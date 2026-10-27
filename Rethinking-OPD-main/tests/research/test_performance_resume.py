"""Execution override boundaries and production microbatch gradient equivalence."""
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import sys

import torch

from scripts.research.common import file_hash, opsd
from scripts.research.models import Model, ByteTokenizer, create_tiny
from scripts.research.performance_resume import build_command, verify_files
from test_forward_kl import actor_methods, TinyLoRA, Proto


class PerformanceResumeTests(unittest.TestCase):
    def test_only_execution_overrides_allowed(self):
        original = ['python', '-m', 'verl.trainer.main_ppo', 'trainer.total_training_steps=200',
                    'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1', 'trainer.resume_mode="auto"']
        command = build_command(original, {'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':4,
                                           '+opsd.diagnostic_token_budget':8192}, '/saved/global_step_5')
        self.assertIn('trainer.total_training_steps=200', command)
        self.assertIn('actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4', command)
        self.assertIn('trainer.resume_mode="resume_path"', command)
        self.assertIn('trainer.resume_from_path="/saved/global_step_5"', command)
        self.assertFalse(any(item.endswith('per_gpu=1') for item in command))
        for key in ['trainer.total_training_steps', 'actor_rollout_ref.actor.optim.lr', '+opsd.loss_type']:
            with self.assertRaisesRegex(ValueError, 'execution-only'):
                build_command(original, {key:1}, '/saved/global_step_5')
        with self.assertRaises(ValueError):
            build_command(original, {'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':True}, '/x')

    def test_changed_resume_files_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'data.pt'
            p.write_bytes(b'saved dataloader')
            files = {str(p):file_hash(p)}
            verify_files(files)
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'input changed'):
                verify_files(files)

    def test_explicit_shorter_budget_stops_at_absolute_step(self):
        original = ['python', '-m', 'verl.trainer.main_ppo', 'trainer.total_training_steps=200']
        command = build_command(original, {}, '/saved/global_step_20', stop_at_step=50)
        self.assertIn('trainer.total_training_steps=50', command)
        self.assertNotIn('trainer.total_training_steps=200', command)
        for invalid in (20, 19, 201, True):
            with self.assertRaises(ValueError):
                build_command(original, {}, '/saved/global_step_20', stop_at_step=invalid)

    def test_status_uses_requested_budget_without_changing_frozen_config(self):
        from scripts.research.job_control import status
        from scripts.research.common import write_json
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_json(work/'config.json', dict(training=dict(steps=200), monitor=dict(optimizer_steps=[0,20,50,100,200])))
            write_json(work/'run_limits.json', dict(stop_at_step=50))
            result = status(work)['methods']['opsd']
            self.assertEqual(result['total_steps'], 50)
            self.assertEqual(result['expected_analysis_steps'], [0,20,50])

    def test_production_microbatch_one_and_four_same_update(self):
        torch.set_num_threads(2)
        methods = actor_methods()
        with tempfile.TemporaryDirectory() as tmp:
            create_tiny(tmp, 91)
            original = Model(tmp, 'fixture:91').model
            original.requires_grad_(False)
            original.lm_head = TinyLoRA(original.lm_head)
            response = torch.tensor([[10,11,12], [13,14,0], [15,0,0], [16,17,18]])
            mask = torch.tensor([[1,1,1], [1,1,0], [1,0,0], [1,1,1]])
            tensors = opsd.pack_responses([[2,3], [2,4,5], [2,6], [2,7]], response, mask, 0, 32)
            cfg = dict(teacher_update_policy='current_batch_stop_gradient', loss_type='forward_kl',
                       max_teacher_prompt_length=256)
            teacher = opsd.privileged_batch([
                dict(messages=[dict(role='user',content=f'Q{i}')],privilege_text='Reference') for i in range(4)
            ], ByteTokenizer(), response, mask, cfg)
            for key in ('input_ids','attention_mask','position_ids'):
                tensors['opsd_'+key] = teacher[key]
            tensors['response_mask'] = mask
            updated, losses = [], []
            for microbatch in (1,4):
                model = copy.deepcopy(original)
                actor = types.SimpleNamespace(actor_module=model, device_name='cpu', use_remove_padding=False,
                    use_fused_kernels=False, use_ulysses_sp=False,
                    config=types.SimpleNamespace(ppo_epochs=1, ppo_mini_batch_size=4,
                        ppo_micro_batch_size_per_gpu=microbatch, use_kl_loss=False, entropy_coeff=0.))
                actor.actor_optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
                for name in ('_forward_micro_batch','_update_opsd_forward_kl'):
                    setattr(actor, name, types.MethodType(methods[name], actor))

                def optimizer_step():
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
                    actor.actor_optimizer.step()
                    return norm

                actor._optimizer_step = optimizer_step
                data = Proto(tensors, {'opsd':cfg, 'temperature':1., 'teacher_temperature':1.})
                with patch.dict(sys.modules, {'verl.utils.opsd':opsd}):
                    metrics = actor._update_opsd_forward_kl(data)
                updated.append(model.lm_head.lora_B.detach().clone())
                losses.append(metrics['actor/forward_kl'])
            torch.testing.assert_close(updated[0], updated[1], atol=2e-5, rtol=2e-3)
            self.assertAlmostEqual(losses[0], losses[1], places=4)


if __name__ == '__main__':
    unittest.main()
