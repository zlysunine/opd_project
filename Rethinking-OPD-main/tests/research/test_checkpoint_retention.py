"""Checkpoint rotation must preserve completed analyses and survive failures."""
import copy
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from analysis.common import source_hashes, step_dir
from scripts.research import compare, retention
from scripts.research.common import ROOT, digest, file_hash, load_verl_utility, write_json
from scripts.research.launch import hydra_options, hydra_value, resolve


class CheckpointRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)/'run'
        self.method = self.run/'opsd'
        self.root = self.method/'train/checkpoints'
        self.probe = self.run/'shared/probe'
        self.probe.mkdir(parents=True)
        manifest = self.probe/'probe_manifest.jsonl'
        manifest.write_text('{"problem_id":"fixture"}\n')
        write_json(self.probe/'complete.json', dict(files={manifest.name:file_hash(manifest)}))
        self.analysis = json.loads((ROOT/'configs/analysis_deepmath.json').read_text())
        self.analysis.update(method='opsd', probe_root=str(self.probe), output_root=str(self.method/'analysis'))
        self.analysis['model'].update(path='/fixture/Base', revision='fixture-model')
        self.analysis['dataset']['revision'] = 'fixture-data'
        self.analysis['monitor']['optimizer_steps'] = [0,1,2]
        self.config = self.method/'analysis_config.json'
        write_json(self.config, self.analysis)
        self.monitor = dict(optimizer_steps=[0,1,2], probe_manifest=str(manifest),
                            probe_manifest_hash=file_hash(manifest), analysis_config=str(self.config),
                            method='opsd', job_dir=str(self.method/'jobs'), checkpoint_retention='latest')
        self.training = dict(checkpoint_retention='latest', pipeline_workdir=str(self.method),
                             lora_rank=16, lora_alpha=32, training_mode='lora', monitor=self.monitor, steps=2)
        write_json(self.method/'training_config.json', self.training)
        self.module = load_verl_utility('research_monitor')
        self.calls = []
        self.addCleanup(patch.stopall)
        patch('analysis.checkpoints.verify_initial_adapter', return_value={'zero_b_verified':True}).start()
        self.export = patch('analysis.checkpoints.export_checkpoint', side_effect=self.fake_export).start()
        self.extract = patch('analysis.extract.extract', side_effect=self.fake_extract).start()

    def save(self, step):
        checkpoint = self.root/f'global_step_{step}'
        actor = checkpoint/'actor'
        actor.mkdir(parents=True, exist_ok=True)
        for name in ('model_world_size_1_rank_0.pt','optim_world_size_1_rank_0.pt',
                     'extra_state_world_size_1_rank_0.pt','lora_adapter/adapter_model.safetensors'):
            path = actor/name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'complete fixture')
        (checkpoint/'data.pt').write_bytes(b'dataloader')
        (self.root/'latest_checkpointed_iteration.txt').write_text(str(step))
        self.module.write_analysis_job(self.monitor, checkpoint, step) if step in (0,1,2) else None
        return checkpoint

    def fake_export(self, training, step):
        target = self.method/'hf/solution'/f'global_step_{step}'
        target.mkdir(parents=True, exist_ok=True)
        (target/'model.safetensors').write_bytes(b'export')
        write_json(target/'export_complete.json', dict(identity=dict(step=step,
                   source=str(self.root/f'global_step_{step}'/'actor'))))
        return target

    def fake_extract(self, analysis, checkpoint, revision, step):
        self.calls.append(step)
        target = step_dir(analysis,step)
        target.mkdir(parents=True)
        (target/'metrics.jsonl').write_text('{"fixture":true}\n')
        identity = dict(step=step, config_hash=digest(analysis), sources=source_hashes(),
                        probe_manifest_hash=self.monitor['probe_manifest_hash'])
        write_json(target/'complete.json', dict(identity=identity,
                   files={'metrics.jsonl':file_hash(target/'metrics.jsonl')}))
        return target

    def process(self, step):
        checkpoint = self.save(step)
        retention.process_checkpoint(self.config, checkpoint, step)
        return checkpoint

    def test_only_latest_weights_remain_and_all_analysis_results_can_be_reused(self):
        for step in (0,1,2):
            self.process(step)
            self.assertEqual([p.name for p in self.root.glob('global_step_*')], [f'global_step_{step}'])
        self.assertEqual(self.calls,[0,1,2])
        self.assertTrue((self.method/'initial_adapter.json').is_file())
        self.assertEqual(list((self.method/'hf/solution').glob('global_step_*')),[])
        for step in (0,1,2):
            self.assertTrue((step_dir(self.analysis,step)/'complete.json').is_file())
        self.export.reset_mock()
        self.extract.reset_mock()
        compare.analyze(dict(workdir=str(self.run),model=self.analysis['model']),'opsd')
        self.export.assert_not_called()
        self.extract.assert_not_called()

    def test_analysis_failure_preserves_old_new_checkpoints_and_export_then_retries(self):
        old = self.process(0)
        new = self.save(1)
        self.extract.side_effect = RuntimeError('analysis failed')
        with self.assertRaisesRegex(RuntimeError,'analysis failed'):
            retention.process_checkpoint(self.config,new,1)
        self.assertTrue(old.is_dir())
        self.assertTrue(new.is_dir())
        self.assertTrue((self.method/'hf/solution/global_step_1').is_dir())
        self.extract.side_effect = self.fake_extract
        retention.process_checkpoint(self.config,new,1)
        self.assertFalse(old.exists())
        self.assertFalse((self.method/'hf/solution/global_step_1').exists())

    def test_incomplete_new_checkpoint_never_deletes_old(self):
        old = self.process(0)
        new = self.save(1)
        (new/'actor/optim_world_size_1_rank_0.pt').unlink()
        with self.assertRaisesRegex(ValueError,'Incomplete training checkpoint'):
            retention.process_checkpoint(self.config,new,1)
        self.assertTrue(old.is_dir())

    def test_missing_rank_is_rejected(self):
        new = self.save(0)
        for path in (new/'actor').glob('*world_size_1_rank_0.pt'):
            path.rename(path.with_name(path.name.replace('world_size_1','world_size_2')))
        with self.assertRaisesRegex(ValueError,'Incomplete checkpoint ranks'):
            retention.process_checkpoint(self.config,new,0)

    def test_changed_analysis_results_block_cleanup(self):
        old = self.process(0)
        (step_dir(self.analysis,0)/'metrics.jsonl').write_text('tampered')
        new = self.save(1)
        with self.assertRaisesRegex(ValueError,'Analysis cache file changed'):
            retention.process_checkpoint(self.config,new,1)
        self.assertTrue(old.is_dir())
        self.assertTrue(new.is_dir())

    def test_periodic_unmonitored_save_rotates_and_unmanaged_paths_are_preserved(self):
        self.process(0)
        unmanaged = self.root/'global_step_0000'
        unmanaged.mkdir()
        outside = Path(self.temp.name)/'outside'
        outside.mkdir()
        linked = self.root/'global_step_00'
        linked.symlink_to(outside,target_is_directory=True)
        notes = self.root/'notes'
        notes.mkdir()
        self.process(3)
        self.assertFalse((self.root/'global_step_0').exists())
        self.assertTrue((self.root/'global_step_3').exists())
        self.assertTrue(unmanaged.is_dir())
        self.assertTrue(linked.is_symlink())
        self.assertTrue(notes.is_dir())
        self.assertEqual(self.calls,[0])

    def test_driver_restores_gpu_allocation_and_waits_for_analysis(self):
        checkpoint = self.save(0)
        for visible in ('GPU-assigned',None):
            config = dict(self.monitor,cuda_visible_devices=visible)
            def run(command,**kwargs):
                if visible is None:
                    self.assertNotIn('CUDA_VISIBLE_DEVICES',kwargs['env'])
                else:
                    self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],visible)
                self.assertTrue(kwargs['check'])
                retention.process_checkpoint(self.config,checkpoint,0)
            with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':''}), patch.object(self.module.subprocess,'run',run):
                self.assertGreaterEqual(self.module.finish_checkpoint(config,checkpoint,0),0)
        self.assertEqual(self.calls,[0])
        self.assertTrue((self.method/'checkpoint_maintenance.jsonl').is_file())

    def test_default_options_disable_early_worker_rotation_and_keep_legacy_deferred_mode(self):
        config = resolve(ROOT/'configs/research/base_geometry.json')
        config['monitor'] = copy.deepcopy(self.monitor)
        options = hydra_options(config,Path('/tmp/fixture'),'BASE')
        self.assertIsNone(options['trainer.max_actor_ckpt_to_keep'])
        self.assertEqual(options['+trainer.research_monitor']['checkpoint_retention'],'latest')
        legacy = dict(self.monitor)
        legacy.pop('checkpoint_retention')
        checkpoint = self.save(0)
        with patch.object(self.module.subprocess,'run') as run:
            self.assertEqual(self.module.finish_checkpoint(legacy,checkpoint,0),0)
            run.assert_not_called()

    def test_actual_hydra_parser_accepts_monitor_and_budget_dictionaries(self):
        from hydra.core.override_parser.overrides_parser import OverridesParser
        parser = OverridesParser.create()
        budget = dict(optimized_tokens=None,rollout_tokens=123,wall_seconds=1.5)
        monitor = dict(self.monitor,cuda_visible_devices='0',flags=[True,False,None])
        for key,value in (('research_monitor',monitor),('research_budget',budget)):
            override = parser.parse_overrides([f'+trainer.{key}={hydra_value(value)}'])[0]
            self.assertEqual(override.value(),value)

    def test_analysis_pause_does_not_consume_training_wall_budget(self):
        cls = load_verl_utility('research_budget').ResearchBudget
        with patch('time.monotonic',side_effect=[10.,25.]):
            budget = cls(Path(self.temp.name)/'metrics.jsonl')
            budget.exclude_pause(12.)
            budget.update(1,1,1,1)
        self.assertEqual(budget.totals['wall_seconds'],3.)

    def test_failed_checkpoint_publication_preserves_previous_recovery_pointer(self):
        import torch
        from omegaconf import OmegaConf
        tree = ast.parse((ROOT/'verl/verl/trainer/ppo/ray_trainer.py').read_text())
        trainer = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RayPPOTrainer')
        method = next(n for n in trainer.body if isinstance(n,ast.FunctionDef) and n.name=='_save_checkpoint')
        namespace = dict(os=os,torch=torch)
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<actual trainer save>','exec'),namespace)
        self.root.mkdir(parents=True)
        tracker = self.root/'latest_checkpointed_iteration.txt'
        tracker.write_text('0')
        config = OmegaConf.create(dict(trainer=dict(default_local_dir=str(self.root),default_hdfs_dir=None)))
        instance = types.SimpleNamespace(config=config,global_steps=1,use_critic=False,
                    actor_rollout_wg=types.SimpleNamespace(save_checkpoint=lambda *a,**k:None),
                    train_dataloader=types.SimpleNamespace(state_dict=lambda:{'position':8}))
        fs = types.SimpleNamespace(local_mkdir_safe=lambda path:Path(path).mkdir(parents=True,exist_ok=True))
        with patch.dict(sys.modules,{'verl.utils.fs':fs}), patch('os.replace',side_effect=OSError('publish failed')):
            with self.assertRaisesRegex(OSError,'publish failed'):
                namespace['_save_checkpoint'](instance)
        self.assertEqual(tracker.read_text(),'0')
        self.assertTrue((self.root/'global_step_1/data.pt').is_file())

    def test_lora_export_contains_merged_bf16_weights_without_shadowing_files(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from scripts.research.lora import finalize_lora_merge
        target = Path(self.temp.name)/'hf_fixture'
        config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, tie_word_embeddings=True)
        base = Qwen3ForCausalLM(config).bfloat16()
        base.save_pretrained(target)
        adapted = get_peft_model(base.float(), LoraConfig(r=2,lora_alpha=4,
                                  target_modules='all-linear',task_type='CAUSAL_LM'))
        with torch.no_grad():
            for name, parameter in adapted.named_parameters():
                if 'lora_' in name:
                    parameter.fill_(0.1)
        adapted.save_pretrained(target/'lora_adapter')
        expected = adapted.merge_and_unload(safe_merge=True).bfloat16()
        adapter_config = target/'lora_adapter/adapter_config.json'
        settings = json.loads(adapter_config.read_text())
        settings['lora_alpha'] = 0
        write_json(adapter_config,settings)
        finalize_lora_merge(target,dict(lora_rank=2,lora_alpha=4))
        actual = Qwen3ForCausalLM.from_pretrained(target,local_files_only=True,torch_dtype=torch.bfloat16)
        for name, value in actual.state_dict().items():
            self.assertEqual(value.dtype,torch.bfloat16)
            torch.testing.assert_close(value,expected.state_dict()[name],atol=0,rtol=0)
        self.assertEqual([p.name for p in target.glob('*.safetensors')],['model.safetensors'])


if __name__ == '__main__':
    unittest.main()
