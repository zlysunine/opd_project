"""Exact forward-KL and production actor CPU checks; no Ray/GPU dependencies."""
import ast
import contextlib
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from scripts.research.common import ROOT, opsd, write_json, write_jsonl
from scripts.research.launch import resolve, hydra_options, validate_experiment
from scripts.research.models import ByteTokenizer, create_tiny, Model
from scripts.research import pipeline


class Proto:
    def __init__(self, tensors, meta_info=None):
        self.batch=tensors
        self.meta_info=meta_info or {}

    def __len__(self):return len(self.batch['responses'])

    def select(self,batch_keys):return Proto({k:self.batch[k] for k in batch_keys},self.meta_info)

    def split(self,size):
        return [Proto({k:v[i:i+size] for k,v in self.batch.items()},self.meta_info)
                for i in range(0,len(self),size)]

    def to(self,device):return Proto({k:v.to(device) for k,v in self.batch.items()},self.meta_info)


def actor_methods():
    """Extract real production methods; replace only distributed import/transport."""
    tree=ast.parse((ROOT/'verl/verl/workers/actor/dp_actor.py').read_text())
    names=('_forward_micro_batch','_update_opsd_forward_kl','update_policy')
    nodes=[copy.deepcopy(n) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
    for node in nodes:node.decorator_list=[]
    namespace={'torch':torch,'DataProto':Proto,'get_device_id':lambda:'cpu'}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<production actor methods>','exec'),namespace)
    return namespace


class TinyLoRA(torch.nn.Module):
    """Small low-rank linear fixture; production uses PEFT, not this test module."""
    def __init__(self,base):
        super().__init__()
        self.base_layer=base
        self.lora_A=torch.nn.Parameter(torch.randn(2,base.in_features)*.05)
        self.lora_B=torch.nn.Parameter(torch.zeros(base.out_features,2))

    def forward(self,x):
        return self.base_layer(x)+(x@self.lora_A.T)@self.lora_B.T


class ForwardKLTests(unittest.TestCase):
    def test_exact_direction_mask_gradient_and_checkpoint_chunks(self):
        torch.manual_seed(6)
        mask=torch.tensor([[1,1,0],[1,0,0]])
        teacher=torch.randn(2,3,7,dtype=torch.float64,requires_grad=True)
        student=torch.randn(2,3,7,dtype=torch.float64,requires_grad=True)
        expected=(teacher.detach().softmax(-1)*(teacher.detach().log_softmax(-1)-student.log_softmax(-1)))
        expected=(expected.sum(-1)*mask).sum()
        reverse=(student.softmax(-1)*(student.log_softmax(-1)-teacher.detach().log_softmax(-1)))
        reverse=(reverse.sum(-1)*mask).sum()
        self.assertGreater(abs(float(expected-reverse)),.01)
        actual=opsd.forward_kl_sum(student,teacher,mask,chunk_size=1)
        torch.testing.assert_close(actual,expected)
        gradient=torch.autograd.grad(actual,student,retain_graph=True)[0]
        dense_gradient=torch.autograd.grad(expected,student)[0]
        torch.testing.assert_close(gradient,dense_gradient)
        actual.backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(student.grad[mask==0].abs().sum().item(),0.)

    def test_temperatures_and_nonfinite_padding(self):
        s=torch.tensor([[[100.,-100.,0.],[float('nan')]*3]],requires_grad=True)
        q=torch.tensor([[[-100.,100.,0.],[float('nan')]*3]],requires_grad=True)
        mask=torch.tensor([[1,0]])
        value=opsd.forward_kl_sum(s,q,mask,student_temperature=2.,teacher_temperature=.5,chunk_size=1)
        self.assertTrue(torch.isfinite(value))
        value.backward()
        self.assertIsNone(q.grad)
        self.assertEqual(s.grad[0,1].abs().sum().item(),0.)
        empty=opsd.forward_kl_sum(s,q,torch.zeros_like(mask))
        self.assertEqual(empty.item(),0.)

    def test_current_profile_and_effective_command(self):
        c=resolve(ROOT/'configs/research/pilot_1p7b.json')
        self.assertEqual((c['batch'],c['rollouts'],c['steps'],c['learning_rate']),(8,8,500,1e-6))
        opts=hydra_options(c,Path('/tmp/no_train'),'STUDENT')
        self.assertEqual(opts['actor_rollout_ref.model.lora_rank'],16)
        self.assertEqual(opts['actor_rollout_ref.model.lora_alpha'],32)
        self.assertEqual(opts['+opsd']['loss_type'],'forward_kl')
        self.assertFalse(opts['reward_model.enable'])
        self.assertEqual(opts['trainer.total_training_steps'],500)
        paper=resolve(ROOT/'configs/research/paper.json')
        self.assertEqual((paper['batch'],paper['rollouts'],paper['steps']),(8,8,500))
        with self.assertRaises(ValueError):validate_experiment({**c,'lora_rank':0})
        with self.assertRaises(ValueError):validate_experiment({**c,'loss_type':'typo'})

    def test_production_actor_full_logits_causal_alignment(self):
        methods=actor_methods()
        with tempfile.TemporaryDirectory() as tmp:
            create_tiny(tmp,7);model=Model(tmp,'local:fixture').model.eval()
            actor=types.SimpleNamespace(actor_module=model,device_name='cpu',use_remove_padding=False,
                                        use_fused_kernels=False,use_ulysses_sp=False)
            response=torch.tensor([[10,11],[12,13]])
            inputs=opsd.pack_responses([[2,3],[2,4,5]],response,torch.ones_like(response),0,8)
            with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
                expected=model(**{k:v for k,v in inputs.items() if k!='responses'},use_cache=False).logits[:,-3:-1]
                *_,actual=methods['_forward_micro_batch'](actor,inputs,1.,return_response_logits=True)
            torch.testing.assert_close(actual,expected)

            # Verify the real remove-padding index mapping with deterministic logits.
            packed=opsd.pack_responses([[2,3]],response[:1],torch.ones_like(response[:1]),0,8)
            actor.use_remove_padding=True
            def unpad(x,mask):
                ids=mask.flatten().nonzero().flatten()
                return x.reshape(-1,1)[ids],ids,None,None
            methods['_forward_micro_batch'].__globals__.update(
                unpad_input=unpad,index_first_axis=lambda x,i:x[i],
                rearrange=lambda x,pattern:x.reshape(-1,1))
            with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
                expected=model(**{k:v for k,v in packed.items() if k!='responses'},use_cache=False).logits[:,-3:-1]
                *_,actual=methods['_forward_micro_batch'](actor,packed,1.,return_response_logits=True)
            torch.testing.assert_close(actual,expected)

    def test_production_lora_update_keeps_base_frozen_and_teacher_fixed(self):
        torch.set_num_threads(1)
        methods=actor_methods()
        with tempfile.TemporaryDirectory() as tmp:
            create_tiny(tmp,11);model=Model(tmp,'local:fixture').model
            model.requires_grad_(False)
            model.lm_head=TinyLoRA(model.lm_head)
            actor=types.SimpleNamespace(actor_module=model,device_name='cpu',use_remove_padding=False,
                use_fused_kernels=False,use_ulysses_sp=False,
                config=types.SimpleNamespace(ppo_epochs=1,ppo_mini_batch_size=2,
                    ppo_micro_batch_size_per_gpu=1,use_kl_loss=False,entropy_coeff=0.))
            for name in ('_forward_micro_batch','_update_opsd_forward_kl'):
                setattr(actor,name,types.MethodType(methods[name],actor))
            actor.actor_optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1)
            updates=[]
            def step():
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1e6)
                actor.actor_optimizer.step();updates.append(1);return norm
            actor._optimizer_step=step
            base_before={n:p.detach().clone() for n,p in model.named_parameters() if 'lora_' not in n}
            lora_before=model.lm_head.lora_B.detach().clone()
            snapshots=[]
            def hook(module,args):
                if not torch.is_grad_enabled(): snapshots.append(module.lora_B.detach().clone())
            handle=model.lm_head.register_forward_pre_hook(hook)
            response=torch.tensor([[10,11],[12,13]])
            mask=torch.tensor([[1,1],[1,0]])
            tensors=opsd.pack_responses([[2,3],[2,4]],response,mask,0,64)
            cfg={'loss_type':'forward_kl','teacher_update_policy':'current_batch_stop_gradient',
                 'max_teacher_prompt_length':256,'forward_kl_chunk_size':1}
            teacher=opsd.privileged_batch([
                {'messages':[{'role':'user','content':'Q1'}],'privilege_text':'Reference one'},
                {'messages':[{'role':'user','content':'Q2'}],'privilege_text':'Reference two'},
            ],ByteTokenizer(),response,mask,cfg)
            for key in ('input_ids','attention_mask','position_ids'):tensors['opsd_'+key]=teacher[key]
            tensors['response_mask']=mask
            data=Proto(tensors,{'opsd':cfg,'temperature':1.,'teacher_temperature':1.})
            reference_model=copy.deepcopy(model)
            reference_actor=types.SimpleNamespace(**vars(actor))
            reference_actor.actor_module=reference_model
            reference_optimizer=torch.optim.SGD(
                [p for p in reference_model.parameters() if p.requires_grad],lr=.1)
            reference_model.eval()
            with torch.no_grad():
                *_,q=methods['_forward_micro_batch'](
                    reference_actor,teacher,1.,return_response_logits=True)
            reference_model.train()
            *_,p=methods['_forward_micro_batch'](
                reference_actor,tensors,1.,return_response_logits=True)
            dense=(q.float().softmax(-1)*(q.float().log_softmax(-1)-p.float().log_softmax(-1))).sum(-1)
            ((dense*mask).sum()/mask.sum()).backward()
            reference_optimizer.step()
            snapshots.clear()  # deepcopy also copied the reference model's test hook.
            with patch.dict(sys.modules,{'verl.utils.opsd':opsd}):
                metrics=methods['update_policy'](actor,data)
            handle.remove()
            self.assertEqual(len(updates),1)
            self.assertEqual(len(snapshots),2)
            torch.testing.assert_close(snapshots[0],snapshots[1],atol=0,rtol=0)
            self.assertFalse(torch.equal(lora_before,model.lm_head.lora_B))
            torch.testing.assert_close(model.lm_head.lora_B,reference_model.lm_head.lora_B,atol=2e-5,rtol=2e-3)
            for n,p in model.named_parameters():
                if n in base_before:
                    torch.testing.assert_close(p,base_before[n],atol=0,rtol=0)
                    self.assertIsNone(p.grad)
            self.assertGreater(metrics['actor/forward_kl'],0.)
            self.assertEqual(metrics['actor/forward_kl_teacher_tokens'],float(teacher['attention_mask'].sum()))
            actor.config.ppo_mini_batch_size=1
            with patch.dict(sys.modules,{'verl.utils.opsd':opsd}), \
                 self.assertRaisesRegex(ValueError,'one optimizer update'):
                actor._update_opsd_forward_kl(data)

    def test_lora_export_uses_training_alpha_and_bakes_adapter(self):
        import transformers  # Initialize availability checks before installing the transport fixture.
        from scripts.research.lora import finalize_lora_merge
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/'lora_adapter/adapter_config.json',{'r':2,'lora_alpha':0})
            (root/'lora_adapter/adapter_model.safetensors').write_bytes(b'transport fixture')
            base=torch.nn.Linear(2,2,bias=False)
            original=base.weight.detach().clone()
            class Adapter:
                @classmethod
                def from_pretrained(cls,model,path,**kwargs):
                    cls.model=model
                    cls.settings=json.loads((Path(path)/'adapter_config.json').read_text())
                    return cls()
                def merge_and_unload(self,safe_merge):
                    with torch.no_grad():self.model.weight.add_(self.settings['lora_alpha']/self.settings['r'])
                    return self.model
            base.save_pretrained=lambda path,**kwargs:torch.save(base.weight,Path(path)/'fixture_merged.pt')
            with patch.dict(sys.modules,{'peft':types.SimpleNamespace(PeftModel=Adapter)}), \
                 patch('transformers.AutoModelForCausalLM.from_pretrained',return_value=base):
                finalize_lora_merge(root,{'lora_rank':2,'lora_alpha':4})
            torch.testing.assert_close(base.weight,(original+2).to(torch.bfloat16),atol=0,rtol=0)
            self.assertTrue(json.loads((root/'lora_merge.json').read_text())['merged'])

    def test_prepare_refuses_old_128_question_budget_before_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=resolve(ROOT/'configs/research/pilot_1p7b.json')
            native=Path(tmp)/'source.jsonl'
            write_jsonl(native,[{'question':f'Q{i}','difficulty':6,'final_answer':'1','r1_solution_1':'R'}
                               for i in range(128)])
            c.update(pipeline_workdir=tmp,native_input=str(native),dataset_revision='local:test')
            with self.assertRaisesRegex(ValueError,'500 steps require at least 4000'):
                pipeline.prepare(c,'drop',128,ByteTokenizer())


if __name__=='__main__':unittest.main()
