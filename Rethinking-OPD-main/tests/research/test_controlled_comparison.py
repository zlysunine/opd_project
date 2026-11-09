"""CPU checks of shared exposures, actual PEFT SFT/resume and production GRPO math."""
import ast
from collections import defaultdict
import contextlib
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.research.common import ROOT, opsd, read_jsonl, write_json, write_jsonl
from scripts.research.controlled_data import schedule, record_exposure, grpo_signal
from scripts.research.controlled_sft import train, initialize, encode_target, response_ce_sum
from scripts.research.launch import hydra_options,resolve
from scripts.research.reward import compute_score_details
from scripts.research import compare
from analysis.checkpoints import verify_initial_adapter,export_checkpoint


def production_math():
    tree=ast.parse((ROOT/'verl/verl/trainer/ppo/core_algos.py').read_text())
    names={'compute_grpo_outcome_advantage','compute_policy_loss_vanilla','agg_loss'}
    nodes=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    for n in nodes:n.decorator_list=[]
    env=dict(torch=torch,np=np,defaultdict=defaultdict,AlgoConfig=type('AlgoConfig',(),{}),
             verl_F=types.SimpleNamespace(masked_mean=lambda x,m:(x*m).sum()/m.sum()))
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'production_core_algos','exec'),env)
    return env


class Proto:
    def __init__(self,batch):self.batch=batch;self.non_tensor_batch={};self.meta_info={'temperature':1.}
    def __len__(self):return len(self.batch['responses'])
    def select(self,batch_keys,**kwargs):return Proto({k:self.batch[k] for k in batch_keys})
    def split(self,size):return [Proto({k:v[i:i+size] for k,v in self.batch.items()}) for i in range(0,len(self),size)]
    def to(self,device):return self


class Config(dict):
    def __getattr__(self,key):
        try:return self[key]
        except KeyError:raise AttributeError(key) from None


class ControlledTests(unittest.TestCase):
    def test_sft_and_grpo_collect_same_privileged_geometry_without_training_teacher(self):
        from analysis.smoke import smoke
        from analysis.extract import extract
        for method in ['sft','grpo']:
            with tempfile.TemporaryDirectory() as t:
                def diagnostic(c,*args,**kwargs):
                    c['method']=method;c['monitor']['diagnostic_privileged_teacher']=True
                    return extract(c,*args,**kwargs)
                with patch('analysis.smoke.extract',side_effect=diagnostic):
                    smoke(t,compare_teacher=True)
                rows=read_jsonl(Path(t)/'analysis/steps/step_000001/metrics.jsonl')
                direct=[r for r in rows if r['metric']=='rsa_rho_student_teacher']
                self.assertEqual(len(direct),8)
                self.assertTrue(all(r['role']=='student_vs_privileged_teacher' for r in direct))

    def test_historical_probe_adoption_is_verified_and_lossless(self):
        from scripts.research.controlled_data import adopt_probe
        from scripts.research.common import file_hash
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=root/'old';source.mkdir();native=root/'native';native.write_bytes(b'native')
            c=json.loads((ROOT/'configs/controlled_deepmath20.json').read_text())
            c['model']['path']='/fixture/base';c['dataset']['input']=str(native)
            c['dataset'].update(train_n=2,dev_n=1);c['monitor'].update(heldout_probe_n=1,train_probe_n=1)
            for split,count in [('train',2),('validation',1),('heldout_probe',1),('train_probe',1)]:
                write_jsonl(source/f'{split}.jsonl',[dict(problem_id=split+str(i)) for i in range(count)])
            write_jsonl(source/'probe_manifest.jsonl',[dict(problem_id='heldout_probe0')])
            write_jsonl(source/'fixed_pool.jsonl',[dict(problem_id='heldout_probe0',response_ids=[7,8,1])])
            write_json(source/'manifest.json',dict(dataset=c['dataset'],model=dict(revision=c['model']['revision']),
                                                  provenance=dict(config=c)))
            files={p.name:file_hash(p) for p in source.iterdir()}
            marker=dict(files=files,probe_manifest_hash=file_hash(source/'probe_manifest.jsonl'),
                        identity=dict(checkpoint_hash='basehash',input_hash=file_hash(native)))
            write_json(source/'complete.json',marker)
            original=file_hash(source/'complete.json')
            with patch('scripts.research.controlled_data.checkpoint_identity',return_value='basehash'):
                target=adopt_probe(c,source,root/'new/probe')
                self.assertEqual((target/'fixed_pool.jsonl').read_bytes(),(source/'fixed_pool.jsonl').read_bytes())
                self.assertEqual(file_hash(source/'complete.json'),original)
                (source/'fixed_pool.jsonl').write_text('tampered')
                with self.assertRaisesRegex(ValueError,'changed'):adopt_probe(c,source,root/'other/probe')

    def test_paired_report_rejects_changed_exposures_and_keeps_all_methods(self):
        from scripts.research.controlled_report import build
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);write_json(root/'config.json',dict(training=dict(steps=1),monitor=dict(optimizer_steps=[0,1])))
            path=root/'shared/probe/question_schedule.json';write_json(path,dict(batches=[dict(step=1,problem_ids=['train'])]))
            for method in ['opsd','sft','grpo']:
                record_exposure(path,1,['train'],root/method/'train/exposures')
                for step in [0,1]:
                    directory=root/method/'analysis/steps'/f'step_{step:06d}';write_json(directory/'complete.json',{})
                    rows=[dict(problem_id=str(i),split='heldout_probe',correct=(i==0 if step==0 else i<2),extraction_failed=False,truncated=False) for i in range(3)]
                    write_jsonl(directory/'live_rollouts.jsonl',rows)
            output=build(root)
            self.assertTrue(output.is_file())
            result=json.loads((output.parent/'controlled_summary.json').read_text())
            self.assertEqual(set(result['methods']),{'opsd','sft','grpo'})
            self.assertEqual(result['comparisons']['sft_minus_opsd']['delta'],0.)
            write_json(root/'grpo/train/exposures/step_000001.json',dict(problem_ids=['wrong'],schedule_sha256='wrong'))
            with self.assertRaises(ValueError):build(root)

    def test_schedule_is_deterministic_nonrepeating_and_runtime_enforced(self):
        rows=[dict(problem_id=str(i)) for i in range(20)]
        a,b=schedule(rows,2,3,42)
        self.assertEqual((a,b),schedule(rows,2,3,42));self.assertEqual(len({r['problem_id'] for r in a}),6)
        with self.assertRaises(ValueError):schedule(rows,8,3,42)
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'schedule.json';write_json(path,dict(batches=b))
            record_exposure(path,1,b[0]['problem_ids'],Path(t)/'seen')
            with self.assertRaises(ValueError):record_exposure(path,1,list(reversed(b[0]['problem_ids'])),Path(t)/'seen')

    def test_prepare_emits_three_common_configs(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);probe=root/'shared/probe'
            for name in ['probe_manifest','train','validation','train_schedule']:
                write_jsonl(probe/f'{name}.jsonl',[])
            write_json(probe/'question_schedule.json',dict(batches=[]))
            c=json.loads((ROOT/'configs/controlled_deepmath20.json').read_text())
            c.update(workdir=str(root),output_root=str(root/'shared'))
            c['model']['path']='/fixture/base';c['dataset']['input']='/fixture/native'
            write_json(root/'models.json',{'qwen3_1p7b_base':dict(path='/fixture/base',revision='immutable')})
            with patch('scripts.research.controlled_data.prepare_shared',return_value=probe):compare.prepare(c)
            configs=[json.loads((root/m/'training_config.json').read_text()) for m in ['opsd','sft','grpo']]
            fields=['train_data','validation_data','seed','learning_rate','batch','rollouts','steps','lora_rank','lora_alpha','adam_betas','adam_eps','weight_decay','question_schedule']
            for field in fields:self.assertEqual(len({json.dumps(x[field]) for x in configs}),1,field)
            for method in ['opsd','sft','grpo']:
                from analysis.common import load_config
                self.assertTrue(load_config(root/method/'analysis_config.json')['monitor']['diagnostic_privileged_teacher'])
            opts=[hydra_options(configs[i],root/'plan','BASE') for i in [0,2]]
            self.assertFalse(opts[0]['data.shuffle']);self.assertFalse(opts[1]['data.shuffle'])
            self.assertEqual(opts[1]['algorithm.adv_estimator'],'grpo')
            self.assertFalse(opts[1]['reward_model.enable']);self.assertNotIn('+opsd',opts[1])
            self.assertTrue(opts[1]['+actor_rollout_ref.actor.research_global_token_mean'])
            self.assertEqual(opts[1]['custom_reward_function.name'],'compute_score_details')

    def test_grpo_reward_matches_evaluation_including_truncation(self):
        row=dict(answer_type='math')
        text=r'Final answer: \boxed{2}'
        self.assertEqual(compute_score_details('DeepMath-103K',text,'2',row)['score'],1.)
        self.assertEqual(compute_score_details('DeepMath-103K',text,'2',{**row,'_research_truncated':True})['score'],0.)
        self.assertEqual(compute_score_details('DeepMath-103K','unfinished','2',row)['extraction_failed'],1.)

    def test_actual_grpo_actor_is_microbatch_invariant_and_zero_groups_have_no_signal(self):
        env=production_math();advantage=env['compute_grpo_outcome_advantage']
        rewards=torch.tensor([[0.,0.,1.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
        mask=torch.tensor([[1.,1.,1.],[1.,0.,0.],[1.,1.,0.],[1.,0.,0.]])
        adv,_=advantage(rewards,mask,np.array(['q']*4))
        self.assertGreater(adv[0,0],0);self.assertLess(adv[1,0],0)
        zero,_=advantage(torch.zeros_like(rewards),mask,np.array(['q']*4))
        self.assertEqual(float(zero.abs().sum()),0.)
        self.assertEqual(grpo_signal([0]*4,['q']*4)['grpo/zero_variance_group_fraction'],1.)
        tree=ast.parse((ROOT/'verl/verl/workers/actor/dp_actor.py').read_text())
        node=next(copy.deepcopy(n) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='update_policy')
        node.decorator_list=[]
        ns=dict(torch=torch,DataProto=Proto,get_device_id=lambda:'cpu',
                get_policy_loss_fn=lambda name:env['compute_policy_loss_vanilla'],
                append_to_dict=lambda destination,values:[destination.setdefault(k,[]).append(v) for k,v in values.items()])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'production_actor','exec'),ns)
        results=[]
        for micro in [1,4]:
            table=torch.nn.Parameter(torch.zeros(4,3));optimizer=torch.optim.SGD([table],lr=.1)
            cfg=Config(ppo_mini_batch_size=4,ppo_epochs=1,use_dynamic_bsz=False,ppo_micro_batch_size_per_gpu=micro,
                research_global_token_mean=True,loss_agg_mode='token-mean',use_kl_loss=False,entropy_coeff=0.,
                policy_loss={},clip_ratio=.2,clip_ratio_low=.2,clip_ratio_high=.2)
            actor=types.SimpleNamespace(actor_module=torch.nn.Identity(),actor_optimizer=optimizer,config=cfg)
            actor._forward_micro_batch=lambda inputs,**kwargs:(None,table[inputs['responses'][:,0]],None,None)
            def update():
                norm=table.grad.norm();optimizer.step();return norm
            actor._optimizer_step=update
            batch=dict(responses=torch.tensor([[i,0,0] for i in range(4)]),response_mask=mask,
                       input_ids=torch.ones(4,3),attention_mask=mask,position_ids=torch.zeros(4,3),
                       old_log_probs=torch.zeros(4,3),advantages=adv)
            with contextlib.redirect_stdout(__import__('io').StringIO()):reported=ns['update_policy'](actor,Proto(batch))
            self.assertAlmostEqual(reported['actor/pg_loss'][0],float((-adv*mask).sum()/mask.sum()),places=6)
            results.append(table.detach().clone())
        torch.testing.assert_close(results[0],results[1],atol=1e-8,rtol=1e-6)
        self.assertGreater(float(results[0][0,0]),0.)
        self.assertLess(float(results[0][1,0]),0.)


class NativeSFTTests(unittest.TestCase):
    def fixture(self,root):
        from transformers import Qwen3Config,Qwen3ForCausalLM
        modeldir=root/'base';modeldir.mkdir()
        torch.manual_seed(3)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=259,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
              num_attention_heads=2,num_key_value_heads=2,head_dim=8,max_position_embeddings=1024,
              bos_token_id=2,eos_token_id=1,pad_token_id=0,attention_dropout=0.))
        model.save_pretrained(modeldir);write_json(modeldir/'byte_tokenizer.json',{})
        c=resolve(ROOT/'configs/research/base_geometry.json')
        common=json.loads((ROOT/'configs/controlled_deepmath20.json').read_text())['training'];c.update(common)
        c.update(method='sft',loss_type='sft',teacher_update_policy='none',controlled_comparison=True,
             student_model=dict(path=str(modeldir)),registry=str(root/'models.json'),steps=2,batch=2,rollouts=2,
             sft_target_repeats=2,seed=42,lora_rank=2,lora_alpha=4,learning_rate=.01,
             max_prompt_length=800,max_response_length=100,monitor={})
        write_json(root/'models.json',{'qwen3_1p7b_base':dict(path=str(modeldir))})
        rows=[dict(problem_id=str(i),question=str(i)+'+1?',privilege_text=r'Final answer: \boxed{2}',split='train') for i in range(4)]
        write_jsonl(root/'train.jsonl',rows);c['train_data']=str(root/'train.jsonl')
        write_json(root/'schedule.json',dict(batches=[dict(step=i+1,problem_ids=[str(2*i),str(2*i+1)]) for i in range(2)]))
        c['question_schedule']=str(root/'schedule.json')
        return c,rows

    def test_real_cpu_freeze_builds_and_reuses_three_method_inputs(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture,_=self.fixture(root)
            c=json.loads((ROOT/'configs/controlled_deepmath20.json').read_text())
            c['model'].update(path=fixture['student_model']['path'],device='cpu',dtype='float32')
            c['training'].update(steps=2,batch=2,rollouts=2,lora_rank=2,lora_alpha=4,
                                 max_prompt_length=800,max_teacher_prompt_length=800,max_response_length=100)
            c['monitor'].update(optimizer_steps=[0,1,2],heldout_probe_n=2,train_probe_n=1,
                                max_probe_response_tokens=8,behavior_max_response_tokens=8,quality_gate={'enabled':False})
            native=root/'native.jsonl'
            write_jsonl(native,[dict(question=f'Fixture problem {i}',difficulty=6,final_answer='2',
                           r1_solution_1=r'Final answer: \boxed{2}',topic='Math -> Arithmetic') for i in range(7)])
            c['dataset'].update(input=str(native),train_n=4,dev_n=1)
            c.update(workdir=str(root/'run'),output_root=str(root/'run/shared'))
            write_json(root/'run/models.json',{'qwen3_1p7b_base':dict(path=c['model']['path'],revision=c['model']['revision'])})
            frozen=compare.prepare(c)
            self.assertEqual(len(read_jsonl(frozen/'train_schedule.jsonl')),4)
            self.assertEqual(len(read_jsonl(frozen/'fixed_pool.jsonl')),3)
            self.assertEqual(compare.prepare(c),frozen)
            for method in ['opsd','sft','grpo']:
                settings=json.loads((root/f'run/{method}/training_config.json').read_text())
                self.assertEqual(Path(settings['train_data']),frozen/'train_schedule.jsonl')

    def test_initial_guard_stops_before_update_and_native_job_recovers(self):
        from scripts.research.common import file_hash
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config,rows=self.fixture(root);method=root/'sft'
            manifest=root/'probe_manifest.jsonl';write_jsonl(manifest,[dict(problem_id=r['problem_id']) for r in rows])
            config['monitor']=dict(optimizer_steps=[0,1,2],method='sft',controlled_comparison=True,
                  probe_manifest=str(manifest),probe_manifest_hash=file_hash(manifest),
                  analysis_config=str(method/'analysis_config.json'),job_dir=str(method/'jobs'))
            write_json(method/'training_config.json',config)
            other=root/'opsd/initial_adapter.json';write_json(other,dict(rank=999))
            with self.assertRaisesRegex(ValueError,'Initial LoRA differs'):
                train(config,method/'train','cpu',fixture=True)
            self.assertFalse((method/'train/checkpoints/global_step_1').exists())
            other.unlink()
            train(config,method/'train','cpu',fixture=True,stop_after=1)
            (method/'jobs/step_000001.json').unlink()
            train(config,method/'train','cpu',fixture=True)
            self.assertTrue(all((method/'jobs'/f'step_{i:06d}.json').is_file() for i in [0,1,2]))
            self.assertTrue(json.loads((method/'initial_adapter.json').read_text())['zero_b_verified'])

    def test_actual_lora_sft_causal_labels_frozen_base_resume_and_export(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config,rows=self.fixture(root)
            initial,tokenizer=initialize(config,'cpu');initial.eval()
            prompt,response=encode_target(rows[0],tokenizer,config)
            loss=response_ce_sum(initial,prompt,response,'cpu')
            ids=torch.tensor([prompt+response]);labels=ids.clone();labels[:,:len(prompt)]=-100
            expected=initial(input_ids=ids,labels=labels).loss*len(response)
            torch.testing.assert_close(loss,expected)
            frozen={n:p.detach().clone() for n,p in initial.named_parameters() if not p.requires_grad}
            config['pipeline_workdir']=str(root/'full')
            full=train(config,root/'full/train','cpu',fixture=True)
            self.assertTrue(all(torch.equal(frozen[n],p) for n,p in full.named_parameters() if n in frozen))
            adapter=verify_initial_adapter(root/'full/train/checkpoints/global_step_0',config)
            self.assertTrue(adapter['zero_b_verified'])
            state={n:p.detach().clone() for n,p in full.named_parameters() if p.requires_grad}
            partial=train(config,root/'resume/train','cpu',fixture=True,stop_after=1)
            del partial
            resumed=train(config,root/'resume/train','cpu',fixture=True)
            for n,p in resumed.named_parameters():
                if p.requires_grad:torch.testing.assert_close(p,state[n],atol=0,rtol=0)
            logs=read_jsonl(root/'resume/train/metrics.jsonl')
            self.assertEqual([r['step'] for r in logs],[1,2])
            self.assertEqual(logs[-1]['budget']['questions'],4)
            self.assertEqual(logs[-1]['budget']['rollout_tokens'],0)
            self.assertEqual(logs[-1]['metrics']['sft/supervised_sequences'],4)
            exported=export_checkpoint(config,2)
            self.assertTrue((exported/'export_complete.json').is_file())
            from scripts.research.audit_adapter_roundtrip import audit
            probes=[dict(rows[0],response_ids=response)]
            checked=audit(config['student_model']['path'],root/'full/train/checkpoints/global_step_0/actor/lora_adapter',probes,'cpu',require_zero=True)
            self.assertTrue(checked['zero_b'])
            checked=audit(config['student_model']['path'],root/'full/train/checkpoints/global_step_2/actor/lora_adapter',probes,'cpu',merged=exported)
            self.assertFalse(checked['zero_b'])
            self.assertLess(checked['comparisons'][0]['active_vs_merged_bf16']['kl_a_to_b_nats'],1e-3)


if __name__=='__main__':unittest.main()
