"""CPU scientific invariants; run with unittest (no GPU, network or Ray)."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import torch

from scripts.research.common import ROOT, opsd, digest, load_verl_utility, write_json, write_jsonl
from scripts.research.distributions import distribution_metrics, pass_at_k, evaluate_samples, gap_recovery, aggregate
from scripts.research.representations import (cosine, rsa, linear_cka, subspace_overlap, neighborhood_overlap,
                                             clustering_ari, state_probe)
from scripts.research.artifacts import extract, align, position_selection
from scripts.research.data import canonical, controlled_dataset, reference_intersection, load_questions
from scripts.research.models import Model, ByteTokenizer, create_tiny
from scripts.research.launch import resolve, matrix, hydra_options, experiment_id


class DistributionTests(unittest.TestCase):
    def metrics(self,p,q,k=2):
        return distribution_metrics(torch.tensor(np.array([p]),dtype=torch.float64).log(),torch.tensor(np.array([q]),dtype=torch.float64).log(),[k])

    def test_formulae(self):
        p=np.array([.5,.3,.15,.05]);q=np.array([.4,.35,.2,.05])
        m=self.metrics(p,q)
        self.assertAlmostEqual(m['token_topk/2/overlap_ratio'][0],1.)
        pb=p[:2]/p[:2].sum();qb=q[:2]/q[:2].sum()
        self.assertAlmostEqual(m['token_topk/2/overlap_advantage'][0],np.sum(pb*(np.log(qb)-np.log(pb)))/2)
        self.assertAlmostEqual(m['token_topk/2/shared_mass_student'][0],.8)
        self.assertAlmostEqual(m['token_topk/2/shared_mass_teacher'][0],.75)
        self.assertAlmostEqual(m['entropy/student_full_vocab'][0],-np.sum(p*np.log(p)))
        self.assertAlmostEqual(m['entropy/gap_absolute'][0],abs(-np.sum(p*np.log(p))+np.sum(q*np.log(q))))
        self.assertAlmostEqual(m['kl/full_vocab_p_to_q'][0],np.sum(p*np.log(p/q)))
        self.assertAlmostEqual(m['kl/full_vocab_q_to_p'][0],np.sum(q*np.log(q/p)))

    def test_identical_empty_singleton_and_extreme(self):
        m=self.metrics([.4,.3,.2,.1],[.4,.3,.2,.1])
        self.assertAlmostEqual(m['kl/full_vocab_p_to_q'][0],0.)
        self.assertAlmostEqual(m['token_topk/2/overlap_advantage'][0],0.)
        m=self.metrics([.4,.3,.2,.1],[.1,.2,.3,.4])
        self.assertTrue(np.isnan(m['token_topk/2/overlap_advantage'][0]))
        self.assertEqual(m['token_topk/2/empty_intersection_rate'][0],1.)
        m=self.metrics([.4,.3,.2,.1],[.4,.1,.3,.2])
        self.assertEqual(m['token_topk/2/overlap_advantage'][0],0.)
        m=distribution_metrics(torch.tensor([[0.,-1000.,-1100.]]),torch.tensor([[-1000.,0.,-1100.]]),[1,2])
        self.assertTrue(np.isfinite(m['kl/full_vocab_p_to_q']).all())
        self.assertGreater(m['kl/full_vocab_p_to_q'][0],900.)

    def test_averaging_question_and_token(self):
        rows=[{'metric':'m','problem_id':'a','value':1.}, {'metric':'m','problem_id':'b','value':0.},
              {'metric':'m','problem_id':'b','value':0.}]
        result=aggregate(rows)['m']
        self.assertAlmostEqual(result['token_weighted'],1/3)
        self.assertAlmostEqual(result['question_mean'],.5)

    def test_avg16_not_pass16(self):
        rows=[{'problem_id':'x','correct':i==0,'truncated':False} for i in range(16)]
        result=evaluate_samples(rows)
        self.assertAlmostEqual(result['avg@16'],1/16)
        self.assertEqual(result['pass@16'],1.)
        self.assertAlmostEqual(pass_at_k(100000,1,1000),.01,places=9)
        with self.assertRaises(ValueError):pass_at_k(2,1,3)
        self.assertIsNone(gap_recovery(.5,.6,.5,teacher_checkpoint='fixed')['gap_recovery'])
        with self.assertRaises(ValueError):gap_recovery(.5,.6,.7)


class AlignmentAndGradientTests(unittest.TestCase):
    def test_padding_eos_and_truncation(self):
        response=torch.tensor([[7,1,0,0],[8,9,10,11]])
        mask=opsd.response_mask_until_eos(response,1)
        self.assertEqual(mask.tolist(),[[1,1,0,0],[1,1,1,1]])
        packed=opsd.pack_responses([[2,3],[2,3,4,5,6]],response,mask,0,6)
        self.assertEqual(packed['responses'].tolist(),response.tolist())
        self.assertEqual(packed['attention_mask'][0].tolist(),[0,0,0,1,1,1,1,0,0])
        self.assertEqual(packed['position_ids'][0,:7].tolist(),[0,0,0,0,1,2,3])
        with self.assertRaises(ValueError):opsd.pack_responses([[2]*7,[2]],response,mask,0,6)

    def test_vocab_and_special_ids_fail(self):
        a,b=ByteTokenizer(),ByteTokenizer()
        opsd.assert_tokenizer_compatible(a,b,259,259)
        b.get_vocab=lambda:{'different':0}
        with self.assertRaises(ValueError):opsd.assert_tokenizer_compatible(a,b)
        with self.assertRaises(ValueError):opsd.assert_tokenizer_compatible(a,a,259,260)
        b=ByteTokenizer();b.eos_token_id=2
        with self.assertRaises(ValueError):opsd.assert_tokenizer_compatible(a,b)

    def test_teacher_no_gradient_and_refresh(self):
        parameter=torch.nn.Parameter(torch.tensor([[.7,.2,-.5]],dtype=torch.float64))
        optimizer=torch.optim.SGD([parameter],lr=.5)
        teachers=[]
        for version in range(2):
            with torch.no_grad():
                p=parameter.log_softmax(-1);q=(parameter+torch.tensor([[.5,-.3,.2]])).log_softmax(-1)
                teachers.append(q.clone())
                old,ids=p.topk(2,-1);reward=opsd.student_topk_reward(old,q.gather(-1,ids))
            self.assertFalse(q.requires_grad);self.assertFalse(reward.requires_grad)
            current=parameter.log_softmax(-1).gather(-1,ids)
            losses,*_=opsd.clipped_surrogate_terms(current,old,reward)
            losses.sum().backward()
            self.assertGreater(float(parameter.grad.norm()),0.)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
        self.assertFalse(torch.equal(teachers[0],teachers[1]))

    def test_existing_opd_reward_and_surrogate_regression(self):
        torch.manual_seed(4)
        p=torch.randn(2,3,4,dtype=torch.float64).log_softmax(-1)
        q=torch.randn(2,3,4,dtype=torch.float64).log_softmax(-1)
        expected=-(p-q)*torch.exp(p-torch.logsumexp(p,dim=-1,keepdim=True))
        torch.testing.assert_close(opsd.student_topk_reward(p,q),expected)
        current=(p+torch.randn_like(p)*.5).requires_grad_()
        advantage=expected
        ratio=torch.exp(torch.clamp(current-p,-20,20))
        l1=-advantage*ratio;l2=-advantage*torch.clamp(ratio,.8,1.2)
        legacy=torch.where(advantage<0,torch.minimum(-advantage*3.,torch.maximum(l1,l2)),torch.maximum(l1,l2))
        actual,*_=opsd.clipped_surrogate_terms(current,p,advantage)
        torch.testing.assert_close(actual,legacy)
        ga=torch.autograd.grad(actual.sum(),current,retain_graph=True)[0]
        gb=torch.autograd.grad(legacy.sum(),current)[0]
        torch.testing.assert_close(ga,gb)

    def test_actual_opsd_worker_method_without_distributed_runtime(self):
        # Execute the production method AST with only the transport/FSDP shell replaced.
        # The teacher forwards use a real local HF causal model and real alignment code.
        import contextlib
        import sys
        from unittest.mock import patch
        tree=ast.parse((ROOT/'verl/verl/workers/fsdp_workers.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='compute_opsd_teacher')
        node.decorator_list=[d for d in node.decorator_list if isinstance(d,ast.Call) and isinstance(d.func,ast.Attribute) and d.func.attr=='no_grad']
        class Proto:
            def __init__(self,tensors,meta_info=None):
                self.batch=tensors;self.meta_info=meta_info or {};self.non_tensor_batch={}
            @classmethod
            def from_dict(cls,tensors,meta_info=None):return cls(tensors,meta_info)
        namespace={'torch':torch,'DataProto':Proto}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<production OPSD method>','exec'),namespace)
        with tempfile.TemporaryDirectory() as tmp:
            create_tiny(tmp,5);m=Model(tmp,'local_test')
            class Actor:
                def logp(self,data):
                    batch=data.batch
                    logits=m.model(**{k:batch[k] for k in ('input_ids','attention_mask','position_ids')}).logits
                    length=batch['responses'].shape[1]
                    return logits[:,-length-1:-1].log_softmax(-1)
                def compute_log_prob(self,data,calculate_entropy=True):
                    lp=self.logp(data);top,ids=lp.topk(data.meta_info['top_k'],-1)
                    return lp.gather(-1,data.batch['responses'][...,None]).squeeze(-1),-(lp.exp()*lp).sum(-1),ids,top
                def compute_log_probs_for_ids(self,data):return self.logp(data).gather(-1,data.batch['target_ids'])
            worker=types.SimpleNamespace(_is_actor=True,_is_offload_param=False,processor=None,
                tokenizer=m.tokenizer,actor=Actor(),world_size=1,ulysses_sharding_manager=contextlib.nullcontext(),
                config=types.SimpleNamespace(rollout=types.SimpleNamespace(log_prob_micro_batch_size_per_gpu=1,
                    log_prob_use_dynamic_bsz=False)))
            data=Proto({'responses':torch.tensor([[10,11]]),'response_mask':torch.ones(1,2,dtype=torch.long),
                        'student_top_k_ids':torch.tensor([[[20,21],[22,23]]])},
                {'opsd':{'teacher_update_policy':'current_batch_stop_gradient','max_teacher_prompt_length':512},
                 'teacher_temperature':1.,'log_prob_top_k':2,'global_steps':1})
            data.non_tensor_batch={'extra_info':[{'messages':[{'role':'user','content':'Q'}],'privilege_text':'R'}]}
            with patch.dict(sys.modules,{'verl.utils.opsd':opsd}):
                first=namespace['compute_opsd_teacher'](worker,data)
                self.assertTrue(all(not value.requires_grad for value in first.batch.values()))
                self.assertTrue(all(p.grad is None for p in m.model.parameters()))
                self.assertEqual(first.batch['opsd_teacher_version'].item(),0)
                with torch.no_grad():next(m.model.parameters()).add_(torch.randn_like(next(m.model.parameters()))*.1)
                data.meta_info['global_steps']=2
                second=namespace['compute_opsd_teacher'](worker,data)
            self.assertEqual(second.batch['opsd_teacher_version'].item(),1)
            self.assertFalse(torch.equal(first.batch['teacher_on_student_log_probs'],second.batch['teacher_on_student_log_probs']))
            self.assertEqual(data.batch['responses'].tolist(),[[10,11]])

    def test_dynamic_metrics_prompt_mask_and_empty(self):
        utility=load_verl_utility('opd_metrics')
        s=torch.log(torch.tensor([[[.8,.1],[.4,.3],[.6,.2]]]))
        q=torch.log(torch.tensor([[[.7,.2],[.2,.4],[.6,.2]]]))
        result=utility.compute_opd_dynamic_metrics(s,q,torch.tensor([[[1,0],[0,0],[1,1]]]),
                                                  torch.tensor([[1,1,0]]),torch.tensor([[1.,2.,float('nan')]]),torch.tensor([[2.,1.,float('nan')]]))
        self.assertEqual(result['opd/entropy_gap'],1.)
        self.assertEqual(result['opd/overlap_token_advantage'],0.)
        self.assertEqual(result['opd/empty_intersection_rate'],.5)

    def test_causal_shift_matches_direct_hf(self):
        with tempfile.TemporaryDirectory() as tmp:
            create_tiny(tmp,3);m=Model(tmp,'local_test')
            prompt=[2,10,11];response=[20,21]
            logits=m.logits(prompt,response)
            packed=m.pack(prompt,response)
            direct=m.model(**{k:v for k,v in packed.items() if k!='responses'}).logits
            torch.testing.assert_close(logits,direct[0,2:4])
            # Different left padding lengths preserve the response-relative state.
            batch=opsd.pack_responses([prompt,[2,15,16,17,18]],torch.tensor([response,response]),torch.ones(2,2,dtype=torch.long),0,10)
            with torch.no_grad():
                padded=m.model(**{k:v for k,v in batch.items() if k!='responses'}).logits
            torch.testing.assert_close(padded[0,4:6],logits,atol=1e-6,rtol=1e-5)


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(7)
        self.x=self.rng.normal(size=(40,8))
    def test_geometry(self):
        np.testing.assert_allclose(cosine(self.x,self.x),1.)
        self.assertTrue(np.isnan(cosine(np.zeros((1,2)),np.zeros((1,2))))[0])
        np.testing.assert_allclose(np.diag(rsa(self.x)),1.)
        self.assertAlmostEqual(linear_cka(self.x,3*self.x+5),1.)
        self.assertIsNone(linear_cka(np.zeros((3,2)),np.zeros((3,4))))
        self.assertAlmostEqual(neighborhood_overlap(self.x,self.x,3)['representation_neighborhood_overlap'],1.)
        self.assertAlmostEqual(neighborhood_overlap(self.x,self.x,3)['random_reference'],3/39)
        self.assertEqual(neighborhood_overlap(self.x,self.x,3,distance='euclidean')['representation_neighborhood_overlap'],1.)
        with self.assertRaises(ValueError):rsa(self.x,max_samples=2)
        with self.assertRaises(ValueError):cosine(self.x,self.x[:,:4])
    def test_subspace_sign_and_orthogonal(self):
        a=np.eye(4)[:2];b=-a;c=np.eye(4)[2:]
        self.assertAlmostEqual(subspace_overlap(a,b,2)['representation_subspace_overlap'],1.)
        self.assertAlmostEqual(subspace_overlap(a,c,2)['representation_subspace_overlap'],0.)
        with self.assertRaises(ValueError):subspace_overlap(np.zeros((2,4)),c,2)
    def test_ari_is_not_adp(self):
        x=np.array([[-10,0],[-10,1],[10,0],[10,1]])
        result=clustering_ari(x,{'answer':['a','a','b','b']},2,42)
        self.assertAlmostEqual(result['ari']['answer'],1.)
        self.assertIn('not_ADP',result['method'])
    def test_probe_prevents_template_leakage(self):
        records=[{'problem_id':str(i),'instance_id':str(i),'template_id':'shared','probe_split':s,
                  'labels':{'intermediate_state':i%2}} for i,s in enumerate(['train']*4+['validation']*4+['test']*4)]
        with self.assertRaises(ValueError):state_probe(np.ones((12,3)),records)
    def test_prefix_alignment_fails_and_reorders(self):
        template={'problem_id':'a','sample_index':0,'layer':0,'hook_point':'block_output',
                  'semantics':'predict_current','token_index':0,'position_bucket':'p','prefix_hash':'h','response_hash':'h'}
        ar=[template,{**template,'problem_id':'b'}]
        _,b,_=align(np.ones((2,3)),ar,np.array([[2.,2,2],[1,1,1]]),ar[::-1])
        np.testing.assert_array_equal(b[0],np.ones(3))
        with self.assertRaises(ValueError):align(np.ones((1,3)),ar[:1],np.ones((1,3)),[{**template,'prefix_hash':'other'}])


class ArtifactAndConfigTests(unittest.TestCase):
    def test_instance_split_and_common_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);rows=controlled_dataset(tmp/'data.jsonl',4,2)
            self.assertEqual(len({r['instance_id'] for r in rows}),len(rows))
            for r in rows:
                self.assertNotIn('intermediate_states',r['messages'][0]['content'])
            a=[{**r,'reference_correctness':i%2==0} for i,r in enumerate(rows)]
            b=[{**r,'reference_correctness':i%3==0} for i,r in enumerate(rows)]
            write_jsonl(tmp/'a.jsonl',a);write_jsonl(tmp/'b.jsonl',b)
            reference_intersection([tmp/'a.jsonl',tmp/'b.jsonl'],tmp/'intersection',True)
            selected=load_questions(tmp/'intersection/source_0.jsonl')
            self.assertEqual(len(selected),4)
            self.assertEqual([r['problem_id'] for r in selected],[r['problem_id'] for r in load_questions(tmp/'intersection/source_1.jsonl')])
    def test_native_split_and_instance_group_partition(self):
        from scripts.research.data import partition_questions
        native=canonical({'prompt':[{'role':'user','content':'native'}],
                          'reward_model':{'ground_truth':'1'},'extra_info':{'split':'test'}})
        self.assertEqual(native['split'],'test')
        with tempfile.TemporaryDirectory() as tmp:
            rows=[canonical({'question':f'q{i}','ground_truth':'1','instance_id':str(i//2)}) for i in range(20)]
            partition_questions(rows,tmp,42,.2,.2)
            groups=[{r['instance_id'] for r in load_questions(Path(tmp)/f'{s}.jsonl')} for s in ('train','validation','test')]
            self.assertFalse(groups[0]&groups[1]);self.assertFalse(groups[0]&groups[2]);self.assertFalse(groups[1]&groups[2])

    def test_plot_never_averages_independent_diagnostic_runs(self):
        import csv
        from scripts.research.plots import plot_directory
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            for name,value in [('a',0.),('b',1.)]:
                write_jsonl(tmp/name/'diagnostics.jsonl',[{'metric':'overlap','relative_bucket':0,'problem_id':'p','value':value}])
            plot_directory(tmp,tmp/'plots')
            with (tmp/'plots/token_metrics.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2)
            self.assertEqual({float(r['question_mean']) for r in rows},{0.,1.})

    def test_matrix_never_runs_and_all_roles_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=resolve(ROOT/'configs/research/pilot_1p7b.json')
            rows=matrix(c,tmp,seeds=[42])
            self.assertEqual(len({r['experiment_id'] for r in rows}),len(rows))
            self.assertEqual(len(rows),13)
            exp1=[r for r in rows if r['experiment_id'].startswith('exp1_opsd_consistency__')]
            self.assertEqual(len(exp1),3)
            self.assertTrue(all(r['self_teacher'] in (None,'qwen3_1p7b_nonthinking') for r in exp1))
            self.assertFalse(any('exp1_3p2' in r['experiment_id'] for r in rows))
            with self.assertRaises(ValueError):matrix(c,Path(tmp)/'removed',groups=['exp1_3p2'])
            for r in rows:
                cfg=json.loads(Path(r['config']).read_text())
                if cfg['method']=='opsd':
                    self.assertEqual(cfg['teacher_update_policy'],'current_batch_stop_gradient')
                    self.assertIsNone(r['external_teacher'])
                if cfg['method']=='opd':self.assertIsNone(r['privilege_generator'])
                if cfg['method'] in ('opsd','opd','rl'):
                    opts=hydra_options(cfg,Path(tmp),'STUDENT','TEACHER')
                    self.assertEqual(opts['reward_model.enable'],cfg['method']=='opd')
                    self.assertEqual(opts['trainer.test_freq'],-1)
    def test_deepmath_inclusive_fractional_difficulty_and_no_reference_leak(self):
        from scripts.research.data import prepare_deepmath,partition_questions
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            rows=[{'question':f'q{i}','final_answer':'1','difficulty':d,'r1_solution_1':'SECRET_REFERENCE'}
                  for i,d in enumerate([4.999,5.,5.5,6.,6.5,7.,7.001])]
            solution=prepare_deepmath(rows,tmp/'solution.jsonl','fixed-source',tokenizer=ByteTokenizer())
            empty=prepare_deepmath(rows,tmp/'empty.jsonl','fixed-source',privilege_type='empty')
            self.assertEqual([r['difficulty'] for r in solution],[5.,5.5,6.,6.5,7.])
            self.assertEqual([r['problem_id'] for r in solution],[r['problem_id'] for r in empty])
            self.assertTrue(all('SECRET_REFERENCE' not in str(r['messages']) for r in solution))
            self.assertTrue(all(r['generator_model'] is None for r in solution))
            manifest=json.loads((tmp/'solution.jsonl.manifest.json').read_text())
            self.assertEqual((manifest['before'],manifest['after']),(7,5))
            self.assertFalse(manifest['difficulty_filter']['rounding'])
            for name,data in [('s',solution),('e',empty)]:partition_questions(data,tmp/name)
            for split in ('train','validation','test'):
                self.assertEqual([r['problem_id'] for r in load_questions(tmp/'s'/f'{split}.jsonl')],
                                 [r['problem_id'] for r in load_questions(tmp/'e'/f'{split}.jsonl')])

    def test_deepmath_missing_metadata_and_reference_fail(self):
        from scripts.research.data import prepare_deepmath
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out.jsonl'
            for row in [{'question':'q','final_answer':'1'},
                        {'question':'q','final_answer':'1','difficulty':True},
                        {'question':'q','final_answer':'1','difficulty':float('nan')},
                        {'question':'q','final_answer':'1','difficulty':6.}]:
                with self.assertRaises(ValueError):prepare_deepmath([row],out,'fixed-source')
            self.assertFalse(out.exists())

    def test_consistency_config_and_dataset_guards(self):
        from scripts.research.launch import validate_experiment,validate_dataset
        from scripts.research.data import prepare_deepmath
        c=resolve(ROOT/'configs/research/pilot_1p7b.json');c['dataset_revision']='fixed-source'
        validate_experiment(c)
        with self.assertRaises(ValueError):validate_experiment({**c,'group':'exp1_3p2'})
        with self.assertRaises(ValueError):validate_experiment({**c,'student':'qwen3_4b_nonthinking','size':'4B'})
        with self.assertRaises(ValueError):validate_experiment({**c,'method':'opd','external_teacher':'qwen3_4b_nonthinking'})
        with tempfile.TemporaryDirectory() as tmp:
            rows=prepare_deepmath([{'question':'q','final_answer':'1','difficulty':6.,'r1_solution_1':'r'}],
                                 Path(tmp)/'data.jsonl','fixed-source')
            validate_dataset(rows,c)
            for changed in [{**rows[0],'difficulty':7.1},{**rows[0],'dataset_revision':'wrong'}]:
                with self.assertRaises(ValueError):validate_dataset([changed],c)

    def test_five_training_curves_are_persisted(self):
        import csv
        from scripts.research.plots import plot_directory
        keys=['opd/overlap_ratio','opd/overlap_token_advantage','opd/student_entropy',
              'opd/teacher_entropy','opd/entropy_gap']
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            write_jsonl(tmp/'metrics.jsonl',[{'step':i,'metrics':dict(zip(keys,[.5,None if i==1 else -.01,2.,3.,1.]))}
                                             for i in (1,2)])
            plot_directory(tmp,tmp/'plots')
            self.assertTrue((tmp/'plots/thinking_consistency.png').exists())
            with (tmp/'plots/thinking_consistency.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2)
            self.assertTrue(all(key in rows[0] for key in keys))
            self.assertEqual(rows[0]['opd/overlap_token_advantage'],'')

    def test_plan_does_not_create_a_resolved_run_or_block_later_execution(self):
        from scripts.research.launch import plan
        with tempfile.TemporaryDirectory() as tmp:
            c=resolve(ROOT/'configs/research/pilot_1p7b.json')
            path,command=plan(c,tmp,False)
            self.assertTrue((Path(tmp)/'plan_config.json').exists())
            self.assertTrue((Path(tmp)/'plan_command.sh').exists())
            self.assertFalse((Path(tmp)/'resolved_config.json').exists())
            self.assertIn('verl.trainer.main_ppo',command)

    def test_cache_identity_includes_model_prefix_and_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);create_tiny(tmp/'model',3);m=Model(tmp/'model','local_test')
            row=canonical({'question':'Q','ground_truth':'1','split':'train'})
            row.update(response_ids=[10,11],tokenizer_hash=m.identity['tokenizer_hash'])
            extract(m,[row],tmp/'cache',[0],bins=1)
            extract(m,[row],tmp/'cache',[0],bins=1) # resume exact shards
            for changed in [{**row,'response_ids':[10,12]}, {**row,'question':'Q2','messages':[{'role':'user','content':'Q2'}]}]:
                with self.assertRaises(ValueError):extract(m,[changed],tmp/'cache',[0],bins=1)
            m.identity['revision']='changed'
            with self.assertRaises(ValueError):extract(m,[row],tmp/'cache',[0],bins=1)
    def test_budget_restore_checkpoint_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cls=load_verl_utility('research_budget').ResearchBudget
            b=cls(Path(tmp)/'metrics.jsonl');b.update(2,4,4,6);b.log(1,{})
            b.update(2,4,4,6);b.log(2,{})
            restored=cls(Path(tmp)/'metrics.jsonl',1)
            self.assertEqual(restored.totals['optimized_tokens'],4)
            self.assertTrue(restored.exceeded({'optimized_tokens':4}))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
