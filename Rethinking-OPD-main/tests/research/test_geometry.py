"""Numerical, causal-indexing, shared-data and checkpoint-isolation tests."""
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from analysis.distributions import distribution_metrics, nll_bits, selected_log_probs
from analysis.extract import collect, response_positions, summarize_curvature
from analysis.metrics.neighbors import knn, neighbor_overlap
from analysis.metrics.rsa import align_rows, cosine_rsm, rsa_to_initial, rsm_summary
from analysis.metrics.spectrum import spectrum
from analysis.metrics.trajectory import curvature
from analysis.prepare_probe import split_rows
from analysis.smoke import smoke
from scripts.research.common import ROOT, file_hash, load_verl_utility, opsd, read_jsonl
from scripts.research.common import write_json, write_jsonl
from scripts.research.compare import prepare as prepare_comparison
from scripts.research.launch import hydra_options, resolve, validate_experiment
from scripts.research.models import create_tiny, Model


class GeometryTests(unittest.TestCase):
    def test_curvature_exact_angles_and_invariances(self):
        examples=[([[0,0],[1,0],[2,0]],0),([[0,0],[1,0],[1,1]],90),([[0,0],[1,0],[0,0]],180)]
        rotation=np.asarray([[0.,-1.],[1.,0.]])
        for points,angle in examples:
            h=np.asarray(points,dtype=float)
            for changed in (h,h+20,h@rotation,h*3):
                value=curvature(changed)
                self.assertAlmostEqual(value['mean_deg'],angle,places=5)
                self.assertEqual(value['valid_angles'],1)
        self.assertTrue(np.isnan(curvature([[0,0],[0,0],[1,0]])['mean_deg']))
        self.assertTrue(np.isnan(curvature([[0,0],[1,0]])['mean_deg']))

    def test_no_angles_across_special_or_discontinuous_segments(self):
        h=np.stack([np.arange(7),np.zeros(7)],axis=1)
        result=curvature(h,[1,1,1,0,1,1,1])
        self.assertEqual(result['valid_angles'],2)
        self.assertTrue(np.isnan(result['angles_deg'][1:4]).all())

    def test_rsa_statistics_and_degenerate_vectors(self):
        h=np.random.default_rng(3).normal(size=(8,5))
        r,valid=cosine_rsm(h)
        result=rsa_to_initial(r,r)
        self.assertAlmostEqual(result['rsa_rho_to_initial'],1.)
        self.assertEqual(result['rsa_rmse_to_initial'],0.)
        self.assertEqual(result['n_pairs'],28)
        stats,hist,bins=rsm_summary(np.eye(3),['a','a','b'])
        self.assertEqual(stats['rsm_mean'],0.)
        self.assertEqual(hist.sum(),3)
        invalid,_=cosine_rsm([[0,0],[1,0]])
        self.assertTrue(np.isnan(invalid[0]).all())
        self.assertTrue(np.isnan(rsa_to_initial(np.ones((4,4)),np.ones((4,4)))['rsa_rho_to_initial']))

    def test_alignment_and_knn_ties_are_id_stable(self):
        h=np.asarray([[0.,0.],[1,0],[-1,0],[0,1]])
        ids=['z','c','a','b']
        indices,dist=knn(h,ids,2)
        self.assertEqual([ids[i] for i in indices[0]],['a','b'])
        np.testing.assert_equal(neighbor_overlap(indices,indices),np.ones(4))
        order=[2,0,3,1]
        restored=align_rows(h[order],[ids[i] for i in order],ids)
        np.testing.assert_equal(restored,h)
        shuffled,_=knn(h[order],[ids[i] for i in order],2)
        original_neighbors={ids[i]:[ids[j] for j in indices[i]] for i in range(4)}
        actual={ids[order[i]]:[ids[order[j]] for j in shuffled[i]] for i in range(4)}
        self.assertEqual(original_neighbors,actual)
        with self.assertRaises(ValueError):knn(h,ids,4)

    def test_participation_ratio(self):
        self.assertAlmostEqual(spectrum(np.arange(8)[:,None]*np.asarray([[1.,2.,3.]]))['participation_ratio'],1.)
        for rank in (2,3,5):
            h=np.concatenate([np.eye(rank),-np.eye(rank)])
            self.assertAlmostEqual(spectrum(h)['participation_ratio'],rank)
        self.assertTrue(np.isnan(spectrum(np.ones((4,8)))['participation_ratio']))

    def test_explicit_padding_positions_and_target_shift(self):
        right=response_positions([[1,1,1,1,0],[1,1,1,1,1]],[2,3],[2,2])
        left=response_positions([[0,1,1,1,1],[1,1,1,1,1]],[2,3],[2,2])
        np.testing.assert_equal(right[0]['response'],[2,3])
        np.testing.assert_equal(left[0]['response'],[3,4])
        np.testing.assert_equal(left[0]['predictions'],[2,3])
        self.assertEqual(left[0]['prompt_last'],2)
        with self.assertRaises(ValueError):response_positions([[1,0,1]],[1],[1])

    def test_real_block_hooks_nll_and_padded_position_encoding(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            create_tiny(tmp,9)
            model=Model(tmp,'fixture',prompt_mode='plain_completion')
            monitor=dict(anchors=['prompt_last','response_2'],curvature_eps=1e-8)
            with torch.inference_mode():
                anchors,curves,h,pos,path=collect(model,[2,5,6],[7,8,9],monitor)
                ids=torch.tensor([[2,5,6,7,8,9]])
                full=model.model(ids,output_hidden_states=True,use_cache=False)
                self.assertEqual(path,'transformer.h')
                self.assertEqual(len(anchors),2)
                # First hidden_states entry is embedding; last is final-normalized,
                # so use an explicit first-block hook, not hidden_states[-1] for block2.
                np.testing.assert_allclose(anchors[1]['response_2'],full.hidden_states[1][0,4].numpy(),rtol=1e-6)
                lp=selected_log_probs(model.model,h,pos['predictions'].tolist(),chunk_size=1)
                np.testing.assert_allclose(lp,full.logits[0,2:5].log_softmax(-1).numpy(),atol=1e-6)
                bits=nll_bits(model.model,h,[7,8,9],pos['predictions'].tolist(),1)
                expected=-full.logits[0,2:5].log_softmax(-1)[range(3),[7,8,9]].mean().item()/np.log(2)
                self.assertAlmostEqual(bits,expected,places=5)
                for mask,values,prefix in [([0,0,1,1,1,1,1,1],[0,0,2,5,6,7,8,9],2),
                                           ([1,1,1,1,1,1,0,0],[2,5,6,7,8,9,0,0],0)]:
                    attention=torch.tensor([mask])
                    padded=model.model(torch.tensor([values]),attention_mask=attention,
                                       position_ids=(attention.cumsum(-1)-1).clamp_min(0)).logits
                    torch.testing.assert_close(padded[0,prefix:prefix+6],full.logits[0],atol=1e-6,rtol=1e-5)
            self.assertFalse(any(block._forward_hooks for block in model.layers()))

    def test_same_weights_same_prompt_kl_zero_and_overlap_one(self):
        logits=torch.randn(200).log_softmax(0).numpy()
        values=distribution_metrics(logits,logits,logits,(10,50,100))
        self.assertAlmostEqual(values['kl_student_teacher_nats'],0.)
        self.assertAlmostEqual(values['kl_to_initial_nats'],0.)
        self.assertEqual(values['token_top100_overlap'],1.)
        changed=torch.randn(200).log_softmax(0).numpy()
        expected=(np.exp(logits)*(logits-changed)).sum()
        self.assertAlmostEqual(distribution_metrics(logits,changed)['kl_student_teacher_nats'],expected,places=5)

    def test_normalized_dedup_and_stratified_disjoint_splits(self):
        c=json.loads((ROOT/'configs/analysis_deepmath.json').read_text())
        c['dataset'].update(train_n=8,dev_n=3)
        c['monitor'].update(heldout_probe_n=4,train_probe_n=2)
        rows=[dict(question=f'Question {i}',problem_id=str(i),ground_truth=str(i),difficulty=5+i%3,
                   topic=f'Mathematics -> topic{i%2}') for i in range(20)]
        rows.append({**rows[0],'question':'  Question   0  ','problem_id':'duplicate'})
        splits,duplicates=split_rows(rows,c)
        self.assertEqual(len(duplicates),1)
        pools={s:{r['problem_id'] for r in v} for s,v in splits.items()}
        self.assertFalse(pools['train'] & pools['heldout_probe'])
        self.assertFalse(pools['train'] & pools['validation'])
        self.assertLessEqual(pools['train_probe'],pools['train'])
        self.assertEqual(splits,split_rows(list(reversed(rows[:-1])),c)[0])
        self.assertEqual({r['topic_label'] for r in splits['heldout_probe']},{'topic0','topic1'})

    def test_paired_curvature_not_difference_of_filtered_means(self):
        rows=[]
        for layer,values in ((1,[10.,200.,30.]),(2,[5.,float('nan'),40.])):
            for index,value in enumerate(values):
                rows.append(dict(role='student',pool='fixed_base_rollout',split='heldout_probe',layer=layer,
                                 problem_id=str(index),mean_deg=value,mean_step_norm=1.,invalid_fraction=0.,surprisal_bits=2.))
        result=[]
        summarize_curvature(rows,None,lambda *args,**kwargs:result.append((args,kwargs)))
        value=[r for r,k in result if r[0]=='curvature_relative_first_deg' and r[5]==2][0]
        self.assertEqual(value[1],2.5)
        self.assertEqual(value[2],2)
        absolute=[r for r,k in result if r[0]=='curvature_deg' and r[5]==1][0]
        self.assertEqual(absolute[1],20.)
        self.assertEqual(absolute[2],2)

    def test_base_prompt_and_both_method_objectives(self):
        messages=opsd.plain_messages('Q?')
        self.assertEqual(messages[0]['content'],'Problem:\nQ?\n\nSolution:\n')
        q=opsd.privileged_messages(messages,'R','plain_completion')
        self.assertEqual(q[0]['content'],'Reference information:\nR\n\nProblem:\nQ?\n\nSolution:\n')
        self.assertEqual(messages[0]['content'],'Problem:\nQ?\n\nSolution:\n')
        c=resolve(ROOT/'configs/research/base_geometry.json')
        validate_experiment(c)
        opts=hydra_options(c,Path('/tmp/fixture'),'BASE')
        self.assertEqual(opts['+opsd']['loss_type'],'forward_kl')
        self.assertNotIn('+data.apply_chat_template_kwargs.enable_thinking',opts)
        self.assertEqual(opts['actor_rollout_ref.model.lora_init_seed'],42)
        c.update(method='rl',loss_type='grpo',teacher_update_policy='none')
        validate_experiment(c)
        rl=hydra_options(c,Path('/tmp/fixture'),'BASE')
        self.assertEqual(rl['algorithm.adv_estimator'],'grpo')
        self.assertEqual(rl['actor_rollout_ref.model.lora_rank'],16)
        self.assertNotIn('+opsd',rl)
        self.assertFalse(rl['reward_model.enable'])

    def test_job_is_atomic_idempotent_rng_independent_and_requires_checkpoint(self):
        module=load_verl_utility('research_monitor')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=root/'probe.jsonl'
            manifest.write_text('{"id":1}\n')
            c=dict(optimizer_steps=[0,1,5],probe_manifest=str(manifest),probe_manifest_hash=file_hash(manifest),
                   analysis_config='config.json',method='opsd',job_dir=str(root/'jobs'))
            self.assertTrue(module.analysis_due(c,5))
            self.assertFalse(module.analysis_due(c,2))
            checkpoint=root/'step1'
            with self.assertRaises(ValueError):module.write_analysis_job(c,checkpoint,1)
            (checkpoint/'actor').mkdir(parents=True)
            (checkpoint/'data.pt').write_bytes(b'fixture')
            (checkpoint/'actor/model_world_size_1_rank_0.pt').write_bytes(b'fixture')
            state=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
            module.write_analysis_job(c,checkpoint,1)
            module.write_analysis_job(c,checkpoint,1)
            self.assertEqual(state[0],random.getstate())
            np.testing.assert_equal(state[1][1],np.random.get_state()[1])
            torch.testing.assert_close(state[2],torch.get_rng_state())
            self.assertEqual(len(list((root/'jobs').glob('*.json'))),1)
            manifest.write_text('changed')
            with self.assertRaises(ValueError):module.write_analysis_job(c,checkpoint,1)

    def test_end_to_end_cpu_extraction_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=smoke(Path(tmp)/'fixture')
            self.assertEqual(result['status'],'passed')
            self.assertEqual(result['checkpoints'],2)

    def test_actual_probe_freeze_and_shared_training_plans(self):
        from analysis.prepare_probe import freeze
        from analysis.evaluate import answer_status
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            base=root/'tiny'
            create_tiny(base,22)
            native=root/'native.jsonl'
            raw=[dict(question=f'{i}+1?',final_answer=str(i+1),difficulty=5+i%3,
                      topic=f'Mathematics -> Topic{i%2}',r1_solution_1=f'Answer: \\boxed{{{i+1}}}') for i in range(14)]
            raw.append(raw[0])  # Exact duplicates must reach normalized-question dedup.
            write_jsonl(native,raw)
            c=json.loads((ROOT/'configs/analysis_deepmath.json').read_text())
            c.update(workdir=str(root/'run'),output_root=str(root/'run/shared'))
            c['model'].update(path=str(base),revision='tiny_fixture',device='cpu',dtype='float32')
            c['dataset'].update(input=str(native),revision='synthetic_fixture',train_n=8,dev_n=2)
            c['training'].update(steps=1)
            c['monitor'].update(heldout_probe_n=2,train_probe_n=2,optimizer_steps=[0,1],
                                max_probe_response_tokens=4,anchors=['prompt_last','response_3'])
            write_json(Path(c['workdir'])/'models.json',{'qwen3_1p7b_base':dict(path=str(base),revision='fixture',model_id='TINY_FIXTURE')})
            # Only answer-equivalence is replaced by the exact synthetic integer
            # verifier. Data adaptation, tokenization, generation and freeze are real.
            checker=lambda text,gt,benchmark,truncated=False:answer_status(text,gt,'controlled_modular',truncated)
            with patch('analysis.prepare_probe.answer_status',checker),patch('analysis.evaluate.answer_status',checker):
                probe=prepare_comparison(c)
                self.assertEqual(freeze(c),probe)
            a=json.loads((Path(c['workdir'])/'opsd/training_config.json').read_text())
            b=json.loads((Path(c['workdir'])/'grpo/training_config.json').read_text())
            self.assertEqual(a['train_data'],b['train_data'])
            self.assertEqual(a['validation_data'],b['validation_data'])
            self.assertEqual(a['seed'],b['seed'])
            self.assertEqual(a['monitor']['probe_manifest_hash'],b['monitor']['probe_manifest_hash'])
            from analysis.common import load_config
            self.assertEqual(load_config(Path(c['workdir'])/'grpo/analysis_config.json')['method'],'grpo')
            self.assertEqual(len(read_jsonl(probe/'duplicates.jsonl')),1)
            from scripts.research.launch import plan
            _,command=plan(a,Path(c['workdir'])/'plan',False)
            custom=[item for item in command if item.startswith('actor_rollout_ref.model.custom_chat_template=')]
            self.assertEqual(len(custom),1)
            self.assertNotIn('enable_thinking',str(hydra_options(a,root,'BASE').get('+opsd',{})))
            self.assertEqual(len(read_jsonl(probe/'fixed_pool.jsonl')),4)
            row=read_jsonl(probe/'fixed_pool.jsonl')[0]
            self.assertEqual(row['position_index']['prompt_last'],len(row['prompt_ids'])-1)
            (probe/'fixed_pool.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Frozen probe file changed'):
                freeze(c)

    def test_initial_lora_identity_and_zero_b_guard(self):
        from analysis.checkpoints import verify_initial_adapter
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            adapter=root/'actor/lora_adapter'
            adapter.mkdir(parents=True)
            write_json(adapter/'adapter_config.json',dict(r=2,lora_alpha=4))
            values={'model.q_proj.lora_A.weight':torch.randn(2,8),
                    'model.q_proj.lora_B.weight':torch.zeros(8,2)}
            save_file(values,adapter/'adapter_model.safetensors')
            state=verify_initial_adapter(root,dict(lora_rank=2,lora_alpha=4))
            self.assertTrue(state['zero_b_verified'])
            self.assertEqual(verify_initial_adapter(root,dict(lora_rank=2,lora_alpha=4)),state)
            values['model.q_proj.lora_B.weight'][0,0]=1
            save_file(values,adapter/'adapter_model.safetensors')
            with self.assertRaisesRegex(ValueError,'Initial LoRA B'):
                verify_initial_adapter(root,dict(lora_rank=2,lora_alpha=4))

    def test_export_failure_is_retryable_and_does_not_modify_training_checkpoint(self):
        from analysis.checkpoints import export_checkpoint
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            actor=root/'raw/actor'
            actor.mkdir(parents=True)
            shard=actor/'model_world_size_1_rank_0.pt'
            shard.write_bytes(b'raw frozen fixture')
            before=file_hash(shard)
            c=dict(pipeline_workdir=str(root/'run'),training_mode='lora',lora_rank=16,lora_alpha=32)
            def failed(command,**kwargs):
                (Path(command[-1])/'partial.bin').write_bytes(b'partial')
                raise subprocess.CalledProcessError(1,command)
            with patch('scripts.research.pipeline.checkpoint_actor',return_value=(1,actor)),patch('analysis.checkpoints.subprocess.run',failed):
                with self.assertRaises(subprocess.CalledProcessError):export_checkpoint(c,1)
            self.assertFalse((root/'run/hf/solution/global_step_1').exists())
            def succeed(command,**kwargs):
                (Path(command[-1])/'model.safetensors').write_bytes(b'export fixture')
            with patch('scripts.research.pipeline.checkpoint_actor',return_value=(1,actor)),patch('analysis.checkpoints.subprocess.run',succeed),patch('scripts.research.lora.finalize_lora_merge') as merged:
                target=export_checkpoint(c,1)
                self.assertEqual(export_checkpoint(c,1),target)
                self.assertEqual(merged.call_count,1)
            self.assertEqual(file_hash(shard),before)
            (target/'model.safetensors').write_bytes(b'changed')
            with patch('scripts.research.pipeline.checkpoint_actor',return_value=(1,actor)):
                with self.assertRaisesRegex(ValueError,'Exported file changed'):export_checkpoint(c,1)

    def test_completed_training_does_not_perform_step_plus_one(self):
        from scripts.research.compare import train
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            tracker=root/'opsd/train/checkpoints/latest_checkpointed_iteration.txt'
            tracker.parent.mkdir(parents=True)
            tracker.write_text('200')
            write_json(root/'opsd/training_config.json',dict(steps=200))
            with patch('scripts.research.pipeline.checkpoint_actor',return_value=(200,tracker.parent/'global_step_200/actor')),patch('scripts.research.compare.plan') as plan:
                train(dict(workdir=str(root)),'opsd',True)
                plan.assert_not_called()

    def test_analysis_configuration_rejects_silent_experiment_changes(self):
        import copy
        from analysis.common import load_config
        original=json.loads((ROOT/'configs/analysis_deepmath.json').read_text())
        original['model']['revision']='fixture_commit'
        original['dataset']['revision']='fixture_dataset_commit'
        changes=[('model','name_or_path','Qwen/Qwen3-1.7B'),
                 ('model','prompt_mode','chat'),('training','teacher_update_policy','frozen_initial'),
                 ('training','loss_type','reverse_kl'),('monitor','heavy_metrics',['gride']),
                 ('monitor','knn_k',[0]),('monitor','anchors',['response_0']),('training','steps',0)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json'
            write_json(path,original)
            load_config(path)
            for section,key,value in changes:
                changed=copy.deepcopy(original)
                changed[section][key]=value
                write_json(path,changed)
                with self.assertRaises(ValueError,msg=f'{section}.{key}'):
                    load_config(path)


if __name__=='__main__':
    unittest.main()
