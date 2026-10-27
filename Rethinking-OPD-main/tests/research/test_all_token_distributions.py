"""Dense causal forwards are the oracle for bounded all-token monitoring."""
from types import SimpleNamespace
import unittest

import torch

from analysis.distributions import pair_token_metrics, TokenAccumulator
from analysis.extract import summarize_distributions
from analysis.trajectory_distributions import PredictionStream, response_mask, trajectory_metrics
from scripts.research.common import load_verl_utility


def adapter(kind,seed):
    from transformers import GPT2Config, GPT2LMHeadModel, Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    if kind=='gpt2':
        model=GPT2LMHeadModel(GPT2Config(vocab_size=32,n_positions=128,n_embd=24,n_layer=2,n_head=2,
                                       resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.))
    else:
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,
                                         num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                                         head_dim=8,max_position_embeddings=128,attention_dropout=0.))
    model.config._attn_implementation='eager'
    return SimpleNamespace(model=model.eval(),device='cpu',tokenizer=SimpleNamespace(eos_token_id=1),
                           prompt=lambda row,privileged=False:row['teacher_prompt_ids' if privileged else 'prompt_ids'])


def dense(policy,prompt,response):
    ids=torch.tensor([prompt+response])
    return policy.model(ids,use_cache=False).logits[0,len(prompt)-1:len(prompt)-1+len(response)].float().log_softmax(-1)


class AllTokenDistributions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @torch.inference_mode()
    def test_incremental_causal_shift_matches_dense_gpt2_and_qwen(self):
        prompt=[2,3,4,5,6]
        response=[7,8,9,10,11,12,1]
        for kind in ('gpt2','qwen3'):
            policy=adapter(kind,7)
            expected=dense(policy,prompt,response)
            for size in (1,3,7):
                with self.subTest(kind=kind,size=size):
                    stream=PredictionStream(policy,prompt,prefill_chunk_size=2)
                    actual=torch.cat([stream.predict(response[i:i+size]) for i in range(0,len(response),size)])
                    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    @torch.inference_mode()
    def test_full_trajectory_initial_policy_and_head_chunk_bound(self):
        row=dict(prompt_ids=[2,3,4,5],teacher_prompt_ids=[2,15,16,17,3,4,5],
                 response_ids=[6,7,8,9,10,11,1,0,0])
        monitor=dict(state_chunk_size=2,distribution_forward_chunk_size=3,
                     distribution_prefill_chunk_size=2,token_top_ks=[2,4],overlap_top_k=2,
                     max_probe_response_tokens=2)  # Geometry window MUST NOT clip these seven tokens.
        for kind in ('gpt2','qwen3'):
            current,initial=adapter(kind,10),adapter(kind,11)
            observed=[]
            handle=current.model.get_output_embeddings().register_forward_pre_hook(
                lambda module,args:observed.append(args[0].shape[0]))
            try:
                actual=trajectory_metrics(current,row,monitor,initial)
            finally:
                handle.remove()
            self.assertLessEqual(max(observed),2)
            self.assertEqual(actual['n_valid_tokens'],7)
            self.assertEqual(actual['aggregation'],'token_mean')
            p=dense(current,row['prompt_ids'],row['response_ids'][:7])
            q=dense(current,row['teacher_prompt_ids'],row['response_ids'][:7])
            p0=dense(initial,row['prompt_ids'],row['response_ids'][:7])
            expected=pair_token_metrics(p,q,[2,4],2)
            for name,vector in expected.items():
                valid=vector[torch.isfinite(vector)]
                self.assertEqual(actual['metric_token_counts'][name],len(valid))
                if len(valid):
                    self.assertAlmostEqual(actual[name],valid.double().mean().item(),places=5,msg=kind+name)
                else:
                    self.assertIsNone(actual[name])
            self.assertAlmostEqual(actual['kl_to_initial_nats'],(p.exp()*(p-p0)).sum(-1).mean().item(),places=5)
            # At step0 the same policy is the exact zero-KL baseline, even on live text.
            base=trajectory_metrics(current,row,monitor,current)
            self.assertEqual(base['kl_to_initial_nats'],0.)
            self.assertEqual(base['token_top2_overlap_to_initial'],1.)

    def test_matches_online_overlap_advantage_and_masked_full_vocab_entropy(self):
        torch.manual_seed(32)
        p=torch.randn(5,32).log_softmax(-1)
        q=torch.randn(5,32).log_softmax(-1)
        p_ids=p.topk(4,-1).indices
        q_ids=q.topk(4,-1).indices
        shared=(p_ids[:,:,None]==q_ids[:,None,:]).any(-1)
        mask=torch.tensor([1,1,1,0,0],dtype=torch.bool)
        hp=-(p.exp()*p).sum(-1)
        hq=-(q.exp()*q).sum(-1)
        online=load_verl_utility('opd_metrics').compute_opd_dynamic_metrics(
            p.gather(-1,p_ids)[None],q.gather(-1,p_ids)[None],shared[None],mask[None],hp[None],hq[None])
        acc=TokenAccumulator()
        acc.update(pair_token_metrics(p,q,[4],4),mask)
        means,_,counts=acc.result()
        for offline,key in [('student_entropy_nats','student_entropy'),('teacher_entropy_nats','teacher_entropy'),
                            ('entropy_gap_abs_nats','entropy_gap'),('token_top4_overlap','overlap_ratio'),
                            ('overlap_token_advantage_nats','overlap_token_advantage')]:
            self.assertAlmostEqual(means[offline],online['opd/'+key],places=6)
        self.assertEqual(counts['student_entropy_nats'],3)
        # Entropy is neither the sampled-token surprisal nor the renormalized top-k entropy.
        truncated=p.topk(4,-1).values.log_softmax(-1)
        self.assertGreater(abs(means['student_entropy_nats']-(-(truncated.exp()*truncated).sum(-1)[:3].mean()).item()),1.)

    def test_empty_intersection_has_own_denominator_and_empty_mask_is_undefined(self):
        p=torch.tensor([[4.,3.,0.,-1.],[4.,3.,0.,-1.]]).log_softmax(-1)
        q=torch.tensor([[-1.,0.,3.,4.],[4.,2.,0.,-1.]]).log_softmax(-1)
        values=pair_token_metrics(p,q,[2],2)
        acc=TokenAccumulator()
        acc.update(values,torch.tensor([True,True]))
        means,_,counts=acc.result()
        self.assertEqual(counts['overlap_token_advantage_nats'],1)
        self.assertEqual(counts['student_entropy_nats'],2)
        self.assertEqual(means['token_top2_overlap'],.5)
        self.assertEqual(means['token_top2_empty_intersection_rate'],.5)
        self.assertAlmostEqual(means['overlap_token_advantage_nats'],values['overlap_token_advantage_nats'][1].item())
        acc=TokenAccumulator()
        acc.update(values,torch.tensor([False,False]))
        self.assertTrue(all(value is None for value in acc.result()[0].values()))

    def test_token_weighting_across_questions_and_pool_isolation(self):
        def record(pool,total,count):
            return dict(pool=pool,split='heldout_probe',metric_sums={'student_entropy_nats':total},
                        metric_token_counts={'student_entropy_nats':count})
        rows=[record('fixed_base_rollout',1.,1),record('fixed_base_rollout',9.,3),record('live_rollout',8.,2)]
        actual={}
        summarize_distributions(rows,lambda name,value,n,pool,split,**kw:actual.update({pool:(value,n,kw)}))
        self.assertEqual(actual['fixed_base_rollout'][:2],(2.5,4))  # NOT (1+3)/2.
        self.assertEqual(actual['live_rollout'][:2],(4.,2))
        self.assertEqual(actual['fixed_base_rollout'][2]['n_questions'],2)

    def test_response_mask_includes_eos_excludes_padding_and_rejects_holes(self):
        self.assertEqual(response_mask([3,4,1,0,0],1).tolist(),[True,True,True,False,False])
        self.assertEqual(response_mask([3,4,5],1,[1,1,0]).tolist(),[True,True,False])
        self.assertEqual(response_mask([3,4,0,0],1,length=2).tolist(),[True,True,False,False])
        policy=adapter('gpt2',3)
        row=dict(prompt_ids=[2,3],teacher_prompt_ids=[2,7,3],response_ids=[])
        monitor=dict(state_chunk_size=2,token_top_ks=[2],overlap_top_k=2)
        empty=trajectory_metrics(policy,row,monitor,policy)
        self.assertEqual(empty['n_valid_tokens'],0)
        self.assertTrue(all(value==0 for value in empty['metric_token_counts'].values()))
        self.assertIsNone(empty['student_entropy_nats'])
        row.update(response_ids=[4,5,6],response_mask=[1,0,1])
        with self.assertRaisesRegex(ValueError,'contiguous valid prefix'):
            trajectory_metrics(policy,row,monitor)


if __name__=='__main__':
    unittest.main()
