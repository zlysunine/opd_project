"""All-token diagnostics with bounded LM-head blocks and incremental KV caches.

No [response_length, vocabulary] arrays are materialized or cached. The initial
policy is rescored on the SAME current trajectory, including live trajectories.
"""
import torch

from scripts.research.common import opsd
from .distributions import TokenAccumulator,pair_token_metrics


def backbone(model):
    if hasattr(model,'model') and hasattr(model.model,'layers'):
        return model.model
    if hasattr(model,'transformer') and hasattr(model.transformer,'h'):
        return model.transformer
    raise ValueError('Unsupported causal backbone for streaming diagnostics')


class PredictionStream:
    """Each block predicts response[s:s+n], using exactly response[:s+n-1]."""
    def __init__(self,adapter,prompt,prefill_chunk_size):
        if not prompt:
            raise ValueError('A causal prompt must be nonempty')
        self.model=adapter.model
        self.base=backbone(self.model)
        self.device=next(self.model.parameters()).device
        self.cache=None
        self.length=0
        self.next_hidden=None
        if prefill_chunk_size<1:
            raise ValueError('Prefill block size must be positive')
        for start in range(0,len(prompt),prefill_chunk_size):
            hidden=self._advance(prompt[start:start+prefill_chunk_size])
            self.next_hidden=hidden[-1]

    def _advance(self,tokens):
        ids=torch.tensor([tokens],device=self.device,dtype=torch.long)
        attention=torch.ones((1,self.length+len(tokens)),device=self.device,dtype=torch.long)
        output=self.base(input_ids=ids,attention_mask=attention,
                         position_ids=torch.arange(self.length,self.length+len(tokens),device=self.device)[None],
                         past_key_values=self.cache,use_cache=True,return_dict=True)
        self.cache=output.past_key_values
        self.length+=len(tokens)
        return output.last_hidden_state[0]

    def prediction_hidden(self,tokens):
        previous=self.next_hidden
        if previous is None:
            raise RuntimeError('Missing causal boundary state')
        after=self._advance(tokens)
        prediction_hidden=torch.cat([previous[None],after[:-1]],0)
        self.next_hidden=after[-1]
        return prediction_hidden

    def predict(self,tokens):
        return self.model.get_output_embeddings()(self.prediction_hidden(tokens)).float().log_softmax(-1)

def response_mask(response,eos_id,explicit=None,length=None):
    """Include the first EOS, exclude padding/positions after it, as online code."""
    ids=torch.tensor([response],dtype=torch.long)
    length=len(response) if length is None else length
    if isinstance(length,bool) or not isinstance(length,int) or not 0<=length<=len(response):
        raise ValueError('Stored response length must fit response IDs')
    mask=opsd.response_mask_until_eos(ids,eos_id,lengths=torch.tensor([length]))[0].bool()
    if explicit is not None:
        supplied=torch.as_tensor(explicit,dtype=torch.bool)
        if supplied.shape!=mask.shape:
            raise ValueError('Stored response_mask must match response IDs')
        mask &= supplied
    return mask


@torch.inference_mode()
def trajectory_metrics(adapter,row,monitor,initial_adapter=None):
    response=list(row['response_ids'])
    prompt=list(row['prompt_ids'])
    teacher_prompt=adapter.prompt(row,True)
    mask=response_mask(response,adapter.tokenizer.eos_token_id,row.get('response_mask'),row.get('response_length'))
    # Trim only trailing invalid padding; all mask-valid tokens, including EOS, remain.
    valid=torch.nonzero(mask,as_tuple=False).flatten()
    end=int(valid[-1])+1 if len(valid) else 0
    response=response[:end]
    mask=mask[:end]
    if not mask.all():
        raise ValueError('Generated responses require a contiguous valid prefix, not internal padding')
    for p in [prompt,teacher_prompt]:
        if len(p)+len(response)>adapter.model.config.max_position_embeddings:
            raise ValueError('Full response diagnostics exceed the actual model context window')
    if initial_adapter is not None and initial_adapter.prompt(row)!=prompt:
        raise ValueError('Initial and current policies must encode the exact same student prompt')
    size=monitor['state_chunk_size']
    forward_size=monitor.get('distribution_forward_chunk_size',256)
    prefill=monitor.get('distribution_prefill_chunk_size',256)
    if min(size,forward_size,prefill)<1:
        raise ValueError('Distribution block sizes must be positive')
    streams=[PredictionStream(adapter,prompt,prefill),PredictionStream(adapter,teacher_prompt,prefill)] if response else []
    separate_initial=initial_adapter is not None and initial_adapter is not adapter
    if separate_initial and response:
        streams.append(PredictionStream(initial_adapter,prompt,prefill))
    accumulator=TokenAccumulator()
    for outer in range(0,max(len(response),1),forward_size):
        block=response[outer:outer+forward_size]
        states=[stream.prediction_hidden(block) for stream in streams] if block else []
        for start in range(0,max(len(block),1),size):
            tokens=block[start:start+size]
            if tokens:
                logp=adapter.model.get_output_embeddings()(states[0][start:start+size]).float().log_softmax(-1)
                logq=adapter.model.get_output_embeddings()(states[1][start:start+size]).float().log_softmax(-1)
            else:
                logp=torch.empty((0,adapter.model.config.vocab_size),device=adapter.device)
                logq=logp
            values=pair_token_metrics(logp,logq,monitor['token_top_ks'],monitor.get('overlap_top_k',16))
            if initial_adapter is not None:
                initial=(initial_adapter.model.get_output_embeddings()(states[2][start:start+size])
                         .float().log_softmax(-1).to(logp.device)) if separate_initial and tokens else logp
                initial_values=pair_token_metrics(logp,initial,monitor['token_top_ks'],monitor.get('overlap_top_k',16))
                values['initial_student_entropy_nats']=initial_values['teacher_entropy_nats']
                values['kl_to_initial_nats']=initial_values['kl_student_teacher_nats']
                values['entropy_gap_to_initial_signed_nats']=initial_values['entropy_gap_signed_nats']
                values['entropy_gap_to_initial_abs_nats']=initial_values['entropy_gap_abs_nats']
                values['overlap_token_advantage_to_initial_nats']=initial_values['overlap_token_advantage_nats']
                for name,value in initial_values.items():
                    if name.startswith('token_top'):
                        values[name+'_to_initial']=value
                del initial,initial_values
            accumulator.update(values,mask[outer+start:outer+start+len(tokens)].to(logp.device))
            del logp,logq,values
        del states
    means,sums,counts=accumulator.result()
    return dict(response_position='all_valid_tokens',distribution_scope='all_valid_tokens',
                aggregation='token_mean',temperature=1.,n_valid_tokens=int(mask.sum()),
                response_length=len(row['response_ids']),metric_sums=sums,metric_token_counts=counts,**means)
