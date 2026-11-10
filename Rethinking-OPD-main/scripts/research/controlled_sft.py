"""Plain-completion, response-only LoRA SFT for the controlled DeepMath comparison.

One optimizer update per shared group of questions. Each offline target is repeated
rollouts times; these are identical references, never sampled model trajectories.
"""
import argparse
import contextlib
import json
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import file_hash, load_verl_utility, opsd, read_jsonl, write_json
from .controlled_data import record_exposure


def encode_target(row, tokenizer, config):
    prompt = tokenizer.encode(opsd.plain_messages(row['question'])[0]['content'],add_special_tokens=False)
    # Generation fixes prompt token IDs. The continuation is encoded separately,
    # so a boundary-spanning BPE merge cannot silently change that prefix.
    response = tokenizer.encode(row['privilege_text'],add_special_tokens=False)
    if not response or not row['privilege_text'].strip():
        raise ValueError('SFT requires a nonempty offline reference')
    if response[-1] != tokenizer.eos_token_id:
        response.append(tokenizer.eos_token_id)
    if len(prompt)>config['max_prompt_length'] or len(response)>config['max_response_length']:
        raise ValueError('SFT prompt/reference exceeds the common limits; no silent truncation')
    return prompt, response


def response_ce_sum(model, prompt, response, device, chunk=16):
    """Causal position P-1 predicts response[0]; all prompt labels are excluded."""
    ids = torch.tensor([prompt+response],device=device)
    output = model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
    logits = output.logits[0,len(prompt)-1:len(prompt)-1+len(response)]
    targets = ids[0,len(prompt):]
    # CE checkpointing avoids retaining full FP32 vocab probabilities.
    from torch.utils.checkpoint import checkpoint
    loss = logits.reshape(-1)[:0].sum()
    for start in range(0,len(response),chunk):
        x,y = logits[start:start+chunk],targets[start:start+chunk]
        def term(a,b):
            return F.cross_entropy(a.float(),b,reduction='sum')
        loss = loss + checkpoint(term,x,y,use_reentrant=False)
    return loss


def response_ce_batch_sum(model, examples, device, pad_token_id=0, chunk=256):
    """Sum response-only CE over real sequences, with right padding masked out.

    The caller divides every microbatch by the SAME optimizer-step response-token
    count. Changing microbatch size therefore never reweights short responses.
    """
    from torch.utils.checkpoint import checkpoint
    lengths = [len(p) + len(r) for p, r in examples]
    ids = torch.full((len(examples), max(lengths)), pad_token_id, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, ((prompt, response), length) in enumerate(zip(examples, lengths)):
        ids[i, :length] = torch.tensor(prompt + response, device=device)
        mask[i, :length] = 1
    # Do not materialize [B,T,151936] logits. Backpropagating hundreds of slices
    # of that tensor repeatedly allocates/scatters a huge dense logits gradient.
    # Run the same decoder and checkpoint the output projection together with CE.
    causal = model.get_base_model() if hasattr(model, 'get_base_model') else model
    decoder = getattr(causal, 'model', None) or getattr(causal, 'transformer', None)
    if decoder is None:
        raise ValueError('Add an explicit decoder adapter for this SFT architecture')
    hidden = decoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    states, targets = [], []
    for i, (prompt, response) in enumerate(examples):
        states.append(hidden[i, len(prompt)-1:len(prompt)-1+len(response)])
        targets.append(ids[i, len(prompt):len(prompt)+len(response)])
    states, targets = torch.cat(states), torch.cat(targets)
    head = causal.get_output_embeddings()
    loss = states.reshape(-1)[:0].sum()
    for start in range(0, len(targets), chunk):
        def term(a, b):
            return F.cross_entropy(head(a).float(), b, reduction='sum')
        loss = loss + checkpoint(term, states[start:start+chunk], targets[start:start+chunk], use_reentrant=False)
    return loss


def initialize(config, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model
    source = config['student_model']['path']
    attention = 'eager' if device=='cpu' else config.get('sft_attention_backend','flash_attention_2')
    base = AutoModelForCausalLM.from_pretrained(source,local_files_only=True,
                    torch_dtype=torch.float32,attn_implementation=attention)
    if (Path(source)/'byte_tokenizer.json').is_file():
        from .models import ByteTokenizer
        tokenizer = ByteTokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(source,local_files_only=True)
    with torch.random.fork_rng():
        torch.manual_seed(config['seed'])
        model = get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=config['lora_rank'],
                     lora_alpha=config['lora_alpha'],target_modules=config['lora_target_modules'],
                     exclude_modules=None,bias='none',lora_dropout=0.))
    if any(p.requires_grad and 'lora_' not in n for n,p in model.named_parameters()):
        raise ValueError('Unexpected trainable base parameter')
    if config.get('activation_checkpointing',True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    return model.to(device),tokenizer


def verify_checkpoint(path):
    path = Path(path)
    marker = json.loads((path/'native_sft.json').read_text())
    for name, expected in marker['files'].items():
        item = (path/name).resolve()
        if path.resolve() not in item.parents or file_hash(item) != expected:
            raise ValueError(f'Native SFT checkpoint file changed: {name}')
    return marker


def save_checkpoint(model,optimizer,root,step,config):
    root=Path(root); target=root/f'global_step_{step}'
    root.mkdir(parents=True,exist_ok=True)
    if target.exists():
        verify_checkpoint(target)
        raise ValueError('Refusing to overwrite an existing SFT checkpoint')
    with tempfile.TemporaryDirectory(prefix='.sft-save-',dir=root) as temporary:
        staging=Path(temporary)
        model.save_pretrained(staging/'actor/lora_adapter',safe_serialization=True)
        model.config.save_pretrained(staging/'actor/huggingface')
        torch.save(dict(step=step,optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).device.type=='cuda' else [],
                   python_rng=random.getstate(),numpy_rng=np.random.get_state()),staging/'data.pt')
        write_json(staging/'native_sft.json',dict(step=step,method='sft',
                   schedule_sha256=file_hash(config['question_schedule']),
                   files={str(p.relative_to(staging)):file_hash(p) for p in staging.rglob('*') if p.is_file()}))
        staging.rename(target)
    write = root/'latest_checkpointed_iteration.tmp'
    write.write_text(str(step)+'\n');write.replace(root/'latest_checkpointed_iteration.txt')
    return target


def restore(model,optimizer,path,config):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    marker=verify_checkpoint(path)
    if marker['schedule_sha256']!=file_hash(config['question_schedule']):
        raise ValueError('Cannot resume SFT with a different question order')
    state=torch.load(Path(path)/'data.pt',map_location='cpu',weights_only=False)
    if state['step']!=marker['step']:
        raise ValueError('SFT checkpoint step mismatch')
    set_peft_model_state_dict(model,load_file(Path(path)/'actor/lora_adapter/adapter_model.safetensors'))
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    if state['cuda_rng'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step']


def train(config,output,device='cuda',fixture=False,stop_after=None):
    if config.get('method')!='sft' or not config.get('controlled_comparison') or config['gpus']!=1:
        raise ValueError('This SFT trainer is only for the single-GPU controlled comparison')
    if not fixture and (device!='cuda' or not torch.cuda.is_available()):
        raise ValueError('Production training requires a CUDA GPU; CPU is reserved for explicit tiny tests')
    if config.get('checkpoint_retention') not in ('all','latest'):
        raise ValueError('Controlled SFT requires an explicit checkpoint retention policy')
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    model,tokenizer=initialize(config,device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],
             betas=tuple(config['adam_betas']),eps=config['adam_eps'],weight_decay=config['weight_decay'])
    rows=read_jsonl(config['train_data'])
    if len(rows)!=config['batch']*config['steps']:
        raise ValueError('SFT consumes exactly the shared fixed schedule, without cycling')
    encoded=[encode_target(r,tokenizer,config) for r in rows]
    if any(len(p)+len(r)>model.config.max_position_embeddings for p,r in encoded):
        raise ValueError('SFT context exceeds the model architecture')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    checkpoints=output/'checkpoints';tracker=checkpoints/'latest_checkpointed_iteration.txt'
    step=restore(model,optimizer,checkpoints/f'global_step_{int(tracker.read_text())}',config) if tracker.is_file() else 0
    if step>config['steps']:raise ValueError('Checkpoint exceeds the planned step cap')
    metrics_path=output/'metrics.jsonl'
    if metrics_path.is_file():
        retained=[r for r in read_jsonl(metrics_path) if r['step']<=step]
        from .common import write_jsonl
        write_jsonl(metrics_path,retained)
    monitor=config.get('monitor',{})
    budget=None
    def publish(number):
        saved=save_checkpoint(model,optimizer,checkpoints,number,config)
        if monitor:
            pause=load_verl_utility('research_monitor').finish_checkpoint(monitor,saved,number)
            if budget is not None:
                budget.exclude_pause(pause)
    if not tracker.is_file():
        publish(0)
    elif monitor and step in monitor.get('optimizer_steps',[]):
        # Recover a crash between publishing weights and the analysis job receipt.
        load_verl_utility('research_monitor').finish_checkpoint(monitor,checkpoints/f'global_step_{step}',step)
    budget=load_verl_utility('research_budget').ResearchBudget(metrics_path,step)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    limit=min(config['steps'],stop_after) if stop_after is not None else config['steps']
    for current in range(step+1,limit+1):
        started=time.monotonic();begin=(current-1)*config['batch'];end=current*config['batch']
        record_exposure(config['question_schedule'],current,[r['problem_id'] for r in rows[begin:end]],output/'exposures')
        examples=[example for example in encoded[begin:end] for _ in range(config['sft_target_repeats'])]
        denominator=sum(len(response) for _,response in examples)
        optimizer.zero_grad(set_to_none=True);model.train();numerator=0.
        microbatch = config.get('sft_microbatch', config.get('microbatch', 1))
        if isinstance(microbatch, bool) or not isinstance(microbatch, int) or microbatch < 1:
            raise ValueError('SFT microbatch must be a positive integer')
        for begin in range(0, len(examples), microbatch):
            autocast=torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else contextlib.nullcontext()
            with autocast:
                loss=response_ce_batch_sum(model, examples[begin:begin+microbatch], device,
                    tokenizer.pad_token_id or 0, config.get('sft_ce_chunk_size', 256))
            if not torch.isfinite(loss):raise ValueError('Nonfinite SFT response cross entropy')
            (loss/denominator).backward();numerator+=float(loss.detach())
        grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config['grad_clip'],error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        metrics=dict(budget.update(config['batch'],0,denominator,0),**{
            'actor/loss':numerator/denominator,'sft/response_ce':numerator/denominator,
            'actor/lr':config['learning_rate'],'actor/grad_norm':float(grad),
            'sft/supervised_sequences':len(examples),'sft/distinct_questions':config['batch'],
            'sft/microbatch':microbatch,
            'timing_s/step':time.monotonic()-started})
        if device=='cuda':
            metrics.update({'perf/max_memory_allocated_gb':torch.cuda.max_memory_allocated()/1024**3,
                            'perf/max_memory_reserved_gb':torch.cuda.max_memory_reserved()/1024**3})
        budget.log(current,metrics)
        if current in monitor.get('optimizer_steps',[]) or current%config['save_every']==0 or current==limit:
            publish(current)
    write_json(output/'sft_complete.json',dict(optimizer_step=limit,planned_steps=config['steps'],
               schedule_sha256=file_hash(config['question_schedule'])))
    return model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();train(json.loads(Path(args.config).read_text()),args.output)


if __name__=='__main__':main()
