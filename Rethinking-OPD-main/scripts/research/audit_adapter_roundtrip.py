"""Measure Base/active-LoRA/serialized-merged logits on identical fixed inputs.

Run for step0 AND a trained adapter. This measures HF merge/precision effects;
it does not validate the FSDP/vLLM worker synchronization path.
"""
import argparse
import contextlib
import gc
import json
from pathlib import Path
import tempfile

import torch

from .common import file_hash, opsd, read_jsonl, write_json


@torch.no_grad()
def score(model, prompt, response, device, autocast):
    underlying=model.get_base_model() if hasattr(model,'get_base_model') else model
    backbone=underlying.model if hasattr(underlying,'model') else underlying.transformer
    ids=torch.tensor([prompt+response],device=device)
    with torch.autocast('cuda',dtype=torch.bfloat16) if autocast else contextlib.nullcontext():
        hidden=backbone(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).last_hidden_state
        # At most three prediction positions; never retain T x vocabulary arrays.
        offsets=sorted({0,min(31,len(response)-1),min(127,len(response)-1)})
        values=underlying.get_output_embeddings()(hidden[0,[len(prompt)-1+i for i in offsets]])
    return values.float().cpu()


def difference(a,b):
    p,q=a.double().log_softmax(-1),b.double().log_softmax(-1)
    return dict(max_logit_abs=float((a-b).abs().max()),mean_logit_abs=float((a-b).abs().mean()),
                kl_a_to_b_nats=float((p.exp()*(p-q)).sum(-1).mean()),
                argmax_agreement=float((a.argmax(-1)==b.argmax(-1)).float().mean()))


@torch.no_grad()
def audit(base,adapter,rows,device='cuda',merged=None,require_zero=False):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    from safetensors.torch import load_file
    from .models import ByteTokenizer
    weights=load_file(Path(adapter)/'adapter_model.safetensors')
    bs=[v for k,v in weights.items() if 'lora_B' in k]
    zero_b=bool(bs) and all(torch.count_nonzero(v)==0 for v in bs)
    if require_zero and not zero_b:raise ValueError('Expected the initial zero-B adapter')
    tokenizer=ByteTokenizer() if (Path(base)/'byte_tokenizer.json').is_file() else AutoTokenizer.from_pretrained(base,local_files_only=True)
    raw=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager')
    active=PeftModel.from_pretrained(raw,adapter,local_files_only=True).to(device).eval()
    comparisons=[];expected=[];inputs=[]
    for row in rows:
        response=row['response_ids'][:128]
        if not response:raise ValueError('Roundtrip audit requires nonempty frozen response IDs')
        for privileged in (False,True):
            messages=opsd.plain_messages(row['question'])
            if privileged:messages=opsd.privileged_messages(messages,row['privilege_text'],'plain_completion')
            prompt=tokenizer.encode(messages[0]['content'],add_special_tokens=False)
            a=score(active,prompt,response,device,device=='cuda')
            with active.disable_adapter(): b=score(active,prompt,response,device,device=='cuda')
            if require_zero and not torch.equal(a,b):raise ValueError('Zero-B active adapter differs from Base on identical inputs')
            comparisons.append(dict(problem_id=row['problem_id'],privileged=privileged,n_positions=len(a),
                                    active_vs_base=difference(a,b)))
            expected.append(a);inputs.append((prompt,response))
    with tempfile.TemporaryDirectory(prefix='opd-roundtrip-') as temp:
        if merged is None:
            target=Path(temp)
            active=active.to('cpu')
            active.merge_and_unload(safe_merge=True).to(torch.bfloat16).save_pretrained(target,safe_serialization=True)
        else:target=Path(merged)
        del active,raw;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
        reloaded=AutoModelForCausalLM.from_pretrained(target,local_files_only=True,
                    torch_dtype=torch.bfloat16,attn_implementation='eager').to(device).eval()
        for record,(prompt,response),a in zip(comparisons,inputs,expected):
            record['active_vs_merged_bf16']=difference(a,score(reloaded,prompt,response,device,False))
        del reloaded;gc.collect()
    return dict(zero_b=zero_b,base_path=str(Path(base).resolve()),adapter_path=str(Path(adapter).resolve()),
                merged_path=str(Path(merged).resolve()) if merged else 'temporary_verified_serialization',
                adapter_sha256=file_hash(Path(adapter)/'adapter_model.safetensors'),
                adapter_config_sha256=file_hash(Path(adapter)/'adapter_config.json'),
                scope='HF eager; active FP32 parameters with CUDA BF16 autocast vs serialized BF16 merge; no FSDP/vLLM claim',
                comparisons=comparisons)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','adapter','input','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--merged');p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--samples',type=int,default=4);p.add_argument('--require-zero',action='store_true')
    a=p.parse_args()
    if a.samples<1:raise ValueError('Positive sample count required')
    result=audit(a.base,a.adapter,read_jsonl(a.input)[:a.samples],a.device,a.merged,a.require_zero)
    result['input_sha256']=file_hash(a.input)
    write_json(a.output,result);print(a.output)


if __name__=='__main__':main()
