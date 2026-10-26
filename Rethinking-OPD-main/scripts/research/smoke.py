"""CPU end-to-end wiring check using actual HF generation and shared verl OPD terms."""
from pathlib import Path
import json
import re
import time

import numpy as np
import torch

from .common import digest, opsd, seed_all, write_json, write_jsonl, provenance, load_verl_utility
from .data import controlled_dataset, canonical
from .models import create_tiny, Model
from .artifacts import extract, analyze


def smoke(output, config):
    from .run import main
    torch.set_num_threads(1)
    seed_all(config['seed'])
    out=Path(output).resolve()
    if (out/'smoke_result.json').exists():
        raise ValueError('Completed smoke directory exists; choose another output to preserve artifacts')
    out.mkdir(parents=True,exist_ok=True)
    initial=out/'initial'
    create_tiny(initial,config['seed'])
    data=out/'questions.jsonl'
    rows=controlled_dataset(data,config['problems_per_split'],config['seed'])
    main(['reference','--input',str(data),'--output',str(out/'references.jsonl'),'--oracle','--type','solution'])
    refs=__import__('scripts.research.common',fromlist=['read_jsonl']).read_jsonl(out/'references.jsonl')
    write_jsonl(out/'validation_questions.jsonl',[r for r in refs if r['split']=='validation'])
    model=Model(initial,'smoke_seed_'+str(config['seed']))
    optimizer=torch.optim.AdamW(model.model.parameters(),lr=1e-3)
    budget=load_verl_utility('research_budget').ResearchBudget(out/'metrics.jsonl')
    model.model.eval() # keep dropout disabled in both branches; eval does not disable student autograd
    teacher_versions=[]
    before=[p.detach().clone() for p in model.model.parameters()]
    for step in range(config['steps']):
        row=[r for r in refs if r['split']=='train'][step%config['problems_per_split']]
        ids,_,truncated=model.generate(row,config['response_tokens'])
        sp,tp=model.prompt(row),model.prompt(row,True)
        with torch.no_grad():
            sl=model.logits(sp,ids).log_softmax(-1)
            ql=model.logits(tp,ids).log_softmax(-1)
            old,top=sl.topk(16,-1)
            reward=opsd.student_topk_reward(old,ql.gather(-1,top))
            assert not reward.requires_grad and not ql.requires_grad
            assert all(p.grad is None for p in model.model.parameters())
            qtop=ql.topk(16,-1).indices
            dynamic=load_verl_utility('opd_metrics').compute_opd_dynamic_metrics(
                old[None],ql.gather(-1,top)[None],(top[:,:,None]==qtop[:,None,:]).any(-1)[None],
                torch.ones(1,len(ids),dtype=torch.long),-(sl.exp()*sl).sum(-1)[None],-(ql.exp()*ql).sum(-1)[None])
            teacher_versions.append(step)
        current=model.logits(sp,ids).log_softmax(-1).gather(-1,top)
        losses,*_=opsd.clipped_surrogate_terms(current,old,reward)
        loss=losses.sum(-1).mean()
        optimizer.zero_grad();loss.backward()
        grad_norm=float(torch.nn.utils.clip_grad_norm_(model.model.parameters(),1.))
        assert grad_norm>0 and np.isfinite(grad_norm)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        metrics=budget.update(1,len(ids),len(ids),len(tp)+len(ids))
        metrics.update(dynamic)
        metrics.update({'loss/repo_clipped_surrogate':float(loss.detach()),'opsd/teacher_parameter_version':step,
                        'gradient_norm':grad_norm,'truncated':truncated})
        budget.log(step+1,metrics)
        ckpt=out/f'checkpoint_{step+1}'
        model.model.save_pretrained(ckpt)
        (ckpt/'byte_tokenizer.json').write_text('{"type":"byte_smoke_v1"}')
        torch.save({'optimizer':optimizer.state_dict(),'step':step+1,'rng':torch.get_rng_state()},ckpt/'optimizer.pt')
        main(['evaluate','--model',str(ckpt),'--revision',f'smoke_step{step+1}',
              '--input',str(out/'validation_questions.jsonl'),'--split','validation','--samples','1','--pass-k','1',
              '--max-tokens',str(config['response_tokens']),'--output',str(out/f'validation_step{step+1}.jsonl'),
              '--training-run',str(out),'--step',str(step+1)])
    assert any(not torch.equal(a,b) for a,b in zip(before,model.model.parameters()))
    final=out/f'checkpoint_{config["steps"]}'
    train_rows=[r for r in refs if r['split']=='test_instance']
    write_jsonl(out/'evaluation_questions.jsonl',train_rows)
    main(['evaluate','--model',str(final),'--revision','smoke_final','--input',str(out/'evaluation_questions.jsonl'),
          '--output',str(out/'evaluation.jsonl'),'--samples','16','--max-tokens',str(config['response_tokens']),'--fixed'])
    main(['diagnose','--model',str(final),'--revision','smoke_final','--input',str(out/'evaluation.jsonl'),
          '--output',str(out/'diagnostics.jsonl'),'--privileged','--limit','4','--ks','4','16','--chunk','4'])
    fixed=[]
    # Oracle prefixes provide real intermediate-state annotations for a probe wiring check.
    for row in refs:
        if row['split'] not in ('train','validation','test_template'): continue
        text=row['oracle_solution']
        annotations=[]
        for match in re.finditer(r's(\d+)=(\d+)',text):
            annotations.append({'token_index':len(model.tokenizer.encode(text[:match.start(2)])),
                                'label':'state_'+match.group(1),'intermediate_state':int(match.group(2))})
        fixed.append({**row,'response_ids':model.tokenizer.encode(text), 'tokenizer_hash':model.identity['tokenizer_hash'],
                      'step_token_positions':annotations,'trajectory_source':'fixed_oracle_diagnostic',
                      'correct':True,'sample_index':0})
    write_jsonl(out/'fixed_prefixes.jsonl',fixed)
    after=Model(final,'smoke_final')
    initial_model=Model(initial,'smoke_seed_'+str(config['seed']))
    for name,m,priv in [('initial_plain',initial_model,False),('initial_privileged',initial_model,True),
                        ('final_plain',after,False),('final_privileged',after,True)]:
        extract(m,fixed,out/name,[0,1],privileged=priv,strategy='annotated',max_samples=512)
    analyze(out/'final_plain',out/'final_privileged',out/'analysis',k=2,clusters=2,
            baseline_a=out/'initial_plain',baseline_b=out/'initial_plain',rank=2,seed=config['seed'],probe=True)
    main(['plot','--input',str(out),'--output',str(out/'plots')])
    result={'status':'passed','teacher_versions':teacher_versions,'student_parameters_changed':True,
            'covered':['rollout','same-prefix privileged teacher no_grad','repository reward and clipped PPO terms',
                       'backward','HF checkpoint reload','avg@16 and pass@k','full-vocabulary chunk diagnostics',
                       'sharded activation extraction','cosine/RSA/NO/CKA/ARI/subspace/state probe','offline plots'],
            'not_tested':['Ray/vLLM/FSDP distributed execution','Qwen large-model memory','large-checkpoint performance'],
            'scientific_result':'Tiny random-model smoke only; no evidence for a research hypothesis'}
    write_json(out/'smoke_result.json',result)
    write_json(out/'provenance.json',provenance(config,[data]))
    print(json.dumps(result,ensure_ascii=False))
    return result
