"""CPU-only tiny GPT-2 extraction fixture. No Qwen, DeepMath, LoRA or training evidence."""
import argparse
import json
from pathlib import Path
import shutil

import torch

from scripts.research.common import digest, file_hash, read_jsonl, seed_all, write_json, write_jsonl
from scripts.research.models import Model, create_tiny
from .evaluate import rollout
from .extract import extract


def smoke(out, make_report=False):
    torch.set_num_threads(1)
    root=Path(out).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use an empty smoke output directory')
    base=root/'tiny_initial'
    create_tiny(base,41)
    model=Model(base,'tiny_fixture_initial',device='cpu',dtype='float32',prompt_mode='plain_completion')
    c=json.loads((Path(__file__).resolve().parents[1]/'configs/analysis_deepmath.json').read_text())
    c.update(run_id='tiny_cpu_fixture',method='fixture',table_format='jsonl',output_root=str(root/'analysis'))
    c['model'].update(name_or_path='TINY_GPT2_CPU_FIXTURE_NOT_QWEN',path=str(base),revision='tiny_fixture_initial',
                      dtype='float32',device='cpu')
    c['dataset'].update(reference_field='synthetic_oracle',revision='synthetic_fixture')
    c['training']={'status':'NOT_RUN', 'note':'Step1 is a deterministic weight perturbation, not OPSD or GRPO training'}
    c['monitor'].update(optimizer_steps=[0,1],anchors=['prompt_last','response_3'],knn_k=[2,3],
                         max_probe_response_tokens=8,distribution_positions=[1,3],token_top_ks=[10,50,100],
                         teacher_geometry_steps=[],plot_full_rsm_steps=[0,1],plot_relative_depths=[0.,1.])
    probe=root/'analysis/probe'
    rows=[]
    for index in range(12):
        row=dict(problem_id=f'fixture_{index:03d}',question=f'{index}+1?',ground_truth=str(index+1),
                 benchmark='controlled_modular',privilege_text=f'Answer: \\boxed{{{index+1}}}',
                 topic_label='addition' if index%2 else 'arithmetic',difficulty=5+index%3,
                 split='heldout_probe' if index<8 else 'train_probe')
        result=rollout(model,row,c['monitor'],'fixed_base_rollout',0)
        result['anchor_membership']={a:a=='prompt_last' or len(result['response_ids'])>=int(a.split('_')[1]) for a in c['monitor']['anchors']}
        rows.append(result)
    write_jsonl(probe/'fixed_pool.jsonl',rows)
    write_jsonl(probe/'probe_manifest.jsonl',[dict(problem_id=r['problem_id'],split=r['split'],prompt_hash=digest(r['prompt_ids']),response_hash=digest(r['response_ids'])) for r in rows])
    for split in ('heldout_probe','train_probe'):
        write_jsonl(probe/f'{split}.jsonl',[r for r in rows if r['split']==split])
    write_json(probe/'manifest.json',dict(model=model.identity,status='SYNTHETIC_CPU_FIXTURE'))
    write_json(probe/'complete.json',dict(identity={'fixture':True},probe_manifest_hash=file_hash(probe/'probe_manifest.jsonl'),
               files={p.name:file_hash(p) for p in probe.iterdir() if p.is_file()}))
    write_json(root/'config.json',c)
    extract(c,base,'tiny_fixture_initial',0,model)
    with torch.no_grad():
        seed_all(67)
        target=model.model.transformer.h[0].mlp.c_fc.weight
        target.add_(torch.randn_like(target)*.01)
    changed=root/'tiny_perturbed'
    model.model.save_pretrained(changed)
    shutil.copyfile(base/'byte_tokenizer.json',changed/'byte_tokenizer.json')
    changed_model=Model(changed,'tiny_fixture_perturbed',device='cpu',dtype='float32',prompt_mode='plain_completion')
    extract(c,changed,'tiny_fixture_perturbed',1,changed_model)
    # Repeating extraction verifies hashes and reuses exactly the same frozen outputs.
    extract(c,changed,'tiny_fixture_perturbed',1,changed_model)
    initial=read_jsonl(Path(c['output_root'])/'steps/step_000000/metrics.jsonl')
    current=read_jsonl(Path(c['output_root'])/'steps/step_000001/metrics.jsonl')
    assert all(abs(r['value']-1)<1e-10 for r in initial if r['metric'].startswith('neighbor_overlap') and r['value'] is not None)
    assert all(abs(r['value'])<1e-10 for r in initial if r['metric']=='rsa_rmse_to_initial' and r['value'] is not None)
    assert any(r['value'] is not None and r['value']>0 for r in current if r['metric']=='rsa_rmse_to_initial')
    result=dict(status='passed',initial_metric_rows=len(initial),perturbed_metric_rows=len(current),
                questions=12,checkpoints=2,model='tiny GPT-2 fixture, 2 blocks, hidden_size24',
                limitations=['No GPU/FSDP/PEFT run','Not Qwen or DeepMath evidence','Step1 is weight perturbation, not training'],
                table_format='jsonl; pyarrow is not required for this CPU fixture')
    if make_report:
        from .report import report
        result['report']=str(report(c['output_root']))
    write_json(root/'smoke_result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--report',action='store_true')
    args=parser.parse_args()
    print(json.dumps(smoke(args.out,args.report),indent=2))


if __name__=='__main__':
    main()
