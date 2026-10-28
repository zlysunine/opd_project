"""Verify real optimizer logs before a pilot can authorize the formal run."""
import json
import math
from pathlib import Path


def validate_training_metrics(workdir, final_step):
    workdir=Path(workdir)
    rows=[json.loads(line) for line in (workdir/'opsd/train/metrics.jsonl').open() if line.strip()]
    latest={row['step']:row for row in rows}
    missing=[step for step in range(1,final_step+1) if step not in latest]
    if missing:raise ValueError(f'Missing optimizer metrics: {missing}')
    required=['actor/forward_kl','actor/grad_norm','opd/overlap_ratio','opd/overlap_token_advantage',
              'opd/student_entropy','opd/teacher_entropy','opd/entropy_gap']
    summaries=[]
    for step in range(1,final_step+1):
        values=latest[step]['metrics']
        absent=[key for key in required if key not in values]
        if absent:raise ValueError(f'Step {step} missing metrics: {absent}')
        for key in required:
            value=values[key]
            if key=='opd/overlap_token_advantage' and value is None:continue
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f'Step {step} invalid {key}: {value}')
        for key in ['actor/forward_kl','actor/grad_norm','opd/student_entropy','opd/teacher_entropy','opd/entropy_gap']:
            if values[key] < -1e-5:raise ValueError(f'Step {step} negative {key}: {values[key]}')
        if not -1e-6<=values['opd/overlap_ratio']<=1+1e-6:
            raise ValueError(f'Step {step} overlap outside [0,1]')
        if 'opsd/teacher_parameter_version' in values and values['opsd/teacher_parameter_version']!=step-1:
            raise ValueError(f'Step {step} teacher parameter version mismatch')
        summaries.append(dict(step=step,**{key:values[key] for key in required},
            response_truncation_rate=values.get('response_length/clip_ratio'),
            response_length_mean=values.get('response_length/mean'),
            optimized_tokens=latest[step].get('budget',{}).get('optimized_tokens')))
    if not any(row['actor/grad_norm']>0 for row in summaries):raise ValueError('All optimizer gradients are zero')
    config=json.loads((workdir/'opsd/training_config.json').read_text())
    expected=dict(steps=final_step,batch=8,rollouts=8,max_prompt_length=2048,max_response_length=16384,
                  max_teacher_prompt_length=2048,lora_rank=16,lora_alpha=32,loss_type='forward_kl')
    for key,value in expected.items():
        if config.get(key)!=value:raise ValueError(f'Training configuration mismatch: {key}={config.get(key)}')
    from safetensors import safe_open
    adapter=workdir/f'opsd/train/checkpoints/global_step_{final_step}/actor/lora_adapter/adapter_model.safetensors'
    with safe_open(adapter,framework='pt',device='cpu') as weights:
        names=[name for name in weights.keys() if 'lora_B' in name]
        if not names or not any(weights.get_tensor(name).count_nonzero().item()>0 for name in names):
            raise ValueError('Final LoRA B weights have not changed from zero initialization')
    from analysis_readiness import validate_analysis_readiness
    config=json.loads((workdir/'config.json').read_text())
    analysis=validate_analysis_readiness(workdir,config['monitor']['optimizer_steps'])
    if not analysis['passed']:raise ValueError('Full analysis readiness has not passed')
    return dict(passed=True,optimizer_updates=final_step,required_fields=required,
                nonzero_lora_B=True,steps=summaries,analysis_readiness=analysis)
