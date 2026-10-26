"""Checks before Ray/GPU allocation. No installation, downloads or cluster mutation."""
import importlib
from importlib import metadata
from pathlib import Path
import subprocess
import sys

from .common import ROOT, write_json


TRAIN_MODULES = ('torch','transformers','ray','pyarrow','pandas','omegaconf','hydra','tensordict',
                 'torchdata.stateful_dataloader','vllm','flash_attn','peft','codetiming','psutil',
                 'pylatexenc','sympy','matplotlib')


def check_environment(gpus=1, require_cuda=True):
    errors=[]
    versions={}
    for name in TRAIN_MODULES:
        try:
            module=importlib.import_module(name)
            versions[name]=str(getattr(module,'__version__','imported'))
        except Exception as error:
            errors.append(f'{name}: {type(error).__name__}: {error}')
    try:
        from packaging.version import Version
        if Version(metadata.version('transformers'))<Version('4.51.0'):
            errors.append('Qwen3 requires transformers>=4.51.0')
        if Version(metadata.version('numpy'))>=Version('2.0.0'):
            errors.append('Vendored verl requirements specify numpy<2.0.0')
    except metadata.PackageNotFoundError as error:
        errors.append(str(error))
    if require_cuda:
        try:
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count()<gpus:
                errors.append(f'Require {gpus} visible CUDA device(s); found {torch.cuda.device_count()}')
        except Exception as error:
            errors.append(f'CUDA probe failed: {error}')
    return {'ok':not errors,'errors':errors,'versions':versions,'python':sys.executable,
            'cuda_required':require_cuda,'requested_gpus':gpus}


def compose_training_config(command, env, output):
    """Print resolved Hydra config and instantiate rollout schema; never call main()."""
    output=Path(output)
    result=subprocess.run([*command,'--cfg','job','--resolve'],cwd=ROOT,env=env,capture_output=True,text=True)
    (output/'hydra_preflight.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(f'Hydra configuration failed before Ray startup; see {output}/hydra_preflight.log')
    # Hydra's --cfg checks overrides, but not unexpected dataclass constructor args.
    validation_code='''import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass
with initialize_config_dir(config_dir=sys.argv[1], version_base=None):
    cfg=compose(config_name="ppo_trainer", overrides=sys.argv[2:])
    omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
print("Rollout dataclass validated")
'''
    result=subprocess.run([sys.executable,'-c',validation_code,str(ROOT/'verl/verl/trainer/config'),*command[3:]],
                          cwd=ROOT,env=env,capture_output=True,text=True)
    (output/'rollout_schema_preflight.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(f'Rollout configuration schema failed; see {output}/rollout_schema_preflight.log')
    return {'hydra':'passed','rollout_schema':'passed'}


def validate_training_inputs(c, train, validation, tokenizer, context_length):
    """Check exact prompts and full references; fail before the first paid rollout."""
    from .common import opsd
    lengths=[]
    for split,rows in [('train',train),('validation',validation)]:
        for row in rows:
            kwargs={'tokenize':True,'add_generation_prompt':True,'enable_thinking':c['enable_thinking']}
            p=tokenizer.apply_chat_template(row['messages'],**kwargs)
            q=tokenizer.apply_chat_template(opsd.privileged_messages(row['messages'],row['privilege_text'],
                                            c.get('prompt_mode','chat')),**kwargs)
            if len(p)>c['max_prompt_length']:
                raise ValueError(f'{split}/{row["problem_id"]}: student prompt {len(p)} > {c["max_prompt_length"]}; filter jointly or increase max_prompt_length')
            if c['method']=='opsd' and len(q)>c['max_teacher_prompt_length']:
                raise ValueError(f'{split}/{row["problem_id"]}: privileged prompt {len(q)} > {c["max_teacher_prompt_length"]}; use pipeline prepare --overlength drop or increase teacher context')
            maximum=len(q) if c['method']=='opsd' else len(p)
            if maximum+c['max_response_length']>context_length:
                raise ValueError('Prompt plus configured response exceeds model context window')
            lengths.append({'problem_id':row['problem_id'],'split':split,'student_prompt_tokens':len(p),
                            'teacher_prompt_tokens':len(q),'reference_tokens':len(tokenizer.encode(row['privilege_text'],add_special_tokens=False))})
    batch=c['batch']
    if batch<1 or c['microbatch']<1 or c['gpus']<1 or c['rollouts']<1:
        raise ValueError('Batch, microbatch, GPU count and rollout count must be positive')
    if (batch*c['rollouts'])%c['gpus']:
        raise ValueError('Rollout batch must divide evenly across GPUs')
    available=(len(train)//batch)*c['epochs']
    if len(train)<batch or (c.get('steps') is not None and available<c['steps']):
        raise ValueError(f'Train split supports {available} complete updates, but configured steps={c.get("steps")}; add data or explicitly reduce steps')
    if c['save_every']<1:
        raise ValueError('save_every must be positive so the final checkpoint is retained')
    return lengths
