"""Supported FSDP export into an atomic staging directory, independent of training state."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.research.common import ROOT, file_hash, write_json


def verify_initial_adapter(checkpoint, training):
    """Prove step0 is the Base function and fingerprint the paired trainable initialization."""
    import torch
    from safetensors.torch import load_file
    path=Path(checkpoint)/'actor/lora_adapter'
    settings=json.loads((path/'adapter_config.json').read_text())
    if settings['r']!=training['lora_rank'] or settings['lora_alpha']!=training['lora_alpha']:
        raise ValueError('Initial adapter rank/alpha differs from the training configuration')
    tensors=load_file(path/'adapter_model.safetensors',device='cpu')
    matrices_b={key:value for key,value in tensors.items() if 'lora_B' in key}
    if not matrices_b or any(torch.count_nonzero(value).item()!=0 for value in matrices_b.values()):
        raise ValueError('Initial LoRA B must be zero before using Base weights as step0')
    checksums={}
    for key,value in sorted(tensors.items()):
        raw=value.contiguous().view(torch.uint8).numpy().tobytes()
        checksums[key]=dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(raw).hexdigest())
    return dict(tensors=checksums,zero_b_verified=True,rank=settings['r'],alpha=settings['lora_alpha'])


def export_checkpoint(training, step):
    from scripts.research.pipeline import checkpoint_actor
    from scripts.research.lora import finalize_lora_merge
    _, actor = checkpoint_actor(training,'solution',step)
    identity = dict(step=step,source=str(actor),shards={p.name:file_hash(p) for p in sorted(actor.glob('model_*.pt'))},
                    training_mode=training['training_mode'],rank=training['lora_rank'],alpha=training['lora_alpha'])
    target = Path(training['pipeline_workdir'])/'hf/solution'/f'global_step_{step}'
    if (target/'export_complete.json').is_file():
        saved=json.loads((target/'export_complete.json').read_text())
        if saved['identity'] != identity:
            raise ValueError('Export checkpoint identity changed')
        for name,expected in saved['files'].items():
            if file_hash(target/name)!=expected:
                raise ValueError(f'Exported file changed: {name}')
        return target
    if target.exists():
        raise ValueError('Unrecognized existing HF export; choose a new workspace')
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.export-{step}-',dir=target.parent) as temporary:
        staging=Path(temporary)
        command=[sys.executable,'-m','verl.model_merger','merge','--backend','fsdp','--local_dir',str(actor),
                 '--target_dir',str(staging)]
        env=os.environ.copy()
        env['PYTHONPATH']=os.pathsep.join([str(ROOT),str(ROOT/'verl'),env.get('PYTHONPATH','')])
        subprocess.run(command,cwd=ROOT,env=env,check=True)
        if training['training_mode']=='lora':
            finalize_lora_merge(staging,training)
            marker=staging/'lora_merge.json'
            if marker.is_file():
                merge_info=json.loads(marker.read_text())
                merge_info['source_adapter']=str(target/'lora_adapter')
                write_json(marker,merge_info)
        if not list(staging.glob('*.safetensors')) and not list(staging.glob('pytorch_model*.bin')):
            raise ValueError('HF export produced no model weights')
        files={str(p.relative_to(staging)):file_hash(p) for p in staging.rglob('*') if p.is_file()}
        write_json(staging/'export_complete.json',dict(identity=identity,files=files))
        staging.rename(target)
    return target
