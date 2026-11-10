"""Analyze saved states before rotating run-owned checkpoints; keep result tables."""
import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile

from .common import digest, file_hash, write_json


def latest_only(training):
    return training.get('checkpoint_retention', 'all') == 'latest'


def completed_analysis(job_path, config_path, analysis):
    """Validate a result receipt without needing already discarded model weights."""
    from analysis.common import source_hashes, step_dir
    job_path, config_path = Path(job_path), Path(config_path)
    job = json.loads(job_path.read_text())
    status_path = config_path.parent/'job_status'/job_path.name
    if not status_path.is_file():
        return None
    status = json.loads(status_path.read_text())
    if status.get('status') != 'complete':
        return None
    target = step_dir(analysis, job['optimizer_step'])
    receipt = status.get('receipt', {})
    expected = dict(job_sha256=file_hash(job_path), config_sha256=file_hash(config_path),
                    complete_sha256=file_hash(target/'complete.json'))
    if receipt != expected or Path(status['analysis_path']).resolve() != target.resolve():
        raise ValueError('Completed analysis receipt changed; refusing checkpoint cleanup')
    saved = json.loads((target/'complete.json').read_text())
    identity = saved['identity']
    if (identity['step'] != job['optimizer_step'] or identity['config_hash'] != digest(analysis)
            or identity['sources'] != source_hashes()
            or identity['probe_manifest_hash'] != job['probe_manifest_hash']):
        raise ValueError('Completed analysis identity changed; discarded weights cannot be reanalyzed')
    frozen = json.loads((Path(analysis['probe_root'])/'complete.json').read_text())
    for name, expected_hash in frozen['files'].items():
        if file_hash(Path(analysis['probe_root'])/name) != expected_hash:
            raise ValueError(f'Frozen probe changed: {name}')
    for name, expected_hash in saved['files'].items():
        if file_hash(target/name) != expected_hash:
            raise ValueError(f'Analysis cache file changed: {name}')
    return target


def record_analysis(job_path, config_path, target):
    job_path, config_path, target = Path(job_path), Path(config_path), Path(target)
    write_json(config_path.parent/'job_status'/job_path.name,
               dict(status='complete', analysis_path=str(target),
                    receipt=dict(job_sha256=file_hash(job_path), config_sha256=file_hash(config_path),
                                 complete_sha256=file_hash(target/'complete.json'))))


def discard_export(training, step):
    """Delete only this run's recognized, successfully analyzed HF export."""
    target = Path(training['pipeline_workdir'])/'hf/solution'/f'global_step_{step}'
    if not target.exists():
        return
    if target.is_symlink() or not (target/'export_complete.json').is_file():
        raise ValueError(f'Refusing to delete an unrecognized HF export: {target}')
    identity = json.loads((target/'export_complete.json').read_text())['identity']
    actor = Path(training['pipeline_workdir'])/'train/checkpoints'/f'global_step_{step}'/'actor'
    if identity['step'] != step or Path(identity['source']).resolve() != actor.resolve():
        raise ValueError('HF export belongs to a different checkpoint')
    shutil.rmtree(target)


def _saved_checkpoint(checkpoint):
    if (checkpoint/'native_sft.json').is_file():
        from .controlled_sft import verify_checkpoint
        verify_checkpoint(checkpoint)
        return
    models = list((checkpoint/'actor').glob('model_*.pt'))
    if not (checkpoint/'data.pt').is_file() or not models:
        raise ValueError(f'Incomplete training checkpoint: {checkpoint}')
    ranks = set()
    worlds = set()
    for model in models:
        match = re.fullmatch(r'model_world_size_([0-9]+)_rank_([0-9]+)\.pt', model.name)
        if not match:
            raise ValueError(f'Unrecognized checkpoint shard: {model}')
        worlds.add(int(match[1]))
        ranks.add(int(match[2]))
        for prefix in ('optim_', 'extra_state_'):
            if not model.with_name(model.name.replace('model_', prefix, 1)).is_file():
                raise ValueError(f'Incomplete training checkpoint: {checkpoint}, missing {prefix}')
    if len(worlds) != 1 or ranks != set(range(next(iter(worlds)))):
        raise ValueError(f'Incomplete checkpoint ranks: {checkpoint}')
    if not (checkpoint/'actor/lora_adapter/adapter_model.safetensors').is_file():
        raise ValueError(f'Missing LoRA adapter: {checkpoint}')


def archive_adapter(checkpoint, method_root, step):
    """Keep every monitored LoRA independently of rotating full recovery states."""
    source = checkpoint/'actor/lora_adapter'
    if source.is_symlink() or not source.is_dir():
        raise ValueError('Adapter archive requires a real checkpoint-owned directory')
    expected = {str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    root = method_root/'adapter_archive'
    target = root/f'global_step_{step}'
    identity = dict(step=step,source=str(checkpoint.resolve()),files=expected)
    if target.exists():
        marker = target/'archive_complete.json'
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise ValueError('Existing adapter archive differs; refusing to overwrite')
        if any(not (target/name).is_file() or file_hash(target/name)!=value for name,value in expected.items()):
            raise ValueError('Adapter archive contents changed')
        return target
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.adapter-',dir=root) as temp:
        staging = Path(temp)/'adapter'
        shutil.copytree(source,staging)
        write_json(staging/'archive_complete.json',identity)
        staging.rename(target)
    return target


def process_checkpoint(config_path, checkpoint, step):
    """Called in a separate process while the training driver waits."""
    from analysis.common import load_config
    from .compare import analyze
    from .job_control import record_progress
    config_path, checkpoint = Path(config_path).resolve(), Path(checkpoint)
    analysis = load_config(config_path)
    training = json.loads((config_path.parent/'training_config.json').read_text())
    if not latest_only(training):
        raise ValueError('Checkpoint rotation requires explicit latest retention')
    root = config_path.parent/'train/checkpoints'
    expected = root/f'global_step_{step}'
    if checkpoint.is_symlink() or checkpoint.resolve() != expected.resolve():
        raise ValueError('Checkpoint is outside the configured run')
    if int((root/'latest_checkpointed_iteration.txt').read_text()) != step:
        raise ValueError('Refusing to rotate without the latest completed checkpoint')
    _saved_checkpoint(checkpoint)
    owner = dict(config_sha256=file_hash(config_path), method=analysis['method'])
    marker = checkpoint/'research_checkpoint.json'
    if marker.exists() and json.loads(marker.read_text()) != dict(owner, step=step):
        raise ValueError('Checkpoint ownership marker changed')
    write_json(marker, dict(owner, step=step))
    method = analysis['method']
    run = dict(workdir=str(config_path.parent.parent), model=analysis['model'])
    if step in analysis['monitor']['optimizer_steps']:
        analyze(run, method, {step})
        if training.get('controlled_comparison'):
            archive_adapter(checkpoint, config_path.parent, step)

    record_progress(run['workdir'],'checkpoint_cleanup',method,step)
    # Inspect and validate every candidate before deleting any old checkpoint.
    candidates = []
    for old in root.iterdir():
        match = re.fullmatch(r'global_step_([0-9]+)', old.name)
        if not match or int(match[1]) >= step or old.is_symlink() or not old.is_dir():
            continue
        old_step = int(match[1])
        old_marker = old/'research_checkpoint.json'
        if not old_marker.is_file():
            # Never sweep pre-existing/unrecognized directories into this policy.
            continue
        if json.loads(old_marker.read_text()) != dict(owner, step=old_step):
            raise ValueError('Old checkpoint belongs to a different run/configuration')
        if old_step in analysis['monitor']['optimizer_steps']:
            job_path = config_path.parent/'jobs'/f'step_{old_step:06d}.json'
            if not job_path.is_file() or completed_analysis(job_path, config_path, analysis) is None:
                raise ValueError(f'Old checkpoint has unfinished analysis: {old}')
        candidates.append(old)
    for old in candidates:
        shutil.rmtree(old)
        print(f'Removed analyzed checkpoint: {old}', flush=True)
    record_progress(run['workdir'],'training',method,step)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--step', required=True, type=int)
    args = p.parse_args()
    process_checkpoint(args.analysis_config, args.checkpoint, args.step)


if __name__ == '__main__':
    main()
