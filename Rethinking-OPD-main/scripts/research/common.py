"""Artifact identity, provenance and dependency-light imports of shared verl utilities."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def load_verl_utility(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'verl/verl/utils' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


opsd = load_verl_utility('opsd')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Strict JSON: undefined metrics are null, never invented zero values.
    def clean(x):
        import math
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def provenance(config, files=()):
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for name in ['torch', 'transformers', 'numpy', 'scikit-learn', 'matplotlib', 'ray', 'pyarrow', 'vllm']:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    source_files = list((ROOT/'scripts/research').glob('*.py')) + list((ROOT/'configs/research').glob('*.json'))
    source_files += [ROOT/p for p in ['scripts/__init__.py','run_controlled_comparison.sh',
                                     'on_policy_distillation.sh','grpo.sh','configs/controlled_deepmath20.json']]
    source_files += [ROOT/'verl/verl'/p for p in ['utils/opsd.py','utils/opd_metrics.py','utils/research_budget.py',
                      'trainer/main_ppo.py','trainer/ppo/core_algos.py','trainer/ppo/ray_trainer.py',
                      'workers/actor/dp_actor.py','workers/fsdp_workers.py','workers/config/rollout.py']]
    source_files += list((ROOT/'analysis').rglob('*.py'))
    source_files += [ROOT/p for p in ['configs/analysis_deepmath.json','run_base_comparison.sh',
                                    'configs/research/plain_completion.jinja',
                                    'verl/verl/utils/research_monitor.py','verl/verl/workers/config/model.py',
                                    'verl/verl/trainer/config/model/hf_model.yaml']]
    return {'source_hashes': {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(source_files)},
            'config': config, 'config_hash': digest(config), 'git_commit': git('rev-parse', 'HEAD'),
            'git_diff_hash': digest(git('diff', 'HEAD')), 'git_status': git('status', '--short'),
            'python': sys.version, 'platform': platform.platform(), 'versions': versions,
            'data_hashes': {str(p): file_hash(p) for p in files},
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def checkpoint_identity(path):
    """Content hash, including weight shards, prevents stale activation cache hits."""
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f'Local checkpoint missing: {p}; no implicit download')
    files = sorted(x for x in p.iterdir() if x.is_file() and
                   (x.suffix in ('.json', '.safetensors', '.bin', '.pt') or 'tokenizer' in x.name))
    if not files:
        raise ValueError(f'No checkpoint artifacts in {p}')
    return digest({x.name: file_hash(x) for x in files})
