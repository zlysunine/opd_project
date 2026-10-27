"""Content-addressed manifests and strict, recoverable analysis tables."""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from scripts.research.common import ROOT, file_hash, write_jsonl


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def table(path, rows, format='parquet'):
    path = Path(path)
    rows = clean(rows)
    write_jsonl(path.with_suffix('.jsonl'), rows)
    if format == 'parquet':
        import pyarrow as pa
        import pyarrow.parquet as pq
        keys = sorted(set().union(*(r.keys() for r in rows))) if rows else []
        normalized = [{key:row.get(key) for key in keys} for row in rows]
        pq.write_table(pa.Table.from_pylist(normalized), path.with_suffix('.parquet'))
    elif format != 'jsonl':
        raise ValueError('table_format must be parquet or jsonl (explicit CPU smoke mode)')


def csv_table(path, rows):
    keys = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(clean(rows))


def load_config(path):
    c = json.loads(Path(path).read_text())
    from scripts.research.answer_protocol import VERSION
    if c['dataset'].get('answer_protocol',VERSION)!=VERSION:
        raise ValueError('Unsupported answer protocol; do not reuse another protocol configuration')
    batch_size = c['monitor'].get('generation_batch_size', 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('monitor.generation_batch_size must be a positive integer')
    if c['model']['name_or_path'] != 'Qwen/Qwen3-1.7B-Base':
        raise ValueError('This experiment explicitly requires Qwen/Qwen3-1.7B-Base')
    for key in ('model', 'dataset'):
        if not c[key].get('revision') or c[key]['revision'] in ('main', 'master'):
            raise ValueError(f'Lock an immutable {key} revision')
    if c['model'].get('prompt_mode') != 'plain_completion':
        raise ValueError('Base monitoring requires the recorded plain completion protocol')
    if c['monitor'].get('heavy_metrics'):
        raise ValueError('Gride/ADP/ARI are not implemented; leave heavy_metrics empty')
    monitor=c['monitor']
    for field in ('max_probe_response_tokens','behavior_max_response_tokens'):
        value=monitor.get(field,monitor['max_probe_response_tokens'])
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(f'monitor.{field} must be a positive integer')
    gate=monitor.get('quality_gate',{})
    if not isinstance(gate,dict) or not isinstance(gate.get('enabled',False),bool):
        raise ValueError('quality_gate must specify a boolean enabled flag')
    if gate.get('enabled',False):
        limit=gate.get('max_extraction_failure_rate',.01)
        if isinstance(limit,bool) or not isinstance(limit,(int,float)) or not np.isfinite(limit) or not 0<=limit<=1:
            raise ValueError('Quality failure-rate limit must be finite and in [0,1]')
        if not monitor.get('live_generation',True):
            raise ValueError('Quality gate requires live_generation')
        splits=gate.get('splits',['heldout_probe'])
        if not isinstance(splits,list) or not splits or any(s not in ('heldout_probe','train_probe') for s in splits):
            raise ValueError('Quality gate requires explicit nonempty probe splits')
    if 'distribution_positions' in monitor:
        raise ValueError('distribution_positions is obsolete; use distribution_scope=all_valid_tokens')
    if monitor.get('distribution_scope','all_valid_tokens')!='all_valid_tokens':
        raise ValueError('Paper-compatible distributions require all_valid_tokens')
    if not isinstance(monitor.get('compare_to_initial',True),bool):
        raise ValueError('compare_to_initial must be a boolean')
    if not isinstance(monitor.get('compare_student_teacher',False),bool):
        raise ValueError('compare_student_teacher must be a boolean')
    for field in ('overlap_top_k','distribution_prefill_chunk_size','distribution_forward_chunk_size','state_chunk_size'):
        value=monitor.get(field,16 if field in ('overlap_top_k','state_chunk_size') else 256)
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(f'monitor.{field} must be a positive integer')
    for field in ('knn_k','token_top_ks'):
        if any(not isinstance(v,int) or isinstance(v,bool) or v<1 for v in monitor[field]):
            raise ValueError(f'{field} must contain positive integers')
    if any(a!='prompt_last' and not re.fullmatch(r'response_[1-9][0-9]*',a) for a in monitor['anchors']):
        raise ValueError('Anchors must be prompt_last or response_<positive integer>')
    if 'prompt_last' not in monitor['anchors'] or monitor['max_probe_response_tokens']<1 or monitor['curvature_eps']<=0:
        raise ValueError('Require prompt_last, positive response window and positive curvature epsilon')
    if c['dataset']['topic_level']<1 or c['dataset']['train_n']<1 or c['dataset']['dev_n']<1:
        raise ValueError('Topic level and split counts must be positive')
    if not 0 < monitor['train_probe_n'] <= c['dataset']['train_n'] or monitor['heldout_probe_n']<1:
        raise ValueError('Require a nonempty heldout probe and train_probe within the train set')
    training=c['training']
    expected=('grpo','none') if c.get('method')=='grpo' else ('forward_kl','current_batch_stop_gradient')
    if (training['loss_type'],training['teacher_update_policy'])!=expected:
        raise ValueError('Preserve OPSD forward KL/current-batch teacher, or outcome-only GRPO/no training teacher')
    if training['training_mode']!='lora' or training['lora_rank']<1 or training['lora_alpha']<=0:
        raise ValueError('This paired experiment requires positive-rank LoRA')
    for field in ('batch','rollouts','microbatch','steps','epochs','save_every','gpus',
                  'max_prompt_length','max_response_length','max_teacher_prompt_length'):
        if not isinstance(training[field],int) or isinstance(training[field],bool) or training[field]<1:
            raise ValueError(f'training.{field} must be a positive integer')
    if training['learning_rate']<=0 or monitor['state_chunk_size']<1:
        raise ValueError('Learning rate and state chunk size must be positive')
    c['output_root'] = str(Path(c['output_root']).resolve())
    return c


def config_argument(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', required=True)
    return parser


def source_hashes():
    paths=list((ROOT/'analysis').rglob('*.py'))
    paths += [ROOT/'scripts/research/answer_protocol.py',ROOT/'scripts/val/eval/utils.py',
              ROOT/'scripts/research/models.py',ROOT/'scripts/research/data.py',ROOT/'scripts/research/reward.py',
              ROOT/'configs/research/plain_completion.jinja',ROOT/'verl/verl/utils/opsd.py']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def step_dir(c, step):
    return Path(c['output_root'])/'steps'/f'step_{step:06d}'


def finite_mean(values):
    v = np.asarray([np.nan if x is None else x for x in values], dtype=float)
    mask = np.isfinite(v)
    return (float(v[mask].mean()) if mask.any() else np.nan, int(mask.sum()))
