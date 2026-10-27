"""Live behavior is kept separate from teacher-forced geometry and token confidence."""
import re
import json
from pathlib import Path
import time

import numpy as np

from scripts.research.common import ROOT, digest, file_hash, seed_all
from scripts.research.data import verify


def answer_status(text, ground_truth, benchmark, truncated=False):
    # Same extraction and equivalence implementation as the training rule reward.
    if benchmark == 'controlled_modular':
        extracted = re.findall(r'\\boxed\{\s*([-+]?\d+)\s*\}', text)
        extracted = extracted[-1] if extracted else ''
    else:
        import sys
        directory = str(ROOT/'scripts/val/eval')
        if directory not in sys.path:
            sys.path.insert(0, directory)
        from utils import extract_answer
        extracted = extract_answer(text)
    failed = not bool(extracted)
    return dict(extracted_answer=extracted, extraction_failed=failed,
                correct=bool(not truncated and not failed and verify(text, ground_truth, benchmark)),
                verifier_revision=file_hash(ROOT/'scripts/val/eval/utils.py'))


def rollout(model, row, config, pool, step):
    seed = int(digest([config['seed'], row['problem_id'], pool])[:8], 16)
    seed_all(seed)
    fixed = pool == 'fixed_base_rollout'
    greedy = not fixed and config.get('live_greedy', True)
    ids, text, truncated = model.generate(row, max_tokens=config['max_probe_response_tokens'],
                                         temperature=1., top_p=1., greedy=greedy)
    status = answer_status(text, row['ground_truth'], row['benchmark'], truncated)
    grams = [tuple(ids[i:i+4]) for i in range(max(0, len(ids)-3))]
    return {**row, **status, 'pool':pool, 'source_optimizer_step':step, 'prompt_ids':model.prompt(row),
            'response_ids':ids, 'response_text':text, 'response_length':len(ids),
            'eos':not truncated, 'truncated':truncated, 'seed':seed,
            'decoding':dict(greedy=greedy, temperature=None if greedy else 1., top_p=1.),
            'repeated_4gram_rate':1-len(set(grams))/len(grams) if grams else None}


def rollout_many(model, rows, config, pool, step, cache_dir, progress_path):
    """Recoverable batched monitoring; preserve row order, seeds and decoding metadata."""
    from scripts.research.common import write_json
    from .common import source_hashes

    batch_size = config.get('generation_batch_size', 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('generation_batch_size must be a positive integer')
    if batch_size == 1:
        return [rollout(model, row, config, pool, step) for row in rows]
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    backend = 'hf_batched_independent_multinomial_v1'
    identity = dict(model=model.identity, rows_hash=digest(rows), monitor=config, pool=pool, step=step,
                    sources=source_hashes(), model_adapter_hash=file_hash(ROOT/'scripts/research/models.py'),
                    backend=backend)
    manifest = directory/'identity.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('Generation cache identity changed; use a new run/cache')
    else:
        if any(directory.iterdir()):
            raise ValueError('Generation cache has files but no identity')
        write_json(manifest, identity)
    results, pending = [None]*len(rows), []
    for i, row in enumerate(rows):
        path = directory/f'{i:06d}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['row_hash'] != digest(row) or saved['result_hash'] != digest(saved['result']):
                raise ValueError(f'Generation cache item changed: {path}')
            results[i] = saved['result']
        else:
            pending.append(i)
    started, initial = time.monotonic(), len(rows)-len(pending)

    def progress():
        completed = sum(r is not None for r in results)
        elapsed = time.monotonic()-started
        rate = (completed-initial)/elapsed if elapsed > 0 else 0
        value = dict(pool=pool, optimizer_step=step, completed=completed, total=len(rows),
                     batch_size=batch_size, backend=backend, elapsed_seconds=elapsed,
                     remaining_seconds=(len(rows)-completed)/rate if rate > 0 else None,
                     updated_unix=time.time(), status='completed' if completed == len(rows) else 'running')
        write_json(progress_path, value)
        print(f'GENERATION_PROGRESS pool={pool} step={step} completed={completed}/{len(rows)} '
              f'batch={batch_size} elapsed_seconds={elapsed:.1f}', flush=True)

    progress()
    for start in range(0, len(pending), batch_size):
        indices = pending[start:start+batch_size]
        selected = [rows[i] for i in indices]
        seeds = [int(digest([config['seed'], row['problem_id'], pool])[:8], 16) for row in selected]
        fixed = pool == 'fixed_base_rollout'
        greedy = not fixed and config.get('live_greedy', True)
        generated = model.generate_batch(selected, seeds, max_tokens=config['max_probe_response_tokens'],
                                         temperature=1., top_p=1., greedy=greedy)
        if len(generated) != len(indices):
            raise ValueError('Batched generator returned the wrong number of responses')
        for i, row, seed, (ids, text, truncated) in zip(indices, selected, seeds, generated):
            checked = answer_status(text, row['ground_truth'], row['benchmark'], truncated)
            grams = [tuple(ids[j:j+4]) for j in range(max(0, len(ids)-3))]
            result = {**row, **checked, 'pool':pool, 'source_optimizer_step':step,
                      'prompt_ids':model.prompt(row), 'response_ids':ids, 'response_text':text,
                      'response_length':len(ids), 'eos':not truncated, 'truncated':truncated, 'seed':seed,
                      'decoding':dict(greedy=greedy, temperature=None if greedy else 1., top_p=1.,
                                      backend=backend, batch_size=batch_size),
                      'repeated_4gram_rate':1-len(set(grams))/len(grams) if grams else None}
            write_json(directory/f'{i:06d}.json',
                       dict(row_hash=digest(row), result_hash=digest(result), result=result))
            results[i] = result
        progress()
    return results


def behavior(rows):
    if not rows:
        return {}
    topics = sorted({r['topic_label'] for r in rows})
    return dict(accuracy_micro=float(np.mean([r['correct'] for r in rows])),
                accuracy_topic_macro=float(np.mean([np.mean([r['correct'] for r in rows if r['topic_label']==t])
                                                   for t in topics])),
                response_length=float(np.mean([r['response_length'] for r in rows])),
                eos_rate=float(np.mean([r['eos'] for r in rows])),
                truncation_rate=float(np.mean([r['truncated'] for r in rows])),
                extraction_failure_rate=float(np.mean([r['extraction_failed'] for r in rows])))
