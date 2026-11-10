"""Live behavior is kept separate from teacher-forced geometry and token confidence."""
import re
import json
from pathlib import Path
import time

import numpy as np

from scripts.research.common import ROOT, digest, file_hash, seed_all
from scripts.research.data import verify


def answer_status(text, ground_truth, benchmark, truncated=False, row=None):
    # Same extraction and equivalence implementation as the training rule reward.
    if benchmark == 'controlled_modular':
        extracted = re.findall(r'\\boxed\{\s*([-+]?\d+)\s*\}', text)
        extracted = extracted[-1] if extracted else ''
    else:
        from scripts.research.answer_protocol import evaluate_answer
        return evaluate_answer(text, ground_truth, truncated, row=row)
    failed = not bool(extracted)
    return dict(extracted_answer=extracted, extraction_failed=failed,
                correct=bool(not truncated and not failed and verify(text, ground_truth, benchmark)),
                verifier_revision=file_hash(ROOT/'scripts/val/eval/utils.py'))


def generation_limit(config, pool):
    return config.get('behavior_max_response_tokens', config['max_probe_response_tokens']) if pool == 'live_rollout' else config['max_probe_response_tokens']


def score_row(text, row, truncated):
    # Synthetic integer fixtures retain their separate, explicit benchmark protocol.
    if row['benchmark'] == 'controlled_modular':
        return answer_status(text, row['ground_truth'], row['benchmark'], truncated)
    return answer_status(text, row['ground_truth'], row['benchmark'], truncated, row=row)


def generation_record(model, row, config, pool, step, seed, ids, text, truncated):
    status = score_row(text, row, truncated)
    grams = [tuple(ids[i:i+4]) for i in range(max(0, len(ids)-3))]
    return {**row, **status, 'pool':pool, 'source_optimizer_step':step, 'prompt_ids':model.prompt(row),
            'response_ids':ids, 'response_text':text, 'response_length':len(ids),
            'eos':not truncated, 'truncated':truncated,
            'finish_reason':'length' if truncated else 'eos',
            'geometry_window_clipped':len(ids)>config['max_probe_response_tokens'],
            'seed':seed, 'decoding':dict(greedy=pool!='fixed_base_rollout' and config.get('live_greedy',True),
                                      max_new_tokens=generation_limit(config,pool),
                                      temperature=1. if pool=='fixed_base_rollout' or not config.get('live_greedy',True) else None,
                                      top_p=1.),
            'repeated_4gram_rate':1-len(set(grams))/len(grams) if grams else None}


def rollout(model, row, config, pool, step):
    seed = int(digest([config['seed'], row['problem_id'], pool])[:8], 16)
    seed_all(seed)
    fixed = pool == 'fixed_base_rollout'
    greedy = not fixed and config.get('live_greedy', True)
    ids, text, truncated = model.generate(row, max_tokens=generation_limit(config,pool),
                                         temperature=1., top_p=1., greedy=greedy)
    return generation_record(model,row,config,pool,step,seed,ids,text,truncated)


def length_grouped_batches(rows, batch_size, length_hints):
    """Partition every row exactly once; long answers never share short-answer buckets."""
    buckets={}
    for index,row in enumerate(rows):
        length=length_hints[row['problem_id']]
        if isinstance(length,bool) or not isinstance(length,int) or length<0:
            raise ValueError('Generation length hints must be nonnegative integers')
        bucket=max(512,1 << max(0,length-1).bit_length())
        buckets.setdefault(bucket,[]).append((length,index))
    result=[]
    for bucket,values in sorted(buckets.items()):
        indices=[i for _,i in sorted(values)]
        result.extend(indices[start:start+batch_size] for start in range(0,len(indices),batch_size))
    return result


def rollout_many(model, rows, config, pool, step, cache_dir, progress_path, length_hints=None):
    """Recoverable batched monitoring; preserve row order, seeds and decoding metadata."""
    from scripts.research.common import write_json
    from .common import source_hashes

    batch_size = config.get('generation_batch_size', 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('generation_batch_size must be a positive integer')
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    attention_backend=config.get('generation_attention_backend')
    engine=config.get('generation_engine','hf')
    if engine not in ('hf','vllm'):
        raise ValueError('Unsupported monitoring generation engine')
    bucketed=config.get('generation_length_bucketing',False)
    backend = ('hf_'+(attention_backend or 'eager')+'_length_buckets_v2' if bucketed or attention_backend
               else 'hf_batched_independent_multinomial_v1')
    if engine == 'vllm':
        if pool == 'fixed_base_rollout' or not config.get('live_greedy',True):
            raise ValueError('The vLLM monitoring profile supports live greedy evaluation only')
        backend = 'vllm_continuous_greedy_v1'
    identity = dict(model=model.identity, rows_hash=digest(rows), monitor=config, pool=pool, step=step,
                    sources=source_hashes(), model_adapter_hash=file_hash(ROOT/'scripts/research/models.py'),
                    backend=backend)
    schedule=None
    if bucketed:
        if length_hints is None:
            length_hints={r['problem_id']:int(r.get('reference_token_length',len(model.prompt(r)))) for r in rows}
        length_hints={r['problem_id']:length_hints[r['problem_id']] for r in rows}
        schedule=length_grouped_batches(rows,batch_size,length_hints)
        identity.update(length_hints_sha256=digest(length_hints),generation_schedule=schedule)
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
    if engine == 'vllm' and pending:
        from .vllm_generation import generate
        seeds = [int(digest([config['seed'], row['problem_id'], pool])[:8], 16) for row in rows]
        def save_vllm(i, generated):
            if results[i] is not None:
                return
            ids, text, truncated = generated
            result = generation_record(model, rows[i], config, pool, step, seeds[i], ids, text, truncated)
            result['decoding'].update(backend=backend,batch_size=64,attention_backend='vllm_flash_attention',
                                      continuous_batching=True)
            write_json(directory/f'{i:06d}.json', dict(row_hash=digest(rows[i]),result_hash=digest(result),result=result))
            results[i] = result
            progress()
        generate(model,rows,seeds,generation_limit(config,pool),directory,save_vllm,
                 config.get('generation_gpu_memory_utilization',.80))
        return results
    if not pending:
        return results
    batches=schedule if schedule is not None else [pending[start:start+batch_size] for start in range(0,len(pending),batch_size)]
    for indices in batches:
        if all(results[i] is not None for i in indices):
            continue
        selected = [rows[i] for i in indices]
        seeds = [int(digest([config['seed'], row['problem_id'], pool])[:8], 16) for row in selected]
        fixed = pool == 'fixed_base_rollout'
        greedy = not fixed and config.get('live_greedy', True)
        if batch_size == 1 and attention_backend is None:
            seed_all(seeds[0])
            generated = [model.generate(selected[0], max_tokens=generation_limit(config,pool),
                                        temperature=1.,top_p=1.,greedy=greedy)]
        else:
            kwargs={'attention_backend':attention_backend} if attention_backend is not None else {}
            generated = model.generate_batch(selected, seeds, max_tokens=generation_limit(config,pool),
                                             temperature=1., top_p=1., greedy=greedy,**kwargs)
        if len(generated) != len(indices):
            raise ValueError('Batched generator returned the wrong number of responses')
        for i, row, seed, (ids, text, truncated) in zip(indices, selected, seeds, generated):
            result = generation_record(model,row,config,pool,step,seed,ids,text,truncated)
            result['decoding'].update(backend=backend,batch_size=len(indices),attention_backend=attention_backend or 'eager')
            if results[i] is not None:
                if digest(results[i])!=digest(result):
                    raise ValueError('Regenerated partial-batch result differs from its cached record')
                continue
            write_json(directory/f'{i:06d}.json',
                       dict(row_hash=digest(row), result_hash=digest(result), result=result))
            results[i] = result
        progress()
    return results


def behavior(rows):
    if not rows:
        return {}
    topics = sorted({r['topic_label'] for r in rows})
    result = dict(accuracy_micro=float(np.mean([r['correct'] for r in rows])),
                accuracy_topic_macro=float(np.mean([np.mean([r['correct'] for r in rows if r['topic_label']==t])
                                                   for t in topics])),
                response_length=float(np.mean([r['response_length'] for r in rows])),
                eos_rate=float(np.mean([r['eos'] for r in rows])),
                truncation_rate=float(np.mean([r['truncated'] for r in rows])),
                extraction_failure_rate=float(np.mean([r['extraction_failed'] for r in rows])))
    if all('format_compliant' in r for r in rows):
        result['format_compliance_rate']=float(np.mean([r['format_compliant'] for r in rows]))
    return result


def enforce_quality(rows, config, output):
    """Persist complete diagnostics before stopping. No retries or sample removal."""
    from collections import Counter
    from scripts.research.common import write_json, write_jsonl
    settings = config.get('quality_gate',{})
    if not settings.get('enabled',False):
        return
    limit=settings.get('max_extraction_failure_rate',.01)
    splits=settings.get('splits',['heldout_probe'])
    if isinstance(limit,bool) or not isinstance(limit,(int,float)) or not np.isfinite(limit) or not 0<=limit<=1:
        raise ValueError('Invalid quality-gate threshold')
    if not isinstance(splits,list) or not splits or any(s not in ('heldout_probe','train_probe') for s in splits):
        raise ValueError('Quality gate requires nonempty probe splits')
    output=Path(output)
    summaries=[]
    for split in splits:
        selected=[r for r in rows if r['split']==split]
        n=len(selected)
        failed=sum(r['extraction_failed'] for r in selected)
        passed=bool(n) and failed/n<=limit
        summaries.append(dict(split=split,n=n,failures=failed,successes=n-failed,
                              extraction_success_rate=(n-failed)/n if n else None,
                              minimum_success_rate=1-limit,
                              extraction_failure_rate=failed/n if n else None,passed=passed,
                              parse_status_counts=dict(Counter(r.get('parse_status','legacy') for r in selected))))
    write_jsonl(output/'responses.jsonl',rows)
    write_jsonl(output/'failed_responses.jsonl',[r for r in rows if r['extraction_failed']])
    write_json(output/'quality.json',dict(settings=settings,splits=summaries,passed=all(s['passed'] for s in summaries)))
    if not all(s['passed'] for s in summaries):
        raise ValueError(f'Answer extraction quality gate failed; inspect {output}/quality.json and failed_responses.jsonl. Training must not continue.')
