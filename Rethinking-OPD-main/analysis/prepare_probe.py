"""Freeze normalized-question splits, raw prompt/response IDs and anchor membership."""
from collections import Counter
import copy
import json
from pathlib import Path
import random
import re
import time
import unicodedata

from scripts.research.common import checkpoint_identity, digest, file_hash, opsd, provenance, write_json, write_jsonl
from scripts.research.data import load_questions, prepare_deepmath
from scripts.research.models import Model
from .common import config_argument, load_config, source_hashes
from .evaluate import answer_status, rollout_many, enforce_quality
from scripts.research.preparation_progress import PreparationProgress, preparation_workspace


def normalized_question(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


def stratified_take(rows, count, seed):
    """Deterministic round-robin coverage over (topic, floor difficulty) strata."""
    if count > len(rows) or count < 0:
        raise ValueError(f'Requested {count} rows from {len(rows)} eligible questions')
    groups = {}
    for row in sorted(rows, key=lambda r:r['problem_id']):
        groups.setdefault((row['topic_label'], int(row['difficulty'])), []).append(row)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < count:
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                if len(selected) == count:
                    break
    chosen = {r['problem_id'] for r in selected}
    return selected, [r for r in rows if r['problem_id'] not in chosen]


def split_rows(rows, c):
    unique, duplicates = {}, []
    level = c['dataset']['topic_level']
    for raw in rows:
        sid = digest(normalized_question(raw['question']))[:20]
        if sid in unique:
            if raw['ground_truth'] != unique[sid]['ground_truth']:
                raise ValueError(f'Conflicting answers for normalized duplicate {sid}')
            duplicates.append(dict(sample_id=sid, original_id=raw['problem_id']))
            continue
        if not isinstance(raw.get('topic'),str) or not raw['topic'].strip():
            raise ValueError('Native DeepMath rows require a nonempty hierarchical topic field')
        parts = [p.strip() for p in raw['topic'].split('->')]
        row = {**raw, 'original_id':raw['problem_id'], 'problem_id':sid,
               'topic_label':parts[min(level-1, len(parts)-1)], 'topic_label_level':min(level,len(parts)),
               'messages':opsd.plain_messages(raw['question'])}
        unique[sid] = row
    remaining = list(unique.values())
    splits = {}
    for offset, (name, count) in enumerate((('heldout_probe', c['monitor']['heldout_probe_n']),
                                           ('validation', c['dataset']['dev_n']),
                                           ('train', c['dataset']['train_n']))):
        selected, remaining = stratified_take(remaining, count, c['monitor']['seed']+offset)
        splits[name] = [{**r, 'split':name} for r in selected]
    selected, _ = stratified_take(splits['train'], c['monitor']['train_probe_n'], c['monitor']['seed']+3)
    splits['train_probe'] = [{**r, 'split':'train_probe'} for r in selected]
    for key in splits:
        splits[key].sort(key=lambda r:(r['topic_label'], r['difficulty'], r['problem_id']))
    return splits, duplicates


def freeze(c, model=None):
    started=time.monotonic()
    root = Path(c['output_root'])/'probe'
    progress = PreparationProgress(root.parent/'preparation_progress.json')
    try:
        identity = dict(config_hash=digest(c), input_hash=file_hash(c['dataset']['input']), sources=source_hashes(),
                        checkpoint_hash=checkpoint_identity(c['model']['path']))
        marker = root/'complete.json'
        if marker.exists():
            previous = json.loads(marker.read_text())
            if previous['identity'] != identity:
                raise ValueError('Frozen probe/config/source changed; use a new output_root')
            for name, expected in previous['files'].items():
                if file_hash(root/name) != expected:
                    raise ValueError(f'Frozen probe file changed: {name}')
            progress.phase('complete',reused_probe=True,accepted=None,excluded=None)
            progress.finish()
            return root
        root.parent.mkdir(parents=True, exist_ok=True)
        if root.exists():
            raise ValueError('Probe directory is incomplete; inspect it before choosing a new output_root')
        progress.phase('model_loading')
        if model is None:
            model = Model(c['model']['path'], c['model']['revision'], device=c['model']['device'],
                          dtype=c['model']['dtype'], prompt_mode='plain_completion')
        with preparation_workspace(root.parent) as out:
            progress.phase('selection_cache', workspace=str(out))
            from scripts.research.preparation_cache import restore, save_selection
            selection_checkpoint = root.parent/'preparation_selection'
            restore_config = c
            if not c['dataset'].get('preparation_cache') and (selection_checkpoint/'selection_complete.json').is_file():
                restore_config = copy.deepcopy(c)
                restore_config['dataset']['preparation_cache'] = str(selection_checkpoint)
            cached = restore(restore_config, out)
            if cached is not None:
                splits, duplicates = cached
                cache_receipt = json.loads((Path(restore_config['dataset']['preparation_cache'])/'selection_complete.json').read_text())
                counts = cache_receipt.get('screening_counts',{})
                progress.value.update(accepted=counts.get('accepted'),excluded=counts.get('excluded'),
                                      exclusion_counts=counts.get('exclusion_counts',{}),reused_selection=True)
            else:
                progress.phase('native_read')
                native = load_questions(c['dataset']['input'],require_unique=False)
                # Native duplicate problem IDs are made unique BEFORE the adapter's ID
                # assertion; normalized question hash below is the actual split identity.
                native = [{**r,'source_problem_id':r['problem_id'],'problem_id':f"{r['problem_id']}__row{i}"}
                          for i,r in enumerate(native)]
                progress.phase('native_adaptation',total=len(native))
                rows = prepare_deepmath(native, out/'eligible.jsonl', c['dataset']['revision'],
                                        c['dataset']['difficulty_min'], c['dataset']['difficulty_max'],
                                        solution_field=c['dataset']['reference_field'], tokenizer=model.tokenizer,
                                        progress=lambda done,total,n: progress.update(done,total,accepted=n,excluded=done-n))
                from scripts.research.answer_protocol import answer_metadata
                accepted, checks, quarantined = [], [], []
                exclusion_counts = Counter()
                progress.phase('eligibility_checks',total=len(rows),accepted=0,excluded=0,exclusion_counts={})
                for index,row in enumerate(rows):
                    progress.update(index,len(rows),accepted=len(accepted),excluded=index-len(accepted),
                                    exclusion_counts=dict(exclusion_counts),current_problem_id=row['problem_id'])
                    metadata = answer_metadata(row)
                    row = {**row, **metadata}
                    if metadata['answer_encoding_status'] != 'ok':
                        quarantined.append(row)
                        exclusion_counts[metadata['answer_encoding_status']] += 1
                        checks.append(dict(problem_id=row['problem_id'],reference_verified=False,
                                           student_tokens=None,teacher_tokens=None,
                                           exclusion_reason=metadata['answer_encoding_status']))
                        continue
                    reference = answer_status(row['privilege_text'], row['ground_truth'], row['benchmark'], row=row)
                    row = {**row, 'reference_correctness':reference['correct'], 'messages':opsd.plain_messages(row['question'])}
                    p, q = model.prompt(row), model.prompt(row, True)
                    reason = None
                    if c['dataset'].get('require_verified_reference', True) and not reference['correct']:
                        reason = 'reference_final_answer_not_verified'
                    elif len(p) > c['training']['max_prompt_length'] or len(q) > c['training']['max_teacher_prompt_length']:
                        reason = 'overlong_prompt'
                    elif max(len(p), len(q))+c['training']['max_response_length'] > model.model.config.max_position_embeddings:
                        reason = 'context_limit'
                    checks.append(dict(problem_id=row['problem_id'], reference_verified=reference['correct'],
                                       student_tokens=len(p), teacher_tokens=len(q), exclusion_reason=reason))
                    if reason is None:
                        accepted.append(row)
                    else:
                        exclusion_counts[reason] += 1
                progress.update(len(rows),len(rows),accepted=len(accepted),excluded=len(rows)-len(accepted),
                                exclusion_counts=dict(exclusion_counts),current_problem_id=None,force=True)
                write_jsonl(out/'eligibility.jsonl', checks)
                write_jsonl(out/'answer_encoding_quarantine.jsonl', quarantined)
                # Keep selection diagnostics even when too few valid rows remain.
                write_jsonl(root.parent/'preparation_audit/answer_encoding_quarantine.jsonl',quarantined)
                write_jsonl(root.parent/'preparation_audit/eligibility.jsonl',checks)
                progress.phase('split_freeze')
                splits, duplicates = split_rows(accepted, c)
                write_jsonl(out/'duplicates.jsonl', duplicates)
                for name, values in splits.items():
                    write_jsonl(out/f'{name}.jsonl', values)
                progress.phase('selection_checkpoint')
                save_selection(out,c,selection_checkpoint)
            progress.value['screening_completed'] = True
            progress.value['selection_counts'] = {name:len(values) for name,values in splits.items()}
            probes = splits['heldout_probe']+splits['train_probe']
            if c['monitor'].get('quality_gate',{}).get('enabled',False):
                progress.phase('baseline_generation',total=len(probes))
                baseline = rollout_many(model, probes, c['monitor'], 'live_rollout', 0,
                                        root.parent/'generation_cache/baseline_live', root.parent/'generation_progress.json')
                progress.update(len(baseline),len(probes),force=True)
                progress.phase('baseline_quality',total=len(probes))
                enforce_quality(baseline,c['monitor'],root.parent/'quality/baseline')
                progress.update(len(baseline),len(probes),force=True)
                write_jsonl(out/'baseline_live_rollouts.jsonl',baseline)
            progress.phase('fixed_pool_generation',total=len(probes))
            fixed = rollout_many(model, probes, c['monitor'], 'fixed_base_rollout', 0,
                                 root.parent/'generation_cache/fixed_base_rollout', root.parent/'generation_progress.json')
            progress.update(len(fixed),len(probes),force=True)
            progress.phase('publishing_probe')
            for result in fixed:
                result['anchor_membership'] = {a:a=='prompt_last' or len(result['response_ids']) >= int(a.split('_')[1])
                                                for a in c['monitor']['anchors']}
                result['position_index'] = {a:len(result['prompt_ids'])-1+(0 if a=='prompt_last' else int(a.split('_')[1]))
                                            for a,valid in result['anchor_membership'].items() if valid}
            write_jsonl(out/'fixed_pool.jsonl', fixed)
            if c['monitor'].get('additional_fixed_reference'):
                references = []
                for row in probes:
                    if not row['reference_correctness']:
                        continue
                    # Tokenize FULL string and find the exact boundary. Reject boundary-spanning merges.
                    text = opsd.plain_messages(row['question'])[0]['content']
                    prompt = model.tokenizer.encode(text, add_special_tokens=False)
                    full = model.tokenizer.encode(text+row['privilege_text'], add_special_tokens=False)
                    if full[:len(prompt)] != prompt:
                        raise ValueError('Reference token crosses prompt boundary; choose a tokenizer-stable prompt suffix')
                    ids = full[len(prompt):][:c['monitor']['max_probe_response_tokens']]
                    references.append({**row, 'pool':'fixed_reference', 'prompt_ids':prompt, 'response_ids':ids,
                                       'source_optimizer_step':None, 'reference_truncated':len(full)-len(prompt)>len(ids),
                                       'anchor_membership':{a:a=='prompt_last' or len(ids)>=int(a.split('_')[1])
                                                            for a in c['monitor']['anchors']}})
                write_jsonl(out/'fixed_reference.jsonl', references)
            manifest = [{k:r[k] for k in ('problem_id','split','topic_label','difficulty','anchor_membership')}
                        | {'prompt_hash':digest(r['prompt_ids']), 'response_hash':digest(r['response_ids'])} for r in fixed]
            write_jsonl(out/'probe_manifest.jsonl', manifest)
            write_json(out/'manifest.json', dict(identity=identity, model=model.identity, dataset=c['dataset'],
                       architecture=dict(blocks=len(model.layers()),hidden_size=model.model.config.hidden_size,
                                         vocab_size=model.model.config.vocab_size),
                       strata='round_robin(topic_at_fixed_level, floor(difficulty))',
                       counts={k:len(v) for k,v in splits.items()}, duplicate_count=len(duplicates),
                       preparation_wall_seconds=time.monotonic()-started,
                       topic_status='nondegenerate' if len({r['topic_label'] for r in probes})>1 else 'degenerate',
                       source_correctness='fixed_pool correctness belongs to step0 generation; not current encoding checkpoint',
                       provenance=provenance(c)))
            files = {str(p.relative_to(out)):file_hash(p) for p in out.rglob('*') if p.is_file()}
            write_json(out/'complete.json', dict(identity=identity, files=files, probe_manifest_hash=file_hash(out/'probe_manifest.jsonl')))
            out.rename(root)
        progress.phase('complete')
        progress.finish()
        return root
    except BaseException as error:
        progress.finish(error)
        raise


def main():
    args = config_argument(__doc__).parse_args()
    print(freeze(load_config(args.config)))


if __name__ == '__main__':
    main()
