"""Canonical question/reference schema, controlled modular skill and fair intersections."""
import importlib.util
import random
import re
import sys
from pathlib import Path

from .common import ROOT, digest, write_json, write_jsonl, read_jsonl


def canonical(row, index=0, split='train'):
    question = row.get('question')
    messages = row.get('prompt')
    if question is None:
        if not isinstance(messages, list):
            raise ValueError('Expected question or prompt chat messages')
        question = '\n'.join(m['content'] for m in messages if m['role']=='user')
    if not isinstance(question, str):
        raise ValueError('Question must be text')
    gt = row.get('ground_truth', row.get('answer', row.get('final_answer', (row.get('reward_model') or {}).get('ground_truth'))))
    if gt is None:
        raise ValueError(f'Missing ground_truth at row {index}')
    # Identity depends on question, never split or reference source.
    pid = row.get('problem_id', digest(question)[:20])
    return {**row, 'problem_id': str(pid), 'split': row.get('split', (row.get('extra_info') or {}).get('split', split)), 'question': question,
            'messages': row.get('messages', messages or [{'role': 'user', 'content': question}]),
            'ground_truth': str(gt), 'privilege_text': row.get('privilege_text', ''),
            'privilege_type': row.get('privilege_type', 'empty'),
            'generator_model': row.get('generator_model'), 'generator_revision': row.get('generator_revision'),
            'generation_seed': row.get('generation_seed'), 'reference_correctness': row.get('reference_correctness'),
            'reference_token_length': row.get('reference_token_length', 0),
            'generation_config': row.get('generation_config', {}),
            'benchmark': row.get('benchmark', row.get('data_source', 'controlled_modular'))}


def load_questions(path, split=None, require_unique=True):
    if Path(path).suffix == '.parquet' or Path(path).is_dir():
        import pandas as pd
        rows = pd.read_parquet(path).to_dict('records')
        # Arrow nested arrays -> plain JSON types.
        import numpy as np
        def plain(x):
            if isinstance(x, np.ndarray): return [plain(v) for v in x]
            if isinstance(x, dict): return {k: plain(v) for k,v in x.items()}
            if isinstance(x, list): return [plain(v) for v in x]
            if isinstance(x, np.generic): return x.item()
            return x
        rows = [plain(r) for r in rows]
    elif Path(path).suffix == '.json':
        import json
        rows = json.loads(Path(path).read_text())
        if not isinstance(rows, list):
            raise ValueError('JSON input must be a list; use JSONL otherwise')
    else:
        rows = read_jsonl(path)
    rows = [canonical(row, i, split or 'train') for i,row in enumerate(rows)]
    if require_unique and len({r['problem_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate problem_id')
    return [r for r in rows if split is None or r['split'] == split]


def controlled_dataset(out, count=64, seed=42, modulus=11, steps=3):
    """Unique operation instances across splits; held-out templates and longer steps.

    Intermediate state metadata never enters messages. Reference solution is a
    separate field, available only to SFT targets/privileged teacher.
    """
    if count < 1 or modulus < 2 or steps < 1:
        raise ValueError('Invalid controlled data parameters')
    rng = random.Random(seed)
    seen = set()
    rows = []
    templates = {
        'train': ['Start at {start} mod {mod}; apply {ops}. Final state? Answer as \\boxed{{n}}.',
                  'State={start}, modulus={mod}. Updates: {ops}. Return final \\boxed{{n}}.'],
        'validation': ['Modulo {mod}, initialize {start}, then {ops}. Give final \\boxed{{n}}.'],
        'test_instance': ['Start at {start} mod {mod}; apply {ops}. Final state? Answer as \\boxed{{n}}.',
                          'State={start}, modulus={mod}. Updates: {ops}. Return final \\boxed{{n}}.'],
        'test_template': ['Register begins at {start}. Execute {ops}, reducing mod {mod} each time. Final \\boxed{{n}}?'],
        'test_long': ['Machine initial value {start}; actions {ops}; reduce mod {mod} after each. Final \\boxed{{n}}?'],
    }
    for split, tmps in templates.items():
        for i in range(count):
            for attempt in range(10000):
                start = rng.randrange(modulus)
                actions = [(rng.choice(['+', '*']), rng.randrange(1, modulus)) for _ in range(steps*(2 if split=='test_long' else 1))]
                instance = digest([modulus, start, actions])[:20]
                if instance not in seen:
                    seen.add(instance)
                    break
            else:
                raise ValueError('Instance space exhausted; reduce count or increase modulus/steps')
            values = [start]
            for op, value in actions:
                values.append(((values[-1]+value) if op=='+' else (values[-1]*value)) % modulus)
            template_index = i%len(tmps)
            template = tmps[template_index]
            question = template.format(start=start, mod=modulus, ops=', '.join(f'{o}{v}' for o,v in actions))
            solution = '; '.join(f's{j}={v}' for j,v in enumerate(values)) + f'. Answer: \\boxed{{{values[-1]}}}'
            row = canonical({'question': question, 'problem_id': instance, 'ground_truth': values[-1],
                             'split': split, 'instance_id': instance, 'template_id': digest(template)[:12],
                             'analysis': {'skill': 'modular_updates', 'intermediate_states': values,
                                          'answer_label': values[-1], 'operations': actions, 'modulus': modulus},
                             'oracle_solution': solution, 'probe_split': {'train':'train', 'validation':'validation',
                                          'test_template':'test', 'test_instance':'test', 'test_long':'test'}[split]})
            rows.append(row)
    write_jsonl(out, rows)
    write_json(str(out)+'.manifest.json', {'seed': seed, 'count_per_split': count, 'modulus': modulus,
               'steps': steps, 'data_hash': digest(rows), 'splits': list(templates),
               'probe_note': 'Use train/validation/test_template only for template-disjoint probes'})
    return rows


def verify(text, ground_truth, benchmark='controlled_modular', extra_info=None):
    if benchmark == 'controlled_modular':
        answers = re.findall(r'\\boxed\{\s*([-+]?\d+)\s*\}', text)
        return bool(answers) and int(answers[-1]) == int(ground_truth)
    from .answer_protocol import evaluate_answer
    return evaluate_answer(text, ground_truth, row=extra_info)['correct']


def reference_intersection(paths, output_dir, correct_only=False):
    sources = [load_questions(p) for p in paths]
    maps = [{r['problem_id']: r for r in source} for source in sources]
    common = set.intersection(*(set(m) for m in maps))
    for pid in common:
        if len({digest([m[pid]['question'], m[pid]['split'], m[pid]['ground_truth']]) for m in maps}) != 1:
            raise ValueError(f'Conflicting question/split/answer for {pid}')
    if correct_only:
        common = {pid for pid in common if all(m[pid]['reference_correctness'] is True for m in maps)}
    if not common:
        raise ValueError('No common eligible questions')
    out = Path(output_dir)
    report = []
    for i, (path, source) in enumerate(zip(paths, sources)):
        selected = [r for r in source if r['problem_id'] in common]
        write_jsonl(out/f'source_{i}.jsonl', selected)
        lengths = [r['reference_token_length'] for r in selected]
        report.append({'source': str(path), 'before': len(source), 'after': len(selected),
                       'coverage': len(selected)/len(source), 'lengths': lengths,
                       'correctness_before': [r['reference_correctness'] for r in source]})
    write_json(out/'coverage.json', {'correct_only': correct_only, 'problem_ids': sorted(common), 'sources': report})


def export_verl(rows, path):
    import pandas as pd
    data = []
    for row in rows:
        data.append({'prompt': row['messages'], 'data_source': row['benchmark'],
                     'reward_model': {'style': 'rule', 'ground_truth': row['ground_truth']},
                     'extra_info': {**row, 'index': row['problem_id']}})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(path, index=False)


def partition_questions(rows, output, seed=42, validation_fraction=.1, test_fraction=.1, group_by='instance_id'):
    if not 0<validation_fraction<1 or not 0<test_fraction<1 or validation_fraction+test_fraction>=1:
        raise ValueError('Require positive validation/test fractions summing to <1')
    groups={}
    for row in rows:
        group=row.get(group_by, row['problem_id'] if group_by=='instance_id' else None)
        if group is None:
            raise ValueError(f'Missing split grouping field {group_by}; annotate before partitioning')
        groups.setdefault(str(group),[]).append(row)
    keys=sorted(groups)
    random.Random(seed).shuffle(keys)
    nv=max(1,int(len(keys)*validation_fraction));nt=max(1,int(len(keys)*test_fraction))
    if nv+nt>=len(keys):raise ValueError('Too few groups for train/validation/test')
    sets={'validation':set(keys[:nv]),'test':set(keys[nv:nv+nt]),'train':set(keys[nv+nt:])}
    output=Path(output)
    for split,ids in sets.items():
        chosen=[{**r,'split':split} for key in keys if key in ids for r in groups[key]]
        write_jsonl(output/f'{split}.jsonl',chosen)
    write_json(output/'split_manifest.json',{'seed':seed,'group_by':group_by,'data_hash':digest(rows),
               'groups':{s:sorted(ids) for s,ids in sets.items()},
               'note':'Paraphrases must share instance_id; semantic deduplication is not inferred'})


def prepare_deepmath(rows, output, revision, level_min=5., level_max=7.,
                     privilege_type='solution', solution_field='r1_solution_1', tokenizer=None, progress=None):
    """Adapt native DeepMath rows; difficulty is a continuous score, never rounded.

    References are dataset artifacts, not logits from an external teacher model.
    Missing difficulty/solutions fail rather than silently changing the experiment.
    """
    import math
    if not revision or revision in ('main', 'master'):
        raise ValueError('Provide an immutable DeepMath dataset revision')
    if not math.isfinite(level_min) or not math.isfinite(level_max) or level_min>level_max:
        raise ValueError('Invalid DeepMath difficulty bounds')
    if privilege_type not in ('empty','solution','final_answer'):
        raise ValueError('DeepMath privilege must be empty, solution or final_answer')
    selected=[]
    histogram={}
    for i, raw in enumerate(rows):
        if progress is not None:
            progress(i, len(rows), len(selected))
        extra=raw.get('extra_info') or {}
        score=raw.get('difficulty',raw.get('level',extra.get('difficulty',extra.get('level'))))
        try:
            if isinstance(score,bool): raise ValueError()
            score=float(score)
            if not math.isfinite(score): raise ValueError()
        except (TypeError,ValueError):
            raise ValueError(f'DeepMath row {i} lacks a valid difficulty/level; use the native dataset with metadata') from None
        histogram[str(score)]=histogram.get(str(score),0)+1
        if not level_min<=score<=level_max: continue
        answer=raw.get('final_answer',raw.get('ground_truth',(raw.get('reward_model') or {}).get('ground_truth')))
        if answer is None: raise ValueError(f'DeepMath row {i} lacks final_answer')
        if privilege_type=='empty': reference=''
        elif privilege_type=='final_answer': reference='\\boxed{'+str(answer)+'}'
        else:
            reference=raw.get(solution_field,extra.get(solution_field))
            if not isinstance(reference,str) or not reference.strip():
                raise ValueError(f'DeepMath row {i} lacks reference field {solution_field}')
        question=raw.get('question',extra.get('question'))
        if question is None:
            question=canonical({**raw,'ground_truth':answer})['question']
        retained={k:v for k,v in raw.items() if not k.startswith('r1_solution_')}
        row=canonical({**retained,'question':question,'ground_truth':answer,'data_source':'deepmath',
                       'benchmark':'DeepMath-103K','difficulty':score,'dataset_id':'zwhe99/DeepMath-103K',
                       'dataset_revision':revision,'privilege_type':privilege_type,'privilege_text':reference,
                       'generator_model':None,'generator_revision':None,'generation_seed':None,
                       'reference_source_field':solution_field if privilege_type=='solution' else ('final_answer' if privilege_type=='final_answer' else None),
                       'reference_correctness':True if privilege_type=='final_answer' else None,
                       'reference_token_length':len(tokenizer.encode(reference,add_special_tokens=False)) if tokenizer else (0 if not reference else None),
                       'reference_character_length':len(reference),
                       'messages':raw.get('prompt') or [{'role':'user','content':question+'\nPlease reason step by step, and put your final answer within \\boxed{}.'}],
                       'generation_config':{'source_type':'dataset_reference','dataset_id':'zwhe99/DeepMath-103K',
                                            'dataset_revision':revision,'solution_field':solution_field}})
        selected.append(row)
    if progress is not None:
        progress(len(rows), len(rows), len(selected))
    if not selected: raise ValueError('No DeepMath questions satisfy the configured difficulty bounds')
    if len({r['problem_id'] for r in selected})!=len(selected): raise ValueError('Duplicate DeepMath problem IDs')
    write_jsonl(output,selected)
    write_json(str(output)+'.manifest.json',{'dataset_id':'zwhe99/DeepMath-103K','dataset_revision':revision,
        'difficulty_filter':{'min':level_min,'max':level_max,'inclusive':True,'rounding':False},
        'before':len(rows),'after':len(selected),'coverage':len(selected)/len(rows),'difficulty_counts_before':histogram,
        'privilege_type':privilege_type,'reference_field':solution_field,'data_hash':digest(selected),
        'reference_token_lengths_available':tokenizer is not None or privilege_type=='empty'})
    return selected
