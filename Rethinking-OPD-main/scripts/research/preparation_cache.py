"""Verify and reuse completed data selection without redoing model generation."""
import ast
import copy
import json
from pathlib import Path
import shutil

from .common import ROOT, checkpoint_identity, digest, file_hash, read_jsonl, write_json


FILES = ('eligible.jsonl', 'eligible.jsonl.manifest.json', 'eligibility.jsonl', 'duplicates.jsonl',
         'answer_encoding_quarantine.jsonl',
         'heldout_probe.jsonl', 'validation.jsonl', 'train.jsonl', 'train_probe.jsonl')


def selection_sources(root=ROOT):
    """Hash the actual selection statements, excluding generation-only changes."""
    root = Path(root)
    path = root/'analysis/prepare_probe.py'
    tree = ast.parse(path.read_text())
    functions = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
    selected = {name:digest(ast.dump(functions[name]))
                for name in ('normalized_question', 'stratified_take', 'split_rows')}
    blocks = []
    for node in ast.walk(functions['freeze']):
        for field in ('body', 'orelse'):
            body = getattr(node, field, [])
            if not isinstance(body, list):
                continue
            begin = next((i for i, item in enumerate(body)
                          if isinstance(item, ast.Assign) and isinstance(item.value, ast.Call)
                          and isinstance(item.value.func, ast.Name) and item.value.func.id == 'load_questions'), None)
            if begin is None:
                continue
            for end in range(begin, len(body)):
                item = body[end]
                if (isinstance(item, ast.For) and isinstance(item.iter, ast.Call)
                        and isinstance(item.iter.func, ast.Attribute)
                        and isinstance(item.iter.func.value, ast.Name)
                        and item.iter.func.value.id == 'splits' and item.iter.func.attr == 'items'):
                    blocks.append(digest([ast.dump(stmt) for stmt in body[begin:end+1]]))
                    break
    if len(blocks) != 1:
        raise ValueError('Could not identify the unchanged data-selection block')
    selected['selection_block'] = blocks[0]
    evaluate = ast.parse((root/'analysis/evaluate.py').read_text())
    checked = next(n for n in evaluate.body if isinstance(n, ast.FunctionDef) and n.name == 'answer_status')
    selected['answer_status'] = digest(ast.dump(checked))
    for name in ('scripts/research/data.py', 'verl/verl/utils/opsd.py', 'scripts/val/eval/utils.py',
                 'scripts/research/answer_protocol.py'):
        selected[name] = file_hash(root/name)
    return selected


def selection_identity(c):
    dataset = {k:v for k,v in c['dataset'].items() if k != 'preparation_cache'}
    sampling = {k:c['monitor'][k] for k in ('seed','heldout_probe_n','train_probe_n')}
    if 'heldout_selection' in c['monitor']:
        sampling['heldout_selection'] = copy.deepcopy(c['monitor']['heldout_selection'])
    return dict(dataset=dataset, input_hash=file_hash(dataset['input']), model=c['model'],
                checkpoint_hash=checkpoint_identity(c['model']['path']),
                lengths={k:c['training'][k] for k in ('max_prompt_length', 'max_teacher_prompt_length',
                                                       'max_response_length')},
                sampling=sampling)


def validate_splits(directory, c):
    splits = {name:read_jsonl(Path(directory)/f'{name}.jsonl')
              for name in ('heldout_probe', 'validation', 'train', 'train_probe')}
    expected = dict(heldout_probe=c['monitor']['heldout_probe_n'], validation=c['dataset']['dev_n'],
                    train=c['dataset']['train_n'], train_probe=c['monitor']['train_probe_n'])
    ids = {}
    for name, rows in splits.items():
        ids[name] = {row['problem_id'] for row in rows}
        if len(rows) != expected[name] or len(ids[name]) != len(rows):
            raise ValueError(f'Cached split count or IDs invalid: {name}')
        if any(row['split'] != name for row in rows):
            raise ValueError(f'Cached split label invalid: {name}')
    if (ids['train'] & ids['validation'] or ids['train'] & ids['heldout_probe']
            or ids['validation'] & ids['heldout_probe'] or not ids['train_probe'] <= ids['train']):
        raise ValueError('Cached split isolation failed')
    checks = {row['problem_id']:row for row in read_jsonl(Path(directory)/'eligibility.jsonl')}
    for rows in splits.values():
        for row in rows:
            checked = checks[row['original_id']]
            if (checked['exclusion_reason'] is not None
                    or checked['student_tokens'] > c['training']['max_prompt_length']
                    or checked['teacher_tokens'] > c['training']['max_teacher_prompt_length']
                    or (c['dataset'].get('require_verified_reference', True) and not checked['reference_verified'])):
                raise ValueError('Cached row no longer satisfies the selection rules')
    return splits, read_jsonl(Path(directory)/'duplicates.jsonl')


def seal_snapshot(backup):
    """Seal already-copied files only after validating snapshot hashes and source code."""
    backup = Path(backup)
    snapshot = json.loads((backup/'snapshot.json').read_text())
    c = json.loads((backup/'source_config.json').read_text())
    cache = backup/'prepared'
    expected = snapshot['files']
    for name in FILES:
        if file_hash(cache/name) != expected[name]['sha256']:
            raise ValueError(f'Snapshot copy changed: {name}')
    sources = selection_sources(backup/'source')
    if sources != selection_sources():
        raise ValueError('Data selection changed since the snapshot; cannot reuse it')
    splits, _ = validate_splits(cache, c)
    manifest = dict(schema=1, identity=selection_identity(c), selection_sources=sources,
                    files={name:expected[name]['sha256'] for name in FILES},
                    source_workdir=snapshot['source_workdir'], source_snapshot=str(backup/'snapshot.json'),
                    counts={name:len(rows) for name,rows in splits.items()})
    target = cache/'selection_complete.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Refusing to replace a different selection-cache manifest')
    write_json(target, manifest)
    return target


def restore(c, output):
    directory = c['dataset'].get('preparation_cache')
    if directory is None:
        return None
    directory = Path(directory)
    marker = directory/'selection_complete.json'
    manifest = json.loads(marker.read_text())
    if manifest['schema'] != 1 or manifest['identity'] != selection_identity(c):
        raise ValueError('Prepared selection identity does not match this run')
    if manifest['selection_sources'] != selection_sources():
        raise ValueError('Prepared selection implementation changed')
    if set(manifest['files']) != set(FILES):
        raise ValueError('Prepared selection file list is incomplete')
    for name, expected in manifest['files'].items():
        if file_hash(directory/name) != expected:
            raise ValueError(f'Prepared selection cache changed: {name}')
    splits, duplicates = validate_splits(directory, c)
    for name in FILES:
        shutil.copy2(directory/name, Path(output)/name)
    write_json(Path(output)/'preparation_reuse.json',
               dict(cache=str(directory), manifest_hash=file_hash(marker),
                    source_workdir=manifest['source_workdir'], counts=manifest['counts']))
    print('PREPARATION_CACHE_VERIFIED', json.dumps(manifest['counts']), flush=True)
    return splits, duplicates


def save_selection(directory, c, target):
    """Publish durable selection before any GPU generation or quality gates."""
    import os
    directory, target = Path(directory), Path(target)
    if target.exists():
        raise ValueError(f'Refusing to overwrite selection checkpoint: {target}')
    temporary = target.with_name('.'+target.name+'.building')
    temporary.mkdir(parents=True,exist_ok=False)
    splits, _ = validate_splits(directory,c)
    for name in FILES:
        os.link(directory/name,temporary/name)
    manifest = dict(schema=1,identity=selection_identity(c),selection_sources=selection_sources(),
                    files={name:file_hash(temporary/name) for name in FILES},
                    source_workdir=str(Path(c.get('workdir',directory.parent.parent)).resolve()),
                    source_snapshot=str(target/'selection_complete.json'),
                    counts={name:len(rows) for name,rows in splits.items()})
    from collections import Counter
    counts = Counter(row['exclusion_reason'] or 'accepted'
                     for row in read_jsonl(directory/'eligibility.jsonl'))
    manifest['screening_counts'] = dict(processed=sum(counts.values()),accepted=counts.get('accepted',0),
                                       excluded=sum(counts.values())-counts.get('accepted',0),
                                       exclusion_counts={k:v for k,v in counts.items() if k!='accepted'})
    write_json(temporary/'selection_complete.json',manifest)
    temporary.rename(target)
    return target
