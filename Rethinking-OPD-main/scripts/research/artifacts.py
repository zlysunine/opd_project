"""Resumable extraction of aligned response positions, one problem per disk shard."""
import math
from pathlib import Path

import numpy as np
import torch

from .common import digest, file_hash, read_jsonl, write_json, write_jsonl
from .models import trajectory_ids
from .representations import (cosine, rsa, rsa_correlations, linear_cka, neighborhood_overlap,
                              clustering_ari, effective_rank, subspace_overlap, state_probe)


def position_selection(length, strategy='relative', positions=(0,), bins=5, annotations=None):
    if length < 1:
        raise ValueError('Empty response')
    if strategy == 'last_prompt_token':
        return [(-1, 'last_prompt_token')]
    if strategy == 'fixed':
        return [(i, f'fixed_{i}') for i in positions if 0 <= i < length]
    if strategy == 'relative':
        return [(min(length-1, int((b+0.5)*length/bins)), f'progress_{b}_of_{bins}') for b in range(bins)]
    if strategy == 'annotated':
        if not annotations:
            raise ValueError('Annotated positions require response-relative step_token_positions metadata')
        return [(int(a['token_index']), a['label']) for a in annotations if 0<=int(a['token_index'])<length]
    raise ValueError(strategy)


def extract(model, rows, out, layers, point='block_output', privileged=False, strategy='relative',
            positions=(0,), bins=5, semantics='predict_current', max_samples=1024):
    out = Path(out)
    if len(rows)*max(bins, len(positions)) > max_samples*10:
        raise ValueError('Input too large; explicitly subsample diagnostic problems')
    identity = {'model': model.identity, 'layers': layers, 'hook_point': point, 'privileged': privileged,
                'strategy': strategy, 'positions': positions, 'bins': bins, 'semantics': semantics,
                'data_hash': digest(rows), 'max_samples': max_samples, 'schema_version': 1,
                'collector_source_hash':file_hash(__file__)}
    cache_key = digest(identity)
    manifest = out/'manifest.json'
    if manifest.exists():
        import json
        old = json.loads(manifest.read_text())
        if old['cache_key'] != cache_key:
            raise ValueError('Cache identity mismatch: model/checkpoint/template/prefix/config changed; choose another directory')
    out.mkdir(parents=True, exist_ok=True)
    write_json(manifest, {'cache_key': cache_key, **identity, 'complete': False})
    records = []
    for row in rows:
        response = trajectory_ids(row,model.tokenizer)
        if not response or row.get('tokenizer_hash') != model.identity['tokenizer_hash']:
            raise ValueError('Extraction requires fixed response_ids and matching tokenizer hash; run evaluate first')
        prompt = model.prompt(row, privileged)
        selection = position_selection(len(response), strategy, positions, bins, row.get('step_token_positions'))
        if len(records)+len(selection)*len(layers) > max_samples:
            break
        shard_key = digest([cache_key, row['problem_id'], row.get('sample_index',0), response])[:24]
        shard = out/f'{shard_key}.npz'
        index = out/f'{shard_key}.jsonl'
        if shard.exists() and index.exists():
            previous = read_jsonl(index)
            with np.load(shard, allow_pickle=False) as saved:
                if len(saved['activations']) != len(previous):
                    raise ValueError('Incomplete/corrupt activation shard')
            records.extend(previous)
            continue
        values, local_records = [], []
        handles = []
        captured = {}
        try:
            for layer in layers:
                module, description = model.hook_module(layer, point)
                def hook(module, args, output, layer=layer):
                    hidden = output[0] if isinstance(output, tuple) else output
                    # Immediately retain only selected positions, never all layer activations.
                    indices = [len(prompt)-1 if i==-1 else len(prompt)+i-(semantics=='predict_current') for i,_ in selection]
                    captured[layer] = hidden[0, indices].detach().float().cpu().numpy()
                handles.append(module.register_forward_hook(hook))
            with torch.no_grad():
                batch = model.pack(prompt, response)
                model.model.base_model(**{k:v for k,v in batch.items() if k!='responses'}, use_cache=False)
            for layer in layers:
                _, description = model.hook_module(layer, point)
                for j, (token_index, bucket) in enumerate(selection):
                    activation = captured[layer][j]
                    analysis = row.get('analysis', {})
                    # No fabricated intermediate labels for unannotated generated reasoning.
                    annotations = {a['token_index']: a for a in row.get('step_token_positions', [])}
                    state = annotations.get(token_index, {}).get('intermediate_state')
                    labels = {'skill': analysis.get('skill', row.get('benchmark')), 'intermediate_state': state,
                              'answer': row['ground_truth'], 'correctness': row.get('correct')}
                    local_records.append({'shard': shard.name, 'row': len(values), 'cache_key': cache_key,
                        'model_config_hash': model.identity['model_config_hash'], 'checkpoint': model.identity,
                        'layer': layer, 'hook_point': point, **description, 'semantics': semantics,
                        'problem_id': row['problem_id'], 'instance_id': row.get('instance_id', row['problem_id']),
                        'template_id': row.get('template_id', digest(row['messages'])[:16]),
                        'sample_index': row.get('sample_index', 0), 'token_index': token_index,
                        'position_bucket': bucket, 'prefix_hash': digest([row['messages'], response[:0 if token_index==-1 else token_index+(semantics=='after_current')]]),
                        'actual_context_hash': digest([prompt, response[:0 if token_index==-1 else token_index+(semantics=='after_current')]]),
                        'response_hash': digest(response), 'trajectory_source': row.get('trajectory_source','fixed'),
                        'labels': labels, 'probe_split': row.get('probe_split', row['split']), 'privileged': privileged})
                    values.append(activation)
            if values:
                temp = shard.with_suffix('.tmp.npz')
                np.savez_compressed(temp, activations=np.stack(values))
                temp.replace(shard)
                write_jsonl(index, local_records)
                records.extend(local_records)
        finally:
            for handle in handles:
                handle.remove()
    if not records:
        raise ValueError('No selected activations; check positions/layers')
    write_jsonl(out/'records.jsonl', records)
    write_json(manifest, {'cache_key':cache_key, **identity, 'complete':True, 'samples':len(records)})
    return records


def load_activations(path):
    path = Path(path)
    records = read_jsonl(path/'records.jsonl')
    cache = {}
    values = []
    for row in records:
        if row['shard'] not in cache:
            with np.load(path/row['shard'], allow_pickle=False) as data:
                cache[row['shard']] = data['activations']
        values.append(cache[row['shard']][row['row']])
    return np.stack(values), records


def alignment_key(r):
    return tuple(r[k] for k in ('problem_id','sample_index','layer','hook_point','semantics','token_index',
                                'position_bucket','prefix_hash','response_hash'))


def align(a, ar, b, br):
    amap, bmap = {alignment_key(r):i for i,r in enumerate(ar)}, {alignment_key(r):i for i,r in enumerate(br)}
    if len(amap) != len(ar) or len(bmap) != len(br) or set(amap) != set(bmap):
        raise ValueError('Representation rows/prefixes differ; use the same fixed diagnostics and layer mapping')
    order = [bmap[alignment_key(r)] for r in ar]
    return a, b[order], ar


def analyze(a_dir, b_dir, out, k=5, cap=2048, clusters=4, seed=0, baseline_a=None, baseline_b=None,
            rank=2, centered=False, cross_scale=False, probe=False, distance="cosine"):
    a, records = load_activations(a_dir)
    b, br = load_activations(b_dir)
    a, b, records = align(a, records, b, br)
    if a.shape[1] != b.shape[1] and not cross_scale:
        raise ValueError('Hidden dimensions differ; cross-scale needs explicit --cross-scale (RSA/CKA/NO only)')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base_a = base_b = None
    if bool(baseline_a) != bool(baseline_b):
        raise ValueError('Provide both baseline activation directories for change-vector analysis')
    if baseline_a:
        aa, aar = load_activations(baseline_a)
        bb, bbr = load_activations(baseline_b)
        _, base_a, _ = align(a, records, aa, aar)
        _, base_b, _ = align(b, records, bb, bbr)
        if base_a.shape != a.shape or base_b.shape != b.shape:
            raise ValueError('Baseline coordinate mismatch')
    summaries = []
    for layer in sorted({r['layer'] for r in records}):
        for bucket in sorted({r['position_bucket'] for r in records}):
            indices = [i for i,r in enumerate(records) if r['layer']==layer and r['position_bucket']==bucket]
            if not indices:
                continue
            if len(indices) > cap:
                raise ValueError('RSA/CKA sample cap exceeded; subsample extraction explicitly')
            x, y = a[indices].astype(float), b[indices].astype(float)
            rr = [records[i] for i in indices]
            summary = {'layer':layer, 'position_bucket':bucket, 'n':len(x), 'linear_cka':linear_cka(x,y),
                       'effective_rank_singular_values_a':effective_rank(x),
                       'interpretation':'diagnostic association, not evidence of causal transfer'}
            if not cross_scale:
                cv = cosine(x,y)
                summary['activation_cosine'] = float(np.nanmean(cv)) if np.isfinite(cv).any() else None
            ra, rb = rsa(x, cap), rsa(y, cap)
            orders = {f'order_{label}': np.array(sorted(range(len(rr)), key=lambda i:str(rr[i]['labels'][label])))
                      for label in ('skill','intermediate_state','answer')}
            np.savez_compressed(out/f'rsa_layer{layer}_{bucket}.npz', a=ra, b=rb, **orders)
            write_json(out/f'rsa_layer{layer}_{bucket}.labels.json', [r['labels'] for r in rr])
            summary.update(rsa_correlations(ra,rb))
            if len(x) > k:
                summary.update(neighborhood_overlap(x,y,k,distance=distance,max_samples=cap))
            if len(x) >= clusters:
                labels = {label:[str(r['labels'][label]) for r in rr] for label in ('skill','intermediate_state','answer')
                          if all(r['labels'][label] is not None for r in rr)}
                summary['clustering_a'] = clustering_ari(x,labels,clusters,seed)
                summary['clustering_b'] = clustering_ari(y,labels,clusters,seed)
            if base_a is not None:
                dx, dy = x-base_a[indices], y-base_b[indices]
                if not cross_scale:
                    change_cos = cosine(dx,dy)
                    summary['change_vector_cosine'] = float(np.nanmean(change_cos)) if np.isfinite(change_cos).any() else None
                    try:
                        summary.update(subspace_overlap(dx,dy,rank,centered))
                    except ValueError as error:
                        summary['subspace_unavailable'] = str(error)
                dn, dv = __import__('scripts.research.representations',fromlist=['normalize']).normalize(dx)
                if dv.sum() > 1:
                    gram = dn[dv]@dn[dv].T
                    summary['privileged_shift_mean_pairwise_cosine_a'] = float(gram[np.triu_indices(len(gram),1)].mean())
            if probe:
                if any(r['labels']['intermediate_state'] is None for r in rr):
                    raise ValueError('Probe requires genuine aligned intermediate state annotations')
                summary['probe_a'] = state_probe(x,rr,seed=seed)
                summary['probe_b'] = state_probe(y,rr,seed=seed)
            summaries.append(summary)
    write_json(out/'representation_metrics.json', summaries)
    return summaries
