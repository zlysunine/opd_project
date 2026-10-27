"""Independent checkpoint extraction: raw block hooks, fixed text geometry and live behavior."""
from collections import defaultdict
import json
from importlib import metadata as package_metadata
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from scripts.research.common import checkpoint_identity, digest, file_hash, read_jsonl, write_json
from scripts.research.models import Model
from .common import config_argument, finite_mean, load_config, source_hashes, step_dir, table
from .distributions import nll_bits
from .trajectory_distributions import trajectory_metrics
from .evaluate import behavior, rollout_many, enforce_quality
from .metrics.neighbors import knn, neighbor_overlap
from .metrics.paired import summarize_student_teacher
from .metrics.rsa import align_rows, cosine_rsm, rsa_to_initial, rsm_summary
from .metrics.spectrum import linear_cka, spectrum
from .metrics.trajectory import curvature, paired_difference, pearson


def response_positions(attention_mask, prompt_lengths, response_lengths):
    """Explicit local-to-physical indices, valid for left AND right padding."""
    result = []
    for mask, p, t in zip(np.asarray(attention_mask), prompt_lengths, response_lengths):
        positions = np.flatnonzero(mask)
        if p < 1 or len(positions) != p+t or (len(positions)>1 and np.any(np.diff(positions)!=1)):
            raise ValueError('Expected a contiguous real prompt+response segment')
        result.append(dict(prompt_last=int(positions[p-1]), response=positions[p:p+t],
                           predictions=positions[p-1:p+t-1]))
    return result


def backbone(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model, 'model.layers'
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer, 'transformer.h'
    raise ValueError('Add an explicit block/backbone adapter for this architecture')


def collect(model, prompt, response, monitor, geometry=True):
    """One sample: retain only anchors and per-question curvature, never all layer token states."""
    ids = torch.tensor([prompt+response], device=model.device, dtype=torch.long)
    if len(prompt)+len(response) > model.model.config.max_position_embeddings:
        raise ValueError('Analysis input exceeds model context')
    positions = response_positions(np.ones((1, ids.shape[1])), [len(prompt)], [len(response)])[0]
    offsets = {'prompt_last':positions['prompt_last']}
    offsets.update({a:int(positions['response'][int(a.split('_')[1])-1])
                    for a in monitor['anchors'] if a!='prompt_last' and len(response)>=int(a.split('_')[1])})
    special = set(getattr(model.tokenizer, 'all_special_ids',
                          [model.tokenizer.pad_token_id, model.tokenizer.eos_token_id, model.tokenizer.bos_token_id]))
    mask = np.asarray([token not in special for token in response], dtype=bool)
    anchors, curves, handles = {}, {}, []
    base, path = backbone(model.model)
    def hook(layer):
        def capture(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.ndim != 3 or h.shape[:2] != ids.shape:
                raise ValueError('Block hook must expose [B,T,d] residual output')
            # Convert BEFORE subtraction; detach only inference tensors.
            h = h[0].float().cpu().numpy()
            anchors[layer] = {a:h[i].copy() for a,i in offsets.items()}
            curves[layer] = curvature(h[positions['response']], mask, monitor['curvature_eps'])
        return capture
    if geometry:
        handles = [block.register_forward_hook(hook(i+1)) for i,block in enumerate(model.layers())]
    try:
        output = base(input_ids=ids, attention_mask=torch.ones_like(ids),
                      position_ids=torch.arange(ids.shape[1], device=ids.device)[None], use_cache=False,
                      return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if geometry and len(curves) != len(model.layers()):
        raise RuntimeError('Not all decoder hooks fired')
    return anchors, curves, output.last_hidden_state[0], positions, path


def extract(c, checkpoint, revision, step, model=None):
    if step < 0 or step not in c['monitor']['optimizer_steps']:
        raise ValueError('Step must be one of the configured optimizer monitoring steps')
    started = time.monotonic()
    root = Path(c['output_root'])
    probe = Path(c.get('probe_root', root/'probe'))
    frozen = json.loads((probe/'complete.json').read_text())
    if frozen['identity'].get('sources',source_hashes()) != source_hashes():
        raise ValueError('Analysis implementation changed since probe freezing; use a new workspace')
    for name, expected in frozen['files'].items():
        if file_hash(probe/name) != expected:
            raise ValueError(f'Frozen probe changed: {name}')
    initial_hash=checkpoint_identity(c['model']['path'])
    if frozen['identity'].get('checkpoint_hash',initial_hash)!=initial_hash:
        raise ValueError('Initial policy weights differ from the frozen probe identity')
    identity = dict(checkpoint_hash=checkpoint_identity(checkpoint), initial_checkpoint_hash=initial_hash,
                    checkpoint_revision=revision,
                    config_hash=digest(c), sources=source_hashes(), probe_manifest_hash=frozen['probe_manifest_hash'], step=step)
    target = step_dir(c, step)
    if (target/'complete.json').exists():
        previous = json.loads((target/'complete.json').read_text())
        if previous['identity'] != identity:
            raise ValueError('Analysis cache identity changed; use a new analysis output_root')
        for name, expected in previous['files'].items():
            if file_hash(target/name) != expected:
                raise ValueError(f'Analysis cache file changed: {name}')
        return target
    if step != 0 and not (step_dir(c,0)/'complete.json').is_file():
        raise ValueError('Analyze step0 first to establish fixed-pool initial geometry and distributions')
    if step != 0:
        original=json.loads((step_dir(c,0)/'complete.json').read_text())['identity']
        if original['sources']!=identity['sources'] or original['config_hash']!=identity['config_hash']:
            raise ValueError('All monitored steps must use the same analysis implementation and configuration')
    if target.exists():
        raise ValueError('Incomplete analysis target; inspect before using a new output_root')
    if model is None:
        model = Model(checkpoint, revision, device=c['model']['device'], dtype=c['model']['dtype'],
                      prompt_mode='plain_completion')
    model.model.eval()
    architecture=json.loads((probe/'manifest.json').read_text()).get('architecture')
    actual=dict(blocks=len(model.layers()),hidden_size=model.model.config.hidden_size,vocab_size=model.model.config.vocab_size)
    if architecture is not None and architecture!=actual:
        raise ValueError('Checkpoint architecture differs from the frozen initial model')
    monitor = c['monitor']
    initial_model=None
    if monitor.get('distribution_metrics',True) and monitor.get('compare_to_initial',True):
        if step==0:
            if identity['checkpoint_hash']!=initial_hash:
                raise ValueError('Step0 distribution baseline must use the archived initial policy')
            initial_model=model
        else:
            initial_model=Model(c['model']['path'],c['model']['revision'],device=c['model']['device'],
                                dtype=c['model']['dtype'],prompt_mode='plain_completion')
        if initial_model.identity['tokenizer_hash']!=model.identity['tokenizer_hash']:
            raise ValueError('Initial/current distribution tokenizers must be identical')
    fixed = read_jsonl(probe/'fixed_pool.jsonl')
    if monitor.get('additional_fixed_reference'):
        fixed += read_jsonl(probe/'fixed_reference.jsonl')
    current_tokenizer_hash = model.identity['tokenizer_hash']
    if current_tokenizer_hash != json.loads((probe/'manifest.json').read_text())['model']['tokenizer_hash']:
        raise ValueError('Checkpoint tokenizer differs from frozen probe tokenizer')
    live = []
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.step-{step}-', dir=target.parent) as tmp, torch.inference_mode():
        out = Path(tmp)
        if monitor.get('live_generation', True):
            if step == 0 and (probe/'baseline_live_rollouts.jsonl').is_file():
                live = read_jsonl(probe/'baseline_live_rollouts.jsonl')
            else:
                live = rollout_many(model, read_jsonl(probe/'heldout_probe.jsonl')+read_jsonl(probe/'train_probe.jsonl'),
                                    monitor, 'live_rollout', step,
                                    root/'generation_cache'/f'step_{step:06d}', root/'generation_progress.json')
            enforce_quality(live,monitor,root/'quality'/f'step_{step:06d}')
        table(out/'live_rollouts', live, 'jsonl')
        arrays, metadata, curve_rows, dist_rows, metric_rows = {}, {}, [], [], []
        groups = defaultdict(list)
        params_hash = identity['config_hash']
        base_meta = dict(run_id=c['run_id'], method=c.get('method','opsd'), optimizer_step=step,
                         checkpoint_revision=revision, params_hash=params_hash, hook='block_out',
                         teacher_revision=revision, privileged_type=c['dataset']['reference_field'],
                         checkpoint_content_hash=identity['checkpoint_hash'], teacher_content_hash=identity['checkpoint_hash'],
                         teacher_role='training_self_teacher' if c.get('method')=='opsd' else 'diagnostic_privileged_policy')
        training_metrics = Path(c.get('training',{}).get('pipeline_workdir',root))/'train/metrics.jsonl'
        budget = {}
        if training_metrics.is_file():
            matches=[r for r in read_jsonl(training_metrics) if r['step']==step]
            if matches:
                budget=matches[-1].get('budget',{})
        for key in ('optimized_tokens','rollout_tokens','wall_seconds','teacher_forward_tokens'):
            base_meta['training_'+key]=budget.get(key,0. if step==0 else None)
        def emit(metric, value, n, pool, split, layer=None, anchor=None, role='student', **extra):
            metric_rows.append({**base_meta, 'pool':pool, 'split':split, 'layer':layer, 'anchor':anchor,
                                'role':role, 'metric':metric, 'value':value, 'n_valid':int(n),
                                'status':'ok' if value is not None and np.isfinite(value) else 'undefined', **extra})
        def store_geometry(row, role, anchors, curves, module_path, bits):
            if row['pool'] != 'live_rollout':
                groups[(role, row['pool'], row['split'])].append((row, anchors))
            for layer, values in curves.items():
                curve_rows.append({**base_meta, 'problem_id':row['problem_id'], 'split':row['split'],
                                   'pool':row['pool'], 'role':role, 'layer':layer, 'module_path':f'{module_path}.{layer-1}',
                                   'response_length':min(len(row['response_ids']),monitor['max_probe_response_tokens']),
                                   'geometry_window_clipped':len(row['response_ids'])>monitor['max_probe_response_tokens'],
                                   'surprisal_bits':bits,
                                   **{k:v for k,v in values.items() if k!='angles_deg'}})
        for row in fixed+live:
            response = row['response_ids'][:monitor['max_probe_response_tokens']]
            if row['pool'] != 'live_rollout' and model.prompt(row) != row['prompt_ids']:
                raise ValueError('Prompt encoding changed; fixed token IDs must match the protocol')
            anchors, curves, hidden, pos, path = collect(model, row['prompt_ids'], response, monitor)
            bits = nll_bits(model.model, hidden, response, pos['predictions'].tolist(), monitor['state_chunk_size'])
            store_geometry(row, 'student', anchors, curves, path, bits)
            del hidden
            if monitor.get('distribution_metrics',True):
                values=trajectory_metrics(model,row,monitor,initial_model)
                dist_rows.append({**base_meta,'pool':row['pool'],'split':row['split'],
                                  'problem_id':row['problem_id'],**values})
            teacher_geometry = (row['pool']!='live_rollout' and c.get('method')=='opsd'
                                and (monitor.get('compare_student_teacher',False)
                                     or step in monitor.get('teacher_geometry_steps', [])))
            if teacher_geometry:
                ta,tc,th,tp,tpath=collect(model,model.prompt(row,True),response,monitor,True)
                store_geometry(row, 'privileged_teacher', ta, tc, tpath, np.nan)
                del th
        initial_arrays = np.load(step_dir(c,0)/'anchors.npz', allow_pickle=False) if step else None
        initial_meta = json.loads((step_dir(c,0)/'anchors.json').read_text()) if step else None
        neighbors = {}
        for (role, pool, split), records in groups.items():
            for anchor in monitor['anchors']:
                chosen = [(r,a) for r,a in records if anchor in a[1]]
                ids = [r['problem_id'] for r,a in chosen]
                if not chosen:
                    for layer in range(1,len(model.layers())+1):
                        for name in ('rsm_mean','rsa_rho_to_initial','neighbor_overlap_k16','participation_ratio'):
                            emit(name,np.nan,0,pool,split,layer,anchor,role,reason='no_fixed_samples_reach_anchor')
                    continue
                topics = [r['topic_label'] for r,a in chosen]
                for layer in range(1,len(model.layers())+1):
                    key = f'{role}__{pool}__{split}__{anchor}__layer{layer}'
                    h = np.stack([a[layer][anchor] for r,a in chosen]).astype(np.float32)
                    arrays[key] = h
                    metadata[key] = dict(sample_ids=ids, topics=topics, layer=layer, anchor=anchor,
                                         pool=pool, split=split, role=role, module_path=f'{path}.{layer-1}',
                                         position_indices=[len(model.prompt(r,role=='privileged_teacher'))-1+
                                             (0 if anchor=='prompt_last' else int(anchor.split('_')[1])) for r,a in chosen])
                    rsm, valid = cosine_rsm(h)
                    stats, hist, edges = rsm_summary(rsm, topics)
                    reference = None
                    if step == 0:
                        reference = h
                    elif key in initial_meta:
                        reference = align_rows(initial_arrays[key], initial_meta[key]['sample_ids'], ids)
                    if reference is not None:
                        stats.update(rsa_to_initial(rsm,cosine_rsm(reference)[0],monitor['rsa_max_pairs'],monitor['seed']))
                        if monitor.get('linear_cka', True):
                            stats['linear_cka_to_initial'] = linear_cka(h,reference)
                    stats.update(spectrum(h))
                    for name,value in stats.items():
                        if name!='n_pairs':
                            emit(name,value,int(valid.sum()),pool,split,layer,anchor,role,
                                 n_pairs=stats.get('n_pairs'),rsa_pair_seed=monitor['seed'],
                                 invalid_sample_ids=[sid for sid,v in zip(ids,valid) if not v])
                    np.savez_compressed(out/f'rsm_{key}.npz', rsm=rsm, sample_ids=np.asarray(ids), hist=hist, bins=edges)
                    for k in monitor['knn_k']:
                        if len(ids)<=k or not np.isfinite(h).all() or reference is None:
                            emit(f'neighbor_overlap_k{k}',np.nan,0,pool,split,layer,anchor,role,reason='N<=k_or_no_initial')
                            continue
                        index, distances = knn(h,ids,k)
                        other, _ = knn(reference,ids,k)
                        overlap = neighbor_overlap(index,other)
                        neighbors[f'{key}__k{k}__indices'] = index
                        neighbors[f'{key}__k{k}__distances'] = distances
                        neighbors[f'{key}__k{k}__per_sample_overlap'] = overlap
                        emit(f'neighbor_overlap_k{k}',float(overlap.mean()),len(ids),pool,split,layer,anchor,role,
                             chance_level=k/(len(ids)-1))
        if initial_arrays is not None:
            initial_arrays.close()
        if monitor.get('compare_student_teacher',False) and c.get('method')=='opsd':
            cohorts=[(pool,split,layer,anchor) for role,pool,split in groups if role=='student'
                     for layer in range(1,len(model.layers())+1) for anchor in monitor['anchors']]
            paired_manifest,paired_neighbors=summarize_student_teacher(arrays,metadata,monitor,emit,cohorts)
            write_json(out/'student_teacher_comparisons.json',paired_manifest)
            np.savez_compressed(out/'student_teacher_neighbors.npz',**paired_neighbors)
        np.savez_compressed(out/'anchors.npz', **arrays)
        write_json(out/'anchors.json',metadata)
        np.savez_compressed(out/'neighbors.npz', **neighbors)
        summarize_curvature(curve_rows, step_dir(c,0)/'curvature.jsonl' if step else None, emit)
        summarize_distributions(dist_rows,emit)
        for split in ('heldout_probe','train_probe'):
            rows = [r for r in live if r['split']==split]
            for name,value in behavior(rows).items():
                emit(name,value,len(rows),'live_rollout',split)
        table(out/'curvature',curve_rows,c['table_format'])
        table(out/'distributions',dist_rows,c['table_format'])
        table(out/'metrics',metric_rows,c['table_format'])
        versions = {}
        for package in ('torch','transformers','numpy','scipy','matplotlib','pyarrow','peft'):
            try:
                versions[package]=package_metadata.version(package)
            except package_metadata.PackageNotFoundError:
                versions[package]=None
        write_json(out/'run_manifest.json',dict(identity=identity, model=model.identity, config=c,versions=versions,
                   hooks=dict(point='complete_decoder_block_output', layers=len(model.layers()),
                              hidden_size=model.model.config.hidden_size, numbering='1-based contextualized blocks'),
                   actual_training=c['training'], analysis_wall_seconds=time.monotonic()-started,
                   note='Checkpoint-only eval/inference; training RNG and dataloader live in a different process.'))
        files = {str(p.relative_to(out)):file_hash(p) for p in out.rglob('*') if p.is_file()}
        write_json(out/'complete.json',dict(identity=identity, files=files))
        out.rename(target)
    return target


def summarize_distributions(rows,emit):
    """Paper-compatible token means; keep fixed/live and train/heldout separate."""
    groups=defaultdict(list)
    for row in rows:
        groups[(row['pool'],row['split'])].append(row)
    for (pool,split), records in groups.items():
        keys=set().union(*(r['metric_sums'] for r in records))
        for name in sorted(keys):
            total=sum(r['metric_sums'].get(name,0.) for r in records)
            n=sum(r['metric_token_counts'].get(name,0) for r in records)
            emit(name,total/n if n else None,n,pool,split,temperature=1.,
                 n_questions=sum(r['metric_token_counts'].get(name,0)>0 for r in records),
                 aggregation='token_mean',distribution_scope='all_valid_tokens',
                 diagnostic_kl_direction='metric_name_specifies_direction')


def summarize_curvature(rows, initial_path, emit):
    """All paired changes use common IDs, never differences of differently filtered means."""
    initial = read_jsonl(initial_path) if initial_path else rows
    lookup = {(r['role'],r['pool'],r['split'],r['layer'],r['problem_id']):r for r in initial}
    groups = defaultdict(list)
    for row in rows:
        groups[(row['role'],row['pool'],row['split'])].append(row)
    for (role,pool,split), values in groups.items():
        first = {r['problem_id']:r for r in values if r['layer']==1}
        layers = sorted({r['layer'] for r in values})
        # One common finite cohort across ALL blocks for D_min and layer_star.
        common = set(first)
        for layer in layers:
            common &= {r['problem_id'] for r in values if r['layer']==layer and r['mean_deg'] is not None
                       and np.isfinite(r['mean_deg'])}
        if pool!='live_rollout':
            for layer in layers:
                common &= {sid for sid in common
                           if (role,pool,split,layer,sid) in lookup
                           and lookup[(role,pool,split,layer,sid)]['mean_deg'] is not None
                           and np.isfinite(lookup[(role,pool,split,layer,sid)]['mean_deg'])}
        deltas = []
        for layer in layers:
            current = [r for r in values if r['layer']==layer]
            paired = [r for r in current if r['problem_id'] in common]
            for field,name in [('mean_deg','curvature_deg'),('mean_step_norm','mean_step_norm'),
                               ('invalid_fraction','invalid_angle_fraction'),('surprisal_bits','surprisal_bits')]:
                selected = paired if field=='mean_deg' else current
                avg,n = finite_mean([r[field] for r in selected])
                emit(name,avg,n,pool,split,layer,role=role)
            value,n = paired_difference([r['mean_deg'] for r in paired], [first[r['problem_id']]['mean_deg'] for r in paired])
            emit('curvature_relative_first_deg',value,n,pool,split,layer,role=role)
            deltas.append(value)
            # Live text changes across steps: no paired training-delta claim on different token sequences.
            pairs = [(r,lookup.get((role,pool,split,layer,r['problem_id']))) for r in paired] if pool!='live_rollout' else []
            pairs = [(r,r0) for r,r0 in pairs if r0 is not None]
            value,n = paired_difference([r['mean_deg'] for r,r0 in pairs], [r0['mean_deg'] for r,r0 in pairs])
            emit('curvature_training_delta_deg',value,n,pool,split,layer,role=role)
            for suffix, sample in [('current',paired), ('initial',pairs)]:
                a = [r['mean_deg'] for r in sample] if suffix=='current' else [r['mean_deg'] for r,r0 in sample]
                b = [r['surprisal_bits'] for r in sample] if suffix=='current' else [r0['surprisal_bits'] for r,r0 in sample]
                value,n = pearson(a,b)
                emit(f'curvature_surprisal_{suffix}_pearson',value,n,pool,split,layer,role=role)
        valid = [(v,l) for v,l in zip(deltas,layers) if np.isfinite(v)]
        value,layer = min(valid) if valid else (np.nan,np.nan)
        emit('curvature_d_min_deg',value,len(common),pool,split,role=role)
        emit('curvature_layer_star',layer,len(common),pool,split,role=role)


def main():
    parser = config_argument(__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--revision',required=True)
    parser.add_argument('--step',type=int,required=True)
    args = parser.parse_args()
    print(extract(load_config(args.config),args.checkpoint,args.revision,args.step))


if __name__=='__main__':
    main()
