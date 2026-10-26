"""Offline plots from scalar artifacts. No interpolation or invented missing results."""
import csv
import json
from pathlib import Path

import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/opd-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import read_jsonl, write_json


def plot_directory(source, output):
    source, out = Path(source), Path(output)
    out.mkdir(parents=True,exist_ok=True)
    generated = []
    metric_rows = []
    token_rows = []
    representations = []
    for path in source.rglob('*.jsonl'):
        if path.name in ('records.jsonl','matrix.jsonl'): continue
        rows = read_jsonl(path)
        for row in rows:
            if 'metrics' in row and 'step' in row:
                metric_rows.append({'run':str(path.parent.relative_to(source)),'step':row['step'],
                                    **row.get('metrics',{}),**{'budget/'+k:v for k,v in row.get('budget',{}).items()}})
            elif 'metric' in row and 'relative_bucket' in row:
                token_rows.append({**row, 'diagnostic_run':str(path.relative_to(source))})
    if metric_rows:
        keys = sorted({k for r in metric_rows for k,v in r.items() if isinstance(v,(float,int))})
        with (out/'training_metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['run',*keys]); writer.writeheader()
            for r in metric_rows: writer.writerow({k:r.get(k) for k in ['run',*keys]})
        names = [key for key in keys if key.startswith(('opd/','loss/','eval/','eval_privileged/','validation/','accuracy','kl/'))
                 or key in ('actor/forward_kl','actor/loss')]
        for name in names:
            for xkey in ('step','budget/optimized_tokens','budget/wall_seconds'):
                fig,ax=plt.subplots(figsize=(6,4))
                for run in sorted({r['run'] for r in metric_rows}):
                    valid=[r for r in metric_rows if r['run']==run and r.get(name) is not None and r.get(xkey) is not None]
                    if valid: ax.plot([r[xkey] for r in valid],[r[name] for r in valid],marker='.',label=run or 'run')
                ax.set(xlabel=xkey,ylabel=name); ax.legend(fontsize=6); fig.tight_layout()
                filename=(name+'__'+xkey).replace('/','_')+'.png'
                fig.savefig(out/filename); plt.close(fig); generated.append(filename)
    consistency_keys = ['opd/overlap_ratio','opd/overlap_token_advantage','opd/student_entropy',
                        'opd/teacher_entropy','opd/entropy_gap']
    consistency_rows = [r for r in metric_rows if any(key in r for key in consistency_keys)]
    if consistency_rows:
        with (out/'thinking_consistency.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['run','step',*consistency_keys])
            writer.writeheader()
            for row in consistency_rows:
                writer.writerow({key:row.get(key) for key in ['run','step',*consistency_keys]})
        fig,axes=plt.subplots(5,1,figsize=(9,13),sharex=True)
        for ax,key in zip(axes,consistency_keys):
            for run in sorted({r['run'] for r in consistency_rows}):
                points=sorted([r for r in consistency_rows if r['run']==run],key=lambda r:r['step'])
                ax.plot([r['step'] for r in points],
                        [r.get(key) if r.get(key) is not None else np.nan for r in points],label=run or 'run')
            ax.set_ylabel(key.removeprefix('opd/')); ax.grid(alpha=.25)
        axes[0].legend(fontsize=7);axes[-1].set_xlabel('rollout batch / pre-update diagnostics')
        fig.tight_layout();fig.savefig(out/'thinking_consistency.png');plt.close(fig)
        generated.append('thinking_consistency.png')
    if token_rows:
        runs = sorted({r['diagnostic_run'] for r in token_rows})
        names = sorted({r['metric'] for r in token_rows})
        with (out/'token_metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['run','metric','relative_bucket','question_mean','token_weighted'])
            writer.writeheader()
            for run in runs:
                for name in names:
                    points=[]
                    for bucket in sorted({r['relative_bucket'] for r in token_rows}):
                        rows=[r for r in token_rows if r['diagnostic_run']==run and r['metric']==name and r['relative_bucket']==bucket and r['value'] is not None]
                        if not rows: continue
                        groups={}
                        for r in rows: groups.setdefault(r['problem_id'],[]).append(r['value'])
                        qm=float(np.mean([np.mean(v) for v in groups.values()]))
                        tm=float(np.mean([r['value'] for r in rows]))
                        points.append((bucket,qm,tm));writer.writerow({'run':run,'metric':name,'relative_bucket':bucket,'question_mean':qm,'token_weighted':tm})
                    if not points: continue
                    fig,ax=plt.subplots(figsize=(6,4))
                    ax.plot([r[0] for r in points],[r[1] for r in points],label='question mean')
                    ax.plot([r[0] for r in points],[r[2] for r in points],label='token weighted')
                    ax.set(xlabel='response relative progress bucket',ylabel=name,title=run);ax.legend();fig.tight_layout()
                    filename='positions_'+run.replace('/','_')+'_'+name.replace('/','_')+'.png'
                    fig.savefig(out/filename);plt.close(fig);generated.append(filename)
    for path in source.rglob('representation_metrics.json'):
        reps=json.loads(path.read_text())
        for r in reps: representations.append({'comparison':str(path.parent.relative_to(source)),**r})
    if representations:
        for key in ('activation_cosine','linear_cka','representation_neighborhood_overlap','representation_subspace_overlap'):
            columns=sorted({r['comparison']+':'+r['position_bucket'] for r in representations})
            layers=sorted({r['layer'] for r in representations})
            matrix=np.full((len(layers),len(columns)),np.nan)
            for r in representations:
                if r.get(key) is not None: matrix[layers.index(r['layer']),columns.index(r['comparison']+':'+r['position_bucket'])]=r[key]
            if np.isfinite(matrix).any():
                fig,ax=plt.subplots(figsize=(max(6,len(columns)*.7),4));im=ax.imshow(matrix,aspect='auto')
                ax.set_yticks(range(len(layers)),labels=layers);ax.set_xticks(range(len(columns)),labels=columns,rotation=80,fontsize=6)
                ax.set(xlabel='checkpoint comparison / position',ylabel='layer',title=key);fig.colorbar(im,ax=ax);fig.tight_layout()
                fig.savefig(out/(key+'.png'));plt.close(fig);generated.append(key+'.png')
        aris=[]
        for r in representations:
            for side in ('a','b'):
                for label,value in r.get('clustering_'+side,{}).get('ari',{}).items():
                    aris.append((f'{r["comparison"]} L{r["layer"]} {r["position_bucket"]} {side} {label}',value))
        if aris:
            fig,ax=plt.subplots(figsize=(10,max(4,len(aris)*.16)))
            ax.barh(range(len(aris)),[v for _,v in aris]);ax.set_yticks(range(len(aris)),labels=[n for n,_ in aris],fontsize=5)
            ax.set(xlabel='ARI (K-means engineering baseline)');fig.tight_layout();fig.savefig(out/'ari.png');plt.close(fig);generated.append('ari.png')
    # RSA matrices and metadata are paired; sort by genuine labels, not array order.
    for path in source.rglob('rsa_layer*.npz'):
        arrays=np.load(path,allow_pickle=False)
        for side in ('a','b'):
            for sort_label in ('skill','intermediate_state','answer'):
                arr=arrays[side]
                order=arrays['order_'+sort_label] if 'order_'+sort_label in arrays else np.arange(len(arr))
                arr=arr[np.ix_(order,order)]
                fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(arr,vmin=-1,vmax=1,cmap='coolwarm')
                ax.set(title=f'{path.parent.name}/{path.stem}/{side}',xlabel='sorted by '+sort_label,ylabel='sorted by '+sort_label)
                fig.colorbar(im,ax=ax);fig.tight_layout()
                filename=str(path.relative_to(source)).replace('/','_').replace('.npz','')+'_'+side+'_'+sort_label+'.png'
                fig.savefig(out/filename);plt.close(fig);generated.append(filename)
    # Optional one-row-per-run association table. Stage is always explicit.
    associations=source/'associations.jsonl'
    if associations.exists():
        rows=read_jsonl(associations)
        allowed={'pretraining_predictor','early_predictor','posttraining_correlation'}
        if any(r.get('stage') not in allowed for r in rows): raise ValueError('Association stage missing')
        ids=[(r['run_id'],r['task'],r['stage']) for r in rows]
        if len(set(ids))!=len(ids): raise ValueError('Association unit must be run/task, never token')
        for stage in allowed:
            group=[r for r in rows if r['stage']==stage]
            if not group: continue
            fig,ax=plt.subplots(figsize=(6,4))
            for split in ('fit','validation'):
                rr=[r for r in group if r['analysis_split']==split]
                if rr: ax.scatter([r['representation_metric'] for r in rr],[r['unprivileged_generalization_gain'] for r in rr],label=split)
            fit_runs={r['run_id'] for r in group if r['analysis_split']=='fit'}
            val_runs={r['run_id'] for r in group if r['analysis_split']=='validation'}
            if fit_runs&val_runs: raise ValueError('Run leakage in association fit/validation split')
            ax.set(title=stage,xlabel='representation metric (one run/task)',ylabel='unprivileged generalization gain');ax.legend();fig.tight_layout()
            fig.savefig(out/(stage+'.png'));plt.close(fig);generated.append(stage+'.png')
    write_json(out/'plot_manifest.json',{'source':str(source.resolve()),'figures':generated,
                                       'missing_data':'No plot is invented for missing inputs'})
    return generated
