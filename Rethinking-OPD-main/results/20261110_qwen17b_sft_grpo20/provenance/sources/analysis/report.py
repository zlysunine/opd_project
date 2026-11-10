"""Rebuild all figures and comparison tables from caches, without loading models."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.research.common import read_jsonl, write_json
from .common import csv_table, table


def report(run_dir):
    root = Path(run_dir)
    output = root/'report'
    output.mkdir(parents=True,exist_ok=True)
    rows, steps = [], []
    for path in sorted(root.rglob('metrics.jsonl')):
        if path.parent==output or not (path.parent/'complete.json').is_file():
            continue
        values = read_jsonl(path)
        if values and 'optimizer_step' in values[0]:
            rows.extend(values)
            steps.append(path.parent)
    if not rows:
        raise ValueError('No completed checkpoint analyses to report')
    hashes = {json.loads((step/'complete.json').read_text())['identity']['probe_manifest_hash'] for step in steps}
    if len(hashes)!=1:
        raise ValueError('Comparison requires exactly the same frozen probe manifest')
    formats={json.loads((s/'run_manifest.json').read_text())['config']['table_format'] for s in steps}
    table(output/'metrics',rows,'parquet' if formats=={'parquet'} else 'jsonl')
    csv_table(output/'metrics.csv',rows)
    plots = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['pool'],row['split'],row['role'],row.get('anchor'),row['metric'])].append(row)
    for (pool,split,role,anchor,metric), values in grouped.items():
        if not any(r['value'] is not None for r in values):
            continue
        label = '__'.join(str(x or 'none') for x in (pool,split,role,anchor,metric))
        layered = any(r['layer'] is not None for r in values)
        fig,ax = plt.subplots(figsize=(8,4))
        for run in sorted({r['run_id'] for r in values}):
            subset = [r for r in values if r['run_id']==run]
            if layered:
                for step in sorted({r['optimizer_step'] for r in subset}):
                    points = sorted([r for r in subset if r['optimizer_step']==step], key=lambda r:r['layer'])
                    ax.plot([r['layer'] for r in points], [r['value'] if r['value'] is not None else np.nan for r in points],
                            label=f'{run} step{step}', alpha=.8)
            else:
                points = sorted(subset,key=lambda r:r['optimizer_step'])
                ax.plot([r['optimizer_step'] for r in points],
                        [r['value'] if r['value'] is not None else np.nan for r in points],'.-',label=run)
        ax.set(xlabel='decoder block (1-based)' if layered else 'optimizer step',ylabel=metric,
               title=f'{pool} / {split} / {role} / {anchor or "response"}')
        ax.grid(alpha=.2)
        ax.legend(fontsize=6)
        fig.tight_layout()
        file=output/f'{label}.png'
        fig.savefig(file)
        plt.close(fig)
        plots.append(file.name)
        if layered:
            runs = sorted({r['run_id'] for r in values})
            fig,axes = plt.subplots(1,len(runs),figsize=(7*len(runs),4),squeeze=False)
            for run,ax in zip(runs,axes[0]):
                subset=[r for r in values if r['run_id']==run]
                ss=sorted({r['optimizer_step'] for r in subset})
                ll=sorted({r['layer'] for r in subset})
                lookup={(r['optimizer_step'],r['layer']):r['value'] for r in subset}
                matrix=np.asarray([[lookup.get((s,l)) if lookup.get((s,l)) is not None else np.nan for l in ll] for s in ss])
                image=ax.imshow(matrix,aspect='auto',interpolation='nearest')
                ax.set(xticks=range(len(ll)),xticklabels=ll,yticks=range(len(ss)),yticklabels=ss,
                       xlabel='block',ylabel='optimizer step',title=run)
                ax.tick_params(axis='x',labelsize=5)
                fig.colorbar(image,ax=ax,label=metric)
            fig.tight_layout()
            file=output/f'{label}__heatmap.png'
            fig.savefig(file)
            plt.close(fig)
            plots.append(file.name)
    # Full RSM panels only use layers/steps selected in advance, derived from relative depth.
    for step in steps:
        manifest=json.loads((step/'run_manifest.json').read_text())
        config=manifest['config']['monitor']
        number=manifest['identity']['step']
        if number not in config['plot_full_rsm_steps']:
            continue
        count=manifest['hooks']['layers']
        layers=sorted({1+round(f*(count-1)) for f in config['plot_relative_depths']})
        meta=json.loads((step/'anchors.json').read_text())
        for key,value in meta.items():
            if value['layer'] not in layers:
                continue
            with np.load(step/f'rsm_{key}.npz',allow_pickle=False) as data:
                fig,ax=plt.subplots(figsize=(5,4))
                image=ax.imshow(data['rsm'],vmin=-1,vmax=1,cmap='coolwarm',interpolation='nearest')
                ax.set(title=f"{manifest['config']['run_id']} step{number} {key}",xlabel='fixed question order',ylabel='fixed question order')
                ax.title.set_fontsize(6)
                fig.colorbar(image,ax=ax,label='raw cosine')
                fig.tight_layout()
                file=output/f"rsm__{manifest['config']['run_id']}__step{number}__{key}.png"
                fig.savefig(file)
                plt.close(fig)
                plots.append(file.name)
    # Geometry and behavior together; fixed geometry and live behavior retain separate labels.
    for split in sorted({r['split'] for r in rows}):
        fig,axes=plt.subplots(1,4,figsize=(16,3.5))
        specifications=[('curvature_training_delta_deg','fixed_base_rollout'),('accuracy_micro','live_rollout'),
                        ('response_length','live_rollout'),('truncation_rate','live_rollout')]
        for ax,(metric,pool) in zip(axes,specifications):
            candidates=[r for r in rows if r['split']==split and r['pool']==pool and r['role']=='student' and r['metric']==metric]
            for run in sorted({r['run_id'] for r in candidates}):
                values=[r for r in candidates if r['run_id']==run]
                if metric.startswith('curvature'):
                    last=max((r['layer'] for r in values),default=1)
                    values=[r for r in values if r['layer']==last]
                values.sort(key=lambda r:r['optimizer_step'])
                ax.plot([r['optimizer_step'] for r in values],[r['value'] if r['value'] is not None else np.nan for r in values],'.-',label=run)
            ax.set(xlabel='optimizer step',ylabel=metric,title=pool)
            ax.legend(fontsize=6)
        fig.suptitle(split)
        fig.tight_layout()
        file=output/f'behavior_and_fixed_curvature__{split}.png'
        fig.savefig(file)
        plt.close(fig)
        plots.append(file.name)
    # Per-question scatter: data points are questions, never tokens as independent observations.
    for step in steps:
        manifest=json.loads((step/'run_manifest.json').read_text())
        count=manifest['hooks']['layers']
        layers=sorted({1+round(f*(count-1)) for f in manifest['config']['monitor']['plot_relative_depths']})
        curves=read_jsonl(step/'curvature.jsonl')
        for split in ('heldout_probe','train_probe'):
            fig,axes=plt.subplots(1,len(layers),figsize=(4*len(layers),3),squeeze=False)
            for layer,ax in zip(layers,axes[0]):
                subset=[r for r in curves if r['layer']==layer and r['split']==split and r['role']=='student'
                        and r['pool']=='fixed_base_rollout' and r['mean_deg'] is not None and r['surprisal_bits'] is not None]
                ax.scatter([r['surprisal_bits'] for r in subset],[r['mean_deg'] for r in subset],s=8,alpha=.5)
                ax.set(xlabel='surprisal (bits/token)',ylabel='curvature (degrees)',title=f'block {layer}')
            fig.tight_layout()
            file=output/f"surprisal_scatter__{manifest['config']['run_id']}__step{manifest['identity']['step']}__{split}.png"
            fig.savefig(file)
            plt.close(fig)
            plots.append(file.name)
    write_json(output/'report_manifest.json',dict(source_steps=[str(s) for s in steps],plots=plots,
               probe_manifest_hash=next(iter(hashes)),
               interpretation='Fixed-pool geometry and live behavior are separate. Geometry changes do not establish better mathematical ability.'))
    from .html_report import build
    html,payload=build(root,output/'OPSD_report.html')
    manifest=json.loads((output/'report_manifest.json').read_text())
    manifest.update(html=html.name,html_standalone=True,html_metrics=len(payload['metrics']))
    write_json(output/'report_manifest.json',manifest)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    print(report(args.run_dir))


if __name__=='__main__':
    main()
