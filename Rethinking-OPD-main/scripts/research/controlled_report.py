"""Paired question-level accuracy and exposure audit for the three-method run."""
import itertools
import html
import io
import json
import math
from pathlib import Path

import numpy as np

from .common import file_hash, read_jsonl, write_json


def paired(a,b,seed=42,replicates=2000):
    if set(a)!=set(b) or not a:raise ValueError('Comparison requires identical nonempty question IDs')
    ids=sorted(a);delta=np.asarray([int(b[i])-int(a[i]) for i in ids])
    rng=np.random.default_rng(seed)
    samples=np.asarray([delta[rng.integers(0,len(ids),len(ids))].mean() for _ in range(replicates)])
    gain=int((delta>0).sum());loss=int((delta<0).sum());n=gain+loss
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(gain,loss)+1))/2**n) if n else 1.
    return dict(delta=float(delta.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),
                gained=gain,lost=loss,exact_mcnemar_p=p,n=len(ids),bootstrap_seed=seed,replicates=replicates)


def build(workdir):
    root=Path(workdir);config=json.loads((root/'config.json').read_text())
    schedule_path=root/'shared/probe/question_schedule.json'
    schedule=json.loads(schedule_path.read_text())['batches']
    result=dict(methods={},comparisons={},schedule_sha256=file_hash(schedule_path))
    finals={};baseline=None
    for method in ('opsd','sft','grpo'):
        states={};series=[]
        for step in config['monitor']['optimizer_steps']:
            directory=root/method/'analysis/steps'/f'step_{step:06d}'
            if not (directory/'complete.json').is_file():continue
            rows=[r for r in read_jsonl(directory/'live_rollouts.jsonl') if r['split']=='heldout_probe']
            current={r['problem_id']:bool(r['correct']) for r in rows}
            if len(current)!=len(rows) or not rows:raise ValueError('Duplicate or empty heldout evaluation')
            states[step]=current;series.append(dict(step=step,n=len(rows),accuracy=sum(current.values())/len(rows),
                extraction_failure=sum(r['extraction_failed'] for r in rows)/len(rows),
                truncation=sum(r['truncated'] for r in rows)/len(rows)))
        if not states:
            result['methods'][method]=dict(status='not_analyzed');continue
        if 0 not in states:raise ValueError('Missing method baseline')
        if baseline is None:baseline=states[0]
        if states[0]!=baseline:raise ValueError('Methods have different Base outcomes; investigate before comparison')
        last=max(states);exposure=root/method/'train/exposures'
        for batch in schedule[:last]:
            path=exposure/f"step_{batch['step']:06d}.json"
            record=json.loads(path.read_text())
            if record['schedule_sha256']!=result['schedule_sha256'] or record['problem_ids']!=batch['problem_ids']:
                raise ValueError('Observed training exposure differs across methods')
        last_budget=None
        metrics=root/method/'train/metrics.jsonl'
        if metrics.is_file():
            entries=[r for r in read_jsonl(metrics) if r['step']<=last]
            if entries:last_budget=entries[-1]['budget']
        result['methods'][method]=dict(status='complete' if last==config['training']['steps'] else 'partial',
                    final_step=last,series=series,change=paired(states[0],states[last]),budget=last_budget)
        if last==config['training']['steps']:finals[method]=states[last]
    for a,b in itertools.combinations(finals,2):result['comparisons'][f'{b}_minus_{a}']=paired(finals[a],finals[b])
    out=root/'report';out.mkdir(exist_ok=True)
    write_json(out/'controlled_summary.json',result)
    lines=['# OPSD / SFT / GRPO 配对对照结果','',
           '三组共享冻结题集、Base 初始参数、LoRA 初始化、逐步题目顺序、更新数和评测协议。',
           'SFT 的每题 8 份是同一离线 reference 的重复监督；OPSD/GRPO 是 8 条自主采样回答。相同更新数和序列数不代表相同 token 或计算预算。','',
           '| 方法 | 状态 | 末次 step | 初始→最终正确率 | 配对变化 95% 区间（百分点） |',
           '|---|---|---:|---|---|']
    for name,r in result['methods'].items():
        if 'series' not in r:lines.append(f'| {name} | 尚未分析 | — | — | — |');continue
        a,b=r['series'][0],r['series'][-1];ci=r['change']['ci95']
        lines.append(f"| {name} | {r['status']} | {r['final_step']} | {a['accuracy']:.1%}→{b['accuracy']:.1%} | [{100*ci[0]:.1f}, {100*ci[1]:.1f}] |")
    lines+=['','## 实际训练预算','',
            '| 方法 | 题目呈现 | rollout tokens | optimized tokens | teacher forward tokens | 训练预算秒 |',
            '|---|---:|---:|---:|---:|---:|']
    for name,r in result['methods'].items():
        budget=r.get('budget') or {}
        values=[str(round(budget[k],2)) if k in budget else '—' for k in
                ['questions','rollout_tokens','optimized_tokens','teacher_forward_tokens','wall_seconds']]
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines+=['','## 方法间最终状态的同题比较','']
    for label,r in result['comparisons'].items():
        lines.append(f"- {label}：差 {100*r['delta']:.1f} 个百分点，95% 区间 [{100*r['ci95'][0]:.1f}, {100*r['ci95'][1]:.1f}]，McNemar p={r['exact_mcnemar_p']:.6g}。")
    lines+=['','区间按问题成对重采样；单 seed、50 题且已用于验收，不能据此作跨 seed 或独立外部测试的稳定性结论。',
            '曲率/RSA/CKA、固定前缀与 live 分布、实际训练 token/耗时见同目录 OPSD_report.html。',
            'SFT/GRPO 的特权条件只用于诊断，训练时没有特权 logits 教师。三组损失函数的数值不能直接横向比较。','']
    target=out/'CONTROLLED_COMPARISON_zh.md';target.write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method,r in result['methods'].items():
        if 'series' not in r:continue
        x=[v['step'] for v in r['series']]
        for ax,key in zip(axes,['accuracy','extraction_failure']):
            ax.plot(x,[100*v[key] for v in r['series']],'.-',label=method)
    for ax,title in zip(axes,['Heldout accuracy','Extraction failure']):
        ax.set(xlabel='optimizer step',ylabel='%',title=title);ax.grid(alpha=.2)
        if ax.lines:ax.legend()
    fig.tight_layout();image=io.StringIO();fig.savefig(image,format='svg');plt.close(fig)
    svg='<svg'+image.getvalue().split('<svg',1)[1]
    page=('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
          '<title>OPSD / SFT / GRPO 配对对照</title><style>body{max-width:1050px;margin:24px auto;padding:0 16px;'
          'font:16px/1.8 system-ui;color:#203743}svg{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
          '</style><h1>OPSD / SFT / GRPO 配对对照</h1><p>同题、同初始参数、同训练题序。曲线为描述性比例；配对区间和预算见下文。</p>'
          +svg+'<pre>'+html.escape('\n'.join(lines))+'</pre><script type="application/json" id="controlled-data">'
          +json.dumps(result,ensure_ascii=False).replace('<','\\u003c')+'</script></html>')
    (out/'CONTROLLED_COMPARISON.html').write_text(page)
    return target
