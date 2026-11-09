"""Audit an exported experiment and build a narrative report in the first-run style.

Only published scalar tables are read. Bootstrap units are questions; distribution
draws retain each metric's own token denominator. Source reports are immutable.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import hashlib
from html import escape
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_table(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def num(value):
    return float(value) if value not in ('',None) else np.nan


def truth(value):
    if value in (True,'True','true','1',1):return True
    if value in (False,'False','false','0',0):return False
    raise ValueError(f'Unexpected boolean measurement: {value!r}')


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def write_json(path,value):
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def paired_ids(before,after):
    if set(before)!=set(after):raise ValueError('Paired problem IDs differ')
    return sorted(before)


def bootstrap_change(s0,n0,s1,n1,indices):
    """Ratio-of-sums change, resampling paired questions (not tokens)."""
    a,b,c,d=(np.asarray(x,dtype=float) for x in (s0,n0,s1,n1))
    if not (a.shape==b.shape==c.shape==d.shape) or a.ndim!=1:
        raise ValueError('Aligned per-question numerator/denominator vectors required')
    if not all(np.isfinite(x).all() for x in (a,b,c,d)) or (b<0).any() or (d<0).any():
        raise ValueError('Invalid cluster sums/counts')
    v0=a.sum()/b.sum() if b.sum() else np.nan
    v1=c.sum()/d.sum() if d.sum() else np.nan
    den0=b[indices].sum(1);den1=d[indices].sum(1)
    valid=(den0>0)&(den1>0)
    draws=c[indices].sum(1)[valid]/den1[valid]-a[indices].sum(1)[valid]/den0[valid]
    return dict(initial=v0,final=v1,change=v1-v0,ci95=np.quantile(draws,[.025,.975]) if len(draws) else [np.nan,np.nan],
                n_questions=len(a),initial_tokens=int(b.sum()),final_tokens=int(d.sum()),
                valid_bootstrap_draws=int(valid.sum()))


def exact_mcnemar(gained,lost):
    n=gained+lost
    return min(1.,2*sum(math.comb(n,k) for k in range(min(gained,lost)+1))/2**n) if n else 1.


def wilson(correct,n):
    z=1.959963984540054;p=correct/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half,center+half]


def aggregate_key(row):
    return (int(row.get('optimizer_step',row.get('step'))),row['pool'],row['split'],row['role'],
            row.get('anchor',''),int(row['layer']) if row.get('layer') else None,row['metric'])


def audit(source,replicates=2000,seed=20261028):
    if replicates<1:raise ValueError('Positive bootstrap count required')
    manifest=json.loads((source/'artifact_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        p=(source/name).resolve()
        if not p.is_relative_to(source.resolve()):raise ValueError('Unsafe source manifest path')
        if p.stat().st_size!=expected['bytes'] or sha(p)!=expected['sha256']:
            raise ValueError('Source export changed: '+name)
    config=json.loads((source/'provenance/config.json').read_text())
    original=json.loads((source/'summary.json').read_text())
    training=[json.loads(x) for x in (source/'data/training_metrics.jsonl').read_text().splitlines() if x.strip()]
    monitor=read_table(source/'data/checkpoint_metrics.csv.gz')
    distributions=read_table(source/'data/distributions.csv.gz')
    curvature=read_table(source/'data/curvature.csv.gz')
    behavior=read_table(source/'data/behavior.csv')
    members=read_table(source/'data/probe_membership.csv')
    steps=sorted({int(r['optimizer_step']) for r in monitor});last=steps[-1]
    assert [r['step'] for r in training]==list(range(1,last+1)),'Missing/duplicate training steps'
    lookup={aggregate_key(r):r for r in monitor}
    assert len(lookup)==len(monitor),'Duplicate checkpoint metric contexts'
    counts=Counter()
    def check(value,record,n=None):
        expected=num(record['value'])
        assert (not np.isfinite(value) and not np.isfinite(expected)) or np.isclose(value,expected,rtol=1e-8,atol=1e-10),record['metric']
        if n is not None:assert int(record['n_valid'])==n,(record['metric'],n,record['n_valid'])
    split_ids={s:{r['problem_id'] for r in members if r['split']==s} for s in ('train','validation','heldout_probe','train_probe')}
    assert not split_ids['train']&split_ids['validation'] and not split_ids['train']&split_ids['heldout_probe']
    assert not split_ids['validation']&split_ids['heldout_probe']
    assert split_ids['train_probe']<=split_ids['train']
    for row in training:
        m=row['metrics'];assert m['actor/lr']==config['training']['learning_rate']
        assert row['budget']['questions']==row['step']*config['training']['batch']
        counts['training_rows']+=1
    dgroups=defaultdict(dict)
    for row in distributions:
        assert row['distribution_scope']=='all_valid_tokens' and row['aggregation']=='token_mean'
        key=(int(row['step']),row['pool'],row['split'])
        assert row['problem_id'] not in dgroups[key]
        dgroups[key][row['problem_id']]=row
    names=sorted(k.split('/',1)[1] for k in distributions[0] if k.startswith('metric_sums/'))
    for (step,pool,split),records in dgroups.items():
        assert set(records)==split_ids[split]
        for name in names:
            total=sum(num(r['metric_sums/'+name]) for r in records.values())
            n=sum(int(num(r['metric_token_counts/'+name])) for r in records.values())
            check(total/n if n else np.nan,lookup[(step,pool,split,'student','',None,name)],n)
            counts['distribution_aggregates_checked']+=1
    bstats={};dstats={};cstats={};rstats={}
    for si,split in enumerate(('heldout_probe','train_probe')):
        ids=sorted(split_ids[split]);n=len(ids)
        indices=np.random.default_rng(seed+si).integers(0,n,size=(replicates,n))
        by_step={step:{r['problem_id']:r for r in behavior if int(r['step'])==step and r['split']==split} for step in steps}
        series=[]
        for step,records in by_step.items():
            assert set(records)==set(ids)
            row=dict(step=step,n=n)
            for key in ('correct','truncated','extraction_failed','format_compliant','eos'):
                row[key]=sum(truth(records[i][key]) for i in ids)
            row['accuracy_ci95']=wilson(row['correct'],n)
            row['mean_length']=float(np.mean([int(records[i]['response_length']) for i in ids]))
            row['parse_status']=dict(Counter(r['parse_status'] for r in records.values()))
            for metric,field in [('accuracy_micro','correct'),('truncation_rate','truncated'),('extraction_failure_rate','extraction_failed')]:
                check(row[field]/n,lookup[(step,'live_rollout',split,'student','',None,metric)])
                counts['behavior_aggregates_checked']+=1
            series.append(row)
        initial,final=by_step[0],by_step[last]
        before=np.asarray([truth(initial[i]['correct']) for i in ids],dtype=float)
        after=np.asarray([truth(final[i]['correct']) for i in ids],dtype=float)
        gains=int(((before==0)&(after==1)).sum());losses=int(((before==1)&(after==0)).sum())
        bstats[split]=dict(series=series,gained=gains,lost=losses,unchanged_correct=int(((before==1)&(after==1)).sum()),
                          exact_mcnemar_p=exact_mcnemar(gains,losses),
                          accuracy_change=bootstrap_change(before,np.ones(n),after,np.ones(n),indices))
        dstats[split]={}
        for pool in ('fixed_base_rollout','live_rollout'):
            changes={};base=dgroups[(0,pool,split)];final=dgroups[(last,pool,split)]
            paired_ids(base,final)
            if pool=='fixed_base_rollout':
                for step in steps:
                    assert all(base[i]['response_length']==dgroups[(step,pool,split)][i]['response_length'] for i in ids)
            for name in names:
                sums=lambda records:[num(records[i]['metric_sums/'+name]) for i in ids]
                ns=lambda records:[num(records[i]['metric_token_counts/'+name]) for i in ids]
                changes[name]=bootstrap_change(sums(base),ns(base),sums(final),ns(final),indices)
            dstats[split][pool]=dict(changes=changes,n_questions=n,
                weighting='sum of per-question token sums / sum of per-question token counts')
        rstats[split]={}
        for comparison,role,fields in [
            ('student_initial','student',['rsa_rho_to_initial','linear_cka_to_initial','neighbor_overlap_k16','rsa_rmse_to_initial']),
            ('student_teacher','student_vs_privileged_teacher',['rsa_rho_student_teacher','linear_cka_student_teacher','neighbor_overlap_student_teacher_k16','rsa_rmse_student_teacher']),
            ('teacher_initial','privileged_teacher',['rsa_rho_to_initial','linear_cka_to_initial','neighbor_overlap_k16'])]:
            rstats[split][comparison]={}
            for field in fields:
                selected=[r for r in monitor if r['metric']==field and r['pool']=='fixed_base_rollout' and r['split']==split and r['role']==role]
                before={(r['anchor'],int(r['layer'])):num(r['value']) for r in selected if int(r['optimizer_step'])==0}
                after={(r['anchor'],int(r['layer'])):num(r['value']) for r in selected if int(r['optimizer_step'])==last}
                valid={k:after[k]-v for k,v in before.items() if np.isfinite(v) and np.isfinite(after[k])}
                final_values=[v for v in after.values() if np.isfinite(v)]
                by_anchor={a:dict(initial=before.get((a,28)),final=after.get((a,28)),
                            n=next((int(r['n_valid']) for r in selected if r['anchor']==a and r['layer']=='28' and int(r['optimizer_step'])==last),0))
                           for a in config['monitor']['anchors']}
                rstats[split][comparison][field]=dict(min=min(final_values) if final_values else None,
                    max=max(final_values) if final_values else None,n_measured=len(valid),increased=sum(v>0 for v in valid.values()),
                    decreased=sum(v<0 for v in valid.values()),min_delta=min(valid.values()) if valid else None,
                    max_delta=max(valid.values()) if valid else None,final_layer=by_anchor,
                    minimum_at=list(min(after,key=lambda k:after[k] if np.isfinite(after[k]) else math.inf)) if final_values else None)
        cstats[split]={}
        for role in ('student','privileged_teacher'):
            rows=[r for r in curvature if r['split']==split and r['pool']=='fixed_base_rollout' and r['role']==role]
            layers=sorted({int(r['layer']) for r in rows})
            table={(int(r['step']),int(r['layer']),r['problem_id']):r for r in rows}
            common=[i for i in ids if all(np.isfinite(num(table[(st,l,i)]['mean_deg'])) for st in steps for l in layers)]
            ci_idx=np.random.default_rng(seed+si).integers(0,len(common),size=(replicates,len(common))) if common else np.empty((replicates,0),dtype=int)
            profiles=[]
            for layer in layers:
                a=np.asarray([num(table[(0,layer,i)]['mean_deg']) for i in common])
                b=np.asarray([num(table[(last,layer,i)]['mean_deg']) for i in common])
                delta=bootstrap_change(a,np.ones(len(a)),b,np.ones(len(b)),ci_idx)
                for st in steps:
                    values=[num(table[(st,layer,i)]['mean_deg']) for i in common]
                    check(np.mean(values) if values else np.nan,lookup[(st,'fixed_base_rollout',split,role,'',layer,'curvature_deg')],len(values))
                    counts['curvature_aggregates_checked']+=1
                check(delta['change'],lookup[(last,'fixed_base_rollout',split,role,'',layer,'curvature_training_delta_deg')],len(common))
                profiles.append(dict(layer=layer,**delta))
            cstats[split][role]=dict(n=len(common),profiles=profiles,
                                   initial_min_layer=min(profiles,key=lambda r:r['initial'])['layer'],
                                   final_min_layer=min(profiles,key=lambda r:r['final'])['layer'])
    tstats={}
    for name in sorted(set().union(*(r['metrics'] for r in training))):
        values=np.asarray([r['metrics'].get(name,np.nan) for r in training],dtype=float)
        valid=values[np.isfinite(values)]
        avg=lambda x:float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan
        tstats[name]=dict(first=values[0],last=values[-1],first5=avg(values[:5]),last5=avg(values[-5:]),
                         min=float(valid.min()) if len(valid) else np.nan,
                         max=float(valid.max()) if len(valid) else np.nan,mean=avg(values),n_recorded=len(valid))
    analysis_times={step:json.loads((source/f'provenance/steps/{step:06d}/run_manifest.json').read_text())['analysis_wall_seconds'] for step in steps}
    formal_seconds=original['session']['finished_utc']
    formal_seconds=(datetime.fromisoformat(formal_seconds)-datetime.fromisoformat(original['session']['started_utc'])).total_seconds()
    integrity=dict(source_manifest_sha256=sha(source/'artifact_manifest.json'),verified_source_files=len(manifest['files']),
                   row_counts=dict(training=len(training),checkpoint=len(monitor),distribution=len(distributions),curvature=len(curvature),behavior=len(behavior)),
                   checks=dict(counts),undefined_checkpoint_rows=sum(not np.isfinite(num(r['value'])) for r in monitor),
                   source_scope='Exported scalar tables and archived configuration, not server weights/text/hidden arrays')
    summary=dict(config=config,steps=steps,final_step=last,training=tstats,distributions=dstats,
                 representation=rstats,curvature=cstats,behavior=bstats,final_budget=original['final_budget'],
                 resources=dict(formal_session_seconds=formal_seconds,analysis_seconds_by_step=analysis_times,
                                analysis_seconds_sum=sum(analysis_times.values())),
                 reuse=original['reuse'],topics={sp:dict(Counter(r['topic'] for r in members if r['split']==sp)) for sp in split_ids},
                 bootstrap=dict(replicates=replicates,seed=seed,unit='paired question clusters',multiple_testing_corrected=False),
                 integrity=integrity)
    compact=defaultdict(list)
    for r in monitor:
        k='|'.join([r['metric'],r['pool'],r['split'],r['role'],r['anchor']])
        compact[k].append([int(r['optimizer_step']),int(r['layer']) if r['layer'] else None,num(r['value']),int(r['n_valid']),num(r.get('n_pairs'))])
    return clean(summary),clean(dict(training=[dict(step=r['step'],**r['metrics']) for r in training],monitor=compact))


def markdown(s):
    f=lambda x,n=6:f'{x:.{n}f}'
    h=s['behavior']['heldout_probe'];start=h['series'][0];end=h['series'][-1]
    fixed=s['distributions']['heldout_probe']['fixed_base_rollout']['changes']
    repr=s['representation']['heldout_probe'];cfg=s['config'];c=s['curvature']['heldout_probe']['student']
    text=['# 第二次OPSD实验：20步结果复核与解读','',
          '分析日期：2026-10-28，Asia/Shanghai。按第一次报告的“结论—在线分布—固定前缀—表征—曲率—自主生成—资源—复核”顺序组织。原始服务器报告不覆盖、不修改。','',
          '**核心结论：训练后观测到小幅分布和表征变化；直接学生—教师表征接近程度随层/anchor有升有降，未显示普遍对齐。heldout正确率50%→46%，没有可靠提升证据。**','',
          '## 1. 实际实验范围与协议','',
          f"Qwen3-1.7B-Base，LoRA r16/alpha32，lr=1e-6，seed42，8题×8条轨迹，实际{ s['final_step']}次更新；DeepMath难度5–7，离线r1_solution_1，训练池2814、validation512、heldout50、train_probe4。单卡A80080GB，无GRPO对照。",'',
          '训练/live回答上限16384，学生和教师prompt各≤2048，固定回答/几何窗口256。分布对全部有效回答位置计算全词表熵/KL，主值按token加权；不是第一次的1/32/128选点等题权统计。几何anchor仍为prompt_last/response_32/response_128。','',
          '**实际抽取成功率门槛为70%（失败率≤30%），并非早期设计的失败率≤1%。** 五个监测点均超过70%，只能说通过变更后的门槛。heldout取此前冻结100题顺序的前50，包含失败题，并被用于验收；它未参与梯度训练，但不是从未看过的最终外部测试集。','',
          '正式20步从Base独立初始化，不是从两步验收checkpoint继续；复用验收的固定题目、回答及基线，正式扩充训练池。20×8=160次题目暴露、1280条训练轨迹；候选池2814不等于全部训练过，4题train_probe也不保证已被抽到。','',
          '## 2. 数据一致性','',
          f"已核验导出包{s['integrity']['verified_source_files']}个文件的大小/SHA256，读取20行训练、41490行checkpoint标量、540行逐题分布、22680行曲率、270行行为。复算{s['integrity']['checks']['distribution_aggregates_checked']}个token加权分布汇总、{s['integrity']['checks']['curvature_aggregates_checked']}个曲率汇总及行为率，检查固定ID与成员隔离。",'',
          f"本地核验的是导出标量及配置，不是重新检查服务器权重、原始答案文本或隐藏向量。{s['integrity']['undefined_checkpoint_rows']}条undefined原样保留，不是同等次数的训练失败；train_probe只有4题，k=8/16/32近邻无法定义。该4题分属不同主题，主题内统计也可能无定义；教师surprisal及部分live配对量没有测量。",'',
          '## 3. 在线分布：高重叠，缺少持续下降趋势','',
          '| 指标 | 前5步均值 | 后5步均值 |','|---|---:|---:|']
    for name in ['actor/forward_kl','opd/overlap_ratio','opd/overlap_token_advantage','opd/student_entropy','opd/teacher_entropy','opd/entropy_gap']:
        r=s['training'][name];text.append(f"| {name} | {f(r['first5'])} | {f(r['last5'])} |")
    text+=['','不同batch的题目/回答和自教师参数都变化；在线均值不能当固定测试集的显著性趋势。OTA是交集内归一化的负KL/交集大小，不是GRPO策略优势。','',
           '## 4. 固定前缀：两组比较分开','',
           '| 指标 | step0 | step20 | 配对问题bootstrap的变化95%区间 |','|---|---:|---:|---:|']
    for name in ['kl_student_teacher_nats','kl_teacher_student_nats','kl_to_initial_nats','token_top16_overlap','token_top16_overlap_to_initial',
                 'overlap_token_advantage_nats','overlap_token_advantage_to_initial_nats','entropy_gap_abs_nats','entropy_gap_to_initial_abs_nats']:
        r=fixed[name];text.append(f"| {name} | {f(r['initial'])} | {f(r['final'])} | [{f(r['ci95'][0])}, {f(r['ci95'][1])}] |")
    drift=fixed['kl_to_initial_nats'];gap=fixed['entropy_gap_abs_nats']
    text+=['',f"固定学生/教师KL略降，但绝对熵差{f(gap['initial'])}→{f(gap['final'])}并未同步下降，不能概括为所有指标一致改善。当前/初始KL最终仅{f(drift['final'])}nats；step1已出现绝大部分差别，仍需零更新导出/加载对照估计BF16等数值底噪。",'',
           '区间按问题成簇、同题成对重采样；每次保留原token分子/分母，再算比值之差。OTA使用自己的非空交集分母。不是对token独立bootstrap，也不是对每题均值简单等权。live前缀跨步改变，其趋势混合参数与访问状态变化。','',
           '## 5. 表征：初始保持与直接条件对齐','',
           '| 比较/指标 | step20跨层及anchor范围 | 增加/减少的层-anchor数量 |','|---|---:|---:|']
    for comp,field in [('student_initial','rsa_rho_to_initial'),('student_initial','linear_cka_to_initial'),
                       ('student_teacher','rsa_rho_student_teacher'),('student_teacher','linear_cka_student_teacher')]:
        r=repr[comp][field];text.append(f"| {field} | {f(r['min'],9)}–{f(r['max'],9)} | {r['increased']}/{r['decreased']} |")
    text+=['','| 末层anchor | 学生—教师RSA step0 | step20 | 有效题数 |','|---|---:|---:|---:|']
    for a,r in repr['student_teacher']['rsa_rho_student_teacher']['final_layer'].items():
        text.append(f"| {a} | {f(r['initial'])} | {f(r['final'])} | {r['n']} |")
    text+=['','本次已直接计算S_s/T_s，而T_s/T_0是另一组辅助曲线。直接RSA的84个层-anchor有51个上升、33个下降；CKA也有升有降，不支持“所有层普遍对齐”。response32/128在step0已很接近，与双方读取同一回答前缀的设置相符，其贡献未作独立消融；prompt_last更能显出特权上下文差异，但同时包含提示长度和位置改变。','',
           'RSA/CKA接近1表示题间关系结构接近，不等于向量坐标相同或推理正确。未发布anchors/RSM数值数组，无法在本地用问题bootstrap重新计算RSA/CKA置信区间。初始CKA偶有约2.4e-7的越界，按浮点舍入保留原值，不解读为超过完全相似。','',
           '## 6. 曲率：主要层间形状仍来自初始模型','',
           '| 层 | step0转角 | step20转角 | 配对变化95%区间（度） |','|---|---:|---:|---:|']
    for r in c['profiles']:
        if r['layer'] in [1,16,28]:text.append(f"| {r['layer']} | {f(r['initial'])} | {f(r['final'])} | [{f(r['ci95'][0])}, {f(r['ci95'][1])}] |")
    relative0=min(r['initial'] for r in c['profiles'])-c['profiles'][0]['initial']
    relative1=min(r['final'] for r in c['profiles'])-c['profiles'][0]['final']
    text +=['',f"最低平均曲率层step0/20分别为{c['initial_min_layer']}/{c['final_min_layer']}；相对首层的最低差值为{relative0:.6f}°→{relative1:.6f}°。约7.10°层间下降在初始模型就存在，不能算作训练新增效果。教师与学生的同题曲率差可从本次逐题表派生，但应标为本地派生统计；它不等于逐token向量对齐。所有层的探索性区间未做多重比较校正。live文本跨步改变，不做固定文本的配对训练差值。",'',
            '## 7. 自主生成：没有改善证据','',
            '| step | 正确题数/50 | 准确率 | 抽取失败率 | 截断率 | 格式合规率 |','|---|---:|---:|---:|---:|---:|']
    for r in h['series']:text.append(f"| {r['step']} | {r['correct']}/{r['n']} | {r['correct']/r['n']:.0%} | {r['extraction_failed']/r['n']:.0%} | {r['truncated']/r['n']:.0%} | {r['format_compliant']/r['n']:.0%} |")
    text +=['',f"同题配对：错→对{h['gained']}题，对→错{h['lost']}题，始终正确{h['unchanged_correct']}题；净减2题。准确率差95%问题bootstrap区间[{100*h['accuracy_change']['ci95'][0]:.1f}, {100*h['accuracy_change']['ci95'][1]:.1f}]个百分点，精确McNemar双侧p={h['exact_mcnemar_p']:.6f}。小样本、单seed且区间含0，不支持稳定提升，也不足以断言OPSD损害数学能力。",'',
            f"最终抽取失败{end['extraction_failed']}/50：{end['parse_status']}。没有同步原始回答文本，只能确认解析状态计数，不能逐例确定格式失败的语义原因。截断与抽取失败可能重叠，不能把两者相加当总失败率；格式合规率也不同于抽取成功率。",'',
            '第一次与第二次的题目、回答上限、答案协议及聚合方式不同，不能用旧实验约1%的正确率与本次46%直接计算“OPSD带来的提升”。本次应比较自身step0的50%与step20的46%。','',
            '## 8. 耗时与资源','',
            f"正式受管流程{s['resources']['formal_session_seconds']/3600:.3f}小时；训练预算wall_seconds={s['final_budget']['wall_seconds']:.2f}秒（{s['final_budget']['wall_seconds']/3600:.3f}小时）；5个checkpoint分析计时合计{s['resources']['analysis_seconds_sum']/3600:.3f}小时。各计时覆盖范围不同，不混作完整数据筛选/验收总耗时。训练平均{s['training']['timing_s/step']['mean']:.2f}秒/更新，step开销明显随轨迹长度波动。",'',
            f"训练日志最大allocated={s['training']['perf/max_memory_allocated_gb']['max']:.2f}GiB、reserved={s['training']['perf/max_memory_reserved_gb']['max']:.2f}GiB，CPU memory日志最大{s['training']['perf/cpu_memory_used_gb']['max']:.2f}GiB。allocated包含在reserved里，不能相加；也不是全部并存进程的独立显存总量。此配置已在单A80080GB完成，但显存余量有限。",'',
            '收尾曾有DataLoader worker被Killed的告警；服务器记录主流程退出0、cgroup oom/oom_kill为0，不能据此认定容器OOM，具体来源未确认。global_seqlen异常profiling字段不用于预算结论，critic兼容零值不代表本实验训练critic。','',
            '## 9. 结论边界与后续','',
            '优先做零更新导出/加载测量底噪；对固定heldout继续保留S_s/S_0与S_s/T_s两组；扩展独立且预先固定的评测题、多seed和训练预算后再判断稳定趋势。GRPO尚未运行，不能比较两算法的机制差异。保持原始结果与本地分析分开，不把70%工程质量门槛当成能力验收，也不把统计小差异当成推理机制改变。','',
            '复算：`python -m scripts.research.build_experiment_reading --source results/20261028_qwen17b_opsd20 --output research_runs/20261028_second_analysis`。只读CPU分析，不加载模型、不训练。']
    return '\n'.join(text)+'\n'


def build(source,output,replicates=2000,seed=20261028):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output==source or source in output.parents:raise ValueError('Use a separate derived-analysis directory')
    output.mkdir(parents=True,exist_ok=True)
    summary,data=audit(source,replicates,seed)
    text=markdown(summary)
    write_json(output/'summary.json',summary);write_json(output/'integrity.json',summary['integrity'])
    (output/'ANALYSIS_zh.md').write_text(text)
    from analysis.html_report import plotly_bundle
    payload=clean(dict(summary=summary,**data,markdown=text))
    encoded=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    encoded=encoded.replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template=Path(__file__).with_name('experiment_reading.html').read_text()
    app=Path(__file__).with_name('experiment_reading.js')
    html=template.replace('@@PLOTLY@@',plotly_bundle()).replace('@@DATA@@',encoded).replace('@@APP@@',app.read_text())
    target=output/'OPSD_second_experiment_analysis.html';target.write_text(html)
    write_json(output/'analysis_manifest.json',dict(source=str(source),source_manifest_sha256=sha(source/'artifact_manifest.json'),
               builder_sha256=sha(Path(__file__)),template_sha256=sha(Path(__file__).with_name('experiment_reading.html')),app_sha256=sha(app),
               generated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),bootstrap=summary['bootstrap'],
               files={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='analysis_manifest.json'}))
    print(json.dumps(dict(html=str(target),bytes=target.stat().st_size,checks=summary['integrity']['checks']),ensure_ascii=False))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'results/20261028_qwen17b_opsd20')
    p.add_argument('--output',type=Path,default=ROOT/'research_runs/20261028_second_analysis')
    p.add_argument('--replicates',type=int,default=2000);p.add_argument('--seed',type=int,default=20261028)
    args=p.parse_args();build(args.source,args.output,args.replicates,args.seed)


if __name__=='__main__':main()
