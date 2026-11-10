"""Build a portable, audited browser for every recorded metric through a selected checkpoint.

Only scalar measurements and summaries are exported. Model weights, reference
solutions, question text, token-ID sequences and large hidden-state arrays stay
on the server. Plotly is needed only to build the offline HTML bundle.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
POINT_COLUMNS = ['step', 'layer', 'value', 'n', 'total', 'q05', 'q25', 'median', 'q75', 'q95',
                 'min', 'max', 'std', 'histogram', 'status']
CURVATURE_FIELDS = ['mean_deg', 'valid_angles', 'invalid_fraction', 'mean_step_norm', 'surprisal_bits', 'response_length']
BEHAVIOR_FIELDS = ['correct', 'extraction_failed', 'format_compliant', 'response_length', 'eos', 'truncated', 'repeated_4gram_rate']
NUMERIC_METADATA = {'optimizer_step','training_optimized_tokens','training_rollout_tokens','training_wall_seconds',
                    'training_teacher_forward_tokens','response_position','temperature','student_prediction_index',
                    'teacher_prediction_index','layer','difficulty','topic_label_level','reference_correctness',
                    'reference_token_length','reference_character_length','generation_seed','source_optimizer_step','seed',
                    'geometry_window_clipped','n_valid_tokens','n_questions','response_length'}
TITLES = {
    'actor/forward_kl':'训练前向 KL', 'actor/loss':'训练损失', 'actor/grad_norm':'梯度范数',
    'opd/overlap_ratio':'在线 top-16 词表重叠率', 'opd/overlap_token_advantage':'重叠 token 优势',
    'opd/student_entropy':'在线学生熵', 'opd/teacher_entropy':'在线教师熵',
    'opd/entropy_gap':'在线绝对熵差', 'opd/entropy_gap_signed':'在线有符号熵差',
    'timing_s/step':'每次参数更新耗时', 'timing_s/update_actor':'参数更新阶段耗时',
    'timing_s/gen':'训练轨迹生成耗时', 'perf/max_memory_allocated_gb':'GPU 已分配显存峰值',
    'perf/max_memory_reserved_gb':'GPU 保留显存峰值', 'accuracy_micro':'自主生成正确率（微平均）',
    'accuracy_topic_macro':'按主题宏平均正确率', 'curvature_deg':'轨迹平均转角',
    'curvature_relative_first_deg':'相对第一层曲率变化', 'curvature_training_delta_deg':'相对初始模型曲率变化',
    'curvature_d_min_deg':'最小相对曲率', 'curvature_layer_star':'最小曲率所在层',
    'rsa_rho_to_initial':'与初始表示的 RSA 相关', 'rsa_rmse_to_initial':'与初始 RSM 的均方根差',
    'rsa_rho_student_teacher':'同一步学生与特权教师的 RSA',
    'rsa_rmse_student_teacher':'同一步学生与特权教师的 RSM 均方根差',
    'linear_cka_student_teacher':'同一步学生与特权教师的线性 CKA',
    'linear_cka_to_initial':'与初始表示的线性 CKA', 'participation_ratio':'表示参与率',
    'activation_norm':'表示向量范数', 'top_eigenvalue_fraction':'首特征值占比',
    'rsm_mean':'RSM 平均余弦', 'rsm_std':'RSM 余弦标准差', 'rsm_topic_within':'同主题余弦相似度',
    'rsm_topic_between':'跨主题余弦相似度', 'rsm_topic_gap':'同主题与跨主题相似度差',
    'student_entropy_nats':'学生全词表熵', 'teacher_entropy_nats':'教师全词表熵',
    'initial_student_entropy_nats':'初始学生全词表熵（当前轨迹）',
    'kl_teacher_student_nats':'诊断 KL（教师 || 学生）',
    'overlap_token_advantage_nats':'top-16交集内 token 优势',
    'overlap_token_advantage_to_initial_nats':'相对初始学生的交集内 token 优势',
    'kl_student_teacher_nats':'诊断 KL（学生 || 教师）', 'kl_to_initial_nats':'诊断 KL（当前学生 || 初始学生）',
    'entropy_gap_signed_nats':'教师熵减学生熵', 'entropy_gap_abs_nats':'教师学生绝对熵差',
    'surprisal_bits':'固定文本每 token 惊异度', 'mean_deg':'逐题轨迹平均转角',
    'valid_angles':'有效转角数量', 'invalid_fraction':'无效转角比例', 'mean_step_norm':'隐藏状态相邻步长',
    'response_length':'回答 token 数', 'correct':'逐题最终答案正确标记', 'extraction_failed':'答案抽取失败标记',
    'eos':'正常 EOS 结束标记', 'truncated':'达到长度上限标记', 'repeated_4gram_rate':'重复四元组比例',
    'eos_rate':'EOS 正常结束率', 'truncation_rate':'回答截断率', 'extraction_failure_rate':'答案抽取失败率',
    'invalid_angle_fraction':'无效转角比例',
    'curvature_surprisal_current_pearson':'曲率与当前惊异度的相关',
    'curvature_surprisal_initial_pearson':'当前曲率与初始惊异度的相关',
}


def read_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def finite(value):
    return isinstance(value, (int, float, bool)) and math.isfinite(value)


def check_measurement_fields(row, fields):
    unknown = {key for key,value in row.items() if isinstance(value,(int,float,bool))
               and key not in fields and key not in NUMERIC_METADATA}
    if unknown:
        raise ValueError('Unclassified numeric measurement fields: '+', '.join(sorted(unknown)))


def csv_file(path, fields, rows, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, 'wt', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def info(name, kind):
    title = TITLES.get(name, name)
    if name.startswith('neighbor_overlap_k'):
        title = '题目近邻保持率 k='+name.rsplit('k',1)[1]
    if name.startswith('neighbor_overlap_student_teacher_k'):
        title = '同一步学生与特权教师的近邻重叠率 k='+name.rsplit('_k',1)[1]
    if name.startswith('token_top'):
        title = name.replace('token_top','词表 top-').replace('_shared_mass_student',' 学生共享概率质量').replace(
            '_shared_mass_teacher',' 教师共享概率质量').replace('_overlap',' 重叠率')
    if '_nats' in name or 'entropy' in name or name in ('actor/forward_kl','actor/loss'):
        unit = 'nats'
    elif name.startswith('timing_s/') or name in ('perf/time_per_step','budget/wall_seconds'):
        unit = '秒'
    elif name.startswith('timing_per_token_ms/'):
        unit = '毫秒/token'
    elif name.endswith('_gb'):
        unit = 'GiB（原始日志单位）'
    elif name.endswith('_deg') or name == 'mean_deg':
        unit = '度'
    elif name == 'surprisal_bits':
        unit = 'bits/token'
    else:
        unit = '原始单位'
    notes = []
    if name in ('actor/forward_kl','actor/loss'):
        notes.append('训练目标为全词表 KL(stopgrad(特权教师) || 学生)，按有效 response token 平均。')
    if name == 'kl_student_teacher_nats':
        notes.append('此处是诊断方向 KL(学生 || 特权教师)，与训练损失方向相反。')
    if name.startswith('global_seqlen/'):
        notes.append('原日志出现远高于 token 预算的数值，疑似统计异常；原样保留，不据此推断真实序列长度。')
    if name.startswith('critic/') and not name.startswith('critic/true_reward/'):
        notes.append('本次 OPSD 未启用 critic；这些兼容字段多为占位零，不代表 critic 学习效果。')
    if name.startswith('critic/true_reward/'):
        notes.append('这是规则校验器的 outcome reward 统计，不是独立 critic 网络的输出。')
    if name == 'perf/cpu_memory_used_gb':
        notes.append('原始可见内存统计，不能单独解释为训练进程的 RSS 或容器内存额度。')
    if name == 'participation_ratio':
        notes.append('方差谱参与率，属于表示统计；不是 Gride 内在维度。')
    if kind == 'analysis':
        notes.append('仅使用已完成的配置监测点；教师几何有单独采集计划。缺失处不插值。')
    if name.endswith('_student_teacher') or name.startswith('neighbor_overlap_student_teacher_k'):
        notes.append('相同checkpoint、固定题目/回答、相同层/相对anchor，联合剔除无效向量；这是直接条件比较。')
    if kind=='distribution':
        notes.append('逐题分布描述题间差异；主token加权曲线请查看监测汇总。fixed/live按pool分开。')
    if kind in ('distribution','curvature','behavior'):
        notes.append('图中分位数描述逐题差异，不是置信区间。无有效值时保留空值，不补零。')
    if name in ('curvature_training_delta_deg','curvature_surprisal_initial_pearson'):
        notes.append('live 文本跨 checkpoint 会改变，部分跨步配对量按设计未定义；请查看缺失值和 n。')
    return title, unit, notes


def category(name, kind):
    if kind != 'training':
        return dict(analysis='监测汇总',distribution='逐题概率分布',curvature='逐题曲率',behavior='生成行为')[kind]
    if name.startswith(('opd/','opsd/','teacher/')):
        return '训练·蒸馏诊断'
    if name.startswith(('timing_','perf/','research/')):
        return '训练·性能资源'
    if name.startswith(('critic/',)):
        return '训练·奖励兼容字段'
    if name.startswith(('prompt_','response','global_seqlen/')):
        return '训练·长度统计'
    return '训练·优化与预算'


def metric_entry(name, kind, variants):
    title, unit, notes = info(name, kind)
    return dict(id=kind+'::'+name, name=name, title=title, kind=kind, category=category(name,kind),
                unit=unit, notes=notes, variants=variants)


def variant_label(dims):
    labels = {'fixed_base_rollout':'固定初始轨迹','live_rollout':'当前自主生成','heldout_probe':'独立监测集',
              'train_probe':'训练内监测集','student':'学生','privileged_teacher':'特权教师',
              'student_vs_privileged_teacher':'当前学生与当前特权教师',
              'all_question_mean':'题目等权均值（非主曲线 token 均值）',
              'all_valid_tokens':'全部有效回答 token（逐题值）'}
    return ' / '.join(labels.get(str(v), str(v)) for v in dims.values() if v is not None)


def default_variant(entry):
    def score(variant):
        d = variant['dims']
        return sum((d.get('pool')=='fixed_base_rollout', d.get('split')=='heldout_probe',
                    d.get('role')=='student',d.get('anchor')=='response_128',
                    d.get('position')=='all_question_mean'))
    entry['default_variant'] = max(range(len(entry['variants'])), key=lambda i:score(entry['variants'][i]))
    entry['finite_points'] = sum(finite(p[2]) for v in entry['variants'] for p in v['points'])
    entry['missing_points'] = sum(not finite(p[2]) for v in entry['variants'] for p in v['points'])
    return entry


def scalar_point(step, layer, value, n, total, status):
    return [step, layer, float(value) if finite(value) else None, n, total]+[None]*9+[status]


def sample_entries(groups, kind):
    """Summarize raw samples by the original experimental context, never across layers/roles."""
    result = []
    for name, contexts in sorted(groups.items()):
        variants = []
        for context, records in sorted(contexts.items(), key=lambda item:str(item[0])):
            dims = dict(context)
            values = [value for group in records.values() for value in group if finite(value)]
            edges = np.histogram_bin_edges(values, bins=24) if values else np.linspace(0,1,25)
            points = []
            for (step, layer), raw in sorted(records.items(), key=lambda item:(item[0][0], item[0][1] or 0)):
                valid = np.asarray([v for v in raw if finite(v)], dtype=float)
                if len(valid):
                    q05,q25,median,q75,q95 = np.quantile(valid,[.05,.25,.5,.75,.95]).tolist()
                    point = [step,layer,float(valid.mean()),len(valid),len(raw),q05,q25,median,q75,q95,
                             float(valid.min()),float(valid.max()),float(valid.std()),
                             np.histogram(valid,edges)[0].tolist(),'ok']
                else:
                    point = [step,layer,None,0,len(raw)]+[None]*8+[[0]*24,'undefined']
                points.append(point)
            variants.append(dict(dims=dims,label=variant_label(dims),points=points,histogram_edges=edges.tolist()))
        result.append(metric_entry(name,kind,variants))
    return result


def static_plot(entry, output, steps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    v = entry['variants'][entry['default_variant']]
    layers = [p[1] for p in v['points'] if p[1] is not None]
    selected = [p for p in v['points'] if p[1]==max(layers)] if layers else v['points']
    lookup = {p[0]:p for p in selected}
    x = list(range(1,steps+1)) if entry['kind']=='training' else sorted({p[0] for w in entry['variants'] for p in w['points']})
    y = [lookup[s][2] if s in lookup and lookup[s][2] is not None else np.nan for s in x]
    fig, ax = plt.subplots(figsize=(8.5,4.8))
    ax.plot(x,y,'.-',color='#2563a6',linewidth=1.5)
    if not any(np.isfinite(y)):
        ax.text(.5,.5,'Undefined / not measured in this context',ha='center',va='center',transform=ax.transAxes)
    ax.set(title=entry['name'],xlabel='Optimizer step',ylabel=entry['name'])
    ax.grid(alpha=.2)
    if entry['kind']=='training':ax.axvline(5.5,color='#777',ls='--',alpha=.4)
    if layers:ax.text(.01,-.23,f'layer={max(layers)}; change context/layer in the interactive report',transform=ax.transAxes,fontsize=8)
    fig.tight_layout()
    fig.savefig(output,dpi=130)
    plt.close(fig)


def build(run, output, max_step=20, static=True):
    from plotly.offline import get_plotlyjs
    import plotly
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or run in output.parents:
        raise ValueError('Use a separate report directory to preserve the source experiment')
    output.mkdir(parents=True,exist_ok=True)
    (output/'data').mkdir(exist_ok=True)
    (output/'charts').mkdir(exist_ok=True)
    sources, entries = {}, []

    def source(path):
        sources[str(path.relative_to(run))] = dict(sha256=sha(path),bytes=path.stat().st_size)

    training_path = run/'final_step20/metrics.jsonl'
    if not training_path.exists():training_path=run/'opsd/train/metrics.jsonl'
    source(training_path)
    training = [r for r in read_rows(training_path) if 1<=r['step']<=max_step]
    if [r['step'] for r in training] != list(range(1,max_step+1)):
        raise ValueError('Selected training steps must be complete and unique')
    names = sorted({key for r in training for key in r['metrics']})
    for key in names:
        points = [scalar_point(r['step'],None,r['metrics'].get(key),int(finite(r['metrics'].get(key))),1,
                              'ok' if finite(r['metrics'].get(key)) else 'missing') for r in training]
        entries.append(metric_entry(key,'training',[dict(dims={},label='训练日志',points=points)]))
    for row in training:
        for key,value in row.get('budget',{}).items():
            if 'budget/'+key not in row['metrics'] or row['metrics']['budget/'+key]!=value:
                raise ValueError('Unregistered or conflicting budget metric: '+key)
    csv_file(output/'data/training_metrics.csv',['step',*names],
             ({'step':r['step'],**r['metrics']} for r in training))

    aggregate = defaultdict(lambda:defaultdict(list))
    monitor_rows, completed = [], []
    distributions = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    curvature = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    behavior = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    source_counts, distribution_fields = defaultdict(int), set()
    exports = {}
    export_fields = {
        'distributions':['step','pool','split','distribution_scope','aggregation','n_valid_tokens',
                         'response_position','problem_id'],
        'curvature':['step','split','pool','role','layer','problem_id',*CURVATURE_FIELDS],
        'behavior':['step','split','problem_id',*BEHAVIOR_FIELDS],
    }
    handles = []
    try:
        for directory in sorted((run/'opsd/analysis/steps').glob('step_*')):
            step = int(directory.name.split('_')[1])
            if step>max_step or not (directory/'complete.json').exists():continue
            completed.append(step)
            source(directory/'complete.json')
            for row in read_rows(directory/'metrics.jsonl'):
                if row['optimizer_step']!=step:raise ValueError('Wrong aggregate step')
                context = tuple((k,row.get(k)) for k in ('pool','split','role','anchor'))
                aggregate[row['metric']][context].append(scalar_point(step,row.get('layer'),row['value'],
                                                                      row['n_valid'],row['n_valid'],row['status']))
                monitor_rows.append({**row,'step':step})
            source(directory/'metrics.jsonl')
            for source_name, filename, fields, target in [
                ('distributions','distributions.jsonl',None,distributions),
                ('curvature','curvature.jsonl',CURVATURE_FIELDS,curvature),
                ('behavior','live_rollouts.jsonl',BEHAVIOR_FIELDS,behavior),
            ]:
                path=directory/filename
                source(path)
                pooled=defaultdict(lambda:defaultdict(list))
                for row in read_rows(path):
                    source_counts[source_name]+=1
                    if fields is None:
                        fields=[k for k in row if k.endswith('_nats') or k.startswith('token_top')]
                        distribution_fields.update(fields)
                    check_measurement_fields(row, fields)
                    if source_name not in exports:
                        if source_name=='distributions':export_fields[source_name].extend(fields)
                        handle=gzip.open(output/'data'/f'{source_name}_samples.csv.gz','wt',encoding='utf-8-sig',newline='')
                        handles.append(handle)
                        writer=csv.DictWriter(handle,fieldnames=export_fields[source_name],extrasaction='ignore',
                                              lineterminator='\n')
                        writer.writeheader();exports[source_name]=writer
                    exports[source_name].writerow({'step':step,**row})
                    for key in fields:
                        value=row.get(key)
                        if source_name=='distributions':
                            context=tuple(dict(pool=row.get('pool','fixed_base_rollout'),split=row['split'],
                                               position=row['response_position']).items())
                            target[key][context][(step,None)].append(value)
                            pooled[(key,row.get('pool','fixed_base_rollout'),row['split'])][row['problem_id']].append(value)
                        elif source_name=='curvature':
                            context=tuple((k,row[k]) for k in ('pool','split','role'))
                            target[key][context][(step,row['layer'])].append(value)
                        else:
                            context=(('split',row['split']),)
                            target[key][context][(step,None)].append(float(value) if isinstance(value,bool) else value)
                if source_name=='distributions':
                    for (key,pool,split), problems in pooled.items():
                        context=(('pool',pool),('split',split),('position','all_question_mean'))
                        for values in problems.values():
                            valid=[v for v in values if finite(v)]
                            target[key][context][(step,None)].append(float(np.mean(valid)) if valid else None)
    finally:
        for handle in handles:handle.close()
    if max_step not in completed:raise ValueError('Selected final checkpoint analysis is not complete')
    for name, contexts in sorted(aggregate.items()):
        entries.append(metric_entry(name,'analysis',[dict(dims=dict(context),label=variant_label(dict(context)),points=points)
                                                    for context,points in sorted(contexts.items(),key=lambda item:str(item[0]))]))
    for kind, groups in [('distribution',distributions),('curvature',curvature),('behavior',behavior)]:
        entries.extend(sample_entries(groups,kind))
    for entry in entries:default_variant(entry)
    expected={'training':set(names),'analysis':set(aggregate),'distribution':distribution_fields,
              'curvature':set(CURVATURE_FIELDS),'behavior':set(BEHAVIOR_FIELDS)}
    for kind, keys in expected.items():
        if {e['name'] for e in entries if e['kind']==kind}!=keys:
            raise ValueError('Metric inventory coverage failed: '+kind)
    csv_file(output/'data/monitor_metrics.csv',
             ['metric','step','pool','split','role','anchor','layer','value','n_valid','status','n_pairs','rsa_pair_seed'],monitor_rows)
    counts={kind:len(keys) for kind,keys in expected.items()}
    catalogue=[]
    for i, entry in enumerate(entries):
        entry['png']=f'charts/{i:03d}.png'
        if static:static_plot(entry,output/entry['png'],max_step)
        catalogue.append(dict(id=entry['id'],source=entry['kind'],metric=entry['name'],title=entry['title'],
                              category=entry['category'],unit=entry['unit'],contexts=len(entry['variants']),
                              finite_points=entry['finite_points'],missing_points=entry['missing_points'],png=entry['png']))
        if i%30==0:print('REPORT_METRICS',i+1,'/',len(entries),flush=True)
    csv_file(output/'metric_catalog.csv',list(catalogue[0]),catalogue)
    gallery=[]
    old_manifest=json.loads((run/'report/report_manifest.json').read_text())
    for name in old_manifest['plots']:
        if name.startswith('rsm__'):
            destination=output/'rsm'/name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(run/'report'/name,destination)
            gallery.append(dict(name=name,path='rsm/'+name))
    overview=run/'final_step20/training_overview.png'
    shutil.copy2(overview,output/'training_overview.png')
    payload=dict(meta=dict(model='Qwen3-1.7B-Base',selected_step=max_step,training_steps=list(range(1,max_step+1)),
                 monitor_steps=completed,counts=counts,total_metrics=len(entries),source_rows=dict(source_counts),
                 point_columns=POINT_COLUMNS,generated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                 scope='第21步原始日志保留在服务器，本报告不读取或展示第21步指标。',plotly_version=plotly.__version__),
                 metrics=entries,gallery=gallery)
    encoded=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    (output/'data/report_data.json').write_text(encoded)
    manifest=dict(schema=1,selected_step=max_step,monitor_steps=completed,counts=counts,total_metrics=len(entries),
                  coverage_complete=True,source_rows=dict(source_counts),sources=sources,
                  raw_fields=expected,all_pngs=static,rsm_gallery_images=len(gallery),
                  metadata_exclusions='step/layer/position are selectors; n_valid/status are support fields; input text/IDs are not metrics')
    manifest['raw_fields']={kind:sorted(keys) for kind,keys in expected.items()}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    from analysis.html_report import write_html
    payload['meta']['mode']='archived_export'
    payload['meta']['notices']=['此报告按实际归档结果展示；未采集的新指标保持缺失。']
    write_html(payload,output/'index.html',output,manifest)
    (output/'README.md').write_text(f'''# 第 {max_step} 步全指标报告

在浏览器中打开 `index.html`。主要交互图与数据已内嵌，可单独下载该 HTML 离线查看；
静态 PNG、RSM 图库和数据下载随同本目录提供。

- 训练日志指标：{counts['training']} 个；监测汇总指标：{counts['analysis']} 个。
- 逐题概率分布：{counts['distribution']} 个；逐题曲率：{counts['curvature']} 个；生成行为：{counts['behavior']} 个。
- 共 {len(entries)} 个按来源区分的指标入口，别名和兼容占位字段均保留。
- 训练范围为 1–{max_step}；分析点为 {completed}。不包含第 21 步。
- 可搜索、逐项切换、选择上下文/层/监测点，导出当前图的 PNG/SVG 和指标 CSV。
- 单个静态 PNG 展示默认上下文；全部上下文请在交互报告中选择。
- 缺失值不补零、不连线插值。分位数描述个体差异，不是置信区间。
- 教师几何在当前结果中仅有 0/20；未计算的教师惊异度等保留 undefined。
- RSM 图库保留原先预选的层与 0/20 时点；所有层的 RSM 统计量在指标列表中。

`metric_catalog.csv` 是完整指标清单；`data/` 包含训练、监测汇总和逐题测量导出。
逐题导出仅含测量值及关联 ID，不含题目文本、参考解答、模型权重或隐藏状态大数组。
源文件 SHA256 和覆盖范围见 `manifest.json`。
''')
    print('REPORT_COMPLETE',output,'METRICS',len(entries),'COUNTS',counts,flush=True)
    return payload,manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--max-step',type=int,default=20)
    parser.add_argument('--no-static',action='store_true')
    args=parser.parse_args()
    build(args.run,args.output,args.max_step,not args.no_static)


if __name__=='__main__':main()
