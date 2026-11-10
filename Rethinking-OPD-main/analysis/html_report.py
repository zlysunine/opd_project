"""Portable HTML reports for current checkpoint outputs and archived report exports.

No model is loaded. Token-weighted aggregates stay separate from per-question
summaries; missing newly introduced measurements are never reconstructed from
old selected-position statistics.
"""
import argparse
import base64
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from html.parser import HTMLParser
import json
from pathlib import Path
import re

from scripts.research.build_metric_report import (
    BEHAVIOR_FIELDS, CURVATURE_FIELDS, NUMERIC_METADATA, POINT_COLUMNS,
    default_variant, finite, info, metric_entry, sample_entries, scalar_point, sha, variant_label,
)
from scripts.research.common import ROOT


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside=False
        self.current=[]
        self.scripts=[]

    def handle_starttag(self,tag,attrs):
        if tag=='script':
            self.inside=True
            self.current=[]

    def handle_data(self,data):
        if self.inside:
            self.current.append(data)

    def handle_endtag(self,tag):
        if tag=='script' and self.inside:
            self.scripts.append(''.join(self.current))
            self.inside=False


def plotly_bundle():
    try:
        from plotly.offline import get_plotlyjs
        return get_plotlyjs()
    except ImportError:
        # The checked-in historical report already contains a standalone bundle.
        reader=Scripts()
        reader.feed((ROOT/'results/step20_all_metrics/index.html').read_text())
        libraries=[s for s in reader.scripts if 'plotly.js v' in s[:300]]
        if len(libraries)!=1:
            raise RuntimeError('Install plotly or provide the repository bundled report library')
        return libraries[0]


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def context(row,method=None):
    dims={k:row.get(k) for k in ('run_id','method','pool','split','role','anchor',
                                'aggregation','distribution_scope','comparison') if row.get(k) is not None}
    if method is not None:
        dims.setdefault('method',method)
    name=row.get('metric','')
    if ('entropy' in name or name.startswith(('token_top','kl_','overlap_token_advantage'))):
        dims.setdefault('distribution_scope','legacy_selected_positions')
        dims.setdefault('aggregation','legacy_question_mean')
    return tuple(sorted(dims.items()))


def read_run(run,max_step=None):
    """Read only complete checkpoint analyses; retain every numeric metric field."""
    run=Path(run).resolve()
    candidates=[]
    for path in sorted(run.rglob('metrics.jsonl')):
        if not (path.parent/'complete.json').is_file() or not (path.parent/'run_manifest.json').is_file():
            continue
        manifest=json.loads((path.parent/'run_manifest.json').read_text())
        complete=json.loads((path.parent/'complete.json').read_text())
        manifest['identity'].setdefault('probe_manifest_hash',complete['identity']['probe_manifest_hash'])
        step=manifest['identity']['step']
        if max_step is None or step<=max_step:
            candidates.append((path,manifest,step))
    if not candidates:
        raise ValueError('No completed checkpoint analyses; provide a run directory or archived report directory')
    hashes={m['identity']['probe_manifest_hash'] for p,m,s in candidates}
    if len(hashes)!=1:
        raise ValueError('A report must use exactly one frozen probe manifest; choose one experiment directory')
    sources={}
    aggregate=defaultdict(lambda:defaultdict(list))
    sample_groups={kind:defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
                   for kind in ('distribution','curvature','behavior')}
    counts=defaultdict(int)
    configs=[]
    notices=[]

    def register(path,expected=None):
        digest=sha(path)
        if expected is not None and digest!=expected:
            raise ValueError(f'Completed analysis file changed: {path}')
        sources[str(path.relative_to(run))]=dict(sha256=digest,bytes=path.stat().st_size)

    for path,manifest,step in candidates:
        directory=path.parent
        complete=json.loads((directory/'complete.json').read_text())
        expected=complete.get('files',{})
        configs.append(manifest['config'])
        method=manifest['config'].get('method')
        register(directory/'complete.json')
        register(directory/'run_manifest.json',expected.get('run_manifest.json'))
        register(path,expected.get('metrics.jsonl'))
        for row in rows(path):
            if row['optimizer_step']!=step:
                raise ValueError('Metric step does not match its checkpoint manifest')
            aggregate[row['metric']][context(row,method)].append(
                scalar_point(step,row.get('layer'),row.get('value'),row.get('n_valid',0),
                             row.get('cohort_n',row.get('n_valid',0)),row.get('status','ok')))
            counts['analysis']+=1
        for kind,filename,known in [('distribution','distributions.jsonl',()),
                                   ('curvature','curvature.jsonl',CURVATURE_FIELDS),
                                   ('behavior','live_rollouts.jsonl',BEHAVIOR_FIELDS)]:
            sample=directory/filename
            if not sample.is_file():
                continue
            register(sample,expected.get(filename))
            for row in rows(sample):
                counts[kind]+=1
                fields=[k for k,v in row.items() if (finite(v) or v is None)
                        and (k in known or k not in NUMERIC_METADATA)
                        and k not in ('response_position','optimizer_step','step','n_valid','cohort_n','n_pairs')
                        and not k.endswith(('_seed','_index'))]
                # Strings/lists/dicts are context or raw data, not scalar measurements.
                fields=[k for k in fields if finite(row[k]) or (row[k] is None and
                         (k in known or k.endswith('_nats') or k.startswith('token_top')))]
                dims=dict(context(row,method))
                if kind=='distribution':
                    dims['position']=row.get('response_position','unspecified')
                    dims['distribution_scope']=row.get('distribution_scope','legacy_selected_positions')
                    dims['aggregation']='question_distribution_summary'
                for name in fields:
                    sample_groups[kind][name][tuple(sorted(dims.items()))][(step,row.get('layer'))].append(row[name])
    entries=[]
    for name,contexts in sorted(aggregate.items()):
        variants=[dict(dims=dict(dims),label=variant_label(dict(dims)),points=points)
                  for dims,points in sorted(contexts.items(),key=lambda x:str(x[0]))]
        entries.append(metric_entry(name,'analysis',variants))
    for kind,groups in sample_groups.items():
        entries.extend(sample_entries(groups,kind))
    selected=max(step for p,m,step in candidates)
    training_groups=defaultdict(lambda:defaultdict(dict))
    train_paths=sorted(set(run.glob('*/train/metrics.jsonl'))|set(run.glob('train/metrics.jsonl')))
    if not train_paths and (run/'final_step20/metrics.jsonl').is_file():
        train_paths=[run/'final_step20/metrics.jsonl']
    for path in train_paths:
        register(path)
        method=path.parent.parent.name if path.parent.name=='train' else 'opsd'
        for row in rows(path):
            if not 1<=row['step']<=selected:
                continue
            counts['training']+=1
            for name,value in row.get('metrics',{}).items():
                if not (finite(value) or value is None):
                    continue
                previous=training_groups[name][method].get(row['step'])
                point=scalar_point(row['step'],None,value,int(finite(value)),1,'ok' if finite(value) else 'undefined')
                if previous is not None and previous!=point:
                    raise ValueError('Conflicting duplicate training metric/step; inspect resumed logs')
                training_groups[name][method][row['step']]=point
    for name,methods in sorted(training_groups.items()):
        entries.append(metric_entry(name,'training',[dict(dims={'method':method},label=method,
                            points=[points[s] for s in sorted(points)]) for method,points in sorted(methods.items())]))
    if not training_groups:
        notices.append('此目录未包含训练标量日志；训练曲线保持未采集，不用checkpoint测量代替。')
    gallery=[]
    for path in sorted((run/'report').glob('rsm__*.png')):
        gallery.append(dict(name=path.name,path=str(path.relative_to(run))))
    # Expose immutable settings from the archived manifests, not today's defaults.
    config=configs[-1]
    model=config.get('model',{}).get('name_or_path','未记录模型名称')
    fixture=('TINY' in model.upper() or config.get('training',{}).get('status')=='NOT_RUN')
    meta=dict(model=model,selected_step=selected,monitor_steps=sorted({s for p,m,s in candidates}),
              training_steps=sorted({p[0] for e in entries if e['kind']=='training' for v in e['variants'] for p in v['points']}),
              source_rows=dict(counts),point_columns=POINT_COLUMNS,scope='仅展示该目录已完成checkpoint和已保存的测量。',
              mode='cpu_fixture' if fixture else 'experiment',settings=config.get('training',{}),
              requested_step=max_step,notices=notices,probe_manifest_hash=next(iter(hashes)))
    return dict(meta=meta,metrics=entries,gallery=gallery),dict(schema=2,sources=sources),run


def is_direct(name):
    return name.endswith('_student_teacher') or name.startswith('neighbor_overlap_student_teacher_k')


def enrich(payload):
    payload=deepcopy(payload)
    meta=payload['meta']
    meta.setdefault('notices',[])
    meta.setdefault('mode','archived_export')
    meta.setdefault('point_columns',POINT_COLUMNS)
    for entry in payload['metrics']:
        name,kind=entry['name'],entry['kind']
        title,unit,notes=info(name,kind)
        if kind=='training' and name=='actor/loss':
            title='训练损失（按方法定义）'
            unit='按方法定义'
            notes=['SFT：response-only 交叉熵；GRPO：策略优化损失；OPSD：全词表 KL(stopgrad(教师) || 学生)。'
                   '请按上下文中的 method 区分，损失绝对值不能跨方法直接比较。']
        entry.update(title=title,unit=unit)
        # Show k as recorded. A generic OTA alias need not mean k=16 in custom configs.
        if name=='overlap_token_advantage_nats':
            entry['title']='交集内 token 优势（配置指定k）'
        match=re.fullmatch(r'token_top(\d+)_(.+?)(_to_initial)?',name)
        if match:
            suffix={'overlap':'重叠率','shared_mass_student':'学生共享概率质量','shared_mass_teacher':'对照共享概率质量',
                    'empty_intersection_rate':'空交集率','overlap_token_advantage_nats':'交集内 token 优势'}.get(match[2],match[2])
            entry['title']=f'top-{match[1]} {suffix}'+('（对初始学生）' if match[3] else '')
        entry['notes']=notes
        if is_direct(name):
            entry['category']='表征 · 当前学生与当前教师'
        elif ('to_initial' in name and not ('nats' in name or name.startswith('token_top'))) or name.startswith('neighbor_overlap_k'):
            entry['category']='表征 · 相对初始参数'
        elif name.startswith(('rsm_','participation_','activation_','top_eigen')):
            entry['category']='表征 · 关系与方差结构'
        elif 'curvature' in name or kind=='curvature':
            entry['category']='表征 · 轨迹曲率'
        elif kind=='analysis' and ('entropy' in name or 'nats' in name or name.startswith('token_top')):
            entry['category']='分布 · checkpoint主曲线'
        for variant in entry['variants']:
            dims=variant['dims']
            if kind=='distribution':
                dims.setdefault('distribution_scope','all_valid_tokens' if dims.get('position')=='all_valid_tokens'
                                else 'legacy_selected_positions')
                dims['aggregation']='question_distribution_summary'
            elif kind=='analysis' and ('entropy' in name or 'nats' in name or name.startswith('token_top')):
                dims.setdefault('distribution_scope','legacy_selected_positions')
                dims.setdefault('aggregation','legacy_question_mean')
            if kind=='analysis' and ('rsa_' in name or 'cka' in name or name.startswith('neighbor_overlap')):
                if is_direct(name):
                    dims['comparison']='same_checkpoint_student_vs_privileged_teacher'
                else:
                    dims['comparison']='teacher_vs_initial_teacher' if dims.get('role')=='privileged_teacher' else 'student_vs_initial_student'
            variant['label']=label(dims) or variant.get('label','训练日志')
        default_variant(entry)
    meta['counts']={kind:sum(e['kind']==kind for e in payload['metrics'])
                    for kind in ('training','analysis','distribution','curvature','behavior')}
    meta['total_metrics']=len(payload['metrics'])
    meta['generated']=datetime.now().astimezone().isoformat(timespec='seconds')
    meta['coverage']=coverage(payload)
    meta['metric_plan']=metric_plan(payload)
    return payload


def metric_plan(payload):
    """Separate the experimental recording plan from availability in this result."""
    def cell(description,fields,role='student',full=False,status='implemented'):
        fields=[fields] if isinstance(fields,str) else fields
        roles=role if isinstance(role,list) else [role]
        available=all(any(e['kind']=='analysis' and e['name']==field and any(
            v['dims'].get('role')==r
            and (not full or (v['dims'].get('distribution_scope')=='all_valid_tokens'
                             and v['dims'].get('aggregation')=='token_mean'))
            and any(finite(p[2]) for p in v['points']) for v in e['variants'])
            for e in payload['metrics']) for field in fields for r in roles)
        return dict(description=description,fields=fields,roles=roles,status=status,available=available)
    specs=[
        ('top-16 overlap ratio','词表top-16集合的交集比例',
         cell('当前学生与初始学生','token_top16_overlap_to_initial',full=True),
         cell('当前学生与当前特权教师','token_top16_overlap',full=True)),
        ('Overlap-Token Advantage','交集内归一化；空交集不参与该均值',
         cell('S_s 对 S_0','overlap_token_advantage_to_initial_nats',full=True),
         cell('S_s 对 T_s','overlap_token_advantage_nats',full=True)),
        ('Entropy','分别记录全词表熵，而非把熵当成两模型相似度',
         cell('H(S_s)、H(S_0)',['student_entropy_nats','initial_student_entropy_nats'],full=True),
         cell('H(S_s)、H(T_s)',['student_entropy_nats','teacher_entropy_nats'],full=True)),
        ('Entropy Gap（绝对）','逐位置取熵差绝对值后平均',
         cell('|H(S_0) − H(S_s)|','entropy_gap_to_initial_abs_nats',full=True),
         cell('|H(T_s) − H(S_s)|','entropy_gap_abs_nats',full=True)),
        ('Entropy Gap（有符号）','对照熵减当前学生熵',
         cell('H(S_0) − H(S_s)','entropy_gap_to_initial_signed_nats',full=True),
         cell('H(T_s) − H(S_s)','entropy_gap_signed_nats',full=True)),
        ('KL散度','完整词表；显式保留方向，训练loss另行记录',
         cell('KL(S_s || S_0)','kl_to_initial_nats',full=True),
         cell('KL(S_s || T_s) 及 KL(T_s || S_s)',
              ['kl_student_teacher_nats','kl_teacher_student_nats'],full=True)),
        ('top-16共享概率质量','交集上的双方原始概率质量，不重新归一化',
         cell('S_s、S_0的共享质量',
              ['token_top16_shared_mass_student_to_initial','token_top16_shared_mass_teacher_to_initial'],full=True),
         cell('S_s、T_s的共享质量',
              ['token_top16_shared_mass_student','token_top16_shared_mass_teacher'],full=True)),
        ('top-16空交集率','解释OTA有效位置数量',
         cell('S_s 与 S_0','token_top16_empty_intersection_rate_to_initial',full=True),
         cell('S_s 与 T_s','token_top16_empty_intersection_rate',full=True)),
        ('RSA（Spearman）','固定题集上两个题间余弦矩阵的秩相关',
         cell('S_s 与 S_0','rsa_rho_to_initial'),
         cell('S_s 与 T_s','rsa_rho_student_teacher',role='student_vs_privileged_teacher')),
        ('RSM RMSE','两个题间相似度矩阵的均方根差',
         cell('S_s 与 S_0','rsa_rmse_to_initial'),
         cell('S_s 与 T_s','rsa_rmse_student_teacher',role='student_vs_privileged_teacher')),
        ('Linear CKA','中心化表征的关系结构比较',
         cell('S_s 与 S_0','linear_cka_to_initial'),
         cell('S_s 与 T_s','linear_cka_student_teacher',role='student_vs_privileged_teacher')),
        ('表征kNN overlap（k=8/16/32）','题目近邻集合重叠；与词表top-16不同',
         cell('S_s 与 S_0',[f'neighbor_overlap_k{k}' for k in (8,16,32)]),
         cell('S_s 与 T_s',[f'neighbor_overlap_student_teacher_k{k}' for k in (8,16,32)],
              role='student_vs_privileged_teacher')),
        ('轨迹曲率','逐层连续隐藏状态的转角',
         cell('配对变化 C(S_s) − C(S_0)','curvature_training_delta_deg'),
         cell('分别记录 C(S_s)、C(T_s)；尚无独立的配对曲率差字段','curvature_deg',
              role=['student','privileged_teacher'],status='separate_curves')),
    ]
    return [dict(metric=name,note=note,student_initial=a,student_teacher=b) for name,note,a,b in specs]


def label(dims):
    names={'student':'学生','privileged_teacher':'特权教师','student_vs_privileged_teacher':'学生↔当前特权教师',
           'fixed_base_rollout':'固定初始回答','live_rollout':'当前生成回答','fixed_reference':'固定参考回答',
           'heldout_probe':'heldout监测集','train_probe':'训练内监测集','all_valid_tokens':'全部有效回答token',
           'legacy_selected_positions':'历史选点统计','token_mean':'按有效token加权',
           'legacy_question_mean':'历史题目等权','question_distribution_summary':'逐题摘要（非token加权主曲线）',
           'all_question_mean':'题目等权摘要','student_vs_initial_student':'当前学生↔初始学生',
           'teacher_vs_initial_teacher':'当前教师↔初始教师',
           'same_checkpoint_student_vs_privileged_teacher':'同一checkpoint学生↔教师'}
    return ' / '.join(names.get(str(dims[k]),str(dims[k])) for k in
                    ('method','run_id','pool','split','role','comparison','anchor','position','distribution_scope','aggregation')
                    if dims.get(k) is not None)


def coverage(payload):
    entries=payload['metrics']
    def measurements(name,role=None,full=False):
        return [p for e in entries if e['kind']=='analysis' and e['name']==name for v in e['variants'] if
            (role is None or v['dims'].get('role')==role)
            and (not full or (v['dims'].get('distribution_scope')=='all_valid_tokens'
                             and v['dims'].get('aggregation')=='token_mean'))
            for p in v['points'] if finite(p[2])]
    requirements=[('当前学生 ↔ 当前特权教师 RSA','rsa_rho_student_teacher','student_vs_privileged_teacher',False),
                  ('当前学生 ↔ 当前特权教师 CKA','linear_cka_student_teacher','student_vs_privileged_teacher',False),
                  ('当前学生 ↔ 初始学生 RSA','rsa_rho_to_initial','student',False),
                  ('当前教师 ↔ 初始教师 RSA','rsa_rho_to_initial','privileged_teacher',False),
                  ('离线全回答全词表学生熵','student_entropy_nats',None,True),
                  ('离线全回答全词表教师熵','teacher_entropy_nats',None,True),
                  ('离线全回答交集内 OTA','overlap_token_advantage_nats',None,True)]
    result=[]
    for title,name,role,full in requirements:
        values=measurements(name,role,full)
        result.append(dict(label=title,metric=name,available=bool(values),recorded_steps=sorted({p[0] for p in values})))
    return result


def image_data(root,path):
    target=(root/path).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        return None
    media={'.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg'}.get(target.suffix.lower())
    if media is None:
        return None
    return f'data:{media};base64,'+base64.b64encode(target.read_bytes()).decode()


def write_html(payload,target,asset_root,provenance=None):
    """Embed JS, data and every referenced image; the HTML needs no companion files."""
    target=Path(target)
    asset_root=Path(asset_root)
    payload=enrich(payload)
    images={}
    def embed(path):
        if path not in images:
            images[path]=image_data(asset_root,path)
        return images[path]
    for entry in payload['metrics']:
        entry['png']=embed(entry['png']) if entry.get('png') else None
    payload['gallery']=[dict(name=g['name'],path=embed(g['path'])) for g in payload.get('gallery',[]) if embed(g['path'])]
    overview=embed('training_overview.png')
    payload['overview_image']=overview
    payload['provenance']=provenance or {}
    payload['meta']['standalone']=True
    encoded=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    encoded=encoded.replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template=(ROOT/'analysis/report_template.html').read_text()
    document=template.replace('/*__PLOTLY__*/',plotly_bundle()).replace('/*__REPORT_DATA__*/','window.REPORT_DATA='+encoded+';')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(document)
    return payload


def build(run,output=None,max_step=None):
    run=Path(run).resolve()
    if (run/'data/report_data.json').is_file():
        payload=json.loads((run/'data/report_data.json').read_text())
        if max_step is not None:
            raise ValueError('Archived exports retain their original step scope; select a raw run for --max-step')
        manifest=run/'manifest.json'
        provenance=dict(archived_manifest=json.loads(manifest.read_text()) if manifest.is_file() else {},
                        sources={'data/report_data.json':dict(sha256=sha(run/'data/report_data.json'))})
        payload['meta']['notices']=['历史结果仅反映其原始采集协议；新加入的指标缺少实测值时标记未采集。']
        asset_root=run
    else:
        payload,provenance,asset_root=read_run(run,max_step)
    output=Path(output) if output else run/'report/OPSD_report.html'
    if output.exists() and output.resolve()==(run/'index.html').resolve():
        raise ValueError('Choose a new HTML filename to preserve the archived source report')
    result=write_html(payload,output,asset_root,provenance)
    return output,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-step',type=int)
    args=parser.parse_args()
    target,payload=build(args.run_dir,args.output,args.max_step)
    print(json.dumps(dict(html=str(target.resolve()),metrics=len(payload['metrics']),
                          standalone=True,bytes=target.stat().st_size),ensure_ascii=False))


if __name__=='__main__':
    main()
