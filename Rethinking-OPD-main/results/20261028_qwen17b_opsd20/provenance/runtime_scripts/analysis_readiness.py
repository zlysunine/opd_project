"""Read-only analysis readiness checks before starting the formal experiment."""
from collections import Counter
import html
import json
import math
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def check_numeric(row):
    name,value=row['metric'],row.get('value')
    if row.get('status')!='ok':return
    if not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(f'Invalid measured analysis value: {name}={value}')
    if row.get('n_valid',0)<=0:raise ValueError(f'{name}: finite measurement without valid samples')
    fraction=(name.startswith('neighbor_overlap') or '_shared_mass_' in name or 'empty_intersection_rate' in name
              or name.endswith('_overlap') or name.endswith('_overlap_to_initial')
              or name.endswith('_rate') or name.startswith('accuracy_') or name=='top_eigenvalue_fraction')
    if fraction and not -1e-5<=value<=1+1e-5:raise ValueError(f'{name} outside [0,1]: {value}')
    if ('entropy' in name and 'gap' not in name) or name.startswith('kl_'):
        if value < -1e-5:raise ValueError(f'Negative entropy/KL: {name}={value}')
    if name.startswith('rsa_rho') and not -1.00001<=value<=1.00001:
        raise ValueError(f'RSA correlation outside [-1,1]: {value}')
    if name.startswith('linear_cka') and not -1e-5<=value<=1+1e-5:
        raise ValueError(f'CKA outside [0,1]: {value}')
    if name=='curvature_deg' and not -1e-5<=value<=180.00001:
        raise ValueError(f'Curvature outside degrees range: {value}')


def validate_analysis_readiness(workdir, steps):
    workdir=Path(workdir)
    c=json.loads((workdir/'config.json').read_text())
    expected=c['monitor']['heldout_probe_n']
    cutoff=c['monitor']['quality_gate']['max_extraction_failure_rate']
    probe=workdir/'shared/probe'
    held_ids={row['problem_id'] for row in read_rows(probe/'heldout_probe.jsonl')}
    train_ids={row['problem_id'] for row in read_rows(probe/'train_probe.jsonl')}
    if len(held_ids)!=expected or held_ids & train_ids:raise ValueError('Invalid frozen probe membership')
    required_student={
        'rsa_rho_to_initial','rsa_rmse_to_initial','linear_cka_to_initial','neighbor_overlap_k8',
        'participation_ratio','curvature_deg','student_entropy_nats','teacher_entropy_nats',
        'initial_student_entropy_nats','kl_to_initial_nats','kl_student_teacher_nats','kl_teacher_student_nats',
        'token_top16_overlap','token_top16_overlap_to_initial','entropy_gap_abs_nats','entropy_gap_to_initial_abs_nats'}
    required_pair={'rsa_rho_student_teacher','rsa_rmse_student_teacher','linear_cka_student_teacher',
                   'neighbor_overlap_student_teacher_k8'}
    summaries=[]
    for step in steps:
        directory=workdir/'opsd/analysis/steps'/f'step_{step:06d}'
        metrics=read_rows(directory/'metrics.jsonl')
        for row in metrics:check_numeric(row)
        fixed=[row for row in metrics if row['pool']=='fixed_base_rollout' and row['split']=='heldout_probe']
        measured={row['metric'] for row in fixed if row['role']=='student' and row['status']=='ok'}
        missing=sorted(required_student-measured)
        if missing:raise ValueError(f'Step {step} missing measured student metrics: {missing}')
        if step in c['monitor']['teacher_geometry_steps']:
            paired={row['metric'] for row in fixed if row['role']=='student_vs_privileged_teacher' and row['status']=='ok'}
            missing=sorted(required_pair-paired)
            if missing:raise ValueError(f'Step {step} missing measured student/teacher geometry: {missing}')
        live=read_rows(directory/'live_rollouts.jsonl')
        held=[row for row in live if row['split']=='heldout_probe']
        if len(held)!=expected or {row['problem_id'] for row in held}!=held_ids:
            raise ValueError(f'Step {step} missing or changed heldout question IDs')
        failed=sum(row['extraction_failed'] for row in held)
        truncated=sum(row['truncated'] for row in held)
        if failed/expected>cutoff:raise ValueError(f'Step {step} extraction failure rate exceeds threshold')
        usable=sum(not row['extraction_failed'] and not row['truncated'] for row in held)
        # Preserve the distinction: a truncated boxed answer may be extractable,
        # but cannot count as a completed solution for interpretation.
        warnings=[]
        if usable/expected < 1-cutoff:
            warnings.append('Fewer completed, extractable responses than the configured success target')
        distributions=read_rows(directory/'distributions.jsonl')
        by_id={(row['pool'],row['split'],row['problem_id']):row for row in distributions}
        for row in live:
            dist=by_id.get(('live_rollout',row['split'],row['problem_id']))
            if dist is None:raise ValueError('Missing live all-token distribution record')
            if dist.get('distribution_scope')!='all_valid_tokens' or dist.get('n_valid_tokens')!=row['response_length']:
                raise ValueError('Live distribution was clipped or valid-token count is inconsistent')
        for row in distributions:
            for name,count in row['metric_token_counts'].items():
                if count:
                    value=row['metric_sums'].get(name)
                    if value is None or not math.isfinite(value):raise ValueError(f'Invalid all-token sum: {name}')
        undefined=Counter(row['metric'] for row in metrics if row['status']=='undefined')
        curves=read_rows(directory/'curvature.jsonl')
        if not curves or not (directory/'anchors.npz').is_file():raise ValueError('Missing geometry artifacts')
        summaries.append(dict(step=step,heldout_n=expected,extraction_successes=expected-failed,
            extraction_success_rate=1-failed/expected,extraction_failure_rate=failed/expected,
            truncation_rate=truncated/expected,complete_extractable_rate=usable/expected,
            accuracy=sum(row['correct'] for row in held)/expected,
            format_compliance_rate=sum(row.get('format_compliant',False) for row in held)/expected,
            metric_rows=len(metrics),measured_rows=sum(row['status']=='ok' for row in metrics),
            undefined_rows=sum(undefined.values()),undefined_by_metric=dict(undefined),
            all_valid_tokens_verified=True,student_initial_coverage=True,
            student_teacher_geometry_coverage=step in c['monitor']['teacher_geometry_steps'],warnings=warnings))
    stage_passed=not any(row['warnings'] for row in summaries)
    expected_steps=c['monitor']['optimizer_steps']
    complete=sorted(steps)==sorted(expected_steps)
    passed=stage_passed and complete
    report=dict(passed=passed,complete=complete,checked_steps=list(steps),expected_steps=expected_steps,stage_checks_passed=stage_passed,minimum_extraction_success_rate=1-cutoff,
        interpretation='Readiness validates measurements and coverage; it does not require accuracy or KL improvement.',
        undefined_policy='Undefined short anchors, N<=k, or zero-variance correlations remain null; core heldout metrics must be measured.',steps=summaries)
    destination=workdir/'report';destination.mkdir(exist_ok=True)
    (destination/'analysis_readiness.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    body=''.join('<tr>'+''.join(f'<td>{html.escape(str(value))}</td>' for value in [
        row['step'],row['heldout_n'],f"{row['extraction_success_rate']:.1%}",f"{row['truncation_rate']:.1%}",
        f"{row['accuracy']:.1%}",row['measured_rows'],row['undefined_rows']])+'</tr>' for row in summaries)
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>OPSD实验指标验收</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:16px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border:1px solid #ccc}}th{{background:#eef3f8}}</style>
<h1>实验指标验收：{'等待其余监测点' if not complete else ('通过' if stage_passed else '需检查')}</h1>
<p>已检查 step：{html.escape(str(list(steps)))}；完整验收要求：{html.escape(str(expected_steps))}。</p>
<p>抽取成功率门槛：{1-cutoff:.0%}。抽取成功、完成解答和答对分别统计。</p>
<table><tr><th>Step</th><th>heldout题数</th><th>抽取成功率</th><th>截断率</th><th>正确率</th><th>实测指标行</th><th>未定义行</th></tr>{body}</table>
<p>短回答、N≤k及零方差等未定义量保留为空；它们与数值错误分开。验收通过表示数据可分析，不表示数学能力提升。</p>
<p><a href="OPSD_report.html">打开完整交互式实验报告</a></p></html>'''
    (destination/'analysis_readiness.html').write_text(page)
    if not stage_passed:raise ValueError('Completed extractable response coverage is below the configured target; inspect analysis_readiness.json')
    return report
