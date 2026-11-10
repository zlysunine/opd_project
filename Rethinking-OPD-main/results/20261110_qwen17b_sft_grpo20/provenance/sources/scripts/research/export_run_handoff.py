"""Export a completed, verified OPSD run without weights or raw question text.

The source run is read-only. The exported detailed HTML is self-contained.
"""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import html
import importlib.metadata
import json
import math
from pathlib import Path
import shutil


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in records for key in row})
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def seal(output):
    output = Path(output)
    files = {}
    for path in sorted(output.rglob('*')):
        if not path.is_file() or path.name == 'artifact_manifest.json':
            continue
        if path.suffix.lower() in {'.pt', '.pth', '.bin', '.safetensors', '.ckpt', '.npz'}:
            raise ValueError(f'Model or array artifact cannot be published: {path}')
        if path.stat().st_size >= 50 * 1024 * 1024:
            raise ValueError(f'Export exceeds per-file budget: {path}')
        files[str(path.relative_to(output))] = dict(bytes=path.stat().st_size, sha256=sha(path))
    write_json(output / 'artifact_manifest.json', dict(schema=1, files=files,
               exclusions=['model weights', 'optimizer states', 'raw datasets', 'raw hidden states', 'tokens and answer text']))
    return dict(files=len(files), bytes=sum(row['bytes'] for row in files.values()))


def plot(title, series, percent=False):
    """Small standalone SVG chart; points expose exact values as tooltips."""
    valid = [(x, y) for _, points, _ in series for x, y in points
             if isinstance(y, (int, float)) and math.isfinite(y)]
    if not valid:
        return ''
    xmin, xmax = min(x for x, y in valid), max(x for x, y in valid)
    ymin, ymax = (0, 1) if percent else (0, max(y for x, y in valid) * 1.08 or 1)
    def xy(x, y):
        return 55 + (x - xmin) / max(1, xmax - xmin) * 690, 235 - (y - ymin) / (ymax - ymin) * 205
    parts = ['<svg viewBox="0 0 800 285" role="img" aria-label="' + html.escape(title) + '">']
    for j in range(5):
        value = ymin + (ymax - ymin) * j / 4
        y = 235 - j * 205 / 4
        label = f'{value:.0%}' if percent else f'{value:.3g}'
        parts.append(f'<path d="M55 {y}H745" stroke="#e2e8f0"/><text x="48" y="{y+4}" text-anchor="end">{label}</text>')
    for name, points, color in series:
        coords = [xy(x, y) for x, y in points if isinstance(y, (int, float)) and math.isfinite(y)]
        parts.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in coords)}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in points:
            if not isinstance(y, (int, float)) or not math.isfinite(y):
                continue
            px, py = xy(x, y)
            parts.append(f'<circle cx="{px}" cy="{py}" r="4" fill="{color}"><title>{html.escape(name)} · step {x}: {y:.6g}</title></circle>')
    for x in sorted({x for x, y in valid}):
        if xmax - xmin > 10 and x not in (xmin, xmax) and x % 5:
            continue
        px, _ = xy(x, ymin)
        parts.append(f'<text x="{px}" y="255" text-anchor="middle">{x}</text>')
    parts.append('<text x="750" y="255">step</text></svg>')
    legend = ' · '.join(f'<span style="color:{color}">{html.escape(name)}</span>' for name, _, color in series)
    return f'<section><h2>{html.escape(title)}</h2>{"".join(parts)}<p>{legend}</p></section>'


def overview(summary):
    training, checkpoints = summary['training'], summary['checkpoints']
    charts = plot('抽取成功率、正确率与截断率（heldout 50 题）', [
        (name, [(row['step'], row[field]) for row in checkpoints], color)
        for name, field, color in [('抽取成功率', 'extraction_success_rate', '#2563eb'),
                                  ('正确率', 'accuracy', '#16a34a'), ('截断率', 'truncation_rate', '#dc2626')]], True)
    for title, fields in [
        ('训练前向 KL', [('actor/forward_kl', '#2563eb')]),
        ('梯度范数', [('actor/grad_norm', '#7c3aed')]),
        ('每步耗时（秒，评测另计）', [('timing_s/step', '#2563eb'), ('timing_s/gen', '#ea580c')]),
        ('在线学生与教师熵（nats）', [('opd/student_entropy', '#2563eb'), ('opd/teacher_entropy', '#ea580c')])]:
        charts += plot(title, [(key, [(row['step'], row.get(key)) for row in training], color) for key, color in fields])
    body = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in [
        row['step'], f"{row['extraction_success_rate']:.0%}", f"{row['accuracy']:.0%}",
        f"{row['truncation_rate']:.0%}", f"{row['complete_extractable_rate']:.0%}", row['undefined_rows']]) + '</tr>' for row in checkpoints)
    final = checkpoints[-1]
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3-1.7B OPSD 20步实验结果 · 2026-10-28</title>
<style>body{{font:16px system-ui;color:#18243b;background:#f4f7fb;max-width:1150px;margin:36px auto;padding:20px;line-height:1.7}}section,.card{{background:white;border:1px solid #dbe3ef;border-radius:12px;padding:20px;margin:20px 0}}h1{{line-height:1.3}}h2{{font-size:20px}}svg{{width:100%;font:12px system-ui}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}a{{color:#1d4ed8}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px}}.cards .card{{margin:0}}strong{{font-size:25px}}.note{{color:#52627b}}</style>
<h1>Qwen3-1.7B-Base · OPSD 20 步实验</h1><p>最终验收通过。数据版本、实际配置与文件 SHA256 随报告保存。</p>
<div class="cards"><div class="card">参数更新<br><strong>20 / 20</strong></div><div class="card">最终抽取成功率<br><strong>{final['extraction_success_rate']:.0%}</strong></div><div class="card">最终正确率<br><strong>{final['accuracy']:.0%}</strong></div><div class="card">最终截断率<br><strong>{final['truncation_rate']:.0%}</strong></div></div>
<p><a href="OPSD_report.html">完整交互报告：全部指标、层、fixed/live 与 RSM 图库</a> · <a href="HANDOFF_zh.md">交接文档</a> · <a href="summary.json">汇总数据</a> · <a href="artifact_manifest.json">SHA256 文件清单</a></p>
<section><h2>质量验收</h2><table><tr><th>Step</th><th>抽取成功率</th><th>正确率</th><th>截断率</th><th>完整且可抽取</th><th>未定义指标行</th></tr>{body}</table><p class="note">成功率门槛为 70%；抽取成功、正常完成与答案正确分别统计。N=50，每道题对应 2 个百分点。验收通过代表测量可分析，不代表训练提高了能力。未定义值保持为空，不补零。</p></section>
{charts}<section><h2>适用范围与注意事项</h2><p>本轮与两步验收复用相同的 heldout 50 题、train_probe 4 题、基线与固定回答。正式训练候选池为 2814 题，独立 validation 为 512 题；20×8=160 次训练题目暴露，不表示训练完整遍历候选池。</p><p>固定回答的几何窗口为最多 256 tokens；live 解题上限为 16384 tokens，输出分布对全部有效回答 token 统计。global_seqlen 统计异常、未启用 critic 的占位字段、少量短回答及 N≤k 导致的未定义值均保留并注明。4 题 train_probe 的结果仅供诊断。</p><p>运行时间：{html.escape(summary['session']['started_utc'])} 至 {html.escape(summary['session']['finished_utc'])}（UTC）。模型、优化器与原始隐藏状态保留在服务器，不上传 GitHub。</p></section></html>'''


def export(run, output, verification, acceptance, logdir):
    run, output, logdir = Path(run).resolve(), Path(output).resolve(), Path(logdir).resolve()
    if output == run or output.is_relative_to(run):
        raise ValueError('Export must be outside the source run')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new, empty output directory')
    attempt = read_json(run / '.session/current.json')['attempt']
    state = read_json(run / '.session/attempts' / attempt / 'state.json')
    if state['status'] != 'completed' or state.get('returncode') != 0:
        raise ValueError('Training, analysis and report must finish before export')
    receipt = read_json(verification)
    if Path(receipt['workdir']).resolve() != run or not receipt['quality_gates_passed']:
        raise ValueError('Verification receipt does not authorize this run')
    validation = receipt['training_metrics_validation']
    readiness = validation['analysis_readiness']
    if not validation['passed'] or not readiness['passed'] or not readiness['complete']:
        raise ValueError('Training and full analysis readiness must pass')
    config = read_json(run / 'config.json')
    steps = config['monitor']['optimizer_steps']
    training = list(rows(run / 'opsd/train/metrics.jsonl'))
    if [row['step'] for row in training] != list(range(1, 21)) or steps != [0, 1, 5, 10, 20]:
        raise ValueError('This handoff requires exactly 20 updates and all five analysis points')
    sources = {}
    def copy(source, relative):
        source = Path(source)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sources[str(source)] = dict(sha256=sha(source), bytes=source.stat().st_size)
    output.mkdir(parents=True, exist_ok=True)
    copy(run / 'report/OPSD_report.html', 'OPSD_report.html')
    copy(Path(acceptance) / 'report/OPSD_report.html', 'acceptance_2step.html')
    for relative in ['config.json', 'models.json', 'opsd/training_config.json', 'opsd/analysis_config.json',
                     'shared/probe/manifest.json', 'shared/probe/complete.json', 'shared/probe/preparation_reuse.json',
                     'opsd/train/metric_validation.json', 'opsd/train/provenance.json', 'opsd/train/resolved_config.json',
                     'opsd/train/verl_resolved_config.json', 'opsd/train/command.json', 'opsd/train/preflight.json',
                     'report/analysis_readiness.json', 'report/report_manifest.json']:
        source = run / relative
        if source.is_file():
            copy(source, 'provenance/' + relative)
    copy(run / 'opsd/train/metrics.jsonl', 'data/training_metrics.jsonl')
    train_log = run / 'opsd/train/train.log'
    (output / 'provenance/training_teardown.log').write_text(train_log.read_text()[-3500:])
    sources[str(train_log)] = dict(sha256=sha(train_log), bytes=train_log.stat().st_size)
    copy(verification, 'provenance/formal_verification.json')
    for name in ['acceptance_verification.json', 'formal_full_input_reuse.json', 'active_run.json',
                 'parquet_transport_fix.json', 'final_code_validation.json', 'final_code_validation.log',
                 'pip-freeze.txt', 'environment-check.json']:
        source = logdir / name
        if source.is_file():
            copy(source, 'provenance/' + name)
    behavior, distributions, curvature = [], [], []
    for step in steps:
        directory = run / f'opsd/analysis/steps/step_{step:06d}'
        complete = read_json(directory / 'complete.json')
        for name, digest in complete['files'].items():
            if sha(directory / name) != digest:
                raise ValueError(f'Analysis artifact changed: {directory / name}')
        for name in ['complete.json', 'run_manifest.json']:
            copy(directory / name, f'provenance/steps/{step:06d}/' + name)
        for source_name, collection, fields in [
            ('live_rollouts.jsonl', behavior, {'optimizer_step', 'pool', 'split', 'problem_id', 'correct',
              'extraction_failed', 'parse_status', 'format_compliant', 'response_length', 'eos', 'truncated', 'repeated_4gram_rate'}),
            ('distributions.jsonl', distributions, None), ('curvature.jsonl', curvature, None)]:
            source = directory / source_name
            sources[str(source)] = dict(sha256=sha(source), bytes=source.stat().st_size)
            for raw in rows(source):
                safe = {k: v for k, v in raw.items() if (fields is None or k in fields)
                        and isinstance(v, (int, float, bool, type(None)))}
                for key in ['pool', 'split', 'problem_id', 'role', 'anchor', 'distribution_scope', 'aggregation', 'comparison', 'parse_status']:
                    if key in raw:
                        safe[key] = raw[key]
                safe['step'] = step
                for group in ['metric_sums', 'metric_token_counts']:
                    if group in raw:
                        safe.update({group + '/' + key: value for key, value in raw[group].items()})
                collection.append(safe)
    write_csv(output / 'data/behavior.csv', behavior)
    write_csv(output / 'data/distributions.csv.gz', distributions)
    write_csv(output / 'data/curvature.csv.gz', curvature)
    membership = []
    for split in ['train', 'validation', 'heldout_probe', 'train_probe']:
        source = run / 'shared/probe' / (split + '.jsonl')
        sources[str(source)] = dict(sha256=sha(source), bytes=source.stat().st_size)
        membership.extend(dict(split=split, problem_id=row['problem_id'], difficulty=row.get('difficulty'),
                               topic=row.get('topic_label')) for row in rows(source))
    write_csv(output / 'data/probe_membership.csv', membership)
    source = run / 'report/metrics.csv'
    if source.is_file():
        with source.open('rb') as stream, gzip.open(output / 'data/checkpoint_metrics.csv.gz', 'wb') as dest:
            shutil.copyfileobj(stream, dest)
        sources[str(source)] = dict(sha256=sha(source), bytes=source.stat().st_size)
    for name in ['run_new_experiment.py', 'pilot_validation.py', 'analysis_readiness.py',
                 'reuse_full_pilot_inputs.py', 'adopt_baseline_prefix.py', 'recover_screened_pool.py',
                 'build_recovered_selection_cache.py', 'write_handoff_docs.py', 'complete_and_shutdown.py']:
        copy(logdir / name, 'provenance/runtime_scripts/' + name)
    versions = {}
    for name in ['torch', 'transformers', 'vllm', 'flash-attn', 'peft', 'ray', 'pyarrow', 'numpy', 'pandas']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(output / 'provenance/environment_versions.json', versions)
    metrics = [dict(step=row['step'], **row['metrics']) for row in training]
    summary = dict(schema=1, run=run.name, exported_utc=datetime.now(timezone.utc).isoformat(),
                   session=state, passed=True, configuration=dict(training=config['training'],
                   dataset=config['dataset'], monitor=config['monitor']), checkpoints=readiness['steps'],
                   training=metrics, final_budget=training[-1]['budget'],
                   reuse=read_json(logdir / 'formal_full_input_reuse.json'), source_files=sources)
    write_json(output / 'summary.json', summary)
    (output / 'index.html').write_text(overview(summary))
    (output / 'HANDOFF_zh.md').write_text(
        '# OPSD已验收结果导出\n\n'
        + f'源实验：`{run}`。20步及所有监测点验收通过。\n\n'
        + '打开 index.html 查看总览，OPSD_report.html 查看全部交互指标。\n'
        + '实际配置、划分ID、源码哈希、验收和环境信息在 provenance 与 data 中。\n'
        + '模型、优化器、原始题目与隐藏状态留在服务器；此导出不包含这些文件。\n'
        + '完整数值和运行时间见 summary.json；文件完整性见 artifact_manifest.json。\n')
    (output / 'README.md').write_text(
        '# OPSD 20步结果\n\n打开 index.html 或 OPSD_report.html，均可离线查看。\n'
        + '交接说明见 HANDOFF_zh.md；来源、验证与数据见 provenance/、data/。\n')
    write_json(output / 'provenance/source_files.json', sources)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--verification', required=True, type=Path)
    parser.add_argument('--acceptance', required=True, type=Path)
    parser.add_argument('--logdir', required=True, type=Path)
    args = parser.parse_args()
    export(args.run, args.output, args.verification, args.acceptance, args.logdir)
    print(json.dumps(seal(args.output)))


if __name__ == '__main__':
    main()
