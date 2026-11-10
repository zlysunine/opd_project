"""Export the second reading report as editable Markdown plus 27 static figures.

The HTML's embedded, audited data and original Markdown are the sole inputs.
Only the default HTML view is rendered: heldout_probe, student-teacher RSA at
prompt_last, student curvature. The Markdown keeps all original narrative.
"""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / 'research_runs/20261028_second_analysis/OPSD_second_experiment_analysis.html'
COLORS = ['#117c79', '#d57b36', '#496cb4', '#b5536d', '#8b9d9d']
STEPS = [0, 1, 5, 10, 20]
LAYERS = list(range(1, 29))


class DataScript(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.text = ''

    def handle_starttag(self, tag, attrs):
        if tag == 'script' and dict(attrs).get('id') == 'analysis-data':
            self.inside = True

    def handle_endtag(self, tag):
        if tag == 'script':
            self.inside = False

    def handle_data(self, value):
        if self.inside:
            self.text += value


def load(html_path):
    parser = DataScript()
    parser.feed(html_path.read_text(encoding='utf-8'))
    if not parser.text:
        raise ValueError('No embedded analysis-data found in HTML')
    data = json.loads(parser.text)
    if not {'summary', 'monitor', 'training', 'markdown'} <= data.keys():
        raise ValueError('Incomplete second experiment report payload')
    return data


def scalar(data, metric, pool='fixed_base_rollout', role='student'):
    key = '|'.join((metric, pool, 'heldout_probe', role, ''))
    return sorted((r for r in data['monitor'].get(key, []) if r[1] is None), key=lambda r: r[0])


def layered(data, metric, role, anchor):
    key = '|'.join((metric, 'fixed_base_rollout', 'heldout_probe', role, anchor))
    return data['monitor'].get(key, [])


def setup(title, ylabel, xlabel='Optimizer step'):
    fig, ax = plt.subplots(figsize=(7.4, 4.25))
    ax.set_title(title, loc='left', fontsize=14, weight='bold', pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=.22, linewidth=.8)
    ax.spines[['top', 'right']].set_visible(False)
    return fig, ax


def line(ax, x, y, label, color, *, dash='-', marker='o'):
    ax.plot(x, y, label=label, color=color, linewidth=2, linestyle=dash,
            marker=marker, markersize=4)


def finish(fig, ax, out, *, legend=True):
    if legend:
        ax.legend(frameon=False, fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor='white')
    plt.close(fig)


def draw_lines(out, title, ylabel, items, xlabel='Optimizer step', percent=False):
    fig, ax = setup(title, ylabel, xlabel)
    for index, (label, x, y) in enumerate(items):
        values = [v*100 if percent and v is not None else v for v in y]
        line(ax, x, values, label, COLORS[index % len(COLORS)])
    finish(fig, ax, out, legend=len(items) > 1)


def export_figures(data, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    captions = {}

    def save(key, caption, draw):
        draw(outdir / (key + '.png'))
        captions[key] = caption

    training = data['training']
    steps = [r['step'] for r in training]

    def online(key, title, ylabel, fields, *, percent=False):
        save(key, title, lambda out: draw_lines(out, title, ylabel,
             [(label, steps, [r.get(field) for r in training]) for field, label in fields],
             percent=percent))

    online('01_online_kl', 'Online forward KL', 'nats',
           [('actor/forward_kl', 'KL(teacher || student)')])
    online('02_online_overlap', 'Online top-16 token overlap', '%',
           [('opd/overlap_ratio', 'student vs teacher')], percent=True)
    online('03_online_ota', 'Online Overlap-Token Advantage', 'nats',
           [('opd/overlap_token_advantage', 'OTA')])
    online('04_online_entropy', 'Online full-vocabulary entropy', 'nats',
           [('opd/student_entropy', 'student'), ('opd/teacher_entropy', 'teacher')])
    online('05_online_gap', 'Online entropy gap', 'nats',
           [('opd/entropy_gap', 'absolute gap'), ('opd/entropy_gap_signed', 'teacher - student')])
    for key, caption in {
        '01_online_kl': '在线前向 KL；每步训练题和回答不同，不能视为固定题测试曲线。',
        '02_online_overlap': '在线 top-16 token overlap；与 k16 题目近邻重叠是不同指标。',
        '03_online_ota': '在线 Overlap-Token Advantage；交集内的归一化负 KL，不是强化学习优势。',
        '04_online_entropy': '学生和当前特权教师在学生回答前缀上的全词表熵。',
        '05_online_gap': '逐位置熵差的绝对值与带符号均值，二者定义不同。',
    }.items():
        captions[key] = caption

    def fixed(key, title, ylabel, fields, *, pool='fixed_base_rollout', percent=False):
        items = []
        for field, label in fields:
            rows = scalar(data, field, pool)
            items.append((label, [r[0] for r in rows], [r[2] for r in rows]))
        save(key, title, lambda out: draw_lines(out, title, ylabel, items, percent=percent))

    fixed('06_fixed_kl', 'Fixed response: two KL directions', 'nats',
          [('kl_student_teacher_nats', 'KL(student || teacher)'),
           ('kl_teacher_student_nats', 'KL(teacher || student)')])
    fixed('07_fixed_initial', 'Fixed response: student vs initial', 'nats',
          [('kl_to_initial_nats', 'KL(student || initial student)')])
    fixed('08_fixed_overlap', 'Fixed response: top-16 overlap', '%',
          [('token_top16_overlap', 'student vs teacher'),
           ('token_top16_overlap_to_initial', 'student vs initial')], percent=True)
    fixed('09_fixed_ota', 'Fixed response: OTA', 'nats',
          [('overlap_token_advantage_nats', 'student vs teacher'),
           ('overlap_token_advantage_to_initial_nats', 'student vs initial')])
    fixed('10_fixed_entropy', 'Fixed response: entropy', 'nats',
          [('student_entropy_nats', 'student'), ('teacher_entropy_nats', 'teacher'),
           ('initial_student_entropy_nats', 'initial student')])
    fixed('11_fixed_gap', 'Fixed response: absolute entropy gap', 'nats',
          [('entropy_gap_abs_nats', 'teacher vs student'),
           ('entropy_gap_to_initial_abs_nats', 'student vs initial')])
    fixed('13_live_kl', 'Live response: diagnostic KL', 'nats',
          [('kl_student_teacher_nats', 'student vs teacher'),
           ('kl_to_initial_nats', 'student vs initial')], pool='live_rollout')
    fixed_items = []
    for pool, label in [('fixed_base_rollout', 'fixed'), ('live_rollout', 'live')]:
        rows = scalar(data, 'student_entropy_nats', pool)
        fixed_items.append((label, [r[0] for r in rows], [r[3] for r in rows]))
    save('14_valid_tokens', 'Valid response tokens',
         lambda out: draw_lines(out, 'Valid response tokens', 'token count', fixed_items))
    for key, caption in {
        '06_fixed_kl': '固定回答上的双向 KL，覆盖全部有效回答 token。',
        '07_fixed_initial': '当前学生相对初始学生的固定回答 KL；step0 为自身对照。',
        '08_fixed_overlap': '固定回答上的 top-16 token 集合重叠，两组比较分别标识。',
        '09_fixed_ota': '固定回答上的 OTA；两组比较分别标识。',
        '10_fixed_entropy': '同一固定前缀下，当前学生、当前教师和初始学生的熵。',
        '11_fixed_gap': '绝对熵差；不能用两条平均熵曲线相减替代。',
        '13_live_kl': 'live 回答随 checkpoint 变化，不能把跨步差值完全归因于参数。',
        '14_valid_tokens': '固定池和自主生成池的有效回答 token 数；token 不是独立题目样本。',
    }.items():
        captions[key] = caption

    changes = data['summary']['distributions']['heldout_probe']['fixed_base_rollout']['changes']
    ci_keys = ['kl_student_teacher_nats', 'kl_teacher_student_nats', 'entropy_gap_abs_nats']
    ci_labels = ['S || T', 'T || S', '|entropy gap|']

    def draw_ci(out):
        fig, ax = setup('Fixed response: paired step20 - step0', 'change (nats)', 'Metric')
        center = [changes[k]['change'] for k in ci_keys]
        low = [center[i]-changes[k]['ci95'][0] for i, k in enumerate(ci_keys)]
        high = [changes[k]['ci95'][1]-center[i] for i, k in enumerate(ci_keys)]
        ax.errorbar(ci_labels, center, yerr=[low, high], fmt='o', color=COLORS[0], capsize=5)
        ax.axhline(0, color='#8b9d9d', linestyle=':')
        finish(fig, ax, out, legend=False)
    save('12_fixed_ci', '配对问题 bootstrap 的 step20−step0 变化区间；未做跨训练 seed 推断。', draw_ci)

    role, anchor, metric = 'student_vs_privileged_teacher', 'prompt_last', 'rsa_rho_student_teacher'
    rows = layered(data, metric, role, anchor)
    lookup = {(int(r[0]), int(r[1])): r[2] for r in rows}

    def draw_repr_lines(out):
        fig, ax = setup('Student vs teacher RSA, prompt_last', 'RSA Spearman rho', 'Decoder block')
        for i, step in enumerate(STEPS):
            line(ax, LAYERS, [lookup.get((step, l), np.nan) for l in LAYERS],
                 'step '+str(step), COLORS[i], marker=None)
        finish(fig, ax, out)
    save('15_repr_lines', '当前学生与当前特权教师的直接 RSA；固定题、固定回答、prompt_last。', draw_repr_lines)

    matrix = np.asarray([[lookup.get((step, layer), np.nan) for layer in LAYERS] for step in STEPS])

    def draw_repr_heat(out):
        fig, ax = setup('Student vs teacher RSA by checkpoint', 'Checkpoint', 'Decoder block')
        image = ax.imshow(matrix, aspect='auto', origin='lower', interpolation='nearest',
                          extent=[.5, 28.5, -.5, 4.5], cmap='viridis')
        ax.set_yticks(range(5), [str(s) for s in STEPS]); ax.set_xticks([1, 7, 14, 21, 28])
        fig.colorbar(image, ax=ax, label='RSA Spearman rho')
        finish(fig, ax, out, legend=False)
    save('16_repr_heatmap', '同一步学生—教师 RSA 的 checkpoint×层热图；颜色刻度可能很窄。', draw_repr_heat)

    anchors = ['prompt_last', 'response_32', 'response_128']
    delta = []
    for a in anchors:
        values = layered(data, metric, role, a)
        by_key = {(int(r[0]), int(r[1])): r[2] for r in values}
        delta.append([by_key.get((20, l), np.nan)-by_key.get((0, l), np.nan) for l in LAYERS])
    delta = np.asarray(delta)

    def draw_repr_delta(out):
        fig, ax = setup('Student vs teacher RSA: step20 - step0', 'Anchor', 'Decoder block')
        limit = float(np.nanmax(np.abs(delta)))
        image = ax.imshow(delta, aspect='auto', interpolation='nearest', cmap='RdBu',
                          vmin=-limit, vmax=limit, extent=[.5, 28.5, -.5, 2.5], origin='lower')
        ax.set_yticks(range(3), ['prompt_last', 'response32', 'response128'])
        ax.set_xticks([1, 7, 14, 21, 28])
        fig.colorbar(image, ax=ax, label='delta rho')
        finish(fig, ax, out, legend=False)
    save('17_repr_delta', '3 个 anchor × 28 层的直接 RSA 变化；51 个上升、33 个下降。', draw_repr_delta)

    curves = data['summary']['curvature']['heldout_probe']
    student = curves['student']['profiles']; teacher = curves['privileged_teacher']['profiles']
    def draw_curve_profile(out):
        fig, ax = setup('Fixed-response curvature profile', 'degrees', 'Decoder block')
        for j, (name, profiles) in enumerate([('student', student), ('teacher', teacher)]):
            x = [r['layer'] for r in profiles]
            for i, state in enumerate(['initial', 'final']):
                line(ax, x, [r[state] for r in profiles], name+' '+('step0' if i==0 else 'step20'),
                     COLORS[j], dash='--' if i==0 else '-', marker=None)
        finish(fig, ax, out)
    save('18_curve_profile', '固定回答曲率：学生和教师各自的初始/最终层间形状。', draw_curve_profile)

    def draw_curve_delta(out):
        fig, ax = setup('Student curvature: paired change', 'step20 - step0 (degrees)', 'Decoder block')
        x = [r['layer'] for r in student]; y = [r['change'] for r in student]
        ax.errorbar(x, y, yerr=[[y[i]-r['ci95'][0] for i,r in enumerate(student)],
                                [r['ci95'][1]-y[i] for i,r in enumerate(student)]],
                    color=COLORS[0], capsize=2, linewidth=1.8)
        ax.axhline(0, color='#8b9d9d', linestyle=':')
        finish(fig, ax, out, legend=False)
    save('19_curve_delta', '学生各层同题配对曲率变化和问题 bootstrap 区间。', draw_curve_delta)

    def draw_curve_gap(out):
        draw_lines(out, 'Teacher - student curvature', 'degrees',
                   [(state, [r['layer'] for r in student],
                     [teacher[i][state]-r[state] for i,r in enumerate(student)])
                    for state in ['initial','final']], xlabel='Decoder block')
    save('20_curve_gap', '教师减学生的固定池层均值曲率差；不等于逐 token 向量距离。', draw_curve_gap)

    behavior = data['summary']['behavior']['heldout_probe']; runs = behavior['series']
    x = [r['step'] for r in runs]

    def draw_accuracy(out):
        fig, ax = setup('Heldout accuracy with Wilson intervals', 'accuracy (%)')
        y = [100*r['correct']/r['n'] for r in runs]
        ax.errorbar(x, y, yerr=[[y[i]-100*r['accuracy_ci95'][0] for i,r in enumerate(runs)],
                                [100*r['accuracy_ci95'][1]-y[i] for i,r in enumerate(runs)]],
                    color=COLORS[0], marker='o', capsize=4)
        ax.set_xticks(x)
        finish(fig, ax, out, legend=False)
    save('21_behavior_accuracy', 'heldout 50 题的准确率和各 checkpoint Wilson 区间；配对变化另见正文。', draw_accuracy)

    save('22_behavior_quality', '抽取失败、截断和格式合规可能有重叠，不能相加。',
         lambda out: draw_lines(out, 'Heldout response quality', '%',
         [('extraction failure', x, [100*r['extraction_failed']/r['n'] for r in runs]),
          ('truncation', x, [100*r['truncated']/r['n'] for r in runs]),
          ('format compliant', x, [100*r['format_compliant']/r['n'] for r in runs])]))

    transition = np.array([[runs[0]['n']-behavior['gained']-behavior['lost']-behavior['unchanged_correct'],
                            behavior['lost']], [behavior['gained'], behavior['unchanged_correct']]])

    def draw_transition(out):
        fig, ax = setup('Step0 to step20 correctness', 'step20', 'step0')
        ax.imshow(transition, cmap='Blues', vmin=0)
        ax.set_xticks([0,1], ['wrong','correct']);ax.set_yticks([0,1], ['wrong','correct'])
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(transition[row,col]), ha='center', va='center', fontsize=17)
        finish(fig, ax, out, legend=False)
    save('23_behavior_transition', '同题正误迁移：错→对 4 题，对→错 6 题。', draw_transition)

    status = runs[-1]['parse_status']
    def draw_parse(out):
        fig, ax = setup('Step20 answer extraction status', 'questions', 'Parse status')
        ax.bar(list(status), list(status.values()), color=COLORS[0])
        ax.tick_params(axis='x', labelrotation=15)
        for i,v in enumerate(status.values()): ax.text(i,v+.3,str(v),ha='center')
        finish(fig, ax, out, legend=False)
    save('24_behavior_parse', '最终解析状态；成功抽取不代表答案正确。', draw_parse)

    online('25_runtime_step', 'Training time per step', 'seconds',
           [('timing_s/step','total'),('timing_s/gen','generation'),
            ('timing_s/update_actor','actor update')])
    online('26_runtime_memory', 'Training GPU memory logs', 'GiB',
           [('perf/max_memory_allocated_gb','allocated'),
            ('perf/max_memory_reserved_gb','reserved')])
    times = data['summary']['resources']['analysis_seconds_by_step']
    def draw_analysis_time(out):
        fig, ax = setup('Checkpoint analysis time', 'minutes', 'Checkpoint')
        ax.bar(list(times), [v/60 for v in times.values()], color=COLORS[2])
        finish(fig, ax, out, legend=False)
    save('27_runtime_analysis', '各 checkpoint 的分析耗时，不含最初数据筛选和验收的全部成本。', draw_analysis_time)
    captions['25_runtime_step'] = '每步训练、rollout 生成和 actor 更新耗时。'
    captions['26_runtime_memory'] = '训练日志的 allocated 和 reserved；前者包含于后者，不能相加。'
    if len(captions) != 27:
        raise AssertionError(f'Expected 27 HTML-default charts, got {len(captions)}')
    return captions


def write_markdown(data, captions, output):
    body = data['markdown']
    intro = ('<!-- 可直接修改本文字。图表为网页默认视图的静态导出；重绘图片可使用 --figures-only。 -->\n'
             '<!-- 导出 PDF 时请将本文件与同名 _figures 文件夹保持在一起。 -->\n\n'
             '> 图表说明：本文件保留原报告完整中文分析，并加入 HTML 默认视图的 27 张静态图。'
             '默认视图为 heldout 50 题；表征图选同一步学生/教师、RSA、prompt_last；曲率变化图选学生。'
             'HTML 中可切换的其余视图没有在单张静态图内展开。\n\n')
    sections = {
        3: (1,5), 4: (6,14), 5: (15,17), 6: (18,20),
        7: (21,24), 8: (25,27),
    }
    for section,(first,last) in sections.items():
        images = []
        for number in range(first,last+1):
            key = next(k for k in captions if k.startswith(f'{number:02d}_'))
            relative = output.stem+'_figures/'+key+'.png'
            images.extend([f'![{captions[key]}]({relative})', '',
                           f'*图 {number}：{captions[key]}*', ''])
        marker = f'## {section+1}. '
        cut = body.find(marker)
        if cut == -1:
            raise ValueError(f'Missing section heading: {marker}')
        body = body[:cut] + '\n'.join(images) + '\n' + body[cut:]
    output.write_text(intro+body, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--output', type=Path, default=DEFAULT_REPORT.with_suffix('.md'))
    parser.add_argument('--figures-only', action='store_true', help='Redraw figures without touching edited Markdown')
    parser.add_argument('--force', action='store_true', help='Overwrite an existing Markdown report')
    args = parser.parse_args()
    data = load(args.html)
    if data['summary']['steps'] != STEPS:
        raise ValueError('Unexpected checkpoint steps for this report')
    output = args.output.resolve();output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.figures_only and not args.force:
        raise FileExistsError(f'Existing Markdown is protected: {output}. Use --figures-only or --force.')
    figures = output.with_name(output.stem+'_figures')
    captions = export_figures(data, figures)
    if not args.figures_only:
        write_markdown(data, captions, output)
    print(f'{output}\n{len(captions)} static figures in {figures}'
          + ('; Markdown preserved' if args.figures_only else ''))


if __name__ == '__main__':
    main()
