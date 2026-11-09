"""Build a small, static phone report and a Chinese-font-embedded PDF.

The source is the two audited local summary.json files. Set OPSD_PDF_FONT to a
locally available Chinese font before running; the released PDF embeds it.
"""
import html
import io
import json
import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[2] / 'research_runs'
FIRST = ROOT / 'step20_local_analysis' / 'summary.json'
SECOND = ROOT / '20261028_second_analysis' / 'summary.json'
STEPS = [0, 1, 5, 10, 20]
TEAL, BLUE, MUTED = '#176f75', '#4754a8', '#536875'


def chart(title, y_label, x, lines, *, ylim=None, note=''):
    fig, ax = plt.subplots(figsize=(6.7, 7.2))
    fig.subplots_adjust(top=.86, bottom=.29, left=.17, right=.96)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f6f9fa')
    for label, y, color in lines:
        ax.plot(x, y, marker='o', markersize=5, linewidth=2.3, label=label, color=color)
    ax.set_title(title, fontsize=17, fontweight='bold', loc='left', pad=18)
    ax.set_xlabel('Step' if x == STEPS else 'Decoder block', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(alpha=.25)
    ax.spines[['top', 'right']].set_visible(False)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if len(lines) > 1:
        ax.legend(loc='best', fontsize=10)
    if note:
        fig.text(.10, .23, note, fontsize=10, color=MUTED, va='top')
    return fig


def save_mobile():
    a = json.loads(FIRST.read_text())
    b = json.loads(SECOND.read_text())
    f = a['behavior']['heldout_probe']['series']
    s = b['behavior']['heldout_probe']['series']
    fa = a['fixed_distributions']['heldout_probe']['changes']
    sb = b['distributions']['heldout_probe']['fixed_base_rollout']['changes']
    fc = a['curvature']['heldout_probe']['fixed_base_rollout__student']
    sc = b['curvature']['heldout_probe']['student']['profiles']
    direct = b['representation']['heldout_probe']['student_teacher']['rsa_rho_student_teacher']['final_layer']

    plots = [
        ('第一次：自主生成', chart('第一次实验 · heldout 512 题', '百分比 (%)', STEPS,
            [('正确率', [100*v['correct']/v['n'] for v in f], TEAL),
             ('抽取失败率', [100*v['extraction_failed']/v['n'] for v in f], BLUE)],
            ylim=(0, 105), note='greedy 最长 256 tokens；抽取失败率约 98%。'),
         'step0 正确 6/512，step20 正确 7/512，净增 1 题；问题配对区间包含 0。高抽取失败率限制了能力判断。'),
        ('第二次：自主生成', chart('第二次实验 · heldout 50 题', '百分比 (%)', STEPS,
            [('正确率', [100*v['correct']/v['n'] for v in s], TEAL),
             ('抽取失败率', [100*v['extraction_failed']/v['n'] for v in s], BLUE)],
            ylim=(0, 62), note='greedy 最长 16384 tokens；答案协议 v2。'),
         'step0 正确 25/50，step20 正确 23/50，净减 2 题；95% 区间为 -16 至 +8 个百分点。没有可靠改善证据。'),
        ('第一次：固定状态', chart('第一次 · 固定前缀诊断 KL', 'KL (nats)', [0, 20],
            [('KL(学生 || 特权教师)', [fa['kl_student_teacher_nats']['initial_mean'],
                                      fa['kl_student_teacher_nats']['final_mean']], TEAL)],
            note='只取回答位置 1 / 32 / 128；每题等权。'),
         '固定题与固定回答下，KL 由 0.299998 降至 0.296125。它是学生→教师方向的诊断值，与在线训练损失口径不同。'),
        ('第二次：固定状态', chart('第二次 · 固定前缀诊断 KL', 'KL (nats)', [0, 20],
            [('KL(教师 || 学生)', [sb['kl_teacher_student_nats']['initial'],
                                    sb['kl_teacher_student_nats']['final']], TEAL),
             ('KL(学生 || 教师)', [sb['kl_student_teacher_nats']['initial'],
                                    sb['kl_student_teacher_nats']['final']], BLUE)],
            note='全部有效回答位置；按 token 加权。'),
         '教师→学生 KL 由 0.044307 降至 0.043671；学生→教师 KL 由 0.039777 降至 0.039432。两次实验的数值不能直接横比。'),
        ('第一次：曲率', chart('第一次 · 固定回答表征转角', '平均转角 (度)',
            [v['layer'] for v in fc], [('step0', [v['initial'] for v in fc], BLUE),
                                      ('step20', [v['final'] for v in fc], TEAL)],
            note='28 个 decoder block；同题固定文本。'),
         '两条曲线几乎重合，最低点在第 16 层。约 7.31° 的层间下降在初始模型中已存在。'),
        ('第二次：曲率', chart('第二次 · 固定回答表征转角', '平均转角 (度)',
            [v['layer'] for v in sc], [('step0', [v['initial'] for v in sc], BLUE),
                                      ('step20', [v['final'] for v in sc], TEAL)],
            note='28 个 decoder block；同题固定文本。'),
         '两条曲线几乎重合，最低点均在第 16 层。不能把初始已有的层间形状当作 OPSD 造成的改变。'),
        ('第二次：直接表征', chart('第二次 · 末层学生—教师 RSA', 'RSA Spearman rho',
            [1, 2, 3], [('step0', [direct[k]['initial'] for k in
                                  ['prompt_last','response_32','response_128']], BLUE),
                        ('step20', [direct[k]['final'] for k in
                                   ['prompt_last','response_32','response_128']], TEAL)],
            ylim=(.70, 1.01), note='横轴 1=prompt_last, 2=response32, 3=response128。'),
         '末层 prompt_last 的直接 RSA 从 0.7599 升至 0.7779；32/128 位置变化很小。全部 84 个层×位置中，51 个升、33 个降。'),
    ]

    font_path = Path(os.environ.get('OPSD_PDF_FONT', '/tmp/opd_NotoSansCJKsc-Regular.otf'))
    if not font_path.is_file():
        raise FileNotFoundError(f'Provide an embeddable Chinese font via OPSD_PDF_FONT: {font_path}')
    font = FontProperties(fname=str(font_path))
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['font.sans-serif'] = [font.get_name()]
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'path'
    for _, fig, explanation in plots:
        fig.text(.10, .17, '\n'.join(textwrap.wrap(explanation, width=38)),
                 fontsize=11, color='#203743', va='top')
    # Figures already exist; set their text font explicitly before export.
    for _, fig, _ in plots:
        for item in fig.findobj(match=matplotlib.text.Text):
            item.set_fontproperties(font)

    output_pdf = ROOT / 'OPSD_experiments_mobile.pdf'
    output_html = ROOT / 'OPSD_experiments_mobile.html'
    with PdfPages(output_pdf) as pdf:
        cover = plt.figure(figsize=(6.7, 7.2), facecolor='#ffffff')
        cover.text(.10, .86, 'OPSD 两次实验', fontsize=28, color=TEAL, fontproperties=font)
        cover.text(.10, .78, '手机阅读版 · 2026-10-28', fontsize=16, fontproperties=font)
        cover_text = [
            '两次均为 Qwen3-1.7B-Base 的 LoRA OPSD，',
            '实际训练 20 步，每步 8 题 × 8 条轨迹。',
            '',
            '第一次：heldout 512 题，live 最长 256 tokens。',
            '正确率 6/512 → 7/512；抽取失败约 98%。',
            '',
            '第二次：heldout 50 题，live 最长 16384 tokens。',
            '正确率 25/50 → 23/50；区间包含 0。',
            '',
            '两次设置与统计口径不同，只比较各自的',
            'step0 → step20。均无可靠正确率提升证据。',
            '',
            '后续页面为静态矢量图，离线可读。',
            '详细指标见电脑端完整交互报告。',
        ]
        cover.text(.10, .68, '\n'.join(cover_text), fontsize=13,
                   linespacing=1.7, va='top', fontproperties=font)
        pdf.savefig(cover, metadata={'Title': 'OPSD 两次实验手机阅读版'})
        plt.close(cover)
        for _, fig, _ in plots:
            pdf.savefig(fig, metadata={'Title': 'OPSD 两次实验手机阅读版'})

    cards = []
    for heading, fig, explanation in plots:
        buffer = io.StringIO()
        fig.savefig(buffer, format='svg', transparent=False)
        svg = buffer.getvalue().split('<svg', 1)[1]
        cards.append(f'<section><h2>{html.escape(heading)}</h2><div class="plot"><svg{svg}</div>'
                     f'<p>{html.escape(explanation)}</p></section>')
        plt.close(fig)
    content = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OPSD 两次实验 · 手机阅读版</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f6f7;color:#203743;font:16px/1.7 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}
main{max-width:740px;margin:auto;padding:18px 14px 36px}h1{font-size:27px;line-height:1.3;margin:12px 0}h2{font-size:20px;line-height:1.4;margin:0 0 8px}
p{margin:8px 0 0}.lede{color:#526a73;margin-bottom:22px}section{background:white;border:1px solid #dce7e9;border-radius:12px;padding:16px;margin:16px 0;box-shadow:0 2px 7px #1737440b}
.plot{overflow:hidden}.plot svg{width:100%;height:auto;display:block}.caution{border-left:4px solid #d49b32;padding-left:12px}small{color:#536875}
@media(max-width:500px){main{padding:11px}section{padding:12px;margin:11px 0}h1{font-size:23px}h2{font-size:18px}}
</style></head><body><main><h1>OPSD 两次实验：手机阅读版</h1>
<p class="lede">静态图表、无需联网或脚本。根据两次已核验报告生成；两次均实际训练 20 步。此版精选主要结果，电脑端完整版仍提供更多指标与交互图。</p>
<section><h2>先看结论</h2><p>两次实验的正确率变化都没有可靠提升证据。固定文本的分布与表征有轻微变化，不能单凭这些变化认定数学能力改善。</p>
<p class="caution">两次实验的题集大小、生成长度、答案协议和分布统计口径不同；请分别比较各自 step0→step20，不要直接用第一次与第二次的正确率差衡量训练效果。</p></section>
CARDS
<section><h2>阅读范围</h2><p>第一次：heldout 512 题，live 回答上限 256 tokens，固定分布只取 1/32/128 位置。第二次：heldout 50 题，live 回答上限 16384 tokens，固定分布覆盖全部有效回答位置。均为 Qwen3-1.7B-Base 的 LoRA OPSD、每步 8 题×8 条轨迹。第二次的 heldout 已用于验收，不能称为完全未触及的外部测试集。</p>
<p>此文件保留关键图表与解读。详细数据、置信区间及全部交互图表见电脑端完整报告。</p></section>
<small>来源：research_runs/step20_local_analysis/summary.json 与 research_runs/20261028_second_analysis/summary.json。图表由本地汇总值重建。</small>
</main></body></html>'''.replace('CARDS', '\n'.join(cards))
    output_html.write_text(content, encoding='utf-8')
    print(output_html, output_html.stat().st_size, 'bytes')
    print(output_pdf, output_pdf.stat().st_size, 'bytes')


if __name__ == '__main__':
    save_mobile()
