"""Embed the two self-contained reading reports in a portable comparison HTML.

Run from the repository root: python -m scripts.research.build_offline_comparison
"""
import base64
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2] / 'research_runs'
    paths = [
        'step20_local_analysis/OPSD_step20_analysis.html',
        '20261028_second_analysis/OPSD_second_experiment_analysis.html',
    ]
    template = (root / 'OPSD_experiments_comparison.html').read_text(encoding='utf-8')
    payload = []
    for index, relative in enumerate(paths):
        payload.append(base64.b64encode((root / relative).read_bytes()).decode('ascii'))
        for attribute, replacement in (
            ('href', f'href="#" id="open-{index}"'),
            ('src', f'id="frame-{index}"'),
        ):
            original = f'{attribute}="{relative}"'
            if template.count(original) != 1:
                raise ValueError(f'Expected exactly one {original} in comparison template')
            template = template.replace(original, replacement)
    template = template.replace('OPSD · 两次实验报告对比', 'OPSD · 两次实验报告对比（离线单文件）')
    template = template.replace('两份报告独立滚动、筛选和切换图表；最大化浏览器可获得更多空间。',
                                '全部报告与图表已内嵌；复制本文件即可离线阅读。请保持本页打开以使用独立窗口。')
    embedded = '''
<script id="embedded-reports" type="application/json">PAYLOAD</script>
<script>
const embeddedReports = JSON.parse(document.getElementById('embedded-reports').textContent);
embeddedReports.forEach((encoded, index) => {
  const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
  const url = URL.createObjectURL(new Blob([bytes], {type: 'text/html;charset=utf-8'}));
  document.getElementById('frame-' + index).src = url;
  document.getElementById('open-' + index).href = url;
});
</script>
<noscript>请在浏览器中启用 JavaScript，以显示内嵌报告及交互图表。</noscript>
'''.replace('PAYLOAD', json.dumps(payload))
    template = template.replace('</body>', embedded + '</body>')
    output = root / 'OPSD_experiments_comparison_offline.html'
    output.write_text(template, encoding='utf-8')
    print(f'{output}\n{output.stat().st_size / 1024**2:.2f} MiB')


if __name__ == '__main__':
    main()
