# Qwen3-1.7B-Base OPSD · 2026-10-28

先下载并打开 [index.html](index.html) 查看总览，或打开 [OPSD_report.html](OPSD_report.html) 查看全部交互指标。两个HTML均可单文件离线使用。

- [交接文档](HANDOFF_zh.md)：实际结果、参数、数据、代码改动、验证、恢复路径和局限。
- [两步验收报告](acceptance_2step.html)。
- `data/`：训练指标、逐题行为/分布/曲率标量、候选池划分ID；不含题面、参考、回答文本或权重。
- `provenance/`：实际配置、训练时源码哈希、数据复用证明、验收与测试记录。
- `summary.json`：用于总览的原始汇总数据。
- `artifact_manifest.json`：各文件大小与SHA256；不包含清单自身。

核验导出包（在本目录执行，仅需Python标准库）：

```python
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('artifact_manifest.json').read_text())
for name, expected in manifest['files'].items():
    path = Path(name)
    assert path.stat().st_size == expected['bytes'], name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], name
print('All exported files verified')
```

原始大型文件保留在服务器。旧实验报告在 `results/step20_all_metrics`，不可与本次不同回答协议的曲线直接拼接。
