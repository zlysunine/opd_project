# 2026-10-28 正式20步实验结果

**训练与最终验收已完成。** 最新结果：[HTML总览](results/20261028_qwen17b_opsd20/index.html) · [完整交互报告](results/20261028_qwen17b_opsd20/OPSD_report.html) · [交接文档](results/20261028_qwen17b_opsd20/HANDOFF_zh.md)。

- 正式20步，监测点0/1/5/10/20；全部质量与数值验收通过。
- 最终抽取成功率 **86%**、失败率 **14%**、正确率 **46%**、截断率 **2%**（heldout50题）。
- 正式训练直接复用两步验收已筛选的数据及冻结监测输入；没有重新筛选或生成基线。
- 完成时间：北京时间 2026-10-28 14:50:10。报告单文件离线可用；模型权重保留在服务器，不上传GitHub。
- [结果和来源清单](results/20261028_qwen17b_opsd20/README.md)。正确率变化仅为50题、单seed观测，不代表稳定能力提升。

---

以下为历史实验报告：

# 第 20 步最终结果入口

统一入口：**[全指标交互报告](results/step20_all_metrics/index.html)**。
在浏览器打开，可以按名称搜索、按分类筛选，使用“上一个 / 下一个”逐项查看。
在 VS Code 中可下载 HTML 后用浏览器打开；如需静态图、RSM 图库和完整 CSV，请下载整个报告目录。

报告包括 98 个训练日志指标、48 个监测汇总指标，以及 15 个逐题概率分布、6 个逐题曲率和 6 个生成行为指标。
共 173 个按来源区分的入口，保留重复语义的别名和兼容占位字段，并不表示 173 个独立科学量。
可以切换 fixed/live、train/heldout、student/teacher、位置和 decoder 层；对应维度只在有记录的指标中出现。
支持趋势图、逐层对比、step × layer 热力图和逐题直方图，当前图可导出 PNG/SVG，指标可导出 CSV。

最终采用 Qwen3-1.7B-Base 的 checkpoint20，训练已经停止。
训练曲线仅使用 step1–20，离线分析使用 step0/1/5/10/20。
原始日志中的 step21 保留在服务器供审计，不用于本报告，也没有对应的完整最终 checkpoint。
缺失或未定义测量保留为空；教师几何仅有 step0/20，不补零、不插值。
RSM 图库是预选层的 120 张矩阵图，所有层的 RSM 标量统计可在指标浏览器中查看。

报告数据清单、适用范围和源文件 SHA256 见 [报告说明](results/step20_all_metrics/README.md) 和
[来源清单](results/step20_all_metrics/manifest.json)。
模型权重、数据集、原始隐藏状态等大文件仍位于服务器的 `research_runs` 和输入目录，不随报告复制。

重新生成报告（仅 CPU 读取现有结果，不启动训练）：

```bash
source /root/autodl-tmp/activate_opsd.sh
PYTHONPATH=/root/autodl-tmp/report_tools:$PYTHONPATH python -m scripts.research.build_metric_report \
  --run research_runs/base_opsd_200_train3600_parallel \
  --output results/step20_all_metrics
```

生成需要 NumPy、Matplotlib 和 Plotly；本机 Plotly 6.3.1 位于独立的 `report_tools` 目录。
已经生成的 HTML 内嵌数据及 Plotly，不需要联网或安装 Python 包即可查看主交互图。
