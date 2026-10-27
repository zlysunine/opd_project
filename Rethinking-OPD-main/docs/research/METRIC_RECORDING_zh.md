# 实验指标清单与完整HTML

2026-10-27。记 `S_s` 为当前学生，`S_0` 为初始学生，`T_s` 为当前学生参数加reference的特权自教师。主分析按以下两组比较梳理；`T_s`与`T_0`仅作为独立辅助分析保留。

| metric | 当前学生与初始学生 | 当前学生与当前老师 |
|---|---|---|
| top-16 overlap ratio | Top16(S_s)与Top16(S_0)的交集比例 | Top16(S_s)与Top16(T_s)的交集比例 |
| Overlap-Token Advantage | S_s与S_0交集内的优势 | S_s与T_s交集内的优势 |
| Entropy | 分别记录H(S_s)、H(S_0) | 分别记录H(S_s)、H(T_s) |
| Entropy Gap（绝对） | 平均每位置abs[H(S_0)−H(S_s)] | 平均每位置abs[H(T_s)−H(S_s)] |
| Entropy Gap（有符号） | 平均每位置H(S_0)−H(S_s) | 平均每位置H(T_s)−H(S_s) |
| KL散度 | KL(S_s∥S_0) | KL(S_s∥T_s)、KL(T_s∥S_s) |
| top-16共享概率质量 | 交集上S_s、S_0的概率质量 | 交集上S_s、T_s的概率质量 |
| top-16空交集率 | S_s与S_0的空交集位置比例 | S_s与T_s的空交集位置比例 |
| RSA Spearman | 比较S_s与S_0的题间相似度矩阵 | 比较S_s与T_s的题间相似度矩阵 |
| RSM RMSE | 两者RSM对应元素的均方根差 | 两者RSM对应元素的均方根差 |
| Linear CKA | S_s与S_0中心化表征的关系结构 | S_s与T_s中心化表征的关系结构 |
| 表征kNN overlap，k=8/16/32 | 两者的题目近邻集合重叠率 | 两者的题目近邻集合重叠率 |
| 轨迹曲率 | 已记录成对变化C(S_s)−C(S_0) | 分别记录C(S_s)、C(T_s)，尚无独立的配对差值字段 |

上表核对checkpoint离线记录；训练batch的在线指标另列，不能替代固定题集的两组比较。分布指标读取同一条回答前缀，使用全词表、全部有效回答位置，以有效token加权；top-k仅用于overlap/OTA/共享质量。OTA只排除空交集位置，单独保存分母。KL方向保持显式：当前学生对初始学生目前只记录上表方向，训练loss仍为KL(stopgrad(T_s)∥S_s)。

表征比较使用固定题目、固定回答、相同层与anchor，并按problem_id对齐；heldout与train_probe独立统计。RSA/CKA接近1描述关系结构接近，不保证向量坐标相同。词表top-16与题目表征kNN的k是不同维度。

Entropy、参与率PR、首特征值占比和激活范数是单模型/上下文统计，不是两模型相似度。S_0参数固定，但live回答前缀会改变，所以在不同live前缀上评分的H(S_0)仍可能变化。固定文本条件下的初始基准保持不变。

此外单独记录学生生成正确率（微平均/主题宏平均）、答案抽取失败率、格式合规、截断率、EOS率、回答长度与重复情况，以及训练loss、梯度、学习率、token预算、耗时和显存。T_s在该监测中只评分学生轨迹，没有独立生成的教师正确率曲线。

## HTML的组织方式

`analysis/html_report.py`读取已完成checkpoint，自动发现数值字段；`analysis/report_template.html`提供概览、完整指标浏览、层×step热图、逐题分布、RSM图库和数据导出。

概览包含与上表对应的指标清单，每格可展开代码字段，分别标注“代码已支持”和“当前结果有无实测值”。代码支持某指标不代表历史实验已经采集。全回答token均值、历史选点统计和逐题摘要各自标注；学生—教师直接比较与各自对step0比较分开。

数据、Plotly.js、静态图和RSM图片全部内嵌，不依赖CDN、本地服务器或相邻图片目录。可搜索指标、选择上下文/层、查看曲线/热图、导出PNG/SVG及CSV；文件来源与SHA256可直接导出JSON。

当前本地可读文件：`research_runs/step20_local_analysis/OPSD_metrics_complete.html`。它来自真实step20历史结果，保留173个指标入口和120张RSM图；新增全回答统计与直接S_s/T_s表征指标缺少真实测量时标为未采集，不伪造数据。

## 生成或重新绘图

最新一键流程在 `analysis.report.report()` 中自动写出 `<workdir>/report/OPSD_report.html`。仅重新生成HTML不加载模型、不使用GPU，也无需重跑训练：

```bash
python -m analysis.html_report \
  --run-dir research_runs/base_geometry_50_answer_v2_full_tokens_paired \
  --output research_runs/base_geometry_50_answer_v2_full_tokens_paired/report/OPSD_report.html
```

也可以直接读取已有导出结果，不需要服务器原始目录：

```bash
python -m analysis.html_report \
  --run-dir results/step20_all_metrics \
  --output research_runs/step20_local_analysis/OPSD_metrics_complete.html
```

HTML重绘无法补算未采集的新指标；这些数据需由相应的checkpoint监测产生。图表仅展示完成的checkpoint，未保存的训练尾部不充当最终模型结果。
