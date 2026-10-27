# 当前学生与当前特权自教师的直接表征比较

2026-10-27。默认启用 `monitor.compare_student_teacher=true`，补充同一checkpoint内 `S_s` 与 `T_s` 的直接RSA、CKA和近邻重叠率。原有 `S_s` 对 `S_0`、`T_s` 对 `T_0` 指标继续保留。

## 比较输入与采集时机

每个监测点0/1/5/10/20/50，使用更新完成后的同一模型参数，分别读取：

```text
学生：题目 + 冻结的学生回答前缀
教师：reference + 题目 + 同一份冻结回答前缀
```

教师没有另行生成回答。逐层提取完整decoder block的输出，按相同pool、split、problem_id、层与anchor配对。教师的实际token位置由其更长的prompt决定，不能使用学生的绝对位置索引。

主比较使用 `fixed_base_rollout`，即step0生成并冻结的文本。512道heldout与64道train_probe分别计算；可选fixed_reference也按独立pool处理。当前不对变化的live文本计算这组跨题表征指标。anchor为 `prompt_last/response_32/response_128`，几何窗口仍为256，与全回答熵的统计范围独立。

开启直接比较时，OPSD会在**每个optimizer_steps监测点**采集教师几何，即使 `teacher_geometry_steps` 只列出部分步骤。关闭直接比较后，教师采集恢复由 `teacher_geometry_steps` 控制。当前默认两份step列表均为0/1/5/10/20/50。GRPO训练入口保留，本次直接几何分支仅用于OPSD。

## 指标及解释

对N道共同有效题，得到学生与教师矩阵 `H_S,H_T ∈ R^(N×d)`。各自计算题间余弦矩阵 `R_S,R_T`，比较严格上三角：

| metric字段 | 定义 |
|---|---|
| `rsa_rho_student_teacher` | 两个RSM上三角的Spearman秩相关 |
| `rsa_rmse_student_teacher` | 对应RSM元素的均方根差 |
| `linear_cka_student_teacher` | 中心化表征的线性CKA，可由linear_cka开关关闭 |
| `neighbor_overlap_student_teacher_k8/k16/k32` | 原始向量欧氏k近邻集合的交集比例，排除自身 |

这些字段的 `role=student_vs_privileged_teacher`，`comparison=same_checkpoint_student_vs_privileged_teacher`；`rsa_*_to_initial` 保留原含义，不能混用。

先按problem_id严格对齐，ID不匹配报错；随后联合剔除任一侧的零向量或非有限向量。所有直接比较指标使用同一有效题集。`n_valid`是有效题数，`n_pairs`是实际RSA题对数，不是token数。512题通常有130816对，64题有2016对；超过 `rsa_max_pairs=1000000` 时按固定seed采样。常量RSM的rho、常量表征的CKA、N≤k的近邻指标保持undefined；回答未到达anchor时N=0，不补零。

step0虽然参数相同，但学生与教师的输入不同，直接RSA不要求为1。RSA/CKA接近1表示题间关系结构接近，不保证隐藏向量坐标相同或数学能力更好。随训练升高可描述两个上下文条件下结构趋近；reference内容、提示长度与位置变化共同构成该比较条件，不能只归因于推理机制一致。

## 代码与产物

- `analysis/extract.py`：两种上下文采集，复用同一回答，按checkpoint调用配对汇总。
- `analysis/metrics/paired.py`：ID对齐、共同有效题集、直接指标及配对审计。
- `analysis/metrics/rsa.py::compare_rsms`：通用RSM比较；原rsa_to_initial接口保留。
- `analysis/report.py`：自动将新role和metric写入CSV，绘制逐层曲线和层×step热图。

每个 `opsd/analysis/steps/step_XXXXXX/` 新增：

```text
student_teacher_comparisons.json  # 配对anchor、层、两侧缓存key、有效/无效题目ID
student_teacher_neighbors.npz    # 共同有效ID、两侧邻居索引/距离、逐题重叠率
```

数值写入原 `metrics.jsonl` 和可选Parquet；原 `anchors.npz/json`、`rsm_*.npz` 保存双方表征与RSM。近邻索引对应本文件保存的共同有效ID，不能按过滤前题目顺序解释。`complete.json`覆盖新增文件的hash。

## 执行

源码身份已改变，使用新目录；已有旧probe/cache会被拒绝。配置仍为50步、LoRA、lr=1e-6、8题×8轨迹、训练长度2048/16384：

```bash
bash run_base_comparison.sh \
  --model-dir "$OPSD_MODEL_DIR" --model-revision "$OPSD_MODEL_REV" \
  --input "$OPSD_DATA_FILE" --dataset-revision "$OPSD_DATA_REV" \
  --workdir research_runs/base_geometry_50_answer_v2_full_tokens_paired --method opsd --execute
```

环境和质量门槛见[服务器交接](../../PILOT_HANDOFF_zh.md)、[答案协议](ANSWER_PROTOCOL_zh.md)。本地测试涵盖独立RSA/CKA/近邻公式、ID重排、无效题过滤、空anchor、常量输入和真实tiny checkpoint提取；不代表已完成Qwen1.7B GPU实验。
