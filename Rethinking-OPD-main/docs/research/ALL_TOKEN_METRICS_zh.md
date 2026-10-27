# 全回答分布监测：论文统计口径

2026-10-27。入口为 `run_base_comparison.sh`，配置为 `configs/analysis_deepmath.json`。本次修正新增离线监测的统计范围，OPSD 全词表前向 KL 训练目标保持不变。

## 定义与比较对象

在学生生成回答的每个有效预测位置，学生读取 `(x,y_<t)`，特权自教师读取 `(x,r,y_<t)`。两者评分同一条学生轨迹，教师不独立生成另一条回答。

每个位置计算全词表熵 `H(p)=-sum_v p(v)log p(v)`，自然对数、单位nats。主曲线为 `sum(mask*H)/sum(mask)`，覆盖全部有效回答位置，包含第一个EOS，排除之后的padding；截断但未出现EOS的实际生成token仍计入。提示token不参与平均。

绝对熵差先计算每位置 `abs(H(q)-H(p))` 再平均；有符号熵差为 `H(q)-H(p)` 的平均。两者不能互换。

主要 overlap 的 k=16，与在线一致；额外保存k=10/50/100。`overlap=|TopK(p)∩TopK(q)|/k`。OTA在交集I内分别归一化，计算 `-KL(p_I||q_I)/|I|`；空交集仅从OTA分母排除，并记录 `empty_intersection_rate`。熵和双向KL始终使用全词表。诊断记录两个KL方向，训练目标仍为 `KL(stopgrad(teacher)||student)`。

定义依据原作者[教师熵实现](https://github.com/Thinking-Space/Rethinking-OPD/blob/main/verl/verl/workers/fsdp_workers.py)和[有效token聚合路径](https://github.com/Thinking-Space/Rethinking-OPD/blob/main/verl/verl/trainer/ppo/ray_trainer.py)。本实验为OPSD条件下的迁移分析，不是全部原论文设置的严格复现。

以下分组独立记录，不能混为一条曲线：

| 分组 | 统计对象 |
|---|---|
| 在线训练 | 当前训练batch的学生轨迹、更新前参数 |
| `fixed_base_rollout` | step0生成并冻结的回答，各checkpoint重读相同token IDs |
| `live_rollout` | 各checkpoint重新生成的回答 |
| `heldout_probe` / `train_probe` | 固定512道未训练题 / 固定64道训练集子集；属于train不保证截至该步已抽中 |
| 当前学生 / 当前特权自教师 | 同参数、不同上下文 |
| 当前学生 / 初始学生 | 同一条当前轨迹、相同无特权prompt，不同参数 |

初始模型会重新评分当前轨迹，不能拿初始模型自己的另一条live回答作为相同前缀。相关指标带 `_to_initial` 或名为 `kl_to_initial_nats`；step0的KL为零、overlap为一。在线更新前与离线checkpoint更新后的参数版本不同。live跨步变化包含参数和访问状态变化，fixed可作为对照。

## 长度与内存

`max_probe_response_tokens=256` 仍控制固定回答生成及几何窗口。固定回答只有256个token时，分布统计覆盖这256个实际位置，不代表完整解题过程。

`behavior_max_response_tokens=16384` 控制live回答上限；live分布统计覆盖整条有效回答，**不再被几何窗口256截短**。几何anchor `prompt_last/response_32/response_128` 保留；它们不用于替代全回答分布均值。旧 `distribution_positions=[1,32,128]` 已移除，传入该字段会报错。

| monitor配置 | 默认值 | 含义 |
|---|---|---|
| `distribution_scope` | `all_valid_tokens` | 唯一支持的统计范围 |
| `compare_to_initial` | true | 额外载入初始权重评分当前轨迹 |
| `overlap_top_k` | 16 | 主要overlap/OTA的k |
| `token_top_ks` | [10,16,50,100] | 保存的top-k统计 |
| `state_chunk_size` | 16 | 每个LM head块的预测位置数，仍覆盖全词表 |
| `distribution_forward_chunk_size` | 256 | backbone回答增量块大小 |
| `distribution_prefill_chunk_size` | 256 | prompt预填块大小 |

分块仅改变计算调度。实现使用KV cache，每次只构造 `[state_chunk_size,vocab_size]` 分布，不保存整条 `[response_length,vocab_size]` 数组。初始模型比较额外占用模型权重及KV cache显存，全回答评分比三个选点更耗时。真实Qwen/A800峰值显存和新增耗时尚未实测。

## 代码与输出

- `analysis/trajectory_distributions.py`：因果位置对齐、EOS/mask、分块评分与当前/初始同前缀比较。
- `analysis/distributions.py`：逐位置全词表熵、双向KL、overlap/OTA及分子分母累加。
- `analysis/extract.py`：各pool/split独立采集，跨题按有效token数加权。
- `verl/verl/utils/opd_metrics.py`：已有在线训练诊断，本次核对其数值口径。

逐题 `distributions.jsonl` 和可选Parquet保存均值、`n_valid_tokens`、`metric_sums`、`metric_token_counts`。OTA可有独立有效数量。主 `metrics.jsonl` 的 `aggregation=token_mean`，`n_valid` 为该指标有效token数，另存 `n_questions`。`report/metrics.csv` 和PNG使用这些token加权主曲线。报告中的逐题分位数、题目等权均值描述题间差异，不能替代主曲线。

## 执行与验证

源码/统计身份改变后拒绝旧缓存；旧三点数据无法补算全回答熵。使用新目录重新准备：

```bash
bash run_base_comparison.sh \
  --model-dir "$OPSD_MODEL_DIR" --model-revision "$OPSD_MODEL_REV" \
  --input "$OPSD_DATA_FILE" --dataset-revision "$OPSD_DATA_REV" \
  --workdir research_runs/base_geometry_50_answer_v2_full_tokens_paired --method opsd --execute
```

默认50步，监测0/1/5/10/20/50；保留LoRA、lr=1e-6、8题×8轨迹及训练长度2048/16384。GRPO仅显式指定时执行。环境和质量门槛见[服务器交接](../../PILOT_HANDOFF_zh.md)、[答案协议](ANSWER_PROTOCOL_zh.md)。

CPU测试使用真实tiny GPT-2和tiny Qwen3，对照完整causal forward，验证因果错位、LM head块上限、EOS/padding、全词表熵、空交集OTA、跨题token加权与初始模型同轨迹比较；不代表真实1.7B GPU实验已完成。
