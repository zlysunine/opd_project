# §3.1：30 元以内的早期验证

**服务器自动准备并训练：项目根目录的 `run_pilot_server.sh`；完整流程见 [服务器交接文档](../../PILOT_HANDOFF_zh.md)。** 新入口会创建隔离环境、自动下载权重，把准备时间计入预算后启动实验。

以下介绍底层离线入口 `pilot_thinking_consistency.sh`。它不修改原始训练脚本，不自动下载模型，不租卡，也不自动关闭云实例。

## 实验内容

- 两组都从 **Qwen3-1.7B-Base** 开始，分别接受 **Qwen3-4B Non-thinking** 和 **Qwen3-4B-Base-GRPO** 的 top-16 OPD。
- 默认 **2 张 80GB 级 GPU**；FP32 主权重/优化器状态，沿用原工作器的混合精度计算，全参数更新。
- 固定抽取 512 条 DAPO prompt；batch=8、rollout=2、最多 30 step，每组实际最多用前 240 个 prompt；回答上限 3072。
- 固定抽取 AMC23 的 64 题，每题 2 次采样，初始学生和两组结果使用相同生成与评分设置。
- 两组单独限时。只比较两组**都完整保存的最大共同步数**，不根据分数选 checkpoint；若某组没有完成 10 step，则不会输出不公平的效果比较。
- 每步记录更新前的 overlap ratio、交集内重新归一化的 overlap-token advantage、学生/教师全词表熵及逐位置绝对熵差；同时记录准确率、截断比例和估算计算费。两组后期轨迹不同，曲线不能解读为固定状态上的直接比较。

这是**单 seed 的缩减实验**。不保证 30 元能看到准确率差距；没有明显差距不能否定论文结论。较多截断、评分超时、OOM、无共同 checkpoint 均标记为结果不足。报告中的 20% 截断阈值只是保守的诊断提示，并非论文标准或统计定理。

## 租卡前准备

先在已有环境中按项目 README 安装本地定制 verl/vLLM/FlashAttention 和数学评分依赖。增加 `psutil` 用于清理本次运行的子进程。这里没有锁定另一套新依赖版本，应复用原项目已验证的环境。

模型必须下载完整到本地，默认目录如下，可用命令参数修改：

| 默认目录 | 模型来源 |
|---|---|
| `model/Qwen3-1.7B-Base` | `Qwen/Qwen3-1.7B-Base` |
| `model/Qwen3-4B` | `Qwen/Qwen3-4B` |
| `model/Qwen3-4B-Base-GRPO` | `Thinking-Space/Qwen3-4B-Base-GRPO` |

使用模型原始的、token ID 兼容的 tokenizer。所有学生 prompt 均采用学生 tokenizer、`enable_thinking=False`，教师在同一学生前缀上评分。脚本检查 token ID 对应关系，不会临时发明 chat template。模型名称相同不能保证下载了正确 checkpoint，请核对来源。

在项目根目录，可以不调用 GPU 地检查模型、准备数据：

```bash
bash pilot_thinking_consistency.sh --prepare-only --output pilot_runs/prepared
```

该命令检查数据和模型；GPU 环境、Hydra 配置和包的完整检查会在正式运行的 prepare 阶段完成。环境安装和模型下载不计入脚本计时，若这些操作发生在付费实例开机期间，要计入 `--already-spent`。

默认保存的是用于评估的模型分片，不保存优化器，不能据此精确恢复训练。为保留两组可比较的共同 checkpoint，需要数十 GB 磁盘；连同模型、合并产物，建议准备约 100GB 可用空间。存储费用另计，优先使用实例自带磁盘。

## 运行

先检查计划（无 GPU、无依赖导入、无文件写入）：

```bash
bash pilot_thinking_consistency.sh --dry-run
```

激活项目的训练环境，假设实际租用的就是两张卡、每卡每小时 5.59 元、之前已经花了 2 元：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash pilot_thinking_consistency.sh \
  --gpus 2 --billable-gpus 2 --price 5.59 \
  --budget 30 --already-spent 2 --reserve 5 \
  --output pilot_runs/thinking_30yuan
```

**`--price` 必须填实际单卡时价，`--billable-gpus` 填全部付费卡数。** 例如租了 8 卡只使用 2 卡，仍必须填 `--billable-gpus 8`；此时 30 元的时间很可能不足以完成实验。

计时上限公式：

```
秒数 = (30 - 启动前费用 - 预留费用) / (付费卡数 × 单卡时价) × 3600
```

若启动前费用为 0，默认预留 5 元，2×5.59 元/小时对应约 **134.2 分钟**上限；prepare、初始评估、两组训练、checkpoint 合并和最终评估都计时。两组训练各分配至多总时限的 28%（默认约 37.6 分钟），其余时间留给准备与评估；提前完成会少花钱。

超时会停止本次创建的进程树，包括 Ray/vLLM 子进程。脚本使用私有 Ray 集群，没有全局 `ray stop --force`。仍应在独占 GPU 作业/容器中运行；不要同时复用同一输出目录。不要把原训练脚本的调试变量 `CUDA_LAUNCH_BLOCKING=1` 带入该实验。

**这是按用户填写单价和卡数估算的运行限额，不是云平台的账单硬上限。**脚本不具有云账户的关机权限：退出后实例仍然收费。准备好文件后再租卡；运行结束立即关机，并在平台另设余额/定时关机限制。若预留不足以覆盖存储、安装或空闲费，总账单仍可能超过 30 元。

可通过 `--student`、`--teacher-nonthinking`、`--teacher-grpo` 指定其他本地路径。完整选项用 `--help` 查看。若要改变 steps，需使它能被 `--save-every` 整除，且 `train-size >= batch*steps`。

## 看结果

输出目录包括：

- `summary.md` / `summary.json`：共同步数、准确率、截断率、按题配对 bootstrap 区间、估算费用、结论。
- `overlap.svg`：可直接用浏览器打开的 overlap 曲线；`overlap_*.json` 是原始曲线点。
- `dynamic_metrics.svg` / `dynamic_metrics.png`：上述五项指标的五联曲线；`dynamic_metrics.csv`、`dynamic_metrics.json` 和 `dynamic_metrics_*.json` 保存逐 step 数值。无 Matplotlib 的精简 CPU 环境只输出 SVG。
- `eval_*.jsonl`：逐条生成结果；`eval_*.json`：完整评估汇总。
- `train_nonthinking.log` / `train_grpo.log`：训练指标与故障信息。
- `config.json`、`data_manifest.json`、`model_metadata.json`、`stages.json`：配置、数据抽样和阶段耗时。

初步支持预期的联合信号是：GRPO 组初始 overlap 更高，随后在相同训练步数的独立题目上也更好。报告采取保守判定；即使置信区间跨零，仍保留原始数据供分析。置信区间不包含训练 seed 不确定性。不要只凭 overlap 更高就宣称推理能力改善。

新增指标默认开启，不增加模型前向；新增的统计时间仍计入原费用上限。advantage 使用交集内分别归一化后的分布，不能拿旧 `val-topk/adv_intersection` 替代；entropy gap 是逐位置 `abs(H_teacher-H_student)` 的平均，不能用两个平均熵相减。padding 不参与统计，空交集从 advantage 均值中排除并记录其比例。详细公式和边界处理见交接文档 §7。

对已有实验，可仅在 CPU 上重建图表：

```bash
bash pilot_thinking_consistency.sh --summarize-only pilot_runs/实际实验目录
```

旧日志中未记录的指标保留为缺失，不能仅靠重新画图补出。服务器需同步新版 `scripts/pilot/` 及定制 `verl/` 的改动，继续使用原一键命令即可采集新指标。

当前仓库环境没有本实验的模型与完整 vLLM 训练依赖，脚本只进行 CPU 可执行的逻辑检查；实际 GPU 显存占用、吞吐、得分及 30 元内能否完整跑完尚未测量。
