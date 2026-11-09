# 实验一：按顺序运行

目标：Qwen3-1.7B 无特权学生，对比同一当前模型加离线参考解答的 stop-gradient 自教师；DeepMath-103K 的 `5 <= difficulty <= 7`。使用 non-thinking 模板，`enable_thinking=False`。不启动独立 4B teacher，不运行已取消的 §3.2 迁移。以下命令均从 `Rethinking-OPD-main/` 执行。

2026-11-09：旧shell包装已删除。本页保留历史单组工作流，直接使用 `python -m scripts.research.pipeline run`；当前Base三组对照请使用 [新交接文档](../../CONTROLLED_COMPARISON_HANDOFF_zh.md) 中的 `run_controlled_comparison.sh`。默认只打印计划，`--execute` 才执行；原 `on_policy_distillation.sh`、`grpo.sh` 保留。

以下命令均从项目根目录执行，先设置 `export PYTHONPATH="$PWD:$PWD/verl${PYTHONPATH:+:$PYTHONPATH}"`。

## 0. 准备环境与输入（一次）

需要完整的本地 Qwen3-1.7B 权重、tokenizer/chat template，以及原始 DeepMath-103K parquet 文件/分片目录或 JSONL。数据必须包含 `question`、`final_answer`、`difficulty`、`r1_solution_1`。仓库自带的 `datasets/DeepMath_deduped.parquet` 缺少 difficulty 和参考解答，不能用于本实验。模型和数据 revision 应填写实际不可变 commit 或有内容身份依据的本地版本。

已有兼容训练环境可直接进入第 1 步。否则可按仓库的安装脚本准备专用环境：

```bash
conda create -n opsd python=3.12
conda activate opsd
cd verl
USE_MEGATRON=0 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh
python -m pip install --no-deps -e .
cd ..
python -m pip install "numpy<2" "transformers==4.56.1" "pyarrow>=19" \
  matplotlib scipy scikit-learn psutil math-verify latex2sympy2_extended
```

该安装脚本采用 vLLM 0.11.0 和面向 Python 3.12 / CUDA 12 / Torch 2.8 的 FlashAttention wheel，需要与机器的驱动及 ABI 匹配。这里未在 GPU 上验证这组安装命令；以第 2 步实际检查为准。安装成功后保留本次运行的 `pip freeze`，后续复跑使用同一环境。

仅测试 CPU 连接可以先执行以下命令，不需要真实权重或 DeepMath；它使用随机 tiny GPT-2，不是 Qwen 实验结果：

```bash
python -m unittest discover -s tests/research -v
python -m scripts.research.pipeline smoke --workdir research_runs/my_cpu_smoke
```

## 1. 生成本次运行的独立配置

将下面四个模型/数据参数替换为真实值。`OPSD_WORKDIR` 必须指向本次实验的新目录；之后各阶段沿用它。

```bash
export CUDA_VISIBLE_DEVICES=0
export OPSD_WORKDIR="$PWD/research_runs/consistency_lora_fkl_500"

python -m scripts.research.pipeline configure \
  --model-dir /actual/local/Qwen3-1.7B \
  --model-revision ACTUAL_MODEL_COMMIT \
  --input /actual/local/DeepMath-103K/data \
  --dataset-revision ACTUAL_DATA_COMMIT \
  --profile pilot
```

执行 `scripts/research/pipeline.py`，生成 `$OPSD_WORKDIR/config.json` 和私有 `models.json`，无需手改全局模型注册表。当前默认单 GPU、LoRA rank 16 / alpha 32 / all-linear、学习率 1e-6、500 updates、batch 8 道题、每题 8 rollouts（64 条轨迹/step），全词表前向 KL 损失。学生 prompt/response 上限各 512，教师 prompt 上限 2048，每 50 updates 保存 checkpoint。512 response tokens 会限制解题，不能直接作为正式能力结论。

如需更长上下文，在本步骤显式设置 `--max-prompt-length`、`--max-response-length`、`--max-teacher-prompt-length`。配置和模型身份一旦写入，改变实验应换工作目录，避免旧数据和 checkpoint 混入。

## 2. 检查环境和实际训练配置

```bash
python -m scripts.research.pipeline check
```

执行 `scripts/research/preflight.py`：检查训练依赖、Qwen3 所需 Transformers 版本、NumPy 约束、可见 CUDA 数量、tokenizer，以及实际 Hydra overrides 与 rollout dataclass。报告写入 `preflight.json` 和 `check_plan/*preflight.log`，这一步不启动 Ray 训练。失败时先修复报告中的具体问题。`--cpu-only` 仅跳过 CUDA 检查，仍检查完整训练依赖；CPU smoke 使用第 0 步的独立命令。

## 3. 过滤 level、检查长度、生成配对划分

```bash
python -m scripts.research.pipeline prepare --overlength drop --max-problems 5000
```

执行 `pipeline.py → data.py`，先按 difficulty 5–7 过滤，再用真实 tokenizer 检查学生和完整特权 prompt 长度。这里显式选择丢弃超长题，再固定抽取至多 5000 题；不截断 reference。`data/prepare_manifest.json` 保存各步覆盖率、输入/输出哈希和选中 ID，`data/lengths.jsonl` 保存实际长度。solution 和 empty 的 train/validation/test 使用完全相同的问题 ID。划分由独立 `data_seed=42` 控制，改变训练 seed 不改变题集。

默认不加 `--overlength drop` 时，任何超长题都会报错。当前 500 steps × batch 8、1 epoch 需要至少 4000 道训练题；5000 个独立实例按默认划分可得到 4000 道训练题。若保留题太少，prepare 会在初始评测前拒绝；应换新目录增加数据/长度预算或明确调整实验步数。不要复用旧 128 题缓存。若取消抽样上限，需报告完整覆盖率。

## 4. 评测未训练基线 A

```bash
python -m scripts.research.pipeline evaluate --which initial --split validation --samples 16
```

执行 `run.py → models.py`，不向模型传 reference。默认 response 上限沿用训练配置；初始和训练后评测保持同题集、采样次数、长度和温度。保存 `eval/solution_initial_0_validation.jsonl`、对应 summary，以及 `train/evaluations.jsonl` 的 step 0。每题 16 次采样，分别记录 avg@16 与 pass@k。

## 5. 预览并启动主实验 C

```bash
python -m scripts.research.pipeline train
python -m scripts.research.pipeline train --execute
```

第一条只写计划；第二条才训练。一键入口已包含该阶段，使用一键流程时无需额外执行。

实际调用顺序：`python -m scripts.research.pipeline train → pipeline.py → launch.py → verl.trainer.main_ppo → RayPPOTrainer → ActorRolloutRefWorker.compute_opsd_teacher → actor 的 full-vocabulary forward-KL loss/backward`。

学生 rollout 只读取题目；自教师在本 batch 更新前，以当前 actor 和相同 response prefix 加固定 reference 计算 logits，并停止梯度。训练前再次检查环境、精确长度、模型 context 和完整 batch 数。当前目标已改为全词表前向 KL，直接对学生求梯度；原 OPD loss 分支仍用于其他历史配置。完整训练输出写入 `train/train.log`，另一个终端可执行：

```bash
tail -f "$OPSD_WORKDIR/train/train.log"
```

根 `on_policy_distillation.sh` 已恢复为原仓库外部 OPD 脚本，与 HEAD 逐字一致；保留其原有模型/设备参数，不再充当最新 OPSD 的入口。

## 6. 查看五项训练诊断曲线

训练成功结束时自动出图；训练过程中或结束后也可重画：

```bash
python -m scripts.research.pipeline plot
```

原始标量为 `train/metrics.jsonl`；`train/plots/thinking_consistency.csv` 和 `.png` 保存 overlap ratio、Overlap-Token Advantage、student entropy、teacher entropy、absolute entropy gap 五项曲线。它们均来自优化前的 rollout batch；entropy 为 full-vocabulary entropy，空交集 advantage 缺失而不是伪记零。参数版本和预算计数也随日志保存。精确定义见 [METRICS_zh.md](METRICS_zh.md)。

## 7. 合并训练 checkpoint

```bash
python -m scripts.research.pipeline merge
```

执行 `python -m verl.model_merger merge --backend fsdp`，默认选择 tracker 中最新完整 checkpoint，先检查各 rank 权重及 HF config。输出为 `hf/solution/global_step_N/`。选中间阶段可用 `--step 50`，前提是该步已保存完整 checkpoint。

## 8. 评测训练后学生，更新性能图

```bash
python -m scripts.research.pipeline evaluate --which trained --split validation --samples 16
python -m scripts.research.pipeline plot
```

需先完成第 7 步合并。默认选择与合并相同的最新 checkpoint；使用 `--step 50` 时 merge 和 evaluate 都要指定该步。validation 分数关联到实际训练 step 和预算，保存于 `train/evaluations.jsonl`。LoRA 合并时自动恢复训练 alpha，并将 adapter 合入完整模型；未确认合入时拒绝评测基座。若要阶段性能曲线，针对每个已保存 checkpoint 重复“merge --step N → evaluate --step N”，最后 plot。

完成 checkpoint 选择后再做最终 test：

```bash
python -m scripts.research.pipeline evaluate --which trained --split test --samples 16
```

test 分数单独保存，不放入 validation 学习曲线。所有这些评测均为无特权学生评测。

## 可选：空特权对照 B

沿用同一工作目录和第 3 步的配对数据，在同一张卡依次运行：

```bash
python -m scripts.research.pipeline train --condition empty --execute
python -m scripts.research.pipeline merge --condition empty
python -m scripts.research.pipeline evaluate --condition empty --which trained --split validation --samples 16
python -m scripts.research.pipeline plot --condition empty
```

该对照使用独立 `train_empty/`、`hf/empty/`、`eval/empty_*`，不覆盖主实验。若需要它自己的 step 0 曲线，在训练前另执行 `evaluate --condition empty --which initial`。

## 正式运行与复现材料

正式配方在新目录使用 `configure --profile paper`：默认 8 GPU、response 上限 7168、教师 prompt 上限 8192，继承相同 LoRA/前向 KL、batch 8、每题 8 rollouts、500 steps、1 epoch；需相应设置可见 GPU。这是当前实验的长上下文配置，不是原论文原配方。不要沿用旧 128 题上限。单次 configure 只启动一个训练 seed；多个 seed 使用不同工作目录和 `--seed 42/43/44`，保持相同 `--data-seed 42`、数据 revision 与长度规则。额外上下文及完整词表诊断会增加显存，GPU 显存和吞吐仍需实测。

每次保存整个工作目录，尤其是 config/models、数据 manifest 和划分、训练 resolved config/provenance、preflight、预算/标量日志、checkpoint、合并来源、评测明细/summary 与图表。当前机器只完成 CPU 测试和 tiny smoke；真实 Qwen/DeepMath 的 Ray/vLLM/FSDP 训练尚未执行，详见 [VALIDATION_zh.md](VALIDATION_zh.md)。
