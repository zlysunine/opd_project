# OPD / OPSD / SFT / GRPO 研究管线

**2026-10-28服务器重跑：** 以重写的[服务器交接](../../PILOT_HANDOFF_zh.md)为执行依据，先校验`SERVER_HANDOFF_MANIFEST.json`，再做独立2步工程验收和50步正式运行。旧版交接已归档，不再复制旧200/500步命令。

**2026-10-27 指标清单与HTML：** 按“当前学生—初始学生”和“当前学生—当前教师”梳理记录项，并区分单模型统计。报告流程自动生成单文件离线HTML，涵盖新旧统计口径、直接RSA/CKA、曲线、热图与RSM图库。见[实验指标清单与完整HTML](METRIC_RECORDING_zh.md)。

**2026-10-27 学生—教师直接表征比较：** 默认在0/1/5/10/20/50步计算固定题集上同一checkpoint的RSA、CKA、kNN overlap，独立role为`student_vs_privileged_teacher`，同时保留与step0的比较。实现、输出字段和运行方法见[学生—教师几何指标](ST_GEOMETRY_zh.md)。

**2026-10-27 分布统计修正：** 离线fixed/live对全部有效回答位置计算全词表熵，按token加权；同步记录top-16 overlap/OTA、双向KL和当前/初始同轨迹比较。1/32/128分布选点已移除，几何anchor保留。见[全回答指标说明](ALL_TOKEN_METRICS_zh.md)。

**2026-10-27 当前答案协议为v2：** 统一Base提示、抽取与判分，隔离不明A/True映射；行为评测16384 tokens、几何窗口256、heldout抽取失败率门槛1%。默认50步、新目录`research_runs/base_geometry_50_answer_v2_full_tokens_paired`，用户设置的训练长度2048/16384保留。先读[答案协议与运行说明](ANSWER_PROTOCOL_zh.md)。下文200步等数字保留为历史设计，不能替代当前配置。

**2026-10-26 本轮主入口：Qwen3-1.7B-Base 的 OPSD / GRPO 表征轨迹比较。** 开发与运行说明见 [BASE_GEOMETRY_zh.md](BASE_GEOMETRY_zh.md)，一键脚本 `run_base_comparison.sh`，配置 `configs/analysis_deepmath.json`。保持 LoRA/前向 KL/current-batch self-teacher，同时独立记录 RSA、曲率、kNN、参与率、固定状态分布和 live 行为。两组共享数据及固定 probe。以下保留既有 non-thinking 和其他研究接口；不要将它们的模型起点与本轮 Base 混用。

**当前先跑 OPSD，共50个optimizer step。** 一键脚本及 `compare run/train/analyze` 默认 `--method opsd`：记录 OPSD 在线指标、分析 checkpoint 表征并生成报告。GRPO 的设计、配置和训练接口均保留，只有显式传入 `--method grpo` 或 `--method both` 才执行。

本交付扩展现有训练设施。2026-10-26 按用户要求，实验一改为 LoRA 和全词表前向 KL；没有加入表征蒸馏、PreSO 或额外正则。生产训练复用本仓库 verl / LLaMA-Factory；CPU tiny smoke 仅测试连接与不变量。审计、原论文设置和代码差异见 [AUDIT_zh.md](AUDIT_zh.md)，指标定义见 [METRICS_zh.md](METRICS_zh.md)，验收记录见 [VALIDATION_zh.md](VALIDATION_zh.md)。

所有命令从 `Rethinking-OPD-main/` 运行。CLI：

**服务器交接与可复制给 Codex 的任务说明见 [PILOT_HANDOFF_zh.md](../../PILOT_HANDOFF_zh.md)**。本轮入口为 `run_base_comparison.sh`，默认OPSD only：环境检查 → 冻结数据与probe → OPSD训练及在线指标 → checkpoint表征分析 → 报告；文档提供2-step验收和200-step一键命令、产物核对与失败恢复。不传 `--execute` 只写配置和打印计划。

旧版non-thinking实验仍由 `run_opsd_experiment.sh` / `opsd_pipeline.sh` 执行，其分阶段记录见 [PIPELINE_zh.md](PIPELINE_zh.md)。根 `on_policy_distillation.sh` / `grpo.sh` 保留原仓库用途。下面保留其他研究接口，不能将旧模型/旧CLI参数混入本轮Base入口。

```bash
python -m scripts.research.run --help
python -m scripts.research.run train --help
```

## 1. 旧版 non-thinking 实验分组（非当前 Base 入口）

实验一改为 **OPSD 中学生与特权自教师的思考模式一致性验证**，固定使用 Qwen3-1.7B。默认初始化为 `Qwen/Qwen3-1.7B`，沿用 non-thinking 模板（`enable_thinking=False`）；这里的“思考模式”指学生访问状态上的分布兼容性，不预设共享权重就一致。§3.2 “New Knowledge, Not Just Scale” 的迁移分组已取消。

| 分组 | 学生 | 提供 logits 的教师 | 特权信息 | 数据 |
|---|---|---|---|---|
| A：初始基线 | Qwen3-1.7B，未训练 | 无 | 无 | DeepMath-103K，level 5–7 |
| B：空特权对照 | 当前 Qwen3-1.7B | 当前 Qwen3-1.7B + 空 reference 字段 | empty | 与 C 完全相同的问题 ID |
| C：主实验 | 当前 Qwen3-1.7B | 当前 Qwen3-1.7B + 参考解答 | solution，默认数据集 `r1_solution_1` | DeepMath-103K，level 5–7 |

研究组名为 `exp1_opsd_consistency`。仅 C 是主训练条件，A/B 用于解释初始差异及额外 prompt 字段的影响，不默认展开 4B 学生、独立 4B logits 教师或不同 RL 教师来源。

学生无特权 rollout：`y ~ pi_theta(.|x)`。在完全相同的 `y_<t` 上比较学生 `p_t=pi_theta(.|x,y_<t)` 与自教师 `q_t=stopgrad(pi_theta(.|x,r,y_<t))`。`current_batch_stop_gradient` 在本 batch 优化前使用当前 actor，下一批重新计算；教师不重新生成另一条 response，也不独立优化。参考解答来自固定数据集，不需要额外 4B privilege generator。当前只训练 LoRA（rank 16 / alpha 32 / all-linear），基座冻结；loss 是全词表 `KL(q || p)`，lr=1e-6，batch 8 题 × 8 rollouts，500 步。top-16 只用于五项诊断，和 loss 支持集分开。

每个 rollout batch 保存五项更新前诊断（有效 response token 平均）：

| 日志名 | 曲线 |
|---|---|
| `opd/overlap_ratio` | 学生/教师各自 top-16 的交集比例 |
| `opd/overlap_token_advantage` | 交集内分别归一化，再额外除以交集大小；空交集记 null |
| `opd/student_entropy` | 学生 full-vocabulary entropy |
| `opd/teacher_entropy` | 特权自教师 full-vocabulary entropy |
| `opd/entropy_gap` | 每位置两者熵的绝对差，再平均 |

训练目录的 `metrics.jsonl` 持久化这些值和参数版本。绘图保存 `thinking_consistency.csv`、`thinking_consistency.png` 五联曲线，以及各指标随 step/优化 tokens/wall time 的独立图。step 对应 rollout batch，诊断在该 batch 的参数更新之前；不同 checkpoint 的 on-policy 前缀可能不同，可另用固定前缀诊断排除访问分布变化。overlap 高、advantage 接近 0、entropy gap 小可作为分布一致性的证据，不能单凭这些量断言语义推理机制完全相同。

**实验二保留**：比较相同初始化经 SFT/RL/OPSD 后的能力与内部表征。它与已取消的 §3.2 迁移分支是不同实验。

| 实验二参数状态 | 无特权评测 | 同一份 reference 的有特权评测 | 参数训练时的监督 |
|---|---|---|---|
| M0 | M0 | M0+r | 无 |
| M_SFT | M_SFT | M_SFT+r | reference 作为 assistant targets |
| M_RL | M_RL | M_RL+r | verifier 的 outcome reward |
| M_OPSD（包括各阶段 checkpoint） | M_OPSD | M_OPSD+r（该参数状态的自教师） | 学生轨迹上的特权 logits |

实验二矩阵默认在各规模内以同一 Qwen3 non-thinking checkpoint 出发；若改成 Base，所有方法一起修改注册表/配置。同一问题集合上的监督信息不同，不能把相同 step 当相同预算。有特权评测始终另标，不与无特权学生放在同一性能曲线。

## 2. 环境、资源与模型

先按项目 README 安装**本地定制 verl**，不要用 pip 上另一个版本替代。SFT 使用项目的 LLaMA-Factory 环境。生产 Qwen3 必须 Transformers ≥4.51。分析依赖清单：

```bash
python -m pip install -r scripts/research/requirements-analysis.txt
```

该命令是人工安装入口，不会在训练过程中自动执行。生产 verl 有自己的 NumPy<2 等依赖约束，优先遵循 `verl/requirements.txt` 和已有安装脚本，避免把当前 CPU 环境直接当作 GPU 训练环境。软件实际版本记录在每次运行的 provenance 中。

填写 `configs/research/models.json` 各模型的本地 `path` 与不可变 `revision`。官方 ID 已核对，**本地权重尚不存在**，不会自动下载，也不会默认把 `main` 当固定版本。当前实验一只需 Qwen3-1.7B；实验二按所选条件准备模型。注册表中的旧教师条目只供通用 OPD 入口使用，不属于当前实验矩阵。

如果 Base checkpoint 没有 chat template，在注册表该模型中增加 `chat_template_path`，指向核实来源的模板文本；记录来源与 revision。launcher 将模板同时传给 driver 和 actor，防止数据与教师模板分叉。独立评测/采集对应传 `--template`。SFT 使用 LLaMA-Factory 的 `qwen3` reasoning template，`enable_thinking=False`；自定义模板若不同于该模板，应先核对模板 token IDs，作为设置差异记录。

本地路径可以是已有 HF checkpoint。verl 分片需先合并；LLaMA-Factory 的 HF checkpoint 可直接评测。本轮不自动下载权重，不租卡，不运行全部矩阵。

| 档位配置 | 目标 | 明确缩减 / 限制 |
|---|---|---|
| smoke.json | CPU random tiny GPT-2 | 每 split 4 题、2 更新、8 response tokens，仅验证程序 |
| pilot_1p7b.json | 单 GPU，1.7B LoRA / full-vocab forward KL | batch 8，n=8，500 steps，lr=1e-6；prompt/response 512、teacher prompt 2048 |
| pilot_4b.json | 实验二，单 A800 80GB，4B | batch 2，n=2，response 256，teacher prompt 1536，10 step |
| paper.json | 8 卡长上下文 LoRA / forward KL | 继承 batch 8、n=8、500 steps；prompt 1024、response 7168、teacher prompt 8192；seeds 42/43/44 |

pilot 的微批为 1、累积更新，启用 activation checkpoint/offload 和 optimizer offload。**未在 A800 实测 OOM 或速度**。paper 的微批、OPSD teacher 上下文预算和额外 forward 是扩展设置；长度/批量变化会改变实验。原论文 GRPO teacher 训练 n=8，若专门重训该 teacher，应在 RL 配置将 `rollouts` 改 8；实验二的预算匹配 RL 配置默认沿用研究配置的 n。

## 3. 数据准备

### 自动生成受控技能

```bash
python -m scripts.research.run prepare \
  --output research_data/controlled.jsonl --count 64 --seed 42 --modulus 11 --steps 3
```

包含 train、validation、test_instance、test_template、test_long。跨 split 的操作实例唯一；test_instance 使用训练模板的新实例，validation/test_template 使用不同模板，test_long 加倍步骤。`analysis.intermediate_states` 仅是元数据，学生 messages 不包含状态标签或 oracle_solution。oracle_solution 仅用于显式 oracle reference / 固定诊断前缀。

### 实验一：DeepMath-103K level 5–7

[官方数据说明](https://huggingface.co/datasets/zwhe99/DeepMath-103K)的原始字段包括 `question`、`final_answer`、连续难度分数 `difficulty` 和三份 R1 解答。level 5–7 在本实验严格定义为 **5 ≤ difficulty ≤ 7**，不把 7.8 四舍五入进范围。

**现有 `datasets/DeepMath_deduped.parquet` 只保留 prompt、答案等元数据，footer 中没有 difficulty/level，也没有 R1 solution 字段**，无法确认其 level 组成。不能拿它静默替代；需要本地原始 DeepMath-103K（保留上述字段的 parquet 文件/分片目录或 JSONL），也不会自动下载整个数据集。

先填真实数据目录、不可变数据 revision（并将相同 revision 写入配置的 `dataset_revision`），然后执行：

```bash
export DEEPMATH_INPUT=/absolute/path/to/native/DeepMath-103K/data
export DEEPMATH_REV=FILL_IMMUTABLE_DATASET_REVISION
python -m scripts.research.run prepare-deepmath \
  --input "$DEEPMATH_INPUT" --revision "$DEEPMATH_REV" \
  --level-min 5 --level-max 7 --solution-field r1_solution_1 \
  --privilege-type solution --output research_data/deepmath_l5_7/solution.jsonl
python -m scripts.research.run prepare-deepmath \
  --input "$DEEPMATH_INPUT" --revision "$DEEPMATH_REV" \
  --level-min 5 --level-max 7 --privilege-type empty \
  --output research_data/deepmath_l5_7/empty.jsonl
python -m scripts.research.run partition \
  --input research_data/deepmath_l5_7/solution.jsonl \
  --output research_data/deepmath_l5_7/solution --seed 42
python -m scripts.research.run partition \
  --input research_data/deepmath_l5_7/empty.jsonl \
  --output research_data/deepmath_l5_7/empty --seed 42
```

同一 seed、同一问题集合的 partition 使用同一 ID 分组，两个条件得到相同 train/validation/test 划分。数据难度、dataset_id/revision 和 reference 字段在训练启动前再次检查，越界或缺失则报错。manifest 保存筛选前后覆盖率、难度分布、闭区间与不舍入规则。

特权参考默认直接使用 `r1_solution_1`，不在训练中动态生成；也可显式选择 final_answer 作为后续消融。它是数据集参考文本，不是独立外部 logits teacher，不伪造其生成模型 revision。参考的正确性默认未知，不声称逐条验证通过。`reference_token_length` 可用 `prepare-deepmath --tokenizer /actual/local/Qwen3-1.7B` 计算；未提供 tokenizer 时非空参考长度为 null，另保存字符长度。

学生 messages 只有题目和统一的 boxed 输出指令，不包含 reference 或最终答案。metadata 中的 difficulty、ground_truth、reference 不会拼进学生 prompt；只有自教师读取 reference。

### 其他数据与实验二

统一 JSONL、原有 reference 生成、交集筛选、controlled-skill 入口继续可用。实验二可复用上述同一问题集合，用 `reference --model ... --revision ... --generator-id Qwen/Qwen3-4B` 离线生成其固定 SFT/OPSD 参考；这类 model-generated 缓存与实验一的 dataset reference 分开记录。需要原论文 external OPD 时可使用通用 launcher 和独立配置，当前默认矩阵不再生成它们。

## 4. 训练入口与矩阵

默认只写配置和命令，不分配 GPU；只有单个 `train --execute` 启动一次实验。

```bash
python -m scripts.research.run matrix \
  --config configs/research/pilot_1p7b.json --groups exp1_opsd_consistency \
  --output research_runs/consistency_matrix --seeds 42
python -m scripts.research.run matrix \
  --config configs/research/paper.json --output research_runs/paper_matrix
```

实验一单 seed 仅 3 个配置（A/B/C）。默认完整矩阵为实验一 3 项 + 保留的实验二 10 项，共 13 项/seed；paper 的 3 seeds 共 39 项。`exp1_3p2` 不再被接受。实验二另写 `evaluation_matrix.jsonl`，比较参数状态×有/无上下文。

完成原始数据准备，并在 `models.json` 填写 `qwen3_1p7b_nonthinking` 的真实本地 path/revision 后：

```bash
python -m scripts.research.run train --config configs/research/pilot_1p7b.json
CUDA_VISIBLE_DEVICES=0 python -m scripts.research.run train \
  --config configs/research/pilot_1p7b.json --dataset-revision "$DEEPMATH_REV" --execute
```

第一条只生成 plan，第二条执行实验一主条件 C。运行空特权 B：

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.research.run train \
  --config research_runs/consistency_matrix/exp1_opsd_consistency__1.7B__nonthinking__opsd__none__empty__current_batch_stop_gradient__42__lora_r16__forward_kl_full_vocab.json \
  --dataset-revision "$DEEPMATH_REV" --execute
```

每批都会计算并持久化 overlap、advantage、两条 entropy 和绝对 entropy gap；没有交集的 advantage 保存 null，不补 0。训练后运行：

```bash
python -m scripts.research.run plot --input research_runs --output research_runs/plots
```

查看 `thinking_consistency.png` 与 `thinking_consistency.csv`。若只想画这一次实验，`--input` 直接填该 run 目录。后文 checkpoint 评测与表征采集入口保持不变。

实验二用默认完整 matrix 中的 `exp2__...json`，SFT/RL/OPSD 分别选择对应 method 的配置；4B 单卡目标配置为 `pilot_4b.json`，属于实验二。根 `on_policy_distillation.sh` 与 `grpo.sh` 均与 Git HEAD 原版一致；最新 OPSD 使用独立入口。新 launcher 仍复用相同底层训练路径。实验一配置明确拒绝独立 external teacher 或换为 4B 学生。

每次 run 保存 resolved config、完整 verl 配置、Git/源代码/数据 hash、模型与数据 revision、JSONL、训练 log 和 checkpoint。预算上限 `budget_optimized_tokens`、`budget_rollout_tokens`、`budget_wall_seconds` 仍在完整更新边界检查；相同 step 不代表相同预算。单卡长度和 batch 缩减会改变设置，GPU 显存尚未实测。

## 5. 已有 checkpoint 评测

```bash
PYTHONPATH="$PWD/verl" python -m verl.model_merger merge --backend fsdp \
  --local_dir /absolute/path/to/run/checkpoints/global_step_5/actor \
  --target_dir /absolute/path/to/merged_hf_step5
python -m scripts.research.run evaluate \
  --model /absolute/path/to/merged_hf_step5 --revision local:run-step5 \
  --input research_data/deepmath_l5_7/solution/test.jsonl --samples 16 --max-tokens 7168 \
  --device cuda --dtype bfloat16 --output research_runs/eval_step5.jsonl
```

输出逐次采样与 summary/manifest。avg@16 是每题 16 次正确比例，再对题目平均、跨 benchmark macro；pass@16 是至少一次成功的估计，单独命名。可用 `--pass-k 1 4 16`。不重训已有 checkpoint。评估学生默认无特权；加 `--privileged` 是独立有信息条件。

用于画验证曲线时加 `--split validation --training-run /absolute/run --step 5`，准确关联已记录预算；不存在该 step 会报错，额外写 evaluations.jsonl。测试不能用于 checkpoint 选择。

训练前先用 controlled 的 validation 跑同一评测；summary 的 controlled_skill_calibration 会指出全对/全错。可调整 modulus、steps、模板或选择技能，不能用 tiny 随机模型的全错结果推断 Qwen 技能难度。

性能匹配 checkpoint 输入为一行一个验证结果的 JSONL：
`{"split":"validation","checkpoint":"...","step":5,"optimized_tokens":10000,"accuracy":0.4}`。

```bash
python -m scripts.research.run select-checkpoint --input research_runs/validation_checkpoints.jsonl \
  --target 0.4 --tolerance 0.02 --max-tokens 100000 --output research_runs/selected.json
python -m scripts.research.run gap-recovery --before 0.2 --after 0.3 --teacher 0.5 \
  --teacher-checkpoint fixed_reference_checkpoint --output research_runs/recovery_example.json
```

上面 recovery 数字仅是 CLI 格式示例，不是实验结果；有特权教师必须再传 `--privileged`。非正/接近零分母输出不适用。

## 6. 固定前缀、分布诊断、表征与绘图

先设置 `QWEN17_PATH`、`QWEN17_REV` 为实际 Qwen3-1.7B checkpoint，再用它生成诊断 response IDs，此后所有模型读取相同文件。数据中包含同一份 reference，默认学生仍不读取它：

```bash
python -m scripts.research.run evaluate \
  --model "$QWEN17_PATH" --revision "$QWEN17_REV" \
  --input research_data/deepmath_l5_7/solution/validation.jsonl \
  --samples 1 --pass-k 1 --max-tokens 512 --fixed --device cuda --dtype bfloat16 \
  --output research_runs/fixed_prefixes.jsonl
python -m scripts.research.run diagnose \
  --model "$QWEN17_PATH" --revision "$QWEN17_REV" \
  --input research_runs/fixed_prefixes.jsonl --privileged --ks 8 16 32 \
  --diagnostic-temperature 1 --chunk 16 --device cuda --dtype bfloat16 \
  --output research_runs/diagnostics.jsonl
```

该 diagnose 是同 checkpoint 的 self-teacher（无/有 reference）。external teacher 改传 `--teacher-model ... --teacher-revision ...`，不加 privileged。采样温度与诊断温度独立保存。hidden forward 后以 chunk 做 LM head/full-vocab 指标，只保存 scalar，不写完整 logits 张量。

采集以下两个条件完全相同 response token prefix：

```bash
python -m scripts.research.run extract \
  --model "$QWEN17_PATH" --revision "$QWEN17_REV" --device cuda --dtype bfloat16 \
  --input research_runs/fixed_prefixes.jsonl --layers 0 10 20 --hook block_output \
  --strategy relative --bins 5 --cap 1024 --output research_runs/activations_plain
python -m scripts.research.run extract \
  --model "$QWEN17_PATH" --revision "$QWEN17_REV" --device cuda --dtype bfloat16 \
  --input research_runs/fixed_prefixes.jsonl --layers 0 10 20 --hook block_output \
  --strategy relative --bins 5 --cap 1024 --privileged --output research_runs/activations_privileged
python -m scripts.research.run analyze \
  --a research_runs/activations_plain --b research_runs/activations_privileged \
  --neighbors 5 --clusters 4 --cap 2048 --seed 42 --output research_runs/analysis
python -m scripts.research.run plot --input research_runs --output research_runs/plots
```

跨 checkpoint 用相同 fixed_prefixes，换 model/path/revision 和输出目录即可。当前 on-policy 轨迹可用当时模型的 evaluate 产物，trajectory_source 会区分；不可把不同 on-policy 轨迹强行当作跨模型固定状态对齐。

位置策略：`last_prompt_token`、`fixed --positions 0 8 16`、`relative --bins 5`、`annotated`。默认 `predict_current` 在 P−1+t 提取；`after_current` 在 P+t，二者缓存和对齐键不同。annotated 输入需 `step_token_positions: [{token_index,label,intermediate_state}]`，位置必须对应当前固定 response，不能直接把 oracle 步数标签贴到模型乱生成的 token 上。

hook 支持 attention_output（投影后、不含 residual）、block_output（含 residual、最终 LN 前）、post_mlp（MLP 输出、不含 residual）。目前明确适配 Qwen decoder 和 tiny GPT-2，未知模型结构拒绝猜测。

每问题一 NPZ shard，JSONL 保存 checkpoint/config/template/prefix hash、问题、位置、hook、标签、轨迹方式；相同身份可恢复，模型/模板/前缀/配置变化必须换目录。`--cap` 是所选层×位置的总 activation 行数上限；RSA/CKA/近邻另有每组 N 上限，默认 2048，避免 O(N²) 爆炸。

变化向量与子空间：在 analyze 加 `--baseline-a INITIAL_PLAIN --baseline-b INITIAL_PLAIN --rank 2`；A/B 可以分别是特权状态、SFT/RL/OPSD 后状态，以同一初始化为基线。特权 shift 则 B 的基线应取对应参数状态的 plain。默认 uncentered SVD；显式 `--centered-pca` 才做中心化。秩不足输出原因，不造高 overlap。

跨规模需显式 `--cross-scale`，只比较同样本的 RSA、CKA、近邻结构；coordinate cosine/变化向量子空间被禁用。层对应只支持用户明确选择的共同 layer index，不自动把不同深度当作等价层。

`--probe` 使用真实 annotated intermediate_state，按 probe_split 与 problem/instance/template 同时检查隔离，验证集选择轻量线性探针 C，测试报告多数类基线和类别分布。默认受控数据应选 train/validation/test_template；test_instance 与训练模板重叠，会被严格模板 probe 拒绝。探针可读出信息不代表模型因果使用该信息。

图包括验证性能对 step/优化 tokens/wall time、动态分布指标、相对位置指标、层×checkpoint 比较、按 skill/state/answer 排序的 RSA、K-means ARI。没有输入数据就不伪造图。不同诊断 run 独立汇总，CSV 同时保留 equal-question/token-weighted 口径。

关联图可提供 `research_runs/associations.jsonl`：每行一个 run/task，包含 run_id、task、stage（pretraining_predictor/early_predictor/posttraining_correlation）、analysis_split（fit/validation）、representation_metric、unprivileged_generalization_gain。程序拒绝同一 run 混入拟合与验证，不把 token 当独立 run；这只是明确时间阶段的关联可视化，未提供默认预测器或因果结论。

## 7. Smoke 与验收

```bash
python -m unittest discover -s tests/research -v
python -m unittest discover -s verl/tests/utils -p 'test_opd_dynamic_metrics_on_cpu.py' -v
python -m scripts.research.run smoke --output research_runs/new_smoke
```

旧两教师 pilot 的脚本和专用测试已在用户确认后清理，当前测试命令只针对保留的研究管线和动态指标。已完成的产物目录 `research_runs/smoke_delivery` 不会被 smoke 覆盖。输出有真实 tiny checkpoint、16 次采样、diagnostics、四组 activations、探针、analysis、图和 smoke_result/provenance。

必须区分：CPU smoke 与指标测试已运行；Ray/vLLM/FSDP 端到端、大 Qwen 训练、SFT/GRPO GPU 运行、A800 显存/吞吐和正式实验未运行。当前模型、依赖与 GPU 条件见审计。核心 worker 的生产方法另有隔离 CPU 测试，但不能代替分布式集成测试。

可选扩展已实现 effective rank（明确 singular-value spectrum）、privileged-shift mean pairwise cosine；Gride intrinsic dimension、ADP/DADApy、activation patching/ablation、LoRA 参数子空间仅预留 `extensions.schema.json`，不声称实现或复现。K-means 永远标为工程基线，effective rank 永远不叫 intrinsic dimension。
