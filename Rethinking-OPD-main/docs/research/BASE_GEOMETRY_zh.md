# DeepMath × Qwen3-1.7B-Base：OPSD / GRPO 表征轨迹

> 2026-10-27：已补充同一步学生与特权教师的直接RSA/CKA/kNN overlap，见[学生—教师几何](ST_GEOMETRY_zh.md)。默认每个监测点采集教师，独立role避免与初始模型比较混淆。

> 2026-10-27：当前50步、答案协议v2、训练长度2048/16384。分布监测改为全回答token平均；详见[全回答指标](ALL_TOKEN_METRICS_zh.md)和[答案协议](ANSWER_PROTOCOL_zh.md)。下文未更新的200步/512长度为历史记录。

> 2026-10-27 更新：用户已将当前实验总预算缩短为 **50 步**。默认新运行目录为 `research_runs/base_geometry_50`；正在运行的实验保留原目录及冻结分析配置，通过 `run_limits.json` 和显式续跑配置在第 50 步正常结束。下文 200 步预算为历史设计记录。

更新：2026-10-26，Asia/Shanghai。以本轮用户明确要求为准，开发规范中“本轮不要求 RL 对照”扩展为 OPSD 与 GRPO 两组；其余按固定 probe、独立分析的设计实现。P0/P1 已实现，P2 的 Gride / ADP / ARI 未实现，配置启用会明确报错。没有加入几何正则，也不把平直化解释为能力提升。

当前执行阶段调整为：**默认只训练 OPSD，保存在线 metrics 并分析 checkpoint 的表征变化**。GRPO 设计、共用数据和配置生成逻辑均保留；只有显式传入 `--method grpo` 或 `--method both` 才执行 GRPO。`run`、`train`、`analyze` 三个入口的默认 method 均为 `opsd`。

当前正式预算为 **200个optimizer updates**，监测点为 `0/1/5/10/20/50/100/200`；最终step200同时纳入教师几何与完整RSM绘图。默认目录为 `research_runs/base_geometry_200`。已有500-step配置/缓存不会自动迁移，继续使用时请提供新的workdir。

交给服务器Codex时，使用 [PILOT_HANDOFF_zh.md](../../PILOT_HANDOFF_zh.md) 第1节的可复制任务说明；该文档集中提供环境/路径设置、2-step验收、200-step一键命令和产物核对。本文件用于解释实验与指标的详细实现。

## 1. 训练设计与边界

| 项目 | OPSD | GRPO |
|---|---|---|
| 初始模型 | 同一份本地 `Qwen/Qwen3-1.7B-Base`、固定 revision 和文件 hash | 相同 |
| 数据 | 同一份 DeepMath level 5–7，归一化题干 hash 去重后的 train / dev | 相同 |
| 参数更新 | LoRA r16 / alpha32 / all-linear，lr=1e-6 | 相同；adapter 初始化 seed 与 OPSD 一致 |
| batch / rollout / steps | 8 题 × 8 条，200 optimizer updates | 相同 |
| 学生输入 | `Problem:\n{question}\n\nSolution:\n` | 相同，无 reference |
| 训练监督 | 当前 actor 加固定 `r1_solution_1`，停止梯度；全词表 `KL(teacher || student)` | 原仓库 GRPO advantage + PPO clipped objective；数学 verifier outcome reward，无额外 KL/entropy 正则 |
| 教师更新 | `current_batch_stop_gradient`，每批更新前重算，无独立教师权重 | 无训练教师 |
| 在线诊断 | 保留 top-16 overlap ratio、Overlap-Token Advantage、全词表 entropy / gap | 普通 RL 训练标量；固定状态上的特权分布诊断单独计算 |

当前采用 LoRA / 前向 KL / 200 step 设置，没有采用附件第 10 节的“另建最小训练”建议替换现有目标。Base 使用纯文本续写；通过一个只返回 user content 的 Jinja 适配器接入 verl 的数据接口，不添加 Instruct 的角色 token 或 thinking 开关。模型维度从真实 config 读取，不硬编码 28×2048。官方来源：[Base 模型配置](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/main/config.json)、[DeepMath schema](https://huggingface.co/datasets/zwhe99/DeepMath-103K)。本地权重不会自动下载或替换；调用方必须提供真实来源 revision，manifest 另记录文件内容 hash，不能仅凭尺寸证明 Base 来源。

OPSD 与 GRPO 的监督与计算成本不同；同 step 不是同计算预算。分析表同时携带可用的训练 optimized tokens / rollout tokens / teacher forward tokens / wall seconds，分析耗时独立记录。原始 `grpo.sh`、`on_policy_distillation.sh` 不变。旧shell包装已于2026-11-09退役；历史 non-thinking CLI可通过 `python -m scripts.research.pipeline run` 调用。本阶段使用 `run_base_comparison.sh`，当前三组对照使用 `run_controlled_comparison.sh`。

## 2. 配置与数据冻结

主配置：[configs/analysis_deepmath.json](../../configs/analysis_deepmath.json)。`training` 是两组共用训练参数；`monitor` 控制分析频率、anchor、kNN、固定状态和绘图。基础训练 profile 是 `configs/research/base_geometry.json`。

默认数量：train=10,000，dev=512，heldout_probe=512，train_probe=64。最后一组是 train 的固定子集，单独统计。其余三组互不相交，不复用 dev 当 probe。先按 difficulty 5–7 筛选、核验参考最终答案及两种 prompt 长度，再按 NFKC+空白归一化题干 hash 去重；答案冲突报错。数据集参考的最终答案通过与训练同一 verifier 核验，**不代表中间推理已被人工证实**。默认排除未核验参考和过长 prompt，两组使用完全相同的剩余题目，记录 `eligibility.jsonl`、重复 ID 和所有排除原因；数量不足时报错，绝不自动减小样本量。

topic 默认取 `Mathematics -> 子领域 -> …` 的第 2 级。缺少 topic 报错；层级不够时取最后一级并保存实际级别。按 `(topic_label, floor(difficulty))` 分层轮转抽样，层内固定 seed，保存全部具体 ID。题目排序固定为 `(topic_label, difficulty, problem_id)`。

默认训练 prompt 上限仍为 student=512、teacher=2048，response=512；较长 R1 解答会被明确排除。若希望保留更多长参考，可在新配置中增加 teacher prompt 上限，先验证可用显存。不会静默截短 reference。

三种 pool 分开保存与统计：

- `fixed_base_rollout`：同一初始 Base 在固定题上以 T=1、top_p=1、每题一个固定 seed 生成，最多 256 tokens。以后始终复用真实 prompt / response IDs。
- `live_rollout`：每个 checkpoint 在同一题目集合重新生成，默认 greedy。准确率、长度、截断、EOS、抽取失败和重复 4-gram 单独保存。
- `fixed_reference`：可选，默认关闭。参考先通过最终答案核验，对完整拼接字符串分词并检查精确 prompt 边界；不把单独 tokenize 后的文本直接拼接。若有跨边界 token 合并，明确报错并要求改用 tokenizer 稳定的模板后新建实验。

step0 固定池的正确性属于生成源；不能当成后续 checkpoint 的自主准确率。截断和抽取失败在 live 主准确率中计错，不删除。

## 3. 实现位置

| 模块 | 职责 |
|---|---|
| `scripts/research/compare.py` | 两组共用配置、prepare/train/analyze/report，独立目录和执行状态 |
| `analysis/prepare_probe.py` | 去重、分层划分、固定 pool、真实 token IDs、anchor 有效 ID、hash 锁定 |
| `verl/verl/utils/opsd.py` | Base student / privileged teacher 纯文本协议；原 chat 行为仍保留 |
| `scripts/research/launch.py` | 将 LoRA 同时接入 OPSD / GRPO，设置 monitor 和 Base 模板 |
| `verl/verl/workers/config/model.py`、`fsdp_workers.py` | 显式 LoRA 初始化 seed，创建 adapter 后恢复周围 Torch RNG |
| `verl/verl/trainer/ppo/ray_trainer.py` | 保存 step0 和 monitor 指定 checkpoint，标量持久化后发布分析 job；不运行 hidden-state 分析 |
| `verl/verl/utils/research_monitor.py` | 仅依赖标准库的原子 job，核验 probe hash，不动 RNG |
| `analysis/checkpoints.py` | 使用 vendored `verl.model_merger` 导出 FSDP；临时目录合并真实 alpha 的 LoRA，成功后原子发布，不写训练 checkpoint |
| `analysis/extract.py` | eval + inference_mode，完整 block 输出 hook、anchor、曲率在线归约、逐 checkpoint manifest |
| `analysis/metrics/{rsa,trajectory,neighbors,spectrum}.py` | RSA / 配对相关、曲率、原始 Euclidean kNN、参与率与 CKA |
| `analysis/trajectory_distributions.py` | fixed/live全回答分块评分、同轨迹初始模型比较，KV cache控制中间显存 |
| `analysis/distributions.py` | 全部有效回答位置的全词表T=1 entropy、双向KL、top-k overlap/OTA与token累加；NLL正确causal shift |
| `analysis/evaluate.py` | live rollout、与训练共享的答案抽取和等价判断、行为指标 |
| `analysis/report.py` | 只读缓存重画图、汇总 CSV / JSONL / Parquet，不加载模型 |

hook 路径与实际维度均写 manifest；Qwen 的路径为 `model.layers.<0-based index>`，输出表的 layer 从 1 起。没有用 MLP 裸输出或 final norm 后的 hidden_states 代替 block 输出。每题前向后只保留原始 float32 anchors 与曲率统计，不保存全部层的 token 激活。

主 anchor 为 prompt_last / response_32 / response_128。`response_32` 是读取第 32 个 response token 后的状态；NLL 的第 32 个 target 使用它前一位置的 logits。固定 pool 的 anchor 有效题集跨 step 不变，显式保存位置；文本不够长时该 anchor 的 N=0、指标 undefined，不 padding 成假轨迹。

## 4. 指标定义和输出

| 指标 | 实现口径 |
|---|---|
| RSM | 原始向量归一化后的跨题目余弦矩阵；统计仅取严格上三角，保存直方图、固定排序、无效零向量 ID；topic 少于两类时 topic 统计 undefined |
| 学生—教师直接比较 | 同一checkpoint、固定题目/回答、同层anchor，联合有效题集上的RSA/CKA/Euclidean kNN overlap；独立role=student_vs_privileged_teacher |
| RSA rho / RMSE | 与 step0 相同题 ID、层、pool、anchor 比较；Spearman 常量无定义，固定 pair 抽样 seed / n_pairs 可追溯 |
| 曲率 | response 内连续非 special tokens，先转 float32 再作差；夹角单位 degrees，零步长 invalid；先题内后题间平均 |
| relative-to-first / training delta | 首 block 为参考，按跨层和 step0 共同有限题集配对；同时输出 absolute C、D、TrainDeltaC、D_min、layer_star 和 N |
| kNN overlap | 原始 H 上 Euclidean k=8/16/32，排除自身，距离并列按 sample_id 排序；N<=k 为 undefined，保存 indices、距离、逐题 overlap 和机会水平 |
| PR / CKA | 中心化参与率与 Gram 形式 linear CKA；常数表示 undefined，不称 Gride / intrinsic dimension |
| NLL / surprisal | 固定 response 的真实 target，bits/token；同时有逐层曲率与当前、step0 surprisal 的 Pearson 相关 |
| 固定状态分布 | fixed/live的全部有效回答预测位置，全词表T=1，跨题按有效token加权 |
| 固定状态 KL | 双向student/privileged-current-policy KL，另记录当前/初始学生在同一轨迹上的KL；不替换训练前向KL |
| 固定状态词表 overlap | 主k=16，额外10/50/100；记录overlap/OTA/共享概率质量/空交集率。与题目kNN的k独立 |

GRPO 的 privileged-current-policy 是**额外诊断条件**，不是训练教师，字段 `teacher_role=diagnostic_privileged_policy`。OPSD 监测使用 checkpoint 的当前参数加 reference；训练在线诊断在更新之前，离线 step_s 几何在更新之后。两者不能不加说明地认为是完全相同参数状态。

固定 pool 才计算训练前后的主几何变化；live 曲率单独报告，不把变化的文本当成固定文本做跨 step RSA 或 training delta。默认开启直接学生—教师比较，每个监测点均采集教师几何，独立role、独立矩阵。

```text
<workdir>/
  config.json, models.json, design_manifest.json
  shared/probe/
    train.jsonl, validation.jsonl, heldout_probe.jsonl, train_probe.jsonl
    fixed_pool.jsonl, probe_manifest.jsonl, manifest.json, complete.json
    eligibility.jsonl, duplicates.jsonl
  opsd/                         # grpo/ 结构相同
    training_config.json, analysis_config.json
    initial_adapter.json        # 检查 step0 的 B=0；两组初始 adapter 内容须一致
    train/metrics.jsonl, train/plots/thinking_consistency.png
    train/checkpoints/global_step_*/actor/…
    jobs/step_*.json, job_status/step_*.json
    hf/solution/global_step_*/…  # 原子导出的合并权重
    analysis/steps/step_000000/
      anchors.npz, anchors.json, neighbors.npz, rsm_*.npz
      curvature.parquet, distributions.parquet, metrics.parquet
      curvature.jsonl, distributions.jsonl, metrics.jsonl, live_rollouts.jsonl
      initial_log_probs/…       # 只在 step0 保存小型固定状态的完整 log probabilities
      run_manifest.json, complete.json
  report/metrics.csv, metrics.parquet, metrics.jsonl, *.png, report_manifest.json
```

报告包含 RSM 热图、RSA / NO / PR 的 step×layer 热图、三种曲率层曲线与热图、行为与固定曲率并列图、分布指标 step 曲线、曲率-surprisal 散点和逐层相关。RSM 展示层按预先固定相对深度从 config 派生，不能事后挑层。未定义值 JSON 为 null，Parquet 为 null，图上断开；空交集 / 短文本 / 退化向量不补零。

## 5. 服务器运行顺序

准备完整本仓库、定制 verl、`analysis/`、本地 Base 权重、真实原生 DeepMath 文件及 revision。环境安装沿用 [PIPELINE_zh.md 第 0 节](PIPELINE_zh.md#0-准备环境与输入一次)，加 `scripts/research/requirements-analysis.txt`。Qwen3 需要 Transformers>=4.51；训练还需 Ray、vLLM、FlashAttention、PEFT、TorchData、PyArrow 等。不要用本机 CPU 环境声称已具备 GPU 训练环境；manifest 会记录实际软件版本。

```bash
cd /your/path/Rethinking-OPD-main
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PWD/verl${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="$PWD/research_runs/.matplotlib"

# 1. 固定配置；先修改 configs/analysis_deepmath.json 中想调整的参数。
python -m scripts.research.compare configure \
  --model-dir /models/Qwen3-1.7B-Base --model-revision ACTUAL_MODEL_COMMIT \
  --input /data/DeepMath-103K/train.parquet --dataset-revision ACTUAL_DATA_COMMIT \
  --workdir research_runs/base_geometry_200

# 2. 生成一次固定数据与初始 rollout，写出两组配置。
python -m scripts.research.compare prepare --workdir research_runs/base_geometry_200

# 3. 训练前先预览实际训练命令；加 --execute 才训练。
python -m scripts.research.compare train --workdir research_runs/base_geometry_200 --method opsd
python -m scripts.research.compare train --workdir research_runs/base_geometry_200 --method opsd --execute

# 4. 顺序导出并分析 OPSD 的监测 checkpoint；失败后可重试同一命令。
python -m scripts.research.compare analyze --workdir research_runs/base_geometry_200 --method opsd

# 5. 只读缓存，生成 OPSD 的指标与表征变化报告。
python -m scripts.research.compare report --workdir research_runs/base_geometry_200
```

一键执行上述 OPSD 流程：

```bash
bash run_base_comparison.sh \
  --model-dir /models/Qwen3-1.7B-Base --model-revision ACTUAL_MODEL_COMMIT \
  --input /data/DeepMath-103K/train.parquet --dataset-revision ACTUAL_DATA_COMMIT \
  --workdir research_runs/base_geometry_200 --execute
```

不传 `--execute` 只配置和打印计划，不加载权重/生成 probe/训练。默认 `--method opsd`，计划与实际执行都不包含 GRPO。prepare 仍保存两组配置，供后续使用；生成 GRPO 配置不会启动 GRPO 训练。配置不符时拒绝复用目录，换新 `--workdir`，不会覆盖旧实验。

后续需要 GRPO 对照时，在同一工作目录显式执行：

```bash
python -m scripts.research.compare train --workdir research_runs/base_geometry_200 --method grpo --execute
python -m scripts.research.compare analyze --workdir research_runs/base_geometry_200 --method grpo
python -m scripts.research.compare report --workdir research_runs/base_geometry_200
```

也可在一键入口显式加 `--method both` 依次执行两组。旧命令若带有该选项，本阶段应移除它或改为 `--method opsd`。

**第一次真实服务器验收先缩小到 2 step：** 在一键命令使用新的 workdir，并加：

```bash
--steps 2 --monitor-steps 0,1,2 --train-n 16 --dev-n 8 --probe-n 64 --train-probe-n 8
```

这需要至少 88 个满足参考/长度规则的独立问题；train_probe 是 train 子集。heldout 的 kNN 有足够 N，train_probe=8 时 k>=8 按规范无定义。验收 Base prompt、在线五项指标、LoRA 初始指纹、checkpoint 导出、独立分析与恢复后，再运行 200 step。实际显存和耗时需由服务器测量；未承诺单卡默认长度一定不 OOM。

训练仍每 50 step 保存常规 checkpoint，**另外**保存 monitor 的 `[0,1,5,10,20,50,100,200]`。step 按 optimizer update 计数，不按 microbatch。默认 `training.checkpoint_retention=latest`，正常完成后只保留最新一份完整模型/optimizer/scheduler/RNG/dataloader状态。每次成功保存checkpoint和标量后，训练暂停，由独立进程完成当前监测点分析；结果表/图/manifest和job完成收据保留，临时HF权重删除，然后清理本运行较旧的checkpoint。分析失败则停止并保留现场，重试成功后才轮换。同配置使用resume=auto；已完成的旧分析通过收据、源码、probe和所有产物hash校验后复用，无需旧checkpoint。未识别目录或符号链接不纳入自动清理。设为all可恢复保留全部权重、训练后分析的方式。

当前 FSDP checkpoint 保存完整FP32 PEFT模型状态，含冻结基座，并非仅adapter。latest模式保存期间短暂需要新旧两份checkpoint，分析阶段另需一份临时HF导出；LoRA合并在FP32计算后按分析所用BF16保存，避免原单文件遮蔽新FP32分片。固定状态的初始full-vocab log probabilities和各点分析结果仍累积保留。训练分析峰值与缓存必须由2-step实测，不能把“只留最新”理解为零额外空间。

默认不并发训练和几何分析：driver等待分析子进程结束后才更新下一批，子进程沿用launcher原GPU分配，避免CPU Ray driver清空CUDA_VISIBLE_DEVICES。训练worker显存可能仍驻留，GPU集成验收前不保证峰值可容纳。分析和轮换耗时写checkpoint_maintenance.jsonl，并从训练wall预算扣除。不要同时启动另一个analyze写同一运行目录；命令处理队列快照，不是常驻watcher。

也提供附件要求的独立 CLI，读取已配置的分析配置和已导出的 HF 权重：

```bash
python -m analysis.prepare_probe --config research_runs/base_geometry_200/config.json
python -m analysis.extract \
  --config research_runs/base_geometry_200/opsd/analysis_config.json \
  --checkpoint /path/to/exported_hf_step20 --revision ACTUAL_EXPORT_ID --step 20
python -m analysis.report --run-dir research_runs/base_geometry_200
```

先分析 step0；直接 extract 要由调用方提供正确的合并 HF 权重。常规用 compare analyze，可自动校验初始 adapter、导出 FSDP 并按顺序分析。

## 6. 本地验证与限制

```bash
python -m unittest discover -s tests/research -v
python -m analysis.smoke --out research_runs/base_geometry_cpu_smoke_new --report
```

CPU smoke 使用 2-block、hidden_size=24 的微型 GPT-2、12 个合成问题、两个权重状态；step1 是确定性参数扰动，**不是 OPSD/GRPO 训练**。使用显式 JSONL 模式，不假装本机已经运行 PyArrow/Parquet。覆盖真实 block 前向、response-local 索引、NLL shift、anchor / RSM / kNN / 曲率缓存和无模型绘图。科学计算、真实 tiny forward、共享数据生成与命令构造、初始 adapter 校验、导出失败重试均有测试。

本机没有可用 CUDA，缺少 Ray/vLLM/PEFT/PyArrow/pylatexenc，Transformers 4.44.2 也不能加载 Qwen3。真实 Qwen GPU 训练、分布式 FSDP 导出和恢复、实际 PEFT 合并、Parquet 写入、显存峰值和训练耗时均未运行。导出单元测试验证流程与失败隔离，不能替代服务器 2-step 验收。运行时生成的 manifest 记录真实软件版本，测试输出不用于数学能力结论。
