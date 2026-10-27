# 服务器Codex交接：2026-10-28重跑OPSD

整理日期：2026-10-27；发布复核：2026-10-28，Asia/Shanghai。执行目录：`Rethinking-OPD-main/`。

**本轮只运行Qwen3-1.7B-Base的OPSD，正式预算50步。记录“当前学生与初始学生”“当前学生与当前特权教师”的分布与表征指标，最终生成单文件离线HTML。保留GRPO设计，本次不训练GRPO。**

本文件替代旧版200/500步、512回答长度及分布选点说明。[旧交接原文](docs/research/archive/PILOT_HANDOFF_before_20261028_zh.md)仅作归档，不从中复制运行命令。`grpo.sh`、`on_policy_distillation.sh`保留原用途，本轮入口是`run_base_comparison.sh`。

## 1. 可直接交给服务器Codex的任务

```text
阅读当前PILOT_HANDOFF_zh.md，以configs/analysis_deepmath.json为正式配置。
先核对SERVER_HANDOFF_MANIFEST.json中的文件，确认收到本轮未提交/新增源码。
检查现有成功环境、GPU分配、真实Base模型及原生DeepMath文件和不可变来源revision。
仅执行OPSD：LoRA r16/alpha32/all-linear，lr=1e-6，8题×8轨迹，正式50步。
训练prompt/response上限2048/16384，teacher prompt上限2048。
统一opsd_answer_v2，初始及监测点heldout抽取失败率门槛1%。
监测0/1/5/10/20/50，保留S0基准，计算Ss/S0与Ss/Ts指标并生成离线HTML。
先在独立目录完成2步工程验收，再以正式配置和新目录运行50步。
不得复用旧step20的probe、generation_cache、analysis_config或complete标记。
不得为过门槛删除失败题、猜A=True、缩短长度、减少正式样本或改变模型/损失。
例行依赖和实现错误可修复并记录；科学配置变化须说明影响并另建配置/目录。
失败保留现场，报告阶段、异常、日志与恢复办法，不宣称训练成功。
交付真实命令、环境/源码身份、数据划分、完成状态、各阶段耗时、峰值显存、
最终完整checkpoint、HTML与指标实测覆盖情况。CPU tiny测试不代表1.7B GPU结果。
```

## 2. 当前正式配置

| 项目 | 当前值 |
|---|---|
| 学生 | `Qwen/Qwen3-1.7B-Base`，本地完整权重/tokenizer及真实revision |
| 自教师 | 当前actor含当前LoRA，加离线`r1_solution_1`；无独立旧教师权重 |
| 更新策略 | `current_batch_stop_gradient`，同批更新前参数，下一批重算 |
| 损失 | 全词表`KL(stopgrad(T_s) || S_s)`，有效回答token平均 |
| 参数更新 | LoRA rank16/alpha32/all-linear，基座冻结 |
| 学习率/seed | 1e-6；训练seed42，probe seed20261026 |
| batch/轨迹 | 8个问题，每题8条，共64条轨迹/更新；microbatch配置1，动态分批 |
| 正式预算/GPU | 50个optimizer更新，epochs=1，默认1张GPU |
| 训练采样 | temperature=1、top_p=1；教师分布温度1 |
| 长度 | 学生prompt≤2048、回答≤16384；教师整个prompt≤2048 |
| 在线诊断 | top-16 overlap/OTA，全词表entropy/gap，更新前参数 |
| checkpoint监测 | 更新后0/1/5/10/20/50 |
| 保留策略 | latest；分析成功后轮换旧权重，保留分析产物 |
| live评测 | 同一题集重新生成，默认greedy，最多16384新token |
| 几何窗口/anchor | 最多256回答token；prompt_last、response_32、response_128 |
| 离线分布 | 全部有效回答位置、全词表T=1；live不被几何256窗口裁剪 |
| 输出 | JSONL/CSV/Parquet、PNG、RSM/anchor/邻居缓存及离线HTML |

配置流：`configs/analysis_deepmath.json` → 工作目录`config.json` → `opsd/training_config.json`、`opsd/analysis_config.json` → 训练resolved配置/Hydra日志。底层profile继承旧pilot，必须由新入口覆盖；不能直接运行旧`pilot_1p7b.json`并认为是本轮配置。

离线模型配置dtype为bfloat16；当前`launch.py`显式设置actor FSDP `model_dtype=fp32`，实际混合精度以Hydra/worker日志为准。不能按“所有权重均BF16”估算训练显存。

学生读取`(x,y_<t)`，自教师读取`(x,r,y_<t)`，共享当前参数。教师分支no-grad，学生分支更新后下一批重算教师；固定的是reference文本。T_s会随学生更新，不是冻结的外部OPD教师。

step0核验LoRA初始B为零后使用原始Base函数，保存anchors与RSM作为S0基准。后续表征比较对齐题目ID、固定回答、层和anchor。当前/初始输出分布比较会加载初始权重，重新评分**同一条当前回答前缀**，不拿另一条初始live回答作对照。

## 3. 数据、提示与判分

输入是单个原生DeepMath Parquet/JSONL（或JSON列表）文件，不是目录；包含`question、final_answer、difficulty、topic、r1_solution_1`。缺少难度/R1参考的旧deduped文件不能替代。

准备顺序：读原始数据 → 难度5–7 → 答案编码检查/参考最终答案核验 → 两种prompt精确长度检查 → NFKC/空白归一化去重 → `(topic, floor(difficulty))`分层固定seed抽样。

| 划分 | 正式数量 | 用途 |
|---|---:|---|
| train | 10000 | 训练候选池 |
| validation | 512 | 独立验证 |
| heldout_probe | 512 | 不参与训练的固定监测题 |
| train_probe | 64 | train的固定子集 |

至少需11024道合格唯一题，64题从train内部选。不是先取一万再筛难度；不足时报错，不自动缩减。50×8=400次题目暴露，64道train_probe不保证被抽到；当前未按实际曝光次数分组，不能全部称作“已训练题”。

固定池回答由step0采样并冻结token IDs；live池每个checkpoint重新生成。学生提示含题目和统一格式指令，末行要求`Final answer: \boxed{YOUR_ANSWER}`。教师仅在前面增加`Reference information:\n{r1_solution_1}\n\n`。学生rollout不读reference。

`opsd_answer_v2`统一提示、抽取、reward和独立判分。没有可靠映射的A/B标签不能猜True/False，先隔离。抽取失败与数学答错分开记录；截断回答在主准确率中计错。教师reference+问题+指令超过2048时排除整题，不静默截断参考。

prepare先做初始Base live校准，heldout抽取失败率须≤1%；512题最多5题失败。后续监测点同样检查。超标保存完整回答/原因并中止，不保证Base首次就达标。256固定回答用于几何，不承担完整解题评测；live最多16384。详见[答案协议](docs/research/ANSWER_PROTOCOL_zh.md)。

## 4. 要记录的两组metric

| metric | 当前学生S_s与初始学生S_0 | 当前学生S_s与当前老师T_s |
|---|---|---|
| top-16 overlap ratio | 两者Top16交集比例 | 两者Top16交集比例 |
| Overlap-Token Advantage | S_s/S_0交集内优势 | S_s/T_s交集内优势 |
| Entropy | H(S_s)、H(S_0) | H(S_s)、H(T_s) |
| Entropy Gap绝对/有符号 | 对照熵减学生熵，逐位置绝对值/保留符号后平均 | 同左，对照为T_s |
| KL | KL(S_s∥S_0) | KL(S_s∥T_s)、KL(T_s∥S_s) |
| 共享概率质量/空交集率 | S_s/S_0，主k=16 | S_s/T_s，主k=16 |
| RSA、RSM RMSE、Linear CKA | 固定题集上的S_s/S_0 | 固定题集上的直接S_s/T_s |
| 表征kNN overlap | k=8/16/32，题目近邻重叠 | k=8/16/32，题目近邻重叠 |
| 轨迹曲率 | 已记录配对变化C(S_s)−C(S_0) | 分别记录双方曲率，尚无独立配对差值字段 |

分布主值对所有有效回答预测位置计算全词表熵/KL，跨题按有效token加权。首个EOS计入，之后padding排除。OTA空交集单独排除，有自己的分母；另存词表k=10/50/100。几何是固定文本/anchor统计，与词表top-k不同。heldout与train_probe分开。

直接RSA/CKA/kNN使用`role=student_vs_privileged_teacher`及`student_teacher`字段。`role=privileged_teacher`下`rsa_rho_to_initial`是T_s对T_0，不能替代S_s对T_s；T_s/T_0辅助曲线保留。

另记PR、首特征值占比、激活范数；学生正确率、抽取失败/格式合规/截断/EOS/长度/重复；训练loss、梯度、学习率、tokens、耗时及显存。Entropy/PR是单模型统计。S0权重固定，但live前缀变化时条件熵可变。

字段映射见[指标清单](docs/research/METRIC_RECORDING_zh.md)、[全回答分布](docs/research/ALL_TOKEN_METRICS_zh.md)、[直接学生—教师几何](docs/research/ST_GEOMETRY_zh.md)。

## 5. 同步并核对源码

本轮代码、新增模块与交接文件一起提交发布。服务器应拉取包含本文件及清单的最新提交，再做下述内容校验；不能只凭清单中的`base_git_commit`判断已收到本轮更新。该字段是开发起点，`worktree_dirty_at_generation`记录清单生成时的状态。保留服务器原有改动和旧结果，不强制reset。

在仓库执行下列校验。`SERVER_HANDOFF_MANIFEST.json`覆盖配置、关键训练/分析源码与本交接文档，不含模型、数据、旧结果及清单自身。

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
m=json.loads(Path('SERVER_HANDOFF_MANIFEST.json').read_text())
bad=[]
for name,expected in m['files'].items():
    p=Path(name)
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
        bad.append(name)
if bad:
    raise SystemExit('交接文件缺失或不同：\n'+'\n'.join(bad))
print('交接源码校验通过：',len(m['files']),'个文件')
PY
```

服务器主动改清单内文件时记录diff与原因，保留原清单；不能静默重算清单掩盖不同。模型/数据身份另由运行manifest记录。

## 6. 环境与真实路径

优先复用服务器step20成功环境，先记录版本再处理具体错误，不盲目升级Torch/vLLM。旧环境安装参考在归档第4节，不是本轮GPU锁定环境。

代码要求Linux/NVIDIA CUDA、Torch、vLLM、FlashAttention、PEFT、Ray、Hydra/OmegaConf、TorchData stateful dataloader、TensorDict、PyArrow、数学判分和分析库。依据`verl/setup.py`和`scripts/research/requirements-analysis.txt`：NumPy<2、Transformers≥4.51且<5、PyArrow≥19，本地verl的vLLM extra范围≥0.8.5且≤0.11.0。Torch/vLLM/FlashAttention应配套，以实际CUDA kernel为准。

先激活服务器已有环境。修改下面真实路径和revision；新入口使用当前PATH的python，不读取旧OPSD_PYTHON。

```bash
set -euo pipefail
cd /your/path/Rethinking-OPD-main
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PWD:$PWD/verl${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="$PWD/research_runs/.matplotlib"
export OPSD_MODEL_DIR="/actual/path/Qwen3-1.7B-Base"
export OPSD_MODEL_REV="ACTUAL_IMMUTABLE_MODEL_REVISION"
export OPSD_DATA_FILE="/actual/path/DeepMath-103K/train.parquet"
export OPSD_DATA_REV="ACTUAL_IMMUTABLE_DATASET_REVISION"
export OPSD_RUN_DIR="$PWD/research_runs/base_opsd_20261028_full_tokens_paired"
export OPSD_PILOT_DIR="$PWD/research_runs/base_opsd_20261028_acceptance_2step"
export OPSD_LOG_DIR="$PWD/research_runs/server_handoff_20261028"
mkdir -p "$OPSD_LOG_DIR" "$MPLCONFIGDIR"
command -v python
nvidia-smi
python -m pip freeze > "$OPSD_LOG_DIR/pip-freeze.txt"
python -m pip check > "$OPSD_LOG_DIR/pip-check.txt"
git rev-parse HEAD > "$OPSD_LOG_DIR/git-head.txt"
git status --short > "$OPSD_LOG_DIR/git-status.txt"
```

保留调度器已有GPU分配。revision须真实，不填main/master或编造字符串；不能把Instruct改目录名当Base。默认单卡不表示已验证A800 80GB够用，必须验收16384回答及额外初始模型的开销。

仅缺本仓库editable绑定且依赖已齐时，可`python -m pip install -e ./verl --no-deps`；不能把跳过安装依赖当作依赖通过。缺包按已记录的兼容环境补齐，新环境参照依赖文件配套安装。

```bash
python - <<'PY' 2>&1 | tee "$OPSD_LOG_DIR/environment-check.log"
import json
from pathlib import Path
import sys
import torch
import verl
from scripts.research.preflight import check_environment
print('Python:',sys.executable)
print('verl:',verl.__file__)
assert Path(verl.__file__).resolve().is_relative_to(Path('verl').resolve())
print('Torch/CUDA:',torch.__version__,torch.version.cuda)
print('C++11 ABI:',torch._C._GLIBCXX_USE_CXX11_ABI)
result=check_environment(gpus=1)
print(json.dumps(result,ensure_ascii=False,indent=2))
if not result['ok']:
    raise SystemExit('环境未通过，先修复以上具体错误')
from transformers import Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model
from vllm.lora.request import LoRARequest
print('Qwen3/PEFT/vLLM LoRA imports passed')
from analysis.html_report import plotly_bundle
print('Offline HTML JS bundle available:',len(plotly_bundle()),'characters')
PY

python - <<'PY' 2>&1 | tee "$OPSD_LOG_DIR/cuda-kernel-check.log"
import torch
from flash_attn import flash_attn_func
q=torch.randn(1,32,4,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
k=torch.randn_like(q,requires_grad=True)
v=torch.randn_like(q,requires_grad=True)
y=flash_attn_func(q,k,v,dropout_p=0.0,causal=True)
y.float().square().mean().backward()
torch.cuda.synchronize()
assert torch.isfinite(y).all() and all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q,k,v))
print('BF16 FlashAttention forward/backward passed')
PY
```

HTML构建需可导入Plotly，或保留仓库旧报告中的内嵌JS作为备用；上面在开跑前核验可用性。检查`df -h`、`free -h`、`df -h /dev/shm`及容器/作业内存上限；offload、导出和分析需要主存/磁盘。必要时为本任务设置短的RAY_TMPDIR，不停止共享服务器上其他任务的Ray进程。

## 7. 独立2步工程验收

只减题数和步数，保留正式batch8×8、LoRA、学习率、损失及2048/16384长度。8题heldout在1%门槛下必须零抽取失败；通过不代表正式512题通过，不混入正式曲线。

```bash
python - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('configs/analysis_deepmath.json').read_text())
c['run_id']='acceptance_20261028_2step'
c['dataset'].update(train_n=16,dev_n=8)
c['training'].update(steps=2,save_every=1)
c['monitor'].update(optimizer_steps=[0,1,2],teacher_geometry_steps=[0,1,2],
                    plot_full_rsm_steps=[0,2],heldout_probe_n=8,train_probe_n=4)
p=Path('research_runs/server_handoff_20261028/acceptance_config.json')
p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
print(p.resolve())
PY

PILOT_ARGS=(
  --config "$OPSD_LOG_DIR/acceptance_config.json"
  --model-dir "$OPSD_MODEL_DIR" --model-revision "$OPSD_MODEL_REV"
  --input "$OPSD_DATA_FILE" --dataset-revision "$OPSD_DATA_REV"
  --workdir "$OPSD_PILOT_DIR" --method opsd
)
bash run_base_comparison.sh "${PILOT_ARGS[@]}" > "$OPSD_LOG_DIR/pilot-plan.json"
bash run_base_comparison.sh "${PILOT_ARGS[@]}" --execute
bash run_base_comparison.sh status --workdir "$OPSD_PILOT_DIR" --watch
```

计划必须为methods=[opsd]、steps=2、监测[0,1,2]。验收覆盖：prepare/初始门槛、真实LoRA更新、checkpoint保存/导出、全回答分布、直接RSA/CKA和HTML。测量训练及分析完整阶段显存；训练进程allocated/reserved不等于所有并存进程的显存总量。

保留step0/1/2分析及最终step2 checkpoint。8/4题下k≥N的近邻undefined属预期，不能补0；正式512/64题另验收。失败先查第11节，不直接启动正式50步。

## 8. 正式50步一键运行

使用正式配置及新目录，不传上节小样本配置：

```bash
RUN_ARGS=(
  --config "$PWD/configs/analysis_deepmath.json"
  --model-dir "$OPSD_MODEL_DIR" --model-revision "$OPSD_MODEL_REV"
  --input "$OPSD_DATA_FILE" --dataset-revision "$OPSD_DATA_REV"
  --workdir "$OPSD_RUN_DIR" --method opsd
)
bash run_base_comparison.sh "${RUN_ARGS[@]}" > "$OPSD_LOG_DIR/formal-plan.json"
bash run_base_comparison.sh "${RUN_ARGS[@]}" --execute
bash run_base_comparison.sh status --workdir "$OPSD_RUN_DIR" --watch
```

正式计划必须为50步、8×8、lr1e-6、监测[0,1,5,10,20,50]、train10000/validation512/heldout512/train_probe64。预览会写config/models但不生成校准回答，不代表质量通过。

`--execute`默认受管后台运行，SSH断开后继续；返回只表示提交，不表示完成。排查可末尾加`--foreground`。不重复提交同一目录。

```bash
# 完整流程日志；prepare时尚无train.log，先看此处。
bash run_base_comparison.sh logs --workdir "$OPSD_RUN_DIR" --follow
# 训练详细日志。
bash run_base_comparison.sh logs --workdir "$OPSD_RUN_DIR" --stream train --follow
# 只读JSON状态。
bash run_base_comparison.sh status --workdir "$OPSD_RUN_DIR" --json
```

顺序：环境检查 → 数据筛选/冻结 → 初始live质量门槛 → 固定回答池 → step0保存/分析 → 指定step训练/保存/分析 → 最终报告。prepare既含CPU筛选也含GPU生成；512题长回答和全回答分布评分可能耗时，按generation_progress及流程日志判断进度。

## 9. checkpoint与产物

常规save_every=50之外，监测点也保存checkpoint，覆盖0/1/5/10/20/50。latest流程：保存完整model/optimizer/data状态及标量 → 暂停训练 → 独立进程分析 → 成功后轮换旧checkpoint/HF导出 → 继续。分析失败保留现场并停止推进。

分析进程独立不意味着训练worker显存已释放。分析载入当前模型，非step0另可载入初始模型；全回答分布使用KV cache和分块LM head，不缓存整条T×V分布。新显存/耗时以验收为准，不沿用旧256回答实验估算。

```text
<workdir>/
  config.json, models.json, design_manifest.json, environment.json
  .session/                        # 后台控制与实际退出状态
  shared/
    preparation_audit/             # 筛选/答案隔离；失败也保留
    generation_cache/, generation_progress.json
    quality/baseline/              # quality及完整/失败回答
    probe/
      train.jsonl, validation.jsonl, heldout_probe.jsonl, train_probe.jsonl
      baseline_live_rollouts.jsonl, fixed_pool.jsonl, probe_manifest.jsonl
      manifest.json, complete.json
  opsd/
    training_config.json, analysis_config.json, initial_adapter.json
    train/
      metrics.jsonl, train.log, resolved_config.json, command.sh
      hydra_preflight.log, rollout_schema_preflight.log
      checkpoints/global_step_50/   # 正式完成后的最终完整checkpoint
    jobs/, job_status/
    analysis/
      quality/step_XXXXXX/
      steps/step_XXXXXX/
        complete.json, run_manifest.json
        metrics.jsonl, distributions.jsonl, curvature.jsonl, live_rollouts.jsonl
        anchors.npz, anchors.json, rsm_*.npz, neighbors.npz
        student_teacher_comparisons.json, student_teacher_neighbors.npz
  grpo/training_config.json, grpo/analysis_config.json  # 设计预留，不代表训练
  report/
    metrics.csv, metrics.jsonl, *.png, report_manifest.json
    OPSD_report.html                # 数据/JS/图库内嵌，单文件离线
```

必须保留最终完整checkpoint，不能只有adapter或HTML。latest不保留全部历史权重，但保留各监测点分析表格/anchor。未来改变anchor/窗口/提取算法重算需预先另建retention=all实验或备份权重；已删除权重不能由anchor重建。

## 10. 完成标准与HTML

1. 任务completed且退出码0；不能只看PID/GPU占用。
2. 实际训练到50，最终global_step_50完整可恢复。
3. step0/1/5/10/20/50均有complete及有效产物hash。
4. 两组新指标有正确field/role/pool/split/layer/anchor及n_valid；undefined不补0。
5. 基线和各监测点heldout质量门槛通过，完整失败明细可审计。
6. 有student_teacher_comparisons及邻居缓存，RSA按共同题ID配对。
7. report/OPSD_report.html单文件离线可读，模型/实际step/指标可用性正确；不用旧step20报告交付新实验。
8. 可以生成GRPO配置，但本轮没有GRPO训练日志/checkpoint。

仅重画HTML无需GPU：

```bash
python -m analysis.html_report \
  --run-dir "$OPSD_RUN_DIR" \
  --output "$OPSD_RUN_DIR/report/OPSD_report.html"
```

HTML区分token主均值、逐题摘要、历史选点；概览按两组列13类指标及字段，区分代码支持与实测有值。正确率结合抽取/截断及样本量解释，RSA/CKA高不能证明推理机制或能力相同。

## 11. 失败恢复

源码/配置/模型/数据身份相同时可同目录重试，缓存与分析经hash检查复用。协议、指标算法、数据或模型改变后新建目录，不删complete或关闭身份校验硬续跑。

```bash
# RUN_ARGS保持第8节原参数。
bash run_base_comparison.sh "${RUN_ARGS[@]}" --execute
# 已准备，仅恢复训练；后台运行，resume=auto。
python -m scripts.research.compare train --workdir "$OPSD_RUN_DIR" --method opsd --execute
# 只重试导出/分析，前台执行；step0必须先完成。
python -m scripts.research.compare analyze --workdir "$OPSD_RUN_DIR" --method opsd
# 完整重画表格、PNG与HTML。
python -m scripts.research.compare report --workdir "$OPSD_RUN_DIR"
```

| 失败位置 | 排查依据 |
|---|---|
| import/CUDA/FlashAttention | 环境报告、ABI、真实kernel异常、本地verl导入路径 |
| 合格题不足 | shared/preparation_audit/eligibility与隔离标签；特别检查teacher prompt2048排除量 |
| 抽取门槛失败 | quality/.../quality.json、responses.jsonl、failed_responses.jsonl；区分编码、坏格式、截断 |
| Hydra/Ray/vLLM | hydra_preflight、rollout_schema_preflight、train.log完整异常 |
| 训练/分析OOM | 记录具体阶段、长度及并存进程；不静默改batch/n/长度或跳过指标 |
| LoRA导出 | 保留checkpoint、alpha/r与失败目录；不把未合并的基座当新模型 |
| HTML缺新指标 | 查实际每step metrics与role/aggregation/scope，重绘不能补算未测数据 |
| 源码/probe不符 | 对照同步清单与旧运行身份；新建目录，不伪造完成标记 |

仅在用户明确要求停止时执行`bash run_base_comparison.sh stop --workdir "$OPSD_RUN_DIR"`。退出日志查看不停止训练。到50步后入口复用最终checkpoint，不额外训练第51步。

## 12. 代码入口与最终汇报

| 文件 | 职责 |
|---|---|
| configs/analysis_deepmath.json、scripts/research/compare.py | 当前配置、OPSD默认选择和完整调度 |
| analysis/prepare_probe.py、scripts/research/data.py | 筛选、隔离、分层划分与probe |
| scripts/research/answer_protocol.py、verl/verl/utils/opsd.py | 统一提示/抽取/判分，特权上下文与token打包 |
| scripts/research/launch.py | Hydra覆盖、LoRA/full-vocab forward KL、长度检查 |
| verl/verl/workers/actor/dp_actor.py、fsdp_workers.py | 学生梯度前向、当前特权分支no-grad与更新 |
| verl/verl/utils/opd_metrics.py、ray_trainer.py | 在线诊断、日志及保存/分析监测点 |
| analysis/checkpoints.py | FSDP/LoRA导出、初始零B核验 |
| analysis/extract.py、analysis/metrics/paired.py | anchor/曲率、初始比较和直接S_s/T_s比较 |
| analysis/distributions.py、analysis/trajectory_distributions.py | 全回答全词表、token累加、同轨迹初始模型比较 |
| analysis/evaluate.py | 完整live回答、质量门槛、可恢复缓存 |
| analysis/report.py、analysis/html_report.py、analysis/report_template.html | 表格、图与离线HTML |

本地已通过142项研究测试和11项原OPD指标测试，共153项，包括tiny GPT-2/Qwen3与HTML控件测试；没有运行本轮真实1.7B长上下文GPU实验。本地临时依赖目录不是服务器环境。

最终汇报：真实命令/工作目录、源码清单校验、Python/pip/CUDA/GPU身份、模型/数据revision及hash、各split数量、实际完成步数、质量结果、两组新指标覆盖、最终checkpoint/HTML路径、prepare/训练/分析分别耗时及峰值显存。失败或未测明确标出，不只交一张图或“训练结束”。
