# 服务器 Codex 交接：Qwen3-1.7B-Base 的 OPSD 过程监测

> 2026-10-27 更新：用户已将当前实验总预算缩短为 **50 步**。默认新运行目录为 `research_runs/base_geometry_50`；正在运行的实验保留原目录及冻结分析配置，通过 `run_limits.json` 和显式续跑配置在第 50 步正常结束。下文 200 步预算为历史设计记录。

更新：2026-10-27，Asia/Shanghai。**本阶段只执行 OPSD，记录训练中的分布指标与 checkpoint 表征变化。GRPO 对照的设计和代码保留，当前不运行。**

本文件是服务器操作主说明；指标细节见 [BASE_GEOMETRY_zh.md](docs/research/BASE_GEOMETRY_zh.md)。唯一的一键入口是 `run_base_comparison.sh`，默认且下文显式指定 `--method opsd`。

## 1. 可直接复制给服务器 Codex 的任务说明

将下面整段发给服务器上的 Codex，并提供第 4 节所需的真实路径和版本：

```text
请在当前 Rethinking-OPD-main 仓库执行已实现的 OPSD 实验，不重新设计实验。

先阅读 PILOT_HANDOFF_zh.md、docs/research/BASE_GEOMETRY_zh.md、
configs/analysis_deepmath.json、run_base_comparison.sh、scripts/research/compare.py，
以及修改文件所在目录适用的 AGENTS.md。

研究问题：观察 Qwen3-1.7B-Base 经 OPSD 训练时，学生与特权自教师的输出分布，
以及固定问题、固定文本上的内部表征结构和 token 轨迹曲率如何变化；
另用同题目的 live rollout 观察解题表现与访问轨迹变化。

本阶段只运行 OPSD，显式使用 --method opsd。GRPO 的设计、配置、数据对齐和
后续执行接口保留；不要启动 GRPO、--method both、SFT 或旧的 non-thinking 实验。

模型必须是同一份 Qwen/Qwen3-1.7B-Base，锁定真实来源 revision、tokenizer 和内容 hash；
不得替换为 Qwen/Qwen3-1.7B。使用仓库的 Base 纯文本续写模板。
数据是原生 zwhe99/DeepMath-103K，5 <= difficulty <= 7，
默认离线参考来自 r1_solution_1，按现有规则核验最终答案、去重、检查长度并冻结划分。
学生 rollout 不读 reference。自教师是当前 actor（含当前 LoRA）加离线参考，
在相同学生 response prefix 上给 logits，停止梯度；每个 batch 更新前重算。
不要改成冻结初始 teacher、EMA teacher 或外部独立 teacher。

保持现有训练目标：全词表 KL(teacher || student)，仅训练 LoRA，r16/alpha32/all-linear，
lr=1e-6，8题 x 每题8条回答，microbatch=1，200 optimizer updates，seed=42，默认1张GPU。
采样和教师温度均为1。学生 prompt/response 上限512/512，教师 prompt上限2048。
几何指标仅用于监测，不加入训练 loss。

每个 batch 保存 overlap ratio、Overlap-Token Advantage、student/teacher entropy、
entropy gap、forward-KL loss 和预算。在线词表 overlap 的 k=16。
指定 checkpoint 用独立程序计算 RSA、曲率及其层间/训练变化、Euclidean kNN overlap、
PR、CKA、surprisal、固定状态分布指标，以及 live 行为；保存原始表和可重画图。
固定文本与 live 文本分开，不将更平直或更相似直接解释成数学能力提升。

执行：核对本地输入、CUDA和依赖 -> 新目录2-step GPU验收 -> 检查第7节产物 ->
验收通过后新目录200-step正式运行。使用一键脚本，不另写替代训练循环。
用当前激活环境的 python，安装本仓库定制 verl，不能用另一份 pip verl 替代。
先定位已有环境、模型和数据；缺少真实路径/revision时再询问，不编造版本。
普通依赖和兼容性错误可定位修复并记录；不要静默减少样本/长度、改变目标或换模型来绕过错误。
如确需改变科学配置，明确说明并另建配置和运行目录，保留原始失败记录。

最终交付：真实执行命令、环境版本、模型/数据身份、训练与分析完成状态、checkpoint、
metrics/CSV/Parquet/图的路径、实际训练/分析耗时及可用显存测量。
未运行或失败的部分明确标注，不能把本地 tiny GPT-2 CPU smoke 当作 Qwen GPU 实验结果。
```

## 2. 实验设计与不可混淆的概念

### 2.1 模型、教师与损失

学生输入：

```text
Problem:
{question}

Solution:
```

教师输入：在相同问题前增加 `Reference information:\n{r1_solution_1}\n\n`，保留同一问题/续写后缀，评分时复用学生的真实 response token IDs。教师有自己的真实 prompt 长度和 position IDs。

令学生分布为 p，特权教师分布为 q，训练直接最小化全词表 `sum_v q(v) * (log q(v) - log p(v))`，按有效 response token 平均。教师 no-grad，同一 rollout batch 的全部 microbatch 反传结束后才做一次 optimizer update；下一批重算教师。

**没有独立 teacher 权重，也没有动态生成 reference。** 基座冻结，当前 LoRA 同时参与学生与自教师前向。`current_batch_stop_gradient` 是当前实现支持的策略。

### 2.2 数据划分与固定输入池

| 集合 | 正式默认数量 | 用途 |
|---|---:|---|
| train | 10,000 | 训练候选题集合 |
| validation | 512 | 独立验证划分 |
| heldout_probe | 512 | 不参与训练的固定监测题 |
| train_probe | 64 | train 的固定子集，独立统计 |

train / validation / heldout_probe 不重叠；train_probe 是 train 子集，不另加训练样本。200 step 正常执行时是 1,600 次训练问题呈现、12,800 条训练回答，并不遍历全部 10,000 题。train_probe 表示属于训练划分，不保证其中每题都在这 200 step 内被访问。

按归一化题干 hash 去重，再按 topic 和 difficulty 分层冻结 ID；topic 默认采用层级中的第 2 级。默认要求 reference 最终答案通过同一数学 verifier，**不等同于中间推理经人工验证**。超长 prompt 和未核验参考整题排除，不截断 reference，也不自动减少请求数量。筛选后数量不足会报错。

| 输入池 | 建立方式 | 解释 |
|---|---|---|
| fixed_base_rollout | step0 的初始 Base 在固定题上生成一次；T=1、top_p=1、每题固定 seed，最多256 tokens | 跨 checkpoint 固定真实 prompt/response IDs，主要观察权重变化 |
| live_rollout | 每个监测 checkpoint 在同题上重新生成，默认 greedy，最多256 tokens | 观察准确率、格式、长度、截断和访问轨迹；与固定池分开 |
| fixed_reference | 可选，默认关闭 | 已核验参考的 teacher-forced 分析，不是自主解题能力 |

### 2.3 指标、时间点与三个不同的 k

| 指标 | 默认支持集/定义 | 写入频率 |
|---|---|---|
| overlap ratio / Overlap-Token Advantage | 学生和教师词表 top-16 的交集；不改变 full-vocab loss | 每个 rollout batch |
| student / teacher entropy、entropy gap | 全词表，gap 为逐位置绝对差再平均 | 每个 rollout batch |
| RSA / RSM、rho与RMSE、PR、CKA | 相同 pool / 题ID / block / anchor | 每个监测 checkpoint |
| token轨迹曲率、首层相对量、训练变化 | 完整 block 输出，float32作差，单位degrees；不跨prompt、EOS、padding | 每个监测 checkpoint |
| 题目表示 kNN overlap | 原始向量的 Euclidean kNN，k=8/16/32，排除自身 | 每个监测 checkpoint |
| 固定状态 entropy、KL、词表 overlap/shared mass | T=1、完整词表；词表k=10/50/100 | 每个监测 checkpoint |
| live accuracy、长度、EOS、截断、抽取失败 | 截断/抽取失败保留并计错 | 每个监测 checkpoint |

训练前向 KL 是 `KL(teacher || student)`；固定状态诊断另算 `KL(student || teacher)` 和 `KL(student || initial)`，不能混为同一 loss。

线上指标使用 step_s **更新前**参数，日志的 `opsd/teacher_parameter_version` 为 s-1；离线 step_s 分析读取 **更新后**checkpoint。几何在完整 decoder block 输出上提取，层号从1起；anchor 为 prompt_last / response_32 / response_128。短文本、空交集、零位移、N<=k 等未定义量保留 null / undefined 和 N，不补零。

监测步数默认 `[0,1,5,10,20,50,100,200]`；教师几何默认只在 `[0,20,100,200]`。常规 checkpoint 还每50 step保存一次。**默认 `training.checkpoint_retention=latest`：保存完整 checkpoint 和标量后暂停训练，由独立进程完成当前监测点分析，保留表格/图/manifest，再清理旧 checkpoint 并继续训练。** 分析不进入训练反传图；分析期间训练 worker 的显存可能仍驻留，单卡峰值须由2-step验收实测。

## 3. 上传和选择正确入口

服务器应拉取本项目的 GitHub 仓库，进入其中的 `Rethinking-OPD-main/`，不能只下载原论文仓库或只上传 shell。本地 `research_runs/` 被 gitignore，已有 CPU 产物可选上传，不能当正式结果。

```bash
git clone git@github.com:zlysunine/opd_project.git
cd opd_project/Rethinking-OPD-main
git rev-parse HEAD
```

已有 clone 时，先确认并保留服务器自己的修改，再更新到交接提交；不要直接覆盖工作树。以下命令中的“仓库根目录”指包含 `run_base_comparison.sh` 的目录，不是外层 `opd_project/`。

```text
Rethinking-OPD-main/
  PILOT_HANDOFF_zh.md
  run_base_comparison.sh
  analysis/
  configs/analysis_deepmath.json
  configs/research/                  # 含Base profile、继承配置、plain Jinja
  scripts/__init__.py
  scripts/research/
  tests/research/
  docs/research/
  verl/                              # 本仓库修改的源码、模型配置和merger
  grpo.sh, on_policy_distillation.sh  # 原入口保留
  run_opsd_experiment.sh, opsd_pipeline.sh
```

| 入口 | 用途 |
|---|---|
| `run_base_comparison.sh` | **本阶段唯一一键入口，默认OPSD** |
| `python -m scripts.research.compare` | 同一 Base 实验的 configure / prepare / train / analyze / report |
| `run_opsd_experiment.sh`、`opsd_pipeline.sh` | 旧版 Qwen3-1.7B non-thinking 实验，不能替代本阶段 |
| `on_policy_distillation.sh` | 原仓库外部教师 OPD，保留原行为 |
| `grpo.sh` | 原仓库 GRPO，保留但当前不执行 |

旧版 non-thinking 的完整操作记录在 [PIPELINE_zh.md](docs/research/PIPELINE_zh.md)。不要把其中的 `--profile`、`--overlength`、`--max-problems`、`--with-empty` 等参数套到本阶段入口。

## 4. 服务器环境与输入：只配置一次

### 4.1 硬件、系统与软件版本参考

下表是服务器配环境的起点，**不是经过真实 Qwen GPU 训练验收的锁定环境**。优先复用已有兼容环境；新环境按下文安装，再做第5节的2-step验收。

| 项目 | 当前实验的环境参考 | 核对点 |
|---|---|---|
| 系统 | Linux x86_64；例如 Ubuntu 22.04 | 下文 CUDA wheel 路径面向 Linux/NVIDIA；不要直接套到 macOS、Windows 或 ARM |
| GPU | 默认1张 A800 80GB，支持 BF16 | `CUDA_VISIBLE_DEVICES` 只暴露分配到的卡；单卡容量是否足够由2-step实际峰值决定 |
| CPU / 内存 | 规划16个CPU核、128GB主存 | 是为 offload、checkpoint 导出和分析预留的资源建议，不是已测最低配置；注意容器/作业的实际内存限额 |
| 磁盘 | 模型、数据、缓存和产物放可写的本地 SSD；按pilot实测峰值预留余量 | 默认只留最新完整checkpoint；保存/分析期间短暂存在新旧两份checkpoint及一份临时HF导出。分析表仍会累积；不能只按LoRA文件估算 |
| Python | 推荐独立 Conda/venv 环境，Python 3.12 | 仓库自带 FlashAttention wheel 名含 `cp312`；不要混用系统pip和环境python |
| Torch / vLLM | 参考 `torch==2.8.0`、`vllm==0.11.0` | vLLM 0.11.0 的CUDA依赖指定Torch2.8.0，不能单独升级Torch |
| CUDA / 驱动 | 与选用的Torch/vLLM二进制匹配；vLLM 0.11.0标准预编译包为CUDA12.8 | 不能只看 `nvidia-smi` 中的CUDA字样；还要检查 `torch.version.cuda` 和真实kernel |
| Transformers | 参考 `4.56.1`；研究依赖要求 `>=4.51.0,<5` | 低于4.51不识别Qwen3；不要自动升级到5.x |
| PEFT / FlashAttention | 参考 `peft==0.17.1`、`flash-attn==2.8.1` | LoRA和attention扩展均要实际导入；FlashAttention须匹配Python、Torch、CUDA及C++ ABI |
| NumPy / PyArrow | 参考 `numpy==1.26.4`；`pyarrow>=19` | 本仓库verl要求NumPy<2；正式分析默认写Parquet，不能缺PyArrow |
| Ray / 训练数据流 | `ray[default]>=2.41`，采用vLLM0.11时还须满足其`ray[cgraph]>=2.48`；TorchData；`tensordict>=0.8,<=0.10,!=0.9` | 用pip统一解析依赖，检查 `torchdata.stateful_dataloader` 可导入 |
| 配置 / 数学 / 分析 | Hydra、OmegaConf、pandas、pylatexenc、SymPy、SciPy、scikit-learn、Matplotlib等 | 完整分析依赖见 `scripts/research/requirements-analysis.txt`；训练依赖见 `verl/setup.py` |
| 本地verl | 本工作树 `verl/` 的 editable安装 | 不用另一份PyPI verl替代；训练与分析进程都应导入本仓库源码 |

版本依据：[vLLM 0.11.0 CUDA依赖](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/cuda.txt)、[vLLM 0.11.0 GPU安装说明](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/gpu.html)、[FlashAttention 2.8.1安装说明](https://github.com/Dao-AILab/flash-attention/tree/v2.8.1#installation-and-features)、[Qwen3-1.7B-Base模型说明](https://huggingface.co/Qwen/Qwen3-1.7B-Base#requirements)。这些上游说明不能替代本仓库的GPU集成验收。

`nvidia-smi` 显示驱动及其CUDA兼容能力，`torch.version.cuda` 表示Torch构建使用的CUDA版本，`nvcc --version` 则表示当前CUDA Toolkit编译器。运行预编译包通常不需本地nvcc；源码编译FlashAttention时需要匹配的Toolkit、C++编译器和ninja。驱动太旧时选择匹配的整套二进制或由服务器管理员处理驱动，不通过随意替换单个CUDA包解决。

本阶段采用FSDP和vLLM，不需要Megatron、SGLang或LLaMA-Factory；没有外部teacher模型下载任务。默认logger为console，不需要W&B登录。已有模型和原生DeepMath文件时，实验可从本地路径读取；安装依赖或下载原始资产所需网络应在开跑前解决。

### 4.2 选择Python环境与安装依赖

已有兼容环境直接激活，并在仓库根目录执行：

```bash
cd /your/path/Rethinking-OPD-main
conda activate opsd
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PWD:$PWD/verl${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="$PWD/research_runs/.matplotlib"
command -v python
nvidia-smi
```

新入口直接调用 PATH 中的 `python`，**不读取旧脚本的 OPSD_PYTHON / OPSD_WORKDIR**；用激活环境选择解释器，用 `--workdir` 选择产物目录。

下面是供服务器Codex参考的新环境安装顺序，只在新建的实验环境中执行。版本组合尚未在本机GPU验证；如果pip报冲突，先定位并记录，不能跳过依赖检查。这里用Conda管理Python、用 `python -m pip` 安装Torch/vLLM，避免混装另一套Conda Torch。

```bash
conda create -n opsd python=3.12 -y
conda activate opsd
python -m pip install --upgrade pip setuptools wheel

# 在包含run_base_comparison.sh的目录执行，先装匹配的Torch/vLLM。
python -m pip install "torch==2.8.0" "vllm==0.11.0"

# 一次解析本地verl和分析依赖，显式保持核心版本。
python -m pip install -e ./verl \
  -r scripts/research/requirements-analysis.txt \
  "torch==2.8.0" "vllm==0.11.0" "transformers==4.56.1" \
  "peft==0.17.1" "numpy==1.26.4" packaging ninja psutil
```

FlashAttention单独安装，安装前先记录实际ABI：

```bash
python - <<'PY'
import sys
import torch
print('Python:', sys.version)
print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda)
print('C++11 ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)
PY

# 优先使用与上述信息完全匹配的官方wheel；以下命令也可能触发源码编译。
# 若触发编译，先确认nvcc/GCC可用，CUDA Toolkit与Torch构建匹配。
MAX_JOBS=4 python -m pip install "flash-attn==2.8.1" --no-build-isolation
python -m pip check
```

仓库原有 `verl/scripts/install_vllm_sglang_mcore.sh` 可作为版本线索，但不直接替代以上检查：它硬编码的FlashAttention wheel是 `cp312/cu12/torch2.8/cxx11abiFALSE`，**ABI为True时不能使用该wheel**；默认还装本实验不用的后端，而且没有严格的失败即停保护。即使它最后打印Successfully，也必须检查真实import、CUDA kernel和训练。使用该脚本时至少先审阅内容、显式禁用Megatron/SGLang，并按实际ABI处理FlashAttention。

以后单独补装/升级依赖时仍应保持上述核心版本；安装结束的 `pip freeze` 才记录实际解析出的完整环境。不要把分析依赖文件当作已验证的全套训练锁文件。

### 4.3 填写真实路径、缓存和版本

```bash
export OPSD_MODEL_DIR="/models/Qwen3-1.7B-Base"
export OPSD_MODEL_REV="ACTUAL_MODEL_COMMIT"
export OPSD_DATA_FILE="/data/DeepMath-103K/train.parquet"
export OPSD_DATA_REV="ACTUAL_DATA_COMMIT"
export OPSD_CONFIG="$PWD/configs/analysis_deepmath.json"
export OPSD_PILOT_DIR="$PWD/research_runs/base_opsd_acceptance_2step"
export OPSD_RUN_DIR="$PWD/research_runs/base_opsd_200"
export OPSD_LOG_DIR="$PWD/research_runs/server_handoff_logs"
mkdir -p "$OPSD_LOG_DIR"
```

可按服务器磁盘布局配置缓存，以下目录需有足够空间且可写。作业调度器已设置的变量优先保留；不要覆盖分配到的GPU编号。

```bash
export OPSD_CACHE_ROOT="${OPSD_CACHE_ROOT:-$PWD/research_runs/.cache}"
export HF_HOME="${HF_HOME:-$OPSD_CACHE_ROOT/huggingface}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$OPSD_CACHE_ROOT/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$OPSD_CACHE_ROOT/triton}"
export RAY_TMPDIR="${RAY_TMPDIR:-$OPSD_CACHE_ROOT/ray}"
mkdir -p "$HF_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$RAY_TMPDIR" "$MPLCONFIGDIR"
df -h "$OPSD_CACHE_ROOT" "$OPSD_LOG_DIR"
free -h
```

Ray临时目录宜选本地SSD上的短路径；路径过长时改用例如 `/tmp/opsd_ray`。容器运行需要GPU设备透传和足够的 `/dev/shm`，可用 `df -h /dev/shm` 检查。单机默认不需要手动 `ray start`；不要为此停止共享服务器上其他人的Ray进程。`--execute`默认建立独立后台会话，SSH断开不终止任务；无需tmux/nohup。加`--foreground`才在前台执行。状态、日志、停止与恢复见[训练控制说明](docs/research/TRAINING_CONTROL_zh.md)。

替换所有示例路径和 `ACTUAL_*` 为真实来源版本，不填写 main/master，不编造版本。模型目录需含完整权重、config、tokenizer；不要拿 Instruct 权重改目录名冒充 Base。manifest 记录本地文件hash，但来源声明仍须真实。

**本阶段 Base CLI 的 `--input` 需要单个原生 Parquet / JSONL（也支持JSON列表）文件，不能直接给目录。** 多分片先在输入准备阶段整理为单文件，保留字段和转换来源；不要使用旧版目录参数。必需字段为 `question`、`final_answer`、`difficulty`、`topic`、`r1_solution_1`。仓库自带 `datasets/DeepMath_deduped.parquet` 缺少难度和R1参考，不符合要求。

### 4.4 检查真实源码、CUDA kernel并记录环境

先检查实际Python和本地verl路径，再测试BF16和FlashAttention前向/反向。以下会在GPU上分配少量测试张量，不加载Qwen权重、不启动训练：

```bash
set -o pipefail
python - <<'PY' 2>&1 | tee "$OPSD_LOG_DIR/cuda_kernel_check.log"
import importlib
from pathlib import Path
import sys
import torch
import verl
from flash_attn import flash_attn_func
from transformers import Qwen3ForCausalLM
from vllm.lora.request import LoRARequest

expected = (Path.cwd() / 'verl/verl').resolve()
actual = Path(verl.__file__).resolve().parent
print('Python:', sys.executable)
print('verl:', actual)
assert actual == expected, f'verl不是当前仓库源码：{actual}'
assert torch.cuda.is_available(), 'CUDA不可用'
assert torch.cuda.device_count() == 1, '默认实验需要恰好1张可见GPU；核对CUDA_VISIBLE_DEVICES'
assert torch.cuda.is_bf16_supported(), '当前GPU不支持BF16'
print('Torch/CUDA/ABI:', torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)
print('GPU:', torch.cuda.get_device_name(0), 'capability:', torch.cuda.get_device_capability(0))
importlib.import_module('vllm._C')
q = torch.randn(1, 32, 2, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
out = flash_attn_func(q, q, q, dropout_p=0.0, causal=True)
out.float().square().mean().backward()
torch.cuda.synchronize()
assert torch.isfinite(out).all() and torch.isfinite(q.grad).all(), 'attention输出或梯度非有限'
print('Qwen3/LoRA imports and BF16 FlashAttention forward/backward: PASS')
PY
```

然后调用仓库统一preflight，缺任一必需模块时先处理具体错误。它不加载模型权重；通过仍不等于FSDP/vLLM/LoRA集成已跑通：

```bash
python - <<'PY' 2>&1 | tee "$OPSD_LOG_DIR/environment_preflight.log"
import json
from scripts.research.preflight import check_environment
report = check_environment(gpus=1)
print(json.dumps(report, indent=2, ensure_ascii=False))
if not report['ok']:
    raise SystemExit('环境检查未通过，请先处理上面的具体错误')
PY
python -m pip check 2>&1 | tee "$OPSD_LOG_DIR/pip_check.log"
python -m pip freeze > "$OPSD_LOG_DIR/environment.freeze.txt"
python -m pip list --format=json > "$OPSD_LOG_DIR/environment.packages.json"
nvidia-smi > "$OPSD_LOG_DIR/nvidia-smi.txt"
git rev-parse HEAD > "$OPSD_LOG_DIR/code_commit.txt"
git status --short > "$OPSD_LOG_DIR/code_worktree.txt"
python - <<'PY' > "$OPSD_LOG_DIR/system.json"
import json
import os
import platform
import shutil
import sys
import torch
print(json.dumps({
    'python': sys.version, 'executable': sys.executable,
    'platform': platform.platform(), 'machine': platform.machine(),
    'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
    'cxx11_abi': torch._C._GLIBCXX_USE_CXX11_ABI,
    'nvcc': shutil.which('nvcc'),
    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'gpus': [{'name': torch.cuda.get_device_name(i),
              'total_bytes': torch.cuda.get_device_properties(i).total_memory}
             for i in range(torch.cuda.device_count())],
}, indent=2))
PY
```

环境freeze用于复现，也应记录安装命令和任何兼容性修复；不同CUDA/ABI机器不能仅凭同一份freeze保证二进制可用。若pip使用了带凭据的私有下载地址，分享freeze前需去除其中凭据。完整一键流程还会检查精确长度、实际Hydra配置和rollout schema，最终以2-step训练、checkpoint导出和分析产物验收为准。

| 常见环境问题 | 定位与处理 |
|---|---|
| `KeyError: qwen3` | 检查实际解释器与Transformers版本，按兼容版本安装，不换成Instruct |
| `undefined symbol` / `flash_attn_2_cuda`加载失败 | 对照Torch/CUDA/Python/ABI重新选wheel或编译；不能只重复安装相同的FALSE ABI wheel |
| `No module named torchdata.stateful_dataloader` | 核对TorchData安装和Torch兼容性，避免项目内同名模块遮蔽 |
| NumPy ABI / PyArrow错误 | 满足NumPy<2和PyArrow>=19，再确认pandas等二进制扩展兼容并运行pip check |
| Ray启动失败 / object store空间不足 | 查Ray日志、作业内存限制、临时路径、`/dev/shm`、本机端口；不停止他人的服务 |
| CUDA不可用 / vLLM kernel报错 | 查GPU分配、驱动、容器透传、Torch与vLLM构建版本，不能以CPU smoke通过替代 |

### 4.5 固定命令参数

首次提交时在同一Bash会话中定义共用参数；后台进程继承该环境。重新连接后激活环境，按实际workdir查看状态即可；若要重新提交，仍需恢复原参数数组：

```bash
set -o pipefail
BASE_ARGS=(
  --config "$OPSD_CONFIG"
  --model-dir "$OPSD_MODEL_DIR"
  --model-revision "$OPSD_MODEL_REV"
  --input "$OPSD_DATA_FILE"
  --dataset-revision "$OPSD_DATA_REV"
  --method opsd
)
```

## 5. 先做2-step GPU验收

使用独立新目录，不把测试目录接着改成200步：

```bash
PILOT_ARGS=(
  "${BASE_ARGS[@]}"
  --workdir "$OPSD_PILOT_DIR"
  --steps 2
  --monitor-steps 0,1,2
  --train-n 16
  --dev-n 8
  --probe-n 64
  --train-probe-n 8
)

# 只写配置并打印计划；不会生成probe或训练，但会创建workdir/config.json等。
bash run_base_comparison.sh "${PILOT_ARGS[@]}" \
  > "$OPSD_LOG_DIR/pilot_plan.json"

# 确认计划methods只有opsd后，提交后台验收任务；返回不代表验收完成。
bash run_base_comparison.sh "${PILOT_ARGS[@]}" --execute
bash run_base_comparison.sh status --workdir "$OPSD_PILOT_DIR" --watch
```

验收至少需要88个满足难度、参考核验、长度和去重要求的独立题，固定probe共72题。train_probe=8时其kNN的k>=8按规范undefined；heldout_probe=64可正常计算8/16/32邻域。Base短回答导致某些response anchor无有效题也应明确记录，不用伪造长文本。

## 6. 正式200-step一键执行

第7节验收通过后，在默认配置和另一个新目录执行：

```bash
RUN_ARGS=(
  "${BASE_ARGS[@]}"
  --workdir "$OPSD_RUN_DIR"
)

# 正式运行的计划：200step、method只有opsd。
bash run_base_comparison.sh "${RUN_ARGS[@]}" \
  > "$OPSD_LOG_DIR/formal_plan.json"

# 默认后台运行；SSH断开不停止。验收成功后才提交正式任务。
bash run_base_comparison.sh "${RUN_ARGS[@]}" --execute
bash run_base_comparison.sh status --workdir "$OPSD_RUN_DIR" --watch
```

实际流程为：

```text
configure（本地路径 / revision / 私有注册表）
  -> 环境检查
  -> prepare（筛选 / 去重 / 划分 / 初始固定rollout / 两组配置）
  -> train OPSD（训练 / 在线指标 / 保存后暂停、逐监测点独立分析、清理旧权重）
  -> analyze OPSD（校验已完成的分析缓存；失败时重试当前尚保留的checkpoint）
  -> report（读缓存汇总表格和图）
```

prepare会保存GRPO的配置文件，这是保留对照设计，**不会执行GRPO训练**。预览与实际调度均只选OPSD；本阶段命令不含 `--method both`。

### 正式默认参数在哪里调整

统一入口为 `configs/analysis_deepmath.json`：

| 区域/参数 | 默认值 | 修改方式 |
|---|---|---|
| training：mode / LoRA | lora / r16 / alpha32 / all-linear | 修改JSON，新建实验目录 |
| learning_rate | 1e-6 | 修改JSON；没有 `--learning-rate` CLI |
| batch / rollouts / microbatch | 8 / 8 / 1 | 修改JSON |
| steps / gpus / seed | 200 / 1 / 42 | JSON或同名CLI；gpus需匹配CUDA_VISIBLE_DEVICES |
| epochs / save_every | 1 / 50 | 修改JSON |
| checkpoint_retention | latest | 先分析再清理，保留各监测点结果；all恢复先训练后分析并保留全部权重 |
| prompt / response / teacher prompt上限 | 512 / 512 / 2048 | 修改JSON；Base入口无对应长度CLI |
| sampling_temperature / teacher_temperature | 1 / 1 | 修改JSON |
| dataset：train_n / dev_n | 10000 / 512 | JSON或 `--train-n` / `--dev-n` |
| monitor：heldout_probe_n / train_probe_n | 512 / 64 | JSON或 `--probe-n` / `--train-probe-n` |
| monitor：seed / response窗口 | 20261026 / 256 | 修改JSON，与训练seed独立 |
| optimizer_steps | 0,1,5,10,20,50,100,200 | JSON或 `--monitor-steps` |
| 其他几何/分布指标设置 | 见JSON | 修改JSON，新建实验目录 |

Base入口目前没有 `--batch`、`--rollouts`、`--lora-rank`、`--profile`、长度或学习率CLI；这些参数用JSON修改。在线top-16来自 `base_geometry.json -> pilot_1p7b.json` 的继承；需要改变时在Base profile中明确覆盖并记录。

`configure` 后实际读取 `<workdir>/config.json`。修改模板不会更新已有运行；配置不一致会拒绝复用目录。改变科学配置应复制为新的配置文件，传 `--config` 并选择新 `--workdir`，不手工篡改被冻结的probe或已训练目录。

## 7. 服务器Codex如何验收、看哪些文件

所有路径相对于本次 `<workdir>`；正式目录用 `$OPSD_RUN_DIR`，2-step目录用 `$OPSD_PILOT_DIR`。

| 文件 | 必须核对的内容 |
|---|---|
| `config.json`、`models.json` | Base路径/revision、LoRA、前向KL、batch/n/steps和seed |
| `environment.json`、外部 `environment.freeze.txt` | 实际依赖与CUDA检查，保存实际GPU/驱动信息 |
| `shared/probe/manifest.json`、`complete.json` | 划分数量、版本/内容身份、固定排序和题ID |
| `shared/probe/eligibility.jsonl`、`duplicates.jsonl` | 参考核验、长度筛选和去重记录 |
| `shared/probe/fixed_pool.jsonl` | 原始prompt/response IDs、逐题seed、anchor有效性 |
| `opsd/training_config.json`、`analysis_config.json` | 实际训练与分析配置；当前方法为OPSD |
| `.session/attempts/<id>/console.log`、`state.json`、`progress.json` | 本次后台任务的控制台、进程心跳/退出码和当前阶段；历史attempt保留 |
| `opsd/train/train.log` | 完整训练日志；报错首先定位这里 |
| `opsd/train/metrics.jsonl` | 每step的metrics字典、预算、forward-KL loss与教师版本 |
| `opsd/train/config_preflight.json`、`*preflight.log` | 实际Hydra/rollout schema验收 |
| `opsd/train/plots/thinking_consistency.png`、`.csv` | 五项在线指标曲线与数据 |
| `opsd/train/checkpoints/global_step_*/` | 正常完成后仅最新一份完整模型/optimizer/scheduler/RNG/dataloader恢复状态；保存/失败现场可能暂留新旧两份 |
| `opsd/jobs/step_*.json`、`job_status/step_*.json` | 队列optimizer step、checkpoint、probe hash；最终status=complete |
| `opsd/initial_adapter.json` | step0的LoRA B=0及实际adapter指纹 |
| `opsd/hf/solution/global_step_*/` | 临时BF16合并权重，成功分析后删除；分析失败时保留供重试 |
| `opsd/checkpoint_maintenance.jsonl` | 保存后分析与轮换耗时；从训练wall预算中扣除 |
| `opsd/analysis/steps/step_*/` | anchors、neighbors、RSM、逐题曲率、分布、live回答、指标和manifest |
| `report/metrics.csv`、`.jsonl`、`.parquet` | 按step/layer/pool/split/anchor/role区分的汇总 |
| `report/*.png`、`report_manifest.json` | 曲率/RSA/NO/PR热图与曲线、行为对照、分布曲线及来源 |

验收标准：

1. 2-step训练到step2，仅保留step2完整checkpoint，但保留step0/1/2全部分析结果和3个完成的job；正式训练到200，仅保留step200完整checkpoint，保留全部8个监测step结果。日志能找到 `actor/forward_kl` 及五个 `opd/` 必需字段。
2. 五项字段为 `opd/overlap_ratio`、`opd/overlap_token_advantage`、`opd/student_entropy`、`opd/teacher_entropy`、`opd/entropy_gap`。空交集advantage为null时同时检查覆盖率，不要求所有undefined量变成有限值。
3. 数据、固定pool和anchor题ID跨checkpoint一致；heldout / train_probe 分开，fixed / live 分开，学生 / privileged_teacher 分开。
4. `anchors.npz`、`anchors.json`、`curvature.parquet`、`distributions.parquet`、`neighbors.npz`、`rsm_*.npz` 和 `metrics.parquet` / JSONL 存在，step目录有 `complete.json`。state位置、block hook和N可追溯。
5. 有live accuracy、长度、EOS、截断、抽取失败记录；性能为0或负增益也保留，不通过换模型隐藏。
6. 新目录中可以有 `grpo/training_config.json` 和 `grpo/analysis_config.json`，但不应出现本次GRPO训练、checkpoint或分析任务。
7. 最终汇报实际训练耗时、独立分析耗时、GPU型号/可见数和可用的显存峰值测量。没有测量就标未测，不用理论估算冒充实测。

可在另一终端或重新SSH连接后观察训练（先激活同一环境；workdir使用实际运行目录）：

```bash
bash run_base_comparison.sh status --workdir "$OPSD_RUN_DIR" --watch
bash run_base_comparison.sh logs --workdir "$OPSD_RUN_DIR" --stream train --follow
```

Ctrl+C只关闭查看器，不停止后台任务。停止须显式使用`stop --workdir`。`--watch`只在整个受管命令成功时返回0；训练到200/200后仍可能处于分析或报告阶段。

开始prepare时还没有train.log，这是正常的。使用`logs --workdir ... --follow`查看完整流程控制台；训练详细输出查看`--stream train`。状态读取真实metrics/checkpoint/job记录并显示阶段与更新时间。后台启动命令返回只表示提交成功；退出码和完成状态以`status`为准，不能只看PID存在或PNG判断成功。

## 8. 失败恢复和分阶段排查

先保留原日志和checkpoint，修复具体问题后重试**相同配置和同一workdir**。不要清空运行目录、忽略hash校验或直接覆盖complete标记。

```bash
# 环境变量、BASE_ARGS、RUN_ARGS仍为第4/6节的同一组设置时：
bash run_base_comparison.sh "${RUN_ARGS[@]}" --execute

# 已有prepare结果，仅恢复/继续OPSD训练：
python -m scripts.research.compare train \
  --workdir "$OPSD_RUN_DIR" --method opsd --execute

# 训练已经完成，仅重试checkpoint导出和分析（此独立命令仍前台执行）：
python -m scripts.research.compare analyze \
  --workdir "$OPSD_RUN_DIR" --method opsd

# 不重新训练或分析，仅从缓存重画汇总：
python -m scripts.research.compare report --workdir "$OPSD_RUN_DIR"

# 仅重画已有在线训练日志：
python -m scripts.research.run plot \
  --input "$OPSD_RUN_DIR/opsd/train" --output "$OPSD_RUN_DIR/opsd/train/plots"
```

训练用resume=auto，从最后保存的checkpoint恢复；尚未保存的更新需要重做。已完成200步时入口会复用完整checkpoint，避免额外更新第201步。分析按step顺序，必须先建立step0基线；完成缓存通过job/config/complete收据、源码、probe与产物hash校验后复用，不依赖已清理的旧权重。分析失败时停止训练并保留新旧checkpoint和当前HF导出；同配置恢复会先重试分析再清理。已删除的旧权重不能重新训练/重新提取，只能复用其完整分析结果；若要修改分析配置重新提取，需另行保留全部权重并新建运行目录。硬中断留下临时目录时先检查，不能伪造完成标记。

| 现象 | 应对 |
|---|---|
| 模型路径/revision缺失、误用了Instruct | 确认真实来源和完整Base文件，不改名替代 |
| 依赖/FlashAttention ABI错误 | 按environment报告和import错误修复当前环境，保留版本记录 |
| 有效题不足 | 查看筛选与数量；失败发生在prepare结束前时临时报告可能未发布，结合控制台定位。调整科学设置则另建配置/目录，不能静默缩小N或截参考 |
| OOM | 记录发生阶段和实际配置；先定位占用与资源配置。不得悄悄改全词表loss为top-k或更换模型 |
| 配置、probe、源码hash不符 | 核对原始版本；恢复对应代码/数据，或建立明确的新实验目录，不绕过检查 |
| checkpoint缺分片、LoRA adapter缺失/非零B | 检查原始保存与step0状态，不把未合并的基座当训练结果 |
| 只有训练曲线，没有表征报告 | 查看analyze是否执行、job_status是否failed，再单独重试analyze/report |

默认分析时训练暂停，使用独立进程和原GPU分配，完成后继续；不是后台并发训练。训练worker显存可能仍驻留，先做2-step实测。不要另启analyze与默认同步分析竞争同一workdir；手动analyze只处理调用时已有jobs快照。

## 9. GRPO保留但暂不执行

GRPO将来使用同一Base起点、共享数据和固定probe，LoRA初始化与OPSD对齐，用数学verifier outcome reward训练。两种方法的监督和计算成本不同，同step不能直接视为同预算。

代码保留显式 `--method grpo` / `--method both`，**本阶段不调用这些选项**。后续用户要求开展对照时，再用已有设计单独执行并分析，不需为本轮删改GRPO代码或原始 `grpo.sh`。

## 10. 已验证范围与服务器最终交付

本地已运行69项研究测试，以及此前11项OPD指标测试；CPU tiny GPT-2 的12个合成问题、2个权重状态已完成提取/缓存/报告。step1是参数扰动，不是训练。详情见 [VALIDATION_zh.md](docs/research/VALIDATION_zh.md)。

**真实Qwen GPU训练、FSDP保存恢复、实际PEFT合并、Parquet写入、显存与耗时尚未在本机验证。** 本地缺少可用CUDA及多项生产依赖。因此服务器Codex需要实际做2-step验收，不能照抄本地测试通过就宣称200步已跑通。

服务器完成或受阻时，按以下格式交付：

```text
执行范围：OPSD only；GRPO未运行
实际模型路径/revision/hash：
实际数据文件/revision/hash及筛选数量：
实际配置、代码版本、环境freeze路径：
2-step验收：通过/失败，日志与产物路径
正式训练：完成step / 200，最后完整checkpoint
在线五项指标与forward-KL曲线路径：
完成的分析step列表、失败/undefined指标及原因：
表征/分布/行为数据表、图、manifest路径：
训练耗时 / 分析耗时 / GPU及显存测量：
未完成部分与具体错误：
```
