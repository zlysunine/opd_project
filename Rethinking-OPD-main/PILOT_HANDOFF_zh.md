# §3.1 Thinking-Pattern Consistency 小批量验证：服务器交接文档

交接日期：2026-10-19；动态指标更新：2026-10-20。目标：用总预算上限 30 元寻找早期信号，建立可复查的小实验流程。它不是原论文完整复现，也不保证在该预算内出现准确率差距。

## 1. 接手后运行哪一个入口

把**整个 `Rethinking-OPD-main` 目录**上传到服务器。进入目录后，一条命令完成安装、权重下载、检查、训练和评估：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_pilot_server.sh \
  --gpus 2 --billable-gpus 2 --price 5.59 \
  --budget 30 --already-spent 0 --reserve 5
```

`5.59` 是计算示例，务必换成租用页面的**实际每卡每小时人民币价格**。`already-spent` 填执行命令之前的开机费用；如果已经开机 10 分钟，2 卡、5.59 元/卡时，约填 `1.87`。脚本不会自动读取云平台账单。

默认自动创建 `.pilot/env`，无需提前 `conda activate`。系统没有 Python 3.12 时，会通过项目目录中的 uv 自动安装受管理的 Python 3.12，不修改系统 Python。安装范围局限在本项目环境及缓存；不会安装 GPU 驱动，也不会启动额外的收费服务。

**推荐先把下面的第 3 节读完再开付费 GPU。** 从空环境下载依赖和约 20GB 模型权重可能占去大部分预算；剩余时间不足时，新入口会停止并保留缓存，不会为了跑完继续花钱。

## 2. 服务器条件和必须上传的文件

| 项目 | 本入口的支持范围 |
|---|---|
| 系统 | Linux x86_64，建议 Ubuntu 22.04/24.04 镜像；启动用 Python ≥3.10，具有 pip、venv，或允许 uv 下载 Python |
| GPU | 默认独占 2 张 80GB 级 NVIDIA 卡，例如 A800/A100 80GB |
| 驱动 | 能运行 PyTorch 2.8.0 CUDA wheel；启动后会执行实际 CUDA/FlashAttention 小算子检查 |
| CPU 内存 | 建议至少 64GB，128GB 更宽裕；原始 8 卡脚本的 500GB 不是这次已测定的最低需求 |
| 磁盘 | 建议开始时至少 120GiB 可用；空缓存启动会检查约 100GiB 余量，已有权重时检查约 55GiB |
| 网络 | 可访问所用 pip 源、Hugging Face、GitHub release；下载本身不需要云账户凭据 |

上传后应包含：

```text
Rethinking-OPD-main/
├── run_pilot_server.sh                     # 新的一键在线入口
├── pilot_thinking_consistency.sh           # 只训练的离线入口
├── scripts/pilot/                         # 调度、下载、预算与报告代码
├── verl/                                 # 本仓库的定制 verl，不能换成任意最新版
└── datasets/
    ├── dapo-math-17k.parquet
    └── test_data/AMC23/test.parquet
```

数据已在项目中提供，因此不再猜测下载另一份同名数据。入口会先检查两个文件是否是真正的 Parquet；如果是 Git LFS 指针或上传漏文件，会在下载大模型之前停止。**只上传几个 launcher 不够。**

环境核心版本固定为：Python 3.12、PyTorch 2.8.0、vLLM 0.11.0、Transformers 4.56.2、FlashAttention 2.8.1。其余安装范围见 `scripts/pilot/requirements-server.txt`，每次保存实际 `pip-freeze.txt`。这是固定核心版本的安装配方，尚非带哈希的全依赖锁文件。

vLLM 0.11.0 的官方依赖明确要求 torch 2.8.0、torchvision 0.23.0，Transformers 至少 4.55.2；新入口据此约束版本，避免自动装到不兼容的大版本。[vLLM CUDA 依赖](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/cuda.txt)、[公共依赖](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/common.txt)

FlashAttention 根据 PyTorch 的 C++ ABI 自动选择 TRUE/FALSE wheel，并校验已记录的 SHA256，不沿用原安装脚本固定 ABI=FALSE 的假设。不自动在付费实例上回退到源码编译。[官方 release assets](https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.8.1)

## 3. 正确的执行顺序

### 方案 A：先免费/低成本准备，再启动付费 GPU（优先）

第一步，在同一台服务器的 CPU 模式、尚未计 GPU 费用的环境，或者已有免费 GPU 资源上，运行：

```bash
bash run_pilot_server.sh --setup-only
```

该步骤自动创建环境、下载权重、检查数学评分和两组 Hydra 配置、准备固定数据，不调用 GPU 训练。准备时仍有时限，避免下载无限等待；如果所在实例本来就在收费，准备期间当然也会产生实际费用。

在真正免费的 CPU 环境准备时，日志按假定卡价换算的费用不是实际账单；切到新的付费 GPU 会话后，`already-spent` 只填写该付费会话已经产生的费用。

第二步，在保留同一磁盘的前提下开两张 GPU，再运行第 1 节的一键命令。已有环境及完整权重会复用，不重新下载。GPU 阶段额外检查原生模块导入、可见卡数、显存和 FlashAttention CUDA 算子。

如果 CPU 和 GPU 是不同服务器，需把项目与 `model/` 模型缓存传过去；**不要假定 Python 虚拟环境直接搬机器就能用**。目标服务器重新运行一键入口安装环境，或者使用同一已准备的镜像。

### 方案 B：在刚租到的 GPU 上从零一键运行

直接运行第 1 节命令即可。安装和下载时间自动计入本次估算预算。准备后若少于 30 分钟余量，脚本会停止、保留缓存并提示原因；30 分钟只是防止明显不可能完成的启动门槛，不是能跑完实验的保证。

### 只查看计划

```bash
bash run_pilot_server.sh --dry-run
```

不下载、不安装、不调用 GPU、不创建输出目录。

### 使用已有的训练环境

如果已经有**符合该配方**的环境，可以跳过自动安装，仍自动下载模型：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_pilot_server.sh \
  --python /absolute/path/to/environment/bin/python \
  --price 5.59 --already-spent 0
```

该 Python 必须是 3.12。脚本会检查版本与原生模块，环境不兼容就停止，不修改该环境。若要修复自动管理的环境，可加 `--reinstall`；不会删除已下载模型。

## 4. 自动下载什么，如何避免反复下载

| 角色 | 下载来源 | 默认本地位置 |
|---|---|---|
| 学生 | `Qwen/Qwen3-1.7B-Base` | `model/Qwen3-1.7B-Base` |
| 教师 A | `Qwen/Qwen3-4B` | `model/Qwen3-4B` |
| 教师 B | `Thinking-Space/Qwen3-4B-Base-GRPO` | `model/Qwen3-4B-Base-GRPO` |

每个模型首次解析实际 commit，先写入 `.pilot-source.json`，再下载该 commit 的 safetensors、配置、tokenizer 和模板文件。中断后再次执行会复用同一 commit 和 Hugging Face 的下载缓存；完整模型直接复用。脚本检查权重索引中的所有分片及 safetensors 数据边界，避免把未完成下载当成完整模型。这不是逐字节重新计算全部模型文件的密码学校验。

运行目录另存 `model_sources.json`，记录本次实际来源，方便复核。不要将两个教师指向同一目录，不要把 Qwen3-4B-Base 当作 Qwen3-4B Non-thinking。

下载机制使用官方 `snapshot_download(..., local_dir=..., revision=commit)`。[Hugging Face 下载说明](https://huggingface.co/docs/huggingface_hub/guides/download)

若官方 endpoint 无法连接，先在服务器解决网络；可以通过 `HF_ENDPOINT` 明确指定自己信任的兼容 endpoint。脚本不会静默切换下载源。公共模型通常无需登录；确需令牌时通过环境变量 `HF_TOKEN` 提供，不要写入脚本或发到日志。

## 5. 实际训练流程

```text
校验文件与预算
  → 创建/复用隔离环境
  → 检查依赖、CUDA、数学评分、两组训练参数
  → 下载/复用三组模型
  → 固定抽样、核对 tokenizer token ID
  → 评估初始学生
  → 从初始学生训练 A（Non-thinking 教师）
  → 重新从同一初始学生训练 B（GRPO 教师）
  → 选择两组最大共同完整 checkpoint
  → 合并分片、分别独立评估
  → 汇总费用、分数、overlap 与不确定性
```

不能把 A 训练出来的学生接着给 B 训练：两组必须独立初始化。也不能拿 A 的第 30 step 和 B 的第 10 step 直接比较。

默认配置：

| 项目 | 值 |
|---|---|
| 数据池 | 固定 seed=42，抽取 512 条 DAPO prompt |
| 每组实际 prompt 消耗 | 最多 30×8=240 条；不重复、不基于结果挑题 |
| batch / rollout | 每步 8 个 prompt，每题 2 条回答，共 16 条轨迹 |
| 优化 | 全参数 OPD，lr=1e-6；top-16、only_stu、student_p |
| 长度 | prompt≤1024，response≤3072 |
| checkpoint | 每 10 step 保存模型；两组分别最多 30 step |
| 验证 | 固定 seed 抽取 64 道 AMC23，每题 2 次采样，温度 0.7、top-p=0.95 |
| 模板 | 学生 tokenizer、enable_thinking=False；两位教师评分同一组学生前缀 |

保留 FP32 主权重/优化器状态，计算沿用原工作器的混合精度路径。降低前向 token 上限与每卡 micro-batch，未把全参数训练替换成 LoRA。保存产物仅供评估，不含完整优化器状态，不能精确恢复被打断的训练。

## 6. 30 元的预算如何执行

```text
可用秒数 = (总预算 - 启动前已花费用 - 预留费用)
           / (全部付费卡数 × 实际单卡每小时价格) × 3600
```

默认预算 30 元、预留 5 元、之前花费 0、两卡各 5.59 元/小时，对应约 **134.2 分钟**。**一键入口从安装前开始计时**，环境安装、下载、检查、训练、合并和评估全部占用这个时间；交给离线训练入口时，再扣除已经产生的准备费，不重置成一个新的 30 元预算。

内部两组训练各分配至多“准备后剩余实验时间”的 28%；其他时间留给初始评估、合并和最终评估。GPU/下载步骤失败就停止，不通过换教师或改题目来拼出好结果。

注意计费边界：

- 租了 8 张卡、只让代码看到 2 张，`--billable-gpus` 仍应填 8。
- 存储、快照、启动命令之前的空闲时间，脚本无法从平台自动得知，需填 `already-spent` 或从预留中覆盖。
- 脚本结束只代表计算进程停止，**不会自动关闭云实例**。完成后立即关机；最好另在云平台设置定时关机/余额限制。
- 这是按填写价格估算的限时控制，不能代替云平台账单硬限额。

## 7. 如何看结果，何时算初步有效

运行结束，控制台会打印输出目录，默认 `pilot_runs/年月日_时分秒/`。

| 文件 | 用途 |
|---|---|
| `summary.md` | 优先阅读：共同步数、准确率、截断率、配对区间、判定 |
| `overlap.svg` | 浏览器打开，查看两组随训练的 top-16 overlap |
| `dynamic_metrics.svg` / `dynamic_metrics.png` | 五联曲线：overlap、overlap-token advantage、学生熵、教师熵、绝对熵差 |
| `dynamic_metrics.csv` / `dynamic_metrics.json` | 两组逐 step 数值、定义、共同步数和初始/共同步数快照 |
| `dynamic_metrics_nonthinking.json` / `dynamic_metrics_grpo.json` | 分组逐 step 动态指标 |
| `eval_*.jsonl` | 原始回答、正确性、token 数、结束原因 |
| `train_nonthinking.log` / `train_grpo.log` | 训练 loss、overlap、响应长度、各阶段耗时 |
| `bootstrap_result.json` / `bootstrap_stages.json` | 从准备开始的估算费用、各阶段状态与耗时 |
| `config.json` / `data_manifest.json` | 实际训练参数、抽样题目索引、模板 |
| `model_sources.json` / `pip-freeze.txt` | 权重来源与实际依赖版本 |

**初步有效的联合信号**：在相同初始学生轨迹上，GRPO 教师具有更高初始 overlap；在相同更新步数的独立题目上，GRPO 教师组也更好。只有 overlap 更高，不能单独证明任务能力改善。

`summary.md` 的配对 bootstrap 区间按题计算；它不包含训练 seed 的不确定性。样本少、训练短、响应截断时，只能判断是否值得继续投入。

### §2.3 动态指标的计算口径

新版小实验默认启用 `+trainer.log_opd_dynamic_metrics=True`，每步在学生参数更新前采集。使用该步学生轨迹上的有效 response token，排除 prompt 和 padding；包括 EOS 在内的有效位置遵从 verl 的 response mask。各有效位置等权汇总，不先给每条回答等权平均。对数为自然对数，entropy/advantage 的单位为 nats。

记当前位置的学生/教师分布为 p、q，I=Top16(p)∩Top16(q)：

| 报告字段 / 日志键 | 定义 |
|---|---|
| `overlap_ratio` / `opd/overlap_ratio` | 对有效位置求 mean(|I|/16) |
| `overlap_token_advantage` / `opd/overlap_token_advantage` | 对交集内分别归一化得到 p̄、q̄；先计算 Σ p̄(v)[log q̄(v)−log p̄(v)]/|I|，再对位置取平均 |
| `student_entropy` / `opd/student_entropy` | 全词表 H(p) 的有效位置均值 |
| `teacher_entropy` / `opd/teacher_entropy` | 同一学生前缀上的全词表 H(q) 的有效位置均值 |
| `entropy_gap` / `opd/entropy_gap` | mean(|H(q)−H(p)|)，先逐位置取绝对值，再平均 |
| `overlap_nonempty_fraction` | 有效位置中 I 非空的比例，用于说明 advantage 的覆盖率 |
| `valid_response_tokens` | 该步参与统计的有效 response token 数 |

这遵从本地论文 §2.3、式 (6)–(8) 的定义；熵由已有工作器的全词表计算提供，再以 FP32 汇总，没有把 top-16 重新归一化后的熵冒充全词表熵。

**不能直接复用旧 `val-topk/adv_intersection` 作为论文 advantage。** 旧字段统计训练中的 advantage，交集归一化和平均口径不同。新版 `opd/*` 独立计算，保留原日志，不改变训练 reward 或 loss。

空交集时式 (7) 未定义：只从 advantage 均值中排除这些位置，同时报告非空比例；若整步全空，不记录 advantage，报表显示缺失，不能补成 0。其他指标仍覆盖所有有效位置。由于逐位置 gap 存在正负方向抵消问题，也不能从已有的两个平均熵倒推得到它。

指标复用已有 top-k log-prob、overlap mask 和全词表 entropy 张量，不增加师生模型前向、不额外生成轨迹。新增少量归一化、聚合和日志开销；原 30 元计时约束保持不变。GPU 端开销尚未实测。

图表展示到最大共同训练步数，完整 JSON/CSV 保留所有已记录步数。step 1 是第一次更新前的初始学生轨迹；step N 的指标也是第 N 次更新前，而 `global_step_N` checkpoint 保存的是更新后的权重。后期两组学生访问的前缀不同，因此曲线并非固定状态上的直接配对比较。

正常训练环境使用 Matplotlib 输出 SVG 和 PNG；无 Matplotlib 的精简 CPU 环境仍可重建 JSON/CSV，并回退输出 SVG，不生成 PNG。

### 已有日志重新生成图表

不租 GPU、不重新训练即可执行：

```bash
bash pilot_thinking_consistency.sh --summarize-only pilot_runs/你的实验目录
```

也可对从服务器下载到本机的结果目录使用该命令。它重建动态指标表格与曲线，保留已有费用和结论。若旧实验没有记录新版 `opd/*`，只能恢复已有的 overlap；无法从聚合日志补算交集归一化 advantage 和逐位置 gap，报表会明确显示缺失，需要新版训练采集。

数值解读：overlap-token advantage 理论上等于交集内的负 KL 再除以交集大小，通常≤0，趋近 0 表示交集内分布更接近；交集只有一个 token 时该值也会为 0，因此应结合 overlap 和覆盖率看。entropy gap 接近 0 只说明不确定性接近，不代表两模型把概率放在相同 token 上。最终仍需看独立题目表现。

以下均应记为**结果不足**：训练 OOM、未完成共同 checkpoint、评估超时、准确率都接近零、或较多回答到达长度上限。不要在这些情况下把失败写成“论文论点不成立”。

验证通过后，先补训练 seed 或增加到 60～100 step，再决定是否投入原论文的 200 step 和全量评估。不应同时扩大模型、长度、batch 和算法改动，否则难以判断变化来源。

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| 下载一直失败 | 查看 `download_models.log`；检查 endpoint、网络和磁盘，修复后用新输出目录重跑，权重缓存保留 |
| GitHub wheel 无法下载 | 查看 `install_flash.log`；先解决 release 网络，不在预算内盲目源码编译 |
| FlashAttention undefined symbol | 检查是否复用了错误 Python 环境；使用自动环境或 `--reinstall`，不要混装另一个 torch |
| `pip_check` 失败 | 看冲突包，优先修复隔离环境；不要忽略冲突直接训练 |
| GPU 数量不符 | 设置 `CUDA_VISIBLE_DEVICES=0,1`，并令 `--gpus` 与可见卡数一致 |
| 剩余时间不足而退出 | 缓存已保留；先关机，再用准备好的镜像/磁盘开新实例，正确填写新一轮已产生费用 |
| 训练 OOM | 先保留日志；当前 2×80GB 配方尚未实测，需检查其他进程、批量与 token 上限，不自动改精度换取“跑通” |
| 没有 `summary.md` | 若在安装/下载阶段停止，先看 `bootstrap_result.json`；尚未开始训练就不会有训练汇总 |
| 终端可能断线 | 可在服务器已有 tmux 中执行；不希望终端断线中止时，先安装/使用终端会话管理工具 |

## 9. 已验证内容和待服务器验收内容

本地完成 shell/Python 语法、预算换算、进程超时清理、共同 checkpoint 选择、报告判定，以及模拟下载缓存/分片检查等 CPU 测试。没有在本地安装整套 GPU 依赖、下载约 20GB 权重或运行 GPU 训练。

因此，一键入口实现了完整执行流程，但**还不能声称已在目标服务器端到端跑通**。不同驱动、网络和包镜像仍需实际验收。服务器应依次确认：

1. `check_environment.log` 通过真实 CUDA/FlashAttention 检查；
2. 三个模型均记录来源 commit，`prepare.log` 通过；
3. 两组至少保存一个共同 checkpoint；
4. 初始学生和两组 checkpoint 均完成评估；
5. `summary.md`、`overlap.svg`、`dynamic_metrics.svg`、动态指标数值与原始回答完整；
6. 查看平台账单，保存需要的产物后立即关机。

本地 CPU 测试命令：

```bash
python -m unittest discover -s scripts/pilot -p 'test_*.py' -v
python verl/tests/utils/test_opd_dynamic_metrics_on_cpu.py -v
```
