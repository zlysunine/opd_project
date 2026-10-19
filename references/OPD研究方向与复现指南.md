# OPD 研究方向与第一篇论文复现指南

依据：本地两篇论文全文、`opd思考.md`、本地 `Rethinking-OPD-main` 的训练、奖励、评估代码，以及作者官方网页。整理日期：2026-10-16。本次只新增本文，没有修改原有代码、安装环境、下载模型或运行训练。下文命令供服务器使用，尚未经过 GPU 实测。

**1．先校准会影响选题的前提**

| 你的理解 | 判断 | 更准确的结论与依据 |
| --- | --- | --- |
| 更强的教师不一定更适合蒸馏 | 准确，但有适用范围 | 第一篇 §3、§4 的受控比较支持该现象。教师在基准上的能力、在学生前缀上的指导能力、学生对信号的可学习性，需要分别测量；不能把作者的解释当成所有模型的定理。 |
| top-k overlap 越高，蒸馏一定越好 | 需修正 | 它是诊断量；第一篇 §5.2 甚至出现 overlap 个数较少、共享概率质量较高、效果更好的设置。重合数量、重合质量、任务收益不能互相替代。 |
| 单 query 覆盖 71.5% 的状态 | 部分准确 | 第二篇 §4.1、附录 B.1：用教师最后一层的前缀末 token 隐状态表示状态；PCA 降到 50 维，K-means 取 K=200，统计访问过多少簇。不是精确前缀重合率，也不是覆盖了 71.5% 的推理能力。 |
| 16 query 接近全量数据 | 在论文设置内准确 | 语义多样的 16 个 query 在对应训练预算内达到 98.9% 的簇覆盖并接近全量效果；不能推广到任意 16 题、任意教师或任意训练时长。第二篇附录还显示数值随聚类粒度变化。 |
| 单题训练有效，说明全量训练时第一题贡献最大 | 需修正 | 单题重复训练和全量 mini-batch 训练是不同分布；本地脚本虽 `data.shuffle=False`，仍每批处理多个 prompt。第二篇附录 B.2 已比较两题的顺序与混合调度，不能由单题结果推出逐题边际收益规律。 |
| OPD／OPSD 都是逐 token 反向 KL | 过于宽泛 | 第一篇讨论 sampled-token、full-vocabulary、top-k 变体；其他 OPSD 工作也使用前向 KL 或广义 JSD。on-policy 主要描述轨迹来源，不能唯一决定损失。 |
| 输出分布上算损失，只训练输出层 | 错误 | 损失通过输出层和隐藏状态反传到所有允许训练的学生参数；全参数、LoRA、只训练头部，差别在可训练参数集合。教师通常冻结。 |
| 自蒸馏没有新知识，所以只能记忆 | 证据不足 | 多次采样、筛选和上下文条件化能改变已有行为的概率、组合及搜索策略；是否产生可迁移改进要靠评估，不能仅凭有没有外部知识判断。 |
| U-OPSD 只是多数投票加伪标签 | 不完整 | 官方摘要明确包含一致性阈值、最短伪解答作为教师上下文、最长不一致轨迹作为训练对象等设计。伪标签不是可靠真值，组件常见也不等于方法没有贡献。[原论文](https://arxiv.org/abs/2608.06296) |
| TTPO 惩罚其他答案 | 不完整 | 它对与投票答案一致的轨迹使用 OPSD，对不一致的轨迹使用 grouped RL，并分别设计 token 加权和掩码。不同于统一地把所有轨迹拉向伪答案。[§3](https://arxiv.org/html/2608.27448v1#S3) |
| 特权信息会损害 thinking 模型 | 有具体实验支持，不能普遍化 | 相关论文报告特定 thinking 模型和长推理设置的退化，联系到高熵分叉位置和纠错信号；不代表所有特权信息都无效。[原论文](https://arxiv.org/abs/2607.05184) |
| OPRD、vMF、Spherical Steering 可直接组合 | 暂不支持 | 你没有提供可唯一确定的 OPRD 论文及相关几何方法原文。目前不能确认其损失、层对应、旋转或门控设计，也不能据此声称新颖。 |

两篇主要依据分别是 [第一篇](https://arxiv.org/abs/2604.13016) 和 [第二篇](https://arxiv.org/abs/2609.04172)。涉及主论文的细节以上述本地 PDF 对应章节为准，而非二手解读。

**2．机制上最需要分清的三件事**

设状态为学生生成的前缀 s=(x,y< t)，学生和教师的下一 token 分布为 p、q。三件事分别是：学生是否访问到有用的状态；教师在这些状态上的分布是否有用；学生参数更新是否能吸收这种指导。覆盖率、教师基准分和优化速度分别只触及其中一部分。

第一篇的 overlap ratio 是 `|TopK(p)∩TopK(q)|/K`；共享概率质量则分别是学生、教师在交集上的概率和。共享集合一样大，并不代表承载相同质量。熵差 `|H(p)-H(q)|` 也不能单独证明分布接近：两个分布可以熵一样，却把质量放在不同 token 上。论文的 overlap-token advantage 还涉及交集内归一化，不能直接等同于任意采样 token 的 log-prob gap。

第二篇的距离是学生实际采样 token 上 `|log q(y|s)-log p(y|s)|` 的均值，吸收率为 `(d_t-d_(t+1))/d_t`。这是操作化指标，不是完整 KL，也不是知识量。训练中的状态分布在变，观察到距离降低可能同时包含参数变化和访问状态变化。因此后续研究应增加一个固定、独立的前缀探针集，专门测同一批状态更新前后的变化。

输出损失的反传可写成 `L(pθ) → logits=W hφ → hφ → 各 Transformer 层`，因此 `∂L/∂φ=(∂hφ/∂φ)^T W^T ∂L/∂logits`。只要内部参数可训练、计算图未断开，它们就会更新。只训练 W 可以改变对已有特征的读取；全参数训练还能改变特征与内部计算，但两者是否带来推理泛化都要实测。输出蒸馏允许师生层数、隐藏维度不同，逐 token 的词表分布比较却要求正确的 tokenizer／token ID 对应。

表征坐标不天然可比。例如 h'=Rh、W'=WR⁻¹，在可逆变换下可保持同样输出；这说明单看向量距离可能把功能相同的表示判为不同，不代表任意 Transformer 整体都能自由旋转。维度降低的投影也不能保证保留全部教师信息。因此目前不把“直接逐层 L2 对齐”作为首选。

**3．候选方向一：什么样的状态覆盖，才能预测可迁移的蒸馏收益？——首选**

研究问题：在同样的聚类覆盖率下，两组训练轨迹是否仍有不同的收益？差异来自簇访问频率、师生剩余分歧，还是教师在学生前缀上的指导可靠性？

依据：第一篇 §5.2 表明 overlap 数量和概率质量会分离；§6.1 表明长前缀上教师续写优势会减弱。第二篇 §7 自述覆盖率等权统计每个簇，忽略访问频率和剩余监督信息。它已在附录 B.1 做聚类数量、参考集和近邻距离控制，因此不能把“补一个聚类稳健性实验”当成新贡献。

可证伪假设：粗粒度覆盖相近时，教师有续写优势、且学生仍未吸收的状态，对任务收益更有预测力。假设若成立，加入这两类信息后，应在未参与指标设计的 query 集和 checkpoint 上，比原始覆盖率更好地预测未来收益；若优势不能跨设置复现，就不能支持这一机制。

最小实验先做诊断，不急着改算法：

- 模型用仓库默认的 R1-Distill-1.5B 学生、JustRL-1.5B 教师。优先复用基线 checkpoint 和已有轨迹，减少额外训练。
- 先构造若干等规模 query 子集，至少包括随机 16 题与语义多样 16 题；再用独立 pilot 轨迹挑选覆盖近似相同的集合，匹配采样 token 数、长度和难度分布。**随机组与多样组本身不是充分新实验，关键是匹配覆盖后的残余差异。**
- 复现第二篇的教师隐状态聚类，随后用独立轨迹评价覆盖、簇频率、共享概率质量、固定探针集的分布差异。
- 在少量分层抽样的前缀上，让教师和学生分别多次续写，估计最终答案正确率差。最终答案标签仅用于离线诊断，不能不经说明地进入所谓无监督选择器。
- 比较普通覆盖率与新增指标对下一段训练收益的预测，在独立 query 集检验；以按题 bootstrap 和训练 seed 复核，不把每个 token 当独立样本。

可以探索权重 `w_c=f_c×g_c×r_c`：f 是固定参考分布下的簇重要性，g 是剩余师生分歧，r 是教师续写可靠性。但这只是待检验的候选指标；简单乘法本身不是创新，g 大也可能只是失配或噪声。

继续标准：匹配覆盖和计算量后仍存在稳定收益差异，且指标在独立设置预测这种差异，才进一步试状态采样／加权；若差异消失，或指标只拟合当前模型，就停止扩大方法开发。主要额外成本是隐藏状态提取、聚类和分层续写；先用少量 checkpoint 与每条轨迹少数位置，避免保存全层全 token 激活。

新颖性边界：第二篇已明确提出按覆盖选数据、按剩余教师信号加权。潜在贡献应落在“覆盖相同但收益不同”的受控证据及可泛化的解释上；尚未完成全面新颖性检索。

**4．候选方向二：慢吸收来自状态分布变化，还是更新规则本身？**

研究问题：第二篇吸收率下降，在固定状态分布上是否仍成立？多次利用同一轨迹，能否以相同教师成本和总计算获得更多任务收益？

可证伪假设：部分下降来自失配状态或低价值 token 占比上升；另一部分来自固定状态上的优化变慢。固定探针指标可以分离两者。若只是改变访问分布让训练 gap 变小，而固定探针和任务表现不提升，就不能算提高吸收效率。

最小实验分两步：先对现有 checkpoint 用固定前缀集重新计算 d，并记录更新前后变化；确认现象后，再比较每次新 rollout 一次更新、相同 rollout 多次更新、相同总 actor 更新数的持续新 rollout 对照。另设成本相近的学习率对照，避免把更大的有效步长误称为新机制。

记录 avg@16、固定探针差异、PPO clipping 比例、策略漂移、梯度范数，以及 rollout token、教师评分 token、actor 更新 token 和 GPU 时间。分别给出“相同教师预算”和“相同总计算”两条比较曲线；一个比较不能同时控制所有预算。

关键实施细节：本地代码将旧师生差异作为 detached advantage，经 PPO 式更新。多轮更新时学生概率改变，缓存的教师 log-prob 可复用，但若研究的是当前分歧，学生 log-prob／权重需要相应更新；不能只把 `ppo_epochs` 调大就声称实现了新的吸收算法。

继续标准：在同教师预算下收益更高，且把额外反传成本计入后仍有优势、固定探针及任务指标一致改善，才投入。若只在按训练 step 作图时占优，实际 GPU 时间更长，则不支持效率结论。

成本主要是更多 actor 反传和 checkpoint 评估，可先做短训练。这个方向实现较直接，但第二篇 §7 已直接提出 trust region 下的多轮复用，TTPO 也包含未收敛位置加权，所以单纯“复用＋加权”新颖性偏弱。

**5．候选方向三：让特权教师保留学生的错误恢复能力**

研究问题：特权答案是否在部分分叉状态上过早压低了有用的纠错分支？能否只在这类位置改用不含答案的教师监督，保持学习收益与错误恢复？

可证伪假设：若退化与特权信号在分叉位置的作用有关，在匹配总损失尺度和位置数量后，针对这些位置的干预应比随机位置干预更有效。若随机掩码、全局减小蒸馏强度也同样恢复性能，则“纠错分叉机制”证据不足。

最小比较：普通 OPD、特权 OPD、特权 OPD＋分叉位置切换普通教师、特权 OPD＋等数量随机位置切换。先用一个已报告该失效的 thinking 模型验证现象，不假定仓库默认 R1 模型一定复现相同问题；non-thinking 模式作为后续对照。切换规则先只用熵等可操作信号，纠错词表用于辅助分析，不把 wait/maybe 的频次当成机制定义。

指标必须包括同温度／长度下的 avg@16、pass@k、长度归一化行为统计，以及对已核验错误前缀的恢复正确率。错误前缀需对各模型相同，包含不同错误类型，并另测正确前缀延续，防止方法学会一律推翻上文。

继续标准：在预算、长度和损失尺度匹配后，定向干预持续优于随机干预，且错误恢复与新题收益同时改善；否则缩小或放弃机制叙事。主要成本是第二条教师评分路径及长序列评估。与 [Rethinking OPSD for Thinking Models](https://arxiv.org/abs/2607.05184) 高度接近，需在开发前细查其全部干预实验。优先级排第三。

**6．一套精简、可解释的评估**

| 指标／实验 | 计算或设置 | 能支持什么；不能单独说明什么 |
| --- | --- | --- |
| avg@16 | 每题 16 次独立采样的正确比例，再按题平均；各基准再宏平均 | 估计给定采样策略下单次成功概率，不是 pass@16，也不是多数投票准确率。 |
| pass@k | 每题 n 次采样中 c 次正确时，用 `1-C(n-c,k)/C(n,k)`，n≥k | 评价采样解题覆盖；固定样本数之外，还要比较总生成 token 成本。 |
| 错误前缀恢复 | 同一组核验错误前缀，比较最终正确率及正确前缀延续 | 更直接测恢复能力；受前缀构造影响，不能代表全部推理能力。 |
| 改写／条件反转／组合新题 | 成对构造、核验答案、训练评估分离 | 帮助排除模板和答案记忆；题目本身的难度变化仍需控制。 |
| overlap、共享质量、熵、固定探针 gap | 同一 tokenization、同一状态及温度下计算 | 描述分布与训练动态；不直接证明因果或真正推理。 |
| 成本曲线 | 同时记录教师评分 token、学生生成 token、反传 token、GPU 时间 | 判断实际收益成本；仅训练 step 更少不足以证明便宜。 |

pass@1 提高而 pass@k 不提高，可能是既有成功轨迹概率提高，也可能是多样性下降；有限 k 的结果不能证明模型没有新能力。关键结果至少做多个训练 seed，并按题而非按 rollout/token 估计不确定性。开发集用于阈值和选 checkpoint，测试集用于最终评价，避免反复在 AIME 小样本上调参。

**7．第一篇复现范围：先基线，再机制，最后 recipe**

论文并不是一次 `bash` 就能复现全部图。建议按下面顺序：

| 阶段 | 模型／改动 | 对应目标 |
| --- | --- | --- |
| 起点 | R1-Distill-Qwen-1.5B → JustRL-DeepSeek-1.5B，箭头左为学生、右为教师 | 默认成功组，验证训练、checkpoint 与独立评估流程。 |
| 教师对照 | 同一学生，教师改成 R1-Distill-Qwen-7B | 第一篇 §3.2、§4.1 的弱收益对照；控制训练预算和评估。 |
| token 机制 | 成功组分别设 `only_stu`、`intersection`、`union-intersection`，另做 sampled-token | §4.2、§6.3；它们不是同一语义的 top-k 参数替换。 |
| 思维模式 | Qwen3-1.7B-Base 学生，Qwen3-4B non-thinking 与 Qwen3-4B-Base-GRPO 教师 | §3.1；注意模板及 non-thinking 配置。 |
| 冷启动 | Qwen3-1.7B-Base 与作者 SFT checkpoint 分别接受同一教师 OPD | §5.1；先复现使用已发布中间 checkpoint 的结果，再考虑从零生成 SFT 数据。 |
| prompt／长度 | matched prompt 模板与内容对照，response cap 扫描、教师前缀续写 | §5.2、§6.1；需要额外实验和分析脚本。 |

完整反向蒸馏还应以 JustRL 为学生、R1-1.5B 和 R1-7B 为教师复现 §3.3。单题覆盖率分析来自第二篇，不属于第一篇全部复现的必需项。

**8．服务器环境、模型和数据准备**

作者 README 报告全部实验使用 **8×A800 80GB**。这是原设置，不是最低要求。较小硬件先做 1.5B 配对与短流程验证；缩短生成长度、改精度、改全局 batch 都可能改变结果，应标为缩减实验。当前不知道你的服务器配置，不给无依据的最低显存或耗时。

把整个 `Rethinking-OPD-main` 传到服务器，保留其中定制的 `verl`。以下以服务器根目录 `/workspace/Rethinking-OPD-main` 为示例，实际路径需替换。先在独占分配的 GPU 作业里执行环境检查：

```bash
cd /workspace/Rethinking-OPD-main
nvidia-smi
conda create -n opd-repro python=3.12 -y
conda activate opd-repro
cd verl
USE_MEGATRON=0 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh
python -m pip install -e .
python -m pip install math-verify swanlab matplotlib
python -m pip check
cd ..
python -c 'import torch, vllm, verl; print(torch.__version__, torch.version.cuda, vllm.__version__, verl.__file__); print(torch.cuda.device_count())'
python -m pip freeze > /workspace/opd-repro-environment.txt
```

原 README 只设 `USE_MEGATRON=0`；这里额外禁用未使用的 SGLang，是精简安装选择。安装脚本固定 vLLM 0.11.0，并下载面向 Python 3.12、torch 2.8、CUDA 12、指定 C++ ABI 的 FlashAttention wheel。必须确认实际安装版本与 wheel 匹配，不要遇到问题就混装最新 torch/vLLM。依赖安装没有完全锁版本，首次跑通后保留 freeze、驱动信息、源代码快照及模型 revision。

`pip install -e .` 很关键：原安装脚本没有安装本地 verl 包。检查 `verl.__file__` 应指向服务器上的这个定制版本。若已有其他 verl 环境，应单独建环境。

下载默认配对，下面会在服务器消耗网络与磁盘；当前本地未执行：

```bash
cd /workspace/Rethinking-OPD-main
python - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
models = {
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B': 'DeepSeek-R1-Distill-Qwen-1.5B',
    'hbx/JustRL-DeepSeek-1.5B': 'JustRL-DeepSeek-1.5B',
}
api = HfApi()
revisions = []
for repo, dirname in models.items():
    revision = api.model_info(repo).sha
    snapshot_download(repo_id=repo, revision=revision, local_dir=f'model/{dirname}')
    revisions.append(f'{repo}\t{revision}')
Path('model/revisions.tsv').write_text('\n'.join(revisions) + '\n')
PY
```

JustRL 地址取自 [作者仓库](https://github.com/thunlp/JustRL) 的模型链接，实际为 [hbx/JustRL-DeepSeek-1.5B](https://huggingface.co/hbx/JustRL-DeepSeek-1.5B)。不要凭组织名猜下载地址。论文模型集合当前链接到 [Thinking-Space](https://huggingface.co/collections/lllyx/rethinking-opd)，例如 `Thinking-Space/Qwen3-1.7B-SFT`、`Thinking-Space/Qwen3-4B-Base-GRPO`。

数据本地已有，无需先重新下载。已用 Parquet 元信息只读核对：

| 文件 | 行数 |
| --- | ---: |
| `datasets/dapo-math-17k.parquet` | 17,917 |
| `datasets/dapo-math-17k-processed.parquet` | 17,917 |
| `datasets/OpenThoughts3_opd.parquet` | 30,000 |
| `datasets/DeepMath_deduped.parquet` | 14,116 |
| `datasets/test_data/AIME24/test.parquet` | 30 |
| `datasets/test_data/AIME25/test.parquet` | 30 |
| `datasets/test_data/AMC23/test.parquet` | 83 |

字段含 `prompt`、`reward_model.ground_truth`、`data_source` 等。processed 数据主要涉及模板处理，不能只看行数相同就当内容完全相同；论文不同实验使用的模板要逐组对齐。

**9．训练：原始定制路径与新版示例要分开**

优先使用根目录 `on_policy_distillation.sh` 加本地定制 verl，保持论文实现。`verl_example/opd.sh` 的说明要求另一套 verl≥0.8.0，使用新的 `distillation.*` 接口，其中 `forward_kl_topk` 不能视为原始 reverse-KL 风格 top-k 训练的同义项；也应核实新版 teacher 独立资源池的 GPU 分配。

默认基线的重要配置为：

| 项目 | 原始脚本／论文默认 |
| --- | --- |
| 学生、教师 | `model/DeepSeek-R1-Distill-Qwen-1.5B`、`model/JustRL-DeepSeek-1.5B` |
| advantage | `token_reward_direct`，不是 GRPO |
| top-k | 16，`only_stu`，`student_p` |
| batch | 每步 64 个 prompt，4 个 rollout／prompt；PPO mini-batch 64 |
| 长度 | prompt 1024，训练 response 7168 |
| 优化 | lr=1e-6，1 epoch，token-mean，额外 reference KL 关闭 |
| 采样 | 训练 temperature=1.0、top-p=1.0，teacher temperature=1.0 |
| 硬件／精度 | 脚本 8 GPU、1 节点，`MODEL_DTYPE=fp32` |
| 保存／验证 | 每 20 step 保存，内置验证 `test_freq=-1` |

“64 个 prompt ×4 rollout”和第二篇的“每次更新 64 rollout”不是同一个 batch 定义。不要只抄论文表格数字而忽略代码的展开与 mini-batch 更新次数。也不要把参数 `USE_KL=False` 误读为没有蒸馏 KL：它关闭的是额外 reference-policy KL 正则。

首次正式训练前，**在服务器新建脚本副本**做短流程验证：令 `trainer.total_training_steps=2`、`trainer.save_freq=1`，保留其余核心设置。这里是未来实施建议，本次没有创建或修改任何训练脚本。确认：teacher `rm_scores` 用于优化；log-prob 有限；actor loss、梯度正常；checkpoint 能合并并完成独立评估。

短验证通过后，在独占 GPU 作业中启动默认完整基线：

```bash
cd /workspace/Rethinking-OPD-main
conda activate opd-repro
export SWANLAB_MODE=offline
IS_PLOT=False bash on_policy_distillation.sh
```

`IS_PLOT=False` 关闭额外可视化，核心 top-k 统计仍有独立代码路径；如需复现逐位置图，再打开绘图并保留所需数据。原脚本含 `ray stop --force`，应在自己的作业／容器中运行，避免影响同节点自己的其他 Ray 任务。`#SBATCH` 中的 account、partition、路径是作者集群配置；若用 Slurm，先按你集群新建提交文件并保证日志目录存在。

**容易误操作的地方：**原脚本的学生路径、教师路径、数据路径、GPU 数、长度等很多值被直接赋值；在命令前设置同名环境变量会被覆盖，而且脚本没有把 `"$@"` 传给 Python。做不同实验时必须改服务器上的副本或写新的 launcher；不能假定 `ACTOR_MODEL_PATH=... bash 原脚本`、`bash 原脚本 trainer.total_training_steps=2` 会生效。`MINI_BATCH_SIZE`、`LOG_PROB_TOP_K` 等使用默认值展开的变量才支持这种环境覆盖。

代码上的损失也值得单独记录：`workers/actor/dp_actor.py` 在 `only_stu` 分支计算归一化学生权重乘以**原词表** `log q-log p`，`core_algos.py` 再经 detached advantage 和 PPO ratio 更新。它与论文 §2 写出的“子集内双方重新归一化后直接求 KL”并非逐项完全相同的计算式；复现应保留实际代码路径，若要研究二者差异，另设明确对照。

**10．checkpoint 合并与独立评估**

先合并 FSDP actor checkpoint。下面 `RUN_DIR` 必须换成实际训练目录，STEP 必须是已经保存的步数：

```bash
cd /workspace/Rethinking-OPD-main
RUN_DIR=/workspace/Rethinking-OPD-main/checkpoint/替换为实际实验目录
STEP=20
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$RUN_DIR/global_step_${STEP}/actor" \
  --target_dir "$RUN_DIR/hf_step_${STEP}"
```

独立评估应对初始学生、教师、选定训练步 checkpoint 使用同一套设置。第一篇 §3.1 的口径为 AIME24、AIME25、AMC23，各题 16 次采样，temperature=0.7、top-p=0.95、最大 response=31,744，报告 avg@16，然后对三个基准宏平均。

本地 `scripts/val/eval/gen_vllm.py`、`grade.py` 不是即开即用，服务器上应复制后配置：

| 文件／问题 | 服务器副本应处理 |
| --- | --- |
| `gen_vllm.py`：`MODEL_FOLDER` 定义被注释，但后面调用 `os.listdir(MODEL_FOLDER)` | 删除不用的自动扫描块，或给出有效目录。当前 except 只捕获 FileNotFoundError，捕获不了 NameError。 |
| `MODEL_NAMES` 的默认相对路径 | 换成合并 checkpoint 的绝对路径列表。每个输出目录名须唯一，避免不同实验同名 checkpoint 互相跳过。 |
| `DATA_DIR='../data'` | 从 `scripts/val/eval` 启动时，该路径能指向仓库已有的 `scripts/val/data`。 |
| GPU 列表硬编码 0…7、每 worker TP=1 | 对照作业可见 GPU 数配置；每个模型须能装进单卡，不能把 worker 数当 TP。 |
| `grade.py`：`NAME` 默认与生成模型不一致 | 指向生成输出目录；否则可能没有待评分文件。 |
| `grade.py` 导入时加载 `../../model/Qwen3-1.7B` tokenizer | 改成已有模型的有效绝对路径；即使不开模型 verifier 也会触发此加载。长度比较最好统一 tokenizer 并记录。 |
| 生成 worker 捕获异常并继续；评分假设每题采样数一致 | 评分前核实题号、每题样本数、重复／缺失，不能以进程退出码代替完整性检查。 |

配置好的副本与原始 `utils.py` 放在同一评估目录，例如 `gen_vllm_repro.py`、`grade_repro.py`，然后运行：

```bash
cd /workspace/Rethinking-OPD-main/scripts/val/eval
python gen_vllm_repro.py --disable-thinking
python grade_repro.py
```

该 thinking 参数控制聊天模板开关，不能用它直接判断模型本身是否是 thinking 模型；对 DeepSeek 配对先保持原评估默认，对 Qwen non-thinking 组要显式保持关闭并核对格式。训练和评估 prompt 后缀也要一致核验。

完整的 16 次输出应为 AIME24 480 条、AIME25 480 条、AMC23 1328 条，而且每题必须正好 16 条。评分 `mean_score` 对应 avg@16；`best_score` 是每题至少一次正确的比例，不是 avg@16。默认先统一规则评分，若使用 CompassVerifier，需额外下载并对所有组保持相同；不要拿两种评分口径直接比较。

README 提示 v0.7.0 内置验证可能低估结果，建议关闭后独立评估；这是作者报告，不是本次重新实测的结论。当前生成脚本中的 `seed` 字段主要是 rollout 编号，代码注释也说明它不控制 vLLM RNG，不能把该字段当作完整的随机复现设置。

**11．如何复现 cold start，以及目前仓库的边界**

先下载作者发布的 `Thinking-Space/Qwen3-1.7B-SFT`，与 Qwen3-1.7B-Base 分别接同一 Qwen3-4B non-thinking 教师做 OPD，可复现“使用该中间 checkpoint 后，OPD 是否恢复”的对照。当前集合地址见上文。这一步不等于从零复现 SFT 数据生成。

从零重做时，第一篇附录 C.1 指定从 OpenThoughts3 数学子集中抽取 200K prompts，Qwen3-4B non-thinking 每题生成一条，temperature=0.7、top-p=0.95、最大生成长度 12,288，过滤截断和重复，再 full SFT。SFT 配置为 qwen3 模板、序列长度 14,336、每卡 batch 8、梯度累积 1、lr=1e-5、cosine、warmup 0.05、1 epoch、BF16。应记录实际过滤后样本数，而不是把输入的 200K 当作输出必然行数。

SFT 应单独使用 Python 3.11 环境：

```bash
conda create -n opd-sft python=3.11 -y
conda activate opd-sft
cd /workspace/Rethinking-OPD-main/LlamaFactory
python -m pip install -e .
python -m pip install -r requirements/metrics.txt
```

随后在新的 YAML 副本中配置模型、数据注册、输出目录及 deepspeed 依赖，核对 `data/dataset_info.json` 的数据映射，再用 `llamafactory-cli train 新配置.yaml`。原 YAML 中有相对路径和 `enable_thinking: True`，需要结合实际序列格式检查，不能因教师是 non-thinking 就未经核验地乱改模板。教师 rollout 脚本的 GPU 列表、输入、拒绝采样和长度也应与附录一致。

根目录 `grpo.sh` 中默认训练／验证长度及 dataset 路径与 OPD 组不同，部分路径在当前目录不存在；直接运行不能自动得到论文的 GRPO 教师。初次复现优先下载作者 GRPO checkpoint；若从零训教师，再依据附录 A.1 配置 processed DAPO、64 prompt、8 rollout、lr=1e-6、1 epoch、无额外 KL，并独立评估。

当前代码足以作为训练与评估基础，但不能据此认定每张机制图都已有可直接运行的独立入口。诸如固定状态探针、教师前缀续写和第二篇 PCA 聚类分析，需要额外配置／实现。本次没有改动这些代码，也没有承诺数值完全复现。

**12．建议执行顺序与下一次需决策的事项**

先跑通默认配对的两步训练、checkpoint 合并和独立评估；然后完成一条全量基线并留存中间轨迹；再用这些轨迹开展方向一的诊断。只有诊断发现原覆盖率解释不了、且在独立数据上稳定存在的差异，才决定开发选择／加权算法。方向二可复用 checkpoint 做固定探针分析；方向三暂缓。

今后需要你决定的是：可用服务器 GPU 型号、数量、单卡显存、允许的 GPU 时间和磁盘预算；选择严格复现原设置还是先做缩减验证；诊断结果是否足以支持修改损失或采样代码。这些会实际改变资源消耗或研究路线，应在执行前确认。OPRD 和几何方向若继续推进，需要先补充可唯一确定的论文题目或链接。
