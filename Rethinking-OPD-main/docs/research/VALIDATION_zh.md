# 交付文件与验证记录

## 2026-10-27：2026-10-28服务器重跑交接

重写根`PILOT_HANDOFF_zh.md`，统一当前50步/LoRA/8×8/2048与16384长度/答案协议v2/全回答分布/两组表征比较与离线HTML；旧交接移至带历史标记的archive。补充源码同步校验、真实环境与CUDA/HTML后端检查、独立2步验收、正式50步命令、后台状态/日志、latest保留限制、质量门槛、故障恢复与完成标准。`SERVER_HANDOFF_MANIFEST.json`记录交接文件SHA256；生成时工作树尚未提交/推送，不将Git HEAD冒充本轮全部源码身份。

检查主文档8个Bash代码块语法、4个内嵌Python代码块编译及本地链接；实际执行文档中的小样本配置生成逻辑和真实shell入口的两次无GPU预览。验收计划为2步、监测0/1/2、划分16/8/8/4；正式计划为50步、监测0/1/5/10/20/50、划分10000/512/512/64。两者均保留8×8、lr1e-6、2048/16384长度与质量/新指标开关，并确认没有启动训练或创建train目录。未在本地执行GPU kernel检查、真实模型prepare或服务器训练。

## 2026-10-27：指标分组清单与单文件HTML

新增 `analysis/html_report.py` 与 `analysis/report_template.html`，接入现有报告流程，自动生成 `report/OPSD_report.html`。按“当前学生与初始学生 / 当前学生与当前特权教师”列出13类指标及字段、代码支持情况和当前结果可用性。全回答token均值、历史选点及逐题摘要保持独立；曲率明确区分已有学生/初始配对差与尚无独立字段的学生/教师配对差。

HTML内嵌Plotly.js、数据和全部引用图片，支持指标检索、上下文/层选择、逐层曲线、层×step热图、逐题直方图、CSV/PNG/SVG和来源JSON导出。仅重绘无需GPU或模型。已从真实step20历史导出生成 `research_runs/step20_local_analysis/OPSD_metrics_complete.html`，保留173个指标入口及120张RSM图；未采集的新指标不伪造。旧结果与原HTML保留。

新增7项测试覆盖token加权主值与题目等权摘要分离、初始/当前教师角色与字段映射、旧选点不能冒充新测量、源hash校验、单文件资源闭合、JSON脚本转义、无服务器原始文件时读取归档，以及Node中的DOM/Plotly替身控件测试。研究全套 **142项通过**，原OPD指标 **11项通过**，共153项。Node控件测试不等于真实浏览器渲染验收；没有执行真实1.7B新实验。

## 2026-10-27：同一步学生—特权教师直接表征比较

新增 `analysis/metrics/paired.py`，按固定回答、相同层/anchor及problem_id配对当前学生与当前特权自教师。默认 `compare_student_teacher=true`，在0/1/5/10/20/50步采集，独立role为 `student_vs_privileged_teacher`。指标含RSA rho/RSM RMSE、线性CKA及Euclidean kNN overlap(k=8/16/32)，保留原学生/教师分别对step0的比较。联合过滤无效向量；常量RSM、空anchor及N≤k不补零；保存共同有效题ID、邻居索引/距离与文件hash。报告自动导出CSV、逐层曲线及层×step热图。

新增8项测试覆盖独立RSA/CKA/近邻公式、相同/旋转表征、ID重排及错配拒绝、共同无效样本、常量/空anchor、固定seed采样、真实tiny checkpoint提取和缓存、实际报告PNG/CSV。全套研究测试 **135项通过**，原在线OPD指标 **11项通过**，共146项；`git diff --check`通过。真实提取验证step0的同权重学生/教师可因输入不同产生不同RSA；使用CPU tiny GPT-2及既有tiny Qwen3测试环境，不代表真实1.7B GPU训练已完成。

默认新目录为 `research_runs/base_geometry_50_answer_v2_full_tokens_paired`。实现及科学解释见[学生—教师几何指标](ST_GEOMETRY_zh.md)。

## 2026-10-27：全有效回答token的分布监测

离线fixed/live由1/32/128选点改为全回答有效预测位置，全词表T=1熵和双向KL按response mask做token平均；默认top-16 overlap/OTA与在线函数一致。空交集只从OTA分母排除。跨题保留分子分母，按有效token加权，不混合fixed/live或train/heldout。初始权重重新评分同一条当前轨迹，消除live文本不相同导致的错误比较；不再保存整条全词表分布。保留几何anchor及256窗口，live分布统计不受该窗口裁剪。

新增6项数值测试使用真实tiny GPT-2、tiny Qwen3及完整causal forward作为对照，覆盖KV分块/错一位、head块上限、几何窗口不裁剪分布、当前/初始模型同前缀、在线函数一致性、EOS/padding、空交集、空mask及跨题token加权。配置测试额外拒绝旧选点、错误scope及非法块大小。研究全套 **127项通过**，原在线OPD指标 **11项通过**，共138项。执行环境仍为前述CPU临时依赖，未运行真实1.7B GPU训练或测量显存/耗时。

默认运行目录更新为 `research_runs/base_geometry_50_answer_v2_full_tokens`；新源码/统计身份拒绝旧probe/cache，旧三点指标不能冒充全回答指标。运行及内存说明见[全回答指标](ALL_TOKEN_METRICS_zh.md)。

## 2026-10-27：答案协议v2对齐

新增共享`answer_protocol.py`，将Base模板、参考标签核验、研究reward和独立评测对齐。解析显式Final answer区域与兼容boxed，不从最后数字或ground truth猜答案；最终候选直接进入数学等价比较。真假题缺少A/B语义映射等样本先隔离。行为评测16384tokens、几何前缀256tokens；基座及监测checkpoint的heldout抽取失败率超过1%时持久化诊断并中止。原始OPD/GRPO脚本、旧step20结果与既有科学目标保留；用户设置的训练长度2048/16384未被覆盖。

新增14项回归覆盖格式/嵌套/空白/未闭合/冲突/占位符、抽取成功但数学错误、候选直接判分、True/A映射、真实prepare隔离及基座门槛、serial生成缓存与独立长度。研究全套121项、原OPD指标11项通过；CPU临时依赖目录`/tmp/opd-answer-v2-deps`使用Transformers4.56.1、PEFT0.15.2、Hydra1.3.2/OmegaConf2.3.0与pylatexenc2.10，包括仓库原有tiny Qwen3/PEFT导出测试。没有更改原Python环境，没有下载Qwen权重或启动真实训练。真实Qwen自由生成能否达到1%门槛仍待服务器校准，不能由这些CPU测试宣称已降低实际失败率。

## 2026-10-27：服务器环境交接与发布前检查

`PILOT_HANDOFF_zh.md` 补充Linux/GPU/CPU/内存/磁盘的环境参考、Torch2.8.0与vLLM0.11.0的版本对应、Transformers/PEFT/NumPy/PyArrow限制、本地verl安装、FlashAttention ABI核对、缓存目录和Ray/容器注意事项。安装命令是服务器配置参考，不是本机已通过GPU训练的锁定环境；仍须完成独立新目录2-step验收。

新增可复制的BF16 FlashAttention前向/反向测试、Qwen3/vLLM LoRA导入及本地verl路径核验，并保存pip freeze、包列表、驱动、系统/CUDA/ABI、Git提交与工作树状态。原始 `grpo.sh`、`on_policy_distillation.sh` 保持与发布前HEAD逐字一致；当前Base实验仍为200步、默认仅OPSD。

发布前重新运行：69项研究测试和11项OPD指标测试通过；13个交接文档Bash代码块、5个shell脚本语法通过，4个内嵌Python代码块可编译，本地文档链接存在；`git diff --check`通过。待发布文件没有大于10MiB的新增/修改文件，也没有发现所检查的常见凭据格式；运行产物与数据缓存按gitignore排除。真实服务器依赖安装、GPU kernel检查、Qwen训练、显存和耗时仍未在本机执行。

## 2026-10-26：当前 Base 实验改为200 step

`configs/analysis_deepmath.json` 的训练上限与 `configs/research/base_geometry.json` 的显式预算均改为200；监测步数为 `[0,1,5,10,20,50,100,200]`，教师几何和完整RSM绘图步数为 `[0,20,100,200]`。默认仍仅执行OPSD，GRPO设计保留。默认工作目录改为 `research_runs/base_geometry_200`，旧目录不自动改写。

batch8/n8/lr1e-6/LoRA r16-alpha32和原probe数量保持原设置。200步对应1600次问题呈现、12800条训练回答；8个分析checkpoint，常规每50步保存的step150也保留，因此正常共有9个训练checkpoint（含step0）。旧版non-thinking profile及原始OPD/GRPO脚本仍保留历史预算。

实际验证：69项研究测试通过；新增检查默认配置、实际Hydra训练预算、最终step200监测/绘图、OPSD-only预览和新目录。完成训练的保护测试同步核验200步后不进入第201步。真实一键脚本预览解析为200步、batch8/n8、8个监测点，没有加载模型或启动训练。Python编译、shell与文档Bash代码块语法、`git diff --check`通过。未运行真实GPU训练。

## 2026-10-26：默认仅执行 OPSD

按本轮要求保留 GRPO 设计及配置生成，将 `run_base_comparison.sh` 和 `compare run/train/analyze` 的默认方法改为 `opsd`。预览计划按实际选择的方法生成；只有显式 `--method grpo` / `--method both` 才执行 GRPO。OPSD 的在线指标、checkpoint 表征监测和报告流程保持开启。

`python -m unittest discover -s tests/research -q`：68 项通过，新增 4 项选择与调度测试，覆盖默认入口、预览计划、实际执行分发和显式 GRPO/both；GPU 训练均被 mock，没有启动真实训练。shell 语法、Python 编译、`git diff --check` 通过。下面保留上一轮完整几何监测的验证记录。

## 2026-10-26：Base OPSD / GRPO 几何监测

新增 `run_base_comparison.sh`、`scripts/research/compare.py`、`configs/analysis_deepmath.json`、Base profile / plain Jinja、独立 `analysis/` 和 checkpoint job 接口。完整设计与服务器命令见 [BASE_GEOMETRY_zh.md](BASE_GEOMETRY_zh.md)。旧入口与既有改动均保留；以下旧记录不代表已运行本轮 Base GPU 训练。

本轮真实执行：

- `python -m unittest discover -s tests/research -q`：**64 项通过**，其中 18 项是本轮几何/队列/共享数据/导出隔离测试。
- `python verl/tests/utils/test_opd_dynamic_metrics_on_cpu.py -q`：**11 项通过**。当前环境未安装 pytest，使用文件自带的 unittest 入口；没有把失败的 pytest 调用算作通过。
- Python compileall、五个保留/新增 shell 的 `bash -n`、`git diff --check` 通过；原始 `grpo.sh` / `on_policy_distillation.sh` 与 Git HEAD 逐字一致。
- `python -m analysis.smoke --out research_runs/base_geometry_release_cpu --report`：12 个合成问题、2 个 tiny GPT-2 权重状态；每个状态 242 行聚合指标，生成 256 张可从缓存重画的 PNG。step1 是确定性权重扰动，不是训练。首次输出、重复缓存验证、真实 block 前向及 NLL 索引均已执行。
- 示例记录：`research_runs/base_geometry_release_cpu/smoke_result.json`；汇总：`analysis/report/metrics.csv`；图：`analysis/report/behavior_and_fixed_curvature__heldout_probe.png`（以上后两项相对该 smoke 目录）。这些运行产物由 `.gitignore` 排除，但在本地工作区可直接查看。

本地依赖版本：Torch 2.3.1、Transformers 4.44.2、NumPy 2.1.1、SciPy 1.14.1、Matplotlib 3.10.3。没有可用 CUDA，缺少 Ray / vLLM / PEFT / PyArrow / pylatexenc，因此未执行真实 Qwen、FSDP 多 rank 保存恢复、实际 PEFT 合并或 Parquet 写入。CPU smoke 显式使用 JSONL；服务器必须按新文档准备训练兼容环境，尤其 Qwen3 的 Transformers>=4.51 和 vendored verl 的 NumPy<2。每个分析 checkpoint 另存实际版本、配置、checkpoint 内容与 probe / 源码 hash。

测试覆盖曲率手算与不变性、零位移/特殊 token、严格上三角 RSA、常量/零向量、ID 对齐、稳定 kNN tie-break、PR、左右 padding 和真实模型位置编码、causal target shift、同输入 full-vocab KL / overlap、共享分层划分、Base prompt、不同策略拒绝、初始 LoRA B=0 与指纹、原子 job 的 RNG 隔离、导出失败重试与 checkpoint 不变、已完成训练不再多更新一步。实际 FSDP/PEFT 验收仍需服务器 2-step 命令完成。

以下为之前各阶段的记录。

验证记录日期：2026-10-20 至 2026-10-21，Asia/Shanghai。所有研究输出中立，不声称得到 Qwen 能力提升或论文结论的复现结果。

## 修改和新增文件

| 文件 | 作用 |
|---|---|
| README.md | 新管线入口与文档索引 |
| run_opsd_experiment.sh、opsd_pipeline.sh | 当前 OPSD 一键与分阶段入口，默认只预览，--execute 才执行 |
| on_policy_distillation.sh、grpo.sh | 原仓库训练入口，两者内容均与 HEAD 逐字一致 |
| scripts/__init__.py | 避免被 vendored verl/scripts 同名包遮蔽 |
| .gitignore | 排除 Python cache、本地大数据及运行产物；不影响已跟踪数据 |
| verl/verl/trainer/main_ppo.py | 自定义 chat template 同步到 driver 的 dataset tokenizer，和 actor 保持一致 |
| verl/verl/trainer/ppo/ray_trainer.py | OPSD 教师分支、版本检查、预算与 JSONL、完整 Hydra resolved config；原 OPD 分支保留 |
| verl/verl/workers/fsdp_workers.py | 当前 actor 的 no-grad 特权教师 forward；研究配置下记录跨 rank 峰值显存 |
| verl/verl/workers/actor/dp_actor.py | 默认 only_stu reward 复用纯张量 helper，原数值公式不变 |
| verl/verl/workers/config/rollout.py | RolloutConfig 显式 seed 字段，匹配 rollout engine 与 Hydra 参数 |
| verl/verl/trainer/ppo/core_algos.py | 原 3D PPO terms 抽取为共享 helper，保留 loss/聚合和梯度 |
| verl/verl/utils/opsd.py | prompt/response 对齐、EOS mask、tokenizer 检查、原 reward/surrogate 的纯张量代码 |
| verl/verl/utils/opd_metrics.py | 保留原论文诊断，增加 shared mass、空交集率、signed gap、分别命名的 subset KL/raw log ratio |
| verl/verl/utils/research_budget.py | token/问题访问/时间/更新计数、checkpoint 边界恢复、JSONL |
| scripts/research/run.py | 所有 prepare/partition/reference/train/eval/extract/analyze/plot 等 CLI |
| scripts/research/pipeline.py | run 全流程与独立阶段入口、流程状态、私有注册表、配对数据和完整 checkpoint 检查 |
| scripts/research/lora.py | 修正 FSDP adapter 导出的 alpha，合入 LoRA 后保存可直接评测的完整 HF 权重 |
| scripts/research/preflight.py | 依赖/CUDA/长度/完整 batch 检查；训练前 Hydra 与 rollout schema 检查 |
| scripts/research/common.py | hash、模型/数据/源代码身份、Git/软件 provenance、原子文件输出 |
| scripts/research/data.py | 统一数据、数学数据适配、受控技能、实例分组划分、共同正确交集与覆盖率、verl export |
| scripts/research/models.py | 离线 HF 模型适配、ByteTokenizer/tiny GPT-2、hook 映射；不负责生产优化 |
| scripts/research/distributions.py | 严格 full-vocab/overlap/KL/entropy/mass、avg@16/pass@k、gap recovery |
| scripts/research/representations.py | cosine/RSA/NO/CKA/ARI/subspace/probe/effective-rank/shift 指标 |
| scripts/research/artifacts.py | 对齐、位置采样、分片缓存、离线比较、变化向量、探针 |
| scripts/research/launch.py | 配置继承、角色检查、原 verl/LLaMA-Factory 启动、默认仅计划的实验矩阵 |
| scripts/research/reward.py | verl 与独立评测复用同一 rule verifier |
| scripts/research/sft_train.py | LLaMA-Factory wrapper，真实 label tokens、同步多卡预算停止 |
| scripts/research/plots.py | 标量 CSV、性能/动态/位置曲线、层比较、排序 RSA、ARI、run 级关联图 |
| scripts/research/smoke.py | tiny HF rollout → no-grad teacher → shared loss → backward/checkpoint → eval/activations/probe/plot |
| scripts/research/__init__.py | 模块入口 |
| scripts/research/requirements-analysis.txt | 分析依赖与 Qwen3 版本最低要求 |
| configs/research/models.json | 已核对模型 ID；明确未填的本地路径/revision 与 RL-Math 未确认项 |
| configs/research/smoke.json | 可离线 CPU 的小模型配置 |
| configs/research/pilot_1p7b.json、pilot_4b.json | 单 A800 目标配置，未做显存承诺 |
| configs/research/paper.json | 大配方、多 seed；仍须数据/精确模型来源核实 |
| configs/research/extensions.schema.json | 保留 Gride/干预/LoRA 等扩展记录接口，明确尚未实现 |
| tests/research/test_research.py | 27 项科学性和实验身份测试 |
| tests/research/test_pipeline.py | 启动流程、缓存、seed、配对、预检、checkpoint 及一键调度测试 |
| tests/research/test_forward_kl.py | 全词表 KL 方向/梯度/掩码、生产 actor 隔离 CPU 更新、冻结基座、LoRA 导出与数据量检查 |
| docs/research/AUDIT_zh.md | 版本、原调用链、论文/代码差异、模型与资源核对 |
| docs/research/METRICS_zh.md | 公式、支持集、归一化、平均与 causal shift |
| docs/research/README_zh.md | 分组表与全部实际 CLI 命令、配置、资源和缺失项 |
| docs/research/PIPELINE_zh.md | 实验一依次运行哪些文件、具体命令及输出位置 |
| PILOT_HANDOFF_zh.md | 最新实验的服务器环境/输入准备、一键运行、验收及输出说明 |
| docs/research/SCRIPT_CLEANUP_zh.md、legacy_cleanup_manifest.json | 用户已批准并完成的旧文件清理记录、内容身份和实际删除项 |

## 已执行

```bash
python -m unittest discover -s tests/research -v
python -m unittest discover -s verl/tests/utils -p 'test_opd_dynamic_metrics_on_cpu.py' -v
python -m scripts.research.run smoke --output research_runs/smoke_delivery
bash opsd_pipeline.sh smoke --workdir research_runs/pipeline_validation
bash -n grpo.sh on_policy_distillation.sh run_opsd_experiment.sh opsd_pipeline.sh
```

- 新增 27 项测试通过：合成概率公式、空/单交集、极小概率、逐题/token 平均、avg@16≠pass@16、gap 分母、EOS/padding/截断、词表不兼容、teacher no-grad/版本更新、原 OPD reward/surrogate 值和梯度回归、真实 HF causal shift/左 padding、缓存身份、RSA/NO/CKA/subspace/ARI/probe 隔离、数据 split/交集、矩阵角色、不混合不同 run 绘图、计划文件不阻塞后续执行。
- 11 项原有 OPD 动态指标测试通过。
- 历史记录：清理前 16 项原有 pilot/bootstrap 测试通过。初次用默认 Python 缺 psutil；改用已安装 psutil 的 `/usr/bin/python3` 后通过，没有修改旧测试。2026-10-21 用户批准后这些旧实验专用测试已删除，不再属于当前测试入口。
- `compute_opsd_teacher` 生产方法经 AST 提取，在真实 HF tiny 模型上执行；只替代分布式 transport/manager 外壳，验证其无梯度、相同 response IDs 和参数版本刷新。未以此声称 FSDP 集成已验证。
- CPU smoke 在仓库中真实保存 checkpoint/optimizer、validation 各 step、每题 16 次评测、分布诊断、四组 activations、全部核心表征指标、线性探针和 PNG 图。`smoke_result.json` 是实际结果；随机 tiny 模型全错只表明不适合作技能假设测试。
- OPD/OPSD/SFT/RL 的配置规划命令均实际生成了对应底层命令；当前 pilot 完整矩阵 13 项，paper 三 seed 矩阵 39 项（仅实验一为 3 / 9 项）；没有执行这些大训练。
- 缺模型 `train --execute` 实际在启动 Ray 前触发明确 ValueError。
- 新增/修改 Python 均通过编译，`git diff --check` 通过。

## 未运行和原因

| 项目 | 具体原因 / 后续验证 |
|---|---|
| Ray/vLLM/FSDP 的完整训练集成 | 当前解释器缺 Ray、PyArrow、OmegaConf、TensorDict 等；GPU NVML 被系统禁止访问 |
| Qwen 1.7B / 4B 训练和评测 | 工作区无真实权重，当前 Transformers 4.44.2 不支持 Qwen3；需按项目训练环境安装 |
| SFT/GRPO GPU 训练 | 同上；SFT 实际复用的 LLaMA-Factory 完整运行尚未验证 |
| A800 80GB 显存/吞吐 | 当前无法访问该 GPU；pilot 是保守目标配置，不是已测不 OOM 的保证 |
| 原论文严格复现 | 当前设计是 Qwen3-1.7B 特权自教师一致性实验，§3.2 迁移已取消；不以当前设置声称原实验复现 |
| Gride/ADP/激活干预/LoRA 参数子空间 | 本轮只预留接口；effective rank/K-means 不会被标作它们的替代复现 |

## 仍需填写的资源

1. 注册表中每个实际模型的 path、不可变 revision；缺模板时的核实来源和 chat_template_path。
2. 原始 DeepMath-103K 数据路径与不可变 revision，须保留 difficulty 和选定参考字段；当前去重 parquet 缺少这些字段。当前实验不需要 RL-Math 权重。
3. 选定数学数据的独立 train/validation/test 与去重实例分组；已有 parquet 可直接适配，但语义改写关系需提供实例标签。
4. 每个来源的离线 reference（或已有缓存）、必要时共同正确交集与覆盖率。
5. 可用的训练依赖和 GPU；先单条件 pilot 验证后再扩大，不默认运行全部矩阵。

## 关键限制

- 生产 OPSD 目前仅适配本仓库的 FSDP 文本 actor 和 current-batch stop-gradient 策略；其他后端/策略未声称支持。
- 原 OPD 的 top-k raw-log-ratio reward 与论文双方重归一化 KL 不完全一致，已明确分开日志；未悄悄替换算法。
- teacher context 增加信息和长度，必须比较相同题集并报告长度/质量；有信息教师的分数不能当作无特权模型能力。
- 相同 step 不等于相同计算；预算在 step 边界检查，存在一批的超出幅度。跨 rank 峰值显存的新增采集代码尚待 GPU 实测。
- 固定 oracle response 上的中间状态 probe 是明示的 teacher-forced 诊断；不把这些标签贴到未知 on-policy reasoning 上，也不推断模型因果使用了可读信息。

## 当前设计调整的增量验证

实验一配置、数据适配与矩阵已同步调整，§3.2 不再可选。新增测试覆盖 difficulty 闭区间/小数边界、缺字段拒绝、参考不进入学生 messages、empty/solution 同 ID 同 split、错误模型/数据 revision/难度拒绝、五项训练指标 CSV 与组合曲线保存。当前新增研究测试总计 27 项；历史 smoke 产物保留原 provenance，不改写成新 DeepMath 的实验结果。真实 DeepMath/Qwen 训练仍未运行，需要上列资源。

## 顺序启动流程的增量验证

2026-10-20 再次执行全部测试：研究测试 **36 项**、原有 OPD 动态指标 **11 项**、原有 pilot/bootstrap **16 项**，共 **63 项通过**。新增 9 项验证包括实际 shell 从其他目录启动时的包解析、历史 shell 与 HEAD 原文件逐字一致、配置重用与 revision 变更拒绝、真实 token 长度过滤后的同 ID 配对、数据 seed 与训练 seed 分离、缓存文件损坏拒绝、完整 batch/context 预检、不完整 checkpoint 拒绝、实际 launcher 的预检/导出/子进程/自动五项绘图顺序。最后一项用 mock 替代 GPU 环境、Hydra 子进程和训练 transport，不能当作 GPU 集成通过。

通过新 shell 入口实际运行的 `research_runs/pipeline_validation/smoke/smoke_result.json` 状态为 `passed`，包含随机 tiny HF 模型的真实参数更新、teacher versions `[0, 1]`、checkpoint、评测、分布与表征诊断和图。其 provenance 记录执行当时的源码身份，不回填后续改动。

实际环境探测确认当前机器缺少生产训练依赖且 Transformers/NumPy 版本不满足约束；无可用 Qwen 权重、原始 DeepMath 数据或可访问训练 GPU。因此完整 Hydra 组合/rollout dataclass 子进程、FSDP 权重合并和 Qwen GPU 训练评测仍待在目标环境执行。现在 `check` 会在 Ray 启动前集中报告这些缺项；[PIPELINE_zh.md](PIPELINE_zh.md) 给出安装、准备和依次执行命令。

## 2026-10-21 服务器交接与入口调整

根 `on_policy_distillation.sh` 已按用户要求恢复为 HEAD 原版，`grpo.sh` 保持原样，两者逐字校验一致。最新一键入口独立为 `run_opsd_experiment.sh`，通过 `opsd_pipeline.sh run` 串联环境检查、数据准备、基线评测、主条件训练、合并、训练后评测与绘图；默认只打印计划，显式 `--execute` 才执行。可选 empty/test 阶段也在流程计划与状态记录中明确保存。

启动流程测试增加至 12 项（原 9 项加 3 项），覆盖实际 shell 的无写入计划、两条件/可选 test 的调度顺序、阶段失败停止与相同选项重试；GPU 阶段使用 mock，不代表完成真实大模型训练。保留的两个原仓库入口只校验内容和 shell 语法，没有启动其训练。新版交接见根 `PILOT_HANDOFF_zh.md`。

本次实际执行 `python -m unittest discover -s tests/research -v`，39 项全部通过；Python 编译、四个保留入口的 shell 语法和 `git diff --check` 均通过。

旧双教师 pilot 专用文件与重复 OPD 副本共 9 项，已在用户回复“清理”后逐项核对 SHA256 并删除，完整记录见 `legacy_cleanup_manifest.json`。原仓库两个入口与最新一键/分阶段入口均保留；数据、权重、checkpoint 和实验结果未删除。

清理后重新执行研究测试 39 项、原有动态指标测试 11 项，共 **50 项全部通过**。四个保留入口通过 shell 语法检查，一键入口 `--help` 正常；四个入口的 SHA256 均与清理前一致，原 OPD/GRPO 仍与 HEAD 逐字一致，`git diff --check` 通过。仅移除已获批的 9 个文件及空的 `scripts/legacy/`，未清理清单外的文件。

## 2026-10-26 LoRA 与全词表前向 KL 更新

按用户要求，当前实验一配置为 Qwen3-1.7B LoRA（rank 16、alpha 32、all-linear），lr=1e-6，batch=8 个问题、每题 8 条回答，500 个 rollout-batch updates，默认 1 GPU。长上下文 paper profile 继承相同训练模式/预算，GPU 数仍为 8。4B 实验二配置及实验二矩阵显式保留 full/reverse_kl；原仓库根 OPD/GRPO shell 与 HEAD 逐字一致。

训练目标是 full-vocabulary KL(stopgrad(privileged teacher) || student)，按有效回答 token 平均。新 actor 分支直接重算学生带梯度 logits、构造无梯度教师 logits，通过分块/重计算计算前向 KL；所有 microbatch 完成后才更新一次 adapter。top-16 仅负责既有五项诊断。HF 学生保持 train 模式以启用 gradient checkpointing；教师 eval/no_grad，仍使用当前 adapter。

单 epoch 的 500×8 题目预算要求至少 4000 道训练题；prepare 在初始评测前检查，旧 128 题配置会明确失败。服务器文档示例改为最多抽取 5000 题，并使用新的 consistency_lora_fkl_500 工作目录，避免混用旧缓存/训练状态。

发现并处理 LoRA 导出陷阱：vendored merger 将 adapter alpha 默认写为 0，且完整权重仅含基座。新 merge 阶段恢复本次训练 alpha，再将 adapter 合入完整 HF 权重；保留 lora_merge.json 标记，缺少合入记录时拒绝训练后评测。归档 adapter 指向原始冻结基座，避免再次加载时重复叠加。

本次实际执行结果：研究测试 **46 项通过**（原 39 + 新 7），原动态指标测试 **11 项通过**，总计 **57 项通过**。Python 编译、四个保留入口的 shell 语法、git diff --check 通过；新一键入口只预览时实际打印出 LoRA / forward_kl / 8×8 / 500 / lr1e-6 参数，不创建训练作业。

新增测试将 KL 值/梯度与独立完整词表公式比较，验证教师无梯度、padding/温度、causal shift 和 remove-padding 索引。生产 _forward_micro_batch / _update_opsd_forward_kl / update_policy 方法经 AST 提取，在真实 tiny HF 模型和可训练低秩线性适配器上执行，验证基座逐位不变、adapter 更新、整个 batch 教师固定、一次 optimizer step，microbatch 累积结果与整批 KL 更新一致。这里替代了分布式 transport，低秩适配器是测试 fixture，并非声称运行了 PEFT/FSDP 集成。LoRA 导出测试替代外部 PEFT transport，验证真实 alpha 修正及合入流程；当前解释器无可用 PEFT/完整训练依赖和 CUDA，因此真实 Qwen PEFT/vLLM/FSDP、GPU 显存、速度和最终模型评测仍未验收。历史 tiny smoke 使用旧 surrogate，不作为新 LoRA 前向 KL 的 GPU 运行证据。
