# 仓库结构、脚本职责与实验产物清理清单

盘点日期：2026-11-09。范围为 `Rethinking-OPD-main/` 当前本地工作树，不包括上级 `.git/`、`SP2026_pre/`、`references/` 或服务器其他路径。初次盘点未删除文件；**随后用户删除了两个旧shell包装，本次提交同步解除依赖**。其他清理候选尚未执行。

目录大小来自实际文件；MiB 按文件内容字节数除以 1024² 计算，`du` 还会计入文件系统分配和目录开销，所以显示可能略大。机器清单见 `repository_inventory_20261109.json`；分组清理候选见 `cleanup_candidates_20261109.csv`。

## 1. 最重要的结论

1. `LlamaFactory/` 和 `verl/` 是原仓库内嵌的第三方框架。**`verl/` 已有本项目补丁，不能直接替换成最新版官方 verl，也不能整目录删除。**
2. 后来添加的 RSA/CKA/曲率/近邻等指标，主要在 `analysis/`；旧一代表征实现还在 `scripts/research/representations.py` 和 `artifacts.py`。在线 overlap/entropy 位于 `verl/verl/utils/opd_metrics.py`，不是 `analysis/metrics/rsa.py`。
3. 最新三组对照入口是 `run_controlled_comparison.sh`，配置为 `configs/controlled_deepmath20.json`。`run_base_comparison.sh` 是上一阶段入口；`opsd_pipeline.sh`/`run_opsd_experiment.sh` 是已删除的早期包装，历史Python模块CLI仍保留。
4. **旧入口与旧产物是两件事。** 8 个旧 CPU smoke 目录可作为第一批清理候选，文件内容合计约61.35 MiB、实际文件分配约66.29 MiB。它们不是两次真实 1.7B GPU 研究结果。
5. 两次真实实验的本地归档在 `results/`，约71.32 MiB 文件内容；论文问题的结果证据、配置和指标都在里面。本地没有同步两次真实训练的完整模型权重。
6. 旧shell的 `common.provenance()` 显式依赖已解除。仍需注意：`analysis.html_report.plotly_bundle()` 在未安装 Plotly 时读取第一次归档 HTML 中的 JS，不能直接删除该结果包。

## 2. 原有库与后来研究代码的边界

这里用当前仓库最早的 Git 提交 `e14c811` 作为“最初上传版本”的参照。它已经包含一部分早期 pilot 文件，不能把“首次提交已有”一律等同于“官方上游原版”。当前已提交 HEAD 为 `929ac85`，其后还有未提交/未跟踪的开发文件。

```text
Rethinking-OPD-main/
├── LlamaFactory/                    原有第三方 SFT 框架
├── verl/                           原有 RL/OPD 框架，内部已有本项目修改
│   ├── verl/trainer/ppo/            训练主循环、优势/策略损失
│   ├── verl/workers/                FSDP、actor、rollout、reward worker
│   └── verl/utils/                  原工具 + 新增 OPSD/在线指标/预算/监测工具
├── verl_example/                   原有 OPD 示例，不是另一个训练框架
├── scripts/infer/                  原有离线采样/去重工具
├── scripts/val/                    原有数学评测代码与 benchmark 数据
├── datasets/                       原有训练/评测数据，不是日志目录
├── figs/                           原论文 README 配图
├── analysis/                       后增：checkpoint 分析与几何指标主实现
├── scripts/research/               后增：配置、数据、训练调度、诊断、报表
├── configs/                        后增：研究实验配置
├── tests/research/                 后增：研究管线 CPU 回归测试
├── docs/research/                  后增：设计、指标定义、交接与审查记录
├── results/                        已归档的两次真实实验结果（Git 跟踪）
└── research_runs/                  本地工作产物、CPU 测试、报告副本（Git 忽略）
```

| 目录 | 文件内容大小（MiB） | 文件数 | 分类 |
|---|---:|---:|---|
| `LlamaFactory/` | 12.38 | 526 | 原有SFT框架 |
| `analysis/` | 0.34 | 35 | 后增checkpoint/指标分析 |
| `configs/` | 0.01 | 10 | 后增实验配置 |
| `datasets/` | 108.47 | 51 | 原有数据，不是日志 |
| `docs/` | 0.24 | 21 | 研究文档与审计 |
| `figs/` | 0.23 | 1 | 原README图片 |
| `research_runs/` | 134.77 | 2432 | 本地测试/分析/展示产物 |
| `results/` | 71.32 | 367 | 两次真实实验归档 |
| `scripts/` | 1.54 | 96 | 原始工具+后增research代码 |
| `tests/` | 0.55 | 37 | 后增研究回归测试 |
| `verl/` | 8.81 | 1144 | 原有RL框架+本项目补丁 |
| `verl_example/` | 0.01 | 2 | 原OPD示例 |

### 两个框架具体承担什么

| 目录 | 当前作用 | 与后增代码的关系 | 清理判断 |
|---|---|---|---|
| `LlamaFactory/` | 原通用 SFT 分支使用的训练框架 | `scripts/research/sft_train.py` 调用；新 `controlled_sft.py` 使用 HF+PEFT，不调用它 | 当前新三组不是它的消费者；如决定放弃原通用 SFT，可单独归档。暂不建议作为首批清理 |
| `verl/verl/` | OPSD/GRPO 的真实训练、rollout 与分布式执行 | 新 Python 调度脚本最终启动 `verl.trainer.main_ppo` | 必须保留，特别是定制训练补丁 |
| `verl/tests, examples, recipe, docs, docker, scripts` | 框架测试、示例、配方、部署资料 | 大多不是当前单卡入口直接调用，但属于原框架配套 | 可以以后做部署精简，不能与运行核心混删 |
| `verl_example/` | 原 OPD 独立示例脚本 | 不是当前 OPSD/SFT/GRPO 对照入口 | 可归档示例，体积很小 |

Git 对比显示，`LlamaFactory/` 相对最初上传没有内容变更；`verl/` 相对最初上传的后续变化如下。这里没有声称最初的 verl fork 与官方 pristine 源码相同。

| 后增或修改的 verl 文件 | 职责 |
|---|---|
| **新增** `utils/opsd.py` | 特权提示、回答 token 对齐、全词表前向 KL、共享纯张量函数 |
| **新增** `utils/opd_metrics.py` | 在线 top-k overlap、OTA、熵和熵差汇总 |
| **新增** `utils/research_budget.py` | questions/tokens/updates/wall-time 的实际预算 |
| **新增** `utils/research_monitor.py` | checkpoint 分析任务、初始 LoRA 指纹核对、同步分析钩子 |
| **新增** `tests/utils/test_opd_dynamic_metrics_on_cpu.py` | 在线指标的 CPU 数学回归 |
| 修改 `trainer/ppo/ray_trainer.py` | 学生 rollout→教师评分→更新；研究日志、checkpoint、实际题序/GRPO信号记录 |
| 修改 `trainer/ppo/core_algos.py` | 原 OPD/GRPO 优势与策略损失扩展 |
| 修改 `trainer/main_ppo.py` | 自定义配置/训练入口的对接 |
| 修改 `workers/actor/dp_actor.py` | 当前实验的直接前向 KL 更新；新 GRPO 的全 batch token 归一 |
| 修改 `workers/fsdp_workers.py` | active-LoRA 特权教师、FSDP worker、LoRA 保存/参数更新 |
| 修改 `workers/config/{actor,model,rollout}.py` | 新增训练/LoRA/采样选项的配置字段 |
| 修改 `trainer/config/model/hf_model.yaml` | 对应的模型默认配置字段 |
| 修改 `utils/dataset/rl_dataset.py` | extra_info 元数据读取与共享解码 |
| 修改 `workers/reward_manager/naive.py` | rule reward 适配、截断信息等；与评测协议对齐 |

上表路径均相对于 `verl/verl/`，测试文件例外，位于 `verl/tests/`。

## 3. 训练、诊断与报表的调用关系

```text
run_controlled_comparison.sh
  └─ scripts.research.compare
      ├─ controlled_data + analysis.prepare_probe  冻结数据/题序
      ├─ launch                                  生成配置与命令
      │   ├─ verl.trainer.main_ppo                OPSD / GRPO
      │   └─ controlled_sft                      SFT
      ├─ analysis.checkpoints                    校验/合并 checkpoint
      │   └─ pipeline.checkpoint_actor           仍复用旧模块的路径函数
      ├─ analysis.extract
      │   ├─ analysis.metrics.*                  RSA、CKA、近邻、曲率
      │   ├─ trajectory_distributions            全回答位置的分布指标
      │   └─ evaluate                            自主生成与判分
      └─ analysis.report + html_report + controlled_report
```

因此，`pipeline.py` 虽然包含旧流程，当前新入口仍调用其中的 checkpoint 路径函数。目录叫“旧流程”并不意味着整个文件已无依赖。

## 4. 根目录每个实验 shell 脚本

| 文件 | 实际作用与版本 | 当前建议 |
|---|---|---|
| `grpo.sh` | 原仓库 GRPO 入口，含 Slurm/环境变量/模型和数据路径；当前默认不是新1.7B配对配置 | 按既有要求保留原入口；不要用它替代新对照 |
| `on_policy_distillation.sh` | 原外部 teacher OPD 入口，含 top-k/reward/rollout 参数 | 保留原入口；不等同于当前 self-teacher OPSD |
| `opsd_pipeline.sh`（已删除） | 早期单组阶段shell包装 | 用户已删除；源码记录与测试已改用当前入口 |
| `run_opsd_experiment.sh`（已删除） | 早期500步一键shell包装 | 用户已删除；历史CLI仍可直接用Python模块运行 |
| `run_base_comparison.sh` | Base 表征监测阶段入口，默认只OPS D，可显式GRPO；现参数经 `analysis_deepmath.json` 指定 | 上一阶段入口，可保留供历史复现；当前源码记录也读取它 |
| `run_controlled_comparison.sh` | 最新三组入口，默认OPS D→SFT→GRPO，各20步，支持分阶段/后台控制 | 当前主入口，保留 |
| `verl_example/opd.sh` | 原 OPD 独立示例，和根目录原 OPD 脚本用途相近但实现入口不同 | 可选归档，非当前新流程依赖 |

注意：旧脚本中的“50/500步”注释、继承 profile 和最终生效配置不是同一回事。上一阶段实际训练20步，新三组默认20步；看 run 目录的 resolved_config 和预算，不按文件名判断。

此前旧双外部4B教师脚本的清理已经完成：`run_pilot_server.sh`、`pilot_thinking_consistency.sh` 等9个文件不在当前目录。详见 `SCRIPT_CLEANUP_zh.md`；`scripts/pilot/` 现在只剩 Python 字节码缓存，不是仍在使用的 pilot 代码。

## 5. 后增 `scripts/research/`：逐文件用途

“当前核心”表示当前新流程直接使用；“工具/兼容”表示可能用于历史复现、导出或测试，不表示现在可以无依赖删除。

| 文件 | 职责 | 归类 |
|---|---|---|
| `__init__.py` | Python包声明 | 基础 |
| `compare.py` | 生成各方法配置、训练、checkpoint分析、最终报告调度 | 当前核心 |
| `controlled_data.py` | 验证/采用历史probe，固定逐步题序，检查实际暴露，计算GRPO组信号 | 当前核心 |
| `controlled_sft.py` | Base/plain提示的response-only LoRA SFT、真实token计数、断点恢复 | 当前核心 |
| `controlled_report.py` | 三方法同题准确率/预算/配对统计，导出MD/JSON/HTML | 当前核心 |
| `launch.py` | 配置继承、实验检查、Hydra选项、启动verl或SFT | 当前核心，内部还有旧实验分支 |
| `data.py` | 题目标准化、DeepMath适配、划分、答案验证、原通用数据工具 | 当前核心，兼容旧CLI |
| `training_data.py` | 训练元数据JSON/Parquet无损转换、extra_info解码 | 当前核心 |
| `answer_protocol.py` | Base提示、boxed/final-answer抽取、类型/选项映射、等价判分 | 当前核心，不能删 |
| `models.py` | 分析用HF模型适配器；另含明确标识的tiny模型/ByteTokenizer | 当前核心；tiny定义不是旧结果 |
| `common.py` | JSON、hash、模型身份、provenance、共享工具导入 | 当前核心，已改为记录仍存在的实验入口 |
| `preflight.py` | CUDA/环境、Hydra和输入长度预检 | 当前核心 |
| `job_control.py` | SSH独立后台运行、status/logs/stop、进度记录 | 当前核心 |
| `preparation_cache.py` | 数据筛选结果的内容校验与恢复，避免重复筛选 | 当前核心 |
| `preparation_progress.py` | 数据准备的阶段/计数/进度与原子工作目录 | 当前核心 |
| `lora.py` | 修复导出adapter的alpha、FP32安全合并、保存BF16分析模型 | 当前核心 |
| `retention.py` | 对旧 latest 策略：分析完成后校验并轮换本run的旧checkpoint | 新三组默认all不做轮换；历史恢复需用 |
| `reward.py` | 训练rule reward，受控GRPO额外记录抽取/格式并处理截断 | 当前核心 |
| `audit_adapter_roundtrip.py` | Base/active adapter/序列化BF16合并的固定输入差异检查 | 当前验收工具 |
| `pipeline.py` | 早期单组工作流 + 当前仍使用的checkpoint路径函数 | 混合文件，不能整文件直接删 |
| `performance_resume.py` | 历史OPS D恢复时审计并调整纯执行性能参数 | 历史恢复工具，非当前默认路径 |
| `plots.py` | 从训练标量日志生成普通离线曲线；launcher仍可能调用 | 共享报表工具 |
| `run.py` | 更早的通用CLI：造数据、reference、evaluate、diagnose、extract、train等 | 历史/通用工具，旧pipeline仍调用 |
| `sft_train.py` | 原通用LLaMA-Factory SFT回调与预算统计 | 旧SFT分支；不用于新controlled_sft |
| `smoke.py` | 早期tiny模型端到端测试工具，真实更新toy参数 | 测试源代码；可删旧输出，保留生成器 |
| `artifacts.py` | 旧版按题保存激活、对齐与分析 | 旧诊断栈，`run.py`/smoke/测试仍调用 |
| `representations.py` | 旧版cosine/RSA/CKA/ARI/子空间/线性probe等指标 | 旧诊断栈，非新报告RSA主入口 |
| `distributions.py` | 旧通用分布指标、avg/pass@k等统计 | 旧诊断/评测仍调用；不同于 `analysis/distributions.py` |
| `build_metric_report.py` | 从原始run导出完整指标浏览器、CSV和静态图，第一份归档形式 | 历史/可选导出工具 |
| `metric_report.html` | 上一个浏览器的页面模板 | 与上一个脚本成套 |
| `export_run_handoff.py` | 将已验证run打包成脱离服务器的结果归档，不含大权重 | 服务器归档工具，第二次归档形式 |
| `build_experiment_reading.py` | 按第一次叙述格式复核第二次导出包、生成中文分析HTML/MD/JSON | 第二次报告工具 |
| `experiment_reading.html` | 第二次叙述报告模板 | 与上一个脚本成套 |
| `experiment_reading.js` | 第二次27图和筛选控件的绘图逻辑 | 与上两个文件成套 |
| `build_offline_comparison.py` | 将两份报告内嵌进一个可转发的桌面HTML | 演示/导出工具 |
| `build_mobile_comparison.py` | 两次结果的静态手机版HTML/PDF摘要 | 演示工具；PDF重建需可用中文字体 |
| `export_second_analysis_md.py` | 第二次完整中文分析+27张默认视图图片，供修改导出PDF | 用户当前可编辑报告工具 |
| `requirements-analysis.txt` | 分析环境依赖列表，不是日志 | 环境配置 |

## 6. 新指标究竟在哪实现

| 文件 | 计算内容 |
|---|---|
| `analysis/prepare_probe.py` | 冻结题集/问题ID/回答token及anchor成员关系；它不是训练器 |
| `analysis/extract.py` | 在各decoder block取hidden states，组装逐题矩阵、调用指标、保存缓存；负责位置对齐 |
| `analysis/metrics/rsa.py` | 余弦RSM、严格上三角Spearman RSA、RSM RMSE、题目ID对齐 |
| `analysis/metrics/spectrum.py` | linear CKA、参与率PR、谱占比和激活范数 |
| `analysis/metrics/neighbors.py` | 原始向量欧氏距离kNN与邻居重叠，排除自身 |
| `analysis/metrics/trajectory.py` | 回答连续隐藏位移的转角曲率、同题变化和相关 |
| `analysis/metrics/paired.py` | 同一步S_s/T_s的联合有效题集与直接RSA/CKA/近邻比较 |
| `analysis/trajectory_distributions.py` | 整条回答有效位置、KV缓存/分块LM head的完整分布诊断 |
| `analysis/distributions.py` | 全词表熵/KL、top-k overlap/OTA及按token分子分母汇总 |
| `analysis/evaluate.py` | 当前checkpoint自主生成、正确率/抽取/格式/截断；fixed几何与live评测分开 |
| `analysis/checkpoints.py` | 初始LoRA校验、FSDP/native SFT模型导出，非指标公式 |
| `analysis/common.py` | 分析配置、源hash、表写入和缓存身份，非指标公式 |
| `analysis/report.py` | 从已有分析表画静态图/比较图并调用完整HTML生成 |
| `analysis/html_report.py` | 当前与历史结果的离线HTML、指标目录和内嵌图片/数据 |
| `analysis/report_template.html` | 完整指标HTML的交互UI |
| `analysis/smoke.py` | 独立分析链路的tiny模型测试；step1是人为参数扰动，不是GPU训练 |

训练时在线指标走 `verl/verl/utils/opd_metrics.py`；训练本身走 `dp_actor.py`/`fsdp_workers.py`/`ray_trainer.py`。删除报表输出不会删除算法，但删除这些新增工具会破坏训练或分析。

## 7. 原有推理/评测脚本与配置文件

| 路径 | 用途与清理判断 |
|---|---|
| `scripts/infer/dedup_deepmath.py` | 原DeepMath/DAPO问题去重预处理工具，不是本轮分层probe选择实现 |
| `scripts/infer/vllm_rollout.py` | 原批量vLLM离线采样/输出筛选工具，非本轮三组主入口 |
| `scripts/val/eval/gen_vllm.py` | 原benchmark批量生成评测答案 |
| `scripts/val/eval/grade.py` | 原生成文件批量判分/汇总 |
| **`scripts/val/eval/utils.py`** | 数学答案规范化/等价判定，当前 `answer_protocol.py` 仍加载，必须保留 |
| `scripts/val/data/` | 原AIME/AMC/MATH等benchmark数据；不是旧训练日志 |
| `configs/controlled_deepmath20.json` | 当前三组20步公共参数，固定Base和数据revision |
| `configs/analysis_deepmath.json` | 上一轮20步Base OPSD的分析/训练公共参数 |
| `configs/research/base_geometry.json` | Base分析阶段继承profile；里面有旧50步说明，最终由上层training段覆盖 |
| `configs/research/pilot_1p7b.json` | 早期1.7B训练profile，也是当前继承链的基础；不能因“pilot”字样直接删 |
| `configs/research/pilot_4b.json` | 早期4B profile，当前1.7B对照不直接采用；旧矩阵/测试可能使用 |
| `configs/research/paper.json` | 历史paper档配置，非当前20步预算；与测试/旧CLI相关 |
| `configs/research/models.json` | 旧通用模型注册表；新compare另写每run的models.json |
| `configs/research/plain_completion.jinja` | Base普通提示的模板；删除会破坏新训练 |
| `configs/research/smoke.json` | CPU测试配置，不是旧输出 |
| `configs/research/extensions.schema.json` | 扩展参数说明/schema，不是执行日志 |
| `tests/research/` | 管线、公式、对齐、报表、恢复测试源代码；不等于可清的smoke输出 |

### 根目录说明文档和校验清单

| 文件 | 用途 |
|---|---|
| `README.md` | 原论文/原仓库使用说明，不是新三组实验的最终参数表 |
| `PILOT_HANDOFF_zh.md` | 当前交接入口索引，也保留第二次历史结果入口 |
| `CONTROLLED_COMPARISON_HANDOFF_zh.md` | 最新三组配置、环境、2步验收、20步执行和服务器检查流程 |
| `CONTROLLED_COMPARISON_MANIFEST.json` | 最新开发的源码快照及关键依赖校验，不是实验数据 |
| `FINAL_RESULTS_zh.md` | 已完成第二次20步的结果索引，不代表新SFT/GRPO已运行 |
| `SERVER_HANDOFF_MANIFEST.json` | 更早版本的436文件交接快照，作为历史审计保留 |
| `docs/research/archive/` | 先前交接原文，含已经变更的50/200/500步等设计 |
| `docs/research/SCRIPT_CLEANUP_zh.md`、`legacy_cleanup_manifest.json` | 已获批执行的9个旧pilot文件清理记录，不是本次40项候选的批准 |
| `research_runs/step20_local_analysis/CONTEXT_HANDOFF_zh.md` | 当时分析会话的压缩交接笔记，不是模型训练日志 |
| 两份分析目录中的 `ANALYSIS_zh.md` | 结果解读正文；可编辑稿和科研结论应按文档保留 |

## 8. 当前每个本地实验/分析目录

| 路径 | 文件内容大小（MiB） | 文件数 | 用途 | 建议 |
|---|---:|---:|---|---|
| `research_runs/20261028_second_analysis/` | 7.67 | 33 | 第二次真实实验中文报告/编辑稿/图片 | 保留，MD与图片成套 |
| `research_runs/base_geometry_cpu_smoke/` | 8.93 | 321 | 几何分析CPU smoke，人为权重扰动 | 第一批候选 |
| `research_runs/base_geometry_cpu_validated/` | 8.95 | 321 | 几何分析CPU验证副本 | 第一批候选 |
| `research_runs/base_geometry_release_cpu/` | 8.95 | 321 | 几何分析CPU交付副本 | 第一批候选 |
| `research_runs/controlled_cpu_validation/` | 0.04 | 3 | 当前三方法开发的CPU日志与计划 | 建议保留 |
| `research_runs/pipeline_validation/` | 7.33 | 296 | 早期通用管线CPU端到端验证 | 第一批候选 |
| `research_runs/smoke_cpu/` | 5.80 | 232 | 早期tiny训练/诊断测试 | 第一批候选 |
| `research_runs/smoke_cpu_verified/` | 7.05 | 294 | 旧版CPU验证副本 | 第一批候选 |
| `research_runs/smoke_delivery/` | 7.21 | 294 | 旧版交付CPU验证副本 | 第一批候选 |
| `research_runs/smoke_final/` | 7.14 | 294 | 旧版最终CPU验证副本 | 第一批候选 |
| `research_runs/step20_local_analysis/` | 49.49 | 19 | 第一次真实实验本地复算/中文报告/重建代码 | 保留主体；可清重复完整HTML |
| `results/20261028_qwen17b_opsd20/` | 29.02 | 60 | 第二次真实GPU结果导出 | 整包保留或验证后外部归档 |
| `results/step20_all_metrics/` | 42.30 | 307 | 第一次真实GPU结果导出 | 整包保留或验证后外部归档 |

表中8个旧CPU目录均在 `research_runs/`。`smoke_result.json` 已证实它们是tiny随机模型测试或人为扰动；其中的 `checkpoint_1/2`、`model.safetensors` 不是真实Qwen3-1.7B训练权重。

| 本地单文件/目录 | 用途 | 建议 |
|---|---|---|
| `research_runs/step20_local_analysis/OPSD_step20_analysis.html` | 第一次实验的中文图解分析 | 保留便于研究对比 |
| 同目录 `OPSD_metrics_complete.html` | 从第一次归档重建的完整指标浏览器，约38.52MiB | 可重建副本，第二批可清候选 |
| 同目录 `analyze.py`、`build_reading_html.py` | 第一次本地分析/报告的两个重建脚本 | **代码和产物混放**，不能整目录清空；若归档需一起保留 |
| 同目录 `privilege_examples/` | 抽取失败排查时匹配的reference示例与抽取审查 | 研究排查证据，不是训练数据池 |
| `research_runs/20261028_second_analysis/OPSD_second_experiment_analysis.html` | 第二次中文图解分析，27张交互图 | 保留 |
| 同目录 `.md` 与 `_figures/` | 用户可编辑稿和27张静态图片 | 需成套保留；不能只移动MD |
| 两个分析目录的 `summary.json` | 本地复算的统计与配对区间 | 手机摘要/后续解释会读取，建议保留 |
| `research_runs/OPSD_experiments_comparison.html` | 用相对路径iframe并排查看两份报告，约4KiB | 依赖两份原HTML，删目录会失效 |
| `research_runs/OPSD_experiments_comparison_offline.html` | 两份报告内嵌的桌面单文件，约15.59MiB | 演示副本；可留一份转发，也可归档 |
| `research_runs/OPSD_experiments_mobile.html` | 手机静态摘要，约0.53MiB | 演示副本，可重建 |
| `research_runs/OPSD_experiments_mobile.pdf` | 手机/导师转发用8页PDF，约92KiB | 很小，建议保留 |
| `research_runs/controlled_cpu_validation/` | 当前三组代码的CPU验证日志与无GPU预览，不是新GPU结果 | 只有约43KiB，建议保留 |

### 两个真实结果归档要区别对待

| 目录 | 内容 | 能否直接删 |
|---|---|---|
| `results/step20_all_metrics/` | 第一次：训练CSV、checkpoint标量、逐题分布/曲率/行为、原报告、PNG和manifest | 目前不宜：报告JS备用源和历史复算来源；需先脱离依赖、备份后再移出 |
| `results/20261028_qwen17b_opsd20/` | 第二次：完整报告/2步验收报告、20步训练指标、逐题表、配置/环境/来源指纹 | 研究结果主归档；旧报告builder及一项本地回归读它。优先归档，不建议拆包删单文件 |

两目录受Git跟踪。删工作树文件不会清除Git历史中已有的大对象。源包的manifest逐文件记hash；只删里面看似重复的HTML/PNG会使整包复核失败。保留HTML截图不能替代逐题数据、配置和原始数值数组。

## 9. 本机实际存在的4个文本日志

| 文件 | 内容 | 清理建议 |
|---|---|---|
| `research_runs/controlled_cpu_validation/research_tests.log` | 当前研究CPU回归输出，164项 | 很小，保留当前开发证据 |
| 同目录 `online_tests.log` | 原在线动态指标CPU数学检查输出，11项 | 很小，保留 |
| `results/20261028_qwen17b_opsd20/provenance/final_code_validation.log` | 第二次服务器归档的代码回归检查输出 | 与归档manifest成套保留 |
| 同目录 `training_teardown.log` | 第二次收尾/worker告警记录 | 排查Killed等现象的来源，成套保留 |

**本机未同步两次1.7B训练的完整 `train.log`、GPUcheckpoint目录和服务器进程会话目录。** `provenance/source_files.json` 里列出服务器路径，并不表示那些原始文件都已经下载到本机。不要把“来源路径清单”误当作本地存在的实验数据。

## 10. 指标表/配置/数组也是实验记录，不只是 .log

### 当前真实结果导出包

| 文件或类型 | 作用 |
|---|---|
| 第一次 `data/training_metrics.csv` / 第二次 `data/training_metrics.jsonl` | 每次训练更新的loss、在线指标、资源与预算；重新作图的重要来源 |
| `monitor_metrics.csv` / `checkpoint_metrics.csv.gz` | checkpoint×层×anchor×role×metric的标量，含n_valid/undefined |
| `distributions_samples.csv.gz` / `distributions.csv.gz` | 逐题分布量与有效token分子分母，可重算公平加权和问题bootstrap |
| `curvature_samples.csv.gz` / `curvature.csv.gz` | 逐题逐层曲率，可作同题训练变化、置信区间与相关 |
| `behavior_samples.csv.gz` / `behavior.csv` | 每题每checkpoint的正误、抽取、截断、长度，可作gain/loss配对 |
| `probe_membership.csv` | 题目ID所属split/主题等，区分heldout/train_probe |
| `report_data.json` | 第一次旧浏览器的完整数据载荷，不是模型checkpoint |
| `summary.json` | 运行概要/统计；服务器版与本地复算版有不同字段和用途 |
| `artifact_manifest.json` / `manifest.json` | 导出包逐文件完整性清单 |
| `provenance/config.json`、`opsd/train/*config.json`、`command.json` | 实际运行配置/命令，不能用最新默认配置替代 |
| `pip-freeze.txt`、`environment*.json` | 服务器依赖/设备环境审计 |
| `source_files.json`、`provenance.json`、各step `run_manifest/complete.json` | 来源身份、输入/输出hash与缓存完成收据 |
| `provenance/runtime_scripts/*.py` | 当时服务器使用的辅助脚本快照，属于历史证据，不是当前入口 |

### 未来服务器原始run会生成的记录（当前本机没有这些真实run目录）

| 路径模式 | 作用 | 清理边界 |
|---|---|---|
| 原shell的 `logs/run_*.log`、Slurm `logs/<日期>/output_*/error_*` | 原训练脚本完整输出及调度器stdout/stderr | 目前本机未发现；历史run结束后可压缩归档 |
| 原shell的 `checkpoint/swanlab_log/` 或示例 `swanlog/` | 实验跟踪器本地事件/缓存 | 与具体run关联；不能冒充模型checkpoint |
| `.session/attempts/<id>/console.log` | 后台总调度/阶段异常输出 | 结束后可归档；运行中不要清理会话目录 |
| `.session/.../{request,state,progress}.json` | 后台进程身份、启动命令、阶段和状态 | 当前会话控制所需 |
| `<method>/train/train.log` | verl/native SFT训练stdout/stderr | 保留最近有效run和异常段；历史成功run可压缩 |
| `<method>/train/metrics.jsonl` | 科学标量和实际预算；恢复会读取 | 不等同于可随意丢弃的console日志 |
| `preflight.json`、`hydra_preflight.log`、`rollout_schema_preflight.log` | 环境/配置启动前检查 | 小文件，建议留在归档 |
| `shared/probe/*.jsonl` | 冻结训练/验证/probe题目、reference和固定回答 | **未来严格复现及配对对照必须保留** |
| `shared/probe/train_schedule.jsonl`、`question_schedule.json` | 三方法实际问题顺序与train_probe暴露记录 | 当前新对照必需 |
| `<method>/train/exposures/step_*.json` | 每步实际看到的问题ID收据 | 验证三方法公平性，不建议删 |
| `<method>/jobs/*.json`、`job_status/*.json` | checkpoint分析任务与完成收据 | 恢复/清理状态依赖，不能任意删 |
| `analysis/steps/step_*/live_rollouts.jsonl` | 原始生成文本/token、抽取和正误 | 分析lost/gained和格式问题必须保留 |
| `analysis/steps/step_*/anchors.npz`、`anchors.json` | 原始anchor向量和ID/位置对应 | RSA复算、逐题cosine及后续对比的原材料 |
| `analysis/steps/step_*/rsm_*.npz` | 各层anchor的题间相似矩阵 | 原始几何证据；PNG不能替代 |
| `neighbors.npz`、`student_teacher_*.json/npz` | 近邻索引/距离及联合有效题集 | 用于核对配对与复算 |
| `analysis/steps/step_*/{metrics,curvature,distributions}.*` | 标量和逐题分析表 | 保留可比统计的基础 |
| `analysis/steps/step_000000/` | 相对初始参数的基线缓存 | 继续分析后续step时仍会被读取，不能单独丢 |
| `train/checkpoints/global_step_*/` | 可恢复的参数/优化器/RNG或FSDP数据状态 | 体积最大，但必须按明确的checkpoint保留策略处理 |
| `hf/solution/global_step_*/` | 供分析/评测的合并模型 | 有完整训练checkpoint和正确导出链时才可重建；当前新对照默认all保留 |
| `report/*.png`、`report/OPSD_report.html` | 表格/数组的展示产物 | 最容易重建；保留一份最终报告后可减少重复导出 |

两个历史导出包没有提供完整原始hidden arrays/回答文本，无法凭本机的scalar CSV重新得到全部未采集指标。对服务器清理应另做实盘大小与可恢复性检查，本盘点不能代替服务器盘点。

## 11. 建议的清理顺序

### A：可作为第一批清理的旧CPU产物（尚未删除）

```text
research_runs/pipeline_validation/
research_runs/smoke_cpu/
research_runs/smoke_cpu_verified/
research_runs/smoke_delivery/
research_runs/smoke_final/
research_runs/base_geometry_cpu_smoke/
research_runs/base_geometry_cpu_validated/
research_runs/base_geometry_release_cpu/
```

这8目录共2373个文件，内容约61.35MiB，文件分配约66.29MiB；不影响当前新三组训练逻辑。需要历史验证输出时可重跑对应smoke。`__pycache__/` 和残留 `scripts/pilot/__pycache__/` 也属于可再生缓存，另列在CSV，不属于科研结果。

### B：选择保留一份展示版本，再清理重复导出

优先候选为 `research_runs/step20_local_analysis/OPSD_metrics_complete.html`（约38.52MiB）。保留第一次 `results/step20_all_metrics/` 时可重建：

```bash
python -m analysis.html_report \
  --run-dir results/step20_all_metrics \
  --output research_runs/step20_local_analysis/OPSD_metrics_complete.html
```

A+B 文件分配合计约104.8MiB，不含额外缓存。可选再归档15.59MiB的桌面合并HTML，但它是先前要求制作的转发版本，是否保留取决于使用需求。

不要整目录删除 `step20_local_analysis/`：它包含两个重建脚本和排查证据。不要只保留第二次MD而删除其 `_figures/` 图片。

### C：旧源码入口先做依赖拆分，再决定删除

1. 将 `pipeline.py` 里的 `checkpoint_actor/train_directory` 抽成共享模块，修改 `compare.py`/`analysis.checkpoints.py` 的调用；再讨论旧pipeline是否退役。
2. 已完成：用户删除两个旧shell后，`common.provenance()` 改为记录当前入口，测试改用现有shell和保留的Python模块CLI，新交接清单已刷新。
3. 原通用 `run.py`、旧 `artifacts.py/representations.py/distributions.py`、`sft_train.py` 与LLaMA-Factory分支要按整条调用链归档，不能只删名字像旧文件的单个模块。
4. 旧 `SERVER_HANDOFF_MANIFEST.json` 是历史来源证据，不应伪装更新为新历史；新受控清单则要在真正完成源码调整后重新生成。

删除旧入口节省空间很少，主要收益是减少混淆；它与清理实验产物应分开操作。

### D：真实研究归档、原数据与第三方库的精简

- `results/` 优先移到独立归档存储并验证整包，而不是无备份删掉。先解除Plotly备用依赖、调整报告builder与相关回归。
- `datasets/` 约109MiB的 `du` 占用是原数据。当前新管线使用服务器外部 native DeepMath 和冻结probe，不能把这个目录误认成当前2814题或旧日志。
- 若确定只保留新三组，可将不使用的原benchmark和LLaMA-Factory作为单独精简方案；这会影响原入口/旧通用SFT能力，应单独确认范围。
- 代码清理尽量在新实验prepare之前完成。源码指纹变化会阻止旧冻结run继续执行；不要在服务器训练中途清理源码再绕过校验。

## 12. 当前不能使用“一键扫掉所有未跟踪文件”的原因

最新 `run_controlled_comparison.sh`、`controlled_*.py`、新配置和交接文档当前仍有未跟踪文件。`research_runs/` 又被Git忽略，其中包含你正在阅读和修改的报告。因此本次清单采用具体路径，未执行 `git clean` 或通配删除。

建议后续按“先A、再B、最后C/D”的组别逐项确认；CSV记录了每项实际路径、体积、用途、删除影响和前置依赖。上级 `SP2026_pre/` 与IDE配置不在这次清理范围内。
