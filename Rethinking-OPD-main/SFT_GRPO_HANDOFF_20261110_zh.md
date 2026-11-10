# SFT / GRPO 20 步实验交接报告（2026-11-10）

## 结论和入口

本轮已全部完成：SFT 与 GRPO 各独立训练 20 步，0/1/5/10/20 全部监测点分析、初始 LoRA 一致性检查、题目暴露核验、adapter 合并精度审计、离线浏览器验证及报告打包均通过。全部结果于北京时间 **2026-11-10 14:30** 整理完成。本轮没有重新训练 OPSD，也没有配置实例自动关机。

优先使用以下入口，避免在 `research_runs` 中寻找中间文件：

- [统一报告入口](results/20261110_qwen17b_sft_grpo20/index.html)
- [完整 250 项指标交互报告](results/20261110_qwen17b_sft_grpo20/report/SFT_GRPO_report.html)
- [SFT 单组报告：173 项](results/20261110_qwen17b_sft_grpo20/report/SFT_report.html)
- [GRPO 单组报告：246 项](results/20261110_qwen17b_sft_grpo20/report/GRPO_report.html)
- [同题配对统计与预算](results/20261110_qwen17b_sft_grpo20/report/CONTROLLED_COMPARISON.html)
- [完成验收记录](results/20261110_qwen17b_sft_grpo20/verification.json) 与 [实验原始交付清单](results/20261110_qwen17b_sft_grpo20/RESULT_MANIFEST.json)

HTML 含 Plotly 和报告数据，下载后可直接在浏览器离线打开；GitHub 文件页通常显示源码，不直接运行交互页面。指标入口数包含不同来源、别名和预算字段，不代表 250 个相互独立的科学量。

## 实验做了什么

| 项目 | 实际配置 |
|---|---|
| Base | Qwen/Qwen3-1.7B-Base；两组各自从 Base 初始化，不串联训练 |
| 模型 revision | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |
| 数据 | zwhe99/DeepMath-103K，难度 5–7，reference=`r1_solution_1` |
| 数据 revision | `5cf055d1fe3d7a2eb19719ac020211469736ae44` |
| 冻结划分 | train=2814、validation=512、heldout=50、train_probe=4 |
| 训练暴露 | seed=42，固定顺序的 160 道不同训练题，20 步 × 8 题；两组每步 ID 收据完全匹配 |
| SFT | 每题 reference 重复 8 次，每步实际训练 64 条监督序列；response-only 全局有效 token 平均 CE |
| GRPO | 每题采样 8 条轨迹，每步 64 条；0/1 答案奖励，组内标准化优势，无 KL/熵/格式奖励 |
| LoRA | r=16、alpha=32、all-linear、dropout=0、bias=none；两组初始张量指纹完全一致，B=0 |
| 优化器 | AdamW，lr=1e-6，betas=(0.9,0.999)，eps=1e-8，weight_decay=0.01，clip=1；constant、warmup=0 |
| 精度与设备 | 一张 A800 80GB；FP32 参数/优化器，BF16 计算；FP32 合并后保存 BF16 分析模型 |
| 长度 | prompt≤2048、response≤16384；固定表征窗口为前 256 个回答 token |
| 监测 | 0/1/5/10/20；heldout50 + train_probe4；greedy 自主回答，答案协议 opsd_answer_v2 |
| 表征/分布 | 固定 frozen response IDs，prompt_last/response32/response128，逐层曲率、RSA/CKA、RSM、邻域、谱、分布 KL 等 |
| 最新执行配置 | [controlled_sft_grpo20_throughput_server.json](configs/controlled_sft_grpo20_throughput_server.json) |

完整训练配置和设计收据保存在结果目录的 `config.json`、`design_manifest.json`、`sft/training_config.json`、`grpo/training_config.json`。`controlled_deepmath20.json` 是原三组设计模板；本次仅请求 SFT/GRPO，不能把模板的 OPSD loss 字段当成本轮实际训练方法。

## 最终观察结果

共同新 Base 基线为 24/50（48%）。监测点的 heldout 正确题数如下：

| 优化步 | Base / SFT | Base / GRPO |
|---:|---:|---:|
| 0 | 24/50 | 24/50 |
| 1 | 25/50 | 24/50 |
| 5 | 24/50 | 23/50 |
| 10 | 24/50 | 26/50 |
| 20 | **25/50（50%）** | **26/50（52%）** |

| 最终指标 | SFT | GRPO |
|---|---:|---:|
| 相对基线净变化 | +1 题 / +2 个百分点 | +2 题 / +4 个百分点 |
| 相对基线 gained / lost | 3 / 2 | 3 / 1 |
| 问题 bootstrap 95% 区间 | [-6,+10] 个百分点 | [-4,+12] 个百分点 |
| 配对 McNemar p | 1.0 | 0.625 |
| 抽取失败率 | 22% | 10% |
| 截断率 | 4% | 2% |
| 最后训练 loss | CE 0.6981366 | policy loss 0.0942095 |
| 实际优化 token 数 | 2,048,096 | 938,398 |
| 自主训练 rollout token 数 | 0 | 938,398 |
| 记录的训练时间预算 | 325.37 秒 | 1,443.06 秒 |

GRPO−SFT 为 +2 个百分点，配对 95% 区间 [-4,+8]、McNemar p=1.0。这是单 seed、50 题的小样本观察，不能据此认定 GRPO 稳定优于 SFT，也不能直接比较两种不同定义的 loss。训练时间预算排除了监测暂停，不是环境安装、重启、全量分析及报告生成的端到端时长。SFT 同一参考的 8 次重复不增加独立监督信息；相同步数、题数和序列数不等于相同 token 或计算预算。

统计定义、每个监测点行为和配对结果见 `report/controlled_summary.json`。完整逐步训练标量及每步题目 ID 保存在 `sft/`、`grpo/` 归档中。

## 提速、错误修复和实际执行顺序

1. 迁移后的模型和公开数据按原 revision 补齐，冻结题集 14 个文件逐一校验，原始 Parquet 为 103,022 行，SHA 与历史原生数据完全一致。没有重新筛选或替换历史冻结题集。
2. 先前 SFT 两步 CUDA 验收已确认真实权重更新。完整长回答评测曾阻塞正式训练，旧 eager/fast 目录和源码快照保留作审计。
3. SFT 改为微批 64，输出投影与 CE 分块 checkpoint（1024 tokens），不物化完整 `[B,T,vocab]` logits，保持整步 response-token 归一化和 64 条真实序列。CPU 测试覆盖不同长度、padding 和不同微批的损失/梯度一致性。
4. 新自主评测使用 vLLM0.11 连续批处理、最多 64 条并发与 CUDA Graph；原 16384-token 上限不变。完整 54 题 Base 试跑耗时 114.65 秒，含两条达到上限的回答。生成完成逐条落盘，结束后释放独立引擎，HF eager 继续完成几何/分布提取。
5. GRPO 使用 32768-token 动态微批，保持整步全局有效 token 归一化；关闭 activation CPU offload，rollout 显存比例 0.70，监测生成显存比例 0.80。
6. 实际顺序：SFT20 → SFT0诊断 → GRPO20及在线监测 → SFT剩余监测点 → 清理已分析旧完整权重 → 两组 roundtrip 审计 → 报告与浏览器检查。所有待分析 SFT 权重一直保留。
7. GRPO第1步更新成功后，评测曾因训练进程缓存未释放而失败：空闲56.49GiB，小于引擎申请63.4GiB。保留第1步 checkpoint 后，在 FSDP checkpoint manager 的 load/save 原实现外增加 GC、CUDA同步和 empty_cache。原读写实现体 AST 未改，优化器/RNG/数据与损失不变；从第1步恢复后继续到20，第5步后保存→评测交接也已验证。
8. 报告后处理首次因没有激活项目 Python 导致 `analysis` 导入失败；用配置好的环境重新执行后，三份报告浏览器检查与ZIP CRC均通过。这是报告后处理错误，没有重做训练。

SFT 20步更新计算合计322.46秒，中位16.13秒/步；相同前两步由31.3/29.9秒降至16.5/14.6秒，约提速1.9–2.1倍。物理显存实测峰值80875/81920MiB。GRPO的PyTorch累计峰值会受vLLM睡眠/虚拟池重映射影响，可能超过实体显存；保留这些原计数，但物理占用请看 `gpu_physical_telemetry.jsonl`，该5秒采样从恢复阶段开始，不覆盖最早全部训练。

生成内核和批处理数值路径已改变，因此旧 eager Base 的25/50与本轮新基线24/50不同。两组使用同一新 vLLM Base 缓存，复用时核对行、模型、revision、monitor设置、源码及逐项结果哈希。新旧轨迹并非逐位等价。

## 服务器上的文件和环境

项目根目录：`/root/autodl-tmp/opd_project/Rethinking-OPD-main`。

- 正式运行：`research_runs/sft_grpo20_throughput_20261110/`。
- 每组最新完整恢复状态：`<method>/train/checkpoints/global_step_20/`；目前均只保留这一完整 checkpoint。
- 全部小 LoRA：`<method>/adapter_archive/global_step_{0,1,5,10,20}/`。
- 全部原始分析：`<method>/analysis/steps/step_000000`、`000001`、`000005`、`000010`、`000020`；包含回答、anchors/RSM/邻域数组及完整表格。
- 冻结题集源：`research-runs/base_opsd_20261028_20steps_q70_reuse_pilot_train2814/shared/probe/`，注意此历史目录使用连字符 `research-runs`。
- Base：`/root/autodl-tmp/opsd_inputs/models/Qwen3-1.7B-Base/`。
- 原生数据：`/root/autodl-tmp/opsd_inputs/datasets/DeepMath-103K/5cf055d1fe3d7a2eb19719ac020211469736ae44/native_train.parquet`。
- 环境：`/root/autodl-tmp/envs/sft_grpo/`；激活入口 `source /root/autodl-tmp/activate_sft_grpo.sh`。
- 运维脚本及下载/安装审计：`/root/autodl-tmp/sft_grpo_setup/`。实际脚本副本另归档在结果目录 `provenance/runtime_scripts/`，其中绝对路径属于本次服务器，迁移后需要调整，不能原样启动覆盖历史运行。
- 本机报告ZIP：`/root/autodl-tmp/SFT_GRPO_20step_results.zip`，原始离线交付包约73MiB，不另向Git重复提交ZIP。

环境主要版本：Python3.12.3、PyTorch2.8.0+cu128、Transformers4.56.1、PEFT0.17.1、vLLM0.11.0、FlashAttention2.8.1、Ray2.59.0、PyArrow21.0.0、NumPy1.26.4。完整版本见结果目录 `final-pip-freeze.txt`。venv使用服务器系统PyTorch，不能直接复制到任意新机器当作独立环境；新服务器应重新安装并验收。环境只为本任务配置，不在Git内。

Base内容指纹：`ba288b1460aa43317f35b35e1e61cfb7f975ceaabb439c1b6d455d7bef4644ee`。
原生Parquet SHA256：`e4981f9b3f3b2840b71539a1970858f1448ea798f26b1e976031cbb1fb7780b5`。
共享训练顺序SHA256：`fdf3e1523070d6735cbc2ad3d8a4c4c541f3f2af25842e3fe44de75c08681410`。

## Git归档范围与后续接手

本次提交包含更新后的源码、测试、三种服务器配置（以throughput为最终配置）、迁移清单、此交接报告、完整可视化报告、所有聚合指标、训练标量、逐步暴露收据、精度与浏览器审计、环境/来源/执行脚本记录。原交付 `RESULT_MANIFEST.json` 保留原样，核对原始报告文件；新增的交接和运维来源以 `PUBLICATION_MANIFEST.json` 补充记录。

不提交Base权重、原始数据集/冻结题集正文、venv/cache、完整checkpoint、LoRA权重、原始大数组和重复ZIP。它们仍保留在本机数据盘；换服务器必须按 [数据迁移清单](DATA_MIGRATION_CHECKLIST_zh.md) 另行迁移。只clone仓库足以阅读报告，但不足以恢复原模型和逐步原始回答/表征数组。不要仅凭Git内50题统计重新构造冻结题集。

历史 [OPSD报告](results/20261028_qwen17b_opsd20/index.html) 仅作独立参照：历史训练没有逐步题目ID收据，不能声称本次与历史OPSD构成训练暴露严格配对的三组实验。

当前任务已经结束，不要为“继续检查”再次启动训练。查看状态使用 `bash /root/autodl-tmp/sft_grpo_status.sh`；不要把54/54回答生成完成当作全部几何分析完成。新增实验请使用新目录，先校验所有输入指纹，固定训练与生成配置；禁止改写旧 `complete.json`、缓存身份、原始收据或将旧结果伪装为新执行。

后续若扩展 seed/题数/步数，应重新记录设计与预算。实例不会因本流程结束自动关机。
