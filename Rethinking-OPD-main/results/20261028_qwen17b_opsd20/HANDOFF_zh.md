# 2026-10-28 OPSD 正式20步实验交接

状态：正式20步训练、5个监测点、质量与数值验收均已完成。当前实验和旧的98%抽取失败实验独立存放。

## 结果入口

- [结果总览（离线HTML）](index.html)：质量、KL、梯度、熵、耗时曲线。
- [完整交互报告（离线HTML）](OPSD_report.html)：全部指标、fixed/live、train/heldout、学生/教师、逐层热力图、RSM与下载。
- [两步验收报告](acceptance_2step.html)、[正式验收收据](provenance/formal_verification.json)。
- [完整汇总](summary.json)、[文件SHA256清单](artifact_manifest.json)。两个主要HTML均可下载后离线打开，无需服务器继续运行。

## 实际结果

正式训练于北京时间 2026-10-28 10:21:18 开始，2026-10-28 14:50:10 完成（包含离线监测与报告），总耗时 4.48 小时。
参数训练阶段累计 146.1 分钟，其余主要为checkpoint导出、监测与报告；训练阶段本身包含rollout。

| Step | 抽取成功数 | 成功率 | 失败率 | 正确率 | 截断率 | 完整且可抽取 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 42/50 | 84% | 16% | 50% | 8% | 84% |
| 1 | 41/50 | 82% | 18% | 48% | 8% | 82% |
| 5 | 38/50 | 76% | 24% | 46% | 8% | 76% |
| 10 | 45/50 | 90% | 10% | 46% | 0% | 90% |
| 20 | 43/50 | 86% | 14% | 46% | 2% | 86% |

门槛是抽取成功率≥70%，且完整可抽取覆盖率≥70%；全部监测点通过。最终失败率为 14%，没有再次出现98%抽取失败。
正确率由训练前 50% 变为最终 46%（-4个百分点）；50题、单随机种子的结果不足以证明稳定能力提升。
最终训练前向KL=0.019864，梯度范数=0.059098；全部20步必要数值字段通过有限性与范围检查，LoRA B实际非零，最终checkpoint状态完整。

## 配置、数据和复用关系

Qwen3-1.7B-Base；单张 NVIDIA A800 80GB；LoRA rank=16、alpha=32、all-linear；lr=1e-6；seed=42；每步8题×8条回答；20次参数更新。训练损失是全词表前向 KL(stopgrad(当前参数的特权教师) || 学生)。教师使用离线 R1 reference，随学生更新重算；没有外部冻结教师或 GRPO 对照。

- student prompt≤2048，teacher prompt（含完整reference）≤2048；超长教师参考整题排除，禁止静默截断。
- 训练及live评测回答上限16384；固定回答几何窗口最多256；live分布对全部有效回答token评分，不用256截断完整解答。
- DeepMath原生103022行，difficulty 5–7候选64942；筛选3851合格，去重后3838。
- 正式train候选池2814、独立validation512、heldout_probe50、train_probe4（train子集），剩余462题未使用。
- 20×8=160次训练题目暴露；候选池2814不等于本次全部训练过。train_probe也不保证实际被抽到。
- 两步验收使用16道train、8道validation；正式训练由同一已筛选池扩充分配，保留验收训练/验证题成员并保持隔离，没有重新筛选。
- 正式训练与验收的heldout50、train_probe4、fixed_pool、baseline_live_rollouts逐文件哈希相同；准备阶段新增generation调用为0。
- 正式20步从同一Base初始化独立训练，不是接着验收checkpoint2继续更新。后续checkpoint的live回答随新权重生成是评测必要步骤。
- heldout50取此前100题的固定前50题，保留失败；这是根据耗时作出的中途缩减，且已用于验收，不应作为从未看过的独立最终测试集。

依赖版本见 `provenance/environment_versions.json` 和 `provenance/pip-freeze.txt`；freeze中的flash-attn使用本机wheel路径，迁移机器时需提供匹配Python、Torch和CUDA的wheel，不能原样当作通用安装命令。

实际训练配置以 [training_config.json](provenance/opsd/training_config.json)、[Hydra配置](provenance/opsd/train/verl_resolved_config.json) 为准；默认 `configs/analysis_deepmath.json` 的 training 节覆盖 `configs/research/base_geometry.json` 的50步模板。历史模板中的50不代表本次实际训练步数。
模型 revision：`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`；数据 revision：`5cf055d1fe3d7a2eb19719ac020211469736ae44`。

## 本轮代码改动

- 数据筛选逐阶段进度、速度、ETA；回答生成进度；后台任务状态与质量摘要。
- 数据筛选结果在生成前持久化；异常保留工作目录；复用缓存严格检查来源和内容哈希。
- 质量收据明确记录抽取成功数、成功率和70%门槛。
- 修复空字典及不同选项映射的 Parquet 元数据问题：extra_info 用 JSON 无损传输，Dataset 兼容旧字典格式。
- 默认上层配置更新为20步、2814题候选池、50道heldout、4道train_probe、监测点0/1/5/10/20。
- 新增已验收结果导出工具，保存标量、配置、划分ID、来源哈希和离线可视化。

运行时源码哈希见 [provenance.json](provenance/opsd/train/provenance.json)；交接导出工具和文档在训练完成后追加，不属于训练算法变更。原 `SERVER_HANDOFF_MANIFEST.json` 仍是初次GitHub交接快照，未用新内容覆盖；本轮导出清单单独保存。

## 验证和解释边界

代码回归：69项，68项通过，1项因未安装Node跳过；无失败或错误。详见 [测试日志](provenance/final_code_validation.log)。
训练与分析验收检查：20步日志完整、KL/熵/梯度/overlap范围、教师参数版本、实际LoRA更新、checkpoint完整、监测文件SHA256、heldout题目ID一致、全部回答token分布覆盖、核心几何指标和HTML完整。

未定义值不补零：4题train_probe的N≤k近邻统计、短回答缺少anchor、主题内缺少配对或零方差相关等可合法未定义。核心heldout指标必须实测。`global_seqlen/*`出现明显异常的profiling统计，保留原值但不用于长度或预算结论；未启用critic的兼容占位零不表示critic学习。正确率、抽取成功率、格式合规率与截断率分开解释。
本轮不是多seed、外部测试集或与GRPO的对照实验；验收通过仅表示实现与测量达到约定门槛。

训练100%结束后的Ray清理日志出现 `DataLoader worker ... killed by signal: Killed`。主训练命令已成功返回并进入报告阶段，外层退出状态及最终文件验收见收据；当时cgroup的oom与oom_kill计数均为0，不能据此认定发生容器OOM。具体终止来源未确认。原始日志保留在服务器，交接包保存收尾日志摘录，不把该告警改写为无异常。

## 文件位置和重新打开

服务器项目：`/root/autodl-tmp/opd_project/Rethinking-OPD-main`。

| 内容 | 服务器路径 |
|---|---|
| 正式run | `/root/autodl-tmp/opd_project/Rethinking-OPD-main/research_runs/base_opsd_20261028_20steps_q70_reuse_pilot_train2814` |
| 最终checkpoint | `/root/autodl-tmp/opd_project/Rethinking-OPD-main/research_runs/base_opsd_20261028_20steps_q70_reuse_pilot_train2814/opsd/train/checkpoints/global_step_20` |
| LoRA adapter | `/root/autodl-tmp/opd_project/Rethinking-OPD-main/research_runs/base_opsd_20261028_20steps_q70_reuse_pilot_train2814/opsd/train/checkpoints/global_step_20/actor/lora_adapter` |
| Base模型 | `/root/autodl-tmp/opsd_inputs/models/Qwen3-1.7B-Base` |
| 原生数据 | `/root/autodl-tmp/opsd_inputs/datasets/DeepMath-103K/5cf055d1fe3d7a2eb19719ac020211469736ae44/native_train.parquet` |
| 筛选数据与冻结回答 | `/root/autodl-tmp/opd_project/Rethinking-OPD-main/research_runs/base_opsd_20261028_20steps_q70_reuse_pilot_train2814/shared/probe` |
| 环境 | `/root/autodl-tmp/envs/opsd` |
| 协调与验收脚本 | `/root/autodl-tmp/opd_project/Rethinking-OPD-main/research_runs/server_handoff_20261028` |

GitHub上传代码、报告、标量与来源记录；Base模型、LoRA权重、优化器、完整原始数据与隐藏状态均不上传，仍留在服务器。操作目标是关闭当前实例，不释放或删除实例。
关机在报告导出、提交推送和远端commit核验之后执行；本地执行记录位于 `research_runs/server_handoff_20261028/shutdown_receipt.json`。该记录是请求/命令状态，不能用来声称已从外部观察到控制台关机完成。

开机后查看已有结果，不会自动重新训练：

```bash
source /root/autodl-tmp/activate_opsd.sh
bash run_base_comparison.sh status --workdir research_runs/base_opsd_20261028_20steps_q70_reuse_pilot_train2814
```

从服务器已有缓存重新生成完整HTML（CPU读取结果，不加载模型、不训练）：

```bash
source /root/autodl-tmp/activate_opsd.sh
python -m analysis.html_report \
  --run-dir research_runs/base_opsd_20261028_20steps_q70_reuse_pilot_train2814 \
  --output /tmp/OPSD_report_rebuilt.html --max-step 20
```

重新导出交接包需保留原run及验收收据，输出使用新目录：

```bash
python -m scripts.research.export_run_handoff \
  --run research_runs/base_opsd_20261028_20steps_q70_reuse_pilot_train2814 \
  --output /tmp/opsd20_handoff_rebuilt \
  --verification research_runs/server_handoff_20261028/formal_verification.json \
  --acceptance research_runs/base_opsd_20261028_acceptance_2step_q70_n50 \
  --logdir research_runs/server_handoff_20261028
```

GitHub克隆可以直接离线阅读报告；仅克隆代码不包含训练所需模型、数据与checkpoint。`provenance/runtime_scripts`是本次调度脚本快照，含服务器绝对路径与原目录关系，供审计；不能直接复制到新机器当通用训练命令。重新训练应新建run，严格核验筛选缓存身份，保持候选划分与评测题固定。无需再次生成本轮已冻结数据或基线。
