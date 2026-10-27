# 保存进度后的执行效率调整

`scripts.research.performance_resume` 从完整 checkpoint 恢复现有 OPSD 运行，保留原 frozen probe、
分析配置、完成收据和结果，不重新生成或修改既有分析身份。入口仅接受执行参数白名单：训练微批、
激活/优化器卸载、诊断 token 批量及 vLLM prefill token 批量。学习率、全局 batch、rollout 数量、
训练目标、数据划分、长度限制和总步数不能通过该入口更改。

恢复前核对 checkpoint 的模型、优化器、extra/RNG、数据加载器及其他文件 hash，核对指标与保存步边界，
核对冻结分析源码及本次执行源码，并运行真实 Hydra/schema 配置预检。显式使用 `resume_path`，
缺少数据加载器状态时拒绝启动，不自动从头训练。训练更新与微批数值等价性有 CPU 回归测试。

配置说明：旧 `config.json` 和分析 `run_manifest.json` 中的 `actual_training` 保留最初的冻结配置，
用于维持跨 checkpoint 的分析身份；续跑的实际执行微批和内存策略以本次 `profile.json`、
`effective_command.json`、`hydra_preflight.log` 以及训练器写出的 `opsd/train/verl_resolved_config.json`
为准。执行配置单独记录，不通过修改旧 manifest 或关闭 hash 校验复用历史结果。

每步仍为 8 题 × 8 条轨迹的一次全词表前向 KL 更新。增加微批只减少梯度累积次数；浮点归约及
重启后的生成轨迹可能与不间断运行不同。训练微批提高不会消除监测点的暂停：当前 latest 策略仍按
保存 → 分析 → 清理旧权重 → 继续训练执行。CPU 几何统计期间 GPU 空闲不表示训练进程出错。
