# 最新实验交接入口 · 2026-10-28

本轮两步验收及正式20步训练均已完成。请以 [最终交接文档](results/20261028_qwen17b_opsd20/HANDOFF_zh.md) 和 [结果入口](FINAL_RESULTS_zh.md) 为准。

实际配置：20步、batch8×8、LoRA r16/alpha32、lr1e-6、prompt/teacher prompt2048、response16384；抽取成功率门槛70%。正式train候选池2814、validation512、heldout50、train_probe4，监测点0/1/5/10/20。筛选数据、heldout、train_probe、基线和固定回答按交接记录复用。

[上一阶段交接原文](docs/research/archive/PILOT_HANDOFF_before_final_20261028_zh.md) 含调整前的512题heldout等历史计划，保留供审计；不可当作本次实际运行配置。原始GitHub交接manifest保持不变，本轮结果另有SHA256清单。
