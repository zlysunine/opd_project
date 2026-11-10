# SFT / GRPO 20-step 实验结果

打开 `report/SFT_GRPO_report.html` 查看全部指标；`SFT_report.html` 与 `GRPO_report.html` 为单组报告。
`report/CONTROLLED_COMPARISON.html` 展示两组同题配对统计与预算。

本轮仅执行 SFT/GRPO，各从 Qwen3-1.7B-Base 独立训练20步，SFT通过先前2步CUDA与权重更新验收后正式训练；GRPO在本次正式运行中逐步验收。
使用历史冻结题集，监测0/1/5/10/20；逐步训练题目顺序在两组间一致。
保留所有监测点的分析结果和 LoRA adapter，仅保留最新完整恢复 checkpoint。
自主评测使用vLLM连续批处理与CUDA Graph，最多64条并发，上限16384tokens；SFT微批64，分块输出层CE，先训练20步再评测全部监测点；GRPO按32768 tokens动态微批，rollout显存比例0.70。
step0基线用同一新生成路径重新测量。几何、分布及adapter精度审计保持HF eager。

历史 OPSD 的完整结果位于仓库 `results/20261028_qwen17b_opsd20/`；本轮未重训OPSD。
历史运行缺少逐步训练题目ID收据，不能声称与这两组构成训练暴露严格配对的三组实验。
原始回答、RSM/anchor/曲率等数组仍保存在服务器正式run中，不随轻量报告省略科学测量。
