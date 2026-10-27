# 批量监测生成与可恢复准备

2026-10-27：当前 Base 实验仍为 OPSD、3,600 道训练题、200 次更新、batch8×8 rollout。
训练损失、LoRA 参数、输入长度和监测题目不变。`monitor.generation_batch_size=16` 只控制分析用的回答生成。

`Model.generate_batch` 对问题左填充，共享一次模型前向；固定池采样为每题独立的 Torch Generator，seed
仍由 monitor seed、problem ID 和 pool 派生。默认 live pool 为 greedy。批量计算可能改变旧串行方式的
生成 token，因此新运行建立新固定池并记录后端、batch size 和 seed，不与旧池混用。

每完成一批，将各题结果原子写入 `shared/generation_cache/fixed_base_rollout/`；
分析中的生成缓存位于 `opsd/analysis/generation_cache/step_XXXXXX/`。缓存检查模型、题目、
配置、源码和结果 hash，失败时保留已有项。相同身份重试只生成缺失项，按原顺序返回。

准确计数分别保存在 `shared/generation_progress.json` 和 `opsd/analysis/generation_progress.json`，
控制台每批打印 `GENERATION_PROGRESS`。退出监控不会停止后台任务。

CPU 筛选缓存通过 `dataset.preparation_cache` 显式指定。复用前核对原始数据和模型内容、长度限制、
数据划分参数、选择逻辑 AST、数据适配器与核验器，以及全部文件 hash 和 split 隔离关系。
生成逻辑变更不会使纯数据选择失效；选择规则、输入内容或文件变化会拒绝复用。
新 probe 保存 `preparation_reuse.json` 追溯旧运行，不伪造旧运行的完成标记。

旧串行进程仅在内存中保存尚未完成的回答，不能从其落盘文件恢复这些回答。重启备份保留了全部已落盘
选择结果、配置、源码、任务状态及日志；新运行重新生成固定回答。

```bash
/root/autodl-tmp/envs/opsd/bin/python /root/autodl-tmp/opsd_inputs/watch_preprocessing.py --watch
```

完整分析中的隐藏状态提取与几何计算仍按原协议执行；回答生成吞吐提升不能直接等同于整项实验同比提速。
