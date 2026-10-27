# OPSD答案协议v2：提示、抽取与判分对齐

2026-10-27。实现版本：`opsd_answer_v2`。原step20报告和原始OPD/GRPO脚本保留；新协议必须建立新运行目录，不能复用旧冻结probe或将旧结果当作同协议对照。

## 统一入口

`scripts/research/answer_protocol.py` 同时提供提示模板、抽取、答案编码校验和候选答案判分。

- `verl/verl/utils/opsd.py:plain_messages` 调用统一Base纯文本模板；教师在相同模板前加入离线reference，学生不读取reference。
- 学生与教师共同要求最后一行使用 `Final answer: \boxed{YOUR_ANSWER}`，判断题输出True/False，选择题输出题面提供的选项标签。
- `analysis/evaluate.py:answer_status`、`scripts/research/data.py:verify` 和研究reward入口复用协议，判分直接接收已抽取的candidate，不再对原响应调用旧boxed-only抽取器。
- 数学等价性仍使用仓库的mathd/SymPy比较函数；没有修改原始 `scripts/val/eval/utils.py` 的历史行为。

## 抽取语义

优先解析明确的Final answer区域，兼容普通Final answer单行及无Final answer标记的末尾完整boxed。支持boxed后的空格/换行、嵌套分式、转义集合括号与fbox；记录抽取来源。普通Final answer行可以判分，但不算标准boxed格式合规。

空白答案、占位符、未闭合括号、多个冲突Final answer区域明确失败。最后答案损坏时不回退到前面的中间答案。无标记的普通数字或“答案是42”不会通过“取最后数字”猜测。

回答逐条保存 `parse_status`、`extraction_source`、`format_compliant`、`extracted_answer`、`answer_encoding_status`、`answer_type`、`answer_protocol_version`、`correct`，并保留完整 `response_text`。抽取失败与数学答错分别记录；有正确候选但生成被截断时，当前主准确率仍计错。

## 答案编码

数据准备在参考核验之前检查标签语义：

- 判断题的标准答案必须可映射为True/False，支持大小写、Yes/No及已约定的中英文形式。
- A/B等标签必须在输入数据的 `option_mapping` / `choices` 或题面明确选项中有依据。不会从reference推理内容猜出A=True。
- 判断题仅给A且无映射、选择题缺选项、未知类型等进入 `answer_encoding_quarantine.jsonl`，不进入训练/probe划分。
- 单独字母可能是合法数学符号；无法判别时保守隔离。人工确认后可在原始输入中显式标注 `answer_type="math"`，并记录数据来源版本。
- 可显式标注 `answer_type` 为 `math` / `boolean` / `multiple_choice`。例如判断题有可靠原始映射时填写 `option_mapping={"A":"True","B":"False"}`；没有证据不能补这个映射。

原始数据文件不被就地修改。完整筛选记录和隔离样本同时写到 `shared/preparation_audit/`；有效样本数量不足仍然报错，不自动减少请求题数。

## 长度与几何窗口

当前主配置保留用户设置的训练prompt2048、训练response16384，另设：

```json
"max_probe_response_tokens": 256,
"behavior_max_response_tokens": 16384,
"quality_gate": {
  "enabled": true,
  "max_extraction_failure_rate": 0.01,
  "splits": ["heldout_probe"]
}
```

`max_probe_response_tokens` 控制初始固定文本池的生成上限和几何分析的前缀窗口；`behavior_max_response_tokens` 控制live自主解题完整回答的生成上限。行为准确率对完整回答判分，曲率/NLL只使用前256个token。live记录中的 `geometry_window_clipped` 与回答的 `truncated` 是两件事，曲率表中的 `response_length` 为实际几何窗口长度。

模型在EOS处可提前结束；`finish_reason` 当前为eos或length。没有引入额外自定义stop字符串。教师prompt上限仍为2048；参考+问题+格式指令须共同满足该限制，超长整题排除，不能静默截掉reference。生成函数另检查真实模型的context window。

## 质量门槛与持久化

1. prepare阶段先在初始Base上生成live校准回答，按全部heldout题计算抽取失败率。
2. 默认门槛≤1%，512题允许最多5题失败。小pilot同样应用比例门槛，不代表小样本通过能证明正式512题通过。
3. 超标立即报错，保留完整回答、失败样本和原因统计，不发布完成的probe，也不开始训练。
4. 通过后的初始live记录冻结到probe，并在step0分析复用。训练入口再次验证初始记录hash与质量；各后续监测checkpoint在几何提取前再次检查live质量。
5. 后续checkpoint失败时现有latest轮换流程停止并保留恢复现场；未通过的分析不会发布complete标记。质量门槛不以准确率为条件，数学答错不会被删除。

查看：

```text
<workdir>/shared/preparation_audit/answer_encoding_quarantine.jsonl
<workdir>/shared/quality/baseline/quality.json
<workdir>/shared/quality/baseline/responses.jsonl
<workdir>/shared/quality/baseline/failed_responses.jsonl
<workdir>/opsd/analysis/quality/step_XXXXXX/quality.json
<workdir>/opsd/analysis/quality/step_XXXXXX/failed_responses.jsonl
```

生成缓存也支持batch_size=1，失败/中断后同一身份可复用已完成项。不能为了过门槛反复采样、删除失败题或提高阈值后冒充原协议。模板、判分模块、数学比较器、模型适配代码纳入内容身份，变化后拒绝旧缓存。

## 服务器运行

默认新目录为 `research_runs/base_geometry_50_answer_v2_full_tokens_paired`；预算仍为50步、batch8×8、lr1e-6、LoRA，默认仅OPSD。

```bash
bash run_base_comparison.sh \
  --model-dir "$OPSD_MODEL_DIR" --model-revision "$OPSD_MODEL_REV" \
  --input "$OPSD_DATA_FILE" --dataset-revision "$OPSD_DATA_REV" \
  --workdir research_runs/base_geometry_50_answer_v2_full_tokens_paired --method opsd

# 核对计划后执行；默认后台任务，使用status查看真实退出状态。
bash run_base_comparison.sh \
  --model-dir "$OPSD_MODEL_DIR" --model-revision "$OPSD_MODEL_REV" \
  --input "$OPSD_DATA_FILE" --dataset-revision "$OPSD_DATA_REV" \
  --workdir research_runs/base_geometry_50_answer_v2_full_tokens_paired --method opsd --execute

bash run_base_comparison.sh status --workdir research_runs/base_geometry_50_answer_v2_full_tokens_paired --watch
```

纯预览不生成校准回答，不能证明质量达标。真正的Qwen质量门槛只能在服务器实际生成后判定；代码不能保证自由生成的Base在第一次尝试时就达标。高失败率会被拦截并留下可诊断记录，而不会被掩盖。
