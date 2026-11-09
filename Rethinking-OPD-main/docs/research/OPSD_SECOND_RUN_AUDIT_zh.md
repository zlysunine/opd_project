# 第二次 OPSD 结果与实现审查（2026-11-09）

当前证据不支持“已经证明 OPSD 导致能力下降”，也没有发现足以解释该结果的 KL 反号、教师版本或回答错位错误。此结论来自归档配置、源码指纹、行为表和 CPU 回归；不代表已经在本地重新运行 1.7B GPU 训练。

## 结果到底下降了多少

第二次 heldout 是固定的 50 题，step0 正确 25 题，step20 正确 23 题。4 题错变对，6 题对变错，净少 2 题。

- 同题配对准确率差：-4 个百分点。
- 问题 bootstrap 2000 次、seed=20261028：95% 区间 [-16,+8] 个百分点。
- 精确 McNemar 双侧 p=0.75390625。
- 最终抽取失败 7/50，初始 8/50；下降不是由抽取失败总量增加解释的。
- **丢失的 6 道正确题，最终都为 `parse_status=extracted`、未截断。** 归档没有它们的原始回答文本，所以不能进一步判断具体数学错误、答案等价判定或推理失败原因。

逐题 ID、状态和复算统计保存在 `controlled_20261109_audit.json`。这些区间是这一题目样本的不确定性，不能代替多训练 seed 的实验。该 50 题曾用于质量验收，也不能称为完全未触及的外部测试集。

## 审查的源码是否对应历史实验

归档 `results/20261028_qwen17b_opsd20/provenance/opsd/train/provenance.json` 记录训练基点为 `032d647e7b61e16e4b18dc7cb0cb7918f7dd092e`，服务器另有工作区修改。

本次修改前 HEAD 为 `929ac85961333fa6ea419422de09483ef4d93128`。逐一计算 Git 中以下文件的 SHA256，与历史训练 provenance 一致：

1. `verl/verl/utils/opsd.py`
2. `verl/verl/workers/actor/dp_actor.py`
3. `verl/verl/workers/fsdp_workers.py`
4. `verl/verl/trainer/ppo/ray_trainer.py`
5. `scripts/research/lora.py`
6. `scripts/research/answer_protocol.py`

因此本次对原 OPSD 核心路径的审查有历史源码指纹支持。新对照所加的 GRPO、SFT、诊断和调度逻辑属于后续开发，不反写历史结果。

## 核心路径的检查结论

| 检查点 | 代码位置 | 结论 |
|---|---|---|
| KL 方向和符号 | `verl/verl/utils/opsd.py::forward_kl_sum` | 实际最小化全词表 KL(stopgrad(q) \|\| p)，q 是特权当前 actor，p 是无特权学生；不是把负 KL 当需要最小化的损失 |
| 学生/教师因果位置 | `opsd.pack_responses`、`dp_actor._forward_micro_batch` | P-1+t 的 logits 预测回答 token t；教师拼接原样的学生 response IDs |
| 教师更新 | `dp_actor._update_opsd_forward_kl` | teacher eval/no_grad；所有 microbatch 反向完成之后才 optimizer.step，批内目标参数版本固定，下一批随学生更新 |
| LoRA | `fsdp_workers`、`lora.finalize_lora_merge` | 训练仅 LoRA；初始 B=0；导出恢复真实 alpha=32 后合并，未发现 alpha=0 导致评测 Base 或重复加载 adapter 的路径 |
| 有效位置 | `response_mask`、`forward_kl_sum` | prompt 和 padding 不计损失；EOS/截断前有效 token 按 mask 参与 |
| 累积归一化 | `dp_actor._update_opsd_forward_kl` | 按全 batch 有效 token 总数归一；CPU 实测 microbatch=1/4 更新一致 |
| 奖励与损失 | `ray_trainer` | 前向 KL 分支直接更新 actor；verifier 只供质量/日志，兼容 rm_scores=0 和 advantages=0 不意味着训练没有梯度 |
| 编码/判分 | `answer_protocol.py`、`data.verify`、`analysis.evaluate` | 使用同一抽取与等价判分协议；不能仅凭标量证明所有数学等价答案都判对 |

CPU 检查包括真实生产方法的数学/梯度、真实 tiny Qwen3+PEFT 的 SFT 更新/恢复/导出，以及小模型的表征提取。vLLM LoRA 同步、真实长序列显存和 1.7B 的实际训练内核仍需服务器验收。

## 发现并处理的对照风险

这些风险会破坏新实验的公平性；它们不能直接解释历史 OPSD 的 2 题净变化。

1. 旧通用 SFT launcher 是 `finetuning_type=full` 和 Qwen3 chat template，与当前 Base/plain/LoRA 不一致。新对照使用独立的 `controlled_sft.py`，固定原始问题提示，reference 仅作为回答 targets，response-only CE，LoRA r16/alpha32。
2. 单有同 seed 不能证明不同 trainer 看到同一步的同一道题。现在共享 `train_schedule.jsonl`，取消二次 shuffle，并在三组每一步核对实际 problem_id、写 exposure receipt。
3. 原通用 GRPO 的 microbatch token-mean 再按序列数加权，对不等长回答不等于全 batch token-mean。仅在新对照开启 `research_global_token_mean`，按 microbatch 有效 token 数加权，并先汇总已加权的 microbatch loss 再记录，避免日志额外缩小。生产 updater 的 CPU 测试已覆盖梯度与记录值。历史 OPSD 使用自己的全局 token 归一分支，不受该问题影响。
4. 旧 GRPO rule reward 没有接收生成是否被截断，而 standalone 评测把截断计错。新 GRPO scorer 接收 `_research_truncated` 并按相同规则判分，不添加格式奖励、参考 KL 或熵奖励。
5. 若后训练质量门槛触发就停止分析，会遗漏表现差的方法。新对照仅把 70% 抽取成功率作为训练前基线门槛，训练后所有失败样本保留并计入准确率分母。
6. 原报告没有合并/精度底噪对照。新增 `audit_adapter_roundtrip.py`，用固定输入测 Base、active adapter、实际 BF16 合并产物的 logits 差异。它不声称验证 FSDP/vLLM 同步。

## 为什么代码正确也可能出现这个结果

当前优化目标是匹配“同一个 Base 模型看到 reference 后”的条件分布，并没有直接最大化数学判分正确率。reference 最终答案经过验证，也不代表 Base 模型在特权上下文中形成的整段 logits 更可靠。特权教师自身的自主生成正确率没有在历史实验中测量，不能把它预设为能力更强的教师。

forward KL 的平均降低可以与 greedy 最终答案的变差同时发生。greedy 对开头少量 token 的变化敏感，会沿另一条长轨迹生成；自教师又与学生共享参数，因此目标本身也在移动。这里是用户指定的全词表前向 KL 实验，不等同于原始 OPD 的 sampled-token/反向 KL surrogate 设置。

仅 20 次、lr=1e-6 的 LoRA 更新，当前结果同时兼容小幅学习、采样/题目样本波动和合并/数值影响。需要对照和重复实验来区分，不能从 50%→46% 直接判断 KL 代码有错误，也不能反过来排除运行时错误。

## 服务器需要完成的证据

- 用新入口先跑三组各 2 步，检查共用数据/题目顺序/初始 LoRA 指纹，确认 loss/梯度和 checkpoint 实际更新。
- 对初始和非零 adapter 分别运行 roundtrip 审计，保留原始回答、anchor、RSM 数值数组和 adapter。
- 三组各从 Base 独立重跑 20 步。历史运行没有逐步 question ID 日志，不能声称它与新对照有完全相同的训练暴露顺序。
- 先看同题准确率变化及置信区间，再看固定前缀分布、表征/曲率和实际 token/时间预算；后续扩展多 seed 和独立测试集。

具体执行命令见仓库根目录 `CONTROLLED_COMPARISON_HANDOFF_zh.md`。
