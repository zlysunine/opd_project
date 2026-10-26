# 仓库审计（2026-10-20）

**当前范围已按用户要求修改**：实验一仅研究 Qwen3-1.7B 与同参数特权自教师在 DeepMath-103K difficulty 5–7 上的思考模式一致性，组名 `exp1_opsd_consistency`；§3.2 迁移取消，实验二 SFT/RL/OPSD 表征比较保留。下文原论文 §3.1/§3.2 的核对是历史审计记录，不是当前默认实验分组。

**2026-10-26 更新**：当前实验一改为 LoRA r16/alpha32、lr=1e-6、8 题×8 rollout、500 个 batch updates，直接优化全词表前向 KL（特权教师 || 学生）。下文关于原 OPD reward/PPO 的分析描述保留的历史分支，当前 forward_kl 分支不调用它作为训练目标。原仓库两个 shell 未改。LoRA 导出后恢复真实训练 alpha 并合入模型才评测，避免旧 merger 默认 alpha=0 导致评测基座；完整说明见服务器交接文档。

本次检查本地 `datasets/DeepMath_deduped.parquet` 的 parquet footer：extra_info 仅 question/index，schema 没有 difficulty/level 或 R1 solution，不能证明 level 范围。官方 [DeepMath-103K 数据说明](https://huggingface.co/datasets/zwhe99/DeepMath-103K)确认 difficulty 是连续数值，参考解答为 r1_solution_1/2/3；新入口使用原始字段，闭区间筛选并在缺失时失败。

审计起点 Git：`71a30bab96b998515d54e5de0654bc4a17c1a17a`（更新 OPD 训练代码和配置）。工作区初始 `git status --short` 为空。未覆盖已有文件修改、未执行 checkout/reset/clean、未下载模型、未启动付费资源。唯一适用的子目录规范为 `verl/verl/AGENTS.md`，已读取；新增该目录 Python 文件采用 Apache 2.0 头。

读取了项目 README、两个训练 shell、现有 pilot 和测试、verl requirements/config/actor/reward/trainer、LLaMA-Factory 的 SFT 配置与模板、现有数学评测器。现有 pilot 是此前提交的一部分，保留原入口。

## 论文及来源核对

原论文：[arXiv 2604.13016](https://arxiv.org/html/2604.13016v1)。本地另有 `../references/papers/Li 等 - 2026 - Rethinking On-Policy Distillation of Large Language Models Phenomenology, Mechanism, and Recipe.pdf`；本环境无 PDF 解析器，实际核对采用 arXiv HTML，不声称读取了 PDF 全文。

| 问题 | 学生 | 两个外部教师 / OPSD 的两个 reference 来源 |
|---|---|---|
| §3.1 | Qwen3-1.7B-Base | Qwen3-4B non-thinking；Qwen3-4B-Base-GRPO |
| §3.2 Qwen 分支 | Qwen3-1.7B non-thinking | Qwen3-4B non-thinking；同一 non-thinking 起点的 RL-Math |

OPSD 扩展增加相应的 4B 学生。OPSD 中这些 4B 模型仅生成参考信息，实际 logits 来自当前学生加参考上下文；这是原问题的迁移实验，不能称为严格复现。

模型注册表为 `configs/research/models.json`。以下 ID 的模型卡实际可访问：

- [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base)、[Qwen/Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base)。
- [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)、[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)。模型卡要求 Qwen3 使用 Transformers ≥4.51，non-thinking 显式 `enable_thinking=False`。
- README 的 `lllyx/Qwen3-4B-Base-GRPO` 实际重定向至 [Thinking-Space/Qwen3-4B-Base-GRPO](https://huggingface.co/Thinking-Space/Qwen3-4B-Base-GRPO)。该模型卡确认从 4B-Base 在 processed DAPO 上 GRPO，区别于 non-thinking 起点。
- 仓库仅出现本地名字 `model/Qwen3-4B-Non-Thinking-RL-Math`，未发现可确认精确权重的本地目录、revision 或权威发布对应关系。搜索出现的第三方 Step500/Step1200 不能证明等于论文 checkpoint，未采用。注册表保留空 ID/path/revision，运行明确报错。

模型卡在线可见不等于权重已下载。本工作区未发现 `model/` 或 checkpoint 目录；所有真实模型 path/revision 必须填写。未将 `main` 当作固定 revision。Base tokenizer 的原始模板和本地下载完整性尚待有权重的环境实测；若缺模板，必须提供核实来源的模板，不能自动编造。

## 原有调用链与梯度

```text
on_policy_distillation.sh / grpo.sh
  → verl.trainer.main_ppo (Ray TaskRunner、资源池、reward manager)
  → RayPPOTrainer.fit
  → actor_rollout_wg.generate_sequences
     ActorRolloutRefWorker.rollout_mode → vLLM → trainer_mode
  → actor_rollout_wg.compute_log_prob
     DataParallelPPOActor._forward_micro_batch
  → rm_wg.compute_rm_score
     RewardModelWorker._forward_micro_batch（独立 causal LM，不是分类 reward head）
  → actor_rollout_wg.compute_distillation_reward
  → compute_token_reward_direct_advantage
  → actor_rollout_wg.update_actor → actor.update_policy
     重算带图的 student log-prob → compute_policy_loss_vanilla
     → backward → optimizer.step → scheduler
  → FSDP checkpoint / logger
```

`grpo` 用原有 rule reward 与 group-relative advantage，`log_prob_top_k=0`。未引入新 loss 或表征训练项。

默认 `only_stu/student_p`：取学生完整分布 top-16 的支持集 S，
`w=softmax(log p[S])`，`A=(log q[S]-log p[S])*w`，随后 direct advantage 乘 response mask。

梯度不是对缓存 `log p-log q` 直接 autograd：reward/weights 停梯度；学生重算同一支持集 log-prob，通过原有 clipped/dual-clipped PPO ratio surrogate 更新，K 维求和，response token mean。即使代码注释提到 memory-efficient log-prob 形式，实际代码仍计算 ratio。sampled 路径用 sampled `log q-log p` reward 和二维 PPO surrogate。不能把这个局部 surrogate 宣称为完整序列 KL 的精确梯度（包含未来状态访问分布的项并未单独实现）。

**论文/代码差异**：论文 top-k 描述对 p、q 均在 S 上重归一化；实际 reward 对 w 重归一化，log-ratio 仍是完整分布的原始 log-prob 差。保留行为，将其命名为 repo raw-log-ratio surrogate；离线另算 renormalized-top-k KL。旧 `val-topk/adv_intersection` 也不能替代论文交集内重归一化且额外除 |I| 的 advantage。已有 `utils/opd_metrics.py` 提供正确动态指标，本轮保留并增加 shared mass/空交集率。

本轮仅抽取两段无依赖纯张量代码到 `utils/opsd.py`；回归测试比较原公式的值与梯度。未改 OPD 支持集策略、KL 正则或训练公式。

## 新增 OPSD 链路

`research run train` → 同一 `main_ppo/fit`；配置 `reward_model.enable=False`、`opsd.enable=True`。

1. 学生在原始无特权 `messages` 上 rollout。
2. 原 actor log-prob 获取 p 和学生 top-k。
3. `ActorRolloutRefWorker.compute_opsd_teacher` 从 `extra_info.messages/privilege_text` 重建教师 prompt；在相同 user message 末尾加入固定 reference 字段，empty 保留空字段。
4. 将**原始 response IDs**追加到左侧 padding 的教师 prompt；教师 prompt 独立长度上限，超限报错。使用同一个 actor，`torch.no_grad`；先获取教师自身 top-k/熵，再查询学生 top-k 的 teacher log-prob，因此当前 top-k 实现有两次 teacher forward。
5. 回到原 reward、advantage、actor update。教师 logits 缓存在本 rollout batch 内，全部优化完成后丢弃，下批重新用当前学生参数计算。

策略名 `current_batch_stop_gradient`。版本为 rollout batch 更新前的 `global_step-1`，当前配置每批一次 actor update。没有独立教师权重，没有 frozen-initial 冒充动态教师。其他更新策略直接拒绝。生产支持本仓库 FSDP 文本 actor；Megatron、多模态、异步策略不在已适配范围。

## Token、后端、数据与 checkpoint

- tokenizer 保留原 token→ID、BOS/EOS/PAD。外部 OPD 启动前检查映射和输出 vocab size；不兼容明确失败。
- 外部 OPD `input_tokenizer=None`，教师直接读取学生 prompt IDs，不 decode/re-tokenize response，不单独拼教师 chat prompt。non-thinking 由学生模板 kwargs 控制。
- OPSD 教师与学生使用同一个 tokenizer/template，但教师额外上下文会改变位置与计算，不能据共享权重推断 thinking pattern 必然兼容。
- `pack_responses`：左 pad prompt、右侧固定 response，position IDs 是 attention mask 的 cumsum−1；用 `P-1+t` 的 logits 预测 `response[t]`。EOS 包含第一个 EOS，后续 padding 排除；真实截断末尾仍是有效访问状态。
- 原 actor 支持 remove-padding、Ulysses、FSDP/FSDP2 与混合精度；Ray 管 worker，vLLM 管 rollout。新单卡配方微批 1、梯度累积、activation checkpoint/offload；是否 OOM 未实测。
- checkpoint 为 `global_step_N/actor/model_world_size_*_rank_*.pt` 等分片，含 HF 配置/tokenizer 子目录；评测前用现有 `verl.model_merger` 合并。新配方保存 model/optimizer/extra，原 pilot 的 model-only 不能据此精确恢复。
- 仓库数学数据实际存在：DAPO 原始/processed parquet、DeepMath deduped、OpenThoughts3_opd、AIME24/25、AMC23、MATH-500 等。现有 shell 的部分注释数据路径并不存在，不能当作已核实路径。
- 数学评分复用 `scripts/val/eval/utils.py::grade_answer_verl`，绕开 `grade.py` 的 import-time tokenizer 加载。受控技能用 boxed integer 精确验证。
- SFT 保留 LLaMA-Factory `run_exp`、Qwen3 模板、full SFT；新增 wrapper 仅计数实际非 −100 的 causal labels，不更换训练器。

## 评测及资源差异

README 已指出 verl 0.7 内置 validation 可能低估，建议关闭；新配方 `test_freq=-1`、独立 HF 评测。原论文附录的 validation length=31744，而当前脚本/README 建议与训练 7168 相等；本轮离线 CLI 默认 7168，并在配置/日志显式保存。这是不同评测设置，不能静默称论文完全一致。

已有 logger：console/W&B/SwanLab 等；本轮复用 logger 并附加 JSONL budget/scalars。2026-10-21 按交接要求将根 `on_policy_distillation.sh` 恢复为 HEAD 原版，`grpo.sh` 保持原样。原 shell 的硬编码模型、`ray stop --force` 等行为均保留；新 OPSD 入口独立为 `run_opsd_experiment.sh`，不调用原脚本。重复的 `scripts/legacy/on_policy_distillation.sh` 已在用户确认后按清单删除，原版仍保留在根目录。

顺序入口 `opsd_pipeline.sh` 与 `pipeline.py` 增加本地运行配置、严格长度过滤与配对划分、训练前 Hydra/dataclass 检查、checkpoint 合并和无特权评测。根 `scripts/__init__.py` 修复 vendored `verl/scripts` 遮蔽项目脚本包的问题；`RolloutConfig.seed` 显式声明训练命令所传的 rollout seed，避免 dataclass 拒绝该参数。这些流程及拒绝路径有 CPU 单元测试；完整生产 Hydra/FSDP 配置仍需训练环境实测。

测试机：Python 3.12、Torch 2.3.1、Transformers 4.44.2、NumPy 2.1.1、SciPy 1.14.1、sklearn 1.6.1、Matplotlib 3.10.3、pandas 2.2.3。缺 Ray/PyArrow/OmegaConf/TensorDict；GPU NVML 访问被系统阻止。现有 verl requirements 要求 NumPy<2，与本机也不同。因此本机只验证无分布式依赖的 CPU 科学指标与 tiny HF wiring，不声称跑通完整生产依赖。系统 `/usr/bin/python3` 可用 psutil，现有 pilot 测试用该解释器通过。
