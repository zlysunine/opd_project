# 指标与采集口径

所有概率指标的 p 是学生、q 是教师；都在同一个 response token prefix 上 teacher-force，prompt 和 padding 不计入平均。自然对数，显式 `diagnostic_temperature`，与生成的 sampling temperature 分开。EOS 算第一个 EOS，后续 padding 不计；达到长度上限但未 EOS 的访问状态仍计入，另外报告 truncation_rate。

## 输出分布

令 S_p=TopK(p,k)，S_q=TopK(q,k)，I=S_p∩S_q，m=|I|。

| 名称 | 公式 / 实现 |
|---|---|
| token_topk/k/overlap_ratio | m/k；独立计算两者各自 top-k |
| overlap_advantage | `(1/m) Σ_I p_bar(v)[log q_bar(v)−log p_bar(v)]`，p_bar/q_bar 分别在 I 重归一化 |
| shared_mass_student / teacher | `Σ_I p(v)` / `Σ_I q(v)`，使用完整分布原始概率 |
| entropy/*_full_vocab | `−Σ_vocab p log p` / `−Σ_vocab q log q` |
| gap_absolute | `abs(H(q)−H(p))`，逐位置取绝对值后平均 |
| gap_signed | H(q)−H(p)，附加诊断，不替代绝对差 |
| kl/full_vocab_p_to_q | `Σ_vocab p log(p/q)` |
| kl/full_vocab_q_to_p | `Σ_vocab q log(q/p)` |
| kl/renormalized_student_topk_k_p_to_q | 在 S_p 上分别重归一化 p、q，再算 KL(p_bar||q_bar) |
| kl/sampled_token_estimate_p_to_q | 当前访问 token 的 log p−log q，单个值可负 |
| loss/repo_raw_log_ratio_student_topk_k | `Σ_Sp softmax(log p[Sp])*(log p−log q)`；原仓库 reward 对应数值，**不是**双方重归一化后的 KL |

k 默认 16（论文表 2、原 shell 均一致），支持多个 k。m=0 时 advantage 为 NaN，JSONL 为 null；保留 valid mask、empty_intersection_rate，平均时只从该指标排除；其他 overlap/mass 仍有意义。m=1 时 advantage 恒为 0，不说明原始概率完全一致。

离线 full-vocab 指标对 hidden states 经 chunk LM head 计算，不记录完整 logits。没有拿 top-k 条件熵充当 full-vocab entropy。student/teacher 词表大小或 ID 映射不兼容立即失败。

历史 reverse_kl 配置的训练 loss 保留原 direct advantage + clipped/dual-clipped PPO surrogate：reward 先停止梯度，student 带图重算 log-prob；不是对缓存 sampled log-ratio 做直接 autograd。详见审计。在线 `opd/kl_renormalized_student_topk_p_to_q` 和 `opd/repo_raw_log_ratio_student_topk` 分开命名；极端非有限的非交集概率不能算出可靠 subset KL 时不输出该在线字段，严格离线诊断会明确失败。

`aggregate` 同时提供 token_weighted 与 question_mean。后者先把每题所有有效 response 位置/轨迹平均，再对题平均；前者每个有效 token 权重相同。空交集优势排除后，单独报告有效 token/问题数。训练动态指标默认沿用仓库 token-mean；离线不能将两种口径拼成一条曲线。相对位置分桶同样同时输出两种平均。

## 性能

- avg@16：每题 n=16，c/16，题均值后跨 benchmark macro。多个不同题数 benchmark 不按总题数混在一起加权。n≠16 时不标 avg@16。
- avg_accuracy：每题 c/n，对题、benchmark 同上汇总。
- pass@k：`1−C(n−c,k)/C(n,k)`，用 log1p/expm1 稳定求积，要求 n≥k。n 不足的 requested k 不输出，绝不拿 avg@16 代替。
- pass@1 是 n 次采样估计的单次成功率，不等同于另设 greedy decoding。
- gap recovery：`(after−before)/(teacher−before)`；固定 reference checkpoint 是必填项，分母非正/≤阈值则 null。有特权 teacher 明确保存 teacher_uses_privilege，不能把该比率当纯新能力证明。
- checkpoint 性能匹配只读取 validation 标签；最近目标性能者胜出，tie 按优化 tokens 和 step。返回是否达到 tolerance，不把不匹配的 checkpoint 自动宣称匹配。

## 表征

特征矩阵 X 的行是匹配的问题×轨迹×response-relative 位置，列是模型 hidden 维度。比较前检查问题、sample_index、layer、hook、causal semantics、token_index、bucket、prefix_hash、response_hash；只重新排序，不静默取不同数据的交集。

- activation cosine：逐匹配向量点积除以范数。零向量产生 NaN/空汇总。不同 hidden dimension 明确失败。
- RSA：每条件内部 `R_ij=cos(h_i,h_j)`。输出 NPZ、标签 JSON，以及按技能/状态/答案排序的热图。矩阵上三角 Pearson/Spearman 是**本项目采用的汇总定义**；常数/太少有效样本为 null。
- representation_neighborhood_overlap：排除自身后，`mean_i |N_k^A(i)∩N_k^B(i)|/k`；默认余弦，函数接口支持欧氏。记录 N/k/distance 和随机参考 k/(N−1)。零向量的余弦近邻拒绝定义。与 token_topk 日志分开。
- linear CKA：先在样本维中心化，`||XᵀY||_F²/(||XᵀX||_F ||YᵀY||_F)`。实现等价的样本 Gram 形式，因此可明确用于不同 feature dimension 的结构比较；分母退化为 null。记录 N，普通 CKA，不是 debiased CKA。
- ARI：固定种子的 sklearn KMeans(n_init=10) 与 skill、真实离散 intermediate_state、answer 标签比较。没有中间状态注释时不伪造该列。日志名 `seeded_KMeans_engineering_baseline_not_ADP`；不是 Representation Landscape 的 ADP 复现。
- representation_subspace_overlap：明确指定变化矩阵 Δ_A/Δ_B，在相同 feature 坐标内取等秩**右奇异向量**正交基 U/V，`||UᵀV||_F²/r`。输入行为样本变化，因此右奇异向量才是 feature 子空间。默认不中心化 SVD；centered PCA 须显式开关，秩不足不造数。对符号不敏感，高 overlap 不意味着作用同向或相同。
- 变化向量 cosine：A−baseline_A 与 B−baseline_B 对应位置的 cosine；要求同坐标。比较特权 shift、SFT/RL/OPSD 相对初始化的变化时应选择正确 baseline，程序不猜。
- linear probe：模型冻结，轻量 LogisticRegression，C 仅在 validation 选。按 problem、instance、template 检查 train/validation/test 无交叉，不能随机拆相邻 token。报告 held-out accuracy、validation accuracy、类别数目和多数类基线；仅说明可读出信息。
- effective rank：默认 singular values 归一化后 `exp(entropy)`；函数也支持 squared_singular_values，名字必须注明。它不是 intrinsic dimension。
- privileged-shift mean pairwise cosine：指定变化矩阵的非零行单位化后，两两上三角 cosine 均值，与 effective rank、子空间 overlap 独立命名。

采集 hook 的 residual/LN 标记见实验指南。`predict_current` 是 P−1+t，`after_current` 是 P+t，不能混用。last_prompt_token 保留独立桶。真实中间标签只能放在对应固定 response 的已标注位置；未对随机错误 rollout 贴 oracle 中间状态。

RSA/近邻/CKA 设 N 上限，默认 2048；采集默认离线、层选择、位置选择、每问题 shard 和内容 hash 缓存。跨 1.7B/4B 显式使用 `--cross-scale`，只保留基于匹配样本几何结构的 RSA/CKA/NO，不逐坐标减法/cosine，不跨尺度注入。

表征与增益关系的分析单元必须是 run/task。stage 明确区分训练前、early 和训练后相关性；拟合/验证按 run 隔离。未实现默认收益预测器，不把相关性当因果证据。

## 当前实验一的固定输出

Qwen3-1.7B vs 同参数 Qwen3-1.7B + DeepMath reference（difficulty 闭区间 5–7）。每 rollout batch 保存 overlap_ratio、overlap_token_advantage、student_entropy、teacher_entropy、entropy_gap；概率诊断在更新前计算。`thinking_consistency.csv/png` 汇集这五项，原 JSONL 始终保留空交集 null 和计数。实验二保留，§3.2 迁移分支已取消。

## 当前实验一的训练目标（2026-10-26）

默认 LoRA rank 16 / alpha 32，学习率 1e-6，8 题 × 每题 8 个学生 rollout，500 个 batch updates。

`actor/forward_kl = sum_valid_tokens sum_vocab q(v) * (log q(v) - log p(v)) / valid_tokens`，其中 q 来自当前 actor（包括当前 LoRA）加离线参考，并停止梯度；p 来自无特权学生。实现对完整词表归一化和求和，top-16 只参与 overlap 等诊断。没有将 teacher_p 加权的旧 raw log-ratio 当作前向 KL。

每个 batch 的教师/学生 raw response logits 按 microbatch 计算，KL softmax 按 token 分块重计算；所有 microbatch 累积完才做一次 optimizer step。多 rank 梯度按全局有效 token 总数归一化。教师梯度为零，基座冻结，只有 LoRA 更新。teacher eval、student train，后者保留 HF gradient checkpointing。

原 reward-manager 接口的 rm_scores/advantages 在当前分支为零占位，不参与优化；训练损失看 `actor/forward_kl` / `actor/loss`。五项 `opd/*` 诊断保持原定义和更新前采集时机。预算包含两次诊断教师前向及更新时额外的完整词表教师前向。
