# 2026-11-10：只运行 SFT / GRPO 的服务器执行记录

用户明确要求本轮不重训 OPSD。SFT、GRPO 分别从 Qwen3-1.7B-Base 独立训练20步，
先在独立目录各做2步验收，验收成功后再从 Base 启动正式实验。
历史 OPSD 保留为参考；历史没有逐步训练题目 ID 收据，不能宣称三组训练暴露严格配对。

## 输入与环境

- 历史冻结题集实际迁移到 `research-runs/base_opsd_20261028_20steps_q70_reuse_pilot_train2814/shared/probe/`
  （注意此处为连字符 `research-runs`，是只读来源，不是新运行输出目录）。14个收据登记文件校验通过，
  train/validation/heldout/train_probe 为2814/512/50/4。
- Base 与 DeepMath 使用交接锁定的原 revision；缺失公开文件由用户授权重新下载，
  位于 `/root/autodl-tmp/opsd_inputs/`，不修改迁移来的原始目录。
- Base 内容指纹已匹配历史 `ba288b1460aa43317f35b35e1e61cfb7f975ceaabb439c1b6d455d7bef4644ee`。
  原始 Parquet 将使用 PyArrow21按原下载脚本重建，并匹配历史 `identity.input_hash`，匹配后才能复用 probe。
- 虚拟环境 `/root/autodl-tmp/envs/sft_grpo`，复用镜像系统 PyTorch2.8.0+cu128，其余依赖在虚拟环境安装。
  激活脚本 `/root/autodl-tmp/activate_sft_grpo.sh` 不启动训练。
- FlashAttention 使用与历史一致的2.8.1官方预编译wheel，其SHA256匹配历史记录。

## 本轮执行配置

当前配置 `configs/controlled_sft_grpo20_fast_server.json` 继承新对照的训练设置：
20步、batch8、每题8份序列、microbatch1、LoRA r16/alpha32、lr1e-6、seed42、AdamW和constant调度。
SFT是重复8次的离线reference监督，GRPO是8条自主采样及0/1 outcome reward；两组共享固定题序。
监测点0/1/5/10/20，greedy评测上限16384tokens，几何窗口256tokens。

本机数据盘50GiB，按用户此前“先分析、再清理旧权重”的明确偏好使用 `checkpoint_retention=latest`。
与原交接的 `all` 有如下存储差异，不改变训练数据、损失或更新数：

1. 保存checkpoint后同步完成该监测点的全部分析；失败时不清理旧恢复状态。
2. 每个监测点的LoRA在 `<method>/adapter_archive/global_step_N/` 单独存档并校验，不随完整checkpoint轮转删除。
3. 各方法保留最新完整恢复checkpoint；中间合并HF权重完成分析后清理；原始回答和指标数组全部保留。
4. 用户确认优化评测后，两组生成使用FlashAttention2，按冻结历史回答长度确定相同的静态分组，每批最多8题。
   全部题目均保留，结果恢复到原始题目顺序，每题RNG seed不依赖分组顺序；训练微批仍为1。
5. step0用新生成路径重新测量；原始固定回答token及数据划分保持不变。
   生成完成后将注意力恢复为eager，固定表征、全词表分布和adapter往返审计保持原eager路径。

独立执行入口 `--method sft_grpo` 只选择SFT和GRPO；共享准备阶段可能生成未执行的OPSD配置文件，
配置文件存在不代表启动OPSD训练。配对摘要将OPSD标记为用户未请求。

## 运行位置和进度

```text
research_runs/sft_grpo_campaign_20261110/campaign_state.json
research_runs/sft_grpo_acceptance_fast_20261110/
research_runs/sft_grpo20_20261110/
```

启动后的后台编排器为 `/root/autodl-tmp/sft_grpo_setup/run_campaign.py`。
安装、下载、CPU测试和启动控制日志位于 `/root/autodl-tmp/sft_grpo_setup/`。
只读查看进度（Ctrl+C仅退出查看）：

```bash
bash /root/autodl-tmp/sft_grpo_status.sh
bash /root/autodl-tmp/sft_grpo_status.sh --watch
```

每个方法的真实进度见 `<workdir>/<method>/train/metrics.jsonl`、`train/train.log`、
`job_status/step_*.json` 和 `analysis/generation_progress.json`（文件具体位置以实际产物为准）。

## 报告

目标为 `results/20261110_qwen17b_sft_grpo20/report/SFT_GRPO_report.html`，
单组报告为 `SFT_report.html` 与 `GRPO_report.html`；配对统计为 `CONTROLLED_COMPARISON.html`。
报告包含所有已采集训练、分布、表征、曲率和行为指标，保留未定义值及有效样本数。
只有真实训练、监测、往返精度审计和最终数据核验均通过后，编排器才将状态标记为completed。
本文记录执行方案和已完成检查，不能当作20步训练已完成的证明。

已完成的启动检查：环境/CUDA preflight通过；受控训练测试10项、checkpoint保留测试14项、
forward-KL测试7项全部通过；报告测试7项中6项通过、1项因无Node跳过。
2026-11-10 11:26（北京时间）启动原eager验收；后经用户明确确认优化评测，12:26启动新验收。
优化后新增的长度分组、部分批次恢复和注意力恢复测试均通过：批量生成测试10项，加其余检查共48项，47通过、1因无Node跳过。
实际训练/分析完成情况以campaign状态和结果收据为准。

## 评测优化的实测与旧结果保留

原eager验收目录 `research_runs/sft_grpo_acceptance_20261110/` 完整保留，SFT已训练2步，
CE为0.7771783953、0.7103722454；392个LoRA张量均更新。step0/1分析已完成，step2缓存已完成40/54题。
该目录的 `preserved_server_sources/` 保存变更前源码，不修改旧来源收据，不把旧结果混入新验收。

原实现对一批包含16384-token回答的8题耗时约1189秒。新GPU固定工作量benchmark使用3条4096-token前缀，各生成64tokens：
eager为2.610秒/峰值12.407GiB，FlashAttention2为1.935秒/峰值5.141GiB，吞吐约1.35倍。
长度分组另减少长短回答混批的padding，实际端到端收益以新运行记录为准。

真实BF16 Base的固定位置logits比较已观察到非零差异，3条短greedy前缀中2条改变；
因此不能声称新旧生成逐位相同，也不能将内核差异当作训练效果。
两组Base基线均重新生成，并在配对统计前核对一致性；历史OPSD仅作独立参照。
完整数值及时间审计为 `/root/autodl-tmp/sft_grpo_setup/fast_generation_gpu_audit.json`。

环境只对独立的 `scripts.research.retention` 分析子进程启用可扩展显存段；
Ray/vLLM训练进程保持默认分配器以兼容CUDA IPC。设置记录在 `cuda_allocator_profile.json`。

## 2026-11-10：吞吐优化与正式20步调度

用户进一步要求继续调试、最大程度提高速度并使用显存。旧 fast 验收的源码与评测缓存保留在 `research_runs/sft_grpo_acceptance_fast_20261110/`；原 eager 两步真实权重更新及指标也保留。正式目录改为 `research_runs/sft_grpo20_throughput_20261110/`，不修改旧运行的冻结收据。

- 解除“两个方法全量两步评测完成后才准训练20步”的调度阻塞。先前SFT两步CUDA与实际adapter更新已验证；现在先训练SFT20步，保存0/1/5/10/20全部原生恢复状态、分析任务与LoRA，再完成全部分析，分析验证通过后才清理旧恢复状态。GRPO在正式运行逐步检查有限损失、梯度、题目顺序与checkpoint，不另插入两步全量生成验收。
- SFT仍为每步8题×8份参考回答，64条真实监督序列，全局response-token归一化CE、AdamW参数、学习率、seed、题集与20步不变。改为微批64，输出投影与CE共同分块checkpoint，避免完整 `[B,T,151936]` logits及重复的大梯度散射。FP32参数、BF16前后向、激活checkpoint保持。分块1024 token；微批GPU数值舍入可能不同，非逐bit等价。
- 最长冻结训练样本2035 token，实测微批64/CE块1024单次前后向16.58秒，峰值55.33GiB；与微批1旧路径约31秒/整步相比，完整训练实际耗时以新metrics为准。微批32略高吞吐（约3.96序列/秒），64约3.86序列/秒，选择64以充分使用显存、减少累积次数。
- 自主评测使用独立vLLM0.11子进程、BF16、连续批处理64、CUDA Graph解码、greedy、原16384 token上限与EOS，显存比例0.80。每条完成即落盘，保留可恢复raw/cache。子进程退出后释放KV缓存，几何与分布仍由HF eager提取。新Base0与训练后各点使用同一新后端，历史OPSD/eager结果仅为独立参照，生成轨迹可能因数值路径不同而改变。
- GRPO启用按32768 token预算动态微批，保持整步全局有效token归一化；关闭activation CPU offload；rollout vLLM显存比例由0.25提高到0.70，启用CUDA Graph。20步、8题×8采样、奖励与优势定义不变。按实测错误继续纠错，不能仅凭显存占用判定训练吞吐。
- CPU验证新增不同长度/右padding/不同微批损失与梯度一致性、deferred SFT不提前分析或删checkpoint、vLLM EOS/乱序完成缓存，以及GRPO不等长动态微批的梯度一致性。输入权重与数据revision、SHA校验不变。

进度：`bash /root/autodl-tmp/sft_grpo_status.sh --watch`。退出查看不会停止训练；后台supervisor负责SSH独立运行。报告后处理使用新的正式目录，不将未完成结果标为最终结果。

正式SFT20已实跑完成：20次更新计算合计322.46秒，中位16.13秒/步，末步CE为0.6981366；分配显存峰值55.37GiB，缓存保留峰值70.91GiB。0/1/5/10/20原生checkpoint与全部LoRA均已保存。Base vLLM完整54题试跑114.65秒、53519生成tokens，含两条16384token截断；heldout24/50，解析失败9/54。经rows/model/revision/monitor/source与逐条结果哈希校验，复用这次新vLLM Base缓存供两组step0使用；每组另存adoption收据，正式提取再次校验完整缓存身份。历史eager基线没有替代新基线。

SFT Base0完整诊断已生成8298条metric记录、349个带哈希的产物。为优先推进实际训练，调度进一步调整为：SFT20 → SFT0诊断验收 → GRPO20及在线监测 → SFT剩余监测点分析 → 两组审计与完整报告。SFT所有待分析原生权重继续保留；不会在分析前删除。一个可由已校验原生step1+Base重新生成的临时BF16 HF export已清理以释放3.3GiB，原生恢复状态与LoRA归档不受影响；清理收据位于setup目录。

### 显存交接纠错与恢复验证

GRPO首次第1步真实更新已完成（采样37.57秒、策略更新6.37秒、整步60.77秒）；随后评测子引擎因当时仅56.49GiB空闲、却申请63.4GiB而失败。原因是检查点临时GPU快照退出后仍留在训练进程的CUDA缓存中。保留第1步完整恢复状态、优化器、RNG、指标、题目暴露记录及已完成Base诊断。

在FSDP checkpoint manager中，将既有load/save实现体保持原样，外层返回后执行GC、CUDA同步与empty_cache，只释放未使用缓存，不改变权重或损失。原实现体AST逐项核验一致；16项保留/恢复/缓存测试通过。额外执行补丁的原源码、diff、SHA与原因单独记录在setup的`checkpoint_cuda_handoff_fix.json`和`.patch`中（旧provenance列表未包含该checkpoint manager模块，故显式补记，不重写冻结分析收据）。

从GRPO第1步恢复后，其54题评测完成，含3条16384-token长回答，物理显存峰值80875/81920MiB，未再出现启动显存错误；第1步完整分析通过后继续第2步，第2步53.76秒、策略更新6.21秒，有限loss/gradient，SFT与GRPO前两步题目暴露收据完全一致。未追加SFT步数或改变GRPO20步上限。

GRPO旧PyTorch累计峰值可能超过实体显存，系vLLM睡眠/虚拟池重映射计数，不应当作物理使用量。新增5秒一次的`gpu_physical_telemetry.jsonl`记录nvidia-smi物理占用、利用率、功率；最终报告保留原计数并明确解释。对当前真实GRPO checkpoint1的246项图表已执行Chromium离线检查：逐项渲染、搜索、28层heatmap、CSV/PNG/SVG均通过，无页面错误或外部请求；这仅为中途报告验收，最终20步报告仍须完成所有监测点后生成。
