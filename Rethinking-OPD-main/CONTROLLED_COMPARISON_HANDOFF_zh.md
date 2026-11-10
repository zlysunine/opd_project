# 服务器 Codex 交接：OPSD / SFT / GRPO 严格配对对照

> 2026-11-10 更新：SFT/GRPO 各20步已在服务器完成，本轮按用户要求未重训OPSD。最新实测配置、保留策略、结果与交接以 [SFT_GRPO_HANDOFF_20261110_zh.md](SFT_GRPO_HANDOFF_20261110_zh.md) 为准。下文保留2026-11-09三组设计和执行前说明作历史记录。

更新：2026-11-09。先阅读本文件，再查看 `docs/research/OPSD_SECOND_RUN_AUDIT_zh.md`。

本地无 GPU。本次交付的是源码、CPU 验证和服务器执行流程，不包含新 1.7B SFT/GRPO 实验结果。已有两次报告和原始启动脚本保留。不要把旧实验目录改成新对照的输出目录。

原 `grpo.sh`、`on_policy_distillation.sh` 保留。用户于2026-11-09删除旧包装 `opsd_pipeline.sh`、`run_opsd_experiment.sh`，源码记录和测试已同步解除依赖。本轮使用下述新入口；历史单组CLI可直接运行 `python -m scripts.research.pipeline`。

## 1. 问题与审查结论

第二次 OPSD 是 25/50→23/50，4 gain/6 loss，配对变化 -4 个百分点，95% 问题 bootstrap 区间 [-16,+8]，McNemar p=0.75390625。6 道 lost 题最终均成功抽取、未截断。当前无法据此认定稳定能力损害，也未发现历史 KL 方向/梯度符号/因果对齐/教师版本的直接错误。历史六个核心文件的指纹与修改前 Git 源码一致，证据见 `docs/research/controlled_20261109_audit.json`。

必须把“公式与 CPU 路径检查通过”和“真实 1.7B CUDA/FSDP/vLLM 执行正确”区分。后者需要服务器 2-step 验收。

## 2. 新对照实际控制什么

统一配置：`configs/controlled_deepmath20.json`。新一键入口：`run_controlled_comparison.sh`。

| 设置 | 三组共同值 |
|---|---|
| Base | Qwen/Qwen3-1.7B-Base |
| 模型 revision | ea980cb0a6c2ae4b936e82123acc929f1cec04c1 |
| 数据 | zwhe99/DeepMath-103K，difficulty 5–7，reference=r1_solution_1 |
| 数据 revision | 5cf055d1fe3d7a2eb19719ac020211469736ae44 |
| 冻结数据池 | train=2814，validation=512，heldout=50，train_probe=4 |
| 实际训练暴露 | 20×8=160 道题目呈现，从训练池预先固定顺序取出，无二次 shuffle |
| 每个更新 | 8 道题，每题 8 份序列；microbatch=1；一次 optimizer.step |
| 更新总数 | 20；每组从 Base 独立初始化，不串联训练 |
| 可训练参数 | LoRA r16/alpha32，all-linear，dropout=0，bias=none |
| 初始化/训练 seed | 42；保存实际初始 adapter 指纹并跨方法核对 |
| 优化器 | AdamW，lr=1e-6，betas=(0.9,0.999)，eps=1e-8，weight_decay=0.01，clip_grad=1 |
| 调度 | constant，warmup=0；按全 batch 有效回答 token 归一 |
| 精度 | FP32 参数/优化器，CUDA BF16 前向计算；最终分析模型统一 FP32 合并后存 BF16 |
| 训练上限 | prompt=2048，response=16384；OPSD teacher prompt=2048 |
| 采样 | OPSD/GRPO temperature=1，top_p=1，top_k=-1（不截 top-k） |
| GPU | 1 张；v1 明确拒绝多 GPU、其他 microbatch 或多 epoch |
| checkpoint/诊断点 | 0/1/5/10/20；周期 save_every=20，监测点另行保存 |
| 评测 | 同一 heldout/train_probe；greedy，最长16384；答案协议 opsd_answer_v2 |
| 固定表征 | 同一批 frozen response IDs，前256 tokens，prompt_last/response32/response128 |

仅由训练方法决定的差别：

- OPSD：每题自主采样 8 条；优化完整词表 KL(stopgrad(当前 actor+reference) || 无特权学生)。teacher 批内固定，下一批重算。
- SFT：每题同一离线 reference 重复 8 次，真实执行 64 份监督序列，优化 response-only CE；不是 8 条采样回答。重复不增加新信息；不把监督序列称为 rollout。其采样温度无意义。
- GRPO：每题自主采样 8 条，使用同一答案协议的 0/1 outcome reward，组内减均值/除样本标准差；无 reference KL、熵或格式奖励。

SFT 使用 native HF+PEFT trainer；OPSD/GRPO 使用现有 verl/FSDP/vLLM。模型、数据、LoRA、优化器和监测配置统一，但不声称不同训练器的内核和舍入逐位一致。SFT reference 通常比采样的长回答短；相同更新/序列数不等于相同 token、FLOPs 或 wall-time。每组分别记录这些预算。

为了保留权重供审计，本次三组统一 `checkpoint_retention=all`，最终和监测点权重均不自动删除。不要沿用旧的 latest-only 清理策略。建议准备至少约 150 GiB 的空闲运行磁盘，外加原 Base 和数据；实际空间以 checkpoint 大小为准。此数字是规划余量，不是新实验实测。

## 3. 如何严格复用第二次的数据

推荐显式传入旧服务器的 `shared/probe`。GitHub 中只归档了结果与来源，不包含完整原始问题/reference/固定回答，不能仅靠仓库里的50题计数恢复同一划分。

旧服务器路径参考：

```bash
MODEL=/root/autodl-tmp/opsd_inputs/models/Qwen3-1.7B-Base
NATIVE=/root/autodl-tmp/opsd_inputs/datasets/DeepMath-103K/5cf055d1fe3d7a2eb19719ac020211469736ae44/native_train.parquet
OLD_PROBE=/root/autodl-tmp/opd_project/Rethinking-OPD-main/research_runs/base_opsd_20261028_20steps_q70_reuse_pilot_train2814/shared/probe
```

路径必须在当前服务器真实存在。采用时会验证所有旧文件 SHA256、Base 权重指纹、数据文件哈希、模型/数据 revision、答案协议、长度设置及划分人数；复制固定 ID/文本/token，不修改旧目录。新目录记录 adoption receipt 和当前代码来源，诊断用新代码重新运行。

`--reuse-probe` 传错路径会报错，不会回退重新抽样。没有历史原始 probe 时，可以不传该参数，三组会共用一份新冻结划分，但必须把它标为新的配对实验，不能称为与第二次题集完全相同。

历史训练没有每步题目 ID receipt，无法证明旧 OPSD 与新 SFT/GRPO 看到相同的160题。**因此主比较应重跑新 OPSD、SFT、GRPO 三组。** 新三组共用 `train_schedule.jsonl`，每步实际 ID 会与 `question_schedule.json` 比对。

## 4. 环境和本地验证

复用上次成功的服务器环境。归档环境为 PyTorch 2.8.0+cu128、Transformers 4.56.1、PEFT 0.17.1、vLLM 0.11.0、FlashAttention 2.8.1、Ray 2.59.0、PyArrow 21.0.0。不要为了跑检查盲目升级 CUDA/PyTorch；先保存 `python -m pip freeze` 和 `nvidia-smi`。

```bash
cd /root/autodl-tmp/opd_project/Rethinking-OPD-main
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PWD/verl${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
python - <<'PY'
from scripts.research.preflight import check_environment
import json
r = check_environment(1)
print(json.dumps(r, ensure_ascii=False, indent=2))
assert r['ok'], r['errors']
PY
```

新 SFT 无需 LLaMA-Factory，使用同环境中的 Transformers/PEFT。纯 CPU 检查：

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests/research -p 'test_controlled_comparison.py' -v
OMP_NUM_THREADS=2 python -m unittest discover -s tests/research -p 'test_forward_kl.py' -v
bash -n run_controlled_comparison.sh
```

开发端在隔离的 `/tmp/opd-controlled-venv` 运行，不修改日常 Python。版本/最终检查结果见本文件最后的验证记录。CPU tiny 模型验证不等于真实 Qwen3-1.7B CUDA 验收。

## 5. 先预览，再执行三组各2步验收

以下 `--execute` 省略时仅生成配置/展示计划，不训练、不冻结数据。默认正式预算20步，默认三组顺序为 OPSD→SFT→GRPO。

```bash
bash run_controlled_comparison.sh \
  --model-dir "$MODEL" \
  --model-revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
  --input "$NATIVE" \
  --dataset-revision 5cf055d1fe3d7a2eb19719ac020211469736ae44 \
  --reuse-probe "$OLD_PROBE" \
  --workdir research_runs/controlled_acceptance_2step \
  --steps 2 --monitor-steps 0,1,2
```

确认后使用相同命令加 `--execute`。这会启动可脱离 SSH 的后台任务；`--foreground` 可用于调试。

```bash
bash run_controlled_comparison.sh status --workdir research_runs/controlled_acceptance_2step
bash run_controlled_comparison.sh logs --workdir research_runs/controlled_acceptance_2step --method opsd
bash run_controlled_comparison.sh logs --workdir research_runs/controlled_acceptance_2step --method sft
bash run_controlled_comparison.sh logs --workdir research_runs/controlled_acceptance_2step --method grpo
```

`logs` 的显示以实际进入的方法为准；尚未开始的方法可能没有日志。`stop` 仅在用户要求停止时执行。

验收必须检查：

1. `design_manifest.json` 的 shared hashes；三份 `training_config.json` 的公共字段一致。
2. 三组 `initial_adapter.json` 指纹一致、B=0；不存在基座意外可训练参数。
3. 三组 `train/exposures/step_*.json` 完全匹配共享 schedule，实际两次 optimizer update。
4. SFT 有有限 CE/梯度且 checkpoint adapter 改变；OPSD 有有限 forward KL；GRPO 有组内 reward/advantage 诊断。全组0或全组1的优势为0是算法性质，记录比例，不伪造监督信号。
5. 三组 step0 同题正确性一致，后续评测 ID、固定 token IDs、长度与判分协议一致。
6. 结果中同时存在 student→initial、current student→privileged context 的分布与表征诊断；SFT/GRPO 的“教师”是只用于诊断的有特权条件，不是训练 teacher。
7. `report/OPSD_report.html` 能离线打开，`report/CONTROLLED_COMPARISON_zh.md` 有三组配对表，未完成的方法明确标记，不补假结果。

## 6. 零更新和训练后 adapter 的往返审计

2-step 完成后执行下例。它在固定输入上比较 Base/active adapter/序列化 BF16 合并模型，输出真实差异，不自动把非零舍入误差判成训练效果。

```bash
ACCEPT=research_runs/controlled_acceptance_2step
python -m scripts.research.audit_adapter_roundtrip \
  --base "$MODEL" \
  --adapter "$ACCEPT/opsd/train/checkpoints/global_step_0/actor/lora_adapter" \
  --input "$ACCEPT/shared/probe/fixed_pool.jsonl" \
  --output "$ACCEPT/zero_update_roundtrip.json" --require-zero

python -m scripts.research.audit_adapter_roundtrip \
  --base "$MODEL" \
  --adapter "$ACCEPT/opsd/train/checkpoints/global_step_2/actor/lora_adapter" \
  --merged "$ACCEPT/opsd/hf/solution/global_step_2" \
  --input "$ACCEPT/shared/probe/fixed_pool.jsonl" \
  --output "$ACCEPT/trained_roundtrip.json"
```

对 SFT/GRPO 也运行相同审计，替换方法目录。审计每题最多3个预测位置、前128个固定回答 token，不重跑长回答生成。`--device cpu` 仅适合小模型/专门的CPU审计，默认 CUDA。

初始 B=0 的 active 与同路径 Base 应完全相同；脚本会拒绝错误的零更新 adapter。合并后 BF16 相对 active 路径的 KL、最大 logits 差和 argmax 一致率用于估计数值影响。该脚本使用 HF eager，不代替 FSDP/vLLM 运行路径的检查。还要保留训练日志的 rollout log-prob 对照指标。

## 7. 正式一键运行20步

验收通过后使用全新的 workdir，从 Base 重建三组；不从验收的 step2 继续。

```bash
bash run_controlled_comparison.sh \
  --model-dir "$MODEL" \
  --model-revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
  --input "$NATIVE" \
  --dataset-revision 5cf055d1fe3d7a2eb19719ac020211469736ae44 \
  --reuse-probe "$OLD_PROBE" \
  --workdir research_runs/controlled_deepmath20_20261109 \
  --execute
```

流程：校验环境→验证/复制同一冻结 probe→固定训练题序→生成3组配置→OPSD训练/分析→SFT训练/分析→GRPO训练/分析→完整HTML+配对统计。

同一 workdir 的同一命令可用于恢复，配置变化会拒绝；SFT 恢复 optimizer/RNG，OPSD/GRPO 使用原有 verl 恢复。已完成的最终 step 不额外训练 step21。

准备数据之后不要修改训练/分析源码。源码指纹发生变化时，使用新 workdir；不要编辑完整性收据来绕过检查。

如需单独重跑阶段：

```bash
bash run_controlled_comparison.sh train --workdir research_runs/controlled_deepmath20_20261109 --method sft --execute
bash run_controlled_comparison.sh analyze --workdir research_runs/controlled_deepmath20_20261109 --method all
bash run_controlled_comparison.sh report --workdir research_runs/controlled_deepmath20_20261109
```

数据准备可单独执行 `prepare --workdir ...`，前提是该目录已 configure。`--method sft/grpo/opsd` 用于选择单组，`both` 仍指旧的 OPSD+GRPO，`all` 才是三组。

## 8. 结果和保留内容

```text
<workdir>/
  config.json, models.json, design_manifest.json
  shared/probe/                         # 原训练池/validation/固定题/回答/来源收据
    train_schedule.jsonl, question_schedule.json
  {opsd,sft,grpo}/
    training_config.json, analysis_config.json, initial_adapter.json
    train/metrics.jsonl, train.log, resolved_config.json
    train/exposures/step_*.json
    train/checkpoints/global_step_*/    # 保留所有监测点及最终权重
    hf/solution/global_step_*/          # 统一分析用 BF16 合并模型
    analysis/steps/step_*/              # 原始回答、anchors、RSM及分布/曲率表
  report/OPSD_report.html               # 所有已采集指标的离线完整HTML
  report/CONTROLLED_COMPARISON.html     # 三方法正确率/抽取曲线与配对摘要，静态单文件
  report/controlled_summary.json        # 配对变化/方法间差异和预算
  report/CONTROLLED_COMPARISON_zh.md
```

指标：全词表熵/绝对熵差、双向KL、相对初始KL、top16 token overlap/OTA；RSA/linear CKA/k16题目近邻/RSM RMSE；固定文本逐层曲率；准确率/抽取失败/格式/截断；实际题目暴露/优化token/rollout token/teacher前向token/时间/显存。

GRPO 另有 `grpo/zero_variance_group_fraction`、all-zero/all-one group fraction 与 reward mean。SFT 另有 CE、supervised sequences、distinct questions。其 rollout token 数记0，不能用 reference token 冒充自主生成预算。

50题/4题的小样本局限保留：train_probe 的 k16 未定义；RSA 是题间 RSM 的 Spearman，不是逐题向量余弦平均。解释报告时保留 fixed/live 和普通/特权评测标签。

请保留原始回答和数值数组，以便分析 lost/gained 题和复算表征置信区间。不要只同步 HTML 然后删除全部底层数据。

## 9. 文件入口

| 文件 | 职责 |
|---|---|
| `run_controlled_comparison.sh` | 三组一键/后台/阶段执行 |
| `configs/controlled_deepmath20.json` | 所有公共实验设置 |
| `scripts/research/compare.py` | 三组配置、训练、checkpoint分析和报告调度 |
| `scripts/research/controlled_data.py` | 历史probe审计采用、固定题序、实际暴露检查、GRPO组诊断 |
| `scripts/research/controlled_sft.py` | LoRA response-only SFT、全batch归一、保存恢复 |
| `scripts/research/audit_adapter_roundtrip.py` | adapter/合并/精度审计 |
| `scripts/research/controlled_report.py` | 三组同题变化及方法间配对统计 |
| `tests/research/test_controlled_comparison.py` | 不依赖GPU的真实小模型与生产算法检查 |

本任务未启动 GPU 训练，也未自动 push；服务器应先确认上述源码已完整同步。本地新增与修改源码的校验清单为 `CONTROLLED_COMPARISON_MANIFEST.json`，不同于历史 `SERVER_HANDOFF_MANIFEST.json`，不要覆盖历史清单。

同步后先在仓库根目录执行源码检查（纯标准库，不需要GPU）：

```bash
python - <<'PY'
from pathlib import Path
import hashlib, json
manifest = json.loads(Path('CONTROLLED_COMPARISON_MANIFEST.json').read_text())
for name, expected in manifest['files'].items():
    path = Path(name)
    assert path.is_file(), f'missing: {name}'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], f'changed: {name}'
print('controlled source snapshot verified:', len(manifest['files']), 'files')
PY
```

## 10. 开发端验证记录

2026-11-09 的实际 CPU 结果：研究测试 164/164、在线动态指标 11/11，共175项通过；其中新对照测试10项。研究测试23.631秒，在线指标0.056秒。完整研究测试含本地已有的报告导出器测试，服务器按实际同步的测试集合核对计数。

```bash
OMP_NUM_THREADS=2 PYTHONPATH=.:verl /tmp/opd-controlled-venv/bin/python \
  -m unittest discover -s tests/research -p 'test_*.py' -v
OMP_NUM_THREADS=2 /tmp/opd-controlled-venv/bin/python \
  -m unittest discover -s verl/tests/utils -p 'test_opd_dynamic_metrics_on_cpu.py' -v
```

还完成了新 OPSD/GRPO 完整 Hydra 配置组合、真实 shell 无GPU预览、Python编译、shell语法及 `git diff --check`。没有在本地启动Ray/vLLM训练。

开发端环境：Python3.12、Torch2.3.1+cu121（CUDA不可用）、Transformers4.56.1、PEFT0.17.1、PyArrow21.0.0、Hydra1.3.2。服务器 GPU 环境继续使用第4节的已验证版本，两者不混称。

机器可读记录：`docs/research/controlled_20261109_validation.json`。本地日志与预览在 `research_runs/controlled_cpu_validation/`（该目录被gitignore，不依赖其存在才能执行新实验）。

### 旧入口退役后的上传前复核

2026-11-09保留用户删除两个旧shell的决定，源码记录、当前文档和测试已解除依赖。复测研究164项、在线11项全部通过（32.124秒/0.019秒）。新源码manifest已刷新，历史 `SERVER_HANDOFF_MANIFEST.json` 保留当时快照。
