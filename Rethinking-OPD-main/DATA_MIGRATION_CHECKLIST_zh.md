# SFT / GRPO / OPSD 对照实验：旧实例数据迁移清单

整理日期：2026-11-10。依据 `CONTROLLED_COMPARISON_HANDOFF_zh.md`、
`scripts/research/controlled_data.py::adopt_probe` 和 2026-10-28 正式实验归档。

目标：为新的三组配对实验恢复输入和环境。本清单不启动训练。
当前只能检查新实例；下列旧实例路径来自交接和归档，实际旧文件是否仍在、是否有新增文件，需要在旧实例确认。

## 1. 在迁移界面优先选择这些路径

以下路径全部位于旧实例 `/root/autodl-tmp/` 内。源实例选旧实例，目标实例选当前实例；
迁移后保留同样的绝对路径。复制目录时包含子目录和隐藏内容，避免多套一层同名目录。

| 勾选 | 优先级 | 源实例中的具体文件或目录（绝对路径） | 目标实例位置 | 范围 |
|---|---|---|---|---|
| [ ] | 必须，最优先 | `/root/autodl-tmp/opd_project/Rethinking-OPD-main/research_runs/base_opsd_20261028_20steps_q70_reuse_pilot_train2814/shared/probe/` | 与源路径相同 | **整个目录，含所有子文件** |
| [ ] | 训练必需，可重新下载但优先原样迁移 | `/root/autodl-tmp/opsd_inputs/models/Qwen3-1.7B-Base/` | 与源路径相同 | **整个模型目录，含实际权重和 tokenizer** |
| [ ] | 训练必需，可重新取得但需核对字节哈希 | `/root/autodl-tmp/opsd_inputs/datasets/DeepMath-103K/5cf055d1fe3d7a2eb19719ac020211469736ae44/native_train.parquet` | 与源路径相同 | 原始 Parquet 文件 |
| [ ] | 建议，复用旧 Python 环境 | `/root/autodl-tmp/envs/opsd/` | 与源路径相同 | 整个虚拟环境；镜像若已有可用等价环境，可不迁移 |
| [ ] | 建议，与环境一起迁移 | `/root/autodl-tmp/activate_opsd.sh` | 与源路径相同 | 环境激活脚本 |

新实例截至整理时，上述五个路径均不存在；这只表示文档所列位置缺失，不代表检查了旧实例。
不需要迁移整个 `opd_project/`，当前实例已经有新对照源码。

## 2. 必须完整保留的历史 probe 目录

目录全名：

```text
/root/autodl-tmp/opd_project/Rethinking-OPD-main/research_runs/base_opsd_20261028_20steps_q70_reuse_pilot_train2814/shared/probe/
```

该目录不能通过重新下载公开数据集等价恢复；里面保存了上次的实际划分、参考解答、冻结回答 token 和来源收据。
新入口会逐个验证 `complete.json` 的 `files` 字段登记的全部文件，**该字段是旧目录实际依赖清单的最终依据**。
即使其中某个文件看似只是日志，只要已登记，也要一起迁移，不要手工修改收据来绕过校验。

需重点核对的文件如下。前十项由当前采用路径直接需要或历史归档明确记录；后五项由筛选／缓存路径保存，
以旧目录 `complete.json` 的实际登记为准。不要因为清单没有列出其他文件而将其排除。

| 勾选 | 文件名（相对上述 probe 目录） | 用途 |
|---|---|---|
| [ ] | `complete.json` | 完成收据；每个文件的 SHA256、Base 和数据文件身份 |
| [ ] | `manifest.json` | 数据版本、模型版本、原配置、划分数量及来源 |
| [ ] | `train.jsonl` | 原正式训练候选池，**2814 行** |
| [ ] | `validation.jsonl` | 验证集，**512 行** |
| [ ] | `heldout_probe.jsonl` | 历史评测题，**50 行** |
| [ ] | `train_probe.jsonl` | 训练内监测题，**4 行** |
| [ ] | `fixed_pool.jsonl` | 冻结回答及 prompt/response token IDs、anchor 信息 |
| [ ] | `probe_manifest.jsonl` | 冻结题目和 token 哈希清单 |
| [ ] | `baseline_live_rollouts.jsonl` | 已保存的初始模型自主生成回答 |
| [ ] | `preparation_reuse.json` | 正式运行的历史准备复用记录 |
| [ ] | `eligible.jsonl` | 筛选候选数据 |
| [ ] | `eligible.jsonl.manifest.json` | 候选数据来源说明 |
| [ ] | `eligibility.jsonl` | 每题参考答案和长度资格检查 |
| [ ] | `answer_encoding_quarantine.jsonl` | 答案编码异常记录，可能是空文件 |
| [ ] | `duplicates.jsonl` | 去重记录，可能是空文件 |

**不要误选旧验收目录** `base_opsd_20261028_acceptance_2step_q70_n50/shared/probe/`，
该验收目录的训练池只有16题，不能直接替代新对照要求的2814题正式训练池。
更早的 `base_opsd_200_train3600_parallel` 也不是本次指定的历史来源。

新对照的 `train_schedule.jsonl` 和 `question_schedule.json` 将在新的实验目录生成，
无需到旧目录寻找这两个新文件，也不要预先写入历史 probe。

### 已归档、可用于核对的部分历史哈希

这些值来自仓库的 `results/20261028_qwen17b_opsd20/provenance/source_files.json`，不是本轮从旧实例重新测得。

| 文件 | SHA256 |
|---|---|
| `complete.json` | `7d59891c34774fb33b22233afbde30ba50ff8e52245ab4bb59428ae3db130c8d` |
| `manifest.json` | `2d1daf1275ec301a264fe3775e8dc099be59e19d10606b9f8aae0e98e9655e8c` |
| `train.jsonl` | `d818924af8be184161f2cd00e6ec02e857216eedb5d2346c2a699b51863d2924` |
| `validation.jsonl` | `b8818d4419cd0270bfd7c38374ad784f708a6d11c81cdba7a3142dca9ea9cbb4` |
| `heldout_probe.jsonl` | `c7a156761f962aa192ddadb7becf84f41eb5bae44e092529f473789ee6ee2d10` |
| `train_probe.jsonl` | `7eded9e538f815dd3ec0109a368acc61de02985f72e47c81424794fd3318befd` |

## 3. Base 模型目录中要包含什么

源／目标目录：

```text
/root/autodl-tmp/opsd_inputs/models/Qwen3-1.7B-Base/
```

模型必须是 `Qwen/Qwen3-1.7B-Base`，revision：

```text
ea980cb0a6c2ae4b936e82123acc929f1cec04c1
```

由于新实例没有旧模型目录，不能在这里声称已经看到其实际文件名。迁移时选择整个目录，
按下面的加载依赖检查，不要只复制权重文件：

- [ ] `config.json`。
- [ ] 全部权重：如果是单文件，通常为 `model.safetensors`；如果是分片，则包含
  `model.safetensors.index.json` 和它的 `weight_map` 引用的全部分片。以原目录和索引为准。
- [ ] `tokenizer.json`、`tokenizer_config.json`，以及原目录中的 `vocab.json`、`merges.txt` 等 tokenizer 资源。
- [ ] 原目录中存在的 `generation_config.json`、`special_tokens_map.json`、附加词表或模板等文件；
  这里列出的可选文件不要求在原模型不存在时自行创建。
- [ ] 如果文件是指向 Hugging Face 缓存的符号链接，要复制它们指向的真实文件，
  或一并保留对应缓存目标，不能留下悬空链接。

历史 Base 的内容指纹为：

```text
ba288b1460aa43317f35b35e1e61cfb7f975ceaabb439c1b6d455d7bef4644ee
```

该值是项目 `checkpoint_identity()` 的目录内容指纹，**不是某一份权重文件的 SHA256**。
代码将顶层 JSON、权重和 tokenizer 相关文件一起计入；重新下载同一 revision 后也必须核验目录身份，
不能仅凭模型名称一致就认定与历史 Base 完全相同。

## 4. 原始数据集具体文件

只需原来使用的这一个文件即可作为 `--input`：

```text
/root/autodl-tmp/opsd_inputs/datasets/DeepMath-103K/5cf055d1fe3d7a2eb19719ac020211469736ae44/native_train.parquet
```

来源 `zwhe99/DeepMath-103K`，revision 为 `5cf055d1fe3d7a2eb19719ac020211469736ae44`。
优先直接复制旧文件；新入口会把其 SHA256 与历史 probe 的 `complete.json → identity.input_hash` 比对。
重新导出 Parquet 即使行内容相同，文件字节也可能不同，所以不能未经核验就替代原文件。
迁移整个带 revision 的父目录也可以，便于保留当时的下载／来源记录。

## 5. Python 环境与安装材料

如要复用原环境，建议同时迁移：

```text
/root/autodl-tmp/envs/opsd/
/root/autodl-tmp/activate_opsd.sh
```

镜像可能已经保留部分系统环境，但放在数据盘的虚拟环境不会因此自动出现在新数据盘。
复制虚拟环境后仍需检查 Python 链接、共享库、CUDA/Torch 匹配及 editable 项目路径，
不能把“目录已复制”当作“环境已验证”。

旧激活脚本若引用以下目录，建议一并迁移，或由后续环境恢复步骤核对后重建：

```text
/root/autodl-tmp/opd_project/Rethinking-OPD-main/research_runs/server_environment_20261026T225534Z/
```

其中优先保留 `constraints.txt` 和 `wheels/`。归档明确记录过的 FlashAttention wheel 为：

```text
wheels/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

该 wheel 对应 Python 3.12、Torch 2.8、CUDA 12 和相应 C++ ABI；换环境时不能不检查兼容性就安装。
安装依赖时暂勿照抄旧 `pip-freeze.txt` 的本地文件 URL，先核对 wheel 是否在新实例。

已在当前代码仓库内的环境记录可直接使用，无需另从旧数据盘找回：

```text
results/20261028_qwen17b_opsd20/provenance/pip-freeze.txt
results/20261028_qwen17b_opsd20/provenance/environment_versions.json
```

## 6. 可选历史备份，不是新三组从 Base 训练的必需输入

以下相对路径均位于旧正式实验目录
`/root/autodl-tmp/opd_project/Rethinking-OPD-main/research_runs/base_opsd_20261028_20steps_q70_reuse_pilot_train2814/`。
如果旧实例将被释放或删除，应在其他持久存储中备份这些原始结果；当前磁盘有限，不必全部放进训练盘。

| 勾选 | 相对路径 | 目的 |
|---|---|---|
| [ ] | `opsd/train/checkpoints/global_step_20/actor/lora_adapter/` | 保存历史训练得到的 LoRA，通常比完整训练 checkpoint 小 |
| [ ] | `opsd/train/checkpoints/global_step_20/` | 完整历史参数、优化器、RNG 等，供历史恢复和审计 |
| [ ] | `opsd/analysis/steps/` | 原始生成回答、anchor、RSM、曲率和分布等；复算 lost/gained 分析需要底层数据 |
| [ ] | `opsd/train/metrics.jsonl`、`opsd/train/train.log` | 原始训练指标与日志 |
| [ ] | `config.json`、`models.json`、`opsd/training_config.json`、`opsd/analysis_config.json` | 当时的配置与来源 |

代码仓库中的历史 HTML/标量交接包不能代替所有原始回答、权重和数值数组。
对新实验来说，不需要从这些历史权重继续训练，三组均从 Base 独立初始化。
Hugging Face 下载缓存、Ray 临时文件、Triton 缓存、旧 `outputs/` 等通常不必迁移；
前提是所选模型和环境已经包含实际文件，没有依赖这些缓存的悬空链接。

## 7. 迁移完成后的验收清单

- [ ] 新实例上述必需路径存在，目录未出现 `probe/probe/` 或模型目录再套一层的情况。
- [ ] 历史 `probe/complete.json` 自身与归档哈希一致，且其 `files` 中的全部文件均存在、SHA256 一致。
- [ ] train / validation / heldout_probe / train_probe 分别为 2814 / 512 / 50 / 4 行。
- [ ] `probe_manifest.jsonl` 的哈希与 `complete.json` 的 `probe_manifest_hash` 一致。
- [ ] Base 内容指纹与历史 `identity.checkpoint_hash` 一致；原始 Parquet 哈希与 `identity.input_hash` 一致。
- [ ] 模型权重分片和 tokenizer 齐全，没有悬空符号链接。
- [ ] 若复用 Python 环境，激活脚本及环境检查通过；当前新对照源码保持不被旧代码覆盖。
- [ ] 保留历史目录作为只读来源，新三组使用新的 workdir，不修改旧来源收据。

2026-11-10 新实例只读检查显示训练盘空闲约49.7 GiB。
这是当前可用空间，不是上述待迁移文件的总大小；旧实例实际目录大小尚未实测。
迁移输入与正式实验运行空间是两回事：交接中“保留三组所有监测点权重”的方案建议至少约150 GiB空闲运行空间，
仍需在启动前确定扩容或调整保存方案。
