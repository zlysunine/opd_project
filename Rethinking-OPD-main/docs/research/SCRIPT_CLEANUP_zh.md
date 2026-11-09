# 旧版本实验脚本清理记录

> 本页记录2026-10-21的历史操作。2026-11-09用户又删除了 `opsd_pipeline.sh`、`run_opsd_experiment.sh`，相关运行依赖与测试已解除；最新入口为 `run_controlled_comparison.sh`。下文“必须保留”等描述为当时记录。

日期：2026-10-21。**状态：用户回复“清理”后已执行；以下 9 个文件均已删除。** 删除前逐项核对内容 SHA256，与获批清单一致。

本次先完成服务器交接与入口调整：根 `on_policy_distillation.sh` 已恢复为 HEAD 原版，`grpo.sh` 未改，两者逐字校验一致。最新实验一键入口为 `run_opsd_experiment.sh`，分阶段入口为 `opsd_pipeline.sh`。

## 已清理的具体文件（9 个）

| 路径（相对项目根目录） | 清理原因 | 字节数 |
|---|---|---|
| `run_pilot_server.sh` | 旧双 4B 外部教师实验的在线安装/下载入口 | 312 |
| `pilot_thinking_consistency.sh` | 旧双 4B 外部教师实验的离线入口 | 593 |
| `scripts/pilot/server_bootstrap.py` | 旧实验专用在线安装、三模型下载和费用调度 | 20157 |
| `scripts/pilot/thinking_consistency.py` | 旧 DAPO/AMC23 双外部教师训练、评测和报表 | 43767 |
| `scripts/pilot/requirements-server.txt` | 旧在线 bootstrap 专用环境依赖 | 688 |
| `scripts/pilot/test_server_bootstrap.py` | 仅测试旧 bootstrap 的测试 | 9135 |
| `scripts/pilot/test_thinking_consistency.py` | 仅测试旧双教师 pilot 的测试 | 7531 |
| `scripts/pilot/README.md` | 旧 DAPO/AMC23 双教师 pilot 说明 | 7494 |
| `scripts/legacy/on_policy_distillation.sh` | 原 OPD 的重复副本；原版现已恢复在根目录 | 12944 |

以上均属于旧双外部教师实验或原 OPD 的重复副本。最新 `scripts/research/`、`configs/research/`、`tests/research/` 不导入这些旧模块，一键脚本不调用它们。当前文档中的旧流程记录是历史验证信息，不是运行依赖。

每个文件删除前的完整 SHA256、是否受 Git 跟踪、用户批准记录和实际删除路径见 [legacy_cleanup_manifest.json](legacy_cleanup_manifest.json)。本次仅删除清单中的 9 个指定文件，并移除空的 `scripts/legacy/`；`scripts/pilot/` 内不在清单中的 Python 字节码缓存保留，未递归清理目录。

## 必须保留

- 原仓库根入口：`grpo.sh`、`on_policy_distillation.sh`。
- 最新实验入口与实现：`run_opsd_experiment.sh`、`opsd_pipeline.sh`、`scripts/__init__.py`、`scripts/research/`、`configs/research/`、`tests/research/`。
- 定制训练代码及原有例子：`verl/`、`LlamaFactory/`、`verl_example/` 及其他原仓库训练/评测代码。
- 全部数据、权重、checkpoint、日志和历史结果，尤其 `datasets/`、`model/`、`pilot_runs/`、`research_runs/`。
- 当前服务器交接文档、指标定义、实验设计与验证记录。

## 同步收尾

1. 已删除清单内文件，移除实际存在的空目录。
2. 已更新文档中的旧测试命令/入口引用，保留明确标注的历史验证记录。
3. manifest 已更新为用户批准、执行完成，并记录实际删除项。
4. 最新管线的测试和入口检查结果见 [VALIDATION_zh.md](VALIDATION_zh.md)。

原仓库两个入口和最新一键/分阶段入口的内容哈希与清理前一致；数据、权重和实验产物均保留。
