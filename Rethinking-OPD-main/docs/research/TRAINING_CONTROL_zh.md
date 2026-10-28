# SSH断开后继续训练与进度查看

`run_base_comparison.sh ... --execute` 和 `python -m scripts.research.compare train ... --execute` 默认提交后台任务，随后返回终端。后台监控进程和训练进程各自建立独立会话，输入来自 `/dev/null`、输出写文件，不依赖SSH终端。当前激活的Python、CUDA分配和缓存环境会传给子进程，无需tmux或nohup。

未加 `--execute` 仍只生成计划。需要前台排错时显式加 `--foreground`；此模式依赖当前会话。服务器关机、容器停止、OOM或程序报错不会被自动重启，排错后用相同参数/同一workdir恢复最近完整checkpoint。

## 启动

这台服务器先进入已配置环境：

```bash
source /root/autodl-tmp/activate_opsd.sh
```

真实Base模型、原生DeepMath文件及来源revision按 `PILOT_HANDOFF_zh.md` 填写。原一键命令保持参数不变：

```bash
# 先审阅计划；不启动后台任务。
bash run_base_comparison.sh "${PILOT_ARGS[@]}"

# 提交2-step验收后台任务，返回PID与完整控制台日志路径。
bash run_base_comparison.sh "${PILOT_ARGS[@]}" --execute
```

**启动命令返回只表示任务已提交，不表示验收完成。** 查看状态为“已完成”且3个分析job完成，并按交接文档检查产物后，再启动另一个workdir的200-step正式任务。不要让pilot与正式任务同时争用这张卡。

同一workdir使用内核文件锁阻止重复启动。每次提交保留独立的控制台日志和退出码；再次提交会创建新的attempt，不覆盖历史日志。同一运行的checkpoint和训练指标仍由原恢复机制管理。

## 查看实时进度

重新SSH连接后激活环境，再执行（以下使用交接文档的验收目录；若自定义了workdir，替换为实际路径，不依赖上次SSH会话的变量）：

```bash
# 单次状态
bash run_base_comparison.sh status --workdir research_runs/base_opsd_acceptance_2step

# 每5秒刷新；Ctrl+C只退出查看，不影响训练。
bash run_base_comparison.sh status --workdir research_runs/base_opsd_acceptance_2step --watch

# 机器可读状态
bash run_base_comparison.sh status --workdir research_runs/base_opsd_acceptance_2step --json
```

正式任务将 `--workdir` 换为 `research_runs/base_opsd_200`（或你的实际正式目录）。不用重新填写模型/数据参数即可查看。

状态包含：后台进程身份与心跳、阶段、训练optimizer step/总步数/百分比、最近完整checkpoint、各监测点分析完成数与失败信息，以及最近记录的forward-KL/overlap/entropy。数据准备、权重导出、几何分析和清理旧权重分别显示，训练到200/200但分析尚未完成时仍显示“运行中”。

后台心跳只说明监控进程存活；判断训练是否推进还应看指标更新时间和日志。进度从真实 `metrics.jsonl` 和checkpoint tracker读取，忽略未写完的JSON行。恢复后若尚无本次新指标，暂按恢复checkpoint显示步数，并明确标注旧指标来自上一attempt。没有真实记录时不生成loss或预计完成时间。

`status --watch` 在整个任务成功后返回0，在失败、停止或丢失完成记录时返回非0；Ctrl+C退出查看返回0。`status` 单次查询本身不代表任务成功，自动化应检查JSON中的status。

## 查看日志

```bash
# 完整流程控制台：包含环境检查、数据准备、外层报错等
bash run_base_comparison.sh logs --workdir research_runs/base_opsd_acceptance_2step --follow

# GPU训练及保存后分析的详细日志；prepare时此文件可能尚未创建，会等待
bash run_base_comparison.sh logs --workdir research_runs/base_opsd_acceptance_2step --stream train --follow

# 只看末尾100行
bash run_base_comparison.sh logs --workdir research_runs/base_opsd_acceptance_2step --stream train --lines 100
```

`--method grpo`仅选择查看哪份已有训练日志，不启动GRPO。Ctrl+C仅关闭日志查看。后台任务已结束时follow会输出余下日志并退出。

## 停止与恢复

```bash
# 显式请求停止该workdir的受管任务和训练进程组
bash run_base_comparison.sh stop --workdir research_runs/base_opsd_acceptance_2step

# 排错后使用原参数恢复；仍默认后台执行
bash run_base_comparison.sh "${PILOT_ARGS[@]}" --execute

# 已prepare时，也可只恢复训练
python -m scripts.research.compare train --workdir research_runs/base_opsd_acceptance_2step --method opsd --execute
```

停止不是新checkpoint保存请求，最后一次保存后的更新需要重做。停止前核对workdir；代码使用boot ID、PID启动时间和可用的pidfd核验进程身份，不调用全局 `ray stop` 或停止其他实验。若监控进程意外丢失但子进程仍活着，状态会标记orphaned并拒绝重复启动，不把它当已完成任务。

## 记录位置

```text
<workdir>/.session/
  lock                         # 持续到后台任务结束的内核锁
  current.json                 # 当前attempt指针
  attempts/<id>/
    request.json               # 精确命令、Python及workdir（不写完整环境变量）
    state.json                 # 进程身份、心跳、退出码、完成/失败状态
    progress.json              # 当前阶段/方法/分析step
    console.log                # 本attempt的完整控制台日志
```

现有 `opsd/train/train.log`、`metrics.jsonl`、`job_status/`、各监测点分析结果和“只留最新checkpoint”策略继续生效。

## 数据准备进度与筛选结果保留

数据准备与训练使用同一条状态命令。下列示例为本轮2步验收目录：

```bash
source /root/autodl-tmp/activate_opsd.sh
bash run_base_comparison.sh status --workdir research_runs/base_opsd_20261028_acceptance_2step --watch
# 单次查询 / JSON
bash run_base_comparison.sh status --workdir research_runs/base_opsd_20261028_acceptance_2step
bash run_base_comparison.sh status --workdir research_runs/base_opsd_20261028_acceptance_2step --json
```

状态显示输入身份检查、原生数据读取、难度筛选/参考分词、逐题答案编码/参考核验/长度检查、去重划分、基线回答、基线抽取质量、固定输入池等阶段。
逐题处理每5秒刷新 `shared/preparation_progress.json`，包括已处理/总题数、通过/排除及原因、最近记录题目、耗时、实测速度。
有已完成题目的实测速度后才显示当前阶段的预计剩余时间；读取文件、模型加载、缓存校验等无题数阶段不编造百分比或ETA。
回答生成进度读取 `shared/generation_progress.json`，不将题目筛选完成当作训练完成。

筛选结果在GPU回答生成前保存到 `shared/preparation_selection/`，带数据/模型/选择规则身份和逐文件SHA256。
相同科学配置和源码下重试会校验并复用该检查点，重新处理尚未完成的后续阶段。
生成或质量门槛失败后，工作目录 `.probe-*`、完整筛选检查点、失败回答和进度错误记录保留；不会伪造最终 `probe/complete.json`。
更改协议、长度或筛选规则仍需新配置/目录，不能跳过身份校验。

旧运行没有进度文件时，状态工具可从已完整写出的 `preparation_audit/eligibility.jsonl` 和基线质量收据恢复真实计数；没有中间记录时不会猜测逐题进度。
按 Ctrl+C 只退出状态查看，不停止后台任务。
