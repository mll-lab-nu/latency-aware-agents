# hist8 v2 修复、重训与纠正评测

协议见 [DECISION_HISTORY_VLA_H8](../DECISION_HISTORY_VLA_H8.md)。旧批次处置见
[交接记录](../../outputs/tasks/decision_history_vla_hist8/old_history01_handoff.md)，
九路权重位置、实际步数和 SHA 见
[legacy_catalog.json](../../outputs/tasks/decision_history_vla_hist8_v2/legacy_catalog.json)。

## 已核验修复

- `state_source=transport` 直接传入原始二值 history；连续状态归一化不变。
- seed 42 在构造模型和随机动作头之前设置；完整断点恢复 scheduler 和 RNG。
- 独立 run ID、显式 VLM LR、可选暂停步数；PI 暂停保持 5,000 步 schedule。
- 固定验证样本图像干预、按 episode 的配对 CI 和 PI 选优均已接入训练。
- bundle 和评测记录实际训练步数、原 W&B ID、checkpoint SHA、输入协议版本。

验收结果：105 项相关测试通过，另 1 项 W&B namespace 集成测试通过。
真实双卡 ZeRO-2 检查在保存 step 4、恢复后执行同一 step 5 时，参数更新、LR、
scheduler step 和 Python/NumPy/CPU/CUDA RNG 精确一致。
固定旧 Demon OFT checkpoint SHA
`e1ad9ac05f7046d3f70ec77e22fbe6ca186d7a8147e5372e9d82370d7739e8ae`
的 64 条真实验证样本经过训练/推理路径，图像、提示、state、分数和动作一致；
全零 reset history 保持全零。两步完整 OFT smoke 生成 step 0、2 图像诊断。
这些是修复验收，不是正式训练或性能结果。

## 正式队列启动：2026-09-23 15:55 UTC

三台服务器均使用独立 `hist8-vla-v2-20260923` 目录。任务图共 36 个节点：
6 路 OFT/GR00T 训练、6 路 PI 1,000 步候选、3 次选优、3 次续训、9 路新版评测、
9 路旧权重纠正评测。每台控制器检测实时 GPU 进程与显存，保留其他任务使用的卡。

| 主机 | 当次启动的训练 | GPU | W&B |
|---|---|---|---|
| H200 `34.131.91.104` | Demon OFT | 0,1 | [8d64ae8b](https://wandb.ai/saberrr-zju/starvla_tasks/runs/8d64ae8b) |
| 同上 | Demon GR00T | 2,3 | [7f1c60f6](https://wandb.ai/saberrr-zju/starvla_tasks/runs/7f1c60f6) |
| 同上 | Demon PI `3e-6` | 4,5 | [b2071776](https://wandb.ai/saberrr-zju/starvla_tasks/runs/b2071776) |
| 同上 | Demon PI `3e-5` | 6,7 | [c9ff7291](https://wandb.ai/saberrr-zju/starvla_tasks/runs/c9ff7291) |
| H200 `34.131.223.74` | Flappy OFT | 0,1 | [126a7fee](https://wandb.ai/saberrr-zju/starvla_tasks/runs/126a7fee) |
| 同上 | Flappy GR00T | 2,3 | [8be125b0](https://wandb.ai/saberrr-zju/starvla_tasks/runs/8be125b0) |
| 同上 | Deadly OFT | 5,6 | [1ade216c](https://wandb.ai/saberrr-zju/starvla_tasks/runs/1ade216c) |

15:59 UTC 的真实训练日志已确认 Demon OFT 107 步、Flappy OFT 93 步、Deadly OFT
66 步；另外四路正在 step 0 的 2,048 样本图像诊断，GPU 持续计算。此时尚未
完成任何新版训练、PI 选优或正式环境评测，不能给出新版性能结论。

Deadly GR00T 及 Flappy/Deadly 的四个 PI 候选等待空闲双卡。Demon 旧 OFT 的纠正
评测位于第一台 H200 GPU 7；旧 GR00T/PI 位于 H100 `34.6.91.253` GPU 0/1。
评测采用修正后的 history01，CuLE 在新目录执行首次编译。

每台根目录下的 `outputs/tasks/decision_history_vla_hist8_v2/queue/state.json`
是任务状态；`queue/jobs/<job-id>/attempt_<n>/output.log` 是原始日志。训练目录下
`diagnostics/step_<step>.json` 保存图像利用证据。控制器失败不会自动重试或自动
选取不合格 PI；定位后使用 `--retry <job-id>`，已有工作保留在原 attempt。
本地 [receipts](../../outputs/tasks/decision_history_vla_hist8_v2/receipts/) 保存服务器
配置/代码 SHA、队列和验收回执的副本；副本是采集时快照，不能代替实时进程核验。

## Demon PI 配对起点核验

2026-09-24 00:03 北京时间，两组已完成 step 0 诊断并分别更新到 70/69 步。
两组的 cohort SHA 均为
`d037b492e5b9a517e462f49b7265a5b39564adaa612b7c7dcf9a196b2b9150c9`；
全部 episode 指标、动作统计和视觉特征统计精确相同。原始 loss
`1.243813332002901`，置乱 loss `1.2437829815389705`，视觉相对变化
`0.6170126629173923`。随机动作头初始不显示图像利用，这不是训练后选优结论。
配对资格和 winner 仍按 step 1,000 结果判断。

## 第一批 step 500 图像诊断

新版 OFT 的固定 2,048 样本诊断已确认图像利用，且继续训练：

| 任务 | 原始 loss | 置乱 loss | 增量 95% episode CI | 动作准确率：原始 / 置乱 |
|---|---:|---:|---|---|
| Demon OFT | 0.957759 | 1.959476 | [0.935039, 1.077551] | 0.625977 / 0.355957 |
| Flappy OFT | 0.090371 | 1.237324 | [1.020689, 1.269863] | 0.957031 / 0.766602 |

证据见 [diagnostics_500_oft.json](../../outputs/tasks/decision_history_vla_hist8_v2/diagnostics_500_oft.json)。
这是训练中的条件依赖证据，正式性能仍等待 step 5,000 和 100 局环境评测。
可读队列快照见 [execution_status.md](../../outputs/tasks/decision_history_vla_hist8_v2/execution_status.md)。

## 用户修订：仅用 Demon PI 选择底座 LR

2026-09-24 用户明确取消 Flappy/Deadly 独立的 PI 配对。新依赖关系是 Demon 两组
各到 1,000 步、按原图像利用与 loss 标准选出 LR；Demon 胜者完整续训，Flappy 和
Deadly 各从原始预训练底座直接训练 5,000 步。三者动作头/投影层配置不变。
若 Demon 两组均不合格，三条 PI 的后续训练仍等待定位，不随意选 LR。

相应删除四个尚未启动的候选和另两次选优，改为两路依赖 Demon 结果的正式训练。
总任务图由 36 个节点变为 30 个；九路最终模型及新旧各九路正式评测保持不变。
新增两种 LR 分支的语义测试确认另两任务不带 `--resume` 或 `--stop-after-steps`，
并保存 Demon 选优回执；相关 25 项测试通过。
服务器已重新载入修订后的队列，四个运行中的 Demon worker PID 保持不变。
迁移回执见 [shared_demon_pi_migration.json](../../outputs/tasks/decision_history_vla_hist8_v2/receipts/h200-2/shared_demon_pi_migration.json)。

## 补齐监控与 agent review

用户要求 PI 配对结束后唤醒当前任务审查，原本只有服务器自动选择/续训的队列，
缺少唤醒 agent 的部分。当前工具列表没有 `automation_update`；使用本机已提供的
`codex queue --thread ... --message ...` 和 systemd 用户服务完成同任务唤醒。
首次投递返回消息 ID `01a0cf10-d883-73d0-bd78-6d4e92b2452d`，
见 [wakeup_test.json](../../outputs/tasks/decision_history_vla_hist8_v2/monitor/wakeup_test.json)。

服务每两分钟查询三台，事件去重；两组 PI 完成、失败、进程/日志异常和评测完成
才投递消息。服务异常退出由 systemd 重启，所有队列完成后正常退出。
PI 选择节点增加 review 回执依赖：agent 审查后写入明确 winner，队列核对统计选择
一致再续训/入队，不需要用户再次确认。正在训练的子进程保持原 PID。

相关 28 项测试通过，包括等待 review、完成后只提醒一次和失败通知。第二台队列
控制器已载入 review gate；本机监控服务已启用并处于 `active/running`，主进程
`1014349`，首轮三台 SSH 查询成功。服务回执见
[service_receipt.json](../../outputs/tasks/decision_history_vla_hist8_v2/monitor/service_receipt.json)，
实时轮询快照见 [monitor/state.json](../../outputs/tasks/decision_history_vla_hist8_v2/monitor/state.json)。
托管启动时恢复了本机用户 D-Bus 控制接口；其他用户服务没有重启。

2026-09-24 00:33 北京时间，测试消息自动进入当前任务并触发新一轮执行，端到端
唤醒链路确认通过；`wakeup_test.json` 已更新为 `consumed_and_verified`。此时服务
仍为 `active/running`、重启次数 0，三台控制器存活。Demon 两个候选仍在训练，
review 回执未写入，PI 正式续训及另两游戏的正式训练继续等待真实配对完成事件。

## Demon PI 选优完成：2026-09-24 00:43 北京时间

两组均完整完成 1,000 步。核对 full config 后，差异仅为 run 标识/路径和底座 LR；
cohort、四组噪声及 step 0 诊断一致。两卡 optimizer/RNG 文件齐全，scheduler
`last_epoch=1000`、`_step_count=1001`。正式选择 **`3e-6`**：

| 底座 LR | 原始 loss | 图像置乱 loss | 动作准确率 | 图像增量 95% CI | 视觉相对变化 |
|---|---:|---:|---:|---|---:|
| `3e-6` | 0.164720 | 0.312436 | 64.1846% | [0.140086, 0.155529] | 0.751643 |
| `3e-5` | 0.230279 | 0.235762 | 49.0723% | [0.003484, 0.007835] | 0.006509 |

高 LR 减低 LR 的 loss 配对 CI 为 [0.059816, 0.071057]；两组均满足原门槛，
但低 LR 的 loss 显著更低，图像利用也明显更强。不能因为高 LR 图像增量仍显著
就忽略其视觉特征变化幅度的退化。该比较支持三游戏 PI 底座统一用 `3e-6`，
不代表 5,000 步正式环境结果已经通过。
审查和 checkpoint SHA 见 [选优回执](../../outputs/tasks/decision_history_vla_hist8_v2/selection/demon_attack-pi.reviewed.json)。

## 用户要求统一每 rank 16 workers

用户明确要求 `num_workers=16`。训练/验证均按每 rank 16 设置，双卡训练 loader
合计 32 个 worker，`prefetch_factor=4`、每卡 batch 64、global batch 128。
后台监控在受控切换期间暂由当前任务现场核验，控制器暂停派发；评测继续。
五路有训练更新的任务均完成完整中断保存：

| 任务 | 恢复步数 |
|---|---:|
| Demon OFT | 2,158 |
| Demon GR00T | 1,259 |
| Flappy OFT | 2,347 |
| Flappy GR00T | 1,336 |
| Deadly OFT | 1,890 |

Deadly GR00T 在 step 0 诊断后退出，尚未更新参数；原零步 run 留档，替换 run ID
为 `deadly_corridor-qwengr00t-hist8-scratch-s42-v2-workers16`。它仍是九路目标中的
同一路任务。Demon PI 从被选中的完整 step-1,000 状态续训；Flappy/Deadly PI
直接使用 `3e-6` 与 16 workers 开始正式训练。相关 29 项测试通过。

16 workers 切换后，五路有中断状态的训练与 Demon PI 胜者均已恢复实际参数更新。
第一台各 rank 的真实 `spawn_main` dataloader 子进程数已核对为 16；Flappy GR00T
也从 1,336 推进到 1,363 步，初次取数未复现此前的初始化超时。监控服务已恢复为
`active/running`。九路训练清单见
[queue_inventory.md](../../outputs/tasks/decision_history_vla_hist8_v2/queue_inventory.md)。

## 删除临时诊断代码（本地源码）

按用户要求移除训练中的 hist8 专用 W&B 配置、固定样本图像置乱诊断、统计选优
实现及其测试。新建队列直接采用已确定的底座 LR `3e-6`；已有队列仍可读取已完成
的选择结果，保持 run ID 和断点恢复路径。保留正常训练/验证 loss、各参数组 LR、
正式环境评测及输入/初始化/断点修复。以上历史诊断记录仅为既往选择证据，不再是
当前训练要求；原始结果文件保留。本次只修改本地源码与暂存区，未更新服务器代码
快照或正在运行的服务。
