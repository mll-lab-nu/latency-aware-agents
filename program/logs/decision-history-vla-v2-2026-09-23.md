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

## 扩展为 18 路训练与 27 组评测

用户新增九路 zero-latency 对照，并要求 zero→zero、zero→profile、profile→profile
三类各九组 100 局评测，parallel 32；最新 Flappy/Demon/Deadly 原始帧上限为
3,600/7,200/3,600，FPS/观测频率不变。监控间隔改为 20 分钟。

三个 zero teacher 固定在中央 HF revision `5b533937a72e9fa923ecf28b1c1be5a16bad488d`。
teacher 原始四帧输入保留，学生使用单帧 RGB＋另行导出的原始 0/1 H8。
Flappy rollout 覆盖为 10/10 FPS，rollout seed 42。每游戏一份数据供三个模型共用。
采集短测均完成 10 局；Demon/Deadly 各写入 72 train＋8 val 行，Flappy 288＋32 行。
首次短测暴露 exporter 在完成后清理 probe_state 的生命周期约定；现将 accepted
episode specs 独立保存后再导出，三个端到端采集短测全部通过。

相关 102 项测试通过。H100 的 GR00T/profile 双卡、PI/zero 双卡和 PI/profile 单卡
各完成 32 局×8 原始帧短测；分别耗时 28.71/32.30/31.29 秒（含模型冷启动，不能
据此线性外推完整局数）。这验证 32 并行输入、推理、解码及记录，不是性能结果。

为释放双卡位并采用新上限，保留后停止三份旧 par8 部分评测：Flappy OFT 7/100、
Flappy PI 59/100、Demon PI 48/100。目录加 `-partial-par8-before-horizon-update`，
含明确状态 JSON；已完成的长上限评测继续保留原协议记录。

正式队列含 18 个最终训练目标、27 组新版评测及 9 组旧权重纠正评测；加上已完成
PI 配对/选优、三份数据的采集/转换与 bundle staging，共 82 个依赖节点。
H200 双卡训练优先；常规评测整对占用释放，第一台单独余卡 7 可供独立采集/评测。
profile→profile 的 bundle 由 CPU 任务经内网传给 H100，核验 SHA 后才写 ready 标记；
传输不占 GPU。zero 模型每份 bundle 只封装一次供两个评测条件共用。

截至 2026-09-24 01:52 北京时间，第二台 H200 四对卡均进入训练，Flappy PI 已推进
到 170 步；第一台 GPU 7 正式采集 Demon zero 数据，Deadly zero 在 CPU 同时采集。
Flappy zero 已通过短测，等待同一 GPU 采集位。第一台 GPU 4 仍为 `/mnt/local/talha/`
下的另一项 cross-task 训练使用；其余 15 张 H200 卡均已分配给本批工作。

截至 02:03 北京时间，Flappy OFT 与 Demon OFT 的 profile 训练均已完成 5,000 步，
bundle 正由 CPU staging 任务传至 H100。Demon zero 数据已经采集、转换并同步完成：
200 局、324,000 train＋36,000 val 行、88D H8；六类动作均有覆盖。Flappy zero 和
Deadly zero 正式采集继续运行。当前新模型 100 局性能结果尚未完成。

## 删除临时诊断代码（本地源码）

按用户要求移除训练中的 hist8 专用 W&B 配置、固定样本图像置乱诊断、统计选优
实现及其测试。新建队列直接采用已确定的底座 LR `3e-6`；已有队列仍可读取已完成
的选择结果，保持 run ID 和断点恢复路径。保留正常训练/验证 loss、各参数组 LR、
正式环境评测及输入/初始化/断点修复。以上历史诊断记录仅为既往选择证据，不再是
当前训练要求；原始结果文件保留。本次只修改本地源码与暂存区，未更新服务器代码
快照或正在运行的服务。

## 旧权重纠正重评审计（2026-09-24 02:29 北京时间）

本次监控唤醒对应旧 Deadly PI 的完成通知。已核对该 step 3,266 checkpoint 的
100 个完整 seeds、权重与 profile SHA、逐局回报/局长、原始动作计数和最后一步：
均分 806.8397，标准差 744.0248，平均 129.25 原始帧；1,658 次命令覆盖 24 种
联合动作，最大单动作占比 41.19%，100 局最终 done，均未触及 3,600 帧上限。
该评测使用 history01-v2、parallel 1，不能作为新版 5,000 步同训练量对照。

同时完成另外七份旧权重重评审计，汇总及完整证据见
[评测汇总](../../outputs/tasks/decision_history_vla_hist8_v2/evaluation_summary.md)。
旧 Deadly OFT 在正确 history 下均分 2,049.45、最大单动作占比 34.41%；此前错误
输入记录为 184.24、95.3%。旧 Demon OFT 在正确 history 下仍有 99.91% 命令为
RIGHTFIRE，均分 148.50；输入修复没有消除该旧模型的动作集中问题。Demon OFT/GR00T
旧重评保留原 14,400 帧上限，PI 已采用 7,200 帧，不能直接作同上限排名。
原始错误输入记录保留，并补充指向正确协议结果的链接。

实时核对：三路 profile OFT 已完成 5,000 步，新版 Flappy OFT 的正式评测进行中；
其余训练持续推进，未见失败任务。Demon PI 的既有审查文件确认胜者仍为 `3e-6`，
续训日志已到 3,999 步；没有重复选优或启动。Flappy zero 数据已完成 200 局采集及
转换，647,783 train / 70,375 val 行；zero OFT 已到 191 步，zero GR00T 已被队列
接入。三台控制器及 20 分钟监控服务存活。代码快照与现有训练本次未修改。

## 监控运行环境修复（2026-09-24 02:32 北京时间）

收到已审计的旧 Demon OFT 完成通知后复查，三台控制器存活，训练步数继续增长，
没有重复评测或 PI 选优。发现当前本地 monitor 已复用 `latency_bench.utils.io`，
但服务 Python 缺少该模块需要的 PyYAML；既有进程仍可运行，新进程导入失败。
已在服务环境安装 PyYAML 6.0.3，验证模块入口后重启监控；新进程完成三台查询，
写入最新 state 和事件回执，保持 1,200 秒间隔与原去重状态。服务回执已更新。
该次查询碰到 Deadly OFT bundle staging 退出而 controller 尚未回填的间隙；
再次核对任务结果和 controller 日志均为 completed/returncode 0，未重试该任务。

## 首批新版 OFT 正式评测与回退复核（2026-09-24 03:00 北京时间）

Flappy OFT 和 Deadly OFT 的 step 5,000 正式 profile→profile 评测均完成 100 局。
已核验 seeds 1,000,000–1,000,099、权重/profile SHA、逐局回报/局长、终止记录及
动作计数和原始分数均值；均为 history01-par32-v3、capacity 1、3,600 原始帧上限。

Flappy 均分 373.7790 ± 126.8270，平均 3,030.62 帧，66 局触及上限；FLAP 占
38,075/302,228（12.60%）。相对旧权重正确 history 的 380.6680，按 seed 配对的
回报差 CI 为 [-38.73, 25.62]，尚无清楚提升证据。

Deadly 均分 211.4052 ± 276.2806，平均 73.46 帧；最大单动作占比 33.28%，19 种
联合动作。相对旧权重正确 history 的 2,049.4455，回报差 CI 为 [-1,959.90, -1,701.47]，
性能明显回退。核对新旧均沿用同一数据和 BCE loss、零阈值，全部 raw scores 解码
与发出命令一致。利用第一台空闲 GPU 7 复核新权重前 32 个相同 seeds，与 H100
对应 32 局回报和局长全部完全一致，均分 282.480086；复核完成已释放 GPU。
不是由 H100 主机或单/双卡推理差异造成，具体训练侧原因尚未确定。

正式结果和 SHA 已加入 [评测汇总](../../outputs/tasks/decision_history_vla_hist8_v2/evaluation_summary.md)，
比较与跨机证据见 [首批性能判断](../../outputs/tasks/decision_history_vla_hist8_v2/oft_profile_findings.md)。
此时新版正式评测完成 2/27，旧权重纠正重评完成 8/9；性能判断与训练/评测完成分开。
Flappy GR00T 也已完成 5,000 步并进入 staging，其释放的双卡由 zero Flappy PI 接入；
其余训练与 H100 上 Demon OFT 正式评测继续，未出现队列失败，未修改既定配方。

## Demon OFT 结果与余卡调度（2026-09-24 03:18 北京时间）

新版 Demon OFT step 5,000 的 profile→profile 正式 100 局已完成并通过审计。
seeds 1,000,000–1,000,099、parallel 32、capacity 1、60/15 FPS/Hz、7,200 原始帧
上限；均分 1,452.80 ± 284.18，100 局均到帧上限。90,043 次命令覆盖六类，
最大 LEFTFIRE 占 28.38%，NOOP 占 3.24%，没有全局单动作塌缩。权重 SHA 为
`eb5ff4d920ea4634853ef4c0788a16e6ae1cc3d8d2712a11a2cbcfc22743f03e`，W&B `8d64ae8b`。
旧权重纠正重评的上限为 14,400 帧、parallel 8，未将两者作同协议的配对增益。
结果已加入 [评测汇总](../../outputs/tasks/decision_history_vla_hist8_v2/evaluation_summary.md)。

实时核对，Demon GR00T 和 PI 也已完成 5,000 步并在 staging；正式训练完成 6/18，
新版正式评测完成 3/27。其释放的双卡已分别承接 Deadly PI 和 zero Demon OFT，
训练日志有实际参数更新；PI 沿用已审查的 `3e-6`，没有重复选优。
Deadly zero 采集/转换也已完成：8,250 局、328,296 train / 36,395 val 行，
原始上限 3,600；teacher revision/SHA 与合同一致，train/val manifest 和 mixture 均存在。
三份 zero 数据全部就绪，质量报告保存在本地 receipts/h200-1/data_zero/。

第一台 H200 GPU 7 空闲时，把尚未启动的 Flappy GR00T 新版正式评测从 H100 迁回
已有完整 bundle 的第一台。停调度器后持锁读取最新 state，确认源任务 pending/attempt 0，
从源 manifest/state 移除后才在目标添加，既有 worker 保持运行。两侧原始队列和迁移回执
均留存；现目标仅允许 GPU 7，100 局、parallel 32 与环境协议保持不变。
新 controller 存活，H100 上该任务已不存在，第一台唯一 eval worker 已完成模型初始化。
队列仍为 82 个依赖节点；三台现有训练、H100 旧 Flappy PI 评测继续，未发生失败。
证据见 [迁移回执](../../outputs/tasks/decision_history_vla_hist8_v2/queue_relocations/flappy_gr00t_eval_attached.json)。

## 旧九路纠正重评全部完成（2026-09-24 03:34 北京时间）

最后一项旧 Flappy PI 的 step 5,000 正确 history 重评完成并核验 100 局。
seeds 1,000,000–1,000,099，10/10 FPS/Hz、capacity 1、parallel 32、3,600 原始帧
上限；均分 292.1640 ± 151.5298，平均局长 2,373.10 帧，41 局到上限、59 局最终 done。
101,580 次命令中 FLAP 为 15,298（15.06%）；未出现全局单动作塌缩。权重 SHA
`a93ea26c1c34aab88fd8286aee9e80ba4676a0c8d66bf6967fa5db86dd5ac137` 与旧批次目录、
bundle 标记及评测 provenance 一致，W&B run 为 `2e02a5ed`。

九路旧权重正确输入重评现为 9/9 × 100 局；其中三条未训练满 5,000 步的旧权重
继续保留实际步数标识，旧 Demon OFT/GR00T 已完成的长上限协议也保留原值。
汇总与逐局证据见 [评测汇总](../../outputs/tasks/decision_history_vla_hist8_v2/evaluation_summary.md)。
新版正式评测仍为 3/27，训练完成 6/18，不能把旧重评完成当作新版性能验收。

实时核对三台控制器与 20 分钟监控均存活，无 failed/needs_diagnosis 或失联进程。
H100 已自动进入新版 Demon GR00T 正式评测，第一台 GPU 7 的新版 Flappy GR00T
已产出 episode；七路在训任务继续更新。Deadly zero 转换完成事件此前已核验，
本次未重复采集、启动训练、选优或评测。

## Flappy / Demon GR00T 正式结果（2026-09-24 03:54 北京时间）

两路新版 GR00T 的 step 5,000、profile→profile 正式评测均完成 100 局，并通过
seeds 1,000,000–1,000,099、checkpoint/profile SHA、逐局回报和局长、动作计数、
原始分数均值及终止记录核验。均为 history01-par32-v3、capacity 1、parallel 32。

| 游戏 | 均分 ± 标准差 | 平均原始帧 | 到帧上限 / 100 | 动作审计 |
|---|---:|---:|---:|---|
| Flappy | 414.5240 ± 89.9948 | 3,358.00 | 87（上限 3,600） | 167,903 次命令，FLAP 18,328（10.92%） |
| Demon | 1,203.10 ± 310.3271 | 7,111.02 | 96（上限 7,200） | 82,064 次命令，六类均出现，最大 FIRE 37.80% |

Flappy 权重 SHA `5989ecd1e19c3ea4a4fc479fb90277f11b0cd4962173afa39ed22406daa1d939`，
W&B `8be125b0`；Demon 权重 SHA
`a059ad8efbed631dc1963c9128bd033f6065671aa6b053d6bc3e57438d98fab8`，W&B `7f1c60f6`。
均无全局单动作塌缩；达到帧上限按截断记录，不称为通关。
旧 Flappy GR00T 只有 4,196 训练步，旧 Demon GR00T 使用 14,400 帧/parallel 8，
因此不将新旧均分差归为学习率单因素的配对提升。不同架构沿用各自 profile，
本表也不作为相同延迟下的跨架构排名。

[评测汇总](../../outputs/tasks/decision_history_vla_hist8_v2/evaluation_summary.md)现包含
旧九路全部结果和新版 5/27 项正式结果。H100 已自动接续新版 Demon PI 正式评测；
七路训练有实际步数更新，三台 controller 和监控正常，无失败或失联任务。
第一台 GPU 7 在 Flappy GR00T 结束后释放；当前可在本机评测的后续 zero checkpoint
尚未训练完成，保留既定双卡训练和就绪后评测依赖，未重复启动或修改配方。

## Demon PI 正式结果（2026-09-24 04:35 北京时间）

新版 Demon PI profile→profile 的 step 5,000 正式 100 局完成。bundle 的完整
run ID 为 `demon_attack-qwenpi_v3-hist8-scratch-s42-v2-vlm3e-6`，与既有审查 winner
一致；底座 LR `3e-6`、W&B `b2071776`，并非暂停在 1,000 步的候选权重。
checkpoint SHA 为 `f48e84564276085b48f0a1d6164c48c91e5865c30bef0aa6bd3ea674357b2338`。
100 seeds、bundle step/SHA、profile SHA、history01、逐局数据、原始分数均值和
动作计数均通过核验。

协议为 seeds 1,000,000–1,000,099、60/15 FPS/Hz、capacity 1、parallel 32、
7,200 原始帧上限。均分 907.30 ± 280.3876，平均局长 7,087.70 帧；92 局到上限，
8 局最终 done。56,727 次命令覆盖六类，最大 FIRE 占 57.05%，未发生全局单动作塌缩。

旧 PI 正确 history 重评使用同一 profile、seeds、帧上限和并行数，均分 939.60。
新减旧为 -32.30，按 seed 配对 bootstrap 95% CI 为 [-113.0025, 51.70125]；
未显示明确控制性能提升。1,000 步学习率筛选符合原规则，但不能替代正式性能验收，
新旧训练配方还有初始化/scheduler/恢复行为差异，也不是单独 LR 的因果对照。
比较见 [Demon PI 结果](../../outputs/tasks/decision_history_vla_hist8_v2/demon_pi_profile_comparison.json)，
正式证据已加入 [评测汇总](../../outputs/tasks/decision_history_vla_hist8_v2/evaluation_summary.md)。

新版正式评测完成 6/27，旧纠正重评为 9/9。profile Flappy PI 和 zero Flappy OFT
也已完成 5,000 步，使正式训练完成 8/18；前者正在 staging，后者的 zero→zero
100 局评测已在第一台 GPU 7 推进到首批 32 局。其释放的双卡分别承接 zero Demon
GR00T 和 zero Deadly OFT，保持训练优先；实时核对三台 controller、监控和已有
worker 正常，训练继续更新，没有失败任务、重复选优或启动。

## Flappy OFT 三条件结果齐备（2026-09-24 05:18 北京时间）

zero 训练 OFT 的 zero→zero、zero→profile 均完成并核验 100 局。两项使用同一
step 5,000 权重 SHA `a6bdc9bf2373827958de75649fa7ea0ce2f7ac8b46d9fc3ccd63d66bd5697fd3`，
W&B `47307ef0`，run `flappy-qwenoft-hist8-zero-scratch-s42-v2`；teacher revision/SHA、
zero 数据来源和训练条件均与合同一致。共同 seeds 1,000,000–1,000,099、10/10 FPS/Hz、
capacity 1、parallel 32、3,600 原始帧上限。SHA、逐局记录、动作计数及原始分数
均值均已核对；zero 的每局 mean_latency_ms 均为 0，profile 为 74.93–77.21 ms，
且 profile SHA 与 profile 训练 OFT 的正式评测相同。

| 训练→评测 | 均分 ± 标准差 | 平均原始帧 | 到上限 / 100 | FLAP / 发出命令 |
|---|---:|---:|---:|---:|
| zero→zero | 439.1280 ± 38.4325 | 3,555.94 | 98 | 29,120 / 355,594（8.19%） |
| zero→profile | 19.1870 ± 17.0565 | 173.53 | 0 | 2,361 / 17,312（13.64%） |
| profile→profile | 373.7790 ± 126.8270 | 3,030.62 | 66 | 38,075 / 302,228（12.60%） |

同一 zero 权重从 zero 环境进入 profile 环境后明显退化，两类动作仍有输出。
profile 训练权重在该 profile 下保持较高分数；两种训练条件使用合同中各自的
teacher/数据，不把跨训练条件比较表述为仅改变某个优化参数的消融。
核对回执见 [三条件证据](../../outputs/tasks/decision_history_vla_hist8_v2/flappy_oft_latency_conditions.json)。
[评测汇总](../../outputs/tasks/decision_history_vla_hist8_v2/evaluation_summary.md)新增三条件矩阵，
完整记录中显式区分训练→环境，避免同一个 zero checkpoint 的两项评测混淆。

新版正式评测核验完成 8/27；训练完成 10/18，包括刚完成的 zero Demon OFT 和
zero Flappy GR00T。后者已封装，并由第一台 GPU 7 自动接续 zero→zero 评测，
释放的双卡由 zero Deadly GR00T 接入。H100 的新版 Flappy PI 已到 95/100 局；
其余任务继续，三台 controller 和监控正常，无失败、重复启动或训练配方变更。

## Flappy PI 正式结果（2026-09-24 05:35 北京时间）

新版 profile Flappy PI 的 step 5,000 正式 100 局已核验，完整 run ID 为
`flappy-qwenpi_v3-hist8-scratch-s42-v2-vlm3e-6`，W&B `7660c2a2`，底座 LR `3e-6`。
权重 SHA `314aa15602db143a58011fff17fa4d1fcdbabf8f32135870ceff61877e5e6cce`、
100 个 seeds、profile SHA、动作计数与分数、逐局回报/局长均匹配。
协议为 seeds 1,000,000–1,000,099，10/10 FPS/Hz，capacity 1、parallel 32、
3,600 原始帧上限。

均分 344.4830 ± 145.2504，平均局长 2,794.21 帧；60 局到上限，40 局最终 done。
119,617 次命令中 FLAP 为 18,153（15.18%），没有全局单动作塌缩。
旧权重正确 history 在同一评测协议下均分 292.1640；新减旧 52.3190，按 seeds
配对 bootstrap 95% CI 为 [9.713875, 94.092101]。这是本批完整训练配方的改善，
不能单独归因于底座 LR。比较见
[Flappy PI 配对结果](../../outputs/tasks/decision_history_vla_hist8_v2/flappy_pi_profile_comparison.json)。

[评测汇总](../../outputs/tasks/decision_history_vla_hist8_v2/evaluation_summary.md)现已核验
新版 9/27 项，旧权重纠正重评维持 9/9。初次实时核对没有失败或失联 worker；
H100 双卡在本项完成后空闲，第二台 H200 的 zero Demon OFT 已封装，两个环境
条件的正式评测等待双卡位，正在按空闲资源调整其中一个尚未启动的评测位置。

本次迁移调整为 **zero Demon OFT 的 zero→profile**。原拟迁移的 zero→zero 在
持锁复查时已被源队列启动，因此保留该 worker，并恢复源 controller；没有停止或
重启其评测。仅从最新源队列移除了仍为 pending/attempt 0 的 zero→profile，
回执为 `queue_relocations/demon_oft_zero_eval_detached.json`。目标加入显式 bundle
ready 标记依赖；源队列原来依赖本机 package job，没有 `ready_path` 字段，迁移脚本
已按跨机目标补上该路径。

H200→H100 内网复制当前仍在进行，实测约 11.6 MB/s，尚未标记迁移完成。脚本在
目标权重 SHA 匹配后才复制 `bundle.complete.json` 并原子挂接 H100 队列；正式协议
仍为 100 局、parallel 32、两张推理卡。源 controller `3965290` 存活，原 zero→zero
继续在 H200 本机运行；该迁移未占训练 GPU。

迁移编排由本机 PID `1246360`、工具 session `94232` 持续执行。已启动独立的
`codex-hist8-demon-offload-followup.service`（核对 active/running），等待此编排退出后
按 attached 回执是否写入，向当前任务发出完成/失败唤醒；失败时明确要求依据
detached 回执恢复或完成挂接，避免漏掉这一项。原 20 分钟监控保持运行。
跟进证据与脚本在 `outputs/tasks/decision_history_vla_hist8_v2/queue_relocations/`；
在 attached 回执出现前，这一评测仍计为未完成。后续应核对空闲 GPU，并继续让
就绪评测占用空闲双卡或第一台的独立余卡，避免空卡只因主机归属固定而等待。

## Deadly GR00T 正式结果与迁移完成（2026-09-24 05:55 北京时间）

新版 Deadly GR00T step 5,000 的 profile→profile 正式 100 局已核验。实际训练 run
为 `deadly_corridor-qwengr00t-hist8-scratch-s42-v2-workers16`，W&B `d78243bb`，
权重 SHA `a68c7ec382c18367164a938eae0332cb8e1cbea28b94907e9640717cd5d7d3d2`。
协议为 seeds 1,000,000–1,000,099、35/8.75 FPS/Hz、capacity 1、parallel 32、
3,600 原始帧上限。Seeds、SHA、profile、逐局记录、动作计数与原始分数均值均通过。

均分 286.4717 ± 365.9562，平均局长 76.85 帧；100 局最终 done，无一到帧上限。
999 次命令包含 15 种联合动作，最大 `MOVE_FORWARD+MOVE_LEFT+ATTACK` 占 37.24%。
没有全局单动作塌缩，但不据此宣称 Deadly 控制问题已经解决。旧 GR00T 只有 2,877
训练步，其 231.31 均分不作为同训练量对照。正式结果已加入
[评测汇总](../../outputs/tasks/decision_history_vla_hist8_v2/evaluation_summary.md)，新版已核验 10/27。

上一条 zero Demon OFT 的 zero→profile 迁移现已完成：目标完整权重 SHA
`19105254b128937823b3f46ca52702340a97dfd929237f6cb80ba7e45196b1cb` 与源 bundle 一致，
ready 标记在校验后写入，attached 回执已生成。实时核对源 H200 队列已无该 job，
H100 唯一 worker `737820` 使用 GPU 0/1，进程存在；目标 controller `737761` 存活。
zero→zero 继续由原 H200 worker 运行，未中断或重启。迁移后的依赖节点仍共 82 个。
完成/失败回调也已成功向当前任务排入完成通知；本轮已消费核验，后续同一回调
不应触发重复迁移或评测。证据位于 `queue_relocations/demon_oft_zero_eval_*.json`。

此时训练完成 12/18：profile Deadly GR00T 和 zero Flappy PI 均已完成 5,000 步；
后者释放的双卡已由最后一项待启动训练 zero Deadly PI 接入并产生实际更新。
六项剩余训练持续运行；两台 H200 各有评测推进，三台 controller 和主监控正常，
没有失败任务。训练完成、评测完成和性能判断继续分别记录。

## 三条件结果、就绪任务调度与 Deadly 根因（2026-09-24）

Demon OFT 的 zero→zero / zero→profile / profile→profile 已各完成 100 局，
均分分别为 2,362.15 / 584.35 / 1,452.80。前两项使用同一 step 5,000 SHA
`19105254b128937823b3f46ca52702340a97dfd929237f6cb80ba7e45196b1cb`；实际 zero 延迟
均为 0，profile SHA 与 profile 训练模型的评测一致，协议保持 7,200 帧、parallel 32。

本轮同步核验了 Flappy GR00T 的 zero→zero / zero→profile（428.4760 / 6.6160），
及 zero Flappy PI 的 zero→profile（5.5820）。原拟迁移的 GR00T 评测已由队列先行
完成，因此未重复运行；实际把仍为 pending 的 Flappy PI zero→profile 移至第二台
H200 空闲双卡。两台内网复制校验通过，唯一评测 worker 完成 100 局、returncode 0。
profile Deadly PI 的待执行跨区 staging 改成本机 CPU 封装，评测从 H100 移回第二台。
全部迁移和原始队列留档于 `queue_relocations/rebalance_flappy_zero/`；节点数保持 82。

继续排查 Deadly 弱表现时，确认了独立于 history 编码的真实输入协议错误：
采集使用 Sample Factory 160×120、带 HUD 的原始 RGB；VLA 评测使用 Gymnasium
默认 320×240、隐藏 HUD 的视图。相同 WAD SHA、相同 seed 的真实重置帧即有差异，
统一缩放到 224×224 后仅约 13.2% 通道值相同。原始帧栈在采集时已有独立复制；
本次没有把图像缓存覆写或标签错位当作已证实原因。

固定新版 OFT 原 SHA、同一组 32 seeds，仅对齐渲染设置，均分从 282.4801 提升至
2,087.9641。随后修复入口，六项渲染设置显式进入 eval config；仅注册名称不存在
时尝试旧 Gym ID，渲染配置错误不再被宽泛异常捕获替换。34 项测试通过，包括真实
ViZDoom 分辨率/HUD 回归和队列按指定 runtime_root 调度测试。

三台建立独立 `hist8-vla-eval-sf-render-v4-20260924` 代码快照；训练保留原快照，
Deadly 封装/评测 job 指向 v4，完成的八份错误视图评测沿用 checkpoint 正式重跑。
原目录和逐局记录保留并标记 invalid_renderer_protocol；本机旧证据恢复留档，
v4 证据按协议单独存放。八份重评均已完成并通过 100 seeds、实际步数、SHA、
渲染配置、动作及逐局记录审计。

| Deadly 项目 | 原错误视图均分（失效） | sf-render-v4 均分 |
|---|---:|---:|
| 新 OFT profile→profile，5,000 步 | 211.41 | 2,091.23 |
| 新 GR00T profile→profile，5,000 步 | 286.47 | 2,090.82 |
| 新 PI profile→profile，5,000 步 | 849.77 | 1,450.83 |
| 新 OFT zero→zero，5,000 步 | 1,663.72 | 2,098.67 |
| 新 OFT zero→profile，5,000 步 | 595.83 | 742.35 |
| 旧 OFT，5,000 步 | 2,049.45 | 2,031.24 |
| 旧 GR00T，2,877 步 | 231.31 | 2,101.98 |
| 旧 PI，3,266 步 | 806.84 | 1,496.04 |

此前把 Deadly OFT 低分判断为训练侧回退的结论撤回；两台机器此前共享同一错误
渲染设置，所以跨机复核一致并未排除此类评测错误。正确视图下旧/新 OFT 差值
95% CI 为 [-42.50, 166.35]，没有明确回退证据。源码、图像和固定权重证据见
[根因修复报告](../../outputs/tasks/decision_history_vla_hist8_v2/deadly_renderer_fix.md)。

在排查和重评期间，其他队列继续推进：Flappy PI zero→zero 为 408.7170，Demon
GR00T zero→zero / zero→profile 为 2,370.90 / 231.90，均完成各 100 局并通过审计。
有效结果已更新三条件矩阵；错误视图分数从有效计数中剔除，按协议保留历史证据。

v4 重评及失效原因已回写 W&B，7 个受影响 run 的旧指标带有
`input_protocol_valid=false` 和原因，新 v4 指标为 true；抽查 `1ade216c` 的服务端
summary 已确认旧 false / 新 true / v4 协议字符串。回执为
`renderer_v4_wandb_flags_receipt.json`。本机按协议保存 v4 证据子目录，旧目录恢复
原始逐局文件；共享 bundle 在迁移时捕获的元数据单独命名，避免当成旧执行快照。

后续又完成 Demon PI 的 zero→zero / zero→profile 各 100 局，均分为
2,350.95 / 112.30；计入有效三条件矩阵。当前有效新版评测为 23/27，旧权重纠正
评测恢复为 9/9，另保留 8 份失效的旧 Deadly 渲染协议结果。剩余四项为 zero
Deadly GR00T / PI 的两个环境条件，随权重完成由 v4 队列自动执行。

## Deadly zero GR00T 两条件验收（2026-09-24 08:55 北京时间）

zero GR00T 已完成 5,000 步、最终验证与保存，W&B `f2a666e1`；两组正式评测
使用同一 SHA `004a7092edb951c671c40bcfa896178322aafcaebd3f476b02ebb504bc669a6a`。
zero→zero 均分 2,100.4612 ± 581.2169，平均 174.37 原始帧；zero→profile 为
341.7600 ± 165.5712，平均 76.86 帧。两项均为 sf-render-v4、100 个正式 seeds、
parallel 32、capacity 1、35/8.75 FPS/Hz、3,600 帧上限；100 局最终 done，均未达上限。
逐局 seeds、实际权重 SHA、动作计数和原始分数均值核对通过。两项分别发出
4,404 / 1,004 个动作，覆盖 24 / 15 种联合动作，最大动作占比 21.25% / 45.42%。

zero 环境逐局记录的平均及 p99 延迟均为 0；profile 环境使用的 profile SHA 与
该架构 profile 训练模型正式评测完全一致。后者均分为 2,090.8181；同一个 zero
checkpoint 在加入 profile 延迟后明显退化。两个训练条件还使用不同 teacher 和
采集数据，不把 profile 训练模型的差异解释成仅学习率造成的提升。证据见
[三条件核对](../../outputs/tasks/decision_history_vla_hist8_v2/deadly_gr00t_latency_conditions.json)。

有效新版评测现为 25/27，旧权重纠正评测 9/9；训练完成 17/18。实时核对剩余
zero Deadly PI 约 4,679 步且持续更新，其封装及两个 v4 环境评测已在依赖队列。
第一台 controller 与每 20 分钟监控正常；第二台/H100 的现有队列已全部完成，
controller 正常退出。没有失败任务，也未重复启动已完成工作。

## 最后两组 PI 评测与全部收尾（2026-09-24 09:40 北京时间）

zero Deadly PI 已完成 5,000 步，W&B `2f9d64db`，SHA
`ba72ed7f1ba94daa0eba7d36fdb36eb7ce76db4a448bdbb29499e3f72ed0912f`。
同一权重在 sf-render-v4 的 zero→zero / zero→profile 各完成 100 局，均分为
2,104.0864 ± 572.5117 / 140.9386 ± 170.2019，平均原始帧 175.16 / 75.27。
两项分别发出 4,423 / 985 个动作、覆盖 20 / 15 种联合动作，最大占比为
21.23% / 38.88%；100 局均最终 done，均未达 3,600 帧上限。seeds、SHA、
profile、history、渲染配置、逐局结果、原始分数和动作计数均通过审计。

最终逐项核对完整 3×3×3 矩阵：27/27 组新版、每组 100 局，共 2,700 局；
旧权重纠正重评 9/9，共 900 局。18 个独立新版模型均在 5,000 步完成，最终
配置符合每卡 64/global 128、16 workers、prefetch 4、底座 3e-6。远端核验
18 路最终 state：每路两个 optimizer shard、两个 RNG 文件和 scheduler；
`last_epoch=5000`、`_step_count=5001`。W&B 18 个新版 run 的 27 组正式均分
全部回读匹配本地结果。固定 zero 权重两环境的 SHA、相应 profile SHA 逐组一致。

三台实时队列为 40+31+11=82 个节点全部完成，无失败或运行节点；controller
正常退出。20 分钟监控已记录 `status=completed` 并以 success 正常退出，未
重复启动任务。失效的 8 份 Deadly 旧视图记录继续保留，不计入有效成绩。

全部配置/身份与证据在[收尾清单](../../outputs/tasks/decision_history_vla_hist8_v2/completion_manifest.json)，
三条件矩阵与结论在[最终报告](../../outputs/tasks/decision_history_vla_hist8_v2/final_report.md)。
训练完成、评测完成均已确认；成绩按单一 training seed、各架构不同 profile 及
两种 teacher/采集数据的限制解释，本任务未进行模型仓库发布。

## 18 模型中央 HF 发布完成（2026-09-24 11:36 北京时间）

按用户指令，18 个 step-5000 原始权重发布至 `latency-sensitive-bench/benchmark-models`：
zero-latency 9 个，latency-aware 9 个。使用既有游戏 slug 和 `vla/starvla-<model>-h1/<run-id>`
目录结构；hist8 表示输入历史，h1 表示动作 horizon。总权重 184,111,173,702 字节。

第一台上传尾段连续明显变慢，停止并核对本任务 uploader 后，经两台内网复制
剩余 8 份 bundle 到第二台的独立 publication_peer 目录，保留已验证 receipt；
随后以两路上传完成剩余模型。没有改动训练、评测、原权重或其他任务进程。

最终统一配置/索引提交为 `95aed229d8a90c0146885eb8f496518e98625d5f`。
18 个 checkpoint 的远端 LFS SHA 全部匹配源权重；588 个元数据对象核对通过，
102 个 README、配置、任务合同、provenance、成绩及索引文件实际下载回读一致。
公开元数据使用归档相对引用，部署配置使用包内路径；profile 及 sidecar 保持原字节。

本轮只发布 18 个新版模型及其证据；旧权重和失效评测不进入这批发布。
中央 README 和 reports/hist8-v2 提供 18 个模型路径与 27 组正式成绩。
[发布回执](../../outputs/tasks/decision_history_vla_hist8_v2/hf_publication/README.md)
包含固定 revision、全部路径、SHA 和读回证明；artifact registry 与运行合同已更新。
