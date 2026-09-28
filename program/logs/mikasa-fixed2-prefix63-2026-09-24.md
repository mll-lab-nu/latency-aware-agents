# Fixed-2 prefix63 重训记录（2026-09-24）

用户批准方案 A：只重训 Fixed-2，第二台 H200 GPU 0–1；旧 Fixed 任务由用户处理，
本任务不操作。SFT 5k + DAgger 5k；DAgger step 0 及每 500 steps 五档开发评测，
只按 L2 选优。完整合同见 [协议](../MIKASA_FIXED2_PREFIX_REPAIR.md)。

本地修改先于同步：输入为 63D 对齐 prefix，固定动作输入 min/max，raw v2 可重打包为
v3，DAgger 原生保存五档结果和按单元格恢复。训练暂停时由 resident 服务在 ready
之前释放未用 CUDA cache，无需修改或共享更新其他任务的 StarVLA 子模块。

启动前核实第二台 `34.131.91.104` GPU 0、1 空闲；其余 GPU 不用于本任务。
测试、源代码快照、数据审计和启动证据保存至 `outputs/tasks/mikasa_intercept_grab_fast/`。

## 正式启动

- 本地核心及 resident 回归 121 passed；独立远端源码的完整 CPU 回归 138 passed，
  一条与本任务无关的 Gymnasium/Atari 注册 warning。真实 QwenOFT 构造和 projector
  探针确认输入63维，改变 prefix 能改变实际输入，训练/推理 min-max 一致。
- 运行目录 `/home/ubuntu/codex/mikasa-fixed2-prefix63-20260924`；run ID
  `fixed2-prefix63-20260924`。StarVLA 从本地 `3cadac2e264b5f0013a2d10d2551abfb0a41a081`
  独立复制；MIKASA 和 teacher artifact 复用旧环境只读路径。
- 第二台 GPU 0、1 启动前均为 0 MiB，controller PID `3088981`；正式流程先执行
  原始数据重打包和全量输入/标签审计，再进入 SFT 5k、DAgger 5k、五档终评和绘图。
- [本地部署及验证证据](../../outputs/tasks/mikasa_intercept_grab_fast/prefix63-20260924-launch/launch.json)
  与同目录 `review.patch`、`source.json`、`REVIEW.md`；未执行 commit。

## 数据完成与 SFT 启动

- 原始数据全量审计通过：train 674 episodes / 11,570 rows / 64,368 valid heads；
  validation 74 episodes / 1,250 rows / 6,943 valid heads；划分不交叉。
  proprio 七维全部统计和动作输出统计与旧合同完全一致，新 prefix 输入 min/max 为 -1/+1。
- 第一次启动在 `code_revisions` 遇到 Git 不接受子模块目录符号链接；未进行梯度更新。
  改为独立 MIKASA worktree，覆盖本地 `017490bbb787c6efa8c025f9558e1a691965c0e0`
  的 tracked runtime 源码，复用原 Python 环境；旧目录未修改，已有打包数据完整复用。
- 新 controller PID `3538537`；SFT 已启动双 rank，日志确认 microbatch 64、accumulation 1、
  global 128；配置为 state_dim 63、gradient checkpointing false、seed 0、5000 steps、
  save_interval 500、pretrained_checkpoint null、is_resume false。

SFT 已核实推进至 step 260；早期 step 90 loss 0.22570，双卡显存各约 136 GiB，
未触发 OOM 或 batch 改写。训练尚未完成，DAgger、五档终评与图由控制脚本顺序接续。

本地一次性结果同步进程 PID `1892397` 每两分钟只读检查本次 controller；运行中不发消息，
完成或失败后拉取指标、图和日志到启动证据目录的 `final-evidence/` 并退出，不重启或停止训练。

## SFT 完成后的首轮 DAgger

SFT `run.json` 确认 5000 updates 完成。step 0 五档各 100 局完成，成功数
L0–L4 为 13/13/10/12/5，仍只按 L2 参与选优；证据为启动目录 `step0-status.json`。
控制器已进入 `deltas/round-01/raw` 的新 student rollout；GPU 0 模拟、GPU 1 推理，
目标至少 16000 个有效 suffix heads，之后打包再启动首轮500次 optimizer updates。
本地结果同步曾因 SSH 255 退出，未影响服务器控制器；同步器现记录连接故障并两分钟后
重试，进程 PID `2023875`。未重启或干预正在运行的训练/采集。
