# MIKASA conditioned H8 DAgger 运行记录

## 合同与来源

六路均从 `h8-conditioned-seed-reset-20260922` 对应 SFT run 的
`training/bootstrap/checkpoints/steps_5000_pytorch_model.pt` 初始化新 DAgger 优化器。
初始推理 bundle 对该训练权重使用硬链接；训练期间第 500–4,500 step 的评测与采集
通过 resident rank 直接读取内存权重，不启动第二个 VLA 模型。每路双卡 10 轮、
每轮 500 optimizer steps；每轮先做 100 局评测，再采至少 16,000 个有效 suffix head，
最后续训。第 5,000 步后再做一次 100 局评测及独立 200 局 baseline/best/latest 评测。

逐轮开发评测固定 seeds `4242424442–4242424541`，独立终评为
`4242424542–4242424741`。第 `r` 轮 DAgger 从 seed
`10,000,000 + r×1,000,000` 顺序采集。首轮 bootstrap/current 各占 50%；
其后 bootstrap/current/history 占 40/40/20%，历史各轮等权。
增量数据复用 SFT normalization 与原 validation 集。

| H200 | GPU 0–1 | GPU 2–3 | GPU 4–5 | 模拟器 |
|---|---|---|---|---|
| `34.131.223.74` | Profile OFT | Fixed-2 OFT | Fixed-2 GR00T | GPU 6、7 |
| `34.131.91.104` | Profile PI | Profile GR00T | Fixed-2 PI | GPU 6、7 |

正式输出根为两台机器各自的
`/path/to/user/codex/mikasa-h8/outputs/starvla/runs/mikasa_intercept_grab_fast/h8-conditioned-dagger-20260923/`。
单路入口是 `scripts/tasks/mikasa_intercept_grab_fast/launch_h8_dagger_arm.sh`；
六路启动与恢复入口是 `launch_h8_dagger_six_way.sh`。每路 `controller.pid`、
`controller.log`、`dagger.log`、`evaluations/`、`deltas/` 和 `training/dagger/checkpoints/`
保存在各自目录。训练 W&B 项目为 `placeholder/starvla_tasks`，凭据从远端 `.env` 读取。

## 上线验证与启动

- 两台 H200 的 W&B 凭据均能访问目标项目；第一台与第二台的逐槽 scalar seed reset
  已对齐。第二台补齐了 MIKASA 环境的 `oracle_info[env_idx]` 更新及 LeRobot 打包环境。
- OFT Profile、OFT Fixed-2、PI Profile、GR00T Profile 四组双卡两轮 smoke
  均完成 step-0/2/4 的执行审计、两次 student 数据采集与打包、完整训练状态及
  独立小评测。resident 边界未加载第二个 VLA 进程。首轮原始数据同时保存
  student chunk 与 PPO suffix，prefix 与有效 loss mask 核对通过。
- 相关 H8、resident 和 replay 配比测试通过；重新启动已完成的 OFT smoke 可直接
  回读完成产物，不重复训练。
- 六路正式 step-0 评测均完成 100 局并通过 H8 execution audit；六份首轮
  student 数据均达到至少 16,000 个有效 suffix head，复用 SFT normalization 完成打包。
  六路双卡 trainer 均已从各自 SFT step-5000 模型权重加载并启动，W&B run 均连接
  `placeholder/starvla_tasks`。完整 500-step 边界与终评尚待核验，届时再更新
  `program/PROGRESS.md`。
- Codex task 的 `mikasa-h8-dagger-15` 监控每 15 分钟检查六路状态与可恢复边界，
  仅在故障、恢复行动或完成时通知。

## 首个正式 resident 边界

Fixed-2 OFT 先到 DAgger step 500：`steps_500_pytorch_model.pt` 与两 rank 完整
trainer state 均存在，`step_500.ready` 在 checkpoint 保存后出现。两训练 rank 暂停并
直接处理 100 局评测，没有在训练卡上启动另一个 VLA 进程。评测 5/100、H8 execution
audit 通过，原 SFT step-0 同 seeds 为 7/100；目前不能据此宣称改善。控制器随后
用同一 resident 模型开始第 2 轮 student 采集，待数据打包完成再发布 continue 信号。
