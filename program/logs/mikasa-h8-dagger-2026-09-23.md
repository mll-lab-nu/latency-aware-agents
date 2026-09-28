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
`/home/ubuntu/codex/mikasa-h8/outputs/starvla/runs/mikasa_intercept_grab_fast/h8-conditioned-dagger-20260923/`。
单路入口是 `scripts/tasks/mikasa_intercept_grab_fast/launch_h8_dagger_arm.sh`；
六路启动与恢复入口是 `launch_h8_dagger_six_way.sh`。每路 `controller.pid`、
`controller.log`、`dagger.log`、`evaluations/`、`deltas/` 和 `training/dagger/checkpoints/`
保存在各自目录。训练 W&B 项目为 `saberrr-zju/starvla_tasks`，凭据从远端 `.env` 读取。

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
  `saberrr-zju/starvla_tasks`。完整 500-step 边界与终评尚待核验，届时再更新
  `program/PROGRESS.md`。
- Codex task 的 `mikasa-h8-dagger-15` 监控每 15 分钟检查六路状态与可恢复边界，
  仅在故障、恢复行动或完成时通知。

## 首个正式 resident 边界

Fixed-2 OFT 先到 DAgger step 500：`steps_500_pytorch_model.pt` 与两 rank 完整
trainer state 均存在，`step_500.ready` 在 checkpoint 保存后出现。两训练 rank 暂停并
直接处理 100 局评测，没有在训练卡上启动另一个 VLA 进程。评测 5/100、H8 execution
audit 通过，原 SFT step-0 同 seeds 为 7/100；目前不能据此宣称改善。控制器随后
用同一 resident 模型开始第 2 轮 student 采集，待数据打包完成再发布 continue 信号。

## Fixed-2 OFT 十轮完成

Fixed-2 OFT 已完成 5,000 DAgger optimizer steps、11 次各 100 局开发评测、
10 份各不少于 16,000 有效 head 的增量数据，以及完整双 rank step-5000 trainer
state。独立 seeds `4242424542–4242424741` 的 200 局终评：SFT baseline
`10/200`、best/latest `175/200`；mean return 分别为 `2.962` 与 `31.680`，
三项 H8 execution audit 均通过。best checkpoint 为 DAgger step 5000。
其余五路尚未完成，正式跨路汇总与 `program/PROGRESS.md` 更新待六路及 naive
对照全部核验后进行。

Profile OFT 随后也完成同样的 5,000-step、11 次开发评测和 10 份合格增量数据；
双 rank 最终 trainer state 完整。相同独立 200 seeds 下，SFT baseline `12/200`，
best/latest `138/200`，mean return 分别为 `3.348` 和 `25.752`；三项执行审计均通过，
best 为 DAgger step 5000。其余四路 conditioned 和三路 naive 仍在运行。

Profile GR00T 已完成 5,000-step、11 次开发评测、10 份每份至少 16,000 有效 head
的增量数据及完整双 rank trainer state。独立 200 局终评：SFT baseline `13/200`、
step-4000 best `134/200`、step-5000 latest `125/200`；mean return 分别为
`3.744`、`25.235`、`23.500`，三项 execution audit 均通过。

Fixed-2 GR00T 已完成相同训练和审计合同。独立 200 局终评：SFT baseline
`12/200`、step-5000 best/latest `164/200`；mean return 分别为 `3.509` 与
`29.866`。其余两路 conditioned PI 已训完 5,000 step，独立终评仍在进行。

Profile PI 已完成 5,000-step、11 次开发评测、10 份合格增量数据和完整双 rank
trainer state。独立 200 局：SFT baseline `8/200`、step-5000 best/latest
`89/200`，mean return 分别为 `2.176` 和 `17.558`；执行审计通过。
Fixed-2 PI 训练亦到 5,000 step，独立终评仍在进行。

Fixed-2 PI 的独立终评随后完成：SFT baseline `9/200`、step-5000
best/latest `159/200`，mean return 分别为 `3.149` 和 `29.303`。
至此六路 conditioned 均完成 5,000-step、11 次开发评测、10 份合格增量数据、
完整双 rank state 和独立 200 局评测；所有 execution audit 通过。
三路 naive 仍在运行，九路总汇总与 `program/PROGRESS.md` 更新继续等待。

## Hugging Face 模型归档

六路的 SFT 推理 bundle 和 DAgger 选优 bundle，共 12 份，已上传至
`latency-sensitive-bench/benchmark-models`。目录遵循
`<condition>/mikasa-intercept-grab-fast/vla/starvla-<model>-h8/<run-id>/`；
每份包含 `checkpoints/model.pt`、模型配置、数据统计、任务合同、来源说明和 README。
DAgger bundle 另含独立终评记录。Profile GR00T 的 best 为 step 4000；同一发布
目录的 `checkpoints/latest_model.pt` 保存 step 5000。每份提交的完整 revision
保存在对应本地 run 的 `publication_hf_sft.json` 或 `publication_hf_dagger.json`，
发布后已逐文件核对 Hub 路径与大小。

| 条件/模型 | SFT revision | DAgger revision |
|---|---|---|
| Profile/OFT | `fbc839de9f51360fbf4ee104f279fe122d4aa77b` | `fd38211d9f2a7f49d46124d6a60a6ee93d2e9e36` |
| Profile/PI | `0e7215d09ac6d33cfab7a7e8ab413b04dd407791` | `bb4a5dcd32437018e96854c4d6c82e08a80568d1` |
| Profile/GR00T | `940b5b54d51c4b762b48dd4cad8b4175da521eee` | `80f385349249d4b69281d7e61ca2bc1b5413f70c` |
| Fixed-2/OFT | `9b8f3c799cd2a31739c9da613364aeab354c917c` | `795468e53947d26314e891cbc375775c6e0bd491` |
| Fixed-2/PI | `32c9c532dfbb129ab3b595da3d0e31b0277c3ab8` | `534218d43ae24f183171a963e5128a9c8c5b6b85` |
| Fixed-2/GR00T | `2f81640270b3573caad7a0d6a7d10841e000ea45` | `17d2a224c92c31b2ed141281b89ad85abc0bea1a` |

## 十轮 student/PPO suffix 误差与最终核验

`student_suffix_mae` 在每轮 `collection.json` 中分别统计首请求和后续请求。
下表把十轮各自的 mean 按请求数 `n` 加权；是有效 suffix 上的动作 MAE，
不是终评成功率，也不跨模型直接比较绝对大小。

| 条件/模型 | 首请求 MAE（n） | 后续请求 MAE（n） |
|---|---:|---:|
| Profile/OFT | 0.05545（1,649） | 0.17997（31,769） |
| Profile/PI v3 | 0.07954（2,549） | 0.25233（38,733） |
| Profile/GR00T | 0.05921（1,602） | 0.18898（31,813） |
| Fixed-2/OFT | 0.05591（1,283） | 0.18206（26,704） |
| Fixed-2/PI v3 | 0.05936（1,181） | 0.20767（27,032） |
| Fixed-2/GR00T | 0.05233（1,306） | 0.17562（26,765） |

发布后由 Hugging Face API 对六份 SFT 和六份 DAgger 的 `main` 路径逐文件
比对本地 stage 大小，12 份全部通过。六路 `result.json`、每轮 100 局审计、
10 份有效标签、完整双 rank trainer state 和独立 200 局 seeds 均按发布脚本的
完整性检查通过。正式结果和后续决策已写入 [PROGRESS](../PROGRESS.md)。
