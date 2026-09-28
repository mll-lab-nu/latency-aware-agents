# MIKASA InterceptGrabFast H8 teacher baseline 运行记录

本记录保存 H8 baseline 的准备、smoke、协议修正和正式运行时间线。可复用合同与命令见
[MIKASA_ACTION_CHUNK_H8](../MIKASA_ACTION_CHUNK_H8.md)；最终核验结果见
[结果报告](../../outputs/tasks/mikasa_intercept_grab_fast/h8_teacher_baseline_20260920/REPORT.md)
和 [PROGRESS](../PROGRESS.md)。

## 数据与测试

- H1 来源固定为 `benchmark-datasets@5421c9d272d7a939bb04045b047af84519c1fba8`。
- H8 转换审计通过：train `250 episodes / 8,911 anchors / 64,288 valid labels`；
  validation `25 / 948 / 6,884`。
- 初始 H200 StarVLA 环境相关测试为 `71 passed, 1 warning`。扩大相关集曾为
  `135 passed`，另有两个环境差异导致的失败（系统 ffmpeg 不在 PATH；GPU
  device-count 测试假设与该机实际 8 卡冲突）。删除首动作冻结机制后，H8 数据、
  LeRobot 导出、single/batched simulated executor、realtime executor 与 H8 审计
  相关测试为 `124 passed, 11 skipped, 1 deselected, 1 warning`；deselected 是该主机
  PATH 缺少系统 ffmpeg 的视频写入测试，warning 是无 DISPLAY 的 GLFW 提示。

## 2-step smoke 与 GPU 映射

QwenOFT、QwenPI v3、QwenGR00T 均完成两次 optimizer update、H8 `[B,8,7]`
输出、masked loss、反向更新、最终 checkpoint 写入/重载和 MIKASA rollout。
QwenOFT 与 QwenPI v3 的 micro batch 128 可运行；QwenGR00T 在 micro batch 128
出现真实 CUDA OOM，改为 micro batch 64 / gradient accumulation 2 后通过，global
batch 保持 128。

训练阶段分别只暴露物理 GPU 1/2/3。评测时可见卡重排为 `0,训练卡`：MIKASA
simulator 使用逻辑 `cuda:0`（物理 GPU 0），模型 inference 使用逻辑 `cuda:1`。
两次前置尝试分别暴露了非交互 shell 缺少 `CUDA_HOME`、以及 ManiSkill simulator
必须是评测进程逻辑 `cuda:0`；均未进入正式结果。

## 正式训练

工作树和正式输出根目录：

```text
/home/ubuntu/codex/mikasa-h8
/home/ubuntu/codex/mikasa-h8/outputs/starvla/runs/mikasa_intercept_grab_fast/h8-zero-latency
```

三模型均使用单卡、gradient checkpointing、global batch 128、5,000 optimizer
steps 和 `--no-publish`。QwenOFT/QwenPI v3 使用 micro batch 128 / accumulation 1；
QwenGR00T 正式训练在 micro batch 128 真实 OOM 后，以 micro batch 64 / accumulation
2 完成。三路 step-2,500 recovery state、step-5,000 model 和完整 state 均已保存。

W&B runs：

- [QwenOFT](https://wandb.ai/saberrr-zju/starvla_tasks/runs/mikasa_h8-qwenoft-h8-zero-latency-seed0-bootstrap)
- [QwenPI v3](https://wandb.ai/saberrr-zju/starvla_tasks/runs/mikasa_h8-qwenpi_v3-h8-zero-latency-seed0-bootstrap)
- [QwenGR00T](https://wandb.ai/saberrr-zju/starvla_tasks/runs/mikasa_h8-qwengr00t-h8-zero-latency-seed0-bootstrap)

## 评测协议修正

评测最终冻结为两个 simulated setting：

- Zero：0ms，20Hz 每帧 chunk 当帧到达，因此执行最新 chunk 的 head 0；
- Profile：每模型自己的 1×RTX 3090 temporal profile，revision
  `66414463f2a609316da63935941a79abf0a73ce9`，latency seed 0，worker capacity 1。

两者都从 reset 立即开始 60-transition budget。Profile 首动作到达前执行 noop；worker
忙时丢弃新请求，旧 chunk 按 observation age 继续后续 head。任务源码明确说明球在
reset 时发射、没有单独 observation phase，因此曾新增的 `wait_for_first_action`
冻结机制被完整删除，不作为可选生产配置保留。

以下尝试均被最终协议取代：

- H200 wall-clock measured：它测的是当前 H200，而目标是 RTX 3090 profile；OFT
  `9/50`、PI `10/50` 不作为正式结果。
- RTX 3090 profile + 首动作冻结：OFT `10/50`、PI `6/50`；启动时间线被人为重写，
  不作为最终结果。
- Zero 的 OFT `25/50` 不因“只出现 head 0”而作废：0ms 下每帧新 chunk 当帧到达，
  head 0 正是正确执行，现已作为正式 Zero 结果保留。

## 已验证终点

共同协议为 20/20Hz、50 episodes、seeds `4242424242–4242424291`、每局 60 个实际
transitions。六组执行审计全部通过。

| 模型 | Zero success / return ± SD | Profile success / return ± SD | Profile 初始 noop | Profile 后续 heads |
|---|---:|---:|---:|---:|
| QwenOFT | 25/50；19.42654±16.93815 | 1/50；3.97404±4.89508 | 110 | 2,890/3,000 |
| QwenPI v3 | 34/50；26.42206±13.34111 | 14/50；9.93412±14.21984 | 163 | 2,837/3,000 |
| QwenGR00T | 31/50；22.92212±16.49738 | 18/50；15.01258±16.16588 | 111 | 2,889/3,000 |

六组 mean length 均为 60。三个 Zero setting 达到 `>=25/50` gate；三个 Profile
setting 均未达到。未挑 checkpoint、未追加训练。

## Profile / Fixed-2 扩展至 200 episodes

2026-09-22 在第二台 H200 上复用同一组三个 zero-trained H8 step-5,000 inference
bundle，为 Profile 和 Fixed-2 各补跑 150 episodes。原 seeds 为
`4242424242–4242424291`，新增 seeds 为 `4242424292–4242424441`；合并后的 200 个
seed 已核验唯一。Profile 仍使用 revision
`66414463f2a609316da63935941a79abf0a73ce9`、latency seed 0；Fixed-2 仍为严格
100ms / 2 frames。原 50 局与新增 150 局的 execution audit 全部通过，mean length
均为 60。

| 模型 | Profile success；return mean±SD | Fixed-2 success；return mean±SD |
|---|---:|---:|
| QwenOFT | 29/200；7.45255±11.82559 | 33/200；8.84059±12.99718 |
| QwenPI v3 | 26/200；5.12064±10.51648 | 51/200；10.79818±14.93077 |
| QwenGR00T | 36/200；8.32811±12.75548 | 41/200；8.56929±12.49691 |

逐 episode 合并结果、原 50 / 新 150 分段统计与审计见
[扩展结果](../../outputs/tasks/mikasa_intercept_grab_fast/h8_zero_extended_eval_20260922/RESULTS.md)。

## 发布

正式训练最初按计划使用 `--no-publish`。协议修正和六组终评完成后，用户撤销该限制，
三套精简 H8 inference bundle 已发布到 `benchmark-models` 的
`zero-latency/mikasa-intercept-grab-fast/vla/starvla-*-h8/h8-zero-latency-seed0`。
Hub API 在最终 revision `d546ea62d9be8f741f9b52c52a7a2122db1a1b42` 核验三目录
各 8 个文件；OFT/PI/GR00T checkpoint 大小分别为 `9,785,101,729`、
`10,922,637,469`、`9,976,844,067` bytes。对应首次发布 revisions 为
`694f65a745392177bf26ae1668e8626d965780e8`、
`c49a0fffb7a63ea277a03eb4f6e0db8e07e86af9`、
`d546ea62d9be8f741f9b52c52a7a2122db1a1b42`。

启动诊断使用 OFT seed `4242424252` 的正式 action traces 精确重放；Zero/Profile
return 分别与正式记录逐值一致为 `37.4221151448 / 8.3430095404`。Profile 首个 chunk
在 transition 3 到达；此前三个 noop transition 中球沿 y 方向移动约 11.14cm，而 TCP
保持不动。高清关键帧与并排视频位于
`outputs/visualizations/mikasa_h8_startup/`。
