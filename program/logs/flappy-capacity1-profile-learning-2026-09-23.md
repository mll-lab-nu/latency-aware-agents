# Flappy capacity=1 profile APPO：评测修复与学习方法（2026-09-23）

本轮固定 RTX 3090 temporal profile、10/10 FPS、seed 42、`simulated_worker_capacity=1`、`one_frame_then_noop/latest_ready`、APPO 与 18,000 帧 deterministic 评测。方法选择使用相同的 100 个 episode seeds（104771–104870）；原始逐局数据与 checkpoint 路径见 [baseline_summary.json](../../outputs/tasks/flappy/capacity1_method_search/baseline_summary.json)。Pi 本轮不训练。

## 先前低评测分数的原因

训练时的四帧图像堆叠随每个原始环境帧更新；批量评测器却只在 observation 被 worker 接纳时更新。GR00T 在 capacity=1 下约半数 observation 被丢弃，因而训练与评测输入错位。使用同一个旧 GR00T 20,021,248-step checkpoint、同 100 个 seeds 与 profile，修复前 mean return 4.388，修复后 243.420；配对 bootstrap 95% 区间对应增益 [191.294, 292.376]。旧的个位数评测不能作为训练失败证据。标量与批量评测的帧序列定向测试已通过。

旧 GR00T run 曾从 10M 断点续训。20M checkpoint 的学习率仍约 `1.50e-4`，不是原计划线性日程应有的约 `5e-5`。原因是 Sample Factory 的线性调度器在载入 checkpoint 后从零重新计数。已用 checkpoint 的 optimizer step 初始化调度器，并验证连续训练与恢复后的下一次学习率一致；旧 GR00T run 只作诊断，不作方法对照。

## 修正后的固定场景基线

| 模型与 checkpoint | 100 局 mean return | 训练状态 |
|---|---:|---|
| OFT 15,007,744 | 1358.986 | 旧 uninterrupted capacity=1 训练；按修正评测器重评 |
| OFT 20,021,248 | 1144.829 | 同上 |
| OFT 25,001,984 / best | 1310.971 | 同一 checkpoint；25M−20M 配对 95% 区间 [-11.685, 341.933]，不能证实后期回退 |
| GR00T 5,013,504 | 127.434 | 全新 seed-42 uninterrupted 训练 |
| GR00T 10,027,008 | 419.782 | 同上 |
| GR00T 15,007,744 | 502.558 | 同上 |
| GR00T 20,021,248 | 1091.034 | 同上；checkpoint 学习率 `4.977e-5` |
| GR00T 25,001,984 | 1636.401 | 同上 |
| GR00T best 24,805,376 | 1727.141 | 独立正式评测 |

新 GR00T [W&B run](https://wandb.ai/saberrr-zju/starvla_tasks/runs/flappy-obs10-qwengr00t_rtx3090_temporal_25m_capacity1_clean_s42_v1-s42) 与上述逐局数据已完成；这条基线到 25M 仍在增长。GR00T profile JSON SHA256 为 `0654f7097e292cc837a483ef8e5f7ed9c3410fd684582b31b7d8358719452b60`；OFT profile 为 `c266cba7ebac05c7e37bc83f1f16fe4b27dab97942ff43290e6c41514f6e39fc`。

## 判据的物理上限

管道横向运动不受动作与随机高度影响：初始三根管的横坐标及每次回收位置固定，每原始帧左移 4 像素。按环境现行 `score` 判定，在 18,000 帧内最多经过 477 根管。每帧存活奖励为 `0.1`，过管帧为 `1.0`，所以任何策略的单局 return 至多 `18,000×0.1 + 477×0.9 = 2229.3`。25M 基线 100 局有 50 局跑满 horizon；best 有 58 局跑满，均达到 477 分。原计划的 GR00T 25M 翻倍目标为 `3272.802`，高于上限，因此不可能达标。用户随后要求停止额外对照，本轮按既定评测协议报告实际分数与上限。[基线与上限图](../../outputs/tasks/flappy/capacity1_method_search/baseline_vs_ceiling.png)展示了这一冲突。

## 方法实验

修复了 `training.lr_schedule` 传递，并加入 OFT 的线性衰减下限 `5e-5`。[OFT 学习率下限 run](https://wandb.ai/saberrr-zju/starvla_tasks/runs/flappy-obs10-qwenoft_rtx3090_temporal_25m_capacity1_lr_floor_s42_v1-s42) 从零训练至 25,001,984 steps；20M checkpoint 的学习率 `5.014e-5` 与旧基线的 `5.006e-5` 接近，25M 保持 `5e-5`。同 100 局下，20M mean return 为 1164.722（基线 1144.829），25M 与正式 best 同一模型 mean 1287.345（基线 1310.971），25M 配对增益 −23.626，bootstrap 95% 区间 [−241.343, 195.012]；25M−20M 的配对区间 [−88.139, 335.782]。该下限未达到 +20% 目标，且无法从这一训练 seed 证明有收益。

GPU 1 的 [GR00T actor 掩码 run](https://wandb.ai/saberrr-zju/starvla_tasks/runs/flappy-obs10-qwengr00t_rtx3090_temporal_25m_capacity1_actor_mask_s42_v1-s42) 单独改变 actor、entropy、KL 与 actor advantage 归一化的样本集合，仍保留完整 GAE/critic 序列。20M mean 1045.399（基线 1091.034），配对区间 [−267.436, 181.788]；25M mean 1365.060（基线 1636.401），配对增益 −271.341、区间 [−476.360, −61.873]；正式 best mean 1265.458。前期 5M、10M mean 264.417、511.876，虽高于基线 127.434、419.782，但后段明显变差。末期 `train/entropy` 为 0.034，干净基线为 0.252；发出 flap 比例约 0.261 与 0.366。两者 observation drop 都约 50%，policy lag 末期约 73 与 72.9。这种熵收缩与表现恶化同时发生，尚不能断定因果。该单变量否定了掩码作为当前 APPO 配方下的加速方法。

GPU 0 随后启动 [GR00T 25D 动作历史 run](https://wandb.ai/saberrr-zju/starvla_tasks/runs/flappy-obs10-qwengr00t_rtx3090_temporal_25m_capacity1_history_s42_v1-s42)，与掩码臂彼此独立。OFT 旧基线与下限 run 的末期训练 entropy 分别约 0.053、0.055，均接近确定性；GPU 1 启动 [OFT entropy 系数 0.01 run](https://wandb.ai/saberrr-zju/starvla_tasks/runs/flappy-obs10-qwenoft_rtx3090_temporal_25m_capacity1_entropy01_s42_v1-s42)，仅改变默认系数 0.003。各 run 的 checkpoint、定期评测和正式 best 评测均使用独立目录与 W&B ID。

actor 掩码的损失、rollout 记录与 GPU admission 测试，以及 25D 八帧因果历史的 CPU/CUDA 序列测试已通过。两项各自的 98,304-step GPU 训练联调完成，历史输入 checkpoint 能被独立评测器加载；短跑产物已清理。恒定学习率单变量配置未运行。用户停止广泛对照后，单独要求检验 worker 忙碌位，因此只新增这一项实验。

GR00T 历史输入的 25M 训练与正式 best 评测已完成。同一开发集 100 局中，5/10/15/20/25M mean return 分别为 1781.386 / 1862.363 / 2136.200 / 1955.151 / 2105.671；跑满 18k 帧的局数为 66 / 72 / 91 / 80 / 90。20M 相比干净基线 +864.117、配对 95% 区间 [648.015, 1074.462]；25M +469.270、区间 [315.114, 626.619]。15M→20M 曾显著回退，说明 checkpoint 选择仍重要；当时的训练 entropy 约 0.233→0.232、KL 约 0.0046→0.0030，未见大幅更新信号。

独立 holdout（104871–104970）已完成：历史模型 20M/25M mean 1963.723 / 2132.132，干净基线 1126.173 / 1786.849；配对增益区间分别为 [629.841, 1049.447]、[205.043, 489.219]。25M 历史模型 92/100 局跑满 18k 帧。性能提升已跨两组评测场景重现，但未做全零输入或新增训练 seed 对照，不能把因果机制或训练 seed 稳定性称为已证明。[逐局报告](../../outputs/tasks/flappy/capacity1_method_search/gr00t_history_holdout_comparison.json)。

25D 输入按最近八个原始帧的“观测接纳、接纳的 flap 命令、实际执行 flap”三位历史排列，再附一个 worker 忙碌位。忙碌位表示前一请求的 worker service 尚未完成，训练端每个决策帧都输入 0/1；评测端只有 worker 可接纳时才调用策略，所以评测策略实际看到的忙碌位基本恒为 0。性能提升更可能来自前 24 位历史或额外输入网络参数，不能归功于忙碌位本身。

按用户的单项对照要求，[GR00T 历史忙碌位恒零 run](https://wandb.ai/saberrr-zju/starvla_tasks/runs/flappy-obs10-qwengr00t_rtx3090_temporal_25m_capacity1_history_no_busy_s42_v1-s42) 保留相同 25D 网络输入及前 24 位历史，只把最后一位在训练和评测固定为 0。新配置的 10/10 FPS、capacity=1、profile、seed 42、APPO 与 25M 学习率日程已核对。CPU/CUDA 历史时序、CLI 映射和短 GPU 训练/评测通过；该 run 后来约停在 16.25M steps，没有正式 25M 评测。新一轮使用去掉忙碌位的 24D 决策历史，见[九条训练记录](decision-action-history-nine-profile-2026-09-23.md)。

[OFT entropy 系数 0.01 run](https://wandb.ai/saberrr-zju/starvla_tasks/runs/flappy-obs10-qwenoft_rtx3090_temporal_25m_capacity1_entropy01_s42_v1-s42) 的 25M 与正式 best mean 1323.350，基线 1310.971；配对增益 +12.379，区间 [−185.362, 211.754]，未达到 +20% 目标。

旧 25D history 与 busy-bit 对照的配置原文保存在[历史配方归档](../archive/flappy-capacity1-20260923/README.md)；当前训练代码已不消费这两个旧开关，不能直接使用它们复现旧实验。
