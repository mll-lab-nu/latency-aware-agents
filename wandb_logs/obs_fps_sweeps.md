# Observation FPS Sweeps W&B Figure

This document summarizes the observation frequency sweep.

[W&B report](https://wandb.ai/saberrr-zju/latency-sensitive-bench/reports/obs-fps--VmlldzoxNjkzMTQ3Mg?accessToken=xzvrzatn7rd75jukkq3aae0zb2f7c6v05uy62twe3mcjbma4uqb8tqvq0cj9q8k1)

### Experiment Setup
This experiment studies whether observation FPS affects learning performance under latency settings.

- So far, the sweep has been completed on Demon Attack and Deadly Corridor.
- Each setting was trained and evaluated with 3 different seeds.
- The observation frequency is controlled by `obs_stride = env_fps / obs_fps`. Since different games may have different environment FPS values, the actual observation FPS can vary across games.
- This sweep compares three observation strides: `obs_stride = 1, 2, and 4`. These correspond to observing the environment and running policy inference once every 1, 2, or 4 raw environment frames.

### Training Setting
- Five latency settings are used for training:
    - fixed low latency: l2
    - fixed medium latency: l4
    - fixed high latency: l6
    - latency sampled from normal jitter: N(4, 1)
    - latency sampled from uniform jitter: U(2, 6)

### Eval Setting
- Each checkpoint is evaluated under all latency settings listed above. The main focus is the eval result under the matching training distribution, while other eval settings can also be used as references.
- Eval max steps depends on the environment:
    - Demon Attack: 3600 raw frames.
    - Deadly Corridor: 30000 raw frames, although episodes usually finish within about 200 steps in practice.

### W&B Setting
- Training curves are aggregated with std err and displayed with time-weighted EMA = 0.8.
- Eval curves are aggregated with std err and displayed with running average = 5.

### Key Findings
1. Based on training-time return and survived raw frames, the overall trend is roughly: `obs_stride = 4 > 2 > 1`.
2. During eval, all groups generally survive until max steps, but return does not show a stable pattern.

---

# Observation FPS Sweeps W&B 图表

本文档总结 observation frequency sweep 的实验结果。

[W&B report](https://wandb.ai/saberrr-zju/latency-sensitive-bench/reports/obs-fps--VmlldzoxNjkzMTQ3Mg?accessToken=xzvrzatn7rd75jukkq3aae0zb2f7c6v05uy62twe3mcjbma4uqb8tqvq0cj9q8k1)

### 实验设置
本实验用于探究 observation FPS 是否会影响 latency setting 下的学习表现。

- 目前已在 Demon Attack 和 Deadly Corridor 两个环境中完成该 sweep。
- 每个 setting 都使用 3 个不同 seed 运行了一轮 train + eval。
- Observation frequency 通过 `obs_stride = env_fps / obs_fps` 控制。由于不同游戏的 env FPS 可能不同，实际 obs FPS 在不同游戏中也可能不同。
- 本次对比三组 observation stride：`obs_stride = 1, 2, 4`。它们分别表示每隔 1、2、4 帧 raw env frame 进行一次观测和策略推理，并得到一次 action。

### 训练设置
- 设置五种 latency 环境用于训练：
    - fixed low latency: l2
    - fixed medium latency: l4
    - fixed high latency: l6
    - latency sampled from normal jitter: N(4, 1)
    - latency sampled from uniform jitter: U(2, 6)

### 评估设置
- 同一个 checkpoint 会在上述所有 latency setting 下各 eval 一遍。主要关注与训练分布对应的 eval 结果，其他 eval setting 也可作为参考。
- Eval max steps 按环境设置：
    - Demon Attack: 3600 raw frames。
    - Deadly Corridor: 30000 raw frames；不过实际运行中通常在 200 步以内结束。

### W&B 设置
- Training curve 使用 std err 聚合，并以 time-weighted EMA = 0.8 展示。
- Eval curve 使用 std err 聚合，并以 running average = 5 展示。

### 主要发现
1. 从 training-time return 和 survived raw frames 看，整体趋势基本为：`obs_stride = 4 > 2 > 1`。
2. Eval 阶段各组基本都能 survive 到 max steps；但 return 指标暂未呈现稳定规律。
