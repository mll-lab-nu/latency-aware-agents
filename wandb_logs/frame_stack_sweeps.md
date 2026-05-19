# Frame Stack Sweeps W&B Figure

This document summarizes the frame stack factor sweep.

[W&B report](https://wandb.ai/saberrr-zju/latency-sensitive-bench/reports/frame-stack--VmlldzoxNjkwNjU0Nw?accessToken=4kj8p5ioe95cue48pc9evxt2b68ym6vov7dxfn6pp4yircngpmc3myx1zmdl6q4s)

### Experiment Setup
This experiment studies whether frame stack affects learning performance under latency settings.

- So far, the sweep has been completed on Demon Attack and Deadly Corridor.
- Each setting was trained and evaluated with 3 different seeds.
- Frame stack means concatenating the current observation frame with the previous N consecutive raw environment frames, then feeding them into the model together.
- This sweep compares three frame stack values: fs=4, 6, and 8.

### Training Setting
- Five latency settings are used for training:
    - fixed low latency: l2
    - fixed medium latency: l4
    - fixed high latency: l6
    - latency sampled from normal jitter: N(4, 1)
    - latency sampled from uniform jitter: U(2, 6)

### Eval Setting
- Each checkpoint is evaluated under all latency settings listed above. The main focus is the eval result under the matching training distribution, while other eval settings can also be used as references.
- eval max steps = 3600 raw frames.

### W&B Setting
- Training curves are aggregated with std err and displayed with time-weighted EMA = 0.8.
- Eval curves are aggregated with std err and displayed with running average = 5.

### Key Findings
1. Based on training-time return and survived raw frames, the overall trend is roughly: fs=8 > fs=6 > fs=4.
2. During eval, all groups generally survive until max steps. However, return does not show a stable pattern, and fs=4 is often higher. This may be affected by the max steps limit.

---

# Frame Stack Sweeps W&B 图表

本文档总结 frame stack factor sweep 的实验结果。

[W&B report](https://wandb.ai/saberrr-zju/latency-sensitive-bench/reports/frame-stack--VmlldzoxNjkwNjU0Nw?accessToken=4kj8p5ioe95cue48pc9evxt2b68ym6vov7dxfn6pp4yircngpmc3myx1zmdl6q4s)

### 实验设置
本实验用于探究 frame stack 是否会影响 latency setting 下的学习表现。

- 目前已在 Demon Attack 和 Deadly Corridor 两个环境中完成该 sweep。
- 每个 setting 都使用 3 个不同 seed 运行了一轮 train + eval。
- Frame stack 指将当前 observation frame 与前面连续的 N 帧 raw env frame 拼接后，一起输入模型。
- 本次对比三组 frame stack：fs=4、6、8。

### 训练设置
- 设置五种 latency 环境用于训练：
    - fixed low latency: l2
    - fixed medium latency: l4
    - fixed high latency: l6
    - latency sampled from normal jitter: N(4, 1)
    - latency sampled from uniform jitter: U(2, 6)

### 评估设置
- 同一个 checkpoint 会在上述所有 latency setting 下各 eval 一遍。主要关注与训练分布对应的 eval 结果，其他 eval setting 也可作为参考。
- eval max steps = 3600 raw frames。

### W&B 设置
- Training curve 使用 std err 聚合，并以 time-weighted EMA = 0.8 展示。
- Eval curve 使用 std err 聚合，并以 running average = 5 展示。

### 主要发现
1. 从 training-time return 和 survived raw frames 看，整体趋势基本为：fs=8 > fs=6 > fs=4。
2. Eval 阶段各组基本都能 survive 到 max steps；但 return 指标暂未呈现稳定规律，且 fs=4 普遍更高。推测这可能受到 max steps 上限的影响。
