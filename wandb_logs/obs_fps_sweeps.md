# obs fps sweeps wandb figure

This doc is for observation frequency sweep

[wandb report](https://wandb.ai/saberrr-zju/latency-sensitive-bench/reports/obs-fps--VmlldzoxNjkzMTQ3Mg?accessToken=xzvrzatn7rd75jukkq3aae0zb2f7c6v05uy62twe3mcjbma4uqb8tqvq0cj9q8k1)


### 实验设置
本实验用于探究 obs fps 是否会影响 latency setting 下的学习表现。

- 目前已在 Demon Attack 和 Deadly Corridor 两个环境中完成该 sweep。
- 每个 setting 都使用 3 个不同 seed 运行了一轮 train + eval。
- Frame stack 指将当前 observation frame 与前面连续的 N 帧 raw env frame 拼接后，一起输入模型。
- 本次对比三组 obs fps，不同游戏具体数值可能不同，分别是 env_fps / obs_fps = 1, 2, 4，意味着每隔 1/2/4 帧观测&推理得到action一次

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
