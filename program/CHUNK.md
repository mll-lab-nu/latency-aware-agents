# FastTD3 Action Chunk：修复与实验记录

更新：2026-09-13。

专用 milestone、update 诊断快照及精确停止脚手架已移除；普通 periodic/final 评估保留。下面的精确步数和诊断属于历史实验，第一版消融结果与通用 D/E 用法见 [CHUNK_RL_ABLATION.md](CHUNK_RL_ABLATION.md)。

**本轮修复有效：精确调度窗口、因果 context、保留旧 critic 初始化，恢复了 profile 可变延迟下的 action chunk 学习。**

在相同 QwenOFT temporal profile、训练 seed 1、5,000,064 raw steps、25 个环境 seeds × 4 个 latency seeds 下：

| 指标 | 修复前 `profile_h8_s1_20260913T0403Z` | 修复后 `profile_h8_repair_s1_20260913T0638Z` |
|---|---:|---:|
| 平均存活帧数 | 72.88 | **172.14（2.36 倍）** |
| 平均回报 | 59.45 | **150.37** |
| ≥100 帧比例 | 3% | **93%** |

100 个配对场景的存活长度和回报均提升。修复针对 Q／target 的动作窗口、actor context 和 critic 初始化整体生效；仅截断额外 head 梯度的 first-two 诊断未能恢复学习。

[修复前 5M 评估](../outputs/tasks/humanoidbench_balance_simple/action_chunk_h8_6m/profile_zero_20260913/profile_h8_s1_20260913T0403Z/matched_5000064.json) · [修复后 5M 评估](../outputs/tasks/humanoidbench_balance_simple/action_chunk_h8_6m/profile_repair_20260913/profile_h8_repair_s1_20260913T0638Z/matched_5000064.json) · [修复训练 W&B](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/profile_h8_repair_s1_20260913T0638Z)。

## 移除自定义限幅后的对照

固定 40ms、50Hz、seed 1，精确 6,000,000 raw steps，相同 25 个评估场景：

| 方法 | 平均存活帧数 | ≥100 帧比例 |
|---|---:|---:|
| H8，保留自定义限幅 | 83.40 | 20% |
| H8，移除自定义限幅 | 244.80 | 100% |
| 原生 H1 FastTD3，移除自定义限幅 | 39.84 | 0% |

H8 移除限幅后在 5M 达到 471.60 帧，精确 6M 回退至 244.80；最终 6,000,384 steps 为 302.56。移除限幅的零延迟 H8 在精确 6M 为 335.32，最终 6,000,384 steps 为 365.92。

[H8 保留限幅](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/fixed40_custom_limits_on_s1_20260912T1615Z) · [H8 移除限幅](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/fixed40_custom_limits_off_s1_20260912T1615Z) · [原生 H1](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/fixed40_custom_limits_off_h1_vanilla_s1_20260912T1708Z) · [视频、关节曲线与频谱](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/fixed40_limit_visuals_6m_s1_20260912T1715Z)。

限幅确实降低了高频运动，但也严重改变 PD 目标。三个相同场景的前 1 秒，实际关节 >10Hz 速度 RMS 为保留限幅 0.47、移除限幅 3.40 rad/s；两组为分别训练的策略。上述训练通过运行时替换移除自定义限制，上游 XML 物理边界保持原样。

## Profile 扩展与实验范围

**当前动态 chunk／mask 仅支持 `simulated_worker_capacity=1`。多 worker 的并发、返回乱序与窗口 credit 尚未实现和验证。** Capacity 来自实验配置；当前 QwenOFT profile 本身也是单 worker。

H=8、43 维动作。chunk 在 `s` 发出、`e` 首次执行、`n` 被下一 chunk 接管，则 critic 的计划窗口为 `[e-s, n-s)`。在首次执行时复制 sampler 和调度状态，只推演后续 admission、service、ready、latest-ready；复用现有 timeline 和 scheduler，不推进物理环境或真实 RNG。窗口缓存在当前 event，提前终止不截掉后缀。hold-tail 将超出 H 的次数累加到末 head。

| 当前 ready/service | 下一 ready | 计划 head |
|---|---|---|
| 2/2 | 2 | 2/3 |
| 2/2 | 3 | 2/3/4 |
| 3/3 | 2 | 3/4 |
| 3/3 | 3 | 3/4/5 |
| 3/2 | 2 | 3 |

Actor 保持 550D，只接收已返回 chunk 中、最早可能交接前确定会执行的前缀。最早交接使用 profile ready 下界和已提交请求的 source clocks；不读取未返回请求的 ready/service 抽样。训练、simulated、batched、pooled、realtime 使用同一裁剪函数。计划窗口仅进入 replay，不能进入 actor context。

Critic 为 execution-start 198D + 8 个计划次数。先按旧 198D 初始化 Q，再在物理状态与 action 之间插入 8 个零权重列；旧首层权重、bias、后续层及随机序列保持一致。当前 Q、bootstrap Q 和 actor loss 各用对应的精确窗口。真实 `execution_counts` 仍负责 reward、`gamma^duration` 和统计；timeout 的未完成片段仍 censor。零延迟的 Q 为 head 0、context 为空；fixed40 退化为原来的 head 2/3。

本轮 319 项相关检查和 32,768 步联调通过；真实 temporal sampler 的 1,950 次交接与预测窗口全部一致，真实 RNG／burst 状态保持一致，2,038 次隐藏时序扰动不改变 actor context。初始 checkpoint 与旧成功 run 逐项对齐，新增 critic 列全零。[验证记录](../outputs/tasks/humanoidbench_balance_simple/action_chunk_h8_6m/profile_repair_20260913/VALIDATION.md)。

### 升级前后的历史实验

四组均移除自定义限幅，从零训练 6M、seed 1，W&B online；profile 和零延迟各比较完整 H8 方法与原生 H1。H1 使用 198D actor、43D 输出、原生物理帧 replay，关闭 fixed1/profile-value/history 改动。两者共享 128 个独立训练环境（每环境推理 capacity=1）、50Hz、batch 32768、replay 1M、优化器、学习率和每 raw-frame 更新预算。H8 的执行聚合、mask/credit 与 chunk 内 AR(1) 探索属于完整方法。

原 profile 上界窗口实现的四组 32,768-step smoke 均完成，actor／critic／target 有更新且参数有限；207 项主回归、18 项 profile 后端检查和 73 项 context／policy 检查通过。通用 simulated／realtime 的 context 生产者也已传递可见 service timing，由同一窗口函数裁剪输入。

正式 profile 两组先启动；为减少 GPU 空闲，每条训练任务结束后接续 zero 对照，最多两组训练并行。

- [Profile H8](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/profile_h8_s1_20260913T0403Z)（已完成精确 6M，配对平均存活 84.11 帧）
- [Profile 原生 H1](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/profile_h1_native_s1_20260913T0403Z)（6M 训练和最终评估已完成，平均存活 35.97 帧）
- [Zero 原生 H1](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/zero_h1_native_s1_20260913)
- [Zero H8](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/zero_h8_s1_20260913)

Profile 精确 6M 匹配评估（相同 25 个环境 seeds × 4 个 latency seeds）：H8 平均存活 **84.11** 帧，原生 H1 **35.97** 帧。首个 1,000,064-step 匹配评估分别为 39.44／19.58 帧。这是修复前的历史结果；本轮修复后的学习提升见文首 5M 配对对照。

结果根目录：`outputs/tasks/humanoidbench_balance_simple/action_chunk_h8_6m/profile_zero_20260913/`。

## Profile 新增 head 梯度的单变量诊断

同一步数的 periodic 平均存活，旧 fixed40 在 2M／3M 为 120.00／164.75 帧，新 profile 为 40.69／58.25 帧。Profile sampler 抽样中约 96.3% 的请求仍是 ready=2、service=2；优化器、batch、replay 与更新预算一致。当时优先检验从固定两格推广到所有可能 head 后的信用分配。

Profile 1M 的真实 replay 中，head 4 在 100% 的样本开放 Q 梯度，只有 6.05% 的片段实际执行。它也会通过旧 chunk context 影响下一次 actor，因此执行频率与梯度幅度不匹配还不足以单独确认根因。

历史 first-two 诊断只截断额外 head 的 actor Q 梯度，未修改 Q/target/context。1M 配对均值 40.87（完整窗口梯度为 39.44），2M periodic 为 50.72；没有支持把修复局限于额外梯度。该诊断的程序开关、配置和专属测试已移除，历史输出继续保留。[诊断记录](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/profile_h8_first_two_s1_20260913T0500Z) · [历史探针结果](../outputs/tasks/humanoidbench_balance_simple/action_chunk_h8_6m/profile_zero_20260913/first_two_probe.json)。

旧精确 6M 权重在现 profile 的相同 25 环境 seeds × 4 latency seeds 下平均存活 198.03，原 profile H8 为 84.11。旧权重确实执行了 2/3 帧延迟；这说明现 profile 执行路径本身可以运行有效的 chunk 策略。修复版从零训练 seed 1、精确 6M，保留 1M、2M、5M、6M checkpoints；沿用原来的 periodic 和 final 配对评估场景，不增加消融矩阵。

以下保留 R3 历史修复与实验记录。

## 本轮 R3 修复方案（已实施）

### 1. 修正方差合并公式

位置：[models.py](../third_party/sample-factory/sample_factory/algo/fast_td3/models.py)，`EmpiricalNormalization.update()`。

原实现使用更新均值后的 `delta2`，低估批次间的方差；改用更新均值前的 `delta`，删除 `delta2`：

```text
delta = batch_mean - old_mean
M2 = old_var * n + batch_var * m + delta² * n * m / (n + m)
new_var = M2 / (n + m)
```

验证：两批 `[0,0]`、`[2,2]` 的合并方差由错误的 0.25 修正为 1。

### 2. 修正 actor 梯度与 L2 的 mask

位置：[learner.py](../third_party/sample-factory/sample_factory/algo/fast_td3/learner.py)，`_actor_step()`。

旧行为在 a₂ 后摔倒，不代表当前策略也执行不到 a₃。删除 `execution_counts` 对当前 actor 的梯度门控，Q 输入与动作 L2 统一使用已有 `planned_mask`：

```text
actor_q_action = actor(issue_obs) * planned_mask
actor_loss = -Q(execution_state, actor_q_action).mean() + coefficient * ||actor_q_action||²
```

Zero 仅 head 0、fixed-2 仅 head 2/3 获得梯度。删除 actor 的 counts 参数；counts 仍用于 reward、折扣和执行统计，`actor_credit_fraction` 改为计划窗口比例。

验证：4 个真实 replay 反例中，a₃ 输出层梯度范数由 0 恢复为 2.64154，计划外梯度仍为 0。[反例数据](../outputs/tasks/humanoidbench_balance_simple/action_chunk_h8_6m/r3_validation_20260912/actor_counterfactual.json)

### 3. 将突变制动改为连续安全限速

位置：[humanoidbench_sonic.py](../latency_bench/envs/humanoidbench_sonic.py)，`_condition_velocity()`。

旧制动分支会把 float32 observation 丢失的约 1e-8 控制状态尾差，放大为最大速度的 20%～40% 跳变。删除旧 `_braking_distance()` 和整档制动切换，改为离散停止距离的连续逆映射：

```text
delta = a_max * dt
d = 到对应关节边界的有符号距离
N = max(1, ceil((sqrt(1 + 8 * max(d, 0) / (dt * delta)) - 1) / 2))
safe_velocity = d / (dt * N) + delta * (N - 1) / 2
```

请求速度先受原速度、加速度限制，再投影到两侧安全速度区间，最后 `q_next = q + v * dt`。保留原关节范围、0.1 秒 ramp、PD、reset 和未启用加速度限制的路径；不新增配置。

验证：同 observation 的真实反例中，下一 observation 差从 1.495 降到 2.38e-7，另一例从 0.2 降到 0.000251；速度、加速度、位置约束通过。[反例数据](../outputs/tasks/humanoidbench_balance_simple/action_chunk_h8_6m/r3_validation_20260912/controller_counterfactuals.json)

## 实验结果

R3 将三项修复一起应用，zero／fixed-2 各从零训练 6M raw steps。参数保持 seed 1、H=8、50Hz、128 workers、replay 1M、batch 32768、gamma 0.99；不使用 teacher 或旧权重。

两组已完成 6,000,384 steps；25 场景最终平均存活为 zero 328.72、fixed 104.96。

### Periodic 平均存活

同一组 8 个评估 seeds，单位 raw frames；`—` 表示本次记录没有该结果。

| Raw steps | Zero r2 | Zero r3 | Fixed-2 r2 | Fixed-2 r3 |
|---:|---:|---:|---:|---:|
| 1,000,064 | 80.25 | 76.375 | 47.125 | 54.375 |
| 2,000,000 | 194.125 | 218 | 71 | 65.125 |
| 3,000,064 | 301.625 | 201 | 86.875 | 66 |
| 4,000,000 | 414 | 265.875 | 90.875 | 93.625 |
| 5,000,064 | 145.875 | — | 117.75 | 103 |
| 6,000,000 | 407.875 | — | 102.375 | — |

在线曲线也仍有问题：zero 的 100-episode 平均存活曾由 296.36 回落到 202.13；fixed 在 3M～4.77M 区间约为 63～79 帧。

### 历史 final 与控制器对照

以下为 25 场景平均存活：

| 条件 | Zero | Fixed-2 |
|---|---:|---:|
| R1 final | 150.88 | 106.20 |
| R2 final | 467.04 | 114.00 |
| R2 final 权重 + R3 控制器 | 460.20 | 108.28 |

仅替换控制器没有明显收益。R2 已修过旧 chunk context 和 critic mask；R3 再修上述三处，主要学习瓶颈仍未定位。

## R3 历史 chunk 语义

- Actor 输入 `198 + 344 + 8 = 550D`，输出 `8 × 43 = 344D`；critic 使用 execution-start 198D 状态和计划 mask 后的 344D chunk。
- Observation-aligned、`latest_ready`、worker capacity 1、hold-tail。Zero 每帧执行新 chunk 的 head 0；fixed 40ms 每两帧接纳一次，执行 head 2/3。
- Actor 的旧 chunk context：zero 清空，fixed 仅保留 head 2/3 和 index。只归一化物理 observation。
- Replay 按真实连续执行窗口聚合，切换来源时立即发布；termination 不 bootstrap，timeout 最后一段 censor。更新频率仍按 raw frames 计算。
- 探索和 target smoothing 使用 chunk 内 AR(1) 噪声，rho=0.9；H=1 保留原路径。

```text
R = sum(gamma^k * reward_k), k = 0 ... duration-1
y = R + (1-done) * gamma^duration
      * Q_target(next_execution_state, target_actor(next_issue_obs) * planned_mask)
```

## 历史验证与归档

- G1 环境 77 项、FastTD3 39 项、相关定向检查 29 项通过；两组 32K smoke 完成，参数 finite 且 actor/critic/target 均更新。
- 已有测试问题：oracle 测试导入缺失的 `OracleTransition`；一项 PPO fixture 缺 `fasttd3_action_chunk_horizon`，修改前也失败。
- [本轮代码差分](../outputs/tasks/humanoidbench_balance_simple/action_chunk_h8_6m/r3_validation_20260912/implementation.patch)、[smoke 验证](../outputs/tasks/humanoidbench_balance_simple/action_chunk_h8_6m/r3_validation_20260912/smoke_checkpoints.json)、[历史明细归档](../outputs/tasks/humanoidbench_balance_simple/action_chunk_h8_6m/r3_validation_20260912/CHUNK_before_condense.md)。

| 条件 | GPU | 结果目录／W&B | tmux |
|---|---:|---|---|
| zero | 0 | [结果](../outputs/tasks/humanoidbench_balance_simple/action_chunk_h8_6m/zero_fix_s1_20260912_r3/) · [W&B](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/zero_fix_s1_20260912_r3) | `action_chunk_zero_6m_20260912_r3` |
| fixed-2 | 1 | [结果](../outputs/tasks/humanoidbench_balance_simple/action_chunk_h8_6m/fixed40_fix_s1_20260912_r3/) · [W&B](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/fixed40_fix_s1_20260912_r3) | `action_chunk_fixed40_6m_20260912_r3` |

结果目录保存 `resolved_config.json`、`progress.jsonl`、`periodic/`、`matched_*.json`、`final.pth/json` 和 `old_final_new_controller.json`；日志在父目录的同名 `.log`。Periodic 每 1M 评估 8 场景，final 评估 25 场景，每场上限 1000 帧；recovery gate 为平均存活 ≥300 且 ≥100 比例 ≥0.8。Fixed 的 zero-transfer 使用未训练的 head 0，不作验收指标。

当前配置启动方式和后续 D/E 单独消融计划见 [CHUNK_RL_ABLATION.md](CHUNK_RL_ABLATION.md)。
