# Latency-sensitive VLA 研究总方针

> 当前实现与实验命令见 [`program/MIKASA.md`](MIKASA.md)。本文是通过当前
> zero-delay baseline 后再执行的 latency proposal，不是本次 baseline 的实现范围。

## 研究问题

研究目标不是重新发明一个 manipulation baseline，而是分阶段回答：

1. 成熟 VLA benchmark 接入统一 latency protocol 后会怎样退化；
2. 真实模型 latency 的时序结构如何影响 closed-loop performance；
3. 不给 VLA 显式 latency/queue 输入时，能否用 delayed closed-loop RL teacher
   蒸馏出 latency-aware VLA；
4. latency robustness 会带来多少 zero-latency regression。

baseline solvability、latency degradation 和 latency-aware recovery 必须分阶段证明，不能
用一次长训练把三个问题混在一起。

## 当前 baseline

第一阶段只冻结并验证 MIKASA `InterceptGrabFast-VLA-v0` 的 zero-delay H1 路线：

```text
MIKASA state PPO teacher
  -> 275 successful trajectories
  -> 250 train / 25 validation LeRobot v3 datasets
  -> StarVLA QwenOFT H1 SFT
  -> zero-delay PPO/VLA evaluation
```

当前 baseline 使用官方 MIKASA/ManiSkill control cadence：20 Hz，即每个 control step 50 ms。
Teacher 和 VLA 都输出一个 7D `pd_ee_delta_pose` action；VLA 输入是 top RGB、wrist RGB 和
固定 task instruction，不输入 state、latency 或 queue。

当前 baseline gate：

- PPO：官方 trainer gate 通过，并在固定 canonical seeds 上完成 100 episodes，至少 95 次
  `success_once`；
- dataset：恰好 275 条成功轨迹，250 条 train、25 条 validation；
- VLA：5,000 optimizer steps、global batch 64，canonical zero-delay 50 episodes 至少 25 次
  `success_once`。

在这些 gate 全部通过前，不启动 latency sweep、profile 或 delayed teacher。

## 后续 latency proposal

### 1. 固定 latency degradation

baseline 通过后，在同一 canonical seeds 上评估固定 action delay：

```text
d ∈ {0, 1, 2, 4, 8} control steps
  = {0, 50, 100, 200, 400} ms
```

固定 delay 的语义是：observation 在 control step `t` 产生 action `a_t`，FIFO queue 在
`t + d` 只应用一次；reset 清空 queue，无 ready action 时执行 7D zero action。rollout 标签
仍然是 `issued_action`，不是当前应用的旧 `applied_action`。

只有 fixed-delay 相对 zero-delay 的绝对 success-rate drop 至少 0.10，且 paired bootstrap
confidence interval 排除 0，才进入 profile 与 recovery 阶段。

### 2. 真实 profile

在 fixed-delay 证明任务确实对 latency 敏感后，按
[`program/PROFILE.md`](PROFILE.md) 记录真实 QwenOFT inference profile。profile 应保留
observation cadence、worker service time、capacity drop 和 inference result 到 environment
process 的边界；不得用模拟固定 delay 代替真实测量。

### 3. Latency-aware teacher

从通过 zero-delay gate 的 PPO checkpoint warm-start，在选定的固定 delay 或真实 profile 下
训练 delayed closed-loop PPO teacher。teacher 第一版仍不读取 latency、queue 或
`applied_action`；只有在预算内没有达到 recovery 目标时，才单独立项 queue-aware teacher
ablation。

目标是在相同 latency condition 下比 frozen zero-delay PPO 提升至少 0.10，并恢复到
zero-delay PPO success rate 的至少 90%。

### 4. Robust distillation

只用 delayed teacher 的 issued actions 生成 demonstrations，再训练不接收 latency/queue
输入的 QwenOFT。最终比较：

```text
J_profile(VLA-delay) >= J_profile(VLA-L0) + 0.10
SR_L0(VLA-delay) >= SR_L0(VLA-L0) - 0.05
```

若 queue-aware teacher 成功但无 queue 输入的 VLA 无法蒸馏，结论记录为
`privileged-teacher observability gap`，不把 queue 偷加到 VLA 输入。

## 停止条件与范围边界

- baseline PPO 未达到 95/100 时，不进入 VLA 或 latency 研究；
- baseline VLA 未达到 25/50 时，不开始 profile 或 delayed teacher；
- 不通过平移 zero-delay action 标签制造 latency-aware data；
- 不把 `applied_action` 当 imitation label；
- 本 proposal 不改变 H1 action contract，不引入 H8/K8、prompt latency conditioning、VLA
  state input 或 VLA online RL；
- InterceptMedium-to-GrabFast transfer 和 `TimedTransfer` 另行立项，不与 MIKASA H1 baseline
  并行混合。
