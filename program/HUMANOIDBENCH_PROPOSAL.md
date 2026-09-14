# HumanoidBench latency 总方针

> 状态：讨论稿。本文只记录研究判断、主线和 gate；具体迭代留在
> `program/BALANCE.md`、`program/RUN.md`、`program/HURDLE.md` 和
> `program/PROFILE.md`。

## 最终研究主线

```text
训练 zero-latency small policy / teacher
  -> rollout teacher data
  -> fine-tune zero-latency VLA
  -> 分别评估 paused L0 与 realtime
  -> profile VLA 的真实 latency distribution
  -> 在同一个 delayed action queue 中在线训练 latency-aware small policy
  -> 采集 latency-aware teacher 当前发出的 action
  -> re-fine-tune VLA
  -> realtime latency 下做最终评估
```

Latency robustness 不能由 zero-latency teacher 直接蒸馏得到。robust teacher 必须
在真实 delayed closed loop 中继续 RL rollout，使 reward 反映延迟后真正发生的后果；
蒸馏标签是 teacher 在时刻 `t` 新发出的 action/chunk，而不是环境此刻正在执行的旧
queued action。

## 当前事实

- **Balance：** direct FastTD3 small teacher 容易训练，direct-43D VLA 也已表现出
  明确 scaling signal，但仍有 closed-loop lower-tail / covariate-shift 问题。
- **Run：** direct small policy 很难训练；SONIC trajectory replay 能提供 locomotion
  prior，但现有 direct VLA 多轮迭代仍未形成可靠 closed loop。
- **Hurdle：** 当前可用路线依赖 run/jump trajectory、手写 transition timing 和 learned
  residual，能构造 teacher，但它是 task-specific recipe，不适合作为最终统一框架。
- **已有 latency small-model 实验：** fixed、normal、uniform 和 mixed latency policy
  都是在相应 action queue 中分别在线训练，不是把 L0 action 标签平移若干帧。

这些结果说明眼下最先要解决的不是 latency，而是统一的 zero-latency teacher/action
contract。teacher 不统一时，后续 dataset、VLA head、latency queue 和评估语义都会继续
按任务分叉。

## “统一”的含义

统一的是 controller contract、训练方法、数据字段和 latency 注入位置；不是要求三个
任务共享同一组权重、reward 或 motion asset。

优先研究的共同接口是：

```text
state + H10 history + task goal
  -> H40 high-level plan
       discrete SONIC primitive/phase transition
       + optional 29D physical residual sequence
  -> latency queue
  -> frozen SONIC decoder / low-level controller at 50 Hz
```

- Balance 使用 stand primitive，policy 学 closed-loop correction。
- Run 使用 run primitive，policy 学稳定和方向 correction。
- Hurdle 允许 run/jump primitive，但 transition 必须由同一个 learned policy 决定，
  手写 distance trigger 只能是 oracle/诊断，不能是最终 teacher。

29D residual 指 SONIC 解码后的 physical joint-target residual，不是 64D token residual。
现有证据表明，在连续 64D token space 中做 full control 容易覆盖 SONIC prior，而直接做
token residual 又未必留在合法 motion manifold；因此任意连续 64D RL 暂不作为主线。

这仍是待验证假设，不应先假定它一定优于 direct RL。它的价值在于用一个 contract
同时表达 stand、run、jump、closed-loop correction 和 delayed chunk execution。

## SONIC / VLA 执行语义

当前应验证的部署语义是：VLA 低频产生 H40 plan（目标约 2.5 Hz），SONIC 使用真实
proprio/action history 在本地 50 Hz 解码并执行。Latency 加在 VLA 发出的整个
high-level chunk 上，只加一次；不要延迟 SONIC 内部每个 50 Hz joint target。

如果 chunk 晚到，应按 observation timestamp 跳过过期 prefix，而不是从 chunk index 0
重新播放。实现前仍需用官方 SONIC/GR00T pipeline 和本仓库 action scheduler 做一次
端到端合同核对，尤其确认 78D action（64D motion token + 14D hand）与 H40 的含义。

## 实验顺序与 gate

1. **统一 zero-latency teacher。** 在 Balance、Run、Hurdle 上使用同一 observation、
   policy architecture、action contract 和 RL trainer；只允许 goal、reward 和 primitive
   asset 不同。先用小预算证明三任务都能达到各自现有 teacher gate。
2. **验证 action oracle。** 在训练前确认 stand/run/jump plan 经相同 H40 scheduler 和
   frozen SONIC 能正确执行；oracle 不成立时先修 action/reset/timing contract，不训练 VLA。
3. **zero-latency distillation。** 先做单 episode/tiny closed-loop overfit，再扩大数据。
   offline loss 不能替代 closed-loop gate。
4. **realtime profiling。** 只有 L0 VLA 成功后才测真实推理 latency，并用 paired seeds
   比较 L0 与 realtime；如果性能没有实质下降，不启动 latency-aware training。
5. **latency-aware teacher。** 只对受 latency 明显影响的任务，在 profile distribution
   对应的真实 action queue 下继续在线 RL，并先证明 small teacher 能恢复性能。
6. **robust distillation。** 采集 issued action/chunk 重新训练 VLA；最终同时检查 realtime
   recovery 与 L0 regression。

任一 gate 失败时先判断是 action contract、teacher solvability、VLA imitation 还是 latency
robustness，不能用增加训练步数把不同问题混在一起。

## Direct VLA + SONIC RL

技术上可行，但不作为第一步：3B VLA 在线 RL 的吞吐、稳定性和样本成本都远高于 small
policy。更合理的顺序是 small RL teacher → token/residual SFT；若 SFT 已有部分 closed-loop
能力但存在稳定的 recovery gap，再做一个有边界的 action-head/LoRA online RL 实验，而不
是从头对整套 VLA 做 RL。

## 现有 VLA tricks 如何处理

- deterministic diffusion、fixed noise、one-step inference、action mask 和一致的
  normalization 仍可复用，它们解决的是优化与可复现性。
- `repeat-current H40` 只适用于 50 Hz 重规划、只执行 `k0` 的 direct-action setup；若改为
  2.5 Hz 执行真正 future H40 chunk，就必须训练 true-future labels。
- direct-43D recovery/history 数据不能自动变成 SONIC token 或 residual dataset；更换
  action contract 后必须重新采集与 issued action 对齐的数据。

## 暂不继续的方向

- 不为每个任务继续堆独立特例，再在最后尝试拼成 latency benchmark。
- 不把 zero-latency teacher action 简单向未来平移来制造 robust labels。
- 不把 arbitrary full-64D 或 token-residual RL 当作默认 SONIC prior 用法。
- 不在 zero-latency VLA 尚未闭环成功前 profile/训练 latency VLA。
- 不把 full-model VLA online RL 当成统一 teacher contract 尚未解决时的替代品。

## 文档分工

- `program/BALANCE.md`：direct teacher/VLA、closed-loop scaling 与 recovery 细节。
- `program/RUN.md`：locomotion prior、history/recovery 和 unified teacher 迭代。
- `program/HURDLE.md`：primitive transition、jump timing 和 clean-pass gate。
- `program/PROFILE.md`：真实 latency profile、paired degradation 和 sim2real。
- 本文：只维护跨任务的主线、判断、gate 和停止条件。
