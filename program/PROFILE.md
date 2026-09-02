# Profiling 工程状态

## Goal and scope

把 realtime profiling 记录成一个完整、按 observation 顺序排列的 request sequence，明确 latency 的边界，并让 simulated worker 的 admission、busy state 和 capacity drop 从前序 inference 自然产生。

本轮只修正 profiling data collection 和 profile input；保留已经做过 sim2real 验证的 IID/temporal 两种 modeling，不同时简化 hidden-regime。用户操作说明维护在 `docs/profiling/cookbook.md`；术语和设计说明放在同目录的设计文档中。

## Current status

- `program/PROFILE.md` 是 profiling 的工程状态和决策记录；第一步 data correction 已经落地。
- 新的 canonical raw artifact 是 `observation_attempts.jsonl`，每个 observation opportunity 恰好一行，并按 `obs_id` 排列。
- canonical metric 名称是 `observation_to_action_latency`。
- worker 内部使用 `worker_service_time`，不再把 `worker_occupancy_duration` 当作独立 profile variable。
- `worker available`、`worker busy_until`、`capacity drop` 是由前序 admitted inference 的 service time 和 observation arrival time 推导出的执行状态，不是独立采样量。
- temporal 必须保留：已有 sim2real 实验表明，IID 打散连续 burst 后可能产生过于乐观的估计；本轮只修正它使用的原始 sequence。
- 旧 profiling artifacts 和旧 paper results 作为历史快照保留；使用新数据得到的 profile 和 simulated results 需要重新生成，不能与旧结果直接混为一谈。

## Locked definitions

一次 observation attempt 的执行链是：

```text
observation attempt
    -> admission decision
    -> worker service, if admitted
    -> result arrives at parent / ResultCoordinator
    -> ordering and scheduler enqueue
    -> frame application
```

时间边界：

```text
observation_to_action_latency
    = result received by the parent
      - observation capture started

worker_service_time
    = result received and worker released by the parent
      - worker reserved immediately before request send
```

parent 收到 result 后，它才进入 `ResultCoordinator`。FIFO/latest-ready ordering、scheduler enqueue 和固定 frame boundary 的 application 都发生在这个边界之后，不属于 `observation_to_action_latency`。因此 result 在 55ms ready、环境在 60ms frame apply 时，profile 记录 55ms 而不是 60ms；simulated replay 再执行同一套 ordering/application 规则，避免重复计算 FIFO 等待。

Action chunk 也只记录整个 chunk result ready 的一次 latency；chunk 内 action 的逐帧使用属于 scheduler/application 语义。

`capacity_drop` 表示 observation attempt 到达时没有 idle worker，因此没有进入 inference。它没有 service time 或 action latency 样本；不能用伪造的 latency 填充 drop。

## Current data problems

当前 artifacts 主要记录 completion/action stream，而不是完整的 observation-attempt stream：

- worker 忙时的 admission drop 没有统一的 request row，主要散落在 step annotation 中；
- 多 worker 下 completion order 不等于 observation/submission order；
- episode 结束时仍在执行的 request 可能只被 drain/release，没有进入最终 profile sequence；
- 当前 `worker_service_ms` 是若干 stage duration 的组合，不一定等于实际 worker busy interval；
- 当前 `action_ready_ms` 不是从 observation capture start 到 parent result receipt 的直接测量，并且可能受 applied/drop 筛选影响；
- profile builder 不能从 action-only input 恢复每个 observation attempt 的 admission 因果关系。

因此，旧 artifacts 不能无损转换成新的完整 sequence。它们继续用于复现历史结果，但不能作为新 profiling contract 的唯一输入。

## This-round data correction

每个 observation attempt 都在 session sequence 中占一条记录，包括 admitted 和 capacity-dropped attempt。logger 在 episode boundary 追加文件，但 `obs_id` 在 session 内持续递增；builder 按 `obs_id` 恢复 observation/submission order。

Canonical schema 是：

```text
episode_id
obs_id
obs_env_step
submitted

# 仅 submitted=true 时存在
worker_id
observation_to_action_latency_ms
worker_service_time_ms
```

`submitted=false` 就是 capacity drop，不写 null worker、伪造 latency 或 service 字段。两个 duration 在真实边界直接计时，而不是再把 pipeline stage 拼接成一个估计值。由 admitted rows 的 service interval 和 observation cadence 推导：

```text
worker busy intervals
worker availability at each attempt
capacity drops
```

不把 `worker_available`、`worker_busy_until` 或 `in_flight_count` 作为 profile model 的独立输入；它们是 replay/simulation state 或诊断信息。logger flush 之前会 drain 并补全本 episode 的 tail completion，但这些 completion 不会在 episode 结束后再 enqueue/apply，也不会改写 episode action metrics。

本轮不引入通用 event framework，不引入 exact trace replay，不改变 scheduler 的 queue/apply 语义。

## Profile consumers

完整 sequence 是唯一的 profiling source；profile builder 不再从 `actions.jsonl` 和 applied/drop outcome 反推 profiling 样本。IID 和 temporal 是 sequence 的两种不同消费方式。

### IID

从 admitted rows 中抽取单次联合样本：

```text
(worker_service_time, observation_to_action_latency)
```

同一次 inference 的两个值保持配对。IID 丢弃相邻 request 的顺序，因此不表达连续 burst；capacity drop 仍由 simulated worker 根据 service time 和 arrival cadence 推导。

### Temporal

使用按 `obs_id` 排列的 admitted sequence，保留连续 latency/burst 的结构。temporal 仍然生成与 IID 相同语义的 latency pair，但会让相邻 inference 共享随时间变化的 regime，使长 service time 可能连续出现。

temporal 不直接采样 drop。每个 generated request 先经过 worker admission；worker 是否 busy 以及后续 drop 由已经生成的 service time 决定。这正是连续 burst 能影响 closed-loop behavior 的路径。

本轮先保持现有 hidden-regime 算法，只修正它的输入顺序和 measurement boundary；算法简化是后续独立实验，不能和数据修正混在同一次归因中。

## Sample Factory small-model training

Sample Factory small-model training 支持 `iid` profile replay：

```yaml
latency:
  method: iid
  profile_path: results/profiling/profiles/<model>/<gpu>/<workload>/<instance>/profile.json
  profile_worker_slot: 0
  seed: 42
```

训练读取 canonical profile 的指定 `worker_slots/<slot>/all` 联合分布，并使用其中的
`observation_to_action_latency_ms` 作为 action queue latency。每个训练 env 使用独立的
seeded IID stream；`profile_worker_slot` 必须明确指定，不会在 slots 之间隐式切换。

这是 action-only training replay：`worker_service_time` 不会驱动 Sample Factory 的 worker
occupancy 或 capacity drop。需要完整 admission、service 和 drop 语义时，使用
`latency_bench` 的 simulated executor replay。Sample Factory training 当前不支持
`temporal`；旧的 top-level inverse-CDF profile 也不属于 canonical training 输入，必须先
按本文件定义重新生成 profile artifacts。

## Verification

当前回归已经验证：

- sequence record 数量覆盖全部 observation attempts，而不只是完成的 actions；
- accepted rows 的时间顺序是 observation/submission order；
- accepted rows 都有 worker service 和 parent result receipt 的真实边界；
- capacity-dropped rows 没有伪造 latency；
- episode tail completions 被保留；
- IID/temporal builder 和 sampler 能读取新 schema，并保留 measurement session boundary。

双 GPU Balance smoke 已验证 realtime trace 能记录两个 worker slot、capacity drop，并生成 per-slot temporal profile。仍待运行完整的 5×100 profiling，再分别运行 simulated 和 realtime 对照实验。

旧 realtime return 不变，旧 IID/temporal 数值也不原地改写；变化来自新 sequence 重新拟合后的 profile。temporal 是否继续优于 IID，要在数据修正后重新验证。

## Next decision

1. 在真实 model inference profiling run 中检查 `observation_attempts.jsonl`。
2. 用新 sequence 重新生成现有 IID 和 temporal profile。
3. 比较新旧 profile 的 latency marginal、burst/run length、worker saturation 和 capacity-drop pattern。
4. 重新检查 sim2real gap，再决定是否有必要简化 hidden-regime。

## Deferred

- hidden-regime 的简化和删除 rank/copula 等机制；
- exact trace replay；
- 在 `docs/profiling/` 增加完整设计文档。

## Implementation and verification

- Realtime executor、inference pool 和 trajectory logger 已生成 `observation_attempts.jsonl`，并保留 capacity drop、out-of-order completion 的 observation order 和 episode tail completion。
- Profile builder 已切换到 canonical sequence；distribution schema 为 3，hidden-regime schema 为 12，旧 schema 不提供 compatibility path。
- 新 manifest 记录 `n_observation_attempts`、`n_admitted_observations` 和 `n_capacity_drops`；per-slot summary 只统计 admitted jobs。
- 当前聚焦回归：realtime/logging `50 passed`；profile/burst/sampler/cross-device `68 passed`。
- 尚未运行完整的 5×100 profiling 或 sim2real 实验，因此本文档不宣称新的 latency distribution 或 performance result。
