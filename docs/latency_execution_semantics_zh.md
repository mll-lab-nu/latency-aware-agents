# Latency Execution Semantics Specification（中文）

**Status:** Draft v0.1  
**Scope:** This document is the stable semantic contract for the latency-sensitive benchmark.  
**Important:** This document should not contain implementation TODOs, bug lists, temporary workarounds, or current-code status. Those belong in `latency_execution_todo.md`.

---

## 1. 目的

本文档定义 latency-sensitive benchmark 的执行语义。它的目标不是描述当前代码“已经实现了什么”，而是定义项目中所有 executor、wrapper、scheduler、logger 后续必须对齐的语义契约。

核心原则：

1. 环境推进、observation 发送、模型推理、action ready、action apply 是不同事件。
2. action 必须作用在 environment timeline 上，而不是 observation timeline 上。
3. observation 是 raw environment timeline 的下采样视图。
4. latency 表示从 observation 被发给模型，到 action 可以被环境采用之间的延迟。
5. paused、simulated、realtime 是三种不同 executor semantics，不应该混在一起。
6. 所有涉及时间的变量都必须显式说明单位。

---

## 2. 总体架构

系统分为两层 wrapper：

```text
RawEnvWrapper -> ObservationWrapper -> Policy / Model
```

其中 action 的应用位置在 `RawEnvWrapper` 层，而不是 `ObservationWrapper` 层。

---

## 3. Raw environment layer

Raw environment layer 是最低层环境推进层。

它的最小时间单位是：

```text
raw_frame
```

`env_fps` 控制 raw environment frame 的刷新频率。

如果：

```text
env_fps = 60
```

则：

```text
1 raw_frame = 1 / 60 seconds
```

Raw environment layer 的职责：

1. 按 raw frame 推进环境；
2. 在每个 raw frame 上应用当前 active action；
3. 维护 environment timeline；
4. 暴露当前 raw state 给 observation layer；
5. 不负责决定什么时候把 observation 发给模型。

关键约束：

> Action latency must be applied at raw environment frame granularity.

也就是说，latency 不能只在 observation wrapper step 上生效。即使 observation 每 4 个 raw frames 才发一次，action 也仍然应该能够在 raw frame 级别 ready/apply。

---

## 4. Observation layer

Observation layer 位于 raw environment layer 之上。

它的职责：

1. 控制什么时候给模型发送 observation；
2. 控制模型看到的 observation 形式；
3. 处理 resize、frame stack、history、afterimage、trajectory overlay、多模态 prompt 等；
4. 维护 observation timeline。

`obs_fps` 控制 observation 发送频率。

必须满足：

```text
env_fps % obs_fps == 0
```

定义：

```text
obs_stride_raw_frames = env_fps / obs_fps
```

例如：

```text
env_fps = 60
obs_fps = 10
obs_stride_raw_frames = 6
```

含义是：环境每秒推进 60 个 raw frames，但每 6 个 raw frames 才向模型发送一次 observation。

Observation layer 只负责发送 observation，不负责决定 action 何时生效。

---

## 5. 两条 timeline

系统中有两条 timeline。

### 5.1 Environment timeline

Environment timeline 是 raw frame 级别的时间轴：

```text
r = 0, 1, 2, 3, ...
```

其中 `r` 表示 raw frame index。

Action apply 发生在这条 timeline 上。

### 5.2 Observation timeline

Observation timeline 是 environment timeline 的下采样：

```text
o_k = k * obs_stride_raw_frames
```

第 `k` 个 observation 在 raw frame `o_k` 被发出。

例如：

```text
obs_stride_raw_frames = 6
observation raw frames = 0, 6, 12, 18, ...
```

重要规则：

> Observation belongs to the observation timeline, but action belongs to the environment timeline.

---

## 6. Action lifecycle

每个 action request 有如下生命周期：

```text
observe -> issue inference -> action ready -> action applied -> action held / released
```

设第 `k` 个 observation 在 raw frame `o_k` 发出。

模型基于该 observation 产生 action `a_k`。

如果 latency 是 `delta_k_raw_frames`，则 action ready frame 为：

```text
ready_raw_frame_k = o_k + delta_k_raw_frames
```

action 不能在 `ready_raw_frame_k` 之前被应用。

最终实际应用时间为：

```text
apply_raw_frame_k >= ready_raw_frame_k
```

如果使用 FIFO ordering，还需要满足：

```text
apply_raw_frame_k > apply_raw_frame_{k-1}
```

或更一般地：

```text
apply_raw_frame_k = max(ready_raw_frame_k, apply_raw_frame_{k-1} + 1)
```

---

## 7. Action hold policy

Action 被 apply 之后，executor 必须定义在下一个 action apply 之前如何维持当前 action。

本项目支持两类 hold policy。

### 7.1 `one_frame_then_noop`

当前 action 只执行一个 raw frame，然后立刻变成 noop/default action。

默认候选场景：Flappy Bird 这类 action 是瞬时 impulse 的环境。

例子：

```text
frame 10: FLAP
frame 11: NOOP
frame 12: NOOP
frame 13: NOOP
```

语义：模型输出一次 `FLAP` 只代表一次 flap，而不是持续按住 flap。

### 7.2 `hold`

当前 action 持续一个 observation period：

```text
duration_raw_frames = obs_stride_raw_frames = env_fps / obs_fps
```

如果这一个 observation period 结束后下一个 action 还没有 ready，则系统使用 noop/default action。

这种策略可能在 jitter 下产生 action gap。

---

## 8. Executor modes

本项目定义三种 executor mode：

```text
paused
simulated
realtime
```

三者必须共享尽可能一致的 action scheduler 和 logging schema，但它们的时间推进语义不同。

---

## 9. Paused mode semantics

Paused mode 是 solvability baseline。

语义：

1. 环境发出 observation；
2. 环境暂停推进；
3. 模型运行推理；
4. 模型返回 action；
5. action 应用到下一次 raw environment update；
6. wall-clock inference time 被记录，但不影响 environment timeline。

Paused mode 回答的问题是：

> 如果世界愿意等待模型，模型是否能完成任务？

Paused mode 不代表 real-time performance。

Paused mode 中也应该记录：

```text
latency_ms
policy_runtime_ms
```

但这些值不会导致 environment frame 前进。

---

## 10. Simulated mode semantics

Simulated mode 用可控 latency source 模拟从 observation 到 action ready 的延迟。

语义：

1. raw environment 按 raw frame 推进；
2. observation layer 按 `obs_fps` 发出 observation；
3. latency source 为每个 observation 采样或指定一个 latency；
4. action 在 `ready_raw_frame` 后进入 scheduler；
5. scheduler 按 ordering policy 决定 `apply_raw_frame`；
6. action hold policy 决定 action 被 apply 后如何持续。

对于第 `k` 个 observation：

```text
obs_env_step_k = k * obs_stride_raw_frames
latency_raw_frames_k = LatencySource.sample(k)
ready_raw_frame_k = obs_env_step_k + latency_raw_frames_k
```

如果使用 issue-order FIFO：

```text
apply_raw_frame_k = max(ready_raw_frame_k, apply_raw_frame_{k-1} + 1)
```

Simulated mode 的关键目标：

> 用 deterministic/reproducible 的方式复现 action delayed execution。

---

## 11. Latency source semantics

运行配置通过唯一的 `latency.method` 选择采样方式。合成采样方式统一使用毫秒参数，
executor 根据 `env.env_fps` 把每次采样转换为 raw frames。

### 11.1 Fixed latency

```yaml
latency:
  method: fixed
  fixed_latency_ms: 100.0
```

表示每个 action 都在 observation 发出 100ms 后 ready。

### 11.2 Millisecond latency

转换规则：

```text
latency_raw_frames = ceil(latency_ms / 1000 * env_fps)
```

例如：

```text
env_fps = 60
latency_ms = 100
latency_raw_frames = ceil(100 / 1000 * 60) = 6
```

### 11.3 Normal latency

```yaml
latency:
  method: normal
  mean_latency_ms: 100.0
  std_latency_ms: 20.0
  min_latency_ms: 0.0
  max_latency_ms: 200.0
```

采样流程：

```text
sample milliseconds -> clamp -> ceil to raw frames
```

### 11.4 Uniform latency

```yaml
latency:
  method: uniform
  min_latency_ms: 50.0
  max_latency_ms: 150.0
```

### 11.5 Profile-sampled latency

```yaml
latency:
  method: temporal  # iid | temporal
  profile_path: path/to/profile.json
  seed: 0
```

`iid` 从 profile 的 inverse-CDF distribution 独立采样；`temporal` 使用 profile
中拟合的 hidden-regime burst model。两者都不再从旧 sampler 名称推断行为。

Profile sampling 必须记录：

```text
profile_path
profile_id
sample_index
converted_latency_raw_frames
```

---

## 12. Realtime mode semantics

Realtime mode 使用 wall clock。

语义：

1. raw environment loop 按 `env_fps` 尝试推进；
2. observation layer 按 `obs_fps` 发送 observation；
3. 每个 observation 发出后立即提交一个 asynchronous inference job；
4. environment 不等待模型推理完成；
5. inference job 完成后，action 进入 action scheduler；
6. action 在最近的 eligible raw frame 被应用。

Realtime mode 的核心原则：

> The environment must not intentionally block on model inference.

但是，系统不能保证在任意 CPU/GPU contention 下 environment 都完美保持 `env_fps`。因此 realtime mode 必须记录 deadline miss 和 actual env FPS。

---

## 13. Realtime inference pool

Realtime executor 使用 inference pool 管理异步推理。

### 13.1 `executor.inference_devices`

`executor.inference_devices` 显式列出 realtime executor 用于推理的设备。

每个 device 启动一个模型副本，因此当前实现中：

```text
max_in_flight = len(executor.inference_devices)
```

没有 pending observation queue；如果所有 worker 都 busy，新 observation 按 queue-full policy 处理。

注意：

> `max_in_flight` 不应该简单等同于 CUDA stream 数量。CUDA streams 不提供严格资源隔离，也不能保证不被 synchronize 或 GPU contention 影响。

### 13.2 Throughput stability rule

设：

```text
model_inference_time_s = 每个 inference request 的期望耗时，单位秒
max_in_flight = 最大并发 inference 数
obs_fps = 每秒发起 observation/inference request 的数量
```

稳定服务能力约为：

```text
capacity = max_in_flight / model_inference_time_s
```

必须满足：

```text
obs_fps < capacity
```

也就是：

```text
obs_fps < max_in_flight / model_inference_time_s
```

等价地：

```text
utilization = obs_fps * model_inference_time_s / max_in_flight
utilization < 1
```

推荐实际使用更保守条件：

```text
utilization <= 0.7 或 utilization <= 0.8
```

并且使用 P95/P99 inference time，而不是只用 mean inference time。

---

## 14. Ordering policy

Ordering policy 决定多个 inference result 完成顺序和 apply 顺序不一致时如何处理。

### 14.1 Benchmark policy: issue-order FIFO

Benchmark config 必须显式选择 issue-order FIFO；省略
`scheduler.ordering_policy` 会直接报错。

如果 `obs_1` 先于 `obs_2` 被发出，则 `a_2` 不能先于 `a_1` 被 apply，即使 `a_2` 更早完成。

原因：

1. simulated 和 realtime 更容易对齐；
2. action stream 保持因果顺序；
3. 更容易分析 action age 和 latency effect；
4. 避免 silently turning the benchmark into a latest-frame reactive controller。

如果 `a_2` 先完成但 `a_1` 未完成，则 `a_2` 等待。

如果 `a_1` timeout，则 `a_1` 被标记为 skipped，FIFO 被 unblock。

### 14.2 Optional ablation: latest-ready

`latest_ready` 是可选 ablation。

语义：

1. 任何 ready action 都可以被应用；
2. 如果有多个 ready action，应用最新的；
3. older un-applied actions 可以被丢弃。

这个模式更接近某些 reactive robotics/controller 系统，但不是默认 benchmark semantics。

---

## 15. Queue full policy

当新的 observation 到来，但 inference pool 已满时，需要 queue full policy。

默认推荐：

```text
drop_new_observation
```

原因：正在运行的 GPU inference 通常不能被可靠地即时取消。即使杀掉 worker，也未必能立刻释放 GPU kernel 或 memory。

当前 contract 不定义 pending observation queue，也不定义 “kill oldest running job”。正在运行的 GPU inference 通常不能被可靠地即时取消；即使杀掉 worker，也未必能立刻释放 GPU kernel 或 memory。

---

## 16. Long-running inference jobs

当前 realtime 语义不定义 outlier 或 timeout。

如果某个 inference job 长时间未返回：

1. 该 worker 在物理上仍然 busy；
2. 新 observation 只能分配给其他 idle worker；
3. 如果没有 idle worker，新 observation 按 queue-full policy 被 dropped；
4. `latest_ready` 可以使用后续已经 ready 的 action，并在旧 job 之后返回时 ignore 它；
5. `issue_order_fifo` 会继续等待更早的 job，不通过 timeout 自动 unblock。

Realtime executor 不假设正在运行的 GPU inference 可以被中途取消并立即复用模型。

---

## 17. Resource isolation semantics

Realtime mode 的目标是：环境 loop 不主动等待模型推理。

推荐实现策略：

1. environment loop 和 inference worker 使用不同 process；
2. 尽量让 raw environment 使用 CPU；
3. 如果必须 GPU rendering，优先使用与 inference 不同的 GPU；
4. 如果 env rendering 和 inference 共用 GPU，必须记录 contention metrics；
5. environment path 中避免全局 `cuda.synchronize()`；
6. 不把 CUDA streams 当作硬隔离机制。

语义上不能承诺：

> 在任意系统负载下 environment 一定完美维持 env_fps。

只能承诺：

> executor 不会有意等待模型；实际 env timing 必须被测量和记录。

---

## 18. Required logging

所有 executor mode 都应该记录统一字段。

### 18.1 Timing fields

```text
raw_frame_idx
obs_idx
obs_env_step
issue_wall_time_ns
ready_wall_time_ns
apply_raw_frame
latency_ms
latency_raw_frames_sampled
latency_raw_frames_effective
action_age_raw_frames
```

### 18.2 Realtime environment timing fields

```text
target_env_fps
actual_env_fps
env_frame_wall_time_ms
env_deadline_miss_count
env_deadline_miss_rate
wall_clock_lag_ms
catchup_step_count
```

### 18.3 Queue / scheduler fields

```text
in_flight_count
queue_full_count
dropped_observation_count
dropped_action_count
fifo_blocked_count
ignored_late_result_count
```

### 18.4 Action fields

```text
action_id
action_value
action_source_obs_idx
action_issue_order
action_apply_order
hold_policy
active_action_before
active_action_after
```

---

## 19. Naming rules

所有时间变量必须在变量名中写清单位。

避免：

```text
frame
step
latency
delay
```

推荐：

```text
raw_frame
obs_frame
env_step
obs_env_step
latency_raw_frames
latency_ms
ready_raw_frame
apply_raw_frame
obs_stride_raw_frames
```

规则：

> If a variable stores time or duration, its unit must be visible in the name.

---

## 20. Minimal semantic invariants

任何实现只要声称符合本文档，就必须满足以下 invariants。

### 20.1 Timing invariants

1. `env_fps % obs_fps == 0`。
2. `obs_stride_raw_frames = env_fps / obs_fps`。
3. observation 只在 `k * obs_stride_raw_frames` 发出。
4. action ready 不早于 `obs_env_step + latency_raw_frames`。
5. action apply 发生在 raw environment timeline 上。

### 20.2 Simulated mode invariants

1. fixed latency 必须 deterministic。
2. stochastic latency 在相同 seed 下必须 reproducible。
3. ms latency 必须用明确公式转换成 raw frames。
4. replay trace 必须可复现。

### 20.3 Realtime mode invariants

1. environment loop 不主动等待 inference result。
2. inference jobs 可以 overlap。
3. queue full、timeout、drop 都必须被记录。
4. actual env FPS 必须被记录。
5. 如果 deadline miss 超过阈值，该 run 必须被 flag。

### 20.4 Action ordering invariants

1. benchmark config 显式选择 issue-order FIFO。
2. latest-ready 只能作为显式配置的 ablation。
3. timeout 可以 unblock FIFO。
4. dropped/skipped actions 必须被记录。

---

## 21. Recommended config keys

示例：

```yaml
executor:
  mode: simulated  # paused | simulated | realtime

env:
  env_fps: 60
  obs_fps: 10

latency:
  method: fixed  # zero | fixed | normal | uniform | iid | temporal
  fixed_latency_ms: 50.0

scheduler:
  hold_policy: one_frame_then_noop
  ordering_policy: issue_order_fifo  # issue_order_fifo | latest_ready

```

---

## 22. What this document intentionally excludes

本文档不包含：

1. 当前代码是否已经满足这些语义；
2. 当前有哪些 bug；
3. 下一步实现计划；
4. refactor phase；
5. open questions；
6. 临时 workaround。

这些内容应该写在 `latency_execution_todo.md`。
