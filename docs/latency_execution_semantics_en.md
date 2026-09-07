# Latency Execution Semantics Specification

**Status:** Draft v0.1  
**Scope:** This document is the stable semantic contract for the latency-sensitive benchmark.  
**Important:** This document should not contain implementation TODOs, bug lists, temporary workarounds, or current-code status. Those belong in `latency_execution_todo.md`.

---

## 1. Purpose

This document defines the intended execution semantics for the latency-sensitive benchmark.

It is not a description of what the current code already implements. Instead, it is the semantic contract that all executors, wrappers, schedulers, and loggers should eventually follow.

Core principles:

1. Environment stepping, observation emission, model inference, action readiness, and action application are separate events.
2. Actions are applied on the environment timeline, not the observation timeline.
3. Observations are a subsampled view of the raw environment timeline.
4. Latency means the delay from sending an observation to the model until the resulting action becomes available to the environment.
5. `paused`, `simulated`, and `realtime` are different executor semantics and should not be conflated.
6. Every time-related variable must make its unit explicit.

---

## 2. High-level architecture

The system is organized into two wrapper layers:

```text
RawEnvWrapper -> ObservationWrapper -> Policy / Model
```

Actions are applied at the `RawEnvWrapper` layer, not the `ObservationWrapper` layer.

---

## 3. Raw environment layer

The raw environment layer is the lowest-level environment stepping layer.

Its minimal time unit is:

```text
raw_frame
```

`env_fps` controls how frequently raw environment frames advance.

If:

```text
env_fps = 60
```

then:

```text
1 raw_frame = 1 / 60 seconds
```

The raw environment layer is responsible for:

1. stepping the environment by raw frames;
2. applying the active action at each raw frame;
3. maintaining the environment timeline;
4. exposing the current raw state to the observation layer;
5. not deciding when observations are sent to the model.

Key constraint:

> Action latency must be applied at raw environment frame granularity.

That means latency must not only take effect at observation-wrapper step granularity. Even if observations are emitted once every several raw frames, actions should still become ready and be applied at raw-frame granularity.

---

## 4. Observation layer

The observation layer sits above the raw environment layer.

It is responsible for:

1. deciding when to emit observations to the model;
2. deciding what observation format the model receives;
3. constructing resized frames, frame stacks, histories, afterimages, trajectory overlays, or multimodal prompts when needed;
4. maintaining the observation timeline.

`obs_fps` controls the observation emission frequency.

The following must hold:

```text
env_fps % obs_fps == 0
```

Define:

```text
obs_stride_raw_frames = env_fps / obs_fps
```

Example:

```text
env_fps = 60
obs_fps = 10
obs_stride_raw_frames = 6
```

This means the environment advances at 60 raw frames per second, while the model receives one observation every 6 raw frames.

The observation layer only emits observations. It does not decide when actions take effect.

---

## 5. Two timelines

The system has two conceptual timelines.

### 5.1 Environment timeline

The environment timeline is the raw-frame-level timeline:

```text
r = 0, 1, 2, 3, ...
```

Here `r` is the raw frame index.

Action application happens on this timeline.

### 5.2 Observation timeline

The observation timeline is a subsampled view of the environment timeline:

```text
o_k = k * obs_stride_raw_frames
```

The `k`-th observation is emitted at raw frame `o_k`.

Example:

```text
obs_stride_raw_frames = 6
observation raw frames = 0, 6, 12, 18, ...
```

Important rule:

> Observations belong to the observation timeline, but actions belong to the environment timeline.

---

## 6. Action lifecycle

Each action request has the following lifecycle:

```text
observe -> issue inference -> action ready -> action applied -> action held / released
```

Suppose the `k`-th observation is emitted at raw frame `o_k`.

The model produces action `a_k` from that observation.

If the latency is `delta_k_raw_frames`, then the action ready frame is:

```text
ready_raw_frame_k = o_k + delta_k_raw_frames
```

The action cannot be applied before `ready_raw_frame_k`.

The actual application time must satisfy:

```text
apply_raw_frame_k >= ready_raw_frame_k
```

If issue-order FIFO is used, it must additionally satisfy:

```text
apply_raw_frame_k > apply_raw_frame_{k-1}
```

or, more concretely:

```text
apply_raw_frame_k = max(ready_raw_frame_k, apply_raw_frame_{k-1} + 1)
```

---

## 7. Action hold policy

After an action is applied, the executor must define what happens until the next action is applied.

This project supports two hold policies.

### 7.1 `one_frame_then_noop`

The current action is applied for exactly one raw frame, then immediately replaced with the noop/default action.

Default candidate: environments like Flappy Bird, where an action such as `flap` is an instantaneous impulse.

Example:

```text
frame 10: FLAP
frame 11: NOOP
frame 12: NOOP
frame 13: NOOP
```

Semantics: one model output of `FLAP` means one flap, not holding the flap button.

### 7.2 `hold`

The latest applied action is held until the next action arrives. Observation
frequency, pending requests, and capacity drops do not expire it. Before the
first arrival, and after an episode reset, the action is noop/default.

---

## 8. Executor modes

The project defines three executor modes:

```text
paused
simulated
realtime
```

They should share the same action scheduler and logging schema whenever possible, but their time-advancement semantics are different.

---

## 9. Paused mode semantics

Paused mode is the solvability baseline.

Semantics:

1. the environment emits an observation;
2. the environment stops advancing;
3. the model runs inference;
4. the model returns an action;
5. the action is applied to the next raw environment update;
6. wall-clock inference time is logged but does not advance the environment timeline.

Paused mode answers:

> Can the model solve the task if the world waits?

Paused mode does not represent real-time performance.

Paused mode should still log:

```text
latency_ms
policy_runtime_ms
```

but these values do not advance environment frames.

---

## 10. Simulated mode semantics

Simulated mode uses a controllable latency source to simulate the delay from observation emission to action readiness.

Semantics:

1. the raw environment advances by raw frames;
2. the observation layer emits observations according to `obs_fps`;
3. the latency source samples or provides a latency for each observation;
4. the action enters the scheduler after `ready_raw_frame`;
5. the scheduler decides `apply_raw_frame` according to the ordering policy;
6. the action hold policy decides how the action persists after application.

For the `k`-th observation:

```text
obs_env_step_k = k * obs_stride_raw_frames
latency_raw_frames_k = LatencySource.sample(k)
ready_raw_frame_k = obs_env_step_k + latency_raw_frames_k
```

If issue-order FIFO is used:

```text
apply_raw_frame_k = max(ready_raw_frame_k, apply_raw_frame_{k-1} + 1)
```

The main goal of simulated mode is:

> Reproduce delayed action execution in a deterministic and reproducible way.

---

## 11. Latency source semantics

The runtime config selects one explicit `latency.method`. Synthetic methods use
millisecond parameters; the executor converts each draw to raw frames with the
configured `env.env_fps`.

### 11.1 Fixed latency

```yaml
latency:
  method: fixed
  fixed_latency_ms: 100.0
```

This means every action becomes ready 100 ms after its observation is emitted.

### 11.2 Millisecond latency

Conversion rule:

```text
latency_raw_frames = ceil(latency_ms / 1000 * env_fps)
```

Example:

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

Sampling procedure:

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

`iid` draws from the selected worker slot's fitted joint distribution.
`temporal` combines each slot's steady distribution with the shared fitted
hidden-regime burst model. Neither method infers behavior from a legacy sampler
name.

Profile sampling must log:

```text
model_id
gpu_class
workload_id
instance_id
source_run_id
profile_ref
latency_raw_frames
```

---

## 12. Realtime mode semantics

Realtime mode uses wall clock time.

Semantics:

1. the raw environment loop attempts to advance according to `env_fps`;
2. the observation layer emits observations according to `obs_fps`;
3. each emitted observation immediately submits an asynchronous inference job;
4. the environment does not wait for model inference to complete;
5. when an inference job completes, its action enters the action scheduler;
6. the action is applied at the nearest eligible raw frame.

The core realtime principle is:

> The environment must not intentionally block on model inference.

However, the system cannot guarantee perfect `env_fps` under arbitrary CPU/GPU contention. Therefore, realtime mode must log deadline misses and actual environment FPS.

---

## 13. Realtime inference pool

The realtime executor uses an inference pool to manage asynchronous inference.

### 13.1 `executor.inference_devices`

`executor.inference_devices` explicitly lists the devices used by the realtime executor for inference.

Each device starts one model replica, so in the current implementation:

```text
max_in_flight = len(executor.inference_devices)
```

There is no pending observation queue; if all workers are busy, the new observation is handled by the queue-full policy.

Important:

> `max_in_flight` should not be treated as the number of CUDA streams. CUDA streams do not provide hard resource isolation and do not guarantee immunity from synchronization or GPU contention.

### 13.2 Throughput stability rule

Let:

```text
model_inference_time_s = expected seconds per inference request
max_in_flight = maximum concurrent inference jobs
obs_fps = number of observation/inference requests per second
```

The approximate serving capacity is:

```text
capacity = max_in_flight / model_inference_time_s
```

The system must satisfy:

```text
obs_fps < capacity
```

that is:

```text
obs_fps < max_in_flight / model_inference_time_s
```

Equivalently:

```text
utilization = obs_fps * model_inference_time_s / max_in_flight
utilization < 1
```

In practice, a more conservative condition is recommended:

```text
utilization <= 0.7 or utilization <= 0.8
```

The P95/P99 inference time should be preferred over the mean inference time.

---

## 14. Ordering policy

The ordering policy decides what happens when inference completion order differs from observation issue order.

### 14.1 Benchmark policy: issue-order FIFO

Benchmark configs must explicitly select issue-order FIFO; omitting
`scheduler.ordering_policy` is an error.

If `obs_1` is issued before `obs_2`, then `a_2` cannot be applied before `a_1`, even if `a_2` finishes first.

Reasons:

1. simulated and realtime modes become easier to align;
2. the action stream preserves causal order;
3. action age and latency effects are easier to analyze;
4. the benchmark does not silently become a latest-frame reactive controller.

If `a_2` finishes first while `a_1` is still running, `a_2` waits.

If `a_1` times out, `a_1` is marked as skipped and FIFO ordering is unblocked.

### 14.2 Optional ablation: latest-ready

`latest_ready` is an optional ablation.

Semantics:

1. any ready action may be applied;
2. if multiple actions are ready, the latest one is applied;
3. older un-applied actions may be dropped.

This mode may be closer to some reactive robotics/controller systems, but it is not the default benchmark semantics.

---

## 15. Queue full policy

When a new observation arrives but the inference pool is full, the executor needs a queue-full policy.

Recommended default:

```text
drop_new_observation
```

Reason: a running GPU inference job usually cannot be reliably canceled instantly. Even killing a worker may not immediately free GPU kernels or memory.

The current contract does not define a pending observation queue or `kill_oldest_running_job`. A running GPU inference job usually cannot be reliably canceled instantly; even killing a worker may not immediately free GPU kernels or memory.

---

## 16. Long-running inference jobs

The current realtime semantics do not define outlier or timeout behavior.

If an inference job runs for a long time:

1. that worker remains physically busy;
2. new observations may only be assigned to other idle workers;
3. if no worker is idle, the new observation is dropped according to the queue-full policy;
4. `latest_ready` may use a later ready action and ignore the older job when it eventually returns;
5. `issue_order_fifo` continues waiting for the earlier job and is not automatically unblocked by timeout.

The realtime executor does not assume that running GPU inference can be canceled mid-flight while immediately reusing the loaded model.

Whether the implementation kills the worker process is an implementation detail, not part of the semantic contract.

---

## 17. Resource isolation semantics

The realtime target is that the environment loop does not intentionally wait for model inference.

Recommended implementation choices:

1. use separate processes for the environment loop and inference workers;
2. keep the raw environment CPU-only when possible;
3. if GPU rendering is required, prefer using a GPU different from the inference GPU;
4. if environment rendering and inference share a GPU, log contention metrics;
5. avoid global `cuda.synchronize()` calls on the environment path;
6. do not treat CUDA streams as hard isolation.

The semantics must not claim:

> The environment always perfectly maintains `env_fps` under any system load.

It may only claim:

> The executor does not intentionally wait for the model; actual environment timing must be measured and logged.

---

## 18. Required logging

All executor modes should use a unified logging schema.

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

Every time-related variable must include its unit in the name.

Avoid:

```text
frame
step
latency
delay
```

Prefer:

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

Rule:

> If a variable stores time or duration, its unit must be visible in the name.

---

## 20. Minimal semantic invariants

Any implementation that claims to follow this document must satisfy the following invariants.

### 20.1 Timing invariants

1. `env_fps % obs_fps == 0`.
2. `obs_stride_raw_frames = env_fps / obs_fps`.
3. observations are emitted only at `k * obs_stride_raw_frames`.
4. action readiness cannot occur before `obs_env_step + latency_raw_frames`.
5. action application happens on the raw environment timeline.

### 20.2 Simulated mode invariants

1. fixed latency must be deterministic.
2. stochastic latency must be reproducible under the same seed.
3. millisecond latency must be converted into raw frames using an explicit formula.
4. trace replay must be reproducible.

### 20.3 Realtime mode invariants

1. the environment loop does not intentionally wait for inference results.
2. inference jobs may overlap.
3. queue-full, timeout, and drop events must be logged.
4. actual environment FPS must be logged.
5. if deadline misses exceed the configured threshold, the run must be flagged.

### 20.4 Action ordering invariants

1. benchmark configs explicitly select issue-order FIFO.
2. latest-ready is only used as an explicitly configured ablation.
3. timeout may unblock FIFO ordering.
4. dropped/skipped actions must be logged.

---

## 21. Recommended config keys

Example:

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

This document does not include:

1. whether the current code already satisfies these semantics;
2. current bugs;
3. next implementation steps;
4. refactor phases;
5. open questions;
6. temporary workarounds.

Those belong in `latency_execution_todo.md`.
