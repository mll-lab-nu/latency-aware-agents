# Latency execution semantics

LAGEN separates environment steps, observation requests, action readiness, and action application.
Select the execution mode, ordering policy, and hold policy in the run configuration.
See [configuration](configuration.md) for field ownership.

## Clocks and observations

`env.env_fps` defines raw environment steps per second.
`env.obs_fps` defines observations sent to the policy per second.
Their ratio must be a positive integer:

```text
raw_frame_ms = 1000 / env_fps
observation_stride = env_fps / obs_fps
observation_frame_k = k * observation_stride
```

For example, 60 environment FPS and 10 observation FPS give one observation every six raw steps.
Actions can become ready and apply between observations.
The environment adapter advances raw steps. Observation scheduling selects when the policy receives input.

A simulated environment clock depends on raw steps, not simulator or model wall time.
Changing observation FPS changes decision cadence. It does not change the granularity of action application.
`evaluation.max_env_steps` counts raw steps, including steps between observations.

## Execution modes

| `executor.mode` | Environment clock | Action readiness |
| --- | --- | --- |
| `paused` | Raw-step clock that stops during inference | Zero simulated delay after synchronous prediction |
| `simulated` | Raw-step clock | Configured sampled delay |
| `realtime` | Actual wall clock paced toward `env_fps` | Actual asynchronous inference completion |

Paused execution waits for each prediction before advancing.
Its recorded inference time does not advance the environment clock.
Use paused execution to test performance when the environment waits for the model.

Simulated execution advances under the declared delay and scheduler rules.
Actual inference speed does not define simulated delay.
Realtime execution advances while inference is pending.
Realtime and paused execution ignore the root `latency` section.

A realtime environment can miss its target FPS under resource contention.
The executor records actual FPS, lag, and deadline misses.
A miss occurs when frame-start lag exceeds one raw-frame period.
There is no automatic run-rejection threshold for these misses.

## Delay and readiness

A request starts from a source observation at raw frame `o_k`.
Its result cannot apply before it becomes ready.
Simulated delay uses milliseconds and converts to raw steps:

```text
latency_raw_frames = ceil(latency_ms * env_fps / 1000)
ready_raw_frame_k = o_k + latency_raw_frames
apply_raw_frame_k >= ready_raw_frame_k
```

At 60 FPS, 100 ms becomes six raw steps.
Ordering and frame eligibility can delay application beyond readiness.
Readiness and application remain separate execution facts.

Select one simulated sampling method:

| `latency.method` | Parameters and behavior |
| --- | --- |
| `zero` | Zero simulated delay |
| `fixed` | `fixed_latency_ms` for each request |
| `normal` | `mean_latency_ms`, `std_latency_ms`, `min_latency_ms`, `max_latency_ms` |
| `uniform` | `min_latency_ms`, `max_latency_ms` |
| `iid` | Independent draws from `profile_path` |
| `temporal` | Fitted temporal structure from `profile_path` |

For a fixed condition:

```yaml
latency:
  method: fixed
  fixed_latency_ms: 100
```

For bounded normal delay:

```yaml
latency:
  method: normal
  mean_latency_ms: 100
  std_latency_ms: 20
  min_latency_ms: 0
  max_latency_ms: 200
  seed: 42
```

The normal sampler draws milliseconds, clamps to the bounds, and converts to raw steps.
A uniform sampler draws between its declared bounds.
For profile replay:

```yaml
latency:
  method: temporal
  profile_path: /ABSOLUTE/PROFILE/profile.json
  seed: 42
```

`iid` uses a fitted distribution for the selected worker slot.
`temporal` combines slot distributions with the fitted shared burst model.
See [profiling](profiling/cookbook.md) for measurement and fitting.

## Ordering policies

Declare `scheduler.ordering_policy` explicitly.
There is no global ordering default.

`issue_order_fifo` applies results in request order.
A later completed result waits for the earlier request.
The scheduler applies at most one selected result per raw step.
For consecutive admitted requests, the earliest application satisfies:

```text
apply_raw_frame_k = max(ready_raw_frame_k, apply_raw_frame_previous + 1)
```

Realtime FIFO has no timeout that automatically releases a blocked request order.
A long-running request can therefore delay later results.

`latest_ready` selects the newest available result.
It discards ready results that the selected result supersedes.
An older result arriving later cannot replace a result from a newer observation.

## Waiting and hold policies

Declare `scheduler.hold_policy` explicitly.
Before the first action arrives, and after reset, the scheduler uses the task's native no-op.
The task author supplies that action.

`one_frame_then_noop` applies a selected single action for one raw step.
It then uses the no-op until the next selected action.
For an impulse action such as Flappy's flap:

```text
frame 10: FLAP
frame 11: NOOP
frame 12: NOOP
```

`hold` retains the latest selected single action until a replacement arrives.
Pending requests, observation intervals, and capacity rejection do not expire that action.

## Action chunks

A chunk contains actions aligned to successive raw frames from its source observation.
If a chunk starts at observation frame `o_k`, head `i` targets frame `o_k + i`.
The scheduler skips heads whose intended frames passed before the result became available.
It does not restart the chunk at head zero on arrival.

For a chunk observed at frame 0 and applied at frame 3, the first eligible head is head 3.
The selected chunk replaces the previously active chunk.

`scheduler.hold_last_chunk_action: true` retains the last chunk action after its horizon expires.
When this setting is false, the scheduler uses the no-op after chunk exhaustion.
Chunk diagnostics record stale prefixes, age, and underflow.
Task-specific action-prefix inputs contain committed actions available at request time.
They do not expose a future sampled delay.

## Capacity and inference workers

`executor.inference_devices` selects actual model worker devices.
Each entry creates a model replica. Repeated names create separate replicas on the same device.
For realtime execution:

```text
max_in_flight = len(executor.inference_devices)
```

Each worker handles one request at a time.
When every worker is busy, the executor rejects the new observation and records the attempt.
It does not queue that observation or cancel a running request to make room.
A long-running request occupies its worker until completion.
Other idle workers can accept new requests.

`executor.simulated_worker_capacity` controls virtual occupancy independently of physical inference resources.
The simulator assigns admitted requests to idle virtual workers.
Profile worker service time and action latency retain their distinct meanings.
Physical inference batching does not set virtual capacity.

A rough throughput estimate is:

```text
capacity_requests_per_second = max_in_flight / mean_service_time_seconds
utilization = obs_fps / capacity_requests_per_second
```

Utilization below one does not prevent rejection during variable service times.
This estimate is not a configuration constraint.
Measure admitted requests and tail latency for the actual workload.

## Parallel episodes and reset

`executor.parallel_envs` controls simulated episode slots. Paused and realtime modes use one environment.
Each slot owns policy state, pending actions, and simulated worker occupancy.
Episode reset clears the addressed slot's policy and scheduler state.
An explicit task soft reset also clears the state required by that task's episode contract.

IID and parametric samplers derive per-episode streams.
Temporal replay keeps one machine-state timeline per replica across episode boundaries.
Changing parallelism or episode completion order can change later temporal draws.
Keep seeds, clocks, replica count, and episode assignment fixed for trajectory reproduction.
See [parallel evaluation](eval_parallelism.md) for backend choices.

## Realtime resources and saved traces

Use separate environment and inference processes.
Use separate GPUs for rendering and inference when the selected deployment allows it.
CUDA streams do not provide hardware isolation.
Record the actual deployment when rendering and inference share resources.

Realtime always measures basic timing.
Logging settings control persistence rather than measurement.
Worker inference latency, observation-to-action latency, worker service time, and applied latency describe different intervals.
The [trace guide](realtime_pipeline_logging.md) defines their boundaries and sample populations.

Keep the resolved configuration with every result.
Include model revision, episode count, environment and observation FPS, latency settings, seed protocol, and hardware for measured results.
