# Realtime Pipeline Logging

This document defines the runtime timing fields emitted by realtime evaluation.
All durations are milliseconds.

## Output Files

- `latencies.jsonl`: worker measurements collected before episode metric finalization.
  `latency_ms` measures worker-side action generation.
- `observation_attempts.jsonl`: every observation attempt, in observation order.
  Admitted requests include completion measurements, including results received after the terminal step.
- `e2e_latencies.jsonl`: one row per applied action. `latency_ms` is
  `inference_latency_ms + bench_overhead_ms` when pipeline profiling is enabled.
  With profiling disabled, it is the frame-derived `applied_latency_ms`.
- `realtime_pipeline_profile.jsonl`: per-action stage breakdown.
- `realtime_pipeline_summary.json`: aggregate statistics for the same fields.

## Core Latency Fields

- `inference_latency_ms`: worker-side action generation time measured by
  `RuntimeLatencyMeasurer`. It includes worker-local observation copy,
  policy input construction, model/action wrapper prediction, and action decode.
- `bench_overhead_ms`: benchmark/runtime overhead around inference. It includes
  env observation, IPC, parent scheduling, worker receive overhead, result
  release, and env enqueue/scheduler overhead. It does not include policy/model
  prediction stages.
- `e2e_latency_ms`: `inference_latency_ms + bench_overhead_ms`.
- `applied_latency_ms`: frame-quantized delay from observation frame to action
  apply frame.

## Measurement Boundaries and Sample Populations

These intervals describe different parts of execution. Do not combine their samples into one latency distribution.

| Field | Start | End | Population |
| --- | --- | --- | --- |
| `latencies.jsonl:latency_ms` | Worker measurement starts, after any initial CUDA synchronization | Policy returns and any final CUDA synchronization finishes | Completion events collected for episode metrics |
| `observation_to_action_latency_ms` | Environment observation starts | Parent receives the inference result | All admitted observation attempts |
| `worker_service_time_ms` | Parent starts dispatching a request to a worker | Parent receives that result | All admitted observation attempts |
| `applied_latency_ms` | Source observation frame | First frame that applies the action | Applied actions |
| `bench_overhead_ms` | Sum of the recorded overhead stages | Not a separate start/end measurement | Applied actions with pipeline profiling |

Episode `num_actions` counts completion events collected during finalization, including explicitly discarded results.
It does not count environment steps or only applied actions.
`num_dropped_actions` counts events marked as dropped by ordering or scheduling rules.
An unapplied result at episode end is not automatically a dropped event.

Realtime finalization collects available results before computing episode metrics.
Remaining requests finish outside that interval and only enrich `observation_attempts.jsonl`.
Thus admitted requests can outnumber `num_actions` and the samples in `latencies.jsonl`.
Capacity-rejected attempts have `submitted: false` and no completion measurements.

Simulated execution finishes its virtual pending results at the episode boundary.
Paused execution records synchronous results.
Preserve these populations when comparing modes.
Compare identical populations when comparing versions of the same mode.

`fifo_blocked_count` counts coordinator polls that encounter FIFO blocking.
It does not count distinct blocked requests, so polling rates affect this diagnostic.

`actual_env_fps` divides completed steps by the episode wall interval measured before tail draining and file writes.
Reset and warmup occur before this interval.
Environment lag compares the frame start with its absolute scheduled time.
A deadline miss occurs when that lag exceeds one frame duration.

The parent pauses automatic cyclic GC during each episode and collects at its boundary.
It restores the caller's GC state and leaves surviving objects collectable after executor shutdown.
Inference workers retain their existing model-freezing and reset barriers within their own processes.
Boundary collection cost remains separate from paced-loop latency.

## Statistical Methods

Episode, pipeline, calibration and profile summaries use linear interpolation for percentiles.
For sorted samples, percentile `p` interpolates at index `(n - 1) * p / 100`.
For `[10, 20, 30, 40]`, p50 is 25 and p90 is 37.
Pipeline and calibration summaries previously used order statistics. Historical summaries retain their original values.

Return and length summaries weight each episode equally.
Task moment summaries combine sample counts, sums and squared sums before calculating their mean and population standard deviation.
Latency summaries calculate percentiles from individual samples. Do not average episode percentiles to obtain a run percentile.

## Bench Overhead Stages

These fields sum to `bench_overhead_ms`:

- `env_observe_ms`: time spent producing the observation in the env process.
- `env_to_parent_ipc_ms`: env-process observation message send to parent receive.
- `parent_observation_handle_ms`: parent handling time before sending to worker.
- `parent_to_worker_ipc_ms`: parent worker-message send to worker receive.
- `worker_recv_overhead_ms`: worker overhead between receive and measured
  inference start. This includes observation attach time.
- `worker_to_parent_ipc_ms`: worker result send to parent receive.
- `parent_release_ms`: parent time to convert a completion into an action event.
- `parent_to_env_ipc_ms`: parent action dispatch send to env receive.
- `env_action_recv_to_enqueue_ms`: env action receive to action queue enqueue.
- `env_scheduler_update_ms`: env-side action scheduler update.

## Worker Diagnostic Stages

These fields are reported separately and do not sum into `bench_overhead_ms`:

- `worker_observation_attach_ms`: worker-side time to attach the observation
  transport. With the slot pool this should be near zero after first attach.
- `worker_observation_localize_ms`: time to copy cross-process observation arrays
  into worker-local contiguous buffers. This is included in
  `inference_latency_ms`.
- `policy_build_example_ms`: StarVLA observation-to-example preprocessing time.
- `policy_wrapper_predict_action_ms`: StarVLA wrapper/model action prediction
  time.
- `policy_decode_ms`: conversion from wrapper output to benchmark `PolicyOutput`.
- `env_step_ms`: env step duration after action selection.
- `env_wall_lag_ms`: wall-clock lag against the target realtime frame schedule.

## Debugging Host Handoff Slow Paths

If realtime inference is much slower than a direct in-process policy baseline,
inspect:

- `worker_observation_localize_ms`: large values mean copying/localizing the
  image handoff is expensive.
- `policy_build_example_ms`: large values mean image preprocessing/PIL conversion
  is expensive.
- `policy_wrapper_predict_action_ms`: large values mean model/wrapper prediction
  is slow under the realtime process shape.
- `worker_recv_overhead_ms` and `worker_observation_attach_ms`: large values mean
  IPC or shared-memory attachment is expensive.

The expected healthy shape is:

- `worker_observation_attach_ms`: near zero after startup.
- `worker_observation_localize_ms`: small relative to model prediction.
- `policy_wrapper_predict_action_ms`: close to a direct worker-local baseline.
