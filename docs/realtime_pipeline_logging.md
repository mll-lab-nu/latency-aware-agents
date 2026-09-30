# Realtime Pipeline Logging

This document defines the runtime timing fields emitted by realtime evaluation.
All durations are milliseconds.

## Output Files

- `latencies.jsonl`: one row per completed inference request. `latency_ms`
  is the measured worker-side action generation latency.
- `e2e_latencies.jsonl`: one row per applied action. `latency_ms` is
  `inference_latency_ms + bench_overhead_ms`.
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
