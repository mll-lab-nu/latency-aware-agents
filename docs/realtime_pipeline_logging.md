# Realtime timing and saved traces

This document defines the runtime timing fields emitted by realtime evaluation.
All durations are milliseconds.

## Output Files

- `latencies.jsonl`: worker measurements collected before episode metric finalization.
  `latency_ms` measures worker-side action generation.
- `observation_attempts.jsonl`: every observation attempt, in observation order.
  Admitted requests include completion measurements, including results received after the terminal step.
- `e2e_latencies.jsonl`: one row per applied action. `latency_ms` is the
  frame-derived `applied_latency_ms`.

Realtime always measures basic timing. `logging.save_latency_records` controls
whether these records are saved. It does not enable measurement.

## Core Latency Fields

- `inference_latency_ms`: worker-side action generation time measured by
  `RuntimeLatencyMeasurer`. It includes worker-local observation copy,
  policy input construction, model/action wrapper prediction, and action decode.
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

Episode, calibration and profile summaries use linear interpolation for percentiles.
For sorted samples, percentile `p` interpolates at index `(n - 1) * p / 100`.
For `[10, 20, 30, 40]`, p50 is 25 and p90 is 37.
Historical summaries retain their original percentile values.

Return and length summaries weight each episode equally.
Task moment summaries combine sample counts, sums and squared sums before calculating their mean and population standard deviation.
Latency summaries calculate percentiles from individual samples. Do not average episode percentiles to obtain a run percentile.
