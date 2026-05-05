# TODOs

## Measured Latency Integration

- `latency.type: measured` is supported on the realtime executor path through `RuntimeLatencyMeasurer`, which wraps `policy.predict(observation)` inside the realtime policy worker and records both inference latency and effective action delay.
- `MeasuredLatencySource` remains only as a compatibility adapter for older code that measured through a latency source. New measured runtime paths should use `RuntimeLatencyMeasurer` directly.
- `SimulatedLatencyExecutor` still calls `policy.predict(observation)` before asking the latency source for `latency_ms`, so simulated mode remains appropriate for zero/fixed/distribution/trace/profile replay latency, not live wall-clock inference measurement.

## Logging Scalability and Output Controls

- `TrajectoryLogger` currently buffers all step/action/latency/metric records in memory before `flush()`. This is fine for smoke tests, but long training runs should use streaming writes or chunked files.
- `TrajectoryLogger.flush()` overwrites JSONL files with mode `"w"`. Resume/append workflows need append mode, run-id directories, or explicit checkpointed log segments.
- Config fields such as `logging.save_step_records`, `logging.save_action_records`, and `logging.save_latency_records` are not wired into `TrajectoryLogger`; the logger currently always writes all four JSONL files.
- `to_jsonable()` handles dataclasses, enums, paths, dicts, and lists, but does not handle numpy arrays, torch tensors, or large binary observations. Real image/tensor logging needs explicit serialization policy.

## Observation Pipeline

- `ObservationBuilder` currently returns the latest observation from `HistoryBuffer`; it does not yet implement frame stacking, resizing, normalization, action history, or latency-aware observation variants.
- The observation pipeline does not inject latency values or prior action history into policy inputs, unlike the older queue-latency scripts.
- The fallback path in `ObservationBuilder` depends on `EnvAdapter.get_state_info()` providing `env_step` and `sim_time_ms`; otherwise it silently defaults both to zero.
- `Observation.data` is typed as `Any`, so real policy integrations need explicit schema conventions for image shapes, dtypes, dict keys, and tensor/array ownership.

## Policy and Scheduler

- `FixedFrequencyDecisionScheduler` rounds `fps_env / frequency_hz` to an integer frame interval, so non-integer ratios can drift from the requested long-run control frequency.
- `DROP_EXPIRED` currently applies the latest fresh ready event and drops older fresh ready events; this stale-action policy should be documented explicitly or split into configurable behavior.
- `ActionScheduler.enqueue()` sorts the full pending list after each append; high-action-rate workloads should use a heap or ordered queue.
- Dropped `ActionEvent`s only store `dropped=True`; they do not record drop time, drop step, or drop reason.
- Policy runners are minimal test runners only. Real model integration still needs model loading, batching, device placement, preprocessing/postprocessing, deterministic/stochastic modes, and measured-latency-aware inference.
