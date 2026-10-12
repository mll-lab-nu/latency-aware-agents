# Runtime responsibilities

LAGEN provides environment and policy interfaces, execution timing, data collection, evaluation, and deployment profiling.
Training frameworks and experiment recipes consume these capabilities.
See [installation](QUICK_START.md) for dependencies and [configuration](configuration.md) for runtime fields.

## Package boundaries

| Area | Responsibility |
| --- | --- |
| `latency_bench/core/` | Observation and action types, clocks, configuration loading |
| `latency_bench/envs/` | Environment adapters, task registration, public Gymnasium latency wrapper |
| `latency_bench/policy/` | Model construction, preprocessing, inference, native action decoding |
| `latency_bench/scheduler/` | Observation cadence, action readiness, ordering, hold, worker capacity |
| `latency_bench/executors/` | Paused, simulated, batched, and realtime execution |
| `latency_bench/latency/` | Delay samplers, measurements, profile fitting, deployment identity |
| `latency_bench/data/` | Canonical collection, sample alignment, splits, storage |
| `latency_bench/integrations/` | Sample Factory and StarVLA boundaries, training-format conversion |
| `latency_bench/eval/`, `logging/` | Episode orchestration, traces, aggregate metrics |
| `latency_bench/profiling/` | Measurement repetition, calibration, saved-run fitting, recovery |
| `training/` | Reference teacher training and framework compatibility |
| `scripts/tasks/`, `scripts/experiments/` | Task recipes, experiment schedules, publication |

Optional simulators and models load at their integration boundaries.
Public environment and policy interfaces allow components in external packages.
See [external components](custom_components.md) for factory arguments and process ownership.

## Environment and training interfaces

An `EnvAdapter` advances one raw step and supplies an `Observation`.
A `PolicyRunner` converts an observation into `PolicyOutput`.
The policy owns model state and resets it at episode boundaries.
Batched policies use stable slot identities rather than batch positions.

`ActionLatencyWrapper` adds the shared timing contract to a Gymnasium training loop.
The caller supplies the native waiting action.
The wrapper shares simulated readiness and worker capacity with evaluation executors.
Training-specific reward transforms and episode statistics stay in the training layer.
See the [Gymnasium guide](gymnasium_adapter.md) for a complete wrapper example.

Raw RGB history remains separate from the teacher's resized or grayscale observations.
The environment owns rendering and frame snapshots.
The collector pairs student snapshots with issued teacher commands.
See [collection](rollout_collection.md) for label and training-data contracts.

## Execution and resources

The [execution contract](latency_execution_semantics_en.md) defines clocks, delays, ordering, hold, capacity, and chunks.
Simulated time depends on the declared raw-step clock.
Realtime uses actual wall time and records environment lag.
Environment processes and inference workers own their constructed components.
Executors release their environments, processes, and shared resources when they close.

`parallel_envs` controls simulated episode slots.
`inference_devices` controls actual model workers.
`simulated_worker_capacity` controls virtual occupancy independently.
See [parallel evaluation](eval_parallelism.md) for batching and environment backends.

## Model and dataset contracts

Model adapters own preprocessing, action decoding, and recurrent state.
Generic Gymnasium training records normalization in the run's `dataset_statistics.json`.
Validation and inference reuse the selected training DataConfig and statistics.
Task-specific policies use their saved task contract and frozen task statistics.
A dataset or checkpoint's action dimensions, state inputs, and horizon must match its consumer.

Canonical rollout storage stays in `data/`.
StarVLA conversion and task-contract preparation stay in `integrations/starvla/`.
Sample Factory inference and reference training use the pinned source and their owning compatibility modules.

## Saved configurations and evidence

YAML `__base__` paths use the containing file's directory.
Runtime filesystem inputs use the caller's working directory.
Entrypoints resolve inputs and save `resolved_config.yaml` before execution.
Model provenance records repository revisions and selected files.
See [artifacts](artifacts.md) for portable model overlays and published result sources.

Keep logs, traces, configurations, checkpoints, and full reports in `outputs/` or a published artifact repository.
Keep reusable commands and user contracts in `docs/` or the owning recipe's README.
Report training completion, evaluation completion, acceptance, and publication separately.
The [trace guide](realtime_pipeline_logging.md) defines measurement populations and latency fields.
