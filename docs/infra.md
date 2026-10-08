# Core infrastructure

This source release keeps benchmark code, experiment entrypoints, and pinned
third-party sources in one checkout. Installation and simulator setup live in
[QUICK_START](QUICK_START.md); task commands live in the
[standard pipeline](standard_pipeline.md) and its linked cookbooks.

## Responsibilities

| Area | Owns |
| --- | --- |
| `latency_bench/core`, `scheduler` | Raw-frame time, action lifecycle, ordering and hold semantics |
| `latency_bench/executors` | Paused, simulated, batched and realtime execution; worker and environment cleanup |
| `latency_bench/latency` | Delay sampling, measurement, profile construction and replay |
| `latency_bench/eval`, `logging` | Episode orchestration, trace output and aggregate metrics |
| `latency_bench/envs`, `policy`, `integrations` | Simulator, model and training-framework boundaries |
| `latency_bench/data`, `training`, `scripts` | Dataset conversion, training recipes and user entrypoints |
| `latency_bench/profiling` | Deployment latency measurement, profile resolution/publication and replay |
| `scripts/experiments/sim2real` | Held-out and fit-budget research protocols, experiment replay and reports |

The profiling package is a shared workflow layer exposed by `scripts/profile.sh`.
It depends on runtime, evaluation and latency primitives. Experiment scripts
consume those capabilities; profiling never imports the experiment scripts.
Physical GPU/topology identity belongs to `latency.hardware_info` and retains
its existing artifact identifiers. Pinned tasks, model revisions, split seeds,
episode counts and report parameters belong to experiment recipes and frozen
manifests, rather than the shared profiling API.

The [English](latency_execution_semantics_en.md) and
[Chinese](latency_execution_semantics_zh.md) contracts define experiment timing.
Changes to code organization must preserve observation cadence, delay rounding,
per-episode seeds, submission/ready/application frames, FIFO/latest-ready
ordering, chunk handling, hold behavior and timeout/late-result handling.

Single-environment, batched and realtime loops have distinct scheduling and
resource contracts. Shared memory, worker pools, GPU batching, render-equivalence
conditions and streaming metrics are intentional performance mechanisms. Keep
measurement boundaries and synchronization points stable when refactoring.
Optional simulator/model imports remain lazy where they isolate environments.

## Configuration and artifacts

YAML `__base__` paths are relative to the containing YAML file. Base files are
merged in order and the current file wins; CLI overrides are applied afterwards.
The evaluation boundary expands explicit filesystem inputs relative to the caller's
working directory and passes absolute paths downstream. Bundled runtime resources
use the shared source-checkout anchor in `latency_bench.utils.paths`. Each evaluation
output directory records `resolved_config.yaml` after defaults and action-prefix
preparation, before episodes run. An omitted experiment seed stays omitted:
policy/suite randomness and the evaluation episode seed have distinct contracts.
Pass external checkpoint, manifest, backbone and output paths explicitly. See
[artifact preparation](artifacts.md#prepare-a-backbone-for-evaluation) for a
portable overlay without editing an archived experiment recipe.

Public task commands own artifact preparation before evaluation starts. Policy
inference consumes prepared local inputs. The task StarVLA adapter requires
`policy.backbone_path`; generic StarVLA adapters use that field to override the
checkpoint's stored model configuration when needed. Choose the original model
identity and revision from the artifact provenance, never from a local basename.

## Training-framework boundaries

The Sample Factory source version is fixed by the benchmark repository commit.
Compatibility code in `training/common` has active training consumers:

- `sf_gpu_sampling_compat.py` adapts batched rollout advancement for GPU tensor
  environments, installed by the Flappy and Demon Attack integrations.
- `sf_episode_extra_stats_compat.py` preserves Flappy episode statistics through
  batched sampling; `sf_raw_reward_stats_compat.py` preserves Demon Attack's raw
  reward reporting independently of learning reward scaling.
- `checkpoints.py` owns atomic checkpoint writes and retained evaluation inputs;
  training and periodic evaluation depend on these lifetimes.

These are explicit upstream integration boundaries. Updating the pinned framework
requires the corresponding training, statistics and checkpoint tests, followed by
real training and performance checks. A local code cleanup alone does not justify
removing a patch or changing its installation order.

## Validation and remaining release gates

Use the owning suites under `tests/` and the existing infrastructure acceleration
benchmark. Compare fixed checkpoints, resolved configs, episode IDs, seeds,
latency inputs and FPS. Deterministic cases should preserve actions, application
frames, termination, returns and metrics; use established numerical tolerances.
Exercise resource cleanup as well as successful runs.

On the same hardware, compare warmed-up alternating baseline/refactor runs for
throughput, CPU/GPU work, memory and realtime deadlines. A median regression over
5% requires investigation; it is not an accepted performance budget. Simulator,
GPU and training checks that were not executed remain unverified.

Streaming batched metrics and retained-trace extraction share
`EpisodeMetricAccumulator`. The final step's `game_score` takes precedence over
`score`; neither field on that step means the score is absent. Raw return uses
the latest available value, preferring top-level fields over `episode_extra_stats`
and using the same alias order in both paths. Task metrics come from the final
step. Record retention does not change these episode metrics; the existing
[executor tests](../tests/executors/test_batched_simulated_executor.py) cover both
retention modes. Historical result artifacts retain their original values.

For release acceptance, exercise a fresh checkout with vendored sources and
explicit external artifacts, including evaluation from outside the checkout.
Record the tested revision, environment, and results with the experiment artifacts.
Public result references belong in the [artifact registry](artifacts.md).
Development plans and operational history belong outside the source repository.
