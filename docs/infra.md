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

## Environment layers

External environments and policies can use importable factories without editing
this repository. See [custom components](custom_components.md) for the factory
contracts, Python construction, and a standalone evaluation example.

`latency_bench/envs/base.py` defines `EnvAdapter`, the raw RGB metadata key and
`RawRgbFrameStackBuffer`. The buffer copies frames and exposes an oldest-to-newest
RGB stack for VLA inference and dataset export, separately from the teacher's
resized or grayscale observations. `factory.py` constructs benchmark adapters;
`atari.py` holds the shared true-episode versus life-loss semantics.

`envs/adapters/` contains the Gymnasium, Flappy, DemonAttack, ViZDoom and dummy
benchmark adapters. `envs/tasks/` contains native task wrappers and registration,
with HumanoidBench controls/assets/contracts grouped under `tasks/humanoidbench/`.
Native tasks keep the Gymnasium interface; the benchmark adapters implement
`EnvAdapter`. The Swimmer MuJoCo 3 XML lives in `tasks/assets/` and is included in
the Python package.

### Public training wrapper

`envs/latency_wrapper.py` exposes `ActionLatencyWrapper` for user-owned
Gymnasium training loops. It imports neither `training` nor the reference
training frameworks. The caller supplies the native action used while no
prediction is ready; the wrapper does not infer that action from the space.

```python
import gymnasium as gym
from latency_bench.envs.latency_wrapper import ActionLatencyWrapper

env = ActionLatencyWrapper(
    gym.make("MountainCar-v0"),
    noop_action=1,  # MountainCar's coast action.
    env_fps=50,
    obs_fps=10,
    fixed_latency_raw_frames=2,
    hold_policy="one_frame_then_noop",
    ordering_policy="latest_ready",
)
try:
    observation, info = env.reset(seed=7)
    observation, reward, terminated, truncated, info = env.step(2)
finally:
    env.close()
```

Here one wrapper step advances up to five native steps. The submitted action
becomes ready after two native steps. Rewards are summed over the steps that
actually run; termination or truncation stops advancement immediately. The
declared `env_fps` defines the benchmark clock, not measured simulator speed.
Changing `obs_fps` changes observation cadence, not action readiness granularity.
Simulator and policy wall time do not advance this simulated clock.

Profile replay uses the existing `profile_path`, `profile_method` (`iid` or
`temporal`), and `profile_worker_slot` arguments. Waiting, result ordering,
worker capacity, chunk timing, and reset seed behavior retain the existing
execution contracts. A two-dimensional action array represents a chunk whose
rows follow the raw-frame timeline from the source observation. Execution
skips rows whose intended frames have already passed.

`envs/latency_info.py` constructs shared scalar/GPU decision metadata.
`scheduler/simulated_timeline.py` owns simulated result delivery and worker
capacity for both wrappers and executors. Executor event construction stays in
`executors/_simulated_events.py`. Reward transforms and training episode
statistics live in `training/common/metrics_wrappers.py`; Sample Factory
argument conversion stays in `training/common/latency_config.py`. Generic
profile sampler operations live in `latency/samplers.py`.

MIKASA environment construction lives in `tasks/mikasa.py`; its native batched
`MikasaEnvStepBackend` lives in `adapters/mikasa.py`. The task module exposes the
pinned MIKASA source checkout to the simulator interpreter, while native imports
remain inside the construction/reset functions so inference-worker spawn does
not require ManiSkill. The existing `scripts/mikasa/evaluate.py` CLI still owns
PPO evaluation and explicitly injects the batched backend for simulated runs.
Realtime uses the factory's existing single-slot backend adapter. Automatic
backend selection through `latency_bench.run` is unchanged.

Current configs and Python callers use these paths. Saved checkpoint and dataset
contracts can still contain the former flat registration module names. The
explicit mapping in `tasks/contracts.py::registration_module_name` is applied at
environment construction and contract-comparison boundaries; it changes only
known module names. It does not rewrite environment IDs, actions, states, clocks
or archived files. Unknown/custom module names keep normal import behavior.
Archived Pi05 source metadata and its SHA256 manifest remain byte-for-byte intact.

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

The executor-facing `policy` package owns `PolicyRunner` implementations and
their observation-to-action conversion. `policy/sample_factory.py` contains
the Sample Factory runner, its environment-specific construction, and the thin
adapter to training integrations. Shared checkpoint/device and action-space
helpers remain in `integrations/sample_factory` for policy, training, and teacher
dataset consumers.

For StarVLA, `integrations/starvla_runtime.py` owns lazy framework imports and
saved model configuration compatibility. `policy/_starvla_input.py` owns the
shared frame plan, raw-image layout conversion, and lazy stitching;
`policy/_starvla_actions.py` converts model action payloads into benchmark actions.
The generic runner assembles prompts/state and orchestrates inference in
`policy/starvla.py`; its existing public image-conversion export delegates to
the input module. `policy/starvla_tasks.py` remains the separate task-contract
path for multi-view inputs and full action chunks. The training evaluation and
resident DAgger integrations continue to reuse these policy interfaces.

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
