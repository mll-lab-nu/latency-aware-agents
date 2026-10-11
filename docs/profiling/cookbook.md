# Measure and reuse deployment latency

Profiling accepts an ordinary realtime evaluation configuration. The same
configuration describes the environment, policy, devices, clocks, episodes,
and seed. The `profiling` section describes independent measurement repetitions
and publication targets. Project examples use this same interface.

A run produces ordered request traces and a fitted profile. The profile can
supply latency to simulated evaluation or the public Gymnasium wrapper.
Local measurement and fitting do not require Hugging Face write permission.

## Start with a CPU example

From the checkout root, run:

```bash
PYTHONPATH="$PWD/configs/examples/custom_env:$PYTHONPATH" \
  python -m latency_bench.profiling.cli run \
  --config configs/examples/custom_env/profile.yaml
```

The example uses user-owned factories in `configs/examples/custom_env/components.py`.
It measures a constant policy in MountainCar. It validates the interface, not
trained policy quality. Copy the files into your own project and update their
paths to use the same workflow outside this checkout.

For a model you already evaluate, keep its environment and policy configuration.
Set `executor.mode: realtime` and explicit
`executor.inference_devices`. Configure the repetitions and episodes:

```yaml
profiling:
  model_id: my_model
  workload_id: my_task
  repeats: 5
evaluation:
  episodes: 20
```

`model_id` and `workload_id` are artifact labels. Each must be one path segment.
They do not select an implementation or require registration.
The saved run configuration records policy paths and factory settings.
HF artifact provenance records the downloaded snapshot revision.

`profiling.repeats` is the total number of independent measurements, including
the first. Each repetition starts a new evaluation process, loads the policy,
and performs warmup. `evaluation.episodes` sets the measured episodes per
repetition. The example measures 5 × 20 = 100 episodes. Warmup is separate.

Use `experiment.seed` for the evaluation seed. Each repetition uses the same
seed protocol and is stored under its own `sessions/<session-id>/` directory.
Profiling requires one environment and one workload, without evaluation sweeps.

`logging.output_dir` is the artifact root. Filesystem values use the caller's
working directory. `__base__` references use the containing YAML's directory,
as in ordinary evaluation. The profiler starts environment workers with `spawn` by default. An explicit
`LATENCY_BENCH_ENV_BACKEND_START_METHOD` overrides that default and is recorded
in the resume contract. Factory modules must be importable in each worker's
Python environment. See [custom components](../custom_components.md).

## Select devices and measurement conditions

Use `[cpu]` for one CPU worker or `[cuda:0, cuda:1]` for two GPU workers.
Repeated device names create multiple workers on the same device.
CUDA device numbers follow `CUDA_VISIBLE_DEVICES`. Only configured inference
devices define the measured deployment. A simulator GPU is not automatically
added to the inference pool. Unknown GPU names need no registration.

The profiler retains the configured FPS, observation cadence, policy inputs,
worker interpreter, and executor settings. It enables the required records
and preserves basic request timing. Video is disabled when no video setting is supplied.
Keep `logging.video.enabled: false` for formal measurements.
Warmup uses the existing realtime evaluator's policy and remains outside the
fitting trace. The profiler does not introduce a separate inference path.

The local hardware record contains the full fingerprint and worker mapping.
The public projection omits host names and GPU UUIDs. CPU-only profiles use
`cpu` in the existing deployment coordinate field, named `gpu_class`.
Known GPU names retain their published aliases. Other GPU names use a normalized
form of the reported hardware name.

Calibration is disabled unless `profiling.calibration` supplies its parameters:

```yaml
profiling:
  calibration:
    target_latency_frames: 2
    worker_utilization_cap: 0.8
```

This explicit operation measures the configured inference devices and selects
new environment and observation clocks. It saves the calibration input and
result. Omit it, or set it to `null`, to retain the task's configured clocks.

## Use a model example

Model examples live under `configs/examples/profiling/<task>/<model>.yaml`.
They inherit the task configuration and explicitly choose model artifacts,
measurement counts, devices, and clocks. For example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m latency_bench.profiling.cli run \
  --config configs/examples/profiling/flappy/openvla.yaml

CUDA_VISIBLE_DEVICES=0,1 python -m latency_bench.profiling.cli run \
  --config configs/examples/profiling/mikasa_intercept_grab_fast/qwenoft.yaml
```

The Flappy example selects logical `cuda:0` for inference. The MIKASA example
selects `cuda:0` for simulation and `cuda:1` for inference. Additional visible
GPUs do not create additional workers. Edit `executor.inference_devices` when
you intend to measure a different worker configuration.

Examples explicitly select `source: hf`. The resolver records the downloaded
snapshot revision. To use a local bundle, inherit an example and override:

```yaml
__base__: /path/to/checkout/configs/examples/profiling/flappy/openvla.yaml
policy:
  source: local
  root: /models/my_bundle
  backbone_path: /models/my_backbone
```

The inherited artifact specifies the checkpoint and model-config filenames
relative to that bundle. Change them if your bundle uses different filenames.
A missing local file fails. It does not trigger an HF download.

Models use one flat declaration in `policy`:

```yaml
policy:
  type: starvla
  source: hf
  hf_repo: org/models
  revision: optional-pinned-revision
  root: models/my_task
  checkpoint: checkpoints/model.pt
  model_config: config.full.yaml
```

For local files, use `source: local` and a local `root`. The two filenames stay
relative to that root. HF resolution records the snapshot revision and produces
a local root before workers start. `profiling.publish_repo` selects the dataset
repository used by the publication command.

Bundles with `backbone` metadata in the declared model config can resolve that
backbone before workers start. The policy loader owns model configuration and
normalization statistics.

## Resume or fit existing records

```bash
python -m latency_bench.profiling.cli run --config profile.yaml --resume RUN_ID
python -m latency_bench.profiling.cli fit --run-dir /path/to/completed/measurement
```

Resume checks the saved configuration, deployment, environment start method,
and evaluation Python path once before continuing. Changing `profiling.publish_repo`
does not change the measurement protocol. Completed sessions remain unchanged.
An incomplete session runs again from its beginning.
Resume and `fit` require the current local run-contract structure.

`fit` uses the saved run contract, hardware record, completed session traces,
and calibration result when calibration was requested.
It does not load the model or inspect the current machine. It returns the
profile inside the source run. An existing completed fit is reused. An
incomplete fit is rebuilt from the recorded sessions.

For independently collected traces, use
`latency_bench.latency.profile_builder.build_profile_from_traces` directly.
Supply each session's `measure_session_id` and `observation_attempts` path,
the measurement coordinates, FPS, and source metadata. The input must follow
the existing `observation_attempts.jsonl` schema. This operation does not certify
that independently supplied traces used LAGEN's measurement protocol.

## Artifacts

Under `logging.output_dir`, a run writes:

```text
runs/<model>/<deployment>/<task>/<instance>/<run-id>/
  run_contract.json
  hardware.json
  calibration.json         # Only when calibration was requested.
  sessions/<session-id>/
    generated.yaml
    resolved_config.yaml
    observation_attempts.jsonl
    episode_metrics.jsonl
  profile/
profiles/<model>/<deployment>/<task>/<instance>/
  profile.json
  latency_distribution.json
  latency_burst_model.json
  latency_profile.png
  hardware.json
```

The run directory retains its own profile. `profiles/` holds the current
completed profile at that coordinate. A profile contains action-return latency,
worker service-time distributions, worker identities, and temporal burst data.
Capacity drops remain in the ordered trace but do not become latency samples.

Observation-to-action timing starts before `env.observe()` and ends when the
inference parent receives the policy action or chunk. It includes observation
extraction, IPC, preprocessing, inference, and model-side decoding. Subsequent
task-owned controllers and simulator steps are outside that latency boundary.

The Python `run_profile(config)` result and CLI output identify `PROFILE_PATH`,
`RUN_DIR`, `RUN_ID`, measurement completion, fitting completion, and publication
status. Successful fitting does not establish simulated agreement or publication.

## Replay locally

For a comparison using the measured model:

```bash
python -m latency_bench.profiling.cli simulate \
  --config profile.yaml --profile-path /path/to/profile.json \
  --method temporal --seeds 0 1 2
```

This command selects the artifact sources recorded by the profile.
Recorded HF revisions remain fixed. To use a local copy, explicitly configure
its artifact root and provide the matching model files. Methods are `zero`, `temporal`,
`iid`, `fixed`, and `normal`. The target configuration supplies the task clocks.
For a matched-cadence comparison after calibration, copy the selected FPS
values from `calibration.json` into the replay config.
Flappy examples with a separate simulation environment have a corresponding
`<model>_replay.yaml` configuration.

To transfer timing conditions to another model or task, use ordinary simulated
evaluation with `latency.profile_path`, or the public wrapper:

```python
import gymnasium as gym
from latency_bench.envs.latency_wrapper import ActionLatencyWrapper

wrapped = ActionLatencyWrapper(
    gym.make("CartPole-v1"), noop_action=0,
    env_fps=50, obs_fps=10,
    profile_path="/path/to/profile.json",
    profile_method="iid", profile_worker_slot=0,
)
try:
    observation, info = wrapped.reset(seed=7)
    observation, reward, terminated, truncated, info = wrapped.step(1)
finally:
    wrapped.close()
```

This uses the selected timing conditions. It does not claim that the target
model or task was measured. Existing profile files remain usable through the
sampler and wrapper. Sample Factory's existing IID profile training configuration remains
supported by its reference integration.

## Download and publish

Remote operations use explicit artifact addresses and do not inspect local GPUs:

```bash
python -m latency_bench.profiling.cli list --repo MLL-Lab/LAGEN-profiles
python -m latency_bench.profiling.cli resolve \
  --repo MLL-Lab/LAGEN-profiles \
  --ref MODEL/DEPLOYMENT/TASK/INSTANCE --output-dir outputs/profile_downloads
python -m latency_bench.profiling.cli publish \
  --run-dir /path/to/completed/run --repo MY_ORG/MY_PROFILES
```

Downloads contain a complete immutable snapshot, including distribution and
burst-model files. Use the returned `PROFILE_PATH` in downstream configs.
To publish at the end of measurement, set `profiling.publish_repo` and pass `--publish`.
Model downloads and profile uploads are separate operations.
Measurement and simulation uploads omit local run contracts, full hardware
fingerprints, and generated or resolved configurations.

The [sim2real study](../../scripts/experiments/sim2real/README.md) retains its
independent experimental protocol and archived evidence.
