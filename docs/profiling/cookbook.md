# Measure and replay deployment latency

Profiling uses a realtime evaluation configuration.
The configuration selects the environment, policy, devices, clocks, seeds, and episode budget.
The `profiling` section adds repetition counts and artifact labels.
Install the selected runtime through the [installation guide](../QUICK_START.md).

## Run the CPU example

From the repository root, run:

```bash
export PYTHONPATH="$PWD/configs/examples/custom_env${PYTHONPATH:+:$PYTHONPATH}"
python -m latency_bench profile run --config configs/examples/custom_env/profile.yaml
```

The example measures a constant policy in MountainCar.
It uses CPU inference, two repetitions of three episodes, and a 50-step limit.
Environment FPS is 50. Observation FPS is 10. The episode seed is 7.
See [external components](../custom_components.md) to run the same example outside the checkout.

## Measure your model

Select the matching task recipe under `configs/examples/<task>/profile_<model>.yaml`.
Create an overlay rather than editing an archived experiment configuration.
For MIKASA QwenOFT, save this as `my_profile.yaml` in the repository root:

```yaml
__base__: configs/examples/mikasa_intercept_grab_fast/profile_qwenoft.yaml
policy:
  source: local
  root: /ABSOLUTE/MODEL_BUNDLE
  backbone_path: /ABSOLUTE/BACKBONE_SNAPSHOT
executor:
  mode: realtime
  inference_devices: [cuda:1]
  parallel_envs: 1
  sync_cuda: true
evaluation:
  episodes: 10
  max_env_steps: 60
  suites: []
logging:
  output_dir: outputs/my_profile
  video:
    enabled: false
profiling:
  model_id: my_model
  workload_id: mikasa_intercept_grab_fast
  repeats: 3
```

Replace the model and backbone paths with compatible prepared artifacts.
The inherited recipe supplies the task, input contract, scheduler, and seeds.
Run the measurement:

```bash
conda activate starvla_rl_games_openvla
CUDA_VISIBLE_DEVICES=0,1 python -m latency_bench profile run --config my_profile.yaml
```

The example uses logical GPU 0 for MIKASA and GPU 1 for inference.
`CUDA_VISIBLE_DEVICES` defines the logical device numbers.
Use `[cpu]` for CPU inference or list each intended model worker explicitly.
Multiple entries create multiple inference workers.
Profiling requires one environment and no evaluation suites.

`repeats: 3` creates three independent measurement processes.
Each process loads the model, warms it, and measures ten episodes.
Warmup episodes are separate from measured episodes.
Each repetition uses the same episode seed protocol.
`model_id` and `workload_id` label artifacts. They do not select implementations.
Each label contains one path segment.

Disable video for formal measurements.
Profiling enables its required traces.
Realtime measures actual latency and ignores the simulated `latency` section.
Keep clocks, policy inputs, and resource settings with the reported measurement.

## Read results and recover a run

The command prints these fields:

```text
RUN_DIR
PROFILE_PATH
MEASUREMENT_STATUS
FIT_STATUS
```

Measurements appear under `logging.output_dir/runs/`.
The fitted profile appears under `logging.output_dir/profiles/`.
Keep the companion files beside `profile.json`.
Check completion status and warmup records before interpreting a latency distribution.

Resume a partial measurement with its recorded run ID:

```bash
python -m latency_bench profile run --config my_profile.yaml --resume RUN_ID
```

Resume checks the recorded configuration, deployment, and process settings.
It reuses complete sessions.
Fit an existing complete measurement without loading the model:

```bash
python -m latency_bench profile fit --run-dir /ABSOLUTE/MEASUREMENT
```

Fitting reads saved traces from every recorded session.
Incomplete sessions prevent a completed fit. The command reuses a completed fit.

Calibration is optional.
Omit `profiling.calibration` or set it to `null` to keep the configured clocks.
To enable calibration, add:

```yaml
profiling:
  calibration:
    target_latency_frames: 2
    worker_utilization_cap: 0.8
```

Calibration records its inputs and selected clocks before measurement.

## Replay the fitted profile

For the CPU example, replace `/ABSOLUTE/PROFILE/profile.json` in
[`replay.yaml`](../../configs/examples/custom_env/replay.yaml) with the returned `PROFILE_PATH`.
Run:

```bash
python -m latency_bench eval --config configs/examples/custom_env/replay.yaml
```

For a task model, save a simulated evaluation overlay.
For example, place `my_replay.yaml` beside the MIKASA `my_profile.yaml` above:

```yaml
__base__: my_profile.yaml
executor:
  mode: simulated
latency:
  method: temporal
  profile_path: /ABSOLUTE/PROFILE/profile.json
  seed: 42
logging:
  output_dir: outputs/my_profile/replay
```

Use the returned profile path.
Run replay through the normal evaluation command:

```bash
conda activate starvla_rl_games_openvla
CUDA_VISIBLE_DEVICES=0,1 python -m latency_bench eval --config my_replay.yaml
```

`iid` samples fitted delays independently.
`temporal` preserves the fitted temporal structure.
The public latency wrapper can also consume a profile.
Transferring a profile to another task transfers timing conditions.
It does not measure that task's deployment latency.

## Paths and task interpreters

YAML `__base__` paths are relative to the file containing them.
Other filesystem inputs are relative to the caller's working directory.
Each model worker must be able to import its configured policy.

MIKASA recipes select `scripts/mikasa/evaluate.py` through `profiling.eval_entrypoint`.
Profiling calls that evaluator with `latency-eval --eval-config <generated.yaml>`.
Run these checkout recipes from the repository root.
Measurement and fitting save local artifacts.
See the [artifact guide](../artifacts.md) for published profiles and their source protocols.
