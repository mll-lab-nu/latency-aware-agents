# Measure deployment latency

Profiling uses an ordinary realtime evaluation configuration. Environment,
policy, devices, clocks, seeds and episode budgets retain their normal meanings.
The additional `profiling` section selects measurement repetitions and artifact labels.

```bash
PYTHONPATH="$PWD/configs/examples/custom_env:$PYTHONPATH" \
  python -m latency_bench profile run --config configs/examples/custom_env/profile.yaml
```

This CPU example measures a constant policy in MountainCar. Its `components.py`
registers the environment and constructs the policy. The same files can live in
an external project when that module is importable by each worker.

## Configuration

For an existing evaluation config, select realtime execution and explicit devices:

```yaml
executor:
  mode: realtime
  inference_devices: [cuda:0]
  parallel_envs: 1
  sync_cuda: true
evaluation:
  episodes: 10
  max_env_steps: 3600
  suites: []
logging:
  output_dir: outputs/profiling
  video:
    enabled: false
profiling:
  model_id: my_model
  workload_id: my_task
  repeats: 3
```

Keep the existing `env`, `policy`, `scheduler` and `experiment` settings.
Model sources use the same local/HF declaration as evaluation.
`model_id` and `workload_id` are artifact labels, each containing one path segment.
They do not select model or task implementations.

`repeats: 3` starts three independent measurement processes. Each loads the model,
warms up, and measures ten episodes. Warmup is separate from the thirty measured
episodes. Each repetition uses the same episode seed protocol.

Profiling requires one environment and no evaluation suites. Multiple inference
workers are declared in `inference_devices`. `[cpu]` selects CPU inference.
Device numbers follow `CUDA_VISIBLE_DEVICES`; a simulator GPU does not implicitly
become an inference worker.

The configured clocks and policy inputs define the measured deployment. Realtime
execution measures actual latency and does not consume a simulated `latency` section.
Formal measurements should disable video. Required traces are enabled by profiling.

Task examples are flat files such as `configs/examples/flappy/profile_openvla.yaml`.
Use `__base__` to share actual common settings. Its paths are relative to the YAML
containing it. Other filesystem inputs follow the caller's working directory.
Workers must be able to import the configured task and policy modules.

## Results and recovery

`run` saves ordered traces, resolved configurations, hardware identity and a fitted
profile. It prints `RUN_DIR`, `PROFILE_PATH`, `MEASUREMENT_STATUS` and `FIT_STATUS`.
Local runs remain under `logging.output_dir/runs/`; the current profile bundle is
under `logging.output_dir/profiles/`. Keep companion files beside `profile.json`.

```bash
python -m latency_bench profile run --config profile.yaml --resume RUN_ID
python -m latency_bench profile fit --run-dir /path/to/completed/measurement
```

Resume checks the recorded configuration, deployment and process settings. Completed
sessions are reused. `fit` requires all recorded sessions to be complete, reads their
saved traces, and does not load the model or measure the current hardware. A completed
fit is reused. Failed or incomplete measurement is not reported as a completed profile.

Calibration is an explicit optional operation. Omit `profiling.calibration` or set
it to `null` to preserve the configured clocks. When configured, calibration saves
its inputs and selected clocks before measurement:

```yaml
profiling:
  calibration:
    target_latency_frames: 2
    worker_utilization_cap: 0.8
```

## Use the profile

Simulated evaluation uses the normal evaluation entrypoint:

```yaml
executor:
  mode: simulated
latency:
  method: temporal
  profile_path: /path/to/profile.json
  seed: 42
```

```bash
python -m latency_bench eval --config eval.yaml
```

The MIKASA profile recipes set `profiling.eval_entrypoint` to
`scripts/mikasa/evaluate.py` to use the task's native evaluator. Profiling invokes
that path with `latency-eval --eval-config <generated.yaml>`. Relative entrypoint
paths use the invocation directory; run these recipes from the repository root.

Specify the environment, policy, resource settings and episode budget in that
configuration. `iid` samples the profile independently; `temporal` preserves its
fitted temporal structure. A profile can also supply the public latency wrapper.
Using a profile on another task transfers timing conditions, not a measurement
claim about that task or model.
