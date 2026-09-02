# Deployment profiling cookbook

Use this pipeline to measure one model/workload on the GPUs that will serve
inference, then build a latency profile that can be replayed in simulation.

## Quick start

Expose only the inference GPUs and run a registered preset:

```bash
CUDA_VISIBLE_DEVICES=0,1 MUJOCO_GL=egl \
  bash scripts/profile.sh run \
  --model official_gr00t \
  --workload g1_sonic_balance_simple
```

This example starts one GR00T worker per visible GPU, runs the measurement
sessions configured for the workload, and writes everything under
`results/profiling/`. Results stay local unless `--publish` is added.

To resume, rerun the command with `--resume "$RUN_ID"`; completed sessions are
skipped and the incomplete session is rerun.

## Configure a profile

Profiles are registered in `configs/profiles.yaml`. Its top-level
`artifact_root` and `hf_repo` select local and published storage. Add unknown
GPU names to `gpu_models`, then add a workload below its model:

```yaml
gpu_models:
  rtx4090:
    aliases: [NVIDIA GeForce RTX 4090]

models:
  my_model:
    workloads:
      my_task:
        base_config: configs/path/to/realtime_eval.yaml
        checkpoint_path: outputs/path/to/checkpoint
        checkpoint_kind: latest
        session_count: 5
        n_episodes: 100
        evaluation_seed: 0
        calibrate_env_fps: false
        simulation_parallel_envs: 1
        simulation_env: {}
```

`base_config` defines the environment, policy, cadence, and evaluation limits.
The profiler overrides the checkpoint, output directory, episode count, and
realtime logging settings. Use `calibrate_env_fps: false` for workloads with a
fixed native cadence; calibrated workloads also require their calibration
settings in the preset.

Each profile has four coordinates:

```text
<model>/<gpu-deployment>/<workload>/<physical-instance>
```

- `model` and `workload` come from the preset.
- `gpu-deployment` is derived from the visible GPU count and registered GPU model.
- `physical-instance` identifies the exact GPU set and its link topology.

Changing the checkpoint alone does not create another coordinate; it is stored
as profile provenance. Use a distinct model ID when a runtime or precision
variant should have its own latency profile.

## How it works

```text
visible GPUs -> realtime sessions -> observation attempts -> fitted profile
```

Every observation opportunity is written to `observation_attempts.jsonl`.
Admitted requests record their worker slot, observation-to-action latency, and
worker service time. If all workers are busy, the row records a capacity drop
instead of inventing a latency sample.

The builder pools admitted samples for per-slot distributions and preserves
session order for the temporal burst model. Pipeline-stage and episode metrics
are useful diagnostics, but they are not profile inputs.

## Artifacts and replay

```text
results/profiling/
  runs/<model>/<gpu>/<workload>/<instance>/<run-id>/
    sessions/<session-id>/observation_attempts.jsonl
    profile/
  profiles/<model>/<gpu>/<workload>/<instance>/
    profile.json
    latency_distribution.json
    latency_burst_model.json
    hardware.json
    DONE
  simulations/<model>/<gpu>/<workload>/<instance>/<run-id>/<method>/<seed>/
```

`runs/` keeps raw measurements, generated configs, warmup diagnostics, pipeline
summaries, and episode metrics. `profiles/` is the current completed profile
for each coordinate. `profile.json` is the human-readable manifest,
`latency_distribution.json` contains per-slot latency/service-time samples, and
`latency_burst_model.json` contains the temporal model.

The run command prints `PROFILE_REF`, `PROFILE_PATH`, `RUN_ID`, `GPU_CLASS`, and
`INSTANCE_ID`. Use those values for the remaining commands:

```bash
bash scripts/profile.sh publish --model my_model --workload my_task
bash scripts/profile.sh resolve --model my_model --workload my_task

bash scripts/profile.sh simulate \
  --model my_model --workload my_task \
  --gpu "$GPU_CLASS" --instance "$INSTANCE_ID" \
  --method temporal --seeds 0 1 2
```

Replay methods are `temporal`, `iid`, `fixed`, and `normal`.
`resolve` and `simulate` use `main` by default; pass `--hf-commit` to pin a
published profile for a reproducible experiment.

## Train a small model with IID profile latency

Sample Factory training can use the same canonical profile manifest with IID sampling:

```yaml
latency:
  method: iid
  profile_path: results/profiling/profiles/<model>/<gpu>/<workload>/<instance>/profile.json
  profile_worker_slot: 0
  seed: 42
```

`profile_worker_slot` selects one real worker slot from the profile. Every training
environment samples that slot's `all` distribution with its own deterministic stream.
The sampled `observation_to_action_latency_ms` is converted to the training environment's
raw-frame clock and applied by the existing action queue.

This training mode is intentionally action-only: it does not model worker service
occupancy or capacity drops. Use `simulate --method iid` when those worker-pool effects
are required. Sample Factory training currently supports `iid` only; temporal burst
replay remains an executor-level feature. Profiles must use the canonical schema 3
distribution generated from `observation_attempts.jsonl`; older single-array profile
artifacts must be rebuilt before training.
