# Deployment profiling cookbook

Use this pipeline to measure one model/workload on the GPUs that will serve
inference, then build a latency profile that can be replayed in simulation.

## Quick start

Use the shared profiler with a registered model/workload pair and the final
model directory:

Use `--hf-repo-id`, `--hf-path`, `--hf-checkpoint`, and `--hf-model-config`
to override the preset Hugging Face artifact coordinates when you want to profile
a different published run without editing `configs/profiles.yaml`.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/tasks/flappy/profile.sh openvla [MODEL_PATH]
CUDA_VISIBLE_DEVICES=0 bash scripts/tasks/flappy/profile.sh wanoft [MODEL_PATH]
CUDA_VISIBLE_DEVICES=0 bash scripts/tasks/flappy/profile.sh pi05 [MODEL_PATH]
CUDA_VISIBLE_DEVICES=0 bash scripts/tasks/flappy/profile.sh gr00t [MODEL_PATH]
CUDA_VISIBLE_DEVICES=0 bash scripts/tasks/demon_attack/profile.sh openvla [MODEL_PATH]
CUDA_VISIBLE_DEVICES=0 bash scripts/tasks/demon_attack/profile.sh wanoft [MODEL_PATH]
CUDA_VISIBLE_DEVICES=0 bash scripts/tasks/demon_attack/profile.sh pi05 [MODEL_PATH]
CUDA_VISIBLE_DEVICES=0 bash scripts/tasks/demon_attack/profile.sh gr00t [MODEL_PATH]
CUDA_VISIBLE_DEVICES=0 bash scripts/tasks/deadly_corridor/profile.sh openvla [MODEL_PATH]
CUDA_VISIBLE_DEVICES=0 bash scripts/tasks/deadly_corridor/profile.sh wanoft [MODEL_PATH]
CUDA_VISIBLE_DEVICES=0 bash scripts/tasks/deadly_corridor/profile.sh pi05 [MODEL_PATH]
CUDA_VISIBLE_DEVICES=0 bash scripts/tasks/deadly_corridor/profile.sh gr00t [MODEL_PATH]
CUDA_VISIBLE_DEVICES=0 bash scripts/tasks/defend_the_line/profile.sh openvla [MODEL_PATH]
CUDA_VISIBLE_DEVICES=0 bash scripts/tasks/defend_the_line/profile.sh wanoft [MODEL_PATH]
CUDA_VISIBLE_DEVICES=0 bash scripts/tasks/defend_the_line/profile.sh pi05 [MODEL_PATH]
CUDA_VISIBLE_DEVICES=0 bash scripts/tasks/defend_the_line/profile.sh gr00t [MODEL_PATH]
CUDA_VISIBLE_DEVICES=0,1 bash scripts/tasks/mikasa_intercept_grab_fast/profile.sh [MODEL_PATH]
CUDA_VISIBLE_DEVICES=1 bash scripts/profile.sh run --model qwenoft --workload g1_sonic_balance_simple --model-path MODEL_PATH
CUDA_VISIBLE_DEVICES=0,1 bash scripts/profile.sh run --model qwenoft --workload extreme_parkour_go2 --model-path MODEL_PATH
CUDA_VISIBLE_DEVICES=0,1 bash scripts/profile.sh run --model qwenoft --workload haic_pull_cart --model-path MODEL_PATH
```

The Demon Attack launchers use the model-specific StarVLA environment active in
the current shell; run them from the matching `starvla_rl_games_*` environment or
pass `--hf-artifacts` to download the registered model artifact.

Balance uses all visible GPUs for inference. The other three tasks use logical
`cuda:0` for the native simulator and `cuda:1..` for inference, so expose at
least two GPUs. The launcher and its inference workers use the currently active
Python environment.

Every task runs five sessions of 10 episodes and writes under
`results/profiling/`. Results stay local unless `--publish` is added. Publishing
starts only after the complete run and fitted profile exist:

```bash
bash scripts/profile.sh run --model MODEL --workload WORKLOAD --model-path MODEL_PATH --publish
```

To resume, rerun the same command with `--resume "$RUN_ID"`; completed
sessions are skipped and the incomplete session is rerun. The local
`run_contract.json` must exactly match the resolved artifacts, interpreter,
entrypoint, deployment and generated config. Legacy runs without this contract,
or runs created with another checkpoint/config, cannot be resumed.

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
        model_artifact:
          local_root: outputs/path/to/model-root
          checkpoint: .
          hf_repo: latency-sensitive-bench/benchmark-models
          hf_path: zero-latency/my-task/vla/my-model-v1
        checkpoint_kind: latest
        session_count: 5
        eval_episodes: 10
        evaluation_seed: 0
        calibrate_env_fps: false
        simulation_parallel_envs: 1
        simulation_gpu: null
        simulation_env: {}
```

`base_config` defines the environment, policy, cadence, and evaluation limits.
The profiler overrides the checkpoint, output directory, episode count, and
realtime logging settings. Use `calibrate_env_fps: false` for workloads with a
fixed native cadence; calibrated workloads also require their calibration
settings in the preset.

`model_artifact` is exact: an absent `local_root` downloads only `hf_path` from
the current Hugging Face repository, while an existing root must contain
`checkpoint`. Native simulator tasks set `eval_entrypoint` and run it with the
Python interpreter active when the command is invoked. A second
`runtime_artifact` may provide a frozen teacher/actor checkpoint. There is no
alternate checkpoint fallback.

The four StarVLA H1 model IDs `qwenoft`, `qwenpi_v3`, `qwengr00t`, and `wanoft`
are registered for both `mikasa_intercept_grab_fast` and
`g1_sonic_balance_simple`. Fetch the latest registered bundle with
`--hf-artifacts`. For Mikasa expose simulator GPU 0 and inference GPU 1; for
Humanoid expose only physical GPU 1, which becomes logical `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/profile.sh run --model qwenoft --workload mikasa_intercept_grab_fast --hf-artifacts
CUDA_VISIBLE_DEVICES=1 bash scripts/profile.sh run --model qwenoft --workload g1_sonic_balance_simple --hf-artifacts
```

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

Observation-to-action starts before `env.observe()` and ends when the inference
parent receives the VLA policy action or action chunk. It includes observation
extraction, IPC, model preprocessing/forward and model-side action decoding. A
task-owned frozen actor, PD controller, actuator and simulator `env.step()` run
after that boundary. Their `env_step_ms` is reported for diagnosis but is not a
profile sample.

The builder pools admitted samples for per-slot distributions and preserves
session order for the temporal burst model. Pipeline-stage and episode metrics
are useful diagnostics, but they are not profile inputs.

Task adapters may publish cumulative episode values through
`StepResult.info["task_metrics"]`. Session summaries report each metric's raw
episode values, mean and standard deviation; these values measure task quality
under latency but never alter the fitted latency distribution.

## Artifacts and replay

```text
results/profiling/
  runs/<model>/<gpu>/<workload>/<instance>/<run-id>/
    run_contract.json                 # local only
    sessions/<session-id>/observation_attempts.jsonl
    profile/
  profiles/<model>/<gpu>/<workload>/<instance>/
    profile.json
    latency_profile.png
    latency_distribution.json
    latency_burst_model.json
    hardware.json
    DONE
  simulations/<model>/<gpu>/<workload>/<instance>/<run-id>/<method>/<seed>/
```

`runs/` keeps raw measurements, generated configs, warmup diagnostics, pipeline
summaries, and episode metrics. `profiles/` is the current completed profile
for each coordinate. `profile.json` is the human-readable manifest,
`latency_profile.png` visualizes the admitted observation-to-action latency density,
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

Replay methods are `zero`, `temporal`, `iid`, `fixed`, and `normal`. `zero`
uses the same generated simulated-eval config as profile replay, but injects
zero virtual latency. `simulate` deliberately has no model-path override: it
uses the artifact coordinates recorded by the selected profile and the current
repository contents.

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
