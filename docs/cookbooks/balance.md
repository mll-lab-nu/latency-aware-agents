# HumanoidBench Balance Simple

Install the task and model dependencies:

```bash
bash install.sh --env g1_sonic_balance_simple --model openvla --tier dev
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate latency
python -m latency_bench.envs.tasks.humanoidbench.assets \
  --cache-dir results/humanoidbench_g1_sonic_assets
```

Run commands from the repository root.
Use `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` for the reference offscreen renderer.
See [installation](../installation.md) for host libraries and interpreters.

## Choose the matching contract

The student uses head-camera RGB and physical state at 50 Hz.
H8 and H16 predict chunks of 43D absolute joint targets.
The SONIC name identifies scene assets and joint order. The student does not load a SONIC policy.
Read the exact input dimensions and horizon from the bundle's `task_contract.json`.

| Existing input | Next step |
| --- | --- |
| Model bundle | Evaluate or profile it with its saved contract |
| H1 FastTD3 teacher | Collect H8/H16 labels and convert them |
| Existing experiment directory | Resume its saved SFT/DAgger schedule |
| Converted data | Use your trainer with the saved task contract |

`zero_full` labels the full teacher chunk.
`profile_conditioned` preserves the committed action prefix and labels its suffix.
The conditioned input includes causal action context.

## Collect and convert teacher labels

Save this overlay as `my_collect.yaml` in the repository root:

```yaml
__base__: configs/examples/g1_sonic_balance_simple/collect.yaml
policy:
  source: local
  root: /ABSOLUTE/TEACHER_BUNDLE
  checkpoint: checkpoint.pth
  model_config: fasttd3_train.yaml
logging:
  output_dir: outputs/balance/dataset
```

Use the teacher weights and resolved FastTD3 training configuration from the same run.
The inherited recipe uses H8 `zero_full` labels and a 71,000-row budget.
Its seed is 4100001. Its latency seed is 271828.
Replace `collection.teacher_revision` with your source revision when using another teacher.

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m latency_bench collect --config my_collect.yaml
python -m latency_bench convert --config configs/examples/g1_sonic_balance_simple/convert.yaml
```

Collection writes canonical data under `outputs/balance/dataset/`.
Conversion writes `outputs/balance/lerobot/`, its validation companion, and the task contract.
For H16, set `collection.action_horizon: 16` and select new output paths.
Save a matching converter overlay as `my_convert_h16.yaml`:

```yaml
__base__: configs/examples/g1_sonic_balance_simple/convert.yaml
conversion:
  task: balance_profile_h16
  input_dir: outputs/balance/h16/dataset
  output_dir: outputs/balance/h16/lerobot
```

Set the H16 collection output to `outputs/balance/h16/dataset`.
Run `python -m latency_bench convert --config my_convert_h16.yaml` after that collection completes.
Keep normalization consistent with the selected training contract.
See [collection](../guides/rollout_collection.md) for converter settings.

## Run the SFT/DAgger recipe

Prepare these inputs:

- H1 FastTD3 teacher weights and the matching `fasttd3_train.yaml`.
- The model backbone selected by the experiment, stored locally.
- The StarVLA interpreter and a new experiment output directory.
- A fitted profile when using `profile_conditioned` labels.

Run a short H8 check:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0,1 \
  python -m scripts.tasks.humanoidbench_balance_simple.vla_profile_dagger \
  --output outputs/balance-smoke \
  --vla-python /ABSOLUTE/VLA/bin/python \
  --backbone /ABSOLUTE/BACKBONE \
  --teacher /ABSOLUTE/TEACHER/checkpoint.pth \
  --teacher-config /ABSOLUTE/TEACHER/fasttd3_train.yaml \
  --model qwenoft --horizon 8 --label-mode zero_full \
  --gradient-checkpointing --smoke
```

For full training, omit `--smoke` and select a new output directory.
The full schedule uses 5,000 bootstrap updates and ten DAgger rounds of 500 updates each.
Global batch is 128. The default uses two training GPUs.
The run saves its exact budgets, seed lists, environment configuration, data references, and bundles.
A short run checks execution. It does not establish closed-loop performance.

Use `--horizon 16` for H16.
For conditioned training, use `--label-mode profile_conditioned --profile-path /ABSOLUTE/PROFILE/profile.json`.
`--resume` continues the saved experiment in its existing output directory.
The recipe saves its selected model bundle, evaluation results, and provenance locally.
W&B is disabled by default. Add `--wandb-project YOUR_PROJECT` and optionally `--wandb-entity YOUR_TEAM` to enable it.

The combined recipe prepares its bootstrap data from the teacher.
For independently converted data, use your training code with its task contract.
The combined recipe does not provide a standalone dataset argument.

## Evaluate an existing bundle

Use the saved experiment environment to preserve the task and latency protocol:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m scripts.starvla.evaluate \
  --task balance --bundle /ABSOLUTE/MODEL_BUNDLE \
  --environment /ABSOLUTE/EXPERIMENT/environment.json \
  --output /ABSOLUTE/NEW_RESULTS/baseline \
  --prefix-mode none \
  --vla-python /ABSOLUTE/VLA/bin/python \
  --backbone /ABSOLUTE/BACKBONE
```

The output directory must be new.
For a conditioned bundle, select a new output directory and use
`--prefix-mode profile_mean --prefix-profile /ABSOLUTE/PROFILE/profile.json`.
The evaluator fixes the prefix length at `ceil(profile.summary.mean_ms * env_fps / 1000)`.
Temporal profile draws still determine actual action readiness.
Use the same environment and profile when comparing bundles.

Published comparisons identify their checkpoint, episode count, clocks, seeds, and model-specific hardware profile.
The [artifact guide](../artifacts.md) links those experiment manifests.
Keep those protocols with the reported results.

## Measure deployment latency

Save this overlay as `my_profile.yaml` in the repository root:

```yaml
__base__: configs/examples/g1_sonic_balance_simple/profile_qwenoft.yaml
policy:
  source: local
  root: /ABSOLUTE/MODEL_BUNDLE
  backbone_path: /ABSOLUTE/BACKBONE
logging:
  output_dir: outputs/balance/profiling
```

Select a bundle compatible with the inherited task and state inputs.
For a conditioned bundle, also provide its matching `policy.action_prefix` declaration.
Run measurement:

```bash
conda activate starvla_rl_games_openvla
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 \
  python -m latency_bench profile run --config my_profile.yaml
```

The inherited recipe uses five repetitions of 50 episodes, seed 0, 50/50 Hz, and a 1,000-step limit.
It uses logical GPU 0 for inference.
The command returns the fitted profile path and measurement status.
The [profiling cookbook](profiling.md) explains replay and recovery.
