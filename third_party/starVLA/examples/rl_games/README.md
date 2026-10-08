# RL Games Pipeline

Detailed guide: `examples/rl_games/USAGE.md`

## Installation

The installer creates one isolated environment per explicitly selected model.
Every model environment includes Flappy, Demon Attack, and Deadly Corridor:

- Python 3.10.x
- CUDA Toolkit 12.8.1
- PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, torchaudio 2.11.0+cu128
- Triton 3.6.0
- CUDA extension targets sm_90 and sm_120

Install one model:

```bash
bash examples/rl_games/install/bootstrap.sh \
  --model openvla
```

Install all five model environments:

```bash
bash examples/rl_games/install/bootstrap.sh \
  --model all
```

The environments are `starvla_rl_games_openvla`, `starvla_rl_games_pi0`,
`starvla_rl_games_pi05`, `starvla_rl_games_gr00t`, and
`starvla_rl_games_wan_oft`. The default tier is `dev`; pass `--tier use` for
runtime-only model packages.

There is no Torch/CUDA auto-detection or alternate version profile. The
bootstrap script fixes the CUDA/PyTorch stack in every environment, while
the Demon Attack installer accepts the AutoROM license automatically when its
ROM is missing.
Transformers, Datasets, Diffusers, JAX, and other model dependencies are
installed independently by each model installer. No dependency lock file is
used.

## Train

The current training flow is Hydra-driven. Compose the run from the canonical config tree and pass the active groups on the command line.

`scratch` trains with the native task action width. `bridge` starts from the released StarVLA Bridge/RT-1 checkpoints, uses `Qwen/Qwen3-VL-4B-Instruct` as the base backbone, and trains through a shared 7D action/state carrier. Losses and inference are masked to the active task action dimensions: 2 for Flappy, 6 for Demon Attack, and 7 for Deadly Corridor.

These pi-0.5 configs use `Qwen/Qwen3-VL-4B-Instruct` as the base backbone and initialize from `StarVLA/Qwen3VL-PI_v3-Bridge-RT_1`. Setup first checks the local initializer at `playground/Pretrained_models/Qwen3VL-PI_v3-Bridge-RT_1/checkpoints/steps_50000_pytorch_model.pt`; if it is missing, it falls back to the Hugging Face repo.

Edit `workspace_dir`, `auth`, `wandb`, `dataset`, `base_model`, `checkpoint`, `launch`, `train_data`, and `trainer` in the YAML. Relative asset paths are resolved under `workspace_dir`.

Authentication tokens are read from `HF_TOKEN` and `WANDB_API_KEY` by default. The RL-games train launchers log into Hugging Face and W&B before asset setup/training when those values are available:

```bash
export HF_TOKEN=HF_TOKEN_VALUE
export WANDB_API_KEY=WANDB_API_KEY_VALUE
```

You can also copy `examples/rl_games/auth.env.example` to a private file such as `WORKSPACE_DIR/auth.env`, then set `auth.env_file: auth.env` in the Hydra config. If `auth.env_file` is null, the launcher auto-detects `WORKSPACE_DIR/auth.env` and `examples/rl_games/auth.env`. Do not commit real tokens.

Mixed-latency Flappy:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=scratch \
  mode=mixed_latency
```

The archived `benchmark-datasets` Flappy config stores one image per row instead of an
explicit context-image column. Convert it directly into the existing WanOFT
LeRobot interface by deriving each row's history inside its episode:

```bash
python examples/rl_games/bash_scripts/gr00t/data_conversion/convert_flappy_history_to_starvla_lerobot.py \
  --dataset-revision 8a5af732702f31824156975ed248ce6534e736ee \
  --image-sequence-length 5

bash commands/wanoft/train_flappy_wan_oft.sh 3
```

The converter writes the final `flappy_train__bridge` and validation datasets
directly. It uses the Hugging Face cache for source parquet shards but does not
materialize an intermediate dataset with duplicated context images.

The pipeline script runs the same conversion and training path
for fixed latency 3 only:

```bash
bash scripts/run_flappy_wan_oft_pipeline.sh
```

The pipeline selects `flappy_fixed_latency_3_200ep_7k2steps` from the fixed
`benchmark-datasets` revision shown above.

The corresponding Demon Attack config can be converted in the same way:

```bash
python examples/rl_games/bash_scripts/gr00t/data_conversion/convert_demon_attack_history_to_starvla_lerobot.py \
  --dataset-revision 8a5af732702f31824156975ed248ce6534e736ee \
  --image-sequence-length 5
```

This writes `demon_attack_train__bridge` and its validation dataset under
`data/demon_attack_fix_latency_6_200ep_context5`. The source latency is recorded
faithfully as 6 raw ALE frames (100 ms), with observations at 15 FPS. It must
not be interpreted as six environment decision steps: with Demon Attack's
frameskip of four, the existing step-based eval queue would turn that into 24
raw frames. Exact online evaluation for this dataset therefore requires a
raw-frame latency evaluator instead of relabeling the converted data.

The archived `benchmark-datasets` Deadly Corridor config also stores only the current
image on each row. Its actions are joint-54 IDs rather than an explicit action
vector. The history converter derives the previous four images within each
episode, validates `action_id` against `action_text`, and writes the decoded 7D
semantic multi-hot action directly:

```bash
python examples/rl_games/bash_scripts/gr00t/data_conversion/convert_deadly_corridor_history_to_starvla_lerobot.py \
  --dataset-revision 8a5af732702f31824156975ed248ce6534e736ee \
  --image-sequence-length 5

MAX_EPISODES=1000 \
  bash commands/wanoft/train_deadly_corridor_wan_oft.sh 6
```

This writes `deadly_corridor_train__bridge` and its validation dataset under
`data/deadly_corridor_fix_latency_6_1000ep_context5`; source parquet shards are
kept in the Hugging Face cache, so no context-enriched intermediate dataset is
materialized. The source has 8.75 FPS observations and a latency of 6 raw
VizDoom frames, which is 1.5 decision steps with frameskip four. The current
online evaluator accepts integer decision-step latency only, so both core
environment evaluation and post-train evaluation are disabled by default
instead of silently changing the latency.

Asterix and Atlantis history conversion requires an explicit dataset config,
source prefix and fixed revision. Their former 1000-episode release sources
were retired. Supply data with the timing and action contracts described below.
The generic training presets accept explicit `dataset.revision` and
`dataset.source_subdir` overrides. Environment-based launches require
`LATENCY_BENCH_DATASET_REVISION` and `LATENCY_BENCH_DATASET_SUBDIR`.

```bash
python examples/rl_games/bash_scripts/gr00t/data_conversion/convert_asterix_history_to_starvla_lerobot.py \
  --dataset-config-name DATASET_CONFIG --dataset-source-prefix DATASET_PREFIX \
  --dataset-revision DATASET_COMMIT --output-dir OUTPUT_DIR --image-sequence-length 5

python examples/rl_games/bash_scripts/gr00t/data_conversion/convert_atlantis_history_to_starvla_lerobot.py \
  --dataset-config-name DATASET_CONFIG --dataset-source-prefix DATASET_PREFIX \
  --dataset-revision DATASET_COMMIT --output-dir OUTPUT_DIR --image-sequence-length 5

python examples/rl_games/bash_scripts/gr00t/data_conversion/convert_defend_the_line_history_to_starvla_lerobot.py \
  --image-sequence-length 5
```

Defend the Line uses its byte-identical retained canonical zero-latency data.
Fixed-source launchers without retained identical datasets were retired.
Deadly Corridor profile launchers use their unchanged nested experiment data.

Asterix and Atlantis preserve the Atari timing used by their VLA datasets:
15 FPS observations, 60 FPS environment frames, and raw-frame frameskip four.
Defend the Line preserves the VizDoom timing used by its VLA dataset: 8.75 FPS
observations, 35 FPS environment frames, and raw-frame frameskip four. Asterix
defaults to the Bridge-compatible `factorized_6` action layout.

Single-latency Deadly Corridor with bridge initialization:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=pi05 \
  env=deadly_corridor \
  init=bridge \
  mode=single
```

Deadly Corridor context-5 training with WanOFT uses the released 7D Bridge
carrier. The raw teacher export stores `deadly_corridor_joint_54` action IDs;
the pipeline decodes those IDs into the equivalent 7D semantic multi-hot
buttons and trains the current action with binary cross-entropy:

```bash
bash scripts/run_deadly_corridor_wan_oft_pipeline.sh
```

The standalone training boundary is:

```bash
bash commands/wanoft/train_deadly_corridor_wan_oft.sh 6
```

Single-latency Deadly Corridor with the pi-0.5 VLA backbone initialized from
Bridge/RT-1, but with a fresh native factorized 11D action head:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=pi05 \
  env=deadly_corridor \
  init=backbone_bridge_factorized11 \
  mode=single \
  run_id=pi05_deadly_corridor_factorized11 \
  checkpoint.load=none
```

Use this mode when you want the trained VLA backbone weights from
`StarVLA/Qwen3VL-PI_v3-Bridge-RT_1`, but do not want to reuse the Bridge action
head or projector. The init config sets:

```text
rl_games.action_carrier=native
rl_games.env_eval.deadly.action_layout=factorized_11
framework.action_model.action_dim=11
framework.action_model.action_env_dim=11
framework.action_model.action_horizon=1
trainer.reload_modules=qwen_vl_interface
```

`trainer.reload_modules=qwen_vl_interface` is the important checkpoint-loading
boundary: setup still resolves the Bridge/RT-1 checkpoint, but training only
loads the `qwen_vl_interface` module from it. `project_layers` and
`action_model` are left as the newly constructed model modules.

The factorized 11D Deadly Corridor layout is:

```text
turn[3] + move[3] + strafe[3] + attack[2]

TURN_NONE, TURN_LEFT, TURN_RIGHT,
MOVE_NONE, MOVE_FORWARD, MOVE_BACKWARD,
STRAFE_NONE, STRAFE_LEFT, STRAFE_RIGHT,
ATTACK_OFF, ATTACK_ON
```

The raw Deadly Corridor source dataset for this mode must provide either:

- `action_tuple`: `[turn, move, strafe, attack]`, where the group sizes are
  `[3, 3, 3, 2]`
- or `action` / `actions`: an already one-hot 11D vector in the order above

If the converted StarVLA/LeRobot dataset already exists under
`paths.dataset_local_dir`, setup reuses it when its manifest matches
`action_layout=factorized_11`. If it does not exist yet, pass the raw source
dataset so setup can verify and convert it:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=pi05 \
  env=deadly_corridor \
  init=backbone_bridge_factorized11 \
  mode=single \
  run_id=pi05_deadly_corridor_factorized11 \
  checkpoint.load=none \
  dataset.source_hf=<deadly_corridor_factorized11_source_dataset>
```

Before launching a long run, dry-run the composition and setup:

```bash
python examples/rl_games/scripts/launch_train.py \
  --dry-run \
  model=pi05 \
  env=deadly_corridor \
  init=backbone_bridge_factorized11 \
  mode=single \
  run_id=pi05_deadly_corridor_factorized11 \
  checkpoint.load=none
```

Do not use `init=bridge` for this experiment. `init=bridge` intentionally uses
the 7D Bridge carrier and is the wrong action surface for factorized 11D.

Override config values from the command line:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=scratch \
  mode=mixed_latency \
  workspace_dir=WORKSPACE_DIR \
  wandb_entity=WANDB_ENTITY \
  run_id=smoke_test \
  trainer.max_train_steps=10 \
  trainer.save_interval=5 \
  trainer.eval_interval=5
```

Fast end-to-end preprocessing debug:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=scratch \
  mode=mixed_latency \
  run_id=debug_flappy_mixed_e2e \
  dataset.debug_subset.enabled=true \
  dataset.debug_subset.max_episodes=5 \
  trainer.max_train_steps=2 \
  datasets.vla_data.per_device_batch_size=1 \
  trainer.distributed_backend=none \
  rl_games.env_eval.mid_train.enabled=false \
  rl_games.env_eval.post_train.enabled=false
```

Cross-task OpenVLA bridge runs use the same launcher with a named setup from
`examples/rl_games/config/cross_task_setup`:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=cross_task \
  init=bridge \
  mode=cross_task \
  cross_task_setup=flappy_mixed_demon_zero
```

Canonical wrappers are available at:

```bash
bash examples/rl_games/bash_scripts/openvla/bridge/cross_task/flappy_mixed_demon_zero.sh
bash examples/rl_games/bash_scripts/openvla/bridge/cross_task/demon_mixed_flappy_zero.sh
```

## Configuration

RL-games training uses one Hydra configuration tree under `examples/rl_games/config`.

The primary config groups are:
- `model`
- `env`
- `init`
- `mode`
- `cross_task_setup`

Use `python examples/rl_games/scripts/launch_train.py ...` to compose runs from those groups. Do not add new YAML files under the legacy `experiments` directory; it is intentionally removed from the active training path.

Single-GPU direct backend with a custom run name:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=scratch \
  mode=single \
  run_id=test_qwen3_flappy_gc_none_backend \
  trainer.distributed_backend=none
```

Debug subsets are written to separate converted dataset folders, for example
`flappy_mixed_latency_train__debug_5ep`, so they do not overwrite the full
preprocessed dataset. The same overrides work for the OpenVLA Demon Attack
single and mixed-latency experiment YAMLs.

The converter also writes a held-out validation LeRobot dataset next to the
training dataset, for example `flappy_mixed_latency_train__val` or
`flappy_mixed_latency_train__debug_5ep__val`. The trainer logs `train/loss`,
`eval/loss`, and `train/grad_norm_pre_clip`.

Deadly Corridor uses the mixed-latency HF dataset for both modes. The single
config filters it to `dataset.latency_filter: [0]`, while the mixed config
keeps all latencies and requires `latency_prompt_map.json`.

Checkpoint fields have separate meanings: `checkpoint.hf_repo_id` is the resume/download source, while `checkpoint.sync.repo_id` is the upload destination when `checkpoint.sync.enabled: true`. The trainer saves full Accelerate training-state directories (`steps_<N>_state/`) for exact resume, including optimizer/scheduler state, and also saves lightweight model files for convenience. A missing `checkpoint.sync.repo_id` repo is created during sync if Hugging Face auth is available. `checkpoint.sync.keep_last_n: 0` keeps all uploaded HF checkpoints.

Environment rollout eval is controlled in the `rl_games` block:

```yaml
rl_games:
  env_eval:
    enabled: true
    mid_train:
      enabled: true
      interval_steps: 100
      latencies: [0, 1, 2, 3, 4, 5]
      num_episodes: 5
      max_steps_per_episode: 2000
    post_train:
      enabled: true
      latencies: [0, 1, 2, 3, 4, 5]
      num_episodes: 5
      max_steps_per_episode: 2000
```

This is separate from `trainer.eval_interval`. `trainer.eval_interval` runs the trainer's action-model eval; `rl_games.env_eval.mid_train` and `rl_games.env_eval.post_train` run the current model in the actual environment.

The launcher activates the conda env from the config, then downloads/prepares the dataset, writes dataset statistics, downloads the base model if needed, checks local/Hugging Face checkpoints, and launches training.

## Eval summary

```bash
bash examples/rl_games/scripts/run_eval.sh --run-dir results/Checkpoints/starvla_rl_games --stage post_train
```
