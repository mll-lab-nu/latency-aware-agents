# Extreme Parkour (Go2)

This cookbook covers the zero-delay Unitree Go2 pipeline on Isaac Lab. The
[Chinese internal runbook](../../program/EXTREME_PARKOUR.md) preserves the
upstream comparison, profiling evidence, and historical failed runs.

## Fixed contract

- Environment: `Isaac-Extreme-Parkour-VLA-Unitree-Go2-v0` for collection,
  DAgger, and VLA evaluation.
- Teacher: privileged PPO, fixed learning rate `2e-4`, 6,144 total environments,
  15,000 iterations, training seed 1.
- VLA input: 256x150 robot RGB at 10 Hz plus masked proprioception. Each call
  encodes the observation once and predicts an H40 chunk of 32D terrain latent
  plus two normalized yaw values (34D per 50 Hz control position).
- Runtime executes chunk positions 0 through 4, then replans at 10 Hz. Their
  labels share one oracle latent/yaw, while the five model predictions are
  consumed separately by the frozen 50 Hz actor.
- Training uses fresh Gaussian noise and the official four-step Euler sampler.
  The actor is frozen and is not included in the VLA checkpoint.
- Bootstrap: 250 successful train episodes and 25 validation episodes, fresh
  action+yaw SFT from the official GR00T base.
- DAgger: 10 rounds, 500 optimizer updates per round, fixed 50/50 bootstrap and
  accumulated-round sampling. Every round uses zero-delay rollout and evaluation.

## Quick start

Install Isaac Sim/Isaac Lab and the pinned official GR00T environment, then run
from the repository root:

```bash
bash scripts/tasks/extreme_parkour_go2/train_small_policy.sh
```

Use the printed `model_14999.pt` teacher checkpoint for the complete VLA path:

```bash
bash scripts/tasks/extreme_parkour_go2/train_vla.sh \
  /absolute/path/to/model_14999.pt \
  --bootstrap-raw outputs/official_gr00t/extreme_parkour_go2_action_data/raw
```

The VLA entrypoint converts the accepted 250/25 action-rich raw dataset, trains
the 5,000-update SFT model, runs
DAgger rounds 0 through 9, and performs the fixed-seed zero-delay evaluation
after each round and at the end. It does not publish artifacts automatically.
Omit `--bootstrap-raw` only when the teacher/oracle gate and collection must be rerun.

To stop after SFT for debugging or a manual baseline, use the runner's explicit
zero-round range:

```bash
bash scripts/tasks/extreme_parkour_go2/train_vla.sh \
  /absolute/path/to/model_14999.pt --end-round 0 \
  --bootstrap-raw outputs/official_gr00t/extreme_parkour_go2_action_data/raw
```

Resume from a completed bootstrap or round with its explicit checkpoint and
round range; for example, continue through round 9 with:

```bash
bash scripts/tasks/extreme_parkour_go2/train_vla.sh \
  /absolute/path/to/model_14999.pt \
  --resume \
  --start-round 4 --end-round 10 \
  --bootstrap-checkpoint /absolute/path/to/bootstrap/checkpoint-5000 \
  --bootstrap-dataset /absolute/path/to/bootstrap/lerobot
```

The exact resume flags are task-runner options; never infer a checkpoint from a
directory name or a `latest` link.

## Profile the final VLA

From the installed Parkour/Isaac Lab Python environment, expose one simulator
GPU plus one or more inference GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/tasks/extreme_parkour_go2/profile.sh [MODEL_PATH]
```

The native environment runs on logical `cuda:0`; GR00T workers use `cuda:1..`.
The fixed actor checkpoint is resolved independently from the registered
small-policy artifact. The action+yaw VLA remote is still pending, so an absent
local model fails until that exact artifact is published. Add `--resume RUN_ID`
or `--publish` after the relevant artifact gate passes.

## Manual stages

The task-owned entrypoint composes these existing implementation stages:

```text
Isaaclab_Parkour/scripts/rsl_rl/parkour_vla.py teacher-eval
Isaaclab_Parkour/scripts/rsl_rl/parkour_vla.py oracle-eval
Isaaclab_Parkour/scripts/rsl_rl/parkour_vla.py collect
scripts/official_gr00t/prepare_parkour_dataset.py
scripts/official_gr00t/train_parkour_dagger.py
Isaaclab_Parkour/scripts/rsl_rl/parkour_vla.py vla-eval
```

The oracle must retain the privileged latent for five 50 Hz control steps and
reach at least 50% of the teacher's mean normalized waypoint progress on the
same seed-2 gate. A failure stops collection. Each accepted raw row retains
10 Hz RGB, 53D state, one 34D action target, five 50 Hz actor observations, and
termination masks. The loader builds H40 from eight consecutive rows and
repeats each row's target five times. Do not mix H1 checkpoints or interleaved
H1 DAgger shards with this H40 schema.

The DAgger lifecycle is:

```text
bootstrap evaluation -> round rollout -> conversion -> 500-update training -> round evaluation
```

Successful training writes the round-complete marker. An interrupted collection
must be restarted from an empty partial round directory; a completed round is
resumed with its explicit checkpoint and zero-based round number.

## Status and artifacts

The privileged teacher and supporting action+yaw implementation are verified.
The current action+yaw bootstrap/DAgger result is still in progress, so no
Parkour VLA is marked publishable. Planned artifact paths and publication gates
are tracked in the [artifact registry](../artifacts.md).

The internal record also documents why the old latent-only 5k result is not a
baseline: its evaluator reused privileged yaw on four of every five control
steps and its objective did not directly constrain motor action.
