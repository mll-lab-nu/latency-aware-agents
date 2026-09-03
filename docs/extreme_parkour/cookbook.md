# Extreme Parkour (Go2)

This cookbook covers the zero-delay Unitree Go2 pipeline on Isaac Lab. The
[Chinese internal runbook](../../program/EXTREME_PARKOUR.md) preserves the
upstream comparison, profiling evidence, and historical failed runs.

## Fixed contract

- Environment: `Isaac-Extreme-Parkour-VLA-Unitree-Go2-v0` for collection,
  DAgger, and VLA evaluation.
- Teacher: privileged PPO, fixed learning rate `2e-4`, 6,144 total environments,
  15,000 iterations, training seed 1.
- VLA input: 256x150 robot RGB at 10 Hz plus masked proprioception. The VLA
  predicts a 32D terrain latent and two normalized yaw values (34D total).
- Fixed actor: the teacher actor maps the VLA output and the five 50 Hz actor
  observations to the 12D Go2 motor action. The actor is frozen and is not
  included in the VLA checkpoint.
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
  /absolute/path/to/model_14999.pt
```

The VLA entrypoint runs the teacher/oracle gate, collects and converts the
250/25 action-rich bootstrap dataset, trains the 5,000-update SFT model, runs
DAgger rounds 0 through 9, and performs the fixed-seed zero-delay evaluation
after each round and at the end. It does not publish artifacts automatically.

To stop after SFT for debugging or a manual baseline, use the runner's explicit
zero-round range:

```bash
bash scripts/tasks/extreme_parkour_go2/train_vla.sh \
  /absolute/path/to/model_14999.pt --end-round 0
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
same seed-2 gate. A failure stops collection. The accepted dataset retains RGB,
53D state, 34D action target, five-step actor observations, and termination
masks. Do not mix the previous latent-only dataset with this action+yaw schema.

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
