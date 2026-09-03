# HAIC Pull Cart (G1)

This cookbook covers the zero-delay `G1/haic/pull_cart` pipeline. The
[Chinese internal runbook](../../program/HAIC.md) is the source for hardware,
runtime diagnostics, and the historical latent-only attempts.

## Fixed contract

- Environment: `G1/haic/pull_cart`, seed 0, zero delay, native 50 Hz control.
- Teacher: native HAIC PPO with a 605D normalized policy state. The completed
  500M-frame teacher is the fixed label source for the VLA route.
- VLA input/output: 256x256 RGB plus the 605D state at 10 Hz, predicting a
  `[40, 256]` privileged-latent chunk. A frozen native `actor_adapt` consumes
  chunk entries 0 through 4 with the corresponding current 50 Hz actor states;
  the VLA then replans. One latent is not repeated five times.
- Bootstrap: 250 successful train episodes and 25 validation episodes, fresh
  action-only SFT from the official GR00T base; the loss is 23D motor-action
  MSE through the frozen actor.
- Sampler: explicitly use the official defaults of horizon 40, fresh Gaussian
  noise, and four Euler steps. The objective remains custom action-only rather
  than the official random-time velocity MSE.
- DAgger: 10 rounds, 500 optimizer updates per round, fixed 50/50 bootstrap and
  accumulated DAgger sampling. Every round uses zero-delay rollout and a fixed
  first-episode-per-environment evaluation.

## Quick start

Install the pinned HAIC/Isaac environment, authenticate to Hugging Face and
W&B, then run from the repository root:

```bash
bash scripts/tasks/haic_pull_cart/train_small_policy.sh
```

Use the printed accepted teacher checkpoint for the complete action-only VLA
route:

```bash
bash scripts/tasks/haic_pull_cart/train_vla.sh \
  /absolute/path/to/checkpoint_final.pt
```

The VLA entrypoint exports the fixed actor, runs the 100-episode oracle gate,
collects and converts the 250/25 bootstrap dataset, trains 5,000 SFT updates,
runs DAgger rounds 0 through 9, and performs the final fixed-seed zero-delay
evaluation. It does not upload artifacts automatically.

For an SFT-only run, stop before round 0:

```bash
bash scripts/tasks/haic_pull_cart/train_vla.sh \
  /absolute/path/to/checkpoint_final.pt --end-round 0
```

Resume a later round with explicit bootstrap and actor artifacts:

```bash
bash scripts/tasks/haic_pull_cart/train_vla.sh \
  /absolute/path/to/checkpoint_final.pt \
  --resume \
  --start-round 4 --end-round 10 \
  --bootstrap-checkpoint /absolute/path/to/bootstrap/checkpoint-5000 \
  --bootstrap-dataset /absolute/path/to/bootstrap/lerobot \
  --actor-checkpoint /absolute/path/to/actor_adapt.pt
```

Use the exact checkpoint and dataset paths from the previous run. The task
runner does not choose a replacement when one is missing.

## Manual stages

The task-owned entrypoint composes the existing implementation:

```text
scripts/official_gr00t/train_haic_dagger.py --run-bootstrap
HAIC oracle evaluation and frozen actor export
teacher rollout -> NPZ + MP4 shards -> video-backed LeRobot
action-only SFT -> DAgger round 0..9 -> fixed-seed evaluation
```

Each raw row is one 50 Hz step carrying RGB, 605D state, 256D teacher latent,
605D actor input, 23D teacher action, and a scalar termination flag. Training
assembles 40-step windows and pads episode tails, with post-terminal entries
masked from the loss. The action-only recipe starts from fresh Gaussian noise,
runs the same four-step Euler sampler used by inference with gradients enabled,
then computes the 23D motor-action MSE through the frozen actor. The teacher
latent is stored for chunk labels/data provenance but is not the optimized
target. Keep the fixed actor outside the VLA checkpoint.

The frozen-actor gate must feed a fresh teacher latent at every 50 Hz control
step. This oracle reaches 94/100; holding one teacher latent for five steps is
a different, invalid test and previously produced the misleading 18/100 result.

The round lifecycle is:

```text
bootstrap evaluation -> round rollout -> conversion -> 500-update training -> round evaluation
```

A completed round keeps its converted dataset, training manifest, checkpoint,
and evaluation result. An interrupted collection is restarted from an empty
partial round directory; a completed round is resumed with its explicit
zero-based round number.

## Status and artifacts

The native 500M-frame teacher is complete and accepted as the VLA label source.
The previous action-only runs are invalidated, and no formal h40 retraining is
currently running; therefore no HAIC VLA is marked publishable. Planned model and dataset paths, plus old source
provenance, are tracked in the [artifact registry](../artifacts.md).

The reader workflow requires an online Hugging Face token and the FFmpeg
runtime used by the pinned GR00T release. Keep both credentials out of scripts,
logs, and published artifacts.
