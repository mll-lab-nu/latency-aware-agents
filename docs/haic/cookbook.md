# HAIC Pull Cart (G1)

Install Isaac Sim, the matching Isaac Lab checkout, HAIC, and StarVLA using
[QUICK_START.md](../QUICK_START.md). Historical runtime evidence remains in
[program/HAIC.md](../../program/HAIC.md).

## Contract

- Environment: `G1/haic/pull_cart`, native 50 Hz control, seed 0.
- Teacher: HAIC PPO with a 605D normalized policy state.
- StarVLA input: current RGB plus the 605D state.
- StarVLA output: H1 256D latent.
- A frozen `actor_adapt` maps the current state and latent to the 23D motor action.

## Train

Train the native teacher:

```bash
conda activate haic
bash scripts/tasks/haic_pull_cart/train_small_policy.sh
```

Run the unified StarVLA smoke or formal schedule:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/tasks/haic_pull_cart/train_qwenoft.sh \
  --smoke --run-id haic-qwenoft-smoke

CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/tasks/haic_pull_cart/train_qwenoft.sh \
  --run-id haic-qwenoft-formal
```

The runner exports the frozen actor, collects the bootstrap data, trains SFT,
runs ten 500-update DAgger rounds, evaluates, and publishes formal artifacts.
Use `--teacher-checkpoint` for an exact local teacher, `--no-publish` for a
local run, or `--resume --run-id ID` to continue it. Replace `qwenoft` with
`qwenpi_v3`, `qwengr00t`, or `wanoft`.

## Profile

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/profile.sh run \
  --model qwenoft \
  --workload haic_pull_cart \
  --model-path MODEL_PATH \
  --hf-artifacts
```

The workload resolves the frozen actor independently from the StarVLA bundle.
See the [profiling cookbook](../profiling/cookbook.md) for the five-session
protocol and publication step.

## Acceptance

Record teacher completion, VLA training, evaluation, acceptance, and publication
separately. Motion progress, cart progress, `pullcart_score`, return, and success
all remain visible because success alone can hide partial progress. Verified
results are listed in [program/PROGRESS.md](../../program/PROGRESS.md).
