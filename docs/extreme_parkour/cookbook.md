# Extreme Parkour (Go2)

Install Isaac Sim, Isaac Lab, the Parkour task, and StarVLA using
[QUICK_START.md](../QUICK_START.md). Historical experiments remain in
[program/EXTREME_PARKOUR.md](../../program/EXTREME_PARKOUR.md).

## Contract

- Environment: `Isaac-Extreme-Parkour-VLA-Unitree-Go2-v0`.
- Teacher: privileged PPO, 6,144 environments, 15,000 iterations, seed 1.
- StarVLA input: 10 Hz RGB plus 53D state.
- StarVLA output: H1 32D terrain latent plus 2D yaw command.
- A frozen teacher actor maps that command and the current actor observation to
  the 12D motor action used by the simulator.

## Train

Train the native teacher:

```bash
conda activate extreme-parkour
bash scripts/tasks/extreme_parkour_go2/train_small_policy.sh
```

Run the unified StarVLA smoke or formal schedule:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/tasks/extreme_parkour_go2/train_qwenoft.sh \
  --smoke --run-id parkour-qwenoft-smoke

CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/tasks/extreme_parkour_go2/train_qwenoft.sh \
  --run-id parkour-qwenoft-formal
```

Formal training performs bootstrap SFT, ten 500-update DAgger rounds, fixed-seed
evaluation, and publication. Use `--teacher-checkpoint` for an exact local
teacher, `--no-publish` for a local run, or `--resume --run-id ID` to continue
the same output. Replace `qwenoft` with `qwenpi_v3`, `qwengr00t`, or `wanoft`.

## Profile

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/profile.sh run \
  --model qwenoft \
  --workload extreme_parkour_go2 \
  --model-path MODEL_PATH \
  --hf-artifacts
```

The simulator and inference worker use separate logical GPUs. The workload and
frozen actor artifact are pinned in `configs/profiles.yaml`. See the
[profiling cookbook](../profiling/cookbook.md) for the five-session protocol.

## Acceptance

The privileged teacher gate, DAgger round completion, final zero-delay result,
profile completion, and publication are separate gates. Use normalized waypoint
progress and episode return from the fixed seed-2 evaluation; the detailed
results are in [program/PROGRESS.md](../../program/PROGRESS.md).
