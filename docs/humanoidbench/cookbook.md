# HumanoidBench Balance Simple (G1)

Install the shared and StarVLA environments using [QUICK_START.md](../QUICK_START.md).
The experiment history and fixed evaluation results remain in
[program/BALANCE.md](../../program/BALANCE.md) and
[program/PROGRESS.md](../../program/PROGRESS.md).

## Contract

- Environment: `g1_sonic_balance_simple`, 50 Hz, 198D state.

The SONIC name identifies the pinned G1 scene assets and their joint-order
contract. The teacher and VLA act directly on joint targets; they do not load
a SONIC policy. Existing environment IDs, asset paths, and action-layout names
are retained so trained models keep the same inputs and control semantics.
- Teacher: FastTD3, 43D absolute G1 joint target.
- StarVLA: 256x256 head-camera RGB plus state, H8 actions.
- Models: `qwenoft`, `qwenpi_v3`, `qwengr00t`, and `wanoft` through one runner.

## Train

Train or reuse the accepted FastTD3 teacher:

```bash
conda activate latency
bash scripts/tasks/humanoidbench_balance_simple/train_small_policy.sh
```

Run a two-update local smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/tasks/humanoidbench_balance_simple/train_qwenoft.sh \
  --smoke --run-id balance-qwenoft-smoke
```

Run the formal SFT and DAgger schedule:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/tasks/humanoidbench_balance_simple/train_qwenoft.sh \
  --run-id balance-qwenoft-formal
```

The runner downloads the pinned teacher unless `--teacher-checkpoint` is given,
collects and converts the dataset, trains, evaluates, and publishes formal
artifacts. Use `--no-publish` for a local formal run and `--resume --run-id ID`
to continue the same run. Replace `qwenoft` in the launcher name with any of the
other three model names.

## Profile

Use the exact inference bundle printed as `MODEL_PATH` by training:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/profile.sh run \
  --model qwenoft \
  --workload g1_sonic_balance_simple \
  --model-path MODEL_PATH \
  --hf-artifacts
```

The workload definition is in `configs/profiles.yaml`; generated profiles use
immutable run and artifact revisions. See the [profiling cookbook](../profiling/cookbook.md)
for resume and publication commands.

## Acceptance

Keep teacher training, VLA training, evaluation, acceptance, and publication as
separate recorded outcomes. A validation loss alone does not establish the
closed-loop Balance gate. Use the fixed seed protocol and return/episode-length
metrics in [program/PROGRESS.md](../../program/PROGRESS.md).
