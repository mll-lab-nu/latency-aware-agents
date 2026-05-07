# OpenVLA Cross-Device Handoff

This note is for running OpenVLA cross-device evals on a machine that has the
OpenVLA base model and SFT checkpoints available.

## Supported Rows

- `openvla-flappy` on `flappy`
- `demon-attack-openvla` on `demon_attack`
- `deadly-corridor-openvla` on `deadly_corridor`

All rows use the same entrypoint:

```bash
bash scripts/bash_scripts/run_cross_device_openvla.sh
```

## Required Paths

Set these paths before dependency checks, smoke runs, or full runs:

```bash
export OPENVLA_BASE_MODEL_PATH=/path/to/openvla_base
export DEMON_ATTACK_OPENVLA_CHECKPOINT_PATH=/path/to/demon_ckpt
export DEADLY_CORRIDOR_OPENVLA_CHECKPOINT_PATH=/path/to/deadly_ckpt
```

For Flappy only, the config default checkpoint is still used unless this
optional override is set:

```bash
export FLAPPY_OPENVLA_CHECKPOINT_PATH=/path/to/flappy_ckpt
```

The base model directory must contain policy-specific statistics files:

- `flappy_dataset_statistics.json` with key `flappy_bird`
- `demon_attack_dataset_statistics.json` with key `demon_attack`
- `deadly_corridor_dataset_statistics.json` with key `deadly_corridor`

Deadly Corridor OpenVLA uses RL-inf's current 11-dim factorized action layout,
`deadly_corridor_factorized_11`. The `deadly_corridor.action` statistics
fields `q01`, `q99`, `min`, `max`, and `mask` must all have length 11.

Each checkpoint directory must contain:

```text
actor/model_state_dict/full_weights.pt
```

## Dependency Check

This checks paths, unnormalization keys, and OpenVLA action-stat dimensions. It
does not load the model, initialize an environment, or write eval output.

```bash
CROSS_DEVICE_HARDWARE=a100 \
OPENVLA_BASE_MODEL_PATH=/path/to/openvla_base \
DEMON_ATTACK_OPENVLA_CHECKPOINT_PATH=/path/to/demon_ckpt \
DEADLY_CORRIDOR_OPENVLA_CHECKPOINT_PATH=/path/to/deadly_ckpt \
OPENVLA_CHECK_DEPS_ONLY=1 \
bash scripts/bash_scripts/run_cross_device_openvla.sh
```

Filter checks the same way as eval:

```bash
OPENVLA_CHECK_DEPS_ONLY=1 ONLY_POLICY=demon-attack-openvla \
bash scripts/bash_scripts/run_cross_device_openvla.sh
```

## Dry Run

Dry-run prints selected rows, generated config paths, output directories, and
resolved OpenVLA paths.

```bash
DRY_RUN=1 CROSS_DEVICE_HARDWARE=a100 \
bash scripts/bash_scripts/run_cross_device_openvla.sh
```

Smoke dry-run:

```bash
DRY_RUN=1 OPENVLA_SMOKE=1 MAX_RUNS=1 ONLY_POLICY=demon-attack-openvla \
CROSS_DEVICE_HARDWARE=a100 \
bash scripts/bash_scripts/run_cross_device_openvla.sh
```

## One-Episode Smoke

Smoke runs are isolated from full runs. They use run IDs like
`smoke_seed_0_n1` and measure-stage smoke profiles under `profiles_smoke/`.
This prevents smoke manifests or profiles from causing a later full run to be
skipped by `--resume`.

Demon Attack smoke:

```bash
CROSS_DEVICE_HARDWARE=a100 \
OPENVLA_BASE_MODEL_PATH=/path/to/openvla_base \
DEMON_ATTACK_OPENVLA_CHECKPOINT_PATH=/path/to/demon_ckpt \
OPENVLA_SMOKE=1 \
OPENVLA_STAGES=measure,realtime \
ONLY_POLICY=demon-attack-openvla \
MAX_RUNS=1 \
bash scripts/bash_scripts/run_cross_device_openvla.sh
```

Deadly Corridor smoke:

```bash
CROSS_DEVICE_HARDWARE=a100 \
OPENVLA_BASE_MODEL_PATH=/path/to/openvla_base \
DEADLY_CORRIDOR_OPENVLA_CHECKPOINT_PATH=/path/to/deadly_ckpt \
OPENVLA_SMOKE=1 \
OPENVLA_STAGES=measure,realtime \
ONLY_POLICY=deadly-corridor-openvla \
MAX_RUNS=1 \
bash scripts/bash_scripts/run_cross_device_openvla.sh
```

Optional smoke step cap:

```bash
OPENVLA_SMOKE=1 OPENVLA_SMOKE_MAX_STEPS=200 \
bash scripts/bash_scripts/run_cross_device_openvla.sh
```

`OPENVLA_SMOKE_MAX_STEPS` overrides `evaluation.eval_max_steps` in generated
configs. It does not change environment stepping semantics such as Atari
wrappers, Doom frameskip, or frame stack.

## Stage Selection

By default, non-smoke runs execute:

```text
doctor, measure, realtime, sim-same, sim-cross
```

By default, smoke runs execute:

```text
doctor, measure, realtime
```

Set explicit stages when needed:

```bash
OPENVLA_STAGES=doctor bash scripts/bash_scripts/run_cross_device_openvla.sh
OPENVLA_STAGES=measure,realtime,sim-same bash scripts/bash_scripts/run_cross_device_openvla.sh
```

Smoke does not run `sim-same` or `sim-cross` unless explicitly requested because
both stages depend on existing latency profiles.

## Full Sweep

After dependency checks and per-environment smoke pass, run the full sweep:

```bash
CROSS_DEVICE_HARDWARE=a100 \
OPENVLA_BASE_MODEL_PATH=/path/to/openvla_base \
DEMON_ATTACK_OPENVLA_CHECKPOINT_PATH=/path/to/demon_ckpt \
DEADLY_CORRIDOR_OPENVLA_CHECKPOINT_PATH=/path/to/deadly_ckpt \
CONTINUE_ON_ERROR=1 \
bash scripts/bash_scripts/run_cross_device_openvla.sh
```

Full results continue to use the established tree:

```text
results/cross_device/runs/<stage>/openvla-flappy/flappy/<device>/best_seed_0_n10/
results/cross_device/runs/<stage>/demon-attack-openvla/demon_attack/<device>/best_seed_0_n10/
results/cross_device/runs/<stage>/deadly-corridor-openvla/deadly_corridor/<device>/best_seed_0_n10/
```

Smoke results use isolated run IDs:

```text
results/cross_device/runs/<stage>/<policy>/<env>/<device>/smoke_seed_0_n1/
```

## Common Failures

- Missing base model: set `OPENVLA_BASE_MODEL_PATH` to an existing directory.
- Missing policy statistics file: ensure the selected policy has its
  `*_dataset_statistics.json` file directly under the base model directory.
- Missing unnorm key: add the selected env key to the matching statistics file.
- Deadly action stats length mismatch: `deadly_corridor.action.q01`,
  `q99`, `min`, `max`, and `mask` must be length 11 for
  `deadly_corridor_factorized_11`.
- Missing `full_weights.pt`: point the checkpoint env var at the directory that
  contains `actor/model_state_dict/full_weights.pt`.
- Demon action meanings mismatch: the latency_bench Demon env will fail fast if
  the Atari action order differs from the configured expected action meanings.
- Deadly VizDoom init failure: verify VizDoom/Sample Factory dependencies,
  scenario files, and display/headless runtime support on the checkpoint machine.
