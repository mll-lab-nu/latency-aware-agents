# Sim2Real experiment scripts

This directory contains the entry points for the paper's Sim2Real experiment. A
standard reproduction has three stages: measure on the target hardware, run the
simulations, and build the paper data and figures. `cross_device.sh` is the
lower-level protocol CLI for running, diagnosing, or recovering an individual
stage.

All commands below can be run from any working directory. The examples assume
the current directory is the repository root.

## Scripts

| Script | Purpose |
|---|---|
| `measure_consecutive.sh <hardware>` | Measure and publish consecutive sessions 0--4, then fit and publish `fit-0-4` |
| `measure_isolated_30s.sh <hardware>` | Measure and publish isolated session 0, then fit and publish `fit-0` |
| `simulate.sh` | Run and publish the complete 120-task simulation matrix for the five paper devices |
| `build_figures.sh` | Run `build_data.py` and `plot.py` from the paper experiment directory in sequence |
| `cross_device.sh` | Activate the OpenVLA environment and invoke `latency_bench.cross_device.protocol` |

## Experiment setting

The shared setting uses 100 evaluation episodes, evaluation seed 0, the OpenVLA
Flappy workload, and four latency methods:

- `temporal`: use the profile's temporal burst model;
- `iid`: sample independently from the profile's empirical inverse CDF;
- `fixed`: always use the profile's `mean_ms`;
- `normal`: use the profile's `mean_ms` and population standard deviation
  `std_ms`, preserving the nonnegative clipping behavior of `NormalSampler`.

The two lifecycles use the following fixed settings:

| Lifecycle | Config / workload | Real sessions | Profile revision | Simulation latency seeds |
|---|---|---:|---|---:|
| Consecutive | `cross_device.yaml` / `openvla-flappy` | 0, 1, 2, 3, 4 | `fit-0-4` | 0, 1, 2, 3, 4 |
| Isolated | `cross_device_isolated_30s.yaml` / `openvla-flappy-isolated-30s` | 0 | `fit-0` | 0 |

Consecutive creates and warms up the runtime once per session, then runs every
episode continuously. Isolated shuts down the runtime and inference workers
before each episode, waits 30 seconds, then rebuilds and warms up the runtime.
It does not wait after the final episode.

The paper matrix uses the following five target devices and measured physical
instances. In this paper, A100 refers specifically to `a100-sxm4-80gb`.

| Paper label | Hardware ID | Consecutive instance | Isolated instance |
|---|---|---|---|
| RTX 3090 | `rtx3090` | `instance_000df10e` | `instance_000df10e` |
| RTX 4090 | `rtx4090` | `instance_9fb8d975` | `instance_e4777ea2` |
| RTX 5090 | `rtx5090` | `instance_02874a5a` | `instance_02874a5a` |
| L40S | `l40s` | `instance_cfa273e0` | `instance_edd21c9a` |
| A100 | `a100-sxm4-80gb` | `instance_1d6030f8` | `instance_1d6030f8` |

`simulate.sh` and the paper's `sources.json` use these exact artifact refs. A
new physical GPU receives a new instance ID based on its GPU UUID and does not
automatically replace these refs. When collecting a new complete paper dataset,
update both the simulation matrix and `sources.json` to point to the newly
generated refs.

## Dependencies and GPU layout

- Install the OpenVLA environment first with
  `bash install.sh --model openvla`. `cross_device.sh` activates
  `/venv/starvla_rl_games_openvla` if it exists; otherwise, it activates the
  Conda environment `starvla_rl_games_openvla`.
- Publishing artifacts requires write access to the Hugging Face dataset
  `latency-sensitive-bench/profiles` configured by the experiment. Downloading
  profiles, aggregating results, and building the paper data require read access
  to that dataset.
- Measurement must run on the target GPU host. The realtime executor uses every
  visible CUDA device as an inference worker, so expose only GPUs belonging to
  that target before measuring.
- `simulate.sh` requires CUDA devices `0..7` and uses four two-GPU slots in
  parallel: `0,1`, `2,3`, `4,5`, and `6,7`. Before starting, the script
  downloads and links the OpenVLA evaluation weights.
- `build_figures.sh` does not activate a Conda environment. The current Python
  environment must already include the repository and plotting dependencies.

The measurement scripts accept the following 11 explicit hardware IDs defined
by the configuration:

```text
a100-sxm4-80gb
a100-sxm4-40gb
a100-pcie-80gb
a100-pcie-40gb
l40s
rtx3090
rtx4090
rtx5090
rtx-pro-6000-blackwell-ws
rtx-pro-6000-blackwell-server
rtx-pro-6000-blackwell-maxq-ws
```

Do not use a generic `A100` without package type and capacity, or a generic
`RTX PRO 6000` in place of one of the IDs above.

## Complete three-stage reproduction

### 1. Measure and publish profiles on each target device

For each of the five paper hardware IDs, run the following commands on the
corresponding host:

```bash
bash scripts/experiments/sim2real/measure_consecutive.sh <hardware>
bash scripts/experiments/sim2real/measure_isolated_30s.sh <hardware>
```

Both scripts run `measure -> publish raw -> fit -> publish profile` and use
`--resume`.

### 2. Run the complete matrix on an eight-GPU simulation host

```bash
bash scripts/experiments/sim2real/simulate.sh
```

The matrix contains five devices times four methods, further multiplied by five
consecutive seeds and one isolated seed, for 120 tasks. Each task is published
immediately after its simulation finishes. A failed task does not stop the other
slots. After all tasks finish, the script lists the failures and returns a
nonzero status.

### 3. Build the paper data and figures

```bash
bash scripts/experiments/sim2real/build_figures.sh
```

The script uses `paper/latency-sensitive-vla/exps/sim2real/sources.json`. It
first generates `data.csv` and `provenance.json`, then produces the four paper
figures. It does not run the generic `aggregate` command described below.

## Lower-level `cross_device.sh` protocol

For a standard complete reproduction, prefer the three stages above. Use the
following commands to run, inspect, or recover an individual artifact leaf.
Set the common prefix as follows:

```bash
PROTOCOL=scripts/experiments/sim2real/cross_device.sh
```

### `target-ref`

Resolve the canonical ref for the current physical target. Its format is
`<device>/<instance>/<workload>`. Later `fit` and `download raw` commands need
this value.

```bash
TARGET_REF=$(bash "$PROTOCOL" target-ref \
  --config configs/experiments/cross_device.yaml \
  --hardware rtx4090)
```

By default, the instance ID is derived from the GPU UUID. Set
`CROSS_DEVICE_INSTANCE_ID` to specify an instance explicitly.

### `measure`

Create calibration data on the target host and run the explicitly selected real
sessions. This command writes only to local `results/cross_device/raw/` and does
not publish automatically.

```bash
bash "$PROTOCOL" measure \
  --config configs/experiments/cross_device.yaml \
  --hardware rtx4090 \
  --sessions 0 1 2 3 4 \
  --resume
```

### `fit`

Fit an explicitly named profile revision from locally completed raw sessions.

```bash
bash "$PROTOCOL" fit \
  --config configs/experiments/cross_device.yaml \
  --target-ref "$TARGET_REF" \
  --profile-revision fit-0-4 \
  --sessions 0 1 2 3 4 \
  --resume
```

If the raw sessions exist only remotely, run `download raw` first. When adding
fit sessions, create a new revision, such as changing `fit-0` to `fit-0-4`,
instead of changing the meaning of an existing revision.

### `sim`

Run the selected method and latency seeds with a profile ref. By default, the
profile is downloaded from Hugging Face. Results are written locally to
`results/cross_device/sim/` and are not published automatically.

```bash
bash "$PROTOCOL" sim \
  --config configs/experiments/cross_device.yaml \
  --profile-ref "$TARGET_REF/fit-0-4" \
  --method temporal \
  --latency-seeds 0 1 2 3 4 \
  --resume
```

### `download raw`

Download calibration data and raw sessions for the selected target. This allows
another host to fit a profile or inspect the data.

```bash
bash "$PROTOCOL" download raw \
  --config configs/experiments/cross_device.yaml \
  --target-ref "$TARGET_REF" \
  --sessions 0 1 2 3 4
```

Add `--overwrite` when an existing local target must be replaced explicitly.

### `publish raw`

Publish the current physical target's calibration data and explicit sessions.
This command resolves the target ref from the current GPU identity, so it
normally runs on the same host that performed the measurement.

```bash
bash "$PROTOCOL" publish raw \
  --config configs/experiments/cross_device.yaml \
  --hardware rtx4090 \
  --sessions 0 1 2 3 4 \
  --resume
```

### `publish profile`

Publish a locally completed profile leaf.

```bash
bash "$PROTOCOL" publish profile \
  --config configs/experiments/cross_device.yaml \
  --profile-ref "$TARGET_REF/fit-0-4" \
  --resume
```

### `publish sim`

Publish local simulation leaves for the selected profile, method, and latency
seeds.

```bash
bash "$PROTOCOL" publish sim \
  --config configs/experiments/cross_device.yaml \
  --profile-ref "$TARGET_REF/fit-0-4" \
  --method temporal \
  --latency-seeds 0 1 2 3 4 \
  --resume
```

### `aggregate`

`aggregate` is a generic cross-device analysis command. It does not measure,
fit profiles, or run simulations. It downloads the explicitly selected profile
refs, real sessions, methods, and latency seeds from Hugging Face, together
with the fit sessions recorded by each profile, then produces analysis CSVs:

```bash
bash "$PROTOCOL" aggregate \
  --config configs/experiments/cross_device.yaml \
  --profile-ref "$TARGET_REF/fit-0-4" \
  --real-sessions 0 1 2 3 4 \
  --methods temporal iid fixed normal \
  --latency-seeds 0 1 2 3 4
```

Repeat `--profile-ref` to include multiple profiles. The default output
directory is `results/cross_device/analysis/`:

| Output | Contents |
|---|---|
| `latency_samples.csv` | The profile's e2e, inference, and overhead latency samples |
| `latency_profile_summary.csv` | Latency summary statistics for each profile |
| `real_vs_sim.csv` | Return and length comparisons between real sessions and each method/seed simulation |

Use `--output-dir <path>` to change the output directory. The paper's
`build_figures.sh` uses `sources.json`, `build_data.py`, and `plot.py` directly;
it does not depend on these aggregate CSVs. You therefore do not need to run
`aggregate` before reproducing the paper figures.

## `--resume`, `--overwrite`, and `--local-profile`

- `--resume`: used by `measure`, `fit`, `sim`, and publish commands. A requested
  leaf with an existing `DONE` marker is skipped; an incomplete local leaf is
  cleaned and rerun. The scripts use this mode by default so interrupted work
  can continue.
- `--overwrite`: used by `download raw` to replace the local selection
  explicitly, and by `publish profile` and `publish sim` to replace the remote
  selection explicitly. For publish commands, `--overwrite` and `--resume` are
  mutually exclusive and affect only the leaf selected by the command.
- `--local-profile`: used only by `sim`. It reads a locally completed profile
  instead of downloading one from Hugging Face. This is useful for validating a
  simulation before its profile has been published. Simulations using an
  ordinarily published profile do not need this option.

Publish uploads the payload first and the `DONE` marker last. By default,
publishing is append-only: if a remote leaf is complete, publishing without
either `--resume` or `--overwrite` fails.
