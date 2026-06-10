# Cross-Device Latency Protocol Runbook

This runbook describes the realtime-inference cross-device protocol: how to
measure latency profiles, reuse them across machines, run replay simulations,
and aggregate the outputs used by the paper figures.

## Main Entrypoint

Use one preset argument instead of exporting many environment variables:

```bash
scripts/bash_scripts/run_cross_device.sh <preset>
```

Presets:

| Preset | Stages | Main use |
| --- | --- | --- |
| `doctor` | `doctor` | Check selected rows, runtime, hardware, configs, and missing profiles. |
| `profile` | `doctor measure` | Measure realtime inference latency, build local profiles, then upload profiles to Hugging Face. |
| `realtime` | `doctor realtime` | Run the true realtime baseline on the current machine. |
| `sim` | `doctor sim-same sim-cross` | Run simulated replay; missing local profiles are downloaded from Hugging Face. |
| `aggregate` | `aggregate` | Write the CSV tables used for figure data snapshots. |
| `custom` | `$STAGES` | Advanced mode when you need an exact custom stage list. |

The script automatically reads `.env` from the repo root and defaults to:

```bash
HF_PROFILE_REPO=latency-sensitive-bench/profiles
HF_PROFILE_REPO_TYPE=dataset
```

Common examples:

```bash
# Print commands without running expensive inference.
DRY_RUN=1 scripts/bash_scripts/run_cross_device.sh profile

# Measure OpenVLA/small-model rows that match the current conda environment.
scripts/bash_scripts/run_cross_device.sh profile

# Run realtime baseline.
scripts/bash_scripts/run_cross_device.sh realtime

# Run same-device and cross-device replay.
scripts/bash_scripts/run_cross_device.sh sim

# Aggregate protocol outputs.
scripts/bash_scripts/run_cross_device.sh aggregate
```

Small overrides are still supported:

```bash
N_EPISODES=3 ONLY_POLICY=OpenVLA-Flappy scripts/bash_scripts/run_cross_device.sh profile
DRY_RUN=1 ONLY_POLICY=OpenVLA-Flappy scripts/bash_scripts/run_cross_device.sh sim
STAGES="doctor measure realtime" scripts/bash_scripts/run_cross_device.sh custom
```

Default paths:

```bash
SPEC=configs/experiments/cross_device_protocol.yaml
OUT=results/cross_device
PROFILES_DIR=results/cross_device/profiles
```

## Experiment Flow

The protocol is run as ordered stages:

1. `doctor`: checks selected rows, hardware mapping, runtime env, configs, and
   missing profiles without running inference.
2. `measure`: runs realtime inference on the current device and writes
   `latencies.jsonl`; then converts it into a profile folder containing
   `profile.json`, `latencies.jsonl`, `profile_summary.json`, and
   `profile_summary.csv`.
3. `realtime`: runs the true realtime baseline on the current device.
4. `sim-same`: replays the current device profile on the same device.
5. `sim-cross`: replays a target device profile on the explicit replay device
   named by the protocol pair.
6. `aggregate`: reads manifests and profiles and writes the CSV tables used to
   build paper figure data snapshots.

Latency profiles are model-and-hardware artifacts. The profile manifest records
the policy/env/device/fps plus model metadata such as `policy_type`,
`checkpoint_kind`, `checkpoint_path_or_id`, optional `checkpoint_fingerprint`,
and OpenVLA `base_model_id_or_path` when available.

Cross-device replay pairs are explicit in the protocol. The `target_device` is
the device whose latency profile is being simulated; the `replay_device` is the
machine that executes the replay:

```yaml
replay_pairs:
  - target_device: a100
    replay_device: b200
  - target_device: a100
    replay_device: h200
```

## Advanced Script Use

The same single bash entrypoint supports explicit overrides:

```bash
scripts/bash_scripts/run_cross_device.sh <preset>
```

Common examples:

```bash
# Print selected rows and generated commands without running inference.
DRY_RUN=1 scripts/bash_scripts/run_cross_device.sh profile

# Run the standard local flow.
scripts/bash_scripts/run_cross_device.sh profile
scripts/bash_scripts/run_cross_device.sh realtime
scripts/bash_scripts/run_cross_device.sh sim

# Run OpenVLA rows from the OpenVLA environment.
conda activate starvla_rl_games_openvla
RUNTIME_ENV=openvla scripts/bash_scripts/run_cross_device.sh profile

# Run small-model rows from the latency environment.
conda activate latency
RUNTIME_ENV=small_model scripts/bash_scripts/run_cross_device.sh profile
```

Useful environment variables:

```bash
SPEC=configs/experiments/cross_device_protocol.yaml
OUT=results/cross_device
PROFILES_DIR=results/cross_device/profiles
TABLES_DIR=results/cross_device/tables
HARDWARE=auto
REPLAY_HARDWARE=auto
RUNTIME_ENV=auto
STAGES="doctor measure realtime sim-same sim-cross aggregate"  # custom preset only
ONLY_POLICY=OpenVLA-Flappy
ONLY_ENV=flappy
ONLY_DEVICE=a100
ONLY_EXPERIMENT=exp2_same_device_calibration
MAX_RUNS=1
N_EPISODES=100
SEED_SET=0,1,2
CHECKPOINT_KIND=best
DRY_RUN=1
RESUME=1
OVERWRITE=0
CONTINUE_ON_ERROR=1
SMOKE=1
SMOKE_EPISODES=1
SMOKE_MAX_STEPS=10
```

## Python CLI

The bash script calls the Python module:

```bash
python -m scripts.cross_device.protocol doctor --spec "$SPEC" --hardware auto --runtime-env auto
python -m scripts.cross_device.protocol measure --spec "$SPEC" --hardware auto --out "$OUT" --resume
python -m scripts.cross_device.protocol realtime --spec "$SPEC" --hardware auto --out "$OUT" --resume
python -m scripts.cross_device.protocol sim-same --spec "$SPEC" --hardware auto --profiles-dir "$PROFILES_DIR" --out "$OUT" --resume
python -m scripts.cross_device.protocol sim-cross --spec "$SPEC" --replay-hardware auto --profiles-dir "$PROFILES_DIR" --out "$OUT" --resume
python -m scripts.cross_device.protocol aggregate --spec "$SPEC" --root "$OUT" --out "$TABLES_DIR" --overwrite
```

## Profile Cache

Profiles are stored locally under:

```text
<PROFILES_DIR>/<policy_slug>/<env_slug>/<profile_id>/profile.json
```

For cross-machine runs, set a Hugging Face dataset repo:

```bash
HF_PROFILE_REPO=latency-sensitive-bench/profiles
```

Sim stages use local-first behavior:

- if `profile.json` exists locally, it is used directly;
- if it is missing and `HF_PROFILE_REPO` is set, the matching profile folder is
  downloaded before skip decisions;
- `doctor` and `DRY_RUN=1` do not download profiles.

Upload measured local profiles. This happens automatically for the `profile`
preset; the explicit form is mainly useful with `custom`:

```bash
UPLOAD_PROFILES=1 \
HF_PROFILE_REPO=latency-sensitive-bench/profiles \
STAGES="measure" \
scripts/bash_scripts/run_cross_device.sh custom
```

Manual profile cache commands:

```bash
python -m scripts.cross_device.profile_cache download \
  --profile-path "$PROFILES_DIR/OpenVLA-Flappy/flappy/L_A100/profile.json" \
  --profiles-dir "$PROFILES_DIR" \
  --hf-repo latency-sensitive-bench/profiles

python -m scripts.cross_device.profile_cache upload \
  --profile-path "$PROFILES_DIR/OpenVLA-Flappy/flappy/L_A100/profile.json" \
  --profiles-dir "$PROFILES_DIR" \
  --hf-repo latency-sensitive-bench/profiles
```

## Aggregation And Paper Data

`aggregate` writes protocol tables under `TABLES_DIR`:

```text
experiment1_latency_profile_summary.csv
experiment2_same_device_calibration.csv
experiment3_cross_device_prediction.csv
```

Those CSVs are derived from measured profiles and run manifests. The plotted
paper snapshot under `paper/latency-sensitive-vla/figures/.../data.csv` is not
written directly by the protocol; it is a curated figure input built from these
aggregate tables.

To aggregate downloaded raw machine outputs:

```bash
python -m scripts.cross_device.protocol aggregate \
  --spec "$SPEC" \
  --root "$OUT" \
  --raw-root results/raw \
  --out "$TABLES_DIR" \
  --overwrite
```

The older full-output upload script is still available for archiving full run
directories:

```bash
HF_REPO=your-name/cross-device-latency-v1 \
bash scripts/bash_scripts/upload_cross_device_to_hf.sh
```
