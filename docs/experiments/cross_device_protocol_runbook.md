# Cross-Device Latency Protocol Runbook

This runbook assumes the repository is available at:

```bash
cd /home/lixinyuan/latency-sensitive-bench
```

The protocol spec is:

```text
configs/experiments/cross_device_protocol.yaml
```

The default output root is:

```text
results/cross_device
```

## Recommended usage

Small models:

```bash
bash scripts/bash_scripts/run_cross_device_small_models.sh
```

OpenVLA:

```bash
bash scripts/bash_scripts/run_cross_device_openvla.sh
```

Aggregate:

```bash
bash scripts/bash_scripts/aggregate_cross_device.sh
```

Upload one machine's collected outputs to Hugging Face:

```bash
HF_REPO=your-name/cross-device-latency-v1 HF_TOKEN=hf_xxx \
bash scripts/bash_scripts/upload_cross_device_to_hf.sh
```

Dry run:

```bash
DRY_RUN=1 bash scripts/bash_scripts/run_cross_device_small_models.sh
DRY_RUN=1 bash scripts/bash_scripts/run_cross_device_openvla.sh
DRY_RUN=1 HF_REPO=your-name/cross-device-latency-v1 bash scripts/bash_scripts/upload_cross_device_to_hf.sh
```

More episodes:

```bash
N_EPISODES=100 SEED_SET=0,1,2 bash scripts/bash_scripts/run_cross_device_small_models.sh
```

Official clean run root:

```bash
OUT=results/cross_device_official \
PROFILES_DIR=results/cross_device_official/profiles \
bash scripts/bash_scripts/run_cross_device_small_models.sh
```

The default protocol keeps most evaluation rows at 10 episodes, but uses a
`stage_overrides.measure.n_episodes` row override for `deadly-corridor-sf-l1`
so its latency profile is measured with 1000 episodes. Passing `N_EPISODES`
explicitly still overrides all row/stage defaults.

Launcher logs are written to:

```text
logs/cross_device_protocol/
```

Each launcher accepts environment-variable overrides:

```bash
REPO_DIR=/path/to/repo
SPEC=configs/experiments/cross_device_protocol.yaml
OUT=results/cross_device
PROFILES_DIR=results/cross_device/profiles
N_EPISODES=100
SEED_SET=0,1,2
CHECKPOINT_KIND=best
DRY_RUN=1
```

The upload launcher also accepts:

```bash
HF_REPO=your-name/cross-device-latency-v1
HF_TOKEN=hf_xxx        # optional if already logged in with hf auth login
HF_TOKEN_FILE=/path/to/token.txt
HF_REPO_TYPE=dataset
OUT=results/cross_device
DEVICE=auto          # or rtx3090 / rtx4090 / l40s / a100 / h200 / b200
RUNTIME=combined     # use this when small_model and OpenVLA both ran on the machine
REMOTE_PREFIX=raw
RUN_TAG=$(hostname)_$(date +%Y%m%d_%H%M%S)
DRY_RUN=1
```

The default remote layout is:

```text
raw/<device>/<runtime>/<hostname>_<timestamp>/cross_device/
```

## 1. Small models in `latency`

Use this environment for Sample Factory / small RL policies:

```bash
conda activate latency
```

Check the machine and selected rows:

```bash
python scripts/experiments/run_cross_device_protocol.py doctor \
  --spec configs/experiments/cross_device_protocol.yaml \
  --hardware auto \
  --runtime-env auto
```

Measure latency profiles on the current hardware:

```bash
python scripts/experiments/run_cross_device_protocol.py measure \
  --spec configs/experiments/cross_device_protocol.yaml \
  --hardware auto \
  --runtime-env auto \
  --out results/cross_device \
  --resume
```

Run true realtime evaluation on the current hardware:

```bash
python scripts/experiments/run_cross_device_protocol.py realtime \
  --spec configs/experiments/cross_device_protocol.yaml \
  --hardware auto \
  --runtime-env auto \
  --out results/cross_device \
  --resume
```

Run same-device simulated replay using the current hardware profile:

```bash
python scripts/experiments/run_cross_device_protocol.py sim-same \
  --spec configs/experiments/cross_device_protocol.yaml \
  --hardware auto \
  --runtime-env auto \
  --profiles-dir results/cross_device/profiles \
  --out results/cross_device \
  --resume
```

## 2. OpenVLA in `latency-openvla`

Use this environment for OpenVLA-Flappy:

```bash
conda activate latency-openvla
```

Check the machine and selected rows:

```bash
python scripts/experiments/run_cross_device_protocol.py doctor \
  --spec configs/experiments/cross_device_protocol.yaml \
  --hardware auto \
  --runtime-env auto
```

Measure OpenVLA latency profiles on the current hardware:

```bash
python scripts/experiments/run_cross_device_protocol.py measure \
  --spec configs/experiments/cross_device_protocol.yaml \
  --hardware auto \
  --runtime-env auto \
  --out results/cross_device \
  --resume
```

Run true realtime OpenVLA evaluation on the current hardware:

```bash
python scripts/experiments/run_cross_device_protocol.py realtime \
  --spec configs/experiments/cross_device_protocol.yaml \
  --hardware auto \
  --runtime-env auto \
  --out results/cross_device \
  --resume
```

Run cross-device replay on the current replay hardware:

```bash
python scripts/experiments/run_cross_device_protocol.py sim-cross \
  --spec configs/experiments/cross_device_protocol.yaml \
  --replay-hardware auto \
  --runtime-env auto \
  --profiles-dir results/cross_device/profiles \
  --out results/cross_device \
  --resume
```

## 3. Aggregation

After collecting or syncing all machine outputs into one `results/cross_device` tree, aggregate in the lightweight `latency` environment:

```bash
conda activate latency
```

```bash
python scripts/experiments/run_cross_device_protocol.py aggregate \
  --spec configs/experiments/cross_device_protocol.yaml \
  --root results/cross_device \
  --out results/cross_device/tables \
  --overwrite
```

## 4. Result locations

Generated configs:

```text
results/cross_device/generated_configs/<stage>/<policy>/<env>/<device_or_pair>/<run_id>.yaml
```

Latency profiles:

```text
results/cross_device/profiles/<policy>/<env>/<profile_id>/
```

Important profile files:

```text
profile.json
latencies.jsonl
profile_summary.json
profile_summary.csv
```

Realtime runs:

```text
results/cross_device/runs/realtime/<policy>/<env>/<hardware>/<run_id>/
```

Same-device simulated replay:

```text
results/cross_device/runs/sim_same/<policy>/<env>/<target_hardware>/<run_id>/
```

Cross-device simulated replay:

```text
results/cross_device/runs/sim_cross/<policy>/<env>/<target_hardware>/replay_<replay_hardware>/<run_id>/
```

Run manifest index:

```text
results/cross_device/manifests/runs.jsonl
```

Final CSV tables:

```text
results/cross_device/tables/experiment1_latency_profile_summary.csv
results/cross_device/tables/experiment2_same_device_calibration.csv
results/cross_device/tables/experiment3_cross_device_prediction.csv
```

## Safety notes

`--runtime-env auto` is the default safety mode. It detects the current environment and only runs compatible rows:

- `latency` runs `small_model` rows.
- `latency-openvla` runs `openvla` rows.

Use `--dry-run` before expensive runs:

```bash
python scripts/experiments/run_cross_device_protocol.py measure \
  --spec configs/experiments/cross_device_protocol.yaml \
  --hardware auto \
  --runtime-env auto \
  --out results/cross_device \
  --dry-run
```

`--runtime-env all` is allowed for dry-run inspection. For real execution it requires `--allow-runtime-env-mismatch`, which should only be used deliberately.
