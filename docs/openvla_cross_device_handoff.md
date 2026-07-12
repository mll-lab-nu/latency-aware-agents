# OpenVLA Cross-Device Handoff

The active OpenVLA cross-device workflow is the single-study sim2real pipeline documented in [`experiments/cross_device_protocol_runbook.md`](experiments/cross_device_protocol_runbook.md).

## Runtime

- Conda environment: `starvla_rl_games_openvla`
- Policy/environment: OpenVLA on Flappy
- Base evaluation config: `configs/experiments/flappy/realtime/starvla_openvla_fix_latency_2_16fps_l2_2gpu_warmup200_3ep.yaml`
- Study configs:
  - `configs/experiments/cross_device_1real_3sim.yaml`
  - `configs/experiments/cross_device_10real_20sim.yaml`
- Remote dataset: configured by `hf_repo` and `hf_repo_type` in `cross_device_study.yaml`

Each measurement host must have the OpenVLA checkpoint and environment assets referenced by the base config. Simulation hosts need the same policy runtime, but they do not need to match the target GPU model.

## Commands

Quick real1 temporal workflow:

```bash
bash start.sh
```

Formal measurement on each target GPU:

```bash
bash measure_10.sh
bash upload_10.sh
```

Formal simulation on any available host:

```bash
bash sim_20.sh
SPEC=configs/experiments/cross_device_10real_20sim.yaml \
  METHOD=temporal bash publish.sh sim

bash sim_iid_20.sh
SPEC=configs/experiments/cross_device_10real_20sim.yaml \
  METHOD=iid bash publish.sh sim
```

Aggregate the published study:

```bash
bash aggregate.sh
```

## Operational Contracts

- `measure` and `sim` are local compute operations; `publish` is explicit.
- `--resume` skips only leaf units carrying `DONE`.
- Partial leaf directories are discarded before retry.
- Publish uploads data before the group `DONE` marker.
- A changed experiment uses a new `study_id`; there is no overwrite mode.
- `CROSS_DEVICE_INSTANCE_ID` overrides the GPU-UUID-derived instance ID when an explicit stable identity is required.
- Simulation host identity is deliberately absent from the study schema and result keys.

## Removed Legacy Interfaces

The repository no longer supports protocol rows, exp1/exp2/exp3 stage selection, replay pairs, `sim_same`, `sim_cross`, target filters, doctor/smoke orchestration, run manifests, profile-build hashes, raw-root aggregation, or automatic publish from compute commands.

Do not use the removed `scripts/bash_scripts/run_cross_device.sh`, `scripts/bash_scripts/upload_cross_device_to_hf.sh`, `cross_device_protocol.yaml`, or `--target`/`--all-target-profiles` flags.
