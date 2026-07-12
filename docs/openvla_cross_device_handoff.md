# OpenVLA Cross-Device Handoff

The active workflow stores reusable target artifacts instead of fixed-size studies. See [`experiments/cross_device_protocol_runbook.md`](experiments/cross_device_protocol_runbook.md) for the complete commands.

## Runtime

- Config: `configs/experiments/cross_device.yaml`
- Workload: `openvla-flappy`
- Conda environment: `starvla_rl_games_openvla`
- Remote dataset: `latency-sensitive-bench/profiles`
- Legacy snapshot: HF tag `v3`

The workload ID is immutable. A behavior-changing policy, checkpoint, environment, or base-config change uses a new workload ID.

## Artifact Contract

```text
raw/<device>/<instance>/<workload>/sessions/<session>/
profiles/<device>/<instance>/<workload>/<revision>/
sim/<device>/<instance>/<workload>/<revision>/<method>/<latency-seed>/
```

- Physical target identity comes from the GPU UUID hash.
- Calibration is created once per target/workload and reused by appended real sessions.
- Profile revisions name exact fitted session sets and are immutable.
- Simulation seeds are explicit, independently appendable leaves.
- Simulation host identity is not part of artifact identity.
- Only calibration/session/profile/seed leaves carry completion markers.

## Commands

Quick workflow:

```bash
bash start.sh
```

Independent operations:

```bash
bash cross_device.sh measure --sessions 0 1 --resume
bash cross_device.sh publish raw --sessions 0 1 --resume

TARGET_REF=$(bash cross_device.sh target-ref)
bash cross_device.sh fit --target-ref "$TARGET_REF" \
  --profile-revision fit-0-1 --sessions 0 1 --resume
bash cross_device.sh publish profile \
  --profile-ref "$TARGET_REF/fit-0-1" --resume

bash cross_device.sh sim --profile-ref "$TARGET_REF/fit-0-1" \
  --method temporal --latency-seeds 0 1 2 --resume
bash cross_device.sh publish sim --profile-ref "$TARGET_REF/fit-0-1" \
  --method temporal --latency-seeds 0 1 2 --resume
```

Aggregation explicitly lists profile refs, real sessions, methods, and latency seeds. It never infers a study size from the artifact tree.

## Removed Interfaces

There are no real1/real10 study configs, numbered shell wrappers, protocol rows, replay pairs, run manifests, profile-build hashes, automatic seed allocation, mutable latest profile, or group-level completion markers.
