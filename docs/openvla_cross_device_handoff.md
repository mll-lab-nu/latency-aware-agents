# OpenVLA Cross-Device Handoff

The active workflow stores reusable target artifacts instead of fixed-size studies. See
[`experiments/cross_device_protocol_runbook.md`](experiments/cross_device_protocol_runbook.md)
for the protocol and
[`scripts/experiments/sim2real/README.md`](../scripts/experiments/sim2real/README.md)
for the maintained experiment entry points and complete command reference.

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

The paper workflow has three stages. Run both measurement scripts on each target
hardware host, then run the simulation matrix on the eight-GPU simulation host,
then build the paper data and figures:

```bash
bash scripts/experiments/sim2real/measure_consecutive.sh <hardware>
bash scripts/experiments/sim2real/measure_isolated_30s.sh <hardware>
bash scripts/experiments/sim2real/simulate.sh
bash scripts/experiments/sim2real/build_figures.sh
```

The measurement scripts accept the 11 explicit hardware IDs listed in the
experiment README. The paper matrix uses RTX 3090, RTX 4090, RTX 5090, L40S,
and A100 SXM4 80GB. It contains 120 tasks and assigns CUDA devices `0..7` to
four two-GPU slots.

Use the low-level protocol only for independent operations or recovery:

```bash
PROTOCOL=scripts/experiments/sim2real/cross_device.sh
CONFIG=configs/experiments/cross_device.yaml

bash "$PROTOCOL" measure --config "$CONFIG" \
  --hardware rtx4090 --sessions 0 1 --resume
bash "$PROTOCOL" publish raw --config "$CONFIG" \
  --hardware rtx4090 --sessions 0 1 --resume

TARGET_REF=$(bash "$PROTOCOL" target-ref \
  --config "$CONFIG" --hardware rtx4090)
bash "$PROTOCOL" fit --config "$CONFIG" --target-ref "$TARGET_REF" \
  --profile-revision fit-0-1 --sessions 0 1 --resume
bash "$PROTOCOL" publish profile --config "$CONFIG" \
  --profile-ref "$TARGET_REF/fit-0-1" --resume

bash "$PROTOCOL" sim --config "$CONFIG" \
  --profile-ref "$TARGET_REF/fit-0-1" \
  --method temporal --latency-seeds 0 1 2 --resume
bash "$PROTOCOL" publish sim --config "$CONFIG" \
  --profile-ref "$TARGET_REF/fit-0-1" \
  --method temporal --latency-seeds 0 1 2 --resume
```

Simulation methods are `temporal`, `iid`, `fixed`, and `normal`. `fixed` uses the profile mean; `normal` passes the profile mean and population standard deviation to `NormalSampler`, which retains its existing zero-clamp behavior.

The protocol also exposes `download raw` and `aggregate`. Aggregation downloads
the explicitly selected profile refs, real sessions, methods, and latency seeds
and writes `latency_samples.csv`, `latency_profile_summary.csv`, and
`real_vs_sim.csv`. It never infers a study size from the artifact tree and is
not required by the paper's `build_figures.sh` workflow. See the experiment
README for every protocol subcommand and the `--resume`, `--overwrite`, and
`--local-profile` semantics.

## Removed Interfaces

There are no real1/real10 study configs, numbered shell wrappers, protocol rows, replay pairs, run manifests, profile-build hashes, automatic seed allocation, mutable latest profile, or group-level completion markers.
