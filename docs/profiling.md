# Deployment latency profiling

The profiling pipeline measures one model/workload on the exact set of GPUs
participating in inference and produces one replayable latency profile.

## Quick start

Install the OpenVLA environment, expose only the GPUs that should serve
inference, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/profile.sh run \
  --model openvla \
  --workload flappy \
  --publish
```

Without `--publish`, every artifact remains local. An interrupted run can be
continued with the run ID created below `results/profiling/runs/`:

```bash
bash scripts/profile.sh run \
  --model openvla \
  --workload flappy \
  --resume <run-id>
```

The shipped preset runs five independent 100-episode sessions. Set its
`checkpoint_path` once for the new machine, then edit `configs/profiles.yaml`
only when changing the preset or adding another model/workload pair.
The launcher selects the installed `starvla_rl_games_<model>` environment from
`--model`; `list` without a model uses the shared `latency` environment.

## Profile identity

A profile is addressed by:

```text
<model>/<gpu-deployment>/<workload>/<physical-instance>
```

For example:

```text
openvla/2x-rtx4090/flappy/instance_a3f81c04d972e6b1
```

The launcher canonicalizes the selected GPU UUIDs into stable worker slots.
The instance ID covers the complete GPU set and the normalized GPU-to-GPU link
topology. Changing a card or a topology link creates another instance. Runtime
health such as clocks, temperature, ECC state, or one temporarily slow card is
measurement data and does not change the instance ID.

The fitted profile keeps separate per-slot distributions, so a persistently
slow card is visible in `profile.json`. Temporal replay also samples observed
slot-to-severity assignments once per burst dwell; the slow rank does not jump
between cards inside that dwell.

Worker-service fitting retains dropped completions, which is essential when a
slow card is overtaken by another GPU. End-to-end action-ready fitting uses only
actions that actually reached the environment scheduler; `profile.json` reports
both `action_ready_count` and `dropped_count` per slot instead of inventing the
missing downstream stages of a dropped action.

## Artifacts

Local artifacts live below `results/profiling/`:

```text
runs/<model>/<gpu>/<workload>/<instance>/<run-id>/
profiles/<model>/<gpu>/<workload>/<instance>/
simulations/<model>/<gpu>/<workload>/<instance>/<run-id>/<method>/<seed>/
```

Each run owns its calibration and raw sessions. A completed run is immutable.
The profile directory is the current profile for that deployment and contains
only:

```text
profile.json
latency_distribution.json
latency_burst_model.json
hardware.json
DONE
```

Remeasuring creates another run and switches current only after the new bundle
is complete. Runs are never merged automatically.

## Publish and resolve

Publish the current completed run later with:

```bash
bash scripts/profile.sh publish --model openvla --workload flappy
```

Publishing writes the run and current profile in one Hugging Face commit and
prints its commit SHA. List current published profiles with:

```bash
bash scripts/profile.sh list --model openvla --workload flappy
```

On the same deployment, the GPU and instance coordinates are detected:

```bash
bash scripts/profile.sh resolve --model openvla --workload flappy
```

On another host, replay requires the exact coordinates rather than a same-SKU
fallback:

```bash
bash scripts/profile.sh resolve \
  --model openvla \
  --workload flappy \
  --gpu 2x-rtx4090 \
  --instance instance_a3f81c04d972e6b1 \
  --hf-commit <commit-sha>
```

Use the commit printed by publish for experiments and papers. Omitting
`--hf-commit` intentionally selects the current `main` profile.

## Replay

Replay one published profile with an explicit method and seeds:

```bash
bash scripts/profile.sh simulate \
  --model openvla \
  --workload flappy \
  --gpu 2x-rtx4090 \
  --instance instance_a3f81c04d972e6b1 \
  --method temporal \
  --seeds 0 1 2 \
  --hf-commit <commit-sha>
```

Supported methods are `temporal`, `iid`, `fixed`, and `normal`. Simulation
outputs include the source run ID in their path, so remeasuring the same
deployment cannot overwrite an earlier replay.

## Adding a preset

Add the exact GPU name under `gpu_models` when it is not already registered.
Add the deployment variant under `models`, then its workload, base eval config,
and checkpoint path. Model IDs should distinguish runtime/precision variants
whose latency differs. Workload IDs should distinguish environment/task
behavior; checkpoint paths remain provenance rather than profile axes.

The first release supports homogeneous CUDA GPU deployments and consecutive
measurement. Mixed GPU types, MIG, model-parallel launch strategies, and a
cross-burst hardware-health state are outside this contract.
