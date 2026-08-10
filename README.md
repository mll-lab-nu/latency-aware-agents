<h1 align="center">Latency-Sensitive Bench</h1>
<h3 align="center"><em>Profiling, evaluating, and training agents under real deployment latency.</em></h3>

Latency-Sensitive Bench studies what happens when real-time agents act through
delayed, variable, device-dependent action pipelines. The benchmark treats
latency as a first-class deployment condition: observations are emitted on an
observation timeline, actions become ready after a latency delay, and the
environment applies actions on the raw environment timeline.

This repository owns the benchmark runtime, latency execution semantics,
rollout/profile generation, and experiment configs. VLA training is launched
from the companion StarVLA repository, usually through
`examples/rl_games/bash_scripts/...`, while consuming datasets, profiles, and
environment dependencies from this repository.

## Overview

Latency-Sensitive Bench provides:

- latency-aware execution for Flappy Bird, Demon Attack, and Deadly Corridor;
- fixed-latency, jittered-latency, and device-profile evaluation semantics;
- teacher rollout generation and Hugging Face dataset export;
- cross-device latency profiling, profile fitting, and profile replay;
- memory, temporal-context, ghost-trail, and KV-memory experiment support;
- integration points for OpenVLA, pi0, pi0.5, GR00T, WanOFT, and human realtime
  evaluation.

## Repository Map

```text
latency-sensitive-bench/
  latency_bench/        # benchmark runtime, latency queues, samplers, profiling, eval
  configs/              # base env configs and experiment configs
  scripts/              # rollout, memory, cross-device, and experiment orchestration
  docs/                 # semantic contracts and detailed runbooks

../starVLA/
  examples/rl_games/bash_scripts/   # canonical VLA training/eval launchers
  examples/rl_games/config/         # StarVLA Hydra experiment configs
  starVLA/training/rl_games/        # eval_core and RL-games integration

../latency-sensitive-bench-realtime-human/
  static/pyodide.html               # browser-local human eval prototype
  server.py                         # FastAPI human realtime eval backend
```

## Installation

Clone this repository with submodules:

```bash
git clone --recurse-submodules https://github.com/ZihanWang314/latency-sensitive-bench.git
cd latency-sensitive-bench
```

The shared `latency` environment and the RL-games model environments use
Python 3.10, CUDA Toolkit 12.8.1, and PyTorch 2.11.0+cu128. The CUDA build
targets H200 (`sm_90`) and RTX PRO 6000 (`sm_120`); there is no automatic
Torch/CUDA profile selection.

Install one model environment:

```bash
bash install.sh --model openvla
```

Install all model environments:

```bash
bash install.sh --model all
```

Accepted model names are `openvla`, `pi0`, `pi05`, `gr00t`, `wan_oft`, and
`all`. Every invocation creates or updates the shared `latency` Conda
environment for benchmark training, teacher rollout, dataset export, and
development tools. Each selected model also creates a separate
`starvla_rl_games_<model>` environment.

Use `--tier use` to omit development/test packages from the selected model
environment. The shared `latency` environment always includes the data and
teacher-training tools used by the experiment scripts.

Create a local `.env` for services used by experiments:

```bash
HF_TOKEN=...
HF_PROFILE_REPO=latency-sensitive-bench/profiles
HF_PROFILE_REPO_TYPE=dataset
WANDB_API_KEY=...
```

## StarVLA Layout

For the current experiment scripts, keep StarVLA as a sibling checkout:

```text
/path/to/workspace/
  latency-sensitive-bench/
  starVLA/
```

Most StarVLA launch scripts assume:

```bash
export WORKSPACE_DIR=/path/to/workspace
export PYTHONPATH="${WORKSPACE_DIR}/latency-sensitive-bench:${PYTHONPATH:-}"
```

The StarVLA bash scripts are the canonical entry points for VLA training runs.
They call StarVLA's launcher and model code, but rely on this repository for
latency datasets, environment wrappers, profile artifacts, and benchmark
dependencies.

Example StarVLA launchers:

```text
../starVLA/examples/rl_games/bash_scripts/openvla/bridge/mixed/flappy.sh
../starVLA/examples/rl_games/bash_scripts/openvla/bridge/mixed/demon_attack.sh
../starVLA/examples/rl_games/bash_scripts/pi_0.5/bridge/single/demon_attack.sh
../starVLA/examples/rl_games/bash_scripts/gr00t/bridge/single/demon_attack_new_lr.sh
../starVLA/examples/rl_games/bash_scripts/openvla/bridge/ghost_trail/final_ghost_trail_demon_attack_latency_6_memory_15_2.sh
../starVLA/examples/rl_games/bash_scripts/openvla/bridge/ghost_trail/final_flappy_latency_4_ablation_memory_2.sh
```

## Core Concepts

### Latency Execution Semantics

The benchmark separates environment stepping, observation emission, model
inference, action readiness, and action application.

```text
RawEnv timeline:       r = 0, 1, 2, 3, ...
Observation timeline:  o_k = k * obs_stride_raw_frames

observation -> policy/model -> delayed action event -> raw-frame action queue
```

Latency is the delay between sending an observation to a policy and the
resulting action becoming available to the environment. Actions are applied on
the raw environment timeline, not only at observation steps. See
[docs/latency_execution_semantics_en.md](docs/latency_execution_semantics_en.md)
for the full semantic contract.

### Latency Conditions

The benchmark is designed around several latency regimes:

- **zero latency:** actions apply immediately;
- **fixed latency:** every action is delayed by the same number of frames;
- **IID jitter:** each action samples its delay independently from a distribution;
- **temporal/profile replay:** actions use a measured or fitted device latency trace;
- **realtime profiling:** latency is measured on a physical target machine.

StarVLA `eval_core` currently supports fixed latency lists. The browser-local
human Pyodide runtime already supports per-action IID jitter. Profile replay and
cross-device simulation are owned by this repository's `latency_bench`
cross-device workflow.

### Rollout Datasets

Teacher policies generate clean rollout datasets under controlled latency
conditions. These datasets are stored locally and on Hugging Face, then converted
into StarVLA-compatible datasets for SFT.

Typical dataset fields include image observations, actions, episode metadata,
and latency labels such as `latency` or `latency_raw_frames`.

### Device Profiles

A deployment profile records how a specific machine behaves under a specific
workload:

- GPU identity and runtime metadata;
- measured end-to-end latency traces;
- inference latency traces;
- fitted latency distributions;
- burst/temporal latency models.

Profiles are stored under:

```text
profiles/<device>/<instance>/<workload>/<revision>/profile.json
```

They can be collected on one machine and replayed on another machine for
profile-shaped evaluation.

### Memory And Temporal Context

The benchmark includes several ways to give agents temporal information:

- raw single-frame VLA baselines;
- mixed-latency datasets;
- ghost-trail image transforms;
- multiframe and stitched-frame inputs;
- KV memory and memory-flex training/evaluation.

See [docs/memory.md](docs/memory.md) for the design and current implementation
status.

## Workflows

### 1. Generate Teacher Rollout Data

Owner: `latency-sensitive-bench`

Teacher rollouts are generated from small policies and exported as datasets:

```bash
conda activate latency

MAX_EPISODES=200 \
MAX_STEPS=3600 \
VAL_FRACTION=0.1 \
FILTER_PRESET=clean_v1 \
DEVICE=cuda \
NUM_ENVS=2048 \
REPLAY_NUM_ENVS=16 \
IMAGE_WRITER_WORKERS=32 \
bash scripts/bash_scripts/run_sf_teacher_dataset_export.sh \
  checkpoints/flappy_sf_same_latency_l0_seed0 \
  outputs/teacher_datasets/flappy_fix_latency_0_200ep
```

Detailed guide: [docs/standard_pipeline.md](docs/standard_pipeline.md).

### 2. Train A VLA With StarVLA

Owner: `starVLA`, using datasets and dependencies from
`latency-sensitive-bench`

Example mixed-latency OpenVLA training:

```bash
cd ../starVLA
bash examples/rl_games/bash_scripts/openvla/bridge/mixed/flappy.sh
```

Example single-latency and model-family launchers:

```text
examples/rl_games/bash_scripts/openvla/bridge/single/demon_attack.sh
examples/rl_games/bash_scripts/pi_0.5/bridge/single/demon_attack.sh
examples/rl_games/bash_scripts/gr00t/bridge/single/demon_attack_new_lr.sh
```

Example memory/ghost-trail launchers:

```text
examples/rl_games/bash_scripts/openvla/bridge/ghost_trail/final_ghost_trail_demon_attack_latency_6_memory_15_2.sh
examples/rl_games/bash_scripts/openvla/bridge/ghost_trail/final_flappy_latency_4_ablation_memory_2.sh
```

### 3. Evaluate A VLA Under Fixed Latency

Owner: `starVLA` for most VLA post-train runs; `latency-sensitive-bench` for
benchmark-level config evaluation.

StarVLA fixed-latency eval is usually configured through:

```text
rl_games.env_eval.post_train.latencies=[0,1,2,3,4]
rl_games.env_eval.eval_backend=eval_core
```

The `eval_core` path evaluates each listed latency as a separate fixed-latency
condition.

### 4. Evaluate Changing Latency

Owner depends on the path:

- `latency-sensitive-bench-realtime-human`: Pyodide human eval supports
  per-action IID jitter with uniform or normal sampling.
- `latency-sensitive-bench`: cross-device simulation supports temporal, IID,
  fixed, and normal profile-derived replay.
- `starVLA`: `eval_core` currently uses fixed latency queues; per-action jitter
  needs a small queue/sampler extension before it exactly matches the Pyodide
  jitter protocol.

### 5. Profile A New Machine

Owner: `latency-sensitive-bench`

Run the target-machine profile workflow:

```bash
cd latency-sensitive-bench

SESSIONS="0 1 2 3 4" \
FIT_SESSIONS="0 1 2 3 4" \
PROFILE_REVISION=fit-0-4 \
METHOD=temporal \
LATENCY_SEEDS="0 1 2" \
bash start.sh
```

This measures real sessions, publishes raw traces, fits `profile.json`, uploads
the profile, runs profile-shaped simulation, and publishes simulation results.

Resolve or download a published profile:

```bash
latency-bench profile resolve --hardware auto --revision latest

PROFILE_REF='rtx4090/<instance-id>/openvla-flappy/fit-0-4' \
  bash scripts/bash_scripts/cross_device/download_profile.sh
```

Detailed guides:
[docs/new_machine_profile_train.md](docs/new_machine_profile_train.md) and
[docs/openvla_cross_device_handoff.md](docs/openvla_cross_device_handoff.md).

### 6. Run Human Realtime Evaluation

Owner: `latency-sensitive-bench-realtime-human`

```bash
cd ../latency-sensitive-bench-realtime-human
bash run.sh
```

Open:

```text
http://127.0.0.1:8002
http://127.0.0.1:8002/pyodide
```

The FastAPI path runs backend-driven human realtime evaluation. The Pyodide path
runs a browser-local prototype with local latency scheduling and export.

## Research Workflows

| Goal | Primary code path |
| --- | --- |
| Measure VLA degradation under fixed latency | StarVLA `bridge/single` scripts plus `rl_games.env_eval.eval_backend=eval_core` |
| Train mixed-latency VLA baselines | StarVLA `bridge/mixed` scripts consuming latency-labeled datasets |
| Compare memory and temporal-context methods | `scripts/experiments/memory/*` plus StarVLA ghost-trail/KV-memory scripts |
| Build fixed-latency and ghost-trail rollout datasets | `scripts/bash_scripts/rollout_data/*` and `scripts/bash_scripts/ghost_trail/*` |
| Profile a new deployment machine | `bash start.sh` and `bash cross_device.sh ...` |
| Replay a target device profile on another machine | `cross_device.sh sim --profile-ref ... --method temporal` |
| Compare humans and VLAs under changing latency | Pyodide human eval plus StarVLA jitter-eval extension |

## Documentation

- [Latency execution semantics](docs/latency_execution_semantics_en.md)
- [Standard rollout, SFT, and eval pipeline](docs/standard_pipeline.md)
- [Memory and KV-memory design](docs/memory.md)
- [New-machine profile and training handoff](docs/new_machine_profile_train.md)
- [Cross-device profile handoff](docs/openvla_cross_device_handoff.md)
- [Cross-device protocol runbook](docs/experiments/cross_device_protocol_runbook.md)
- [WanOFT StarVLA eval](docs/starvla_wan_oft_eval.md)
- [Realtime pipeline logging](docs/realtime_pipeline_logging.md)

## Datasets, Checkpoints, And Profiles

The project uses Hugging Face repositories for public artifacts:

- rollout datasets and memory rollouts;
- fixed-latency and mixed-latency training data;
- ghost-trail datasets;
- model checkpoints;
- device profiles in `latency-sensitive-bench/profiles`.

Exact dataset and checkpoint names are recorded in the corresponding StarVLA
bash scripts and experiment runbooks.

## Citation

If you use this benchmark, cite the paper once the citation entry is available.

```bibtex
@misc{latency_sensitive_bench,
  title = {Latency-Sensitive Bench},
  author = {...},
  year = {...},
}
```
