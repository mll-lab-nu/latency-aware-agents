<h1 align="center">Latency-Sensitive Bench</h1>
<h3 align="center"><em>Profiling, evaluating, and training agents under real deployment latency.</em></h3>

Latency-Sensitive Bench studies what happens when real-time agents act through
delayed, variable, device-dependent action pipelines. The benchmark treats
latency as a first-class deployment condition: observations are emitted on an
observation timeline, actions become ready after a latency delay, and the
environment applies actions on the raw environment timeline.

This repository owns the benchmark runtime, latency execution semantics,
rollout/profile generation, and experiment configs. Most VLA training is
launched from the companion StarVLA repository. NVIDIA's official GR00T path is
the exception: `official_gr00t` trains and runs directly from the pinned
`third_party/Isaac-GR00T` submodule.

## Overview

Latency-Sensitive Bench provides:

- latency-aware execution for Flappy Bird, Demon Attack, and Deadly Corridor;
- fixed-latency, jittered-latency, and device-profile evaluation semantics;
- teacher rollout generation and Hugging Face dataset export;
- cross-device latency profiling, profile fitting, and profile replay;
- memory, temporal-context, ghost-trail, and KV-memory experiment support;
- integration points for OpenVLA, pi0, pi0.5, the existing StarVLA GR00T,
  NVIDIA official GR00T, WanOFT, and human realtime evaluation.

The current manipulation research path integrates the MIKASA
`InterceptGrabFast-VLA-v0` task as a zero-delay H1 baseline: an official 7D
state PPO teacher produces demonstrations, StarVLA trains a QwenOFT policy on
the resulting LeRobot v3 dataset, and PPO/VLA policies are evaluated at the
task's native 20 Hz (50 ms) control cadence. See the executable
[`docs/mikasa/cookbook.md`](docs/mikasa/cookbook.md) reproduction guide, the executable
[`program/MIKASA.md`](program/MIKASA.md) runbook, and the deferred latency
[`program/PROPOSAL.md`](program/PROPOSAL.md).

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

Accepted model names are `openvla`, `pi0`, `pi05`, `gr00t`,
`official_gr00t`, `wan_oft`, and `all`. Every invocation creates or updates the shared `latency` Conda
environment for benchmark training, teacher rollout, dataset export, and
development tools. StarVLA model targets create a separate
`starvla_rl_games_<model>` environment; `official_gr00t` uses Isaac-GR00T's own
uv environment.

MyoChallenge 2025 is an optional Gymnasium task dependency. Install its
pinned runtime only in environments that run those tasks:

```bash
uv pip install -e '.[myochallenge]'
export MUJOCO_GL=egl  # headless MuJoCo rendering
```

Use `--tier use` to omit development/test packages from the selected model
environment. The shared `latency` environment always includes the data and
teacher-training tools used by the experiment scripts.

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

## Official GR00T Layout

`gr00t` remains the existing StarVLA/Qwen implementation. The separate
`official_gr00t` policy loads NVIDIA GR00T N1.7 directly inside one local
inference worker process; it does not use a network policy server.

Official dataset conversion and torchcodec require a system FFmpeg version
supported by the pinned GR00T release (FFmpeg 4–7). Model loading also requires
access to the gated `nvidia/Cosmos-Reason2-2B` backbone, even when the GR00T
checkpoint is already local. The installer does not run `sudo`, install system
packages, or authenticate Hugging Face.

```bash
ffmpeg -version
hf auth login
bash install.sh --model official_gr00t
```

The HumanoidBench integration uses a custom `NEW_EMBODIMENT` contract with a
93D G1 proprioceptive state and a 40-step by 29D absolute body-joint action
chunk. Training uses Isaac-GR00T's official `launch_finetune.py` entry point and
checkpoint format. Training and closed-loop inference use the same
latency-neutral task prompt; measured latency is represented by the action
chunk scheduler, not appended to the language instruction.

The canonical SONIC teacher index is
[`configs/teachers/g1_sonic_gr00t.json`](configs/teachers/g1_sonic_gr00t.json).
It is the single source of truth for the fixed hurdle, run-reference, and
balance inputs, per-task rollout caps, and episode acceptance gates. Hurdle
requires `teacher_accepted=true`; balance and run must survive at least 18
seconds, which is 900 of each 1000 available raw frames at 50 Hz. The exporter
derives each `raw` and `lerobot` output from the teacher name:

```bash
python scripts/teachers/export_g1_sonic_gr00t.py \
  --output-root results/g1_sonic_gr00t_teachers \
  --max-episodes 3 --force
```

The three LeRobot datasets stay separate. Pass one dataset path for single-task
training, or combine paths with `:` for joint training; `--ds-weights-alpha 0`
gives equal task weighting when the loader supports it. A merged root is not
used for task filtering.

A minimal end-to-end smoke run selects only the hurdle teacher and splits its
three accepted episodes into two train and one validation episode:

```bash
python scripts/teachers/export_g1_sonic_gr00t.py \
  --output-root outputs/official_gr00t/teachers \
  --teacher g1_hurdle_sonic_residual_h025_gamma099_1m_v1 \
  --max-episodes 3 --val-fraction 0.34 --device cuda --force

CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
BASE_MODEL_PATH=outputs/official_gr00t/models/GR00T-N1.7-3B \
DATASET_PATH=outputs/official_gr00t/teachers/g1_hurdle_sonic_residual_h025_gamma099_1m_v1/lerobot \
OUTPUT_DIR=outputs/official_gr00t/g1_hurdle_smoke \
bash scripts/official_gr00t/finetune.sh \
  --max-steps 2 --global-batch-size 1 --dataloader-num-workers 0
```

Run the official open-loop evaluator on the held-out episode:

```bash
uv run --project third_party/Isaac-GR00T python \
  third_party/Isaac-GR00T/gr00t/data/stats.py \
  --dataset-path outputs/official_gr00t/teachers/g1_hurdle_sonic_residual_h025_gamma099_1m_v1/lerobot__val \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path latency_bench/policy/official_gr00t_modality.py

uv run --project third_party/Isaac-GR00T python \
  third_party/Isaac-GR00T/gr00t/eval/open_loop_eval.py \
  --dataset-path outputs/official_gr00t/teachers/g1_hurdle_sonic_residual_h025_gamma099_1m_v1/lerobot__val \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path outputs/official_gr00t/g1_hurdle_smoke \
  --traj-ids 0 --execution-horizon 40 --steps 80
```

For the two-step closed-loop check, set the scene and checkpoint paths in
`configs/examples/gymnasium/g1_sonic_hurdle/official_gr00t_eval.yaml`, then run
the realtime executor. Its acceptance artifact must report measured worker
latency rather than a simulated zero-latency condition:

```bash
python -m latency_bench.run \
  --config configs/examples/gymnasium/g1_sonic_hurdle/official_gr00t_eval.yaml \
  --eval-max-steps 2
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
profiles/<model>/<gpu-deployment>/<workload>/<physical-instance>/profile.json
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

Expose the GPUs that participate in inference and run the one-shot profiler:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/profile.sh run \
  --model openvla \
  --workload flappy \
  --publish
```

Resolve the exact published deployment profile:

```bash
bash scripts/profile.sh resolve --model openvla --workload flappy
```

See [the profiling guide](docs/profiling.md) for multi-GPU identity, artifact
layout, publishing, commit-pinned replay, and adding presets.

### 6. Run Human Realtime Evaluation

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
| Profile a new deployment machine | `bash scripts/profile.sh run --model openvla --workload flappy` |
| Replay a target deployment profile | `bash scripts/profile.sh simulate ... --method temporal` |
| Compare humans and VLAs under changing latency | Pyodide human eval plus StarVLA jitter-eval extension |

## Agent-Assisted Gymnasium Adaptation

New Gymnasium environments use a separate, contract-driven path. You can adapt
a task manually, but using an agent is recommended because the process requires
API inspection, explicit scientific decisions, generated configs and tests,
and cross-stage contract checks.

- Use the [Gymnasium adaptation guide](docs/gymnasium_adapter.md) as the manual
  runbook and the primary execution specification for an agent. It explains
  the supported boundary and uses the runnable
  [Inverted Pendulum configs](configs/examples/gymnasium/inverted_pendulum/) as
  a reference.
- Use the [agent-assisted Gymnasium examples](docs/gym-adapt/gymnasium/README.md)
  to understand what an adaptation changes, inspect its evidence, and continue
  from small-model training to later pipeline stages. The Inverted Pendulum and
  Pendulum reports are also the reporting templates an agent should follow.

Give the agent this prompt, replacing the bracketed fields:

```text
Read docs/gymnasium_adapter.md and docs/gym-adapt/gymnasium/README.md. Adapt [ENV_ID] from [PACKAGE + PINNED VERSION] into latency-sensitive-bench through the separate env.name=gymnasium path. My task objective is [OBJECTIVE], and the target latency is [LATENCY].

First, probe the exact Gymnasium API and report its constructor arguments, action and observation spaces, reset/step/render behavior, native timing, and anything you cannot infer. Ask me to decide the action catalog, action order, no-op action, raw-tick meaning, preprocessing, reward and termination policy, task prompt, and experiment budgets whenever these cannot be determined from the API.

After I answer, create a task-owned wrapper only if needed; generate grouped small-model train/eval, latency-bench eval, and StarVLA eval configs; preserve the resulting task contract across checkpoint, rollout, conversion, SFT, and final evaluation; run parity and compatibility tests; and keep every legacy environment path untouched. Record the commands, generated files, artifact paths, and which stages were actually run.

Create a self-contained report at docs/gym-adapt/gymnasium/[TASK_NAME]/index.html following the worked examples. Separate API facts from my decisions, show the frozen contract and generated files, document every pipeline consumer, preserve exact commands and evidence, explain how I can continue rollout/SFT/eval, and mark every stage VERIFIED, NOT RUN, or BLOCKED. Do not invent results.
```

## Documentation

- [Latency execution semantics](docs/latency_execution_semantics_en.md)
- [Standard rollout, SFT, and eval pipeline](docs/standard_pipeline.md)
- [Gymnasium adaptation guide](docs/gymnasium_adapter.md) & [agent-assisted adaptation examples](docs/gym-adapt/gymnasium/README.md)
- [Memory and KV-memory design](docs/memory.md)
- [Deployment latency profiling](docs/profiling.md)
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
