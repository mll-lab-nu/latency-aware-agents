# Standard zero-latency pipeline

The four supported benchmark tasks share the same outer experiment:

```text
small-policy RL teacher
  -> accepted teacher checkpoint
  -> successful teacher rollouts
  -> task dataset conversion
  -> VLA SFT
  -> optional task-specific DAgger
  -> zero-latency evaluation
  -> deployment profile
```

The task implementations do not have identical observation or action spaces.
The common part is the order of the stages and the acceptance boundary between
them. Use the task guide for the exact environment, dimensions, and gates.

## Reader quick start

Run commands from the repository root after installing the task's dependencies.
Every task owns train and profile entrypoints:

| Task | Train the small-policy teacher | Train and evaluate QwenOFT | Profile QwenOFT |
| --- | --- | --- | --- |
| MIKASA Intercept Grab Fast | `bash scripts/tasks/mikasa_intercept_grab_fast/train_small_policy.sh` | `bash scripts/tasks/mikasa_intercept_grab_fast/train_qwenoft.sh --run-id RUN_ID` | `bash scripts/profile.sh run --model qwenoft --workload mikasa_intercept_grab_fast --model-path MODEL_PATH` |
| HumanoidBench Balance Simple | `bash scripts/tasks/humanoidbench_balance_simple/train_small_policy.sh` | `bash scripts/tasks/humanoidbench_balance_simple/train_qwenoft.sh --run-id RUN_ID` | `bash scripts/profile.sh run --model qwenoft --workload g1_sonic_balance_simple --model-path MODEL_PATH` |
| Extreme Parkour Go2 | `bash scripts/tasks/extreme_parkour_go2/train_small_policy.sh` | `bash scripts/tasks/extreme_parkour_go2/train_qwenoft.sh --run-id RUN_ID` | `bash scripts/profile.sh run --model qwenoft --workload extreme_parkour_go2 --model-path MODEL_PATH` |
| HAIC Pull Cart G1 | `bash scripts/tasks/haic_pull_cart/train_small_policy.sh` | `bash scripts/tasks/haic_pull_cart/train_qwenoft.sh --run-id RUN_ID` | `bash scripts/profile.sh run --model qwenoft --workload haic_pull_cart --model-path MODEL_PATH` |

`train_small_policy.sh` runs the native RL recipe and leaves the checkpoint in
the task output directory. Inspect the reported path and pass it with `--teacher-checkpoint`; an unverified or implicit `latest` checkpoint is not accepted. The StarVLA launchers own rollout collection, conversion, SFT, task-specific DAgger, evaluation, and formal publication. Pass `--no-publish` to keep a formal run local.

The [StarVLA pipelines](../program/SCALE_MODELS.md) provide
`train_qwenoft.sh`, `train_qwenpi_v3.sh`, `train_qwengr00t.sh`, and
`train_wanoft.sh` in each task directory. They download teachers from the
model repository's default branch and
automatically publish formal datasets and final models; `--smoke` stays local.

Each VLA trainer finishes by printing `MODEL_PATH=<final inference bundle>`.
Pass that path to `scripts/profile.sh run --model MODEL --workload TASK --model-path MODEL_PATH`. If the local model root is absent, the profiler downloads that exact
artifact from `latency-sensitive-bench/benchmark-models`; an existing but
incomplete root fails instead of selecting another checkpoint. Profiling stays
local unless `--publish` is supplied.

Balance, Parkour, and HAIC run SFT followed by ten DAgger rounds by default. To
stop after SFT, pass `--dagger-rounds 0`. Resume the same `--run-id` with
`--resume`.

## Stage contracts

### 1. Train and accept the teacher

The small policy is trained with privileged state in the native task
environment. Its checkpoint is the label source for all later stages. Run the
task's teacher gate before collecting data. If the gate fails, keep the output
for diagnosis and do not start VLA training from it.

The teacher run is zero-delay: the task applies the current policy output at its
native control cadence. This is the baseline used by all four guides; latency
sweeps and deployment profiles are separate experiments.

### 2. Collect accepted demonstrations

The VLA dataset is collected only from an accepted teacher checkpoint. The
collector writes successful episodes into the task's output tree and records
the task observation/action contract. The train/validation split is fixed by
the task guide rather than chosen after training.

The labels are not interchangeable across tasks:

| Task | VLA label/interface |
| --- | --- |
| MIKASA | Direct 7D current action |
| Balance Simple | StarVLA 198D state to H8 43D absolute joint targets |
| Extreme Parkour | 34D terrain latent plus yaw, then fixed actor to 12D motor action |
| HAIC Pull Cart | 256D latent, then frozen `actor_adapt` with 605D state to 23D motor action |

### 3. Convert and train the VLA

Each task-owned VLA entrypoint converts its accepted rollouts to the format
required by its training backend, trains the documented SFT recipe, and writes
an inference checkpoint. Parkour and HAIC use their fixed actor only to define
the action-level objective; that actor is not part of the VLA checkpoint.

### 4. Run DAgger where the task requires it

MIKASA finishes at the SFT model. Balance, Parkour, and HAIC collect current-VLA
rollouts against the fixed teacher and aggregate them with the bootstrap data
for the configured DAgger rounds. Each completed round keeps its converted
dataset, checkpoint, manifest, and evaluation result; raw shards may be removed
after conversion.

An interrupted collection is restarted from an empty partial round directory,
not from a row count. An interrupted completed round resumes from its explicit
checkpoint and round number. The task runners fail at missing or incompatible
artifacts instead of substituting another checkpoint.

### 5. Evaluate zero latency

The final evaluation uses the task's native zero-delay environment and the
documented fixed seeds/episode count. It is the policy-quality gate. Offline
action or latent loss is useful for diagnosis but is not a replacement for the
closed-loop result.

### 6. Profile the final VLA

The profile entrypoint runs five native realtime sessions of 100 episodes and
fits the canonical `observation_attempts.jsonl` sequence. MIKASA, Parkour, and
HAIC reserve logical `cuda:0` for simulation and use `cuda:1..` for inference;
Balance uses every visible GPU for inference. `--publish` uploads only after all
sessions and profile construction complete.

## Developer artifact workflow

Keep large artifacts outside the source tree when possible. A task run should
leave these durable outputs:

```text
<output-root>/
  teacher/       # accepted small-policy checkpoint and gate summary
  dataset/       # raw/converted train and validation data
  sft/           # VLA inference checkpoint and training metadata
  dagger/        # Parkour/HAIC rounds, if applicable
  eval/          # final zero-latency metrics
```

Publish only the inference bundle, accepted dataset, and concise evaluation
evidence. Do not publish optimizer state, W&B caches, temporary raw shards, or
machine-specific absolute paths. The central artifact layout and historical
source revisions are recorded in [the artifact registry](artifacts.md).

Publishing is an explicit developer operation with the Hugging Face CLI. The
two central repositories are:

- models: `latency-sensitive-bench/benchmark-models`;
- datasets: `latency-sensitive-bench/benchmark-datasets`.

Use the task cookbook to identify the exact local source directories and the
registry path. A reader can reproduce a run without organization write access;
only the publish step requires it.

## Older game and generic adapter paths

The original Flappy Bird, Demon Attack, and Deadly Corridor experiments remain
supported as research workflows, but are not part of the four-task table above.
Their existing rollout exporters, StarVLA conversion scripts, latency configs,
and model-family launchers remain under `scripts/bash_scripts/` and the sibling
StarVLA checkout. Start with the historical commands in the repository history
or adapt the task-specific scripts under `scripts/experiments/`.

For new Gymnasium environments, use the separate [Gymnasium adaptation guide](gymnasium_adapter.md)
and its generated contract report. For latency semantics and device profiles,
use [latency execution semantics](latency_execution_semantics_en.md) and the
[profiling cookbook](profiling/cookbook.md); do not add latency to this
zero-delay training pipeline implicitly.
