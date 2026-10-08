# Standard task pipeline

Install dependencies with [QUICK_START.md](QUICK_START.md). Run commands from
the repository root. The task cookbooks define the required inputs:

| Task | Training guide | Default student |
| --- | --- | --- |
| HumanoidBench Balance Simple | [Balance cookbook](humanoidbench/cookbook.md) | RGB + 198D state, H8 × 43D actions |
| MIKASA Intercept Grab Fast | [MIKASA cookbook](mikasa/cookbook.md) | Two RGB views + 7D state, H1 × 7D action |

## Train a task

Each task directory provides `train_qwenoft.sh`, `train_qwenpi_v3.sh`,
`train_qwengr00t.sh`, and `train_wanoft.sh`. These scripts call
`python -m scripts.starvla.train_task`.

For MIKASA, run a local smoke with the shared launcher:

```bash
conda activate latency
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/tasks/mikasa_intercept_grab_fast/train_qwenoft.sh \
  --smoke --run-id mikasa-qwenoft-smoke
```

For a full local run, replace `--smoke` with `--no-publish` and use a new run ID.
Balance additionally requires a matching H8 teacher checkpoint and training
config. Use the explicit command in its cookbook.

The current shared runner uses these budgets:

| Task | SFT updates | DAgger | Global batch | Final evaluation |
| --- | ---: | --- | ---: | --- |
| Balance | 5,000 | 10 rounds × 500 updates | 128 | 100 episodes, seeds 0–99 |
| MIKASA H1 | 5,000 | None in this runner | 128 | 100 episodes, seeds 4242424242–4242424341 |

These are current defaults, not the settings of every published experiment.
A smoke uses two SFT updates and two evaluation episodes. Balance also uses
one DAgger round with two updates. Smoke results do not establish policy quality.

The runner uses two training GPUs by default. It starts at microbatch 64 per GPU.
On CUDA OOM, it halves the microbatch and adjusts accumulation to preserve the
global batch. Other errors propagate. The training YAML records the resolved
settings. Use `--num-gpus` and `--micro-batch` to select a different allocation.
Their product must divide the formal global batch.

Useful arguments:

| Argument | Input or effect |
| --- | --- |
| `--teacher-checkpoint`, `--teacher-config` | Matching teacher weights and resolved training configuration |
| `--dataset-dir` | Prepared parent directory containing `lerobot/` and `lerobot__val/` |
| `--vla-python`, `--task-python` | Absolute interpreter paths for model and simulator environments |
| `--simulator-device`, `--inference-device` | Default `cuda:0` and `cuda:1` for task evaluation |
| `--output-dir` | Local run directory |
| `--eval-episodes`, `--eval-seed` | Episode count and first seed for final evaluation |
| `--dagger-rounds 0` | Stop Balance training after SFT |
| `--resume --run-id ID` | Continue the matching run from its saved state |
| `--no-publish` | Keep a full run local |

Use `python -m scripts.starvla.train_task --help` for the complete argument list.

## Data and training contracts

The teacher supplies the action labels. Select the teacher using the task's
independent evaluation protocol before generating a new training dataset.

Balance collects closed-loop teacher trajectories into a new bootstrap dataset
unless `--dataset-dir` supplies one. Each label contains the actions selected
across eight physical frames, with a mask for valid targets. Formal collection
uses 90 training episodes and 10 validation episodes.

MIKASA H1 reuses the published teacher dataset and its fixed 250/25 split.
The shared runner prepares its task contract and normalization metadata.
MIKASA H8 uses a separate suffix-label conversion. Its conditioned variant
requires an explicitly prepared dataset. See the task cookbook for those contracts.

Balance DAgger labels branches from student states with the fixed teacher.
Half of the sampling weight remains on bootstrap data. The other half is
shared across accumulated DAgger rounds. Validation data remain separate.
The optimizer continues across rounds with one cosine schedule for the phase.
MIKASA DAgger uses task-specific entrypoints, not this shared runner.

An interrupted partial collection restarts that round. A completed training
state supports `--resume`. Preserve its run ID, inputs, configuration, and data.

## Evaluate and profile

The shared runner evaluates the final bundle at zero latency. It prints
`MODEL_PATH=<checkpoint path>`. Pass that exact path to the profiler:

```bash
bash scripts/profile.sh run \
  --model qwenoft --workload mikasa_intercept_grab_fast \
  --model-path /ABSOLUTE/PATH/inference/checkpoints/model.pt
```

Profiling uses the workload's `session_count` and `eval_episodes` settings.
Current Balance and MIKASA presets use five sessions of 50 episodes.
It fits action latency from `observation_attempts.jsonl`. See the
[profiling cookbook](profiling/cookbook.md) for hardware allocation and replay.
A simulated control rate does not establish realtime inference throughput.

## Artifacts and publication

Formal training publishes data and the final model unless you pass `--no-publish`.
Smoke runs stay local. Profiling uploads only with `--publish`.
Use local runs to reproduce results without organization write access.

Model and dataset destinations are `MLL-Lab/LAGEN-models` and
`MLL-Lab/LAGEN-datasets`. The [artifact registry](artifacts.md) describes their
layout. Record the resolved revisions because the runner downloads some training inputs
from the default branch. Reproduction requires the experiment's fixed inputs.

If publication fails after training, publish the saved run with `--publish-only`
and the same `--output-dir`. Publication replaces files at the selected target path.
Use a new run ID for a new model. Shared bootstrap paths depend on the task,
condition, horizon, and dataset version, not the run ID.
Keep training completion, evaluation, acceptance, and publication as separate states.

## Older game and generic adapter paths

Game experiments use the existing launchers in `scripts/experiments/` and
`third_party/starVLA/examples/rl_games/`. See the
[experiment guide](experiment_runner.md) for the available sweeps.
[Demon Attack evaluation](demon_attack.md) defines full-game scoring and life-loss resets.

Use the [Gymnasium adaptation guide](gymnasium_adapter.md) to add an environment.
Use the [execution semantics](latency_execution_semantics_en.md) to select timing
and scheduling behavior. Preserve each experiment's declared latency conditions.
