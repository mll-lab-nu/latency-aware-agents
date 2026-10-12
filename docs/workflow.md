# Use the stages independently

Start with your existing environment, teacher, dataset, model, or latency profile.
Each stage saves artifacts that the next stage can consume.
Install the selected dependencies through the [installation guide](installation.md).
Run reference recipe commands from the repository root.

| Your input | Stage | Output |
| --- | --- | --- |
| Environment and policy | Evaluation | Episode metrics, resolved configuration, optional traces |
| Environment and teacher | Collection | Canonical demonstrations or task-specific raw data |
| Demonstrations | Conversion | Training data and input/action metadata |
| Training data | Your trainer or reference recipe | Checkpoint and model input/output contract |
| Environment and model | Realtime profiling | Measurements and fitted latency profile |
| Model and profile | Simulated evaluation | Results under the selected timing conditions |

LAGEN owns execution timing and the stage interfaces.
Your training framework owns model training.
Sample Factory and StarVLA provide reference recipes.

## Evaluate an existing model

Select the checkpoint, dataset metadata, and backbone through the [artifact guide](artifacts.md).
Preserve the model's task, observations, actions, normalization, and horizon.
You do not need to train a teacher or collect data first.

Create a local overlay for the appropriate evaluation recipe.
The [MIKASA cookbook](cookbooks/mikasa.md#evaluate-an-existing-model) provides a complete example.
The artifact guide provides a generic Gymnasium overlay.
Run the evaluation:

```bash
python -m latency_bench eval --config my_eval.yaml
```

`paused` waits for each prediction. `simulated` applies the configured delay.
`realtime` measures actual execution timing.
See [configuration](guides/configuration.md) and [execution semantics](reference/latency_execution_semantics_en.md).

## Connect an environment and teacher

Follow the [external component example](guides/custom_components.md).
It includes CPU evaluation, RGB collection, conversion, profiling, and profile replay.
Your own package supplies the environment and policy factories.
The [Gymnasium guide](guides/gymnasium_adapter.md) also explains the public training wrapper.

Collect and convert your demonstrations:

```bash
python -m latency_bench collect --config collect.yaml
python -m latency_bench convert --config convert.yaml
```

The [collection guide](guides/rollout_collection.md) defines student inputs, teacher labels, splits, and converter outputs.
Generic collection supports paused and simulated execution with single-action teacher outputs.
Task-specific recipes own multi-view and chunk labels.

## Use the task training recipes

| Task | Recipe | Student inputs and actions |
| --- | --- | --- |
| MIKASA Intercept Grab Fast | [MIKASA cookbook](cookbooks/mikasa.md) | Two RGB views, 7D state, H1 × 7D action |
| HumanoidBench Balance Simple | [Balance cookbook](cookbooks/balance.md) | Head-camera RGB, physical state, H8/H16 × 43D actions |
| Games and Gymnasium | [Gymnasium guide](guides/gymnasium_adapter.md), [experiment guide](experiments/README.md) | Task and checkpoint contract |

For MIKASA, the shared launcher accepts a teacher checkpoint or a prepared dataset:

```bash
conda activate latency
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/tasks/mikasa_intercept_grab_fast/train_qwenoft.sh \
  --dataset-dir /ABSOLUTE/DATASET_PARENT \
  --run-id mikasa-qwenoft-local
```

The dataset parent must contain `lerobot/` and `lerobot__val/` with matching task contracts.
Omit `--dataset-dir` to prepare the shared published dataset.
Add `--teacher-checkpoint /ABSOLUTE/TEACHER/checkpoint.pt` to select your teacher.
H1 bootstrap data still comes from the prepared or published dataset.
Selecting a teacher does not generate a new H1 dataset automatically.

Add `--smoke` for two training updates and two evaluation episodes.
A smoke run checks execution and checkpoint loading. It does not establish policy quality.
The full MIKASA H1 recipe uses 5,000 SFT updates, global batch 128, and 100 final evaluation episodes.
Its final evaluation seeds are 4242424242–4242424341.

The launcher defaults to two training GPUs with microbatch 64 per GPU.
Use `--num-gpus` and `--micro-batch` to select resources.
Their product must divide the configured global batch.
On CUDA OOM, the runner halves the microbatch and adjusts accumulation to preserve the global batch.
Other errors propagate.

| Argument | Purpose |
| --- | --- |
| `--vla-python`, `--task-python` | Select absolute model and simulator interpreter paths |
| `--simulator-device`, `--inference-device` | Select logical evaluation devices, default `cuda:0` and `cuda:1` |
| `--output-dir` | Select the local run directory |
| `--eval-episodes`, `--eval-seed` | Select final evaluation count and first seed |
| `--resume --run-id ID` | Continue the matching saved run |

Balance uses its own H1 teacher → H8/H16 SFT/DAgger schedule.
Follow its cookbook rather than changing only the task name in the MIKASA command.

## Measure and replay a profile

Select your checkpoint in a realtime profiling configuration.
Run measurement and fitting:

```bash
python -m latency_bench profile run --config my_profile.yaml
```

`profiling.repeats` counts total repetitions.
`evaluation.episodes` counts measured episodes per repetition.
The command returns a measurement directory and fitted profile path.
To fit saved complete measurements, run:

```bash
python -m latency_bench profile fit --run-dir /ABSOLUTE/MEASUREMENT
```

Use `latency.method: temporal` or `iid` with the returned profile in a simulated evaluation config.
The [profiling cookbook](cookbooks/profiling.md) provides complete examples and recovery commands.

## Read local outputs

Keep datasets, checkpoints, logs, configurations, profiles, and reports under `outputs/`.
Record input revisions, paths, episode counts, clocks, latency settings, and seeds with results.
Training completion and evaluation completion remain separate states.

The task trainer saves an `inference/` bundle with the checkpoint, model configuration, task contract, statistics, and provenance.
Balance also saves its selected bundle, held-out results, and completed SFT/DAgger schedule.
Profiling prints its measurement directory, fitted profile path, and completion status.
Paper experiment recipes write local summaries and figures.

W&B is optional. Add `--wandb-project YOUR_PROJECT` to a task training command to enable it.
Use `--wandb-entity YOUR_TEAM` when the project belongs to a team.
The [installation guide](installation.md#access-remote-inputs-and-optional-logs) explains authentication.
See the [artifact guide](artifacts.md) for published inputs and their source protocols.
