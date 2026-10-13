# Run the paper experiment recipes

Run commands from the repository root with the selected task dependencies installed.
Each recipe writes local configurations, checkpoints, evaluation records, summaries, and figures.

| Experiment | Entrypoint |
| --- | --- |
| Observation stride | `scripts/experiments/observation_stride/run.sh` |
| Memory / small model | `scripts/experiments/memory/small_model/run.sh` |
| Jitter robustness | `scripts/experiments/jitter_robustness/run.sh` |
| Held-out Sim2Real | [Experiment guide](../../scripts/experiments/sim2real/README.md) |
| Teacher datasets | [Collection recipes](../../scripts/experiments/memory/README.md) |
| HumanoidBench assets and training | [Experiment guide](../../scripts/experiments/humanoidbench/README.md) |

## Select stages

| Stage | Effect |
| --- | --- |
| `generate` | Generate run configurations and manifests |
| `train` | Train the selected checkpoints |
| `eval` | Evaluate existing checkpoints |
| `build-data` | Aggregate complete local evaluation summaries |
| `plot` | Plot the selected local CSV |
| `all` | Run all five stages in order |

For observation stride:

```bash
bash scripts/experiments/observation_stride/run.sh generate
bash scripts/experiments/observation_stride/run.sh train --gpus 0,1
bash scripts/experiments/observation_stride/run.sh eval --gpus 0,1
bash scripts/experiments/observation_stride/run.sh build-data
bash scripts/experiments/observation_stride/run.sh plot
```

Use `all --gpus 0,1` to run the complete local experiment.
Use `generate --seeds CSV` and `train` or `eval --match TEXT` for partial runs.
Inspect the entrypoint's help before selecting stage-specific arguments.

## Preserve the checkpoint protocol

Observation stride and memory train the extended Flappy and Demon Attack variants and the single Deadly Corridor variant.
Evaluation runs the capped and extended variants.
Capped evaluation reuses the corresponding extended checkpoint.

Jitter robustness uses four specifications under `configs/experiments/jitter_robustness/`.
Each selects five single-profile and two mixed-profile training conditions.
Demon Attack trains from the capped manifest and evaluates at 3,600 and 36,000 steps.
Deadly Corridor uses 35/8.75 Hz and a 30,000-step cap.

## Run one manifest and resume

```bash
python scripts/experiment_runner/run_manifest_jobs.py \
  --manifest /ABSOLUTE/GENERATED/manifest.tsv \
  --stage train --gpus 0,1
```

The manifest runner supports `all`, `train`, and `eval`.
Its default `all` runs training and evaluation.
Evaluation does not start training.

The runner records events in `runner_events.jsonl` beside the manifest.
It reuses successful stages when the same manifest runs again.
Use a new event log after changing a job's training or evaluation protocol.
`--dry-run` prints commands without writing progress events.
`--gpus` selects each worker's visible devices on the current host.

## Read local outputs

Summary CSVs and figures appear under `outputs/<experiment>/paper/`.
Aggregation requires the complete expected result set.
It fails when a seed, profile, or axis value is missing.
Training recovery files remain local.

The released recipes disable W&B by default.
Enable it through your own configuration overlay when needed.
The [artifact guide](../artifacts.md) links published experiment manifests and their fixed inputs.
