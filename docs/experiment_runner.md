# Run the paper experiment recipes

Each sweep has a public Bash entrypoint.
Run commands from the repository root with the selected task dependencies installed.

| Experiment | Entrypoint |
| --- | --- |
| Observation stride | `scripts/experiments/observation_stride/run.sh` |
| Memory / small model | `scripts/experiments/memory/small_model/run.sh` |
| Jitter robustness | `scripts/experiments/jitter_robustness/run.sh` |

## Select stages

| Command | Effect |
| --- | --- |
| `generate` | Write configurations and manifests from source specifications |
| `train` | Train the canonical checkpoint set |
| `eval` | Evaluate capped and extended variants |
| `build-data` | Aggregate complete local evaluation summaries |
| `plot` | Plot the selected local data snapshot |
| `upload` | Publish completed model and evaluation artifacts |
| `all` | Run generation, training, evaluation, aggregation, plotting, and publication |
| `published-plot` | Rebuild a pinned published snapshot from its historical sources |

Run generation and training separately when you want to retain local results:

```bash
bash scripts/experiments/observation_stride/run.sh generate
bash scripts/experiments/observation_stride/run.sh train --gpus 0,1
bash scripts/experiments/observation_stride/run.sh eval --gpus 0,1
```

Use the complete pipeline to publish results:

```bash
bash scripts/experiments/observation_stride/run.sh all --gpus 0,1
```

`all` uses the complete seed set and uploads artifacts.
Use `generate --seeds <csv>` and `train` or `eval --match <substring>` for partial runs.
`build-data` requires the complete result set.
Inspect each entrypoint's help for stage-specific inputs.

## Reuse checkpoints across evaluation variants

Observation stride and memory train the extended Flappy and Demon Attack manifests and the single Deadly Corridor manifest.
Their evaluation stage runs all five manifests.
Capped evaluation reuses its extended checkpoint.

Jitter robustness uses four source specifications under `configs/experiments/jitter_robustness/`.
Each contains five single-profile and two mixed-profile training conditions.
Demon Attack trains once from the capped manifest and evaluates at 3,600 and 36,000 steps.
Deadly Corridor uses 35/8.75 Hz and a 30,000-step horizon.

## Run one manifest and resume

The existing manifest runner supports individual stages:

```bash
python scripts/experiment_runner/run_manifest_jobs.py \
  --manifest /ABSOLUTE/GENERATED/manifest.tsv \
  --stage train --gpus 0,1
```

`--stage` accepts `all`, `train`, `eval`, or `upload`.
Its default `all` runs training, evaluation, and upload.
Separate train and evaluation stages do not upload.
Evaluation does not train implicitly.

The runner records events in `runner_events.jsonl` beside the manifest.
It skips successful stages on rerun and retries failed stages.
`--max-upload-attempts` limits upload retries.
`--dry-run` prints commands without writing progress events.
Resume identifies jobs by manifest job ID.
Use a new event log after changing a job's training, evaluation, or publication semantics.
`--gpus` selects each worker's visible devices. The runner executes on the current host.

## Read outputs and publish

`build-data` requires every expected local summary and fails when a seed, profile, or axis value is missing.
Local CSV and figure outputs appear under `outputs/<experiment>/paper/`.
`published-plot` reads its pinned source records separately.
It does not substitute fresh results for missing historical inputs.

Canonical upload destinations are `MLL-Lab/LAGEN-models` and `MLL-Lab/LAGEN-datasets`.
Paths identify the experiment, task, horizon variant, stage, and job.
The recipe uploads shared checkpoints once.
Model uploads exclude optimizer state, recovery state, event files, and caches.

Authentication uses the Hugging Face login, a supported token environment variable, or the same variable in the repository `.env`.
The `all` stage builds local data and figures before upload.
An upload failure returns nonzero and preserves local artifacts.
Keep training, evaluation, acceptance, and publication as separate states.

`manifest.tsv` and `runner_events.jsonl` identify jobs and completed stages.
The runner generates command listings from the manifest.
See [artifacts](artifacts.md) for published experiment manifests and evidence.
