# Paper Experiment Workflow

Each reproducible sweep experiment has one public shell entrypoint. Run it from
the repository root.

| Experiment | Entrypoint |
| --- | --- |
| Observation stride | `bash scripts/experiments/observation_stride/run.sh <command>` |
| Memory / small model | `bash scripts/experiments/memory/small_model/run.sh <command>` |
| Jitter robustness | `bash scripts/experiments/jitter_robustness/run.sh <command>` |

The supported commands are:

```text
generate       expand source specs into configs and manifests
train          train the canonical checkpoint set
eval           evaluate every capped/extended variant
build-data     aggregate the fresh local evaluation summaries
plot           plot an explicit local data snapshot
upload         upload completed model and evaluation artifacts
all            run generate -> train -> eval -> build-data -> plot -> upload
published-plot rebuild the pinned published snapshot from historical sources
```

For example:

```bash
bash scripts/experiments/observation_stride/run.sh all --gpus 0,1
```

`all` always uses the complete seed set. Use `generate --seeds <csv>` and
`train`/`eval --match <substring>` only for partial development runs; a partial
result set is not accepted by `build-data`.

## Training and evaluation manifests

Observation stride and memory share checkpoints between their capped and
extended variants. Their `train` command runs the Flappy Bird and Demon Attack
extended manifests plus the single Deadly Corridor manifest. Their `eval`
command runs all five manifests, so capped evaluation never schedules duplicate
training.

Jitter robustness uses four source specs under
`configs/experiments/jitter_robustness/`: Flappy Bird, Demon Attack capped,
Demon Attack extended, and Deadly Corridor. Each environment spec contains all
five single-profile and two mixed-profile training settings. Demon Attack is
trained from the capped manifest once and evaluated at both 3,600 and 36,000
steps. Deadly Corridor uses the canonical 35/8.75 clock pair and a 30,000-step
horizon.

## Internal manifest runner

The public scripts delegate one manifest at a time to:

```bash
python scripts/experiment_runner/run_manifest_jobs.py \
  --manifest <generated-root>/manifest.tsv \
  --stage train \
  --gpus 0,1
```

`--stage` accepts `all`, `train`, `eval`, or `upload`; the default `all`
preserves the original train/eval/upload behavior. Stage-specific train and
eval invocations do not upload automatically. Eval never trains implicitly, so
a capped manifest can use a checkpoint produced by its extended counterpart.

Progress is append-only in the manifest's sibling `runner_events.jsonl`.
Successful stages are skipped on rerun, failed train/eval stages are retried at
that stage, and upload retries use `--max-upload-attempts`. `--dry-run` prints
the selected commands without writing progress events. Resume state is keyed by
job ID; remove the corresponding event log after changing a generated job's
training, evaluation, or upload semantics.

The runner is host-local. `--gpus` sets `CUDA_VISIBLE_DEVICES` for each worker;
SSH and remote host selection remain outside the runner.

## Uploads

The canonical specs upload to:

```text
latency-sensitive-bench/paper-experiment-models
latency-sensitive-bench/paper-experiment-evals
```

Remote paths are namespaced by paper experiment, environment, horizon variant,
stage, and manifest job ID. Capped manifests that reuse an extended checkpoint
have no model upload target, preventing duplicate model uploads. W&B data,
training state directories, optimizer directories, event files, and caches are
excluded from model uploads.

Authentication uses the Hugging Face cached login, a supported HF token
environment variable, or the same token variable in the repository `.env`.
Evaluation artifacts are uploaded before the larger model artifacts. `all`
builds the local data and figures before uploading; an upload failure is printed,
recorded in the event log, and returns nonzero without deleting local artifacts.

## Local and published figures

`build-data` reads every expected local summary JSON and fails on a missing
seed, profile, or axis value. It does not fall back to W&B. Local CSV and figure
outputs live under `outputs/<experiment>/paper/`, outside the committed paper
snapshot.

`published-plot` is a separate strict path for pinned W&B histories and explicit
historical JSON roots. Jitter robustness additionally verifies the historical
JSON hashes. It never substitutes fresh local results for missing published
provenance.

## Internal tools

- `scripts/sweeps/generate_sweep_configs.py` expands one source spec.
- `scripts/experiment_runner/run_manifest_jobs.py` executes one manifest.
- `train_commands.txt` and `latency_bench_eval_commands.txt` are derived
  artifacts; `manifest.tsv` and `runner_events.jsonl` are the source of truth.
