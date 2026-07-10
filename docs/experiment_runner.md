# Manifest V2 Experiment Workflow

For experiments that follow `train -> eval -> upload`, use this workflow.

## Standard flow

1. Generate configs and `manifest.tsv`
2. Run jobs with `scripts/experiment_runner/run_manifest_jobs.py`
3. Inspect progress with `scripts/manage_factor_sweeps.py` or `runner_events.jsonl`
4. Do not run `train_commands.txt` line-by-line unless explicitly requested

## Generators

Same-latency:
- `scripts/generate_sample_factory_same_latency_sweep.py`

Factor sweep / jitter / jitter robustness:
- `scripts/sweeps/generate_flappy_factor_sweeps.py`

Generated manifests must use V2 schema.

## Runner

Use:

```bash
python scripts/experiment_runner/run_manifest_jobs.py --manifest <path> --gpus 0,1
```

Useful flags:
- `--match <substring>`
- `--upload-workers <n>`
- `--max-upload-attempts <n>`
- `--dry-run`

The runner is host-local only. SSH and remote machine selection must be handled outside the runner.

## Status

Primary status files:
- `manifest.tsv`
- `runner_events.jsonl`

`train_commands.txt` and `latency_bench_eval_commands.txt` are derived artifacts, not the source of truth.

When automating experiments, prefer manifest V2 plus the runner over ad hoc shell loops.
