# Held-out Sim2Real

Measure deployment latency, fit a profile, and compare simulated returns with independent real runs.
Run commands from the repository root.
Install the selected simulator and StarVLA runtime through the [installation guide](../../../docs/installation.md).

## Measure a real experiment

```bash
conda activate starvla_rl_games_openvla
bash scripts/experiments/sim2real/run.sh \
  --stage real --task flappy --model qwenoft --gpu 0
```

The command uses the frozen inputs and protocol in `configs/experiments/sim2real.yaml`.
Use `--config` to select another registry and protocol.
GPU indices are physical `nvidia-smi` indices.

For MIKASA, select separate simulator and inference GPUs:

```bash
conda activate latency
bash scripts/experiments/sim2real/run.sh \
  --stage real --task mikasa_intercept_grab_fast --model qwengr00t \
  --gpu 1 --simulator-gpu 0 \
  --worker-python /ABSOLUTE/STARVLA/bin/python
```

The simulator owns logical GPU 0. The inference worker owns logical GPU 1.
The worker interpreter defaults to the current Python for ordinary tasks.
Both interpreters must import LAGEN and their selected dependencies.

The command prints `EXPERIMENT_ID`, `OUTPUT_DIR`, and `SOURCE_DIR`.
Outputs default to `outputs/sim2real/<task>/<model>/<instance>/<experiment-id>/`.
Use `--output-root` to select another root.

## Replay the local experiment

Supply the returned real experiment directory:

```bash
bash scripts/experiments/sim2real/run.sh \
  --stage sim --source-dir /ABSOLUTE/REAL_OUTPUT --gpu 0
```

Replay writes `sim/`, `runs.csv`, `comparison.json`, and `SIM_DONE` into that directory.
For MIKASA, also pass `--simulator-gpu` and the model interpreter.
The real stage must complete before replay.
Replay uses its frozen profile and protocol.

Use `--parallel-runs N` to run independent simulated sessions with resident model inference.
Each logical run retains its own environment seeds and latency stream.
Temporal state continues across episodes within that run.
Record the selected parallelism with comparisons.

## Read an existing published experiment

The same replay stage can download fixed published inputs:

```bash
bash scripts/experiments/sim2real/run.sh \
  --stage sim \
  --source-ref experiments/sim2real-calibration/runs/GPU/TASK/MODEL/INSTANCE/EXPERIMENT \
  --source-revision HF_COMMIT --gpu 0
```

Select the source path and revision from the experiment manifest.
The command resolves the supplied revision once.
It downloads experiment records and the separately pinned profile.
Downloaded replay outputs default to `outputs/sim2real/replays/<source-ref>/<revision>/`.
Local `--source-dir` and remote source arguments are mutually exclusive.

## Measurement and result contracts

The real stage runs independent calibration probes before freezing the clocks.
Calibration measures observation-to-action latency.
It scales environment and observation FPS together while preserving the observation stride.
Calibration samples do not enter profile fitting or held-out evaluation.

The experiment manifest records fitting run IDs, validation run IDs, collection order, episodes, seeds, and methods.
Each real run creates a new evaluator process with model warmup.
The profile uses fitting traces only.
Held-out results remain independent of fitting.
Replay preserves the model's observation, action, normalization, scheduler, and physical simulation contract.

Intervals and paired errors use run means as independent sessions.
Preserve checkpoint revisions, profile inputs, clocks, seed protocol, and execution hardware with reported results.
The [published 30-case release](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/50d8f0aef64749a849760fc580ce3efb3c5c5e76/experiments/sim2real-calibration) contains its exact historical protocol and primary evidence.

## Outputs and resume

A real experiment retains `experiment.json`, `eval_template.yaml`, calibration records, the fitted profile, run metrics, and traces.
`REAL_DONE` marks completed real acquisition.
`SIM_DONE` marks completed replay.

Resume real acquisition with the original task, model, devices, interpreter, output root, and experiment ID:

```bash
bash scripts/experiments/sim2real/run.sh \
  --stage real --task flappy --model qwenoft --gpu 0 --resume EXPERIMENT_ID
```

Repeat the same sim command to resume replay.
Completed runs remain intact. Incomplete runs restart from their beginning.
Resume rejects changed execution contracts.
A new comparison needs a separate experiment directory.

`study.py` owns the protocol. `runtime.py` owns runtime overlays and process lifetime.
`assets.py` resolves input artifacts. `replay.py` persists logical simulated runs.
