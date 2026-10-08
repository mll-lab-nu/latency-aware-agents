# Held-out sim2real: one experiment per invocation

Implementation lives in this experiment directory. `study.py` owns the held-out
protocol; `runtime.py` owns runtime overlays and resource lifetime; `replay.py`
owns logical sessions and resident replay; `assets.py` and `publication.py` own
pinned inputs and experiment output delivery. The budget entrypoints and reports
are documented in [the budget protocol](../../../program/SIM2REAL_BUDGET.md).
Shared deployment profiling lives in `latency_bench.profiling`.

The held-out recipe is `configs/experiments/sim2real.yaml`, including its shared
`sim2real/protocol.yaml` base. New experiment manifests freeze the episode count,
seed, seed stride, methods, fitting/validation runs and calibration parameters.
Replay uses the frozen manifest. Historical snapshots retain their original
schema and source revision; no missing protocol fields are inferred from the
current recipe.

`run.sh` runs one task/model on one explicitly selected inference GPU, in the
foreground. It has no job queue, GPU allocator, background process manager,
monitor or retry loop. The calling agent owns machine setup, concurrent jobs,
tmux, monitoring, result transfer and decisions to resume a failed experiment.

From a deployed checkout at `/workspace/latency-sensitive-bench`:

```bash
bash /workspace/latency-sensitive-bench/scripts/experiments/sim2real/run.sh \
  --stage real --task flappy --model qwenoft --gpu 0

bash /workspace/latency-sensitive-bench/scripts/experiments/sim2real/run.sh \
  --stage real --task mikasa_intercept_grab_fast --model qwenpi_v3 \
  --gpu 1 --simulator-gpu 0 --worker-python /venv/sim2real/bin/python
```

GPU numbers are physical `nvidia-smi` indices. Ordinary experiments expose only
their inference GPU. InterceptGrab exposes simulator then inference GPU as
logical `cuda:0` and `cuda:1`; it still has exactly one inference worker.
Ant/InvertedPendulum use EGL rendering on the selected GPU. A caller may run
two ordinary experiments concurrently on different cards, or reserve both
cards for one InterceptGrab experiment.

Run ordinary tasks in the StarVLA model environment. Run InterceptGrab in its
MIKASA/ManiSkill environment and pass the model interpreter with `--worker-python`.
The worker interpreter defaults to the current Python for ordinary tasks. Both
environments must import this repository. Asset downloads use the current Hugging Face
credentials; the pinned profile datasets must be readable. The script does
not install environments or authenticate accounts.

## Frozen assets and task contracts

[`configs/experiments/sim2real.yaml`](../../../configs/experiments/sim2real.yaml)
lists all 18 task/model pairs, exact checkpoint/config files, evaluation archive
provenance, and training profile coordinates. Model repository
`MLL-Lab/LAGEN-models` is pinned to
`180da0effdaaaeacef218cf5fbaa4db41af67495`.

| Task ID | Models | Original env/obs FPS | Archived maximum steps |
|---|---|---|---|
| `flappy` | `qwenoft`, `qwenpi_v3`, `qwengr00t` | 10/10 | 3600 |
| `demon_attack` | same three | 60/15 | 7200 |
| `asterix` | same three | 60/15 | 7200 |
| `mikasa_intercept_grab_fast` | same three, corrected H1 `evalfix_20260923` | 20/20 | 60 transitions |
| `ant` | `qwenoft` | 10/10 | **3600** |
| `ant` | `qwenpi_v3`, `qwengr00t` | 5/5 | **1000** |
| `inverted_pendulum` | `qwenoft` | 10/10 | 1000 |
| `inverted_pendulum` | `qwenpi_v3`, `qwengr00t` | 5/5 | 1000 |

The local comparison below covers selected environment, scheduler, evaluation
and executor fields across the three archived model YAMLs for each task. It
records only values compared in those files. `experiment.name` and
model/checkpoint/profile paths remain model-specific archive or registry inputs;
no equality is claimed for fields omitted from this table.

| Task | Shared archived fields across model YAMLs | Archived model-specific fields or exceptions |
|---|---|---|
| `flappy` | 10/10 FPS, 3,600 steps, 100 episodes, 32 parallel envs, 8-decision history, `one_frame_then_noop` / `latest_ready` | Executor device/batch fields differ: OFT and PI use `cuda:0,cuda:1` with batch 16; GR00T uses `cuda:0` with batch 32. Runtime overlay chooses actual devices and replay capacity. |
| `demon_attack` | 60/15 FPS, 7,200 steps, 100 episodes, 32 parallel envs, 8-decision history, `hold` / `latest_ready`, two inference devices and batch 16 | The three YAMLs share these selected runtime fields; the registry keeps each model's checkpoint, training profile and source provenance. |
| `asterix` | 60/15 FPS, 7,200 steps, 100 episodes, 32 parallel envs, 8-decision history, `hold` / `issue_order_fifo`, two inference devices and batch 16 | Checkpoint/model-config/manifest/output paths and archived `latency.profile_path` differ by model. The study selects its measured profile when constructing replay config. |
| `ant` | Shared action/state contract, 20 episodes and one parallel env | OFT uses 10/10 FPS and 3,600 steps; PI and GR00T use 5/5 FPS and 1,000 steps. Checkpoint, model config, task manifest and prompt-map paths differ by model. |
| `inverted_pendulum` | Shared action/state contract, 20 episodes, one parallel env and 1,000 steps | OFT uses 10/10 FPS; PI and GR00T use 5/5 FPS. Checkpoint, model config, task manifest and prompt-map paths differ by model. |
| `mikasa_intercept_grab_fast` | 20/20 FPS, 60 transitions, 200 archived episodes, 10 parallel envs, `latest_ready`, `hold_last_chunk_action: true`, batch 8 on `cuda:1` | Checkpoint/model-config/task-contract/worker/profile paths differ by model. H1 archives omit action-prefix conditioning; asset preparation sets `action_prefix.mode: none` when resolving their task contract. |

Eval YAMLs under `configs/experiments/sim2real/eval/` are verbatim archived
snapshots, including original host-local paths. Asset resolution relocates the
explicitly listed model/manifest/contract/prompt-map files and backbone.
Runtime configuration changes devices, mode, clocks, episode count/seeds and
logging output. Environment physics, frame skip, observation stride, model
inputs/history, action mapping, prompts, normalization, scheduler and maximum
steps retain their archived values. InterceptGrab preserves
`hold_last_chunk_action: true`.
The archived InterceptGrab H1 policies predate action-prefix conditioning;
asset resolution explicitly selects their original `action_prefix.mode: none`
for the current task-contract runtime. Archived YAML snapshots remain verbatim.

The registry selects the exact source archive and model/training-profile
coordinates. `assets.prepare_model_assets` relocates those assets and
records provenance; `portable_eval_config` prepares the portable published
template; `study.calibrate` freezes the runtime clock; and `runtime.run_config`
applies per-run devices, latency method/seed, episode count, run-seed fields and
logging. The evaluator assigns episode IDs under the existing run protocol;
`run_real` and `run_sim` own stage completion, resume and publication. The table
describes archive values; it does not replace these runtime decisions.

Ant/IP prompt maps contain literal training-clock FPS and millisecond annotations.
Those strings and prompt keys remain exactly as trained while wall-clock pacing
is calibrated; they are not rewritten as target-GPU timing claims.

The active InvertedPendulum registry uses the migrated evaluation archives and
training profiles in the main repositories. Its original Standard-Pipeline
coordinates remain in frozen provenance. MIKASA evaluation snapshots retain
the corrected contract identified by each published model bundle.

The study pins ordinary `Qwen/Qwen3-VL-4B-Instruct` backbone assets to
`ebb281ec70b05090aa6165b016eac8ec08e71b17`. This revision is recorded by the newer
bundles; for historical OFT archives recording only a local backbone path, it
is this study's initialization/processor asset revision, not a claim about an
undocumented training-time revision. Ant OFT's runtime uses ordinary Qwen3-VL,
not the Action tokenizer variant.

## Real stage and immediate publication

The `--stage real` command stops after held-out evaluation and publication:

1. Two independent realtime calibration probes, each warmed once and collecting
   the first 500 admitted observation-to-action records. Each probe finishes the
   episode that reaches the budget, then uploads before the next stage.
2. Upload the final calibration report and frozen FPS.
3. Fit runs **0–4**, 2 episodes each. Upload every run immediately before
   starting the next run. Fit and upload one profile using only these runs.
4. Held-out runs **5–9**, 2 episodes each, again uploading after every run.
5. Write/upload `real_runs.csv`, `real_summary.json`, and `REAL_DONE`.

Experiment manifests and episode results go to dataset
`MLL-Lab/LAGEN-datasets`, under:

```text
experiments/sim2real-calibration/runs/<gpu-class>/<task>/<model>/<instance>/<experiment-id>/
```

Calibration, fitted profiles, action/step traces and runtime logs go to
`MLL-Lab/LAGEN-profiles`, under the same experiment prefix.
Each publication commits measurement files first. It then commits experiment
files, a fixed `profile_source.json` and the stage marker to benchmark-datasets.
The dataset marker establishes successful stage publication. If the dataset
commit fails, resume reuses the completed measurement commit.
Raw hardware and resume contracts stay local. Published manifests/configs use
portable model references and public GPU descriptions; evaluator-generated local
path configs are omitted from directory uploads.
The local `uploads/` receipts retain the commit IDs. Upload failures propagate to
the caller; a subsequent explicit resume first publishes completed local runs
instead of re-evaluating them. A lost local receipt is recovered from the remote
publication marker. No automatic retry or scheduling is performed by this code.

Every real run is a fresh evaluator process with a fresh model/worker and warmup;
its 2 episodes share that runtime. Each model has 20 real episodes; all 18 models
have 360 real episodes, in addition to calibration and warmup. Environment seeds are
`2000000 + 20 * run_id + episode_id`, preserving the original run seed blocks
while evaluating only episode IDs 0 and 1. Calibration uses separate 500-seed blocks
below 2000000. Its samples are excluded from profile fitting and validation.

Let `F_train` and `O_train` be the archived training env/obs FPS, and `L_train`
the exact training profile's mean milliseconds. Calibration targets
`d = L_train * F_train / 1000` frames. Each probe measures complete
observation-to-action latency through the realtime executor. Given probe mean
`L`, the next clock is `F = 1000*d/L`, `O = F/(F_train/O_train)`.
The rates scale together; physical simulation steps and observation/action
stride remain unchanged. The profile acquisition FPS is not substituted for
training FPS. There is no worker-utilization FPS cap. After the second probe,
clocks are frozen; neither fit nor held-out data changes them.

## Replay on another machine

The real stage prints `SOURCE_REF` and `SOURCE_REVISION`. Supply both explicitly; any supplied ref name is resolved once to its immutable commit before replay:

```bash
bash /workspace/latency-sensitive-bench/scripts/experiments/sim2real/run.sh \
  --stage sim \
  --source-ref experiments/sim2real-calibration/runs/GPU/TASK/MODEL/INSTANCE/EXPERIMENT \
  --source-revision HF_COMMIT --gpu 0
```

For InterceptGrab, also specify `--simulator-gpu` and the model worker interpreter.
The replay downloads required experiment inputs at the supplied dataset revision.
It downloads calibration and profile files at the revision in `profile_source.json`.
It requires `REAL_DONE` and uses the uploaded `eval_template.yaml` and exact model coordinates.
It replaces host-local paths/devices, preserving the measured clock and task
contract. It does not recalibrate or select a current/latest profile. Target
hardware is retained in `experiment.json`; execution hardware is recorded
separately in local `simulation_hardware.json` and published `simulation_hardware_public.json`.

Fixed, Normal, IID and Temporal each run five groups of 2 episodes. Group `i`
uses held-out run `5+i`'s environment seeds and latency seed `i`. Simulation
worker capacity is one per run. `--parallel-runs 5` keeps the model resident,
assigns independent runs to five environment slots and batches their inference.
The default remains one run at a time. Each simulation run is uploaded to the
source experiment's `sim/` directory. `--defer-uploads` instead persists a stage
manifest under `pending_uploads/`; the external batch publisher commits completed
stages together and creates `uploads/` receipts only after HF confirms success.
This lets simulation continue during a repository commit cooldown. Computation
completion and publication completion remain separate. Final outputs are
`runs.csv`, `comparison.json`, and `SIM_DONE`.

Parallel replay preserves completed run directories and publishes each missing
run to its original `sim/METHOD/INDEX` path. Environment seeds and latency
streams belong to logical runs; Temporal shares its sampler across the two
episodes as before. PI v3 and GR00T parallel inference use explicit per-episode,
per-frame action-noise seeds. Existing completed runs retain their original
global-RNG behavior; `simulation_parallel.json` records the transition. Batched
GPU arithmetic can change action trajectories despite identical logical random
streams. The RTX PRO comparison and accepted differences are recorded in the
[continuation log](../../../program/logs/sim2real-rtxpro-continuation-20260930.md).

Parallel runs use logical episode IDs `20 * valid_run_id + episode_in_run` and
record both run fields in episode metadata. `parallel_runtime/config.json`
records the resident executor configuration; each run retains its logical
`generated.yaml` and raw traces.

Intervals and paired errors use five run means rather than 10 episodes as
independent machine sessions. They are conditional on the fitted profile.
Task-specific metrics, including InterceptGrab success, also receive run means,
intervals and paired comparisons. Cross-task normalization remains separate.

## Local outputs and resume

Real outputs default to
`results/sim2real/<task>/<model>/<gpu-instance>/<experiment-id>/`.
Replay outputs default to
`results/sim2real/replays/<source-ref>/<source-revision>/`.
`--output-root` can relocate either tree.

Each run directory contains `generated.yaml`, raw traces, episode metrics and a
local `DONE` marker. Serial evaluators additionally write `stdout.log`; parallel
execution uses the caller's case log and `parallel_runtime/config.json`. `REAL_DONE` and `SIM_DONE` identify
completed computation stages; their `uploads/*_complete.json` receipts identify
successful publication. `experiment.json` and `eval_template.yaml` are the
portable source contract. Hardware/config/log records remain available for audit.

Resume a real experiment with the same task, model, GPUs, interpreters and storage:

```bash
bash /workspace/latency-sensitive-bench/scripts/experiments/sim2real/run.sh \
  --stage real --task flappy --model qwenoft --gpu 0 --resume EXPERIMENT_ID
```

Repeat the same pinned sim command with its original local replay directory to
resume it. Its execution contract is checked before reusing results or hardware
metadata. A different GPU/interpreter/config is rejected, as is a fresh directory
when that experiment already has a published simulation. Completed runs and the
frozen profile are retained; incomplete runs restart at their beginning. Upload
completion is independent of computation completion, so failed uploads do not
consume another GPU run.

## Working directory

Run direct Python commands from the repository root. `scripts/profile.sh` sets
that working directory before invoking the profiling CLI.
