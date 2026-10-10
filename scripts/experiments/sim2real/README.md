# Held-out sim2real: one experiment per invocation

Implementation lives in this experiment directory. `study.py` owns the held-out
protocol; `runtime.py` owns runtime overlays and resource lifetime; `replay.py`
owns logical sessions and resident replay; `assets.py` and `publication.py` own
pinned inputs and experiment output delivery.
Shared deployment profiling lives in `latency_bench.profiling`.

The formal release contains 30 cases: RTX 3090, RTX 4090, RTX 5090, L40S and
A100 × QwenOFT/GR00T × Flappy/InvertedPendulum/MIKASA InterceptGrabFast.
Each case contains 25 fitting runs and 25 validation runs, with two episodes
per run. Four replay methods each evaluate the 25 validation runs.
The release contains 3,000 real and 6,000 simulated episodes.
The [fixed 30-case release](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/50d8f0aef64749a849760fc580ce3efb3c5c5e76/experiments/sim2real-calibration)
contains the case index, source manifests, episode outcomes, and paper inputs.
It also links the historical ICLR two-worker protocol, whose results are separate.

The published split seed is `20261001`. Each case preserves its calibrated
clock across measured and simulated execution. Temporal state continues across
the two episodes within a run. Each case uses one inference worker.
The fixed model revision is `180da0effdaaaeacef218cf5fbaa4db41af67495`.
The fixed profiles revision is `01d85ac858121fb2ef4dee706802e533dbd82bc9`.

| Method | NMAE over 30 cases (percentage points) |
| --- | ---: |
| Fixed | 8.9262 |
| Normal | 6.5517 |
| IID | 4.5836 |
| Temporal v13 | 3.8148 |

Each run contributes its two-episode mean. The final NMAE weights cases equally.
The published manifests preserve source reuse and rerun conditions.
These figures describe the fixed release, not a result of executing today's defaults.

The five-GPU, two-model, three-task 25/25 campaign uses
`configs/experiments/sim2real/five_gpu.yaml` and the `campaign` prepare/report entrypoint.
Pass `--config` explicitly for real acquisition and `--parallel-runs 25` for replay.
Its frozen collection order interleaves fitting and validation sessions; the profile
is built when the last designated fitting run completes, using fitting traces only.
Both real and sim accept `--defer-uploads`, keeping HF rate limits outside evaluator
lifetime. Deferred stages require a separate publication step.

The held-out recipe is `configs/experiments/sim2real.yaml`, including its shared
`sim2real/protocol.yaml` base. New experiment manifests freeze the episode count,
seed, seed stride, methods, fitting/validation runs and calibration parameters.
Replay uses the frozen manifest. Historical snapshots retain their original
schema and source revision; no missing protocol fields are inferred from the
current recipe.

`run.sh` runs one task/model on the selected inference GPU in the foreground.
Run these commands from the repository root:

```bash
bash scripts/experiments/sim2real/run.sh \
  --config configs/experiments/sim2real/five_gpu.yaml \
  --stage real --task flappy --model qwenoft --gpu 0

bash scripts/experiments/sim2real/run.sh \
  --config configs/experiments/sim2real/five_gpu.yaml \
  --stage real --task mikasa_intercept_grab_fast --model qwengr00t \
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
stores exact checkpoint/config files, evaluation provenance and training profiles.
The five-GPU campaign selects six task/model pairs from this registry.
The policy revision is `benchmark-models@180da0effdaaaeacef218cf5fbaa4db41af67495`.

| Task ID | Models | Original env/obs FPS | Maximum steps |
|---|---|---|---|
| `flappy` | `qwenoft`, `qwengr00t` | 10/10 | 3600 |
| `inverted_pendulum` | `qwenoft` / `qwengr00t` | 10/10 / 5/5 | 1000 |
| `mikasa_intercept_grab_fast` | corrected H1 `qwenoft`, `qwengr00t` | 20/20 | 60 transitions |

Flappy retains eight-decision history and `one_frame_then_noop` / `latest_ready`
semantics. InvertedPendulum retains each policy's trained action/state contract,
prompt map and original clock. MIKASA retains `latest_ready`,
`hold_last_chunk_action: true` and its corrected H1 input contract.
Runtime overlays select the actual devices and replay capacity.

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

InvertedPendulum prompt maps contain literal training-clock FPS and millisecond annotations.
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
undocumented training-time revision. The OFT runtime uses ordinary Qwen3-VL,
not the Action tokenizer variant.

## Real stage and immediate publication

The `--stage real` command stops after held-out evaluation and publication:

1. Two independent realtime calibration probes, each warmed once and collecting
   the first 500 admitted observation-to-action records. Each probe finishes the
   episode that reaches the budget, then uploads before the next stage.
2. Upload the final calibration report and frozen FPS.
3. Collect the frozen 50 run IDs in their recorded order. Fit one profile
   after the last designated fitting run completes, using the 25 fitting runs.
4. Preserve the 25 held-out runs independently of fitting and calibration.
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
its 2 episodes share that runtime. Each formal case has 100 real episodes.
The six policies are shared across the five GPU conditions. Environment seeds are
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
bash scripts/experiments/sim2real/run.sh \
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

Fixed, Normal, IID and Temporal each run 25 groups of two episodes. Group `i`
uses the frozen validation run at index `i` and latency seed `i`. Simulation
worker capacity is one per run. `--parallel-runs 25` keeps the model resident,
assigns independent runs to 25 environment slots and batches their inference.
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
streams. Preserve the recorded execution mode when reproducing a published run.

Parallel runs use logical episode IDs `20 * valid_run_id + episode_in_run` and
record both run fields in episode metadata. `parallel_runtime/config.json`
records the resident executor configuration; each run retains its logical
`generated.yaml` and raw traces.

Intervals and paired errors use 25 run means rather than 50 episodes as
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
bash scripts/experiments/sim2real/run.sh \
  --config configs/experiments/sim2real/five_gpu.yaml \
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

Run the experiment commands from the repository root. The general profiling
CLI also supports user projects outside the checkout when their modules are
importable and their config paths use the caller’s working directory.
