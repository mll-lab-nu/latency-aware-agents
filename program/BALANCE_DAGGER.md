# Balance QwenOFT DAgger handoff

## Goal and current priority

The Balance DAgger label fix and the corrective run approved on 2026-09-09 are
complete, including all 500 updates and 100 zero-latency eval episodes. The
corrected model recovers to 68.02 mean return but remains below the original
formal QwenOFT checkpoint. Preserve the original as the better model.

User decisions that must be preserved:

- Train with two GPUs, per-device micro-batch 32, gradient accumulation 2, and
  effective global batch 128.
- Keep all generated datasets and models local (`--no-publish`). W&B training
  logging is expected, but Hugging Face writes are not authorized.
- Prefer direct action over speculative infrastructure. Do not add defensive
  checks, hidden fallbacks, or abstractions.

## Original incorrect-label run (preserved)

The run completed successfully at 500 optimizer updates and then completed a
100-episode zero-latency evaluation.

- Run manifest and exact provenance:
  `outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-r1-20260908/run.json`
- Final model:
  `outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-r1-20260908/inference/checkpoints/model.pt`
- Training log:
  `outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-r1-20260908/dagger.log`
- W&B run ID: `balance-qwenoft-dagger-r1-20260908-mb32-dagger`
- Final held-out action loss: `0.1185289919`; final train loss: `0.0745380297`.
- Final rollout result: mean return `22.8633 ± 3.0377`, mean survival length
  `27.44` over 100 episodes. See
  `outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-r1-20260908/inference/eval_l0.json`.

This is a real regression, not an eval-config mismatch. The exact initial
checkpoint scored mean return `107.2190 ± 25.8240`, mean length `118.08` over
50 episodes. Its artifacts are under
`outputs/starvla/eval_backfill/results/balance/qwenoft/`. A diff of the two
`eval_config.yaml` files changes only checkpoint/output paths and episode count;
the environment, seed, zero-latency mode, observation contract, and execution
settings match.

Nothing from this DAgger run was uploaded to Hugging Face.

## Data and preserved checkpoints

- DAgger round 0 contains 64,000 learner-visited rows with counterfactual
  FastTD3 labels:
  `outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-r1-20260908/dagger/round-00/`
- Training used a 50/50 bootstrap/DAgger mixture. The paths and revisions are in
  `run.json` above.
- A prematurely started micro-batch-1 run was stopped at step 187 and preserved
  at
  `outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-r1-20260908/training/dagger-microbatch1-aborted-step187/`.
  It has both the model and full optimizer state and is useful as an intermediate
  checkpoint for regression localization.
- A one-update micro-batch-8 probe, started from the initial checkpoint, is under
  `outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-r1-20260908/batch-probe/probe-mb8/`.
- The final step-500 training checkpoint and full state remain under
  `outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-r1-20260908/training/dagger/checkpoints/`.

## Local implementation state

Relevant work is uncommitted. Inspect the diff rather than reconstructing it:

- `scripts/starvla/train_task.py` adds Balance DAgger orchestration,
  `--dagger-from`, `--no-publish`, `--eval-episodes`, local publication gating,
  and completed-checkpoint resume behavior.
- `scripts/starvla/balance_dagger.py` is the new Balance collector. It executes
  learner actions while recording FastTD3 labels for the same pre-action state.
  This file is not in the base Git index yet.
- `latency_bench/data/lerobot_export.py` fixes generated modality
  metadata from `annotation.human.action.task_description` to the loader-required
  `annotation.human.task_description`.
- `tests/scripts/test_starvla_train_task.py` covers the direct/no-publish path.
- The completed-resume fix moves `training_config(...)` outside the
  `dagger_step < stop_step` branch so a finished checkpoint can proceed through
  inference bundling, eval, and final manifest writing.

Verification already run:

- `tests/scripts/test_starvla_train_task.py`: 9 passed.
- The earlier targeted exporter/direct-DAgger set: 38 passed after the modality
  key fix.
- The real completed-checkpoint resume path built the inference bundle, evaluated
  100 episodes, wrote final provenance, and exited successfully.

The repository has many unrelated user changes. Do not reset, clean, stage, or
rewrite files outside the relevant diff.

## Confirmed root cause and corrective run

The cached bootstrap labels are command targets **after** velocity/acceleration
conditioning. The old DAgger collector recorded absolute teacher requests
**before** conditioning. Mixing those two target meanings 50/50 was incorrect;
the environment's limiter itself was never disabled.

On all 85,958 adjacent bootstrap training pairs, the stored action matches the
next state's decoded command target to maximum error `3.61e-7`. A separate
12-row teacher replay matched the stored labels after conditioning to maximum
error `2.38e-6`, versus mean absolute error `0.59544` for unconditioned requests.

The fix shares one pure label helper between live collection and offline
relabeling. It decodes previous command position/velocity from pre-action state
`[112:198]`, maps the request to MuJoCo joint order, applies the existing
environment limiter, and maps back to policy order. It does not step the teacher
or change the student's environment state.

The approved rerun is `dagger-conditioned-r1-20260909` under the same QwenOFT run
root. Relabel the original 64,000 rows, preserving images, states, seeds, episode
ordering, and the original bootstrap normalization/validation split. Start from
the original 107-return model with a fresh optimizer, not the failed model.
Keep 500 updates, 50/50 sampling, micro-batch 32 on two GPUs, accumulation 2,
base LR `3e-5`, head LR `1e-4`, and the existing warmup/schedule. Evaluate 100
episodes at zero latency; also report the first 50 against the original run.

W&B run: [balance-qwenoft-dagger-conditioned-r1-20260909-dagger](https://wandb.ai/saberrr-zju/starvla_tasks/runs/balance-qwenoft-dagger-conditioned-r1-20260909-dagger).
The run completed all 500 updates and all 100 zero-latency evaluation episodes.
Final train loss is `0.0227174610`; held-out action loss is `0.0213983804`
(old incorrect-label run: `0.1185289919`).

Corrected model, all 100 episodes: return `68.0205 ± 3.0410`, mean survival
`79.73` steps. No invalid actions were recorded. For the same first 50 episode
seeds (0 through 49):

| Model | Mean return | Return std | Mean survival steps |
| --- | ---: | ---: | ---: |
| Original formal checkpoint | 107.2190 | 25.8240 | 118.08 |
| Incorrect-label DAgger | 22.9287 | 3.5819 | 27.82 |
| Corrected-label DAgger | 68.3031 | 3.1028 | 79.72 |

The corrected labels substantially reduce the collapse, but this 500-update,
50/50 DAgger recipe still regresses against the original checkpoint. Do not
promote it as an improvement. This run does not isolate which remaining training
or distribution choice causes that residual regression.

Final artifacts under
`outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-conditioned-r1-20260909/`:

- `inference/checkpoints/model.pt`: completed step-500 model.
- `inference/eval_l0.json`: summary and all 100 episode metrics.
- `evals/final/`: evaluation config, logs, actions, and step records.
- `run.json` and `inference/provenance.json`: final training/data provenance.

The evaluation matches the incorrect-label run except model/output paths;
the Python executable path resolves to the same interpreter. Both the original
and incorrect-label checkpoints/data remain intact. `--no-publish` was retained,
and the corrected dataset/model were not uploaded to Hugging Face.

Corrective-run verification:

- All 64,000 raw rows were relabeled. The mean absolute label change is
  `0.6183085`; every non-action column, the schema, and metadata are preserved.
- Prepared LeRobot state/action arrays equal the corrected raw arrays exactly.
  There are 64,000 training frames in 555 episodes and no new validation rows.
- The task contract and `meta/stats_gr00t.json` equal bootstrap exactly.
  Trainer, framework, and checkpoint configuration match the failed run.
- The shared helper's real 12-episode teacher probe has maximum label error
  `1.024e-6`, without changing environment command state. An extended probe
  found one single-joint discrepancy on a terminal bootstrap row; all 980
  nonterminal transitions in that episode match next-state commands within
  `1.81e-7`. Offline DAgger repair uses saved requests and does not rerun that
  teacher-label reconstruction.
- Collector/orchestration tests: 17 passed. The broader data selection reports
  28 passed and 1 skipped. One existing exporter test was excluded after failing
  because its unknown-layout fixture lacks `action_labels`; the relevant
  exporter validation is unchanged by this fix.
- Environment limiter selection: 1 passed, 2 skipped (environment-dependent).

Do not blindly invoke the existing run with `--resume` expecting a new training
run: it sees the complete step-500 state and skips training. Use a distinct run
directory/ID, or preserve and explicitly replace the expected training output,
for any new experiment.

## Ten-round DAgger: latest recorded result and pending verification

Checked on 2026-09-14 against the [full historical record](archive/DAGGER.md#11-2026-09-10持续十轮-dagger-对照).
The last recorded update is 2026-09-11 09:20 UTC: both scratch and checkpoint
arms completed ten rounds / 5,000 updates and all per-round 50-episode L0
evaluations. Twenty round bundles were reported returned and checked.

| Model / evaluation | Episodes | Mean return | Return SD | Mean survival steps |
| --- | ---: | ---: | ---: | ---: |
| Original model, common round-evaluation seeds | 50 | 102.924974 | 18.104805 | 113.22 |
| Scratch, final step 5,000 | 50 | 144.532006 | Not recorded in the handoff | 157.30 |
| Checkpoint, final step 5,000 | 50 | 377.179249 | Not recorded in the handoff | 397.54 |
| Scratch, independent endpoint confirmation | 100 | 143.576013 | Not recorded in the handoff | Not recorded in the handoff |

These are documented experiment results, separate from the central H1 baseline.
The documented local root `outputs/starvla/experiments/balance-dagger10-20260910/`
is absent from this checkout at this audit; the remote hosts and live process
state have not been rechecked. Do not present the old running messages as a
current running job, or substitute the step-5,000 round evaluation for endpoint
acceptance.

Pending work carried out of the archive:

- Locate the final independent 100-episode checkpoint-arm and original-model
  confirmations using the common endpoint seeds; retain the scratch confirmation.
- Confirm that the final two full training states and confirmations were returned.
- Run the existing final verification on those artifacts, then produce the
  planned comparison / REPORT with paired episode uncertainty.
- Record the verified endpoint results and artifact locations in [PROGRESS](PROGRESS.md).

The experiment protocol, per-round table, W&B links, and recovery details remain
in the archived full record. This handoff does not restart the experiment.

## Suggested skills

- `ponytail:ponytail` at full level for the diagnosis and smallest corrective
  experiment.
- `occam-review:occam-review` before staging or declaring the implementation
  finished, because the orchestration diff has grown across collection,
  training, resume, evaluation, and publication paths.
