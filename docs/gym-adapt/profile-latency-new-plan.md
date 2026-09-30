# Profile-latency-new execution plan

Updated: 2026-09-11. This is a reusable runbook for an environment assigned by the user. Writing this plan does not start an experiment or publish artifacts.

## 1. Assignment and scope

The execution request specifies the target environment and its required `env_fps`. Run only that environment; do not enumerate or schedule every checkpoint in the source repository. A user-specified slowdown is part of the experiment contract.

Before execution, record these inputs in the run ledger:

| Input | Source |
| --- | --- |
| `TASK` | Environment named in the execution request |
| `ENV_FPS` | User-specified environment rate; otherwise the selected baseline's rate |
| `OBS_FPS` | Explicit user value if supplied; otherwise preserve the baseline observation stride as described below |
| `ZERO_BUNDLE` | Matching checkpoint under `<task>/zero_latency/OpenVLA/` in `latency-sensitive-bench/extra-envs-checkpoints` |
| `RUN_ID` | Unique UTC run identifier |
| Baseline recipe | Actual source VLA config, teacher config and rollout metadata |

Resolve the exact checkpoint and Hub revision for the requested environment. Keep its model implementation and state encoding: the repository directory name `OpenVLA` is not a reason to replace a QwenOFT checkpoint with another implementation. Do not select an arbitrary checkpoint if the assignment leaves materially different model variants unresolved.

## 2. Servers and storage

The user has confirmed that both SSH hosts are reachable and each has one GPU:

| Host | Role | GPU policy |
| --- | --- | --- |
| `vast-3090` | Profiling and all latency-bench evaluations | One job at a time; exclusive GPU access during measured latency runs |
| `nebius-H100` | Small-model training, teacher rollout, conversion and VLA training | One GPU job at a time; overlap CPU/IO work when resources permit |

The earlier H100 connection failure is superseded by this confirmation. At execution time, inspect current processes, GPU use, free disk space and available CPU/RAM; do not assume either server is idle merely because SSH works.

Previously located 3090 paths are `/workspace/latency-sensitive-bench`, `/venv/starvla_rl_games_openvla/bin/python`, and `/mnt/results/latency-sensitive-bench`. Confirm these paths and locate the H100 checkout/interpreter before launching. Record actual root and submodule revisions on both hosts. Preserve unrelated changes; use an isolated checkout when needed instead of resetting the shared checkout.

Use `LATENCY_BENCH_ENV_BACKEND_START_METHOD=spawn`. For the 3090 MuJoCo path, the previously working rendering setup is `MUJOCO_GL=osmesa`; keep the renderer identical across profiling and both VLA evaluations. The prior `exps` run encountered an OSMesa/model-initialization fork deadlock, so confirm the first episode completes before leaving a job unattended.

### Keep the 3090 working set small

Storage management is part of execution, not an end-of-project cleanup task.

- Before every large download or transfer, inspect the destination filesystem with `df -h` and inspect relevant model/cache directories with `du -sh`. Account for the incoming bundle, temporary download/staging files and expected evaluation logs.
- Download only the assigned environment's exact checkpoint and required inference assets. Do not mirror the whole checkpoint repository, training dataset, optimizer states or intermediate VLA checkpoints onto the 3090.
- Reuse one local L0 bundle for P and B. After both stages finish and their evidence is saved, remove that local bundle if its exact revision remains available on Hub. Do this before receiving the new VLA bundle when possible.
- After E finishes, remove its local small-model copy once the checkpoint/config are verified on H100 or Hub and no active job references them. H100 retains the copy needed for rollout.
- After F and result synchronization finish, remove the 3090 copy of the new VLA once the required Hub upload is verified. Keep small configs, metrics, hashes, logs and provenance.
- Clean obsolete, reproducible checkpoint copies and task-owned temporary downloads promptly. Inspect symlinks/shared Hub blobs and use the installed cache tooling for shared caches; do not delete files needed by another run or active process.
- Never delete the sole copy of an unpublished checkpoint, unsynchronized evaluation evidence or an ambiguous unrelated artifact. Record the exact paths removed and their durable source revisions in the ledger. Moving a file elsewhere on the same filesystem does not reclaim space.

Keep large rollout/conversion data on H100 and upload from there. Transfer inference bundles directly between servers when available, or download the exact Hub revision; avoid routing large artifacts through the local workstation.

## 3. Experiment contract and user-specified FPS

Create `baseline_contract.json` for this run from the selected VLA's `config.full.yaml`, manifest, normalization statistics, teacher `config.json`/descriptor and rollout metadata. Save resolved YAMLs and actual commands alongside it.

Keep the zero-latency recipe unchanged except for latency and the user's requested clock settings. Preserve training seed, frame budget, optimizer, workers, batch size, rollout length, model structure, frame stack, environment kwargs, action/state order, scheduler, initialization and checkpoint-selection rule. Preserve the original rollout quality gate, episode budget, split and deterministic/stochastic policy settings.

### Apply the requested clock consistently

1. Record the source `env_fps`, source `obs_fps` and observation stride `source_env_fps / source_obs_fps`.
2. Set `env_fps = ENV_FPS` in P, B, S, E, R and F, and in the corresponding conversion/VLA timing metadata. If the user supplies `obs_fps`, use it; otherwise set `obs_fps = ENV_FPS / source_observation_stride` so the number of raw ticks between observations remains unchanged.
3. Preserve environment dynamics, simulator timestep/frame skip, frame stack and raw-frame budgets. A slower execution clock must not silently change the underlying physics or expand the training frame budget. Change those only when explicitly requested.
4. Use the resulting raw-frame clock for the existing milliseconds-to-frame latency conversion. Do not manually rescale latency samples or reuse a profile collected at a different observation cadence.
5. Keep B and F identical in requested FPS, observation cadence, seeds, horizon and renderer. Label the comparison with those rates; do not mix it with an older native-speed evaluation.

The original zero-latency checkpoint remains the source model even when evaluated at a slower requested clock. New teacher training uses the requested clock and new latency distribution. Record both differences from the original training run explicitly.

Do not inherit an old `*_profile_10fps.yaml` wholesale. Its rate is appropriate only if it matches the new assignment, and its other fields still need to match the selected baseline.

### Bind the complete inference bundle

Read VLA image size from `rl_games.env_eval.image_size` and verify exported images, training config, resolved eval config and actual warmup inputs agree. The repaired runs expect 224 by 224 VLA images; this does not change an Atari teacher's separate 84 by 84 observation contract.

Bind checkpoint, model config, manifest, normalization statistics and prompt map/key from the same bundle. Changing only the weight path can leave inherited metadata from another model. Preserve continuous-projector versus text-bin state encoding.

Existing examples contain task-specific traps: an AirRaid L0 eval config inherits profile metadata, an InvertedPendulum generic teacher YAML is fixed-L1, and the AirRaid export shell's return gate differs from the source directory name. When relevant to the assigned environment, resolve these from actual source metadata rather than assuming a filename is authoritative.

## 4. Dependencies and fastest single-environment schedule

```text
vast-3090:   P: profile ----> B: L0 VLA realtime
                |                (overlaps S on H100)
                v
nebius-H100: S: small-model training ----> upload small model
                |
                v
vast-3090:   E: simulated profile-latency small-model eval
                |
                v
nebius-H100: R: gated rollout -> C: conversion -> V: VLA training
                                    |                 |
                              dataset upload     final bundle upload/transfer
                                                      |
                                                      v
vast-3090:                                      F: new VLA realtime
```

Start H100 environment/config/model preparation while 3090 runs P. As soon as P is complete and the immutable profile files are verified, start S on H100 and B on 3090 in parallel. Profile upload does not need to delay S.

When S finishes, transfer its small inference bundle to 3090. Finish the active B job before starting E; do not run two GPU workloads concurrently to shorten this handoff. E provides evidence for the rollout gate. Once E is accepted, run R then C on H100. Start V as soon as conversion is complete, while uploading the dataset in parallel if CPU/RAM/IO permit.

After V completes, transfer the final inference bundle for F while fulfilling the required Hub upload. A verified direct transfer allows F to start without waiting for upload followed by download. All required uploads must still finish before declaring the pipeline complete.

Avoid downloads, compression and heavy CPU work on 3090 during P/B/F: observation-to-action latency includes host-side work. H100 conversion/metadata/upload work may overlap GPU work only when it does not starve the training job. Do not alter the baseline batch/workers or interrupt a healthy VLA training run to make room for another job.

One execution agent can coordinate both machines. If delegation is used, assign one owner per server and a coordinator for the ledger/reports; keep a single GPU queue per host. Check for other assigned tasks before reserving a server, rather than launching competing processes.

## 5. Artifact destinations

Use the requested new prefixes and retain old results:

| Artifact | Repository | Path in repository |
| --- | --- | --- |
| Profile traces and fitted profile | Dataset: `latency-sensitive-bench/Standard-Pipeline` | `<task>/profile_latency/profiling/` |
| Small-model bundle | Model: `latency-sensitive-bench/extra-envs-checkpoints` | `<task>/profile_latency/small_model/` |
| Raw demonstrations | Dataset: `latency-sensitive-bench/Standard-Pipeline` | `<task>/profile_latency/demonstrations/raw/` |
| Final VLA inference bundle | Model: `latency-sensitive-bench/extra-envs-checkpoints` | `<task>/profile_latency/OpenVLA/` |

Model bundles use `<task>/<zero_latency|profile_latency>/small_model/` and `<task>/<zero_latency|profile_latency>/OpenVLA/`. Dataset model-specific records use `OpenVLA/` and `baseline_OpenVLA/` to distinguish future model families. Directory names describe the environment, condition and stage; record run IDs, clocks and timestamps in metadata rather than folder names. Every downstream artifact records upstream run IDs, file hashes and exact Hub revisions. Preserve relative paths required by profile readers and inference loaders. Avoid mutable `latest` references. Verify uploaded file sizes/hashes and commit IDs before using the upload as evidence that local copies can be removed.

## 6. Stage instructions

Command examples below use variables bound to actual paths in the run ledger. Resolve configurations before execution; do not run the examples with unset variables.

### P: profile the zero-latency VLA on 3090

Reuse `scripts/profile.sh` and `latency_bench.profiling.runner.run_profile`. Add or adapt only the requested workload in the isolated checkout's `configs/profiles.yaml`. Point its base config at the selected L0 bundle and its artifact root at a fresh run directory.

Set `session_count: 5`, `eval_episodes: 10`, `evaluation_seed: 42`, `calibrate_env_fps: false`, one inference worker and one environment. Record actual session seeds; five measurement runs are not five training seeds. The CLI does not provide episode/session-count flags, so configure the preset.

Target approximately one minute per episode using a profiling cap of `ceil(60 * ENV_FPS)` raw steps. Respect natural termination and the environment's existing TimeLimit; do not disable them to fill a minute. Record actual episode duration and sample count. Early termination may make the run much shorter; do not silently add episodes.

```bash
export CUDA_VISIBLE_DEVICES=0
export LATENCY_BENCH_ENV_BACKEND_START_METHOD=spawn
# For the established MuJoCo rendering path:
export MUJOCO_GL=osmesa
bash scripts/profile.sh run --model qwenoft --workload "$TASK" --model-path "$ZERO_BUNDLE"
```

Use the selected checkpoint's actual model preset if different. Confirm warmup, actual image dimensions, the first completed episode and `observation_attempts.jsonl` before detaching. Run all five sessions without unrelated GPU work between them.

Keep raw session traces, episode metrics, generated configs, warmup diagnostics, hardware metadata and the completed canonical profile/distribution/burst files. Fit only canonical observation attempts. The sample is `observation_to_action_latency`, including observation extraction, IPC, preprocessing, forward and decoding; isolated inference time is not a substitute. Exclude warmup from formal samples.

Report admitted/attempted observations, capacity drops and latency mean/std/P50/P95/P99/max. Hand downstream stages the immutable profile inside the completed run directory. Upload separately to the specified `<task>/profile_latency/profiling/` prefix instead of assuming the profiler's default `--publish` destination is correct.

### B: zero-latency VLA real-time baseline on 3090

Run 20 episodes using the task's established evaluation seed block and raw-step horizon, with the requested clock from Section 3. Use the same settings later for F. This budget is separate from P's 5 by 10 measurements.

Run `python -m latency_bench.run --config "$BASELINE_RT_YAML"` with `executor.mode: realtime`, `inference_devices: [cuda:0]`, `latency.method: measured`, `sync_cuda: true` and `evaluation.eval_latency_values: null`. Do not inject profile latency on top of measured latency.

P's profiling horizon may differ from B's evaluation horizon, so do not automatically reuse P returns as the baseline. B can run alongside S on the other server. Once P and B evidence is synchronized, reclaim the L0 checkpoint's 3090 disk space as described in Section 2.

### S: train the small model on H100

Derive the configuration from the actual L0 teacher recipe. Apply the requested clock and replace latency with:

```yaml
latency:
  method: iid
  fixed_latency_ms: null
  profile_path: /absolute/task/run/profile/profile.json
  profile_worker_slot: 0
  seed: 0
  add_latency_info: false
```

Preserve the baseline's `add_latency_info` setting if different; do not change its observation space. Run `python -m training.train --config "$SMALL_TRAIN_YAML"` in the established environment, with a fresh experiment/output directory. Train from the same initialization rule as L0, not by continuing the trained L0 teacher.

Complete the baseline frame budget and select best/final according to its original rule. Save the checkpoint, config, rollout descriptor, selection evidence and profile reference. Upload the complete usable small-model bundle and transfer the copy needed by E. An isolated weight file is insufficient if the loader needs its experiment metadata.

Current Sample Factory IID training injects action latency only; it does not reproduce worker occupancy or capacity drops. Preserve this distinction in the report. Do not expand the training implementation to temporal replay as part of this run.

### E: simulated profile-latency small-model eval on 3090

Run 10 episodes with the established seeds/raw-step horizon and the requested clock. Bind the new checkpoint with `policy.type: gymnasium_sf` and run `python -m latency_bench.run --config "$SMALL_EVAL_YAML"`.

Use `executor.mode: simulated`, `simulated_worker_capacity: 1`, `simulated_inference_pool: true`, `inference_devices: [cuda:0]`, `latency.method: iid`, the new profile path, latency seed 0 and `evaluation.eval_latency_values: null`. Follow `profiling/runtime_configs.py::simulation_config` for replay semantics, then bind the small policy. The default profile simulation command may bind the source VLA instead.

This evaluation includes worker occupancy/capacity drops, unlike action-only IID training. Verify the distribution and drop counts in recorded traces. Do not leave the default fixed-latency sweep enabled or replace the measured distribution with a single frame delay. Save results before removing the local small-model copy.

### R/C: teacher rollout and conversion on H100

Use the new teacher with the requested clock and explicit new latency configuration. Preserve the L0 rollout recipe: accepted episode budget, quality gate, raw horizon, split, action/state contract and deterministic/stochastic sampling. The established pipeline uses 100 accepted episodes; resolve the exact source metadata before launching.

Use E and an exporter probe to assess data collection. If no episodes pass the original gate, record attempts and returns and diagnose the failure. Do not lower the gate or let a collection with no progress run indefinitely. Completing the horizon alone does not establish teacher quality.

```bash
python scripts/rollout_data/export_sf_teacher_dataset.py \
  --checkpoint-root "$SMALL_CHECKPOINT" --output-dir "$RAW_DIR" \
  --latency-config "$SMALL_TRAIN_YAML" \
  --max-episodes 100 --max-steps-per-episode "$RAW_HORIZON" \
  --val-fraction 0.1 --split-seed 0 \
  --export-image-size 224,224 --filter-preset none \
  --episode-return-gt "$RETURN_GATE"

python scripts/rollout_data/prepare_starvla_lerobot_dataset.py \
  "$RAW_DIR" "$LEROBOT_ROOT" "$DATA_MIX" --action-carrier native
```

Complete this CLI template using the source contract, including deterministic sampling, context window, environment kwargs, device and rollout parallelism; confirm the shown split/filter settings against that contract. Explicitly pass `--latency-config` and verify the recorded latency source, profile hash and FPS.

The exporter currently uses action-delay IID semantics, not E's complete worker-pool simulation. Record accepted/attempted counts, return range, raw/converted rows, train/validation episode IDs, image size, state/action dimensions, filter report, normalization stats and manifest/prompt map.

Verify the resolved Sample Factory sampler method is actually `iid`: the logging label `profile_sample` means temporal sampling in the flat runtime config. The 10 FPS InvertedPendulum run exposed and fixed an override conversion that confused these values. After image replay, check actual per-episode returns in the materialized data against the original gate and reconcile probe/replay row counts. Probe acceptance metadata and exit code 0 alone do not prove that the exported trajectories satisfy the gate.

Complete conversion before starting V. Upload raw data, including referenced images, while retaining relative paths. Generate converted datasets on the training server; do not publish a converted dataset copy. Keep these datasets off the 3090.

### V: train the VLA on H100

Derive the run from the selected zero-latency VLA's complete recipe. Update dataset paths/mixes, task metadata, timing metadata, run name and output paths. Preserve pretrained initialization, batch/accumulation, optimizer/LR, precision, state encoding, action head, seed, step budget and held-out validation cadence. Do not initialize from the already fine-tuned L0 VLA unless that is explicitly the required recipe.

Set `WANDB_MODE=online`, use the established entity/project and record the actual online run URL. Set `rl_games.env_eval.post_train.enabled=false`. Preserve the source mid-train evaluation setting. `trainer.eval_interval` controls held-out validation and must not be removed merely because post-train environment evaluation is disabled.

The underlying entrypoint, from the StarVLA checkout, is:

```bash
WANDB_MODE=online python starVLA/training/train_starvla.py \
  --config_yaml "$RESOLVED_VLA_CONFIG" \
  rl_games.env_eval.post_train.enabled=false
```

Preserve the source single-GPU torchrun/DeepSpeed launcher when applicable instead of replacing it with this bare entrypoint. Confirm steps advance, W&B is online and output paths are correct, then use delayed monitoring.

Completion requires the specified final step, exit code 0 and a fully written final checkpoint with inference assets. Upload final weights, full config, manifest, normalization/statistics, required prompt map and provenance; exclude optimizer states and caches. Transfer only this final inference bundle to 3090 after reclaiming obsolete checkpoint space. Low training loss is not evidence of closed-loop success.

### F: new VLA real-time eval on 3090

Run 20 episodes under B's exact evaluation contract, including the user-specified FPS. Bind the new complete bundle. Measure actual latency; do not inject the training profile or report its mean as the new model's measured latency.

Record per-episode returns/lengths/seeds, return mean and standard deviation, latency mean/P95, deadline misses, dropped observations/actions and invalid actions. Synchronize evidence and verify the final Hub checkpoint before cleaning up its 3090 copy.

## 7. Long jobs, wakeups and recovery

Use the existing tmux or suitable supervisor setup for remote batch jobs. Save logs and exit-code files so jobs survive SSH disconnection. Use simple sequential stage commands and existing completion artifacts rather than building a new scheduler.

After V starts normally, estimate completion using 50-100 stable steps:

`ETA = remaining_steps * median_step_seconds + remaining_validation_and_save_time`

Set a Codex heartbeat for this execution task based on that ETA. For a roughly four-hour run, arrange the first check around three to four hours later, then adjust from the remaining work. Do not poll every few minutes or assume all environments train in the same time. Short jobs should use completion-driven handoff rather than waiting hours for the next check.

The heartbeat should read this plan and the ledger, inspect the remote process/last step/exit code/final checkpoint, and continue with verification, upload and F once training finishes. Stay quiet when nothing meaningful changes; notify on completion, failure or required user action. Record its ID and pause it when the pipeline finishes. Do not create an idle monitor before a job exists.

Resume P with `--resume RUN_ID` only under the same run contract. Resume training from this run's explicit checkpoint and W&B ID. Use the exporter's existing shard/probe recovery semantics for R/C. Retry failed uploads or evaluations without retraining. Preserve failure evidence and successful session outputs.

## 8. Timing, ledger and report

The main time savings come from running B alongside S, overlapping uploads with H100 work, reusing one checkpoint download for P/B, and avoiding upload-then-download waits for F.

| Stage | Estimate |
| --- | --- |
| P | Nominally 5 * 10 * 60 seconds = 50 minutes, plus loading/warmup; natural early termination can shorten it |
| B/F | Each 20 episodes at the requested clock and established raw horizon; use observed termination lengths |
| S | Fixed training frame budget divided by measured training throughput |
| R/C | Accepted episode target divided by observed acceptance rate, plus replay/image writing/conversion time |
| V | Remaining steps times measured step time, plus validation/saving |
| Transfer | Required bundle size divided by observed bandwidth |

Estimate end-to-end time from the dependency graph: after P, B overlaps S; E starts when both the 3090 and S output are ready; R/C/V follow E; F follows final-bundle availability. Report a measured ETA after startup rather than promising a fixed completion time. A zero acceptance rate has no finite rollout ETA.

Keep a task-specific ledger beside the run configs and link it from the report. Record stage, status, host, run ID, command/config path, input hash/revision, requested env/obs FPS, process/session, log, exit code, output path, Hub commit, cleanup actions, next action and ETA. Use `TODO/RUNNING/DONE/FAILED/BLOCKED`; a completed plan is not a completed experiment.

Update the assigned environment's report under `docs/gym-adapt/`; for the existing Gymnasium reports this is `gymnasium/<task>/index.html`, with evidence under `assets/<task>/profile-latency-new/<run_id>/` and a README link.

| Training condition | Evaluation condition | Episodes | Return mean +/- std | Mean length | Latency mean/P95 | Drops/invalid | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L0 VLA | 3090 realtime, requested FPS | 20 | Pending | Pending | Measured | Pending | B |
| New profile small model | Simulated IID profile, one worker pool, requested FPS | 10 | Pending | Pending | Replayed | Pending | E |
| New profile VLA | 3090 realtime, requested FPS | 20 | Pending | Pending | Measured | Pending | F |

Include exact env/obs FPS, seeds, horizons, the five-run profile summary, training budgets/W&B, dataset acceptance statistics and Hub revisions. Mark old mismatched-image results as historical/superseded and keep their evidence separate. A slower requested clock must be visible in the comparison.

Completion requires P/B/S/E/R/C/V/F evidence, all four requested artifact uploads verified, and the updated report with working links. Check actual report rendering after edits; data validation alone is not visual QA. If the teacher fails its original gate, report the blocked downstream stages rather than filling them with weaker data. Do not automatically commit or push repository changes.

## 9. Execution checklist

1. Read the assignment, bind its environment and requested FPS, inspect both single-GPU hosts and reclaim safe-to-remove 3090 checkpoint storage.
2. Resolve the source bundle/teacher recipe, pin code and artifact revisions, and freeze the clock/image/action/state contract.
3. Prepare only this task's presets and resolved configs; confirm 5 by 10 profiling, output prefixes and usable free space.
4. Run P while preparing H100; then run S and B in parallel. Save P/B evidence and remove the reproducible L0 copy when no longer needed.
5. Run E, then R/C/V; upload artifacts as they become ready. Set the delayed heartbeat after V is confirmed healthy.
6. Transfer the final bundle, run F, synchronize reports/evidence, verify uploads and reclaim unused 3090 checkpoint copies.

Implementation references: `configs/profiles.yaml`, `latency_bench/profiling/{cli,runner,runtime_configs}.py`, `docs/profiling/cookbook.md`, `training/train.py`, `scripts/rollout_data/{export_sf_teacher_dataset,prepare_starvla_lerobot_dataset}.py`, and `third_party/starVLA/starVLA/training/train_starvla.py`. Prior experimental context is in the same project's `exps` task; historical results are not results from this run.
