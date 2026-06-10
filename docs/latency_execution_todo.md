# Latency Execution TODO / Audit

**Status:** Dynamic working document
**Last audit:** 2026-05-08
**Scope:** This file tracks current implementation gaps, audit evidence, refactor phases, open questions, and moving TODOs.
**Important:** Stable semantic definitions belong in `latency_execution_semantics_zh.md` and `latency_execution_semantics_en.md`, not here.

---

## 1. Source of truth split

| File | Purpose | Should contain | Should not contain |
|---|---|---|---|
| `docs/latency_execution_semantics_zh.md` | Stable Chinese semantic contract | definitions, invariants, executor semantics | current-code bugs, TODOs, temporary plans |
| `docs/latency_execution_semantics_en.md` | Stable English semantic contract | definitions, invariants, executor semantics | current-code bugs, TODOs, temporary plans |
| `docs/latency_execution_todo.md` | Dynamic implementation tracker | bugs, phases, audit tables, open questions | stable semantic redefinitions |

### Audit Baseline

Phase 0 audit artifacts:

- `docs/audits/latency_code_alignment_audit.md`
- `docs/audits/latency_refactor_plan.md`

The audit docs are the current detailed baseline for code/semantics mismatches and the corrected Phase 0 through Phase 6 refactor plan.

### Phase 1 Timing Baseline

Phase 1 adds explicit timing helpers and a backward-compatible `clock` config shim:

- `latency_bench.core.clock.validate_clock_rates`
- `latency_bench.core.clock.compute_obs_stride_raw_frames`
- `latency_bench.core.clock.ms_to_raw_frames`
- `latency_bench.core.config.resolve_clock_config`

Executor scheduling, scheduler ordering, realtime behavior, hold policies, and logging schema are intentionally unchanged.

### Phase 3 Simulated Raw-Frame Alignment

Phase 3 is implemented for the simulated executors:

- `ActionEvent` now uses `obs_env_step` as the observation raw-frame index and carries `latency_raw_frames`, `ready_raw_frame`, and `apply_raw_frame`.
- `ActionScheduler` uses `ready_raw_frame` when present and writes `apply_raw_frame` when an action is applied.
- `SimulatedLatencyExecutor` and `BatchedSimulatedLatencyExecutor` schedule observation, readiness, and application on raw-frame indices. Existing millisecond latency sources remain compatible through executor-local conversion; sources may optionally expose `next_latency_raw_frames(context)`.
- Metrics finalization backfills raw-frame fields from legacy aliases when older event producers do not set them.

Phase 4/5 work remains open for realtime queue-full behavior, timeout/skipped/late-result lifecycle, and the complete logging/job-status schema.

---

## 2. Relevant code map

| Area | Files / classes / functions |
|---|---|
| Env adapters | `latency_bench/envs/base.py`, `latency_bench/envs/factory.py`, `latency_bench/envs/flappy.py`, `latency_bench/envs/demon_attack.py`, `latency_bench/envs/deadly_corridor.py` |
| Observation pipelines | `latency_bench/envs/flappy.py`, `latency_bench/envs/demon_attack.py`, `latency_bench/envs/deadly_corridor.py` |
| Executors | `latency_bench/executors/paused.py`, `latency_bench/executors/simulated.py`, `latency_bench/executors/realtime.py`, `latency_bench/executors/batched_simulated.py`, `latency_bench/executors/factory.py` |
| Latency sources | `latency_bench/latency/sources.py`, `latency_bench/latency/measurement.py`, `latency_bench/latency/profile.py`, `latency_bench/latency/trace.py` |
| Action queue / scheduler | `latency_bench/scheduler/action_queue.py`, `latency_bench/scheduler/persistence.py`, `latency_bench/scheduler/decision.py` |
| Policy inference runners | `latency_bench/policy/base.py`, `latency_bench/policy/factory.py`, `latency_bench/policy/sample_factory_common.py` |
| Logging and metrics | `latency_bench/logging/metrics.py`, `latency_bench/logging/trajectory_logger.py`, `latency_bench/logging/summary.py` |
| Training latency wrappers | `training/sample_factory_common/latency_wrapper.py`, `training/flappy_sf/integration.py`, `training/demon_attack_sf/integration.py`, `training/deadly_corridor_sf/integration.py` |
| Tests | `tests/test_action_queue.py`, `tests/test_decision_scheduler.py`, `tests/test_simulated_executor.py`, `tests/test_paused_executor.py`, `tests/test_realtime_executor.py`, `tests/test_latency_sources.py`, `tests/test_config.py`, `tests/test_logging_records.py` |

---

## 3. Top mismatches

1. Config and clock naming still center on `env.env_fps`, `control.obs_fps`, and `obs_fps`; there is no first-class `clock.env_fps`, `clock.obs_fps`, or divisibility validation.
2. Latency sources remain millisecond-first for compatibility; simulated executors now convert to canonical `latency_raw_frames` before scheduling.
3. Phase 2.5 removes legacy scheduler aliases; runtime config should use only `ordering_policy`.
4. Phase 2 resolves normal FIFO head-of-line blocking; timeout/skipped unblocking remains a later realtime phase.
5. Phase 2.5 removes legacy hold aliases; runtime config should use only `hold_policy`.
6. Realtime inference now uses one worker process per configured `executor.inference_devices` entry; observations are dropped when no worker is idle, and `latest_ready` ignores older late results.
7. Realtime logs actual FPS, deadline misses, in-flight/idle worker counts, dropped observations, and ignored late results; no timeout/outlier lifecycle is part of the current realtime contract.
8. Training latency wrappers schedule latency by wrapper decision steps; Atari/Doom configs currently blur wrapper-step FPS and raw-frame FPS.

---

## 4. Audit table

| Component | Current behavior | Intended behavior | Gap / bug | Suggested refactor | Risk level | Evidence |
|---|---|---|---|---|---|---|
| Clock / config | Config requires `env.env_fps`; control uses `obs_fps` or `obs_fps`; fixed frequency rounds `env_fps / obs_fps`. | Use explicit `clock.env_fps`, `clock.obs_fps`, and `obs_stride_raw_frames`, with `env_fps % obs_fps == 0`. | No first-class observation timeline; non-divisible observation rates can be silently rounded. | Add clock schema, backward-compatible aliases, and fail-fast divisibility validation. | High | `latency_bench/core/config.py:100-109`; `latency_bench/scheduler/decision.py:27-36`; `latency_bench/executors/factory.py:362-369` |
| Latency source / conversion | `LatencySource` still exposes `next_latency_ms` by default; simulated executors use an executor-local helper that prefers `next_latency_raw_frames(context)` when available and otherwise converts ms with ceiling semantics. | Latency should be represented and applied at raw-frame granularity, with ms converted by the semantic helper. | Phase 3 simulated scheduling is resolved; full latency source API/schema migration is deferred. | Keep the compatibility helper; consider first-class latency unit schema only if future sources need it. | Done for Phase 3 | `latency_bench/executors/_simulated_timeline.py`; `latency_bench/executors/simulated.py`; `latency_bench/executors/batched_simulated.py`; `latency_bench/core/clock.py` |
| Action ordering default | Phase 2.5 exposes only `ordering_policy` with `issue_order_fifo` and `latest_ready`; omitted ordering defaults to issue-order FIFO. | Default benchmark ordering should be issue-order FIFO; `latest_ready` should be explicit ablation. | Resolved; previous aliases and obsolete ordering variants were removed. | Keep runtime API names limited to the new contract. | Done | `latency_bench/scheduler/action_queue.py`; `latency_bench/executors/factory.py`; `tests/test_action_queue.py`; `tests/test_executor_factory.py` |
| FIFO scheduler semantics | Phase 2 applies only the head-of-line pending action in FIFO mode. | Issue-order FIFO should prevent a later-issued action from applying before an earlier-issued action unless the earlier action is skipped. | Resolved for normal ready actions; timeout/skipped unblocking remains later. | Add timeout/skipped unblocking in the realtime timeout phase. | Done | `latency_bench/scheduler/action_queue.py`; `tests/test_action_queue.py` |
| Hold policies | Phase 2.5 exposes only `hold_policy` with `one_frame_then_noop`, `hold`, and `hold`. | Support explicit `one_frame_then_noop`, `hold`, and `hold`. | Resolved; previous aliases were removed. | Keep runtime API names limited to the new contract. | Done | `latency_bench/core/types.py`; `latency_bench/scheduler/persistence.py`; `latency_bench/executors/factory.py`; `tests/test_persistence.py` |
| Paused executor | Policy prediction runs synchronously; event ready time is current time; scheduler update and env step happen in the same loop iteration. | Paused mode should stop environment time during inference and apply the returned action to the next raw environment update. | Behavior may be semantically acceptable, but current tests describe it as same-frame; off-by-one wording needs a decision. | Define observation-before-update vs observation-after-update ordering, then align tests and logs. | Medium | `latency_bench/executors/paused.py:79-92`; `latency_bench/executors/paused.py:117-137`; `tests/test_paused_executor.py:36-43` |
| Realtime inference pool | Realtime starts one inference worker process per `executor.inference_devices` entry; each worker loads its own policy instance and handles one in-flight job at a time. | Environment progression and inference should be independent; new observations should be dispatched only to idle workers, and dropped when none are idle. | Current implementation matches the new no-timeout/no-outlier contract; further work is analysis/scale testing on real multi-GPU hardware. | Keep pool semantics explicit and avoid reintroducing pending observation queues. | Done for current contract | `latency_bench/executors/realtime.py`; `tests/test_realtime_executor.py` |
| Realtime synthetic latency | Realtime does not accept injected latency; `executor.mode: realtime` measures runtime in the inference worker. | Realtime should measure model runtime rather than add synthetic post-completion delay. | Resolved for current contract. | Keep synthetic latency in simulated executors. | Done | `latency_bench/executors/factory.py`; `latency_bench/executors/realtime.py` |
| Realtime timing/logging | Step info and episode metrics record wall lag, actual FPS, deadline misses, in-flight/idle worker counts, dropped observations, FIFO blocking, and ignored late results. | Logs should be sufficient to audit whether env progression waited for inference and whether worker capacity was saturated. | Current fields cover the new contract without adding new `ActionEvent` top-level fields. | Add downstream analysis only if consumers need derived fields such as action age. | Done for current contract | `latency_bench/executors/realtime.py`; `tests/test_realtime_executor.py` |
| Policy inference API | `PolicyRunner.predict` is synchronous; `predict_batch` is optional and not an async interface. | Realtime needs an async/pool execution boundary independent of policy implementation. | Pool semantics are currently implemented around synchronous calls rather than exposed as a first-class runner contract. | Keep policy API synchronous, but define an executor-owned `InferencePool` wrapper contract. | Medium | `latency_bench/policy/base.py:10-21`; `latency_bench/executors/realtime.py:69-97`; `latency_bench/executors/realtime.py:126-153` |
| Observation layer | Env adapters emit policy-ready observations from `observe()`, while simulated executors interpret decision scheduler input as raw-frame indices. | Observation timeline should be a subsampled view of raw environment timeline with explicit stride. | Phase 3 simulated observation scheduling is raw-frame aligned; observation construction is adapter-local rather than a shared builder layer. | Keep decision scheduler compatibility; avoid reintroducing a shared observation-builder abstraction unless cross-env reuse becomes necessary. | Done for simulated | `latency_bench/envs/flappy.py`; `latency_bench/envs/demon_attack.py`; `latency_bench/envs/deadly_corridor.py`; `latency_bench/scheduler/decision.py`; `latency_bench/executors/simulated.py`; `latency_bench/executors/batched_simulated.py` |
| Env/training raw vs decision frames | `QueueLatencyWrapper` schedules by `env_step_idx`; Demon Attack config comments that raw frames should be `15 x 4 = 60`; Deadly Corridor uses `env_fps` plus `env_frameskip`. | Latency should ultimately apply at raw-frame granularity. | Existing training wrapper and configs mix wrapper decision steps with raw-frame terminology. | Treat current training wrapper as legacy decision-step latency until a dedicated alignment pass. | High | `training/sample_factory_common/latency_wrapper.py:160-179`; `training/sample_factory_common/latency_wrapper.py:234-247`; `configs/base/demon_attack.yaml:20-25`; `configs/base/deadly_corridor.yaml:20-22`; `training/demon_attack_sf/integration.py:138-157` |
| Env adapter implementation hygiene | `FlappySampleFactoryGpuEnvAdapter` is defined twice in the same module; the later definition shadows the earlier one. | Env adapter definitions should have one authoritative implementation. | Not directly a latency semantic bug, but it increases audit/refactor risk. | Remove or rename the shadowed implementation in a separate cleanup pass. | Low | `latency_bench/envs/flappy.py:172`; `latency_bench/envs/flappy.py:391` |
| Tests | Contract and executor tests cover timing helpers, FIFO, hold policies, simulated raw-frame conversion, realtime observation stride, drop-new behavior, latest-ready late-result ignore, and worker/device latency metadata. | Tests should cover semantic invariants before refactor. | Current realtime contract has no timeout/outlier xfails. | Add hardware integration tests separately for real multi-GPU runs. | Done for current contract | `tests/test_latency_execution_semantics_contract.py`; `tests/test_realtime_executor.py`; `tests/test_simulated_executor.py`; `tests/test_batched_simulated_executor.py`; `tests/test_eval_logging.py` |

---

## 5. Proposed phased implementation plan

The canonical Phase 0 through Phase 6 plan is maintained in `docs/audits/latency_refactor_plan.md`.

Summary:

- Phase 0: semantic tests and audit baseline.
- Phase 1: explicit timing units.
- Phase 2: action scheduler and hold policies.
- Phase 2.5: remove legacy scheduler/persistence aliases.
- Phase 3: simulated executor alignment. Done for single and batched simulated executors.
- Phase 4: realtime async inference.
- Phase 5: logging and validation.
- Phase 6: cleanup.

---

## 6. Open questions requiring human decision

1. Confirm default hold policies:
   - Flappy Bird: `one_frame_then_noop`
   - Demon Attack: `hold`
   - Deadly Corridor: `hold`
2. Confirm `issue_order_fifo` as the default benchmark ordering and `latest_ready` as ablation-only.
3. Confirm `drop_new_observation` as default queue-full behavior.
4. Choose integer conversion for stochastic raw-frame latency values that are sampled as floats: `ceil`, `round_half_up`, `floor`, or stochastic rounding.
5. Define realtime degraded/invalid thresholds, for example deadline miss rate, actual FPS ratio, and p95 wall-clock lag.
6. Resolve paused-mode off-by-one wording: whether “next raw environment update” is the same loop iteration after observation emission or the following raw-frame index.
7. Decide whether training `QueueLatencyWrapper` is legacy decision-step latency or should be migrated to raw-frame latency.

---

## 7. Notes for future updates

1. Do not redefine stable semantics here; link to the semantic docs instead.
2. Mark implementation work as `Planned`, `In progress`, `Done`, or `Deferred`.
3. Include exact file paths and line numbers for every audited claim.
4. Keep current bugs, audit status, and refactor phases out of the semantic docs.
