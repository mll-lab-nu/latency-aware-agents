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

---

## 2. Relevant code map

| Area | Files / classes / functions |
|---|---|
| Env adapters | `latency_bench/envs/base.py`, `latency_bench/envs/factory.py`, `latency_bench/envs/flappy.py`, `latency_bench/envs/demon_attack.py`, `latency_bench/envs/deadly_corridor.py` |
| Observation builders | `latency_bench/observation/builder.py`, `latency_bench/observation/history.py`, `latency_bench/observation/sample_factory.py`, `modeling/flappy_ppo/observation.py`, `modeling/demon_attack_ppo/observation.py` |
| Executors | `latency_bench/executors/paused.py`, `latency_bench/executors/simulated.py`, `latency_bench/executors/realtime.py`, `latency_bench/executors/batched_simulated.py`, `latency_bench/executors/factory.py` |
| Latency sources | `latency_bench/latency/sources.py`, `latency_bench/latency/measurement.py`, `latency_bench/latency/profile.py`, `latency_bench/latency/trace.py` |
| Action queue / scheduler | `latency_bench/scheduler/action_queue.py`, `latency_bench/scheduler/persistence.py`, `latency_bench/scheduler/decision.py` |
| Policy inference runners | `latency_bench/policy/base.py`, `latency_bench/policy/factory.py`, `latency_bench/policy/flappy_ppo.py`, `latency_bench/policy/sample_factory_common.py` |
| Logging and metrics | `latency_bench/logging/metrics.py`, `latency_bench/logging/trajectory_logger.py`, `latency_bench/logging/summary.py` |
| Training latency wrappers | `training/sample_factory_common/latency_wrapper.py`, `training/flappy_sf/integration.py`, `training/demon_attack_sf/integration.py`, `training/deadly_corridor_sf/integration.py` |
| Tests | `tests/test_action_queue.py`, `tests/test_decision_scheduler.py`, `tests/test_simulated_executor.py`, `tests/test_paused_executor.py`, `tests/test_realtime_executor.py`, `tests/test_latency_sources.py`, `tests/test_config.py`, `tests/test_logging_records.py` |

---

## 3. Top mismatches

1. Config and clock naming still center on `env.fps_env`, `control.frequency_hz`, and `observe_every_n_frames`; there is no first-class `clock.env_fps`, `clock.obs_fps`, or divisibility validation.
2. Latency sources are millisecond-first, and executors schedule by `ready_time_ms` / float step rather than canonical `latency_raw_frames`.
3. The configured/default scheduler policy is `latest_ready`, while the intended benchmark default is issue-order FIFO.
4. Current FIFO behavior is ready-time FIFO among ready actions; it does not block on an earlier issued action that is not ready yet.
5. Hold behavior is encoded as `hold_previous` or `noop_after_oneshot`; there is no explicit fixed-duration hold policy.
6. Realtime inference uses one serial worker thread and a drop-oldest pending queue; there is no `max_in_flight`, default drop-new behavior, timeout lifecycle, or late-result ignore path.
7. Realtime logs some catchup/drop data but do not yet expose the full required timing, queue, in-flight, timeout, and actual-FPS fields.
8. Training latency wrappers schedule latency by wrapper decision steps; Atari/Doom configs currently blur wrapper-step FPS and raw-frame FPS.

---

## 4. Audit table

| Component | Current behavior | Intended behavior | Gap / bug | Suggested refactor | Risk level | Evidence |
|---|---|---|---|---|---|---|
| Clock / config | Config requires `env.fps_env`; control uses `frequency_hz` or `observe_every_n_frames`; fixed frequency rounds `fps_env / frequency_hz`. | Use explicit `clock.env_fps`, `clock.obs_fps`, and `obs_stride_raw_frames`, with `env_fps % obs_fps == 0`. | No first-class observation timeline; non-divisible observation rates can be silently rounded. | Add clock schema, backward-compatible aliases, and fail-fast divisibility validation. | High | `latency_bench/core/config.py:100-109`; `latency_bench/scheduler/decision.py:27-36`; `latency_bench/executors/factory.py:362-369` |
| Latency source / conversion | `LatencySource` exposes `next_latency_ms`; fixed/normal/uniform/trace values are ms; simulated executor computes `ready_time_ms` and `ready_step_float`. | Latency should be represented and applied at raw-frame granularity, with ms converted by the semantic helper. | No canonical `latency_raw_frames` field or ceil conversion path in the scheduler. | Add a latency unit enum and conversion helper; store sampled/effective raw-frame latency in events/logs. | High | `latency_bench/latency/sources.py:13-16`; `latency_bench/latency/sources.py:24-107`; `latency_bench/executors/simulated.py:86-102`; `latency_bench/core/clock.py:26-29` |
| Action ordering default | Simulated and realtime builders default to `StaleActionMode.LATEST_READY` when config is absent. | Default benchmark ordering should be issue-order FIFO; `latest_ready` should be explicit ablation. | Default behavior does not match intended benchmark policy. | Change default after compatibility migration; require explicit `latest_ready` in legacy configs. | High | `latency_bench/executors/factory.py:118-123`; `latency_bench/executors/factory.py:269-274`; `configs/base/flappy.yaml:59-61`; `configs/base/demon_attack.yaml:51-53` |
| FIFO scheduler semantics | Pending events are sorted by `(ready_time_ms, action_id)`; FIFO selects the first ready event only among ready events. | Issue-order FIFO should prevent a later-issued action from applying before an earlier-issued action unless the earlier action is skipped. | Later-issued ready actions can bypass earlier-issued not-ready actions. | Store issue order separately; make FIFO head-of-line blocking explicit; add skipped/timeout unblocking. | High | `latency_bench/scheduler/action_queue.py:37-54`; `tests/test_action_queue.py:64-75`; `tests/test_action_queue.py:78-97` |
| Hold policies | Persistence modes are only `hold_previous` and `noop_after_oneshot`; noop-after-one-shot depends on `Action.is_oneshot`; fixed duration is absent. | Support explicit `one_frame_then_noop`, `hold_for_fixed_duration`, and `hold_until_next_action`. | Policy names and behavior are implicit; no fixed-duration hold. | Introduce explicit hold policy enum/controller and migrate old names as aliases. | Medium | `latency_bench/core/types.py:14-17`; `latency_bench/scheduler/persistence.py:18-52`; `latency_bench/executors/factory.py:412-418`; `configs/base/flappy.yaml:35-40` |
| Paused executor | Policy prediction runs synchronously; event ready time is current time; scheduler update and env step happen in the same loop iteration. | Paused mode should stop environment time during inference and apply the returned action to the next raw environment update. | Behavior may be semantically acceptable, but current tests describe it as same-frame; off-by-one wording needs a decision. | Define observation-before-update vs observation-after-update ordering, then align tests and logs. | Medium | `latency_bench/executors/paused.py:79-92`; `latency_bench/executors/paused.py:117-137`; `tests/test_paused_executor.py:36-43` |
| Realtime inference pool | `SerialPolicyWorker` owns one thread; queue push drops oldest pending task; executor has `max_pending_observations` but no `max_in_flight` or timeout policy. | Realtime should use an async pool with `max_in_flight`, default `drop_new_observation`, timeout/skipped events, and late-result ignore. | Cannot model inference-pool utilization or intended queue semantics. | Replace serial worker with pool abstraction; add queue-full policy and timeout state machine. | High | `latency_bench/executors/realtime.py:49-97`; `latency_bench/executors/realtime.py:160-178`; `latency_bench/executors/realtime.py:244-253`; `tests/test_realtime_executor.py:346-358` |
| Realtime synthetic latency | Non-measured realtime completions add latency after policy completion: `ready_time_ms = completion_time_ms + source_latency_ms`. | Realtime should clearly distinguish measured model runtime from any additional configured delay. | Synthetic delay after completion may be intentional, but the semantics are not explicit in config/log names. | Split measured inference duration, injected delay, queue wait, and effective latency fields. | Medium | `latency_bench/executors/realtime.py:262-293`; `latency_bench/logging/metrics.py:97-127` |
| Realtime timing/logging | Step info logs catchup, lag, and latest-only submission; metrics log dropped/skipped/catchup counts. | Logs should include actual env FPS, deadline miss rate, in-flight count, pending queue size, timeout count, and queue events. | Missing fields needed to audit realtime behavior. | Extend step/action/latency/episode metric schema with required realtime fields. | Medium | `latency_bench/executors/realtime.py:400-432`; `latency_bench/executors/realtime.py:476-495`; `latency_bench/logging/metrics.py:149-170` |
| Policy inference API | `PolicyRunner.predict` is synchronous; `predict_batch` is optional and not an async interface. | Realtime needs an async/pool execution boundary independent of policy implementation. | Pool semantics are currently implemented around synchronous calls rather than exposed as a first-class runner contract. | Keep policy API synchronous, but define an executor-owned `InferencePool` wrapper contract. | Medium | `latency_bench/policy/base.py:10-21`; `latency_bench/executors/realtime.py:69-97`; `latency_bench/executors/realtime.py:126-153` |
| Observation layer | Observation builder returns history/latest state; observation cadence is controlled by decision scheduler, not an explicit `obs_fps` layer. | Observation timeline should be a subsampled view of raw environment timeline with explicit stride. | Cadence exists, but naming and validation are not aligned with the stable semantics. | Introduce `obs_fps`/stride config and route existing decision scheduler through it. | Medium | `latency_bench/observation/builder.py:8-22`; `latency_bench/scheduler/decision.py:17-36`; `latency_bench/executors/factory.py:362-369` |
| Env/training raw vs decision frames | `QueueLatencyWrapper` schedules by `env_step_idx`; Demon Attack config comments that raw frames should be `15 x 4 = 60`; Deadly Corridor uses `fps_env` plus `env_frameskip`. | Latency should ultimately apply at raw-frame granularity. | Existing training wrapper and configs mix wrapper decision steps with raw-frame terminology. | Treat current training wrapper as legacy decision-step latency until a dedicated alignment pass. | High | `training/sample_factory_common/latency_wrapper.py:160-179`; `training/sample_factory_common/latency_wrapper.py:234-247`; `configs/base/demon_attack.yaml:20-25`; `configs/base/deadly_corridor.yaml:20-22`; `training/demon_attack_sf/integration.py:138-157` |
| Env adapter implementation hygiene | `FlappySampleFactoryGpuEnvAdapter` is defined twice in the same module; the later definition shadows the earlier one. | Env adapter definitions should have one authoritative implementation. | Not directly a latency semantic bug, but it increases audit/refactor risk. | Remove or rename the shadowed implementation in a separate cleanup pass. | Low | `latency_bench/envs/flappy.py:172`; `latency_bench/envs/flappy.py:391` |
| Tests | Tests cover current latest-ready, drop-oldest pending, fixed ms latency, and paused same-frame behavior. | Tests should cover semantic invariants before refactor. | Missing tests for non-divisible `obs_fps`, issue-order blocking FIFO, timeout unblocking, drop-new queue policy, raw-frame conversion, and fixed-duration hold. | Add failing semantic tests before changing implementation. | Medium | `tests/test_action_queue.py:64-97`; `tests/test_realtime_executor.py:346-358`; `tests/test_paused_executor.py:36-43`; `tests/test_decision_scheduler.py:29-41`; `tests/test_simulated_executor.py:133-149` |

---

## 5. Proposed phased implementation plan

The canonical Phase 0 through Phase 6 plan is maintained in `docs/audits/latency_refactor_plan.md`.

Summary:

- Phase 0: semantic tests and audit baseline.
- Phase 1: explicit timing units.
- Phase 2: action scheduler and hold policies.
- Phase 3: simulated executor alignment.
- Phase 4: realtime async inference.
- Phase 5: logging and validation.
- Phase 6: cleanup.

---

## 6. Open questions requiring human decision

1. Confirm default hold policies:
   - Flappy Bird: `one_frame_then_noop`
   - Demon Attack: `hold_until_next_action`
   - Deadly Corridor: `hold_until_next_action`
2. Confirm `fifo` as the default benchmark ordering and `latest_ready` as ablation-only.
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
