# Latency Execution Refactor Plan

**Status:** Phase 0 planning baseline
**Date:** 2026-05-08
**Scope:** Proposed implementation phases for aligning code with the stable latency execution semantics.

The stable semantic contract is maintained in:

- `docs/latency_execution_semantics_en.md`
- `docs/latency_execution_semantics_zh.md`

This plan is dynamic implementation guidance. It must not redefine the contract.

## Phase 0: Semantic tests and audit baseline

**Goal:** Create a safe baseline before behavior changes.

**Likely files:**

- `docs/audits/latency_code_alignment_audit.md`
- `docs/audits/latency_refactor_plan.md`
- `docs/latency_execution_todo.md`
- `tests/test_latency_execution_semantics_contract.py`

**Work:**

- Add audit documentation with exact code evidence.
- Add semantic contract tests for known gaps.
- Mark known current violations with `pytest.mark.xfail(strict=True, reason="...")`.
- Do not change runtime behavior.

**Risks:**

- Contract tests can become noisy if they depend on future class names too strongly.
- Strict xfail tests can become XPASS when later phases partially implement behavior; that is intentional and should force xfail removal or test update.

**Acceptance criteria:**

- Audit docs exist and separate stable semantics from implementation status.
- Contract tests collect and report expected xfails.
- Existing nearby lightweight tests still pass.
- No executor, scheduler, config, latency source, realtime, logging, training, or runtime default behavior changes.

**Independent:** Yes.

## Phase 1: Explicit timing units

**Goal:** Make raw-frame and observation-frame units explicit.
**Status:** Implemented for central helpers and backward-compatible config resolution. Executor alignment remains in later phases.

**Likely files:**

- `latency_bench/core/config.py`
- `latency_bench/core/clock.py`
- `latency_bench/core/types.py`
- `latency_bench/scheduler/decision.py`
- `latency_bench/executors/factory.py`
- `configs/base/*.yaml`
- `scripts/generate_sample_factory_same_latency_sweep.py`
- `tests/test_config.py`
- `tests/test_decision_scheduler.py`
- `tests/test_latency_execution_semantics_contract.py`

**Work:**

- Introduce canonical `env_fps`, `obs_fps`, and `obs_stride_raw_frames`.
- Validate `env_fps % obs_fps == 0`.
- Introduce a central helper for `ceil(latency_ms / 1000 * env_fps)`.
- Preserve legacy field compatibility during migration if that policy is chosen.

**Risks:**

- Existing configs use `env.fps_env`, `control.frequency_hz`, and `observe_every_n_frames`.
- Atari/Doom wrapper FPS and raw-frame FPS are currently easy to confuse.

**Acceptance criteria:**

- Valid fps pairs produce deterministic stride.
- Non-divisible fps pairs fail fast.
- Millisecond latency conversion uses integer ceiling semantics.
- Phase 0 timing-unit xfails are converted to passing tests.

**Independent:** Mostly yes; safer before scheduler/executor changes.

## Phase 2: Action scheduler and hold policies

**Status:** Implemented.

**Goal:** Separate action readiness, application ordering, and hold behavior.

**Likely files:**

- `latency_bench/core/types.py`
- `latency_bench/scheduler/action_queue.py`
- `latency_bench/scheduler/persistence.py`
- `latency_bench/executors/factory.py`
- `configs/base/*.yaml`
- `tests/test_action_queue.py`
- `tests/test_latency_execution_semantics_contract.py`

**Work:**

- Make issue-order FIFO the benchmark default.
- Preserve `latest_ready` as an explicit ablation.
- Keep skipped/timed-out action states for the realtime timeout phase.
- Add explicit hold policies:
  - `one_frame_then_noop`
  - `hold_for_fixed_duration`
  - `hold_until_next_action`

**Risks:**

- Changing default ordering changes benchmark results.
- Existing tests and configs encode current `latest_ready` behavior.
- Hold behavior is currently encoded partly through `Action.is_oneshot`.

**Acceptance criteria:**

- Later issued actions cannot overtake earlier issued unskipped actions in FIFO mode.
- `latest_ready` only occurs when explicitly configured.
- Three hold policies pass direct unit tests using raw-frame counts.
- Legacy hold names remain accepted through compatibility mappings.

**Independent:** Depends on Phase 1 for final raw-frame naming, but unit-level scheduler work can start earlier with compatibility shims.

## Phase 3: Simulated executor alignment

**Goal:** Make simulated latency scheduling occur on the raw environment timeline.

**Likely files:**

- `latency_bench/executors/simulated.py`
- `latency_bench/executors/batched_simulated.py`
- `latency_bench/executors/factory.py`
- `latency_bench/logging/metrics.py`
- `latency_bench/core/types.py`
- `tests/test_simulated_executor.py`
- `tests/test_batched_simulated_executor.py`
- `tests/test_latency_execution_semantics_contract.py`

**Work:**

- Emit observations according to `obs_stride_raw_frames`.
- Convert every sampled latency to `latency_raw_frames`.
- Schedule action readiness and application by raw-frame index.
- Log observation, ready, and apply raw frames.

**Risks:**

- Trace replay and stochastic latency reproducibility can change if conversion moves earlier or later.
- Batched simulated currently supports only zero/fixed latency.
- Paused/simulated/realtime logs need compatible event schemas.

**Acceptance criteria:**

- Zero, fixed, distribution, and trace latency tests agree on raw-frame readiness.
- Applied action frame is deterministic and logged.
- Batched simulated behavior is either aligned or explicitly limited by tests and docs.

**Independent:** Depends on Phase 1 and Phase 2.

## Phase 4: Realtime async inference

**Goal:** Replace serial realtime inference semantics with an explicit asynchronous inference pool.

**Likely files:**

- `latency_bench/executors/realtime.py`
- `latency_bench/executors/factory.py`
- `latency_bench/policy/base.py`
- `latency_bench/latency/measurement.py`
- `latency_bench/core/types.py`
- `tests/test_realtime_executor.py`
- `tests/test_latency_execution_semantics_contract.py`

**Work:**

- Introduce executor-owned `InferencePool` or equivalent worker-pool boundary.
- Add `max_in_flight`.
- Add explicit queue-full policies:
  - default `drop_new_observation`
  - optional `drop_oldest_pending`
- Add timeout state, skipped jobs, and late-result ignore handling.
- Avoid assuming CUDA streams provide hard isolation.

**Risks:**

- Threading and timing tests can be flaky if they rely on real sleeps.
- GPU measured latency needs careful synchronization boundaries.
- Cancellation of running inference must not be treated as guaranteed.

**Acceptance criteria:**

- Fake policy tests prove environment loop does not intentionally wait for inference.
- `max_in_flight` bounds concurrent jobs.
- Queue-full and timeout behavior are deterministic under fake clocks/workers.
- FIFO unblocks after timed-out/skipped jobs and ignores late completions.

**Independent:** Depends on Phase 2 for FIFO/skipped semantics; should follow Phase 1 for unit naming.

## Phase 5: Logging and validation

**Goal:** Make outputs sufficient to audit raw-frame and realtime behavior.

**Likely files:**

- `latency_bench/core/types.py`
- `latency_bench/logging/metrics.py`
- `latency_bench/logging/records.py`
- `latency_bench/logging/summary.py`
- `latency_bench/logging/trajectory_logger.py`
- `latency_bench/analysis/*.py`
- `tests/test_latency_execution_semantics_contract.py`
- Logging/summary tests as needed

**Work:**

- Standardize required action/job fields:
  - `obs_raw_frame`
  - `latency_raw_frames`
  - `ready_raw_frame`
  - `apply_raw_frame`
  - `action_age_raw_frames`
  - `job_status`
- Log realtime queue-full events, dropped observations, skipped/timed-out jobs, actual env FPS, deadline misses, in-flight counts, and queue sizes.
- Mark invalid or contended realtime runs when configured thresholds are exceeded.

**Risks:**

- Changing metric schemas can break downstream analysis scripts.
- Large logs can grow quickly for realtime runs.
- Backward compatibility is needed for existing result files.

**Acceptance criteria:**

- All executor modes write the common required fields.
- Realtime runs expose enough data to verify capacity, deadline, queue, and timeout behavior.
- Analysis scripts either consume the new fields or fail with clear compatibility messages.

**Independent:** Partially; some fields depend on Phase 3 and Phase 4.

## Phase 6: Cleanup

**Goal:** Remove or clearly deprecate legacy paths after behavior is aligned and tested.

**Likely files:**

- `latency_bench/experiments/run_simulated_latency.py`
- `latency_bench/envs/flappy.py`
- `latency_bench/executors/batched_simulated.py`
- `training/sample_factory_common/latency_wrapper.py`
- `scripts/bash_scripts/*.sh`
- Legacy config files under `configs/`
- Tests that preserve old semantics

**Work:**

- Merge duplicate Flappy adapter implementations.
- Deprecate or remove thin legacy entrypoints after a compatibility window.
- Consolidate duplicated eval shell scripts if still needed.
- Rename or isolate training-time decision-step latency wrappers if they remain semantically different.
- Remove old config aliases after migration.

**Risks:**

- Users may still rely on old CLI entrypoints or config names.
- Training scripts may intentionally use old decision-step latency semantics.
- Cleanup should not erase reproducibility for existing experiments.

**Acceptance criteria:**

- No duplicate adapter definitions remain.
- Legacy entrypoints/config names either have documented deprecation or are removed with migration notes.
- Tests no longer encode obsolete semantics except in explicit backward-compatibility tests.

**Independent:** No. Run after Phases 1-5 are stable.

## Open Decisions

1. Whether legacy config fields are supported through aliases for one release or migrated immediately.
2. Whether `hold_for_fixed_duration` defaults to `obs_stride_raw_frames`.
3. Realtime timeout defaults: disabled, configured per run, or derived from model latency percentiles.
4. Whether batched simulated should support every latency source or remain a fixed/zero-only optimization.
5. Whether Sample Factory training latency should migrate to raw-frame semantics or remain a clearly labeled training-time decision-step augmentation.
