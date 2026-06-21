"""Everything that lives at / behind the realtime process boundary.

This module holds the IPC message protocol (dataclasses + their (de)serialization)
shared between the parent and its child processes, plus the two subprocess
"programs" themselves: the inference worker and the env-loop worker. The parent
side (pool, coordinator, executor) lives in sibling modules.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import (
    Action,
    ActionEvent,
    ActionHoldPolicy,
    LatencyRecord,
    Observation,
    OrderingPolicy,
    PolicyOutput,
    StepRecord,
)
from latency_bench.latency.hardware_info import safe_set, torch_backend_flags
from latency_bench.latency.measurement import RuntimeLatencyMeasurer
from latency_bench.logging.metrics import active_action_step_info
from latency_bench.logging.records import step_record_info
from latency_bench.logging.realtime_pipeline import PipelineProfile, REALTIME_PROFILE_METADATA_KEY
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.persistence import HoldPolicy, OneFrameThenNoopPolicy
from latency_bench.executors.realtime.shared_observation import (
    ObservationAttachmentCache,
    ObservationLocalizer,
    ObservationSlotPool,
)


# Number of shared-memory slots pre-allocated per episode. In-flight observations
# (written but not yet released) ~= inference_latency / obs_stride; should_observe
# applies no backpressure, so this must stay well above the worst-case in-flight
# count. 64 is generous for realistic latency distributions.
DEFAULT_OBSERVATION_SLOT_POOL_CAPACITY = 64


def _profile_now_ms() -> float:
    return time.perf_counter() * 1000.0


# --- IPC message types --------------------------------------------------------

@dataclass
class RealtimeObservationTask:
    """
    env -> worker
    """
    obs_id: int
    observation: Observation
    submit_env_step: int
    submit_time_ms: float
    profile: PipelineProfile | None = None
    shared_obs_id: int | None = None


@dataclass
class RealtimePolicyCompletion:
    """
    results returned by workers
    """
    task: RealtimeObservationTask
    policy_output: PolicyOutput
    completion_time_ms: float
    latency_record: LatencyRecord
    worker_id: int
    device: str
    profile: PipelineProfile | None = None
    error: BaseException | None = None


@dataclass
class RealtimePolicyResult:
    """
    slim worker -> parent result; parent reattaches the observation task.
    """
    policy_output: PolicyOutput
    completion_time_ms: float
    latency_record: LatencyRecord
    profile: PipelineProfile | None = None
    error: BaseException | None = None


@dataclass
class RealtimeActionDispatch:
    action_id: int
    action: Action
    obs_env_step: int
    ready_time_ms: float
    ready_step_float: float
    latency_raw_frames: int | None = None
    ready_raw_frame: int | None = None
    apply_raw_frame: int | None = None
    applied: bool = False
    dropped: bool = False
    profile: PipelineProfile | None = None


def _action_event_dispatch_payload(event: ActionEvent) -> dict[str, Any]:
    payload = {
        "action_id": event.action_id,
        "action": event.action,
        "obs_env_step": event.obs_env_step,
        "ready_time_ms": event.ready_time_ms,
        "ready_step_float": event.ready_step_float,
        "latency_raw_frames": event.latency_raw_frames,
        "ready_raw_frame": event.ready_raw_frame,
    }
    profile = event.metadata.get(REALTIME_PROFILE_METADATA_KEY)
    if profile is not None:
        payload["profile"] = profile.to_dict()
    return payload


def _action_event_from_dispatch_payload(payload: dict[str, Any]) -> RealtimeActionDispatch:
    profile = PipelineProfile.from_dict(payload["profile"]) if "profile" in payload else None
    return RealtimeActionDispatch(
        action_id=payload["action_id"],
        action=payload["action"],
        obs_env_step=payload["obs_env_step"],
        ready_time_ms=payload["ready_time_ms"],
        ready_step_float=payload["ready_step_float"],
        latency_raw_frames=payload["latency_raw_frames"],
        ready_raw_frame=payload["ready_raw_frame"],
        profile=profile,
    )


def _action_event_ack_payload(event: RealtimeActionDispatch) -> dict[str, Any]:
    payload = {
        "action_id": event.action_id,
        "applied": event.applied,
        "apply_raw_frame": event.apply_raw_frame,
    }
    if event.profile is not None:
        payload["profile"] = event.profile.to_dict()
    return payload


# --- inference worker ---------------------------------------------------------

def _worker_runtime_metadata(worker_id: int, start_method: str) -> dict[str, Any]:
    import torch

    metadata = {
        "worker_id": int(worker_id),
        "worker_pid": os.getpid(),
        "worker_start_method": start_method,
        "worker_python_executable": sys.executable,
        "worker_torch_version": torch.__version__,
        "worker_torch_cuda_version": torch.version.cuda,
        "worker_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "worker_cuda_device_count": torch.cuda.device_count(),
    }
    safe_set(metadata, "worker_cuda_current_device", lambda: int(torch.cuda.current_device()))
    safe_set(
        metadata,
        "worker_cuda_current_device_name",
        lambda: str(torch.cuda.get_device_name(torch.cuda.current_device())),
    )
    backend_flags = torch_backend_flags(torch)
    for flag_key in ("cuda_flash_sdp_enabled", "cuda_mem_efficient_sdp_enabled", "cuda_math_sdp_enabled"):
        if flag_key in backend_flags:
            metadata[f"worker_{flag_key}"] = backend_flags[flag_key]
    safe_set(metadata, "worker_torch_float32_matmul_precision", torch.get_float32_matmul_precision)
    return metadata


def _inference_worker(connection: Any, config: dict[str, Any], worker_id: int, device: str, start_method: str) -> None:
    try:
        from latency_bench.core.actions import ActionResolver
        from latency_bench.policy.factory import build_policy

        runtime_metadata = _worker_runtime_metadata(worker_id, start_method)
        worker_config = deepcopy(config)
        worker_config.setdefault("policy", {})["device"] = device
        worker_config.setdefault("latency", {})["device"] = device
        action_resolver = ActionResolver(worker_config["env"])
        policy = build_policy(worker_config, action_resolver)
        latency_cfg = worker_config.get("latency", {})
        measurer = RuntimeLatencyMeasurer(
            device=device,
            sync_cuda=bool(latency_cfg.get("sync_cuda", True)),
        )
        attachment_cache = ObservationAttachmentCache(blocks={})
        observation_localizer = ObservationLocalizer()
        connection.send(("ready", {"worker_id": int(worker_id), "device": device}))

        def predict_from_attached_observation(attached_observation, profile):
            localize_start_ms = _profile_now_ms()
            local_observation = observation_localizer.localize(attached_observation.observation)
            if profile is not None:
                profile.worker_observation_localize_ms = _profile_now_ms() - localize_start_ms
            if hasattr(policy, "predict_with_timing"):
                policy_output, timings = policy.predict_with_timing(local_observation)
                if profile is not None:
                    profile.policy_build_example_ms = timings["policy_build_example_ms"]
                    profile.policy_wrapper_predict_action_ms = timings["policy_wrapper_predict_action_ms"]
                    profile.policy_decode_ms = timings["policy_decode_ms"]
                return policy_output
            predict_start_ms = _profile_now_ms()
            policy_output = policy.predict(local_observation)
            if profile is not None:
                profile.policy_build_example_ms = 0.0
                profile.policy_wrapper_predict_action_ms = _profile_now_ms() - predict_start_ms
                profile.policy_decode_ms = 0.0
            return policy_output

        while True:
            command, payload = connection.recv()
            worker_recv_ms = _profile_now_ms()
            payload = payload or {}
            if command == "close":
                attachment_cache.close()
                return
            if command == "predict":
                task = payload["task"]
                attach_start_ms = _profile_now_ms()
                attached_observation = attachment_cache.attach(task.observation)
                attach_ms = _profile_now_ms() - attach_start_ms
                profile = task.profile
                if profile is not None:
                    profile.worker_recv_ms = worker_recv_ms
                    profile.parent_to_worker_ipc_ms = (
                        worker_recv_ms - profile.parent_to_worker_send_start_ms
                    )
                    inference_start_ms = _profile_now_ms()
                    profile.worker_observation_attach_ms = attach_ms
                    profile.worker_recv_overhead_ms = inference_start_ms - worker_recv_ms
                try:
                    policy_output, latency_record = measurer.run_and_measure(
                        predict_from_attached_observation,
                        attached_observation,
                        profile,
                        obs_id=task.obs_id,
                    )
                finally:
                    attached_observation.close()
                latency_record.metadata = {
                    **latency_record.metadata,
                    **runtime_metadata,
                }
                result = RealtimePolicyResult(
                    policy_output=policy_output,
                    completion_time_ms=task.submit_time_ms + latency_record.latency_ms,
                    latency_record=latency_record,
                    profile=profile,
                )
                try:
                    if result.profile is not None:
                        result.profile.worker_result_send_start_ms = _profile_now_ms()
                    connection.send(("result", {"result": result}))
                except (BrokenPipeError, EOFError, OSError):
                    return
    except BaseException:
        try:
            connection.send(("error", {"traceback": traceback.format_exc()}))
        except (BrokenPipeError, EOFError, OSError):
            return


# --- env-loop worker ----------------------------------------------------------

def _drain_env_action_messages(
    *,
    connection: Any,
    action_scheduler: ActionScheduler,
    release_observation: Callable[[int], None],
    now_ms_fn: Callable[[], float] = _profile_now_ms,
) -> bool:
    while connection.poll():
        recv_start_ms = now_ms_fn()
        command, payload = connection.recv()
        recv_end_ms = now_ms_fn()
        payload = payload or {}
        if command == "close":
            return True
        if command == "release_observation":
            release_observation(payload["obs_id"])
            continue
        if command == "action_dispatch":
            action_event = _action_event_from_dispatch_payload(payload)
            if action_event.profile is not None:
                action_event.profile.parent_to_env_ipc_ms = (
                    recv_end_ms - action_event.profile.parent_to_env_send_start_ms
                )
            action_scheduler.enqueue(action_event)
            if action_event.profile is not None:
                action_event.profile.env_action_recv_to_enqueue_ms = now_ms_fn() - recv_start_ms
    return False


def _drain_env_actions_until_wall_time(
    *,
    connection: Any,
    action_scheduler: ActionScheduler,
    release_observation: Callable[[int], None],
    target_wall_s: float,
    time_fn: Callable[[], float] = time.perf_counter,
    now_ms_fn: Callable[[], float] = _profile_now_ms,
) -> bool:
    while True:
        remaining_s = target_wall_s - time_fn()
        if remaining_s <= 0.0:
            return False
        if connection.poll(remaining_s):
            if _drain_env_action_messages(
                connection=connection,
                action_scheduler=action_scheduler,
                release_observation=release_observation,
                now_ms_fn=now_ms_fn,
            ):
                return True
            continue
        return False


def _build_realtime_persistence_policy(scheduler_cfg: dict[str, Any], clock: EnvClock):
    mode = ActionHoldPolicy(str(scheduler_cfg.get("hold_policy", ActionHoldPolicy.ONE_FRAME_THEN_NOOP.value)))
    if mode == ActionHoldPolicy.HOLD:
        return HoldPolicy(duration_raw_frames=clock.obs_stride_raw_frames)
    return OneFrameThenNoopPolicy()


def _env_loop_worker(
    connection: Any,
    config: dict[str, Any] | None,
    episode_id: int,
    seed: int | None,
    eval_max_steps: int,
    video_output_dir: str | None,
    video_fps: float | None,
) -> None:
    try:
        from latency_bench.core.actions import ActionResolver
        from latency_bench.envs.factory import build_env_adapter
        from latency_bench.logging.video import EpisodeVideoRecorder
        from latency_bench.scheduler.decision import ObservationCadenceDecisionScheduler

        runtime_config = config or {}
        profile_enabled = bool(runtime_config.get("logging", {}).get("realtime_pipeline_profile", False))
        action_resolver = ActionResolver(runtime_config["env"])
        env = build_env_adapter(runtime_config, action_resolver.noop_action)
        clock = EnvClock(
            env_fps=float(runtime_config["env"]["env_fps"]),
            obs_fps=float(runtime_config["env"]["obs_fps"]),
        )
        action_scheduler = ActionScheduler(
            clock=clock,
            noop_action=env.noop_action,
            persistence_policy=_build_realtime_persistence_policy(runtime_config.get("scheduler", {}), clock),
            ordering_policy=OrderingPolicy(str(runtime_config.get("scheduler", {}).get("ordering_policy", "issue_order_fifo"))),
        )
        decision_scheduler = ObservationCadenceDecisionScheduler(clock)
        env.reset(seed=seed)
        video_recorder = None
        if video_output_dir is not None:
            video_recorder = EpisodeVideoRecorder(video_output_dir, fps=video_fps)
            video_recorder.start_episode(episode_id=episode_id, metadata={})
            video_recorder.record_frame(env.render_game_frame())
        episode_start_s = time.perf_counter()
        next_obs_id = 0
        applied_event_count = 0
        observation_pool: ObservationSlotPool | None = None

        def release_shared_observation(obs_id: int) -> None:
            observation_pool.release(obs_id)

        def release_all_shared_observations() -> None:
            if observation_pool is not None:
                for obs_id in list(observation_pool.obs_slots):
                    release_shared_observation(obs_id)

        def close_env_and_pool() -> None:
            if observation_pool is not None:
                observation_pool.close()
            env.close()

        for env_step in range(int(eval_max_steps)):
            scheduled_time_ms = clock.step_to_time_ms(env_step)
            target_wall_s = episode_start_s + scheduled_time_ms / 1000.0
            now_s = time.perf_counter()
            if now_s < target_wall_s:
                if _drain_env_actions_until_wall_time(
                    connection=connection,
                    action_scheduler=action_scheduler,
                    release_observation=release_shared_observation,
                    target_wall_s=target_wall_s,
                ):
                    close_env_and_pool()
                    return
            wall_elapsed_ms = (time.perf_counter() - episode_start_s) * 1000.0
            wall_clock_lag_ms = max(0.0, wall_elapsed_ms - scheduled_time_ms)

            if _drain_env_action_messages(
                connection=connection,
                action_scheduler=action_scheduler,
                release_observation=release_shared_observation,
            ):
                close_env_and_pool()
                return

            if decision_scheduler.should_observe(env_step, scheduled_time_ms):
                observe_start_ms = _profile_now_ms()
                observation = env.observe()
                observe_end_ms = _profile_now_ms()
                obs_id = next_obs_id
                next_obs_id += 1
                if observation_pool is None:
                    observation_pool = ObservationSlotPool.from_observation(
                        observation,
                        capacity=DEFAULT_OBSERVATION_SLOT_POOL_CAPACITY,
                    )
                shared_observation = observation_pool.write(obs_id=obs_id, observation=observation)
                profile = None
                if profile_enabled:
                    profile = PipelineProfile(
                        env_observe_ms=observe_end_ms - observe_start_ms,
                        env_wall_lag_ms=wall_clock_lag_ms,
                    )
                task = RealtimeObservationTask(
                    obs_id=obs_id,
                    observation=shared_observation,
                    submit_env_step=env_step,
                    submit_time_ms=wall_elapsed_ms,
                    profile=profile,
                    shared_obs_id=obs_id,
                )
                if profile is not None:
                    profile.env_observation_send_start_ms = _profile_now_ms()
                connection.send(("observation", {"task": task}))

            if _drain_env_action_messages(
                connection=connection,
                action_scheduler=action_scheduler,
                release_observation=release_shared_observation,
            ):
                close_env_and_pool()
                return

            scheduler_update_start_ms = _profile_now_ms()
            active_action = action_scheduler.update(env_step, scheduled_time_ms)
            scheduler_update_ms = _profile_now_ms() - scheduler_update_start_ms
            new_applied_events = action_scheduler.get_applied_events_since(applied_event_count)
            for event in new_applied_events:
                if event.profile is not None:
                    event.profile.env_scheduler_update_ms = scheduler_update_ms

            env_step_start_ms = _profile_now_ms()
            result = env.step(active_action)
            env_step_ms = _profile_now_ms() - env_step_start_ms
            for event in new_applied_events:
                if event.profile is not None:
                    event.profile.env_step_ms = env_step_ms
                connection.send(("action_ack", _action_event_ack_payload(event)))
            applied_event_count = action_scheduler.applied_event_count
            episode_done = bool(result.done or result.truncated)
            if video_recorder is not None and not episode_done:
                video_recorder.record_frame(env.render_game_frame())
            step_info = step_record_info(result.info)
            step_info.update(
                {
                    "scheduled_time_ms": scheduled_time_ms,
                    "wall_elapsed_ms": wall_elapsed_ms,
                    "frame_time_ms": clock.frame_ms,
                    "wall_clock_lag_ms": wall_clock_lag_ms,
                    "observation_submitted": False,
                    "observation_dropped": False,
                    "in_flight_count": None,
                    "idle_worker_count": None,
                }
            )
            step_info.update(
                active_action_step_info(
                    active_event=action_scheduler.latest_applied_event,
                    env_step=env_step,
                    scheduled_time_ms=scheduled_time_ms,
                    frame_ms=clock.frame_ms,
                    latency_type="measured",
                    profile_metadata={},
                )
            )
            connection.send((
                "step",
                {
                    "record": StepRecord(
                        episode_id=episode_id,
                        env_step=env_step,
                        sim_time_ms=scheduled_time_ms,
                        active_action=active_action,
                        reward=float(result.reward),
                        done=episode_done,
                        info=step_info,
                    )
                },
            ))
            if episode_done:
                break

        if video_recorder is not None:
            video_recorder.close_episode({})
        connection.send(("episode_done", {}))
        while True:
            if _drain_env_action_messages(
                connection=connection,
                action_scheduler=action_scheduler,
                release_observation=release_shared_observation,
            ):
                release_all_shared_observations()
                close_env_and_pool()
                return
            time.sleep(0.001)
    except BaseException:
        error_traceback = traceback.format_exc()
        # Best-effort: unlink the slot pool's shared memory so a crash (e.g. pool
        # exhaustion) does not leak it. close_env_and_pool only exists if we got
        # past env construction; it already no-ops when the pool was never created.
        cleanup = locals().get("close_env_and_pool")
        if cleanup is not None:
            try:
                cleanup()
            except BaseException:
                pass
        connection.send(("error", {"traceback": error_traceback}))
