"""Realtime environment-loop subprocess program."""
from __future__ import annotations

import gc
import time
import traceback
from collections.abc import Callable
from typing import Any

from latency_bench.core.clock import EnvClock
from latency_bench.executors.realtime.messages import (
    RealtimeObservationTask,
    _action_event_ack_payload,
    _action_event_from_dispatch_payload,
)
from latency_bench.executors.realtime.shared_observation import ObservationSlotPool
from latency_bench.executors.realtime.timing import monotonic_ms
from latency_bench.logging.records import build_step_record
from latency_bench.logging.realtime_pipeline import PipelineProfile
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.factory import build_action_scheduler


# Number of shared-memory slots pre-allocated per episode. In-flight observations
# (written but not yet released) ~= inference_latency / obs_stride; should_observe
# applies no backpressure, so this must stay well above the worst-case in-flight
# count. 64 is generous for realistic latency distributions.
DEFAULT_OBSERVATION_SLOT_POOL_CAPACITY = 64


# --- env-loop worker ----------------------------------------------------------

def _drain_env_action_messages(
    *,
    connection: Any,
    action_scheduler: ActionScheduler,
    release_observation: Callable[[int], None],
    now_ms_fn: Callable[[], float] = monotonic_ms,
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
    now_ms_fn: Callable[[], float] = monotonic_ms,
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


def _env_loop_worker(
    connection: Any,
    config: dict[str, Any] | None,
    episode_id: int,
    seed: int | None,
    eval_max_steps: int,
    video_output_dir: str | None,
    video_fps: float | None,
) -> None:
    gc_was_enabled = gc.isenabled()
    try:
        from latency_bench.core.actions import ActionResolver
        from latency_bench.envs.factory import build_env_adapter
        from latency_bench.logging.video import EpisodeVideoRecorder
        from latency_bench.scheduler.decision import ObservationCadenceDecisionScheduler

        # This process lives for exactly one episode; a GC pause here misses
        # frame deadlines, and the heap is reclaimed at process exit anyway.
        gc.disable()

        runtime_config = config or {}
        profile_enabled = bool(runtime_config.get("logging", {}).get("realtime_pipeline_profile", False))
        action_resolver = ActionResolver(runtime_config["env"])
        env = build_env_adapter(runtime_config, action_resolver.noop_action)
        clock = EnvClock(
            env_fps=float(runtime_config["env"]["env_fps"]),
            obs_fps=float(runtime_config["env"]["obs_fps"]),
        )
        action_scheduler = build_action_scheduler(
            runtime_config["scheduler"],
            clock=clock,
            noop_action=env.noop_action,
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
                observe_start_ms = monotonic_ms()
                observation = env.observe()
                observe_end_ms = monotonic_ms()
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
                    observation_start_time_ms=observe_start_ms,
                    profile=profile,
                    shared_obs_id=obs_id,
                )
                if profile is not None:
                    profile.env_observation_send_start_ms = monotonic_ms()
                connection.send(("observation", {"task": task}))

            if _drain_env_action_messages(
                connection=connection,
                action_scheduler=action_scheduler,
                release_observation=release_shared_observation,
            ):
                close_env_and_pool()
                return

            scheduler_update_start_ms = monotonic_ms()
            active_action = action_scheduler.update(env_step, scheduled_time_ms)
            scheduler_update_ms = monotonic_ms() - scheduler_update_start_ms
            new_applied_events = action_scheduler.get_applied_events_since(applied_event_count)
            for event in new_applied_events:
                if event.profile is not None:
                    event.profile.env_scheduler_update_ms = scheduler_update_ms

            env_step_start_ms = monotonic_ms()
            result = env.step(active_action)
            env_step_ms = monotonic_ms() - env_step_start_ms
            for event in new_applied_events:
                if event.profile is not None:
                    event.profile.env_step_ms = env_step_ms
                connection.send(("action_ack", _action_event_ack_payload(event)))
            applied_event_count = action_scheduler.applied_event_count
            episode_done = bool(result.done or result.truncated)
            if video_recorder is not None and not episode_done:
                video_recorder.record_frame(env.render_game_frame())
            connection.send((
                "step",
                {
                    "record": build_step_record(
                        episode_id=episode_id,
                        env_step=env_step,
                        scheduled_time_ms=scheduled_time_ms,
                        active_action=active_action,
                        reward=result.reward,
                        done=episode_done,
                        info=result.info,
                        active_event=action_scheduler.latest_applied_event,
                        frame_ms=clock.frame_ms,
                        latency_type="measured",
                        pre_active_info={
                            "scheduled_time_ms": scheduled_time_ms,
                            "wall_elapsed_ms": wall_elapsed_ms,
                            "frame_time_ms": clock.frame_ms,
                            "wall_clock_lag_ms": wall_clock_lag_ms,
                            "observation_submitted": False,
                            "observation_dropped": False,
                            "in_flight_count": None,
                            "idle_worker_count": None,
                        },
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
    finally:
        # The real env process exits with this function; restoring GC state
        # only matters for in-process callers (tests).
        if gc_was_enabled:
            gc.enable()
