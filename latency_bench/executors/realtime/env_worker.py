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
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.factory import build_action_scheduler
from latency_bench.policy.action_prefix import with_action_prefix


# Number of shared-memory slots pre-allocated per environment process. In-flight observations
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
) -> bool:
    while connection.poll():
        command, payload = connection.recv()
        payload = payload or {}
        if command == "close":
            return True
        if command == "release_observation":
            release_observation(payload["obs_id"])
            continue
        if command == "action_dispatch":
            action_event = _action_event_from_dispatch_payload(payload)
            action_scheduler.enqueue(action_event)
    return False


def _drain_env_actions_until_wall_time(
    *,
    connection: Any,
    action_scheduler: ActionScheduler,
    release_observation: Callable[[int], None],
    target_wall_s: float,
    time_fn: Callable[[], float] = time.perf_counter,
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
            ):
                return True
            continue
        return False


def _env_loop_worker(connection, config, video_output_dir, video_fps):
    """Construct, warm, and run the environment in its owning process."""
    # Native environment dependencies belong only in the environment interpreter.
    from latency_bench.core.actions import ActionResolver
    from latency_bench.envs.factory import build_env_adapter
    from latency_bench.executors.env_step_backend import env_action_space

    was_enabled = gc.isenabled()
    gc.disable()
    env = None
    observation_pool = None
    try:
        env = build_env_adapter(config, ActionResolver(config["env"]).noop_action)
        clock = EnvClock(env_fps=config["env"]["env_fps"], obs_fps=config["env"]["obs_fps"])
        scheduler = build_action_scheduler(config["scheduler"], clock=clock, noop_action=env.noop_action)
        action_prefix = config["policy"]["action_prefix"]
        env.reset(seed=None)
        observation_pool = ObservationSlotPool.from_observation(
            with_action_prefix(env.observe(), scheduler, action_prefix),
            capacity=DEFAULT_OBSERVATION_SLOT_POOL_CAPACITY,
        )
        connection.send(("ready", {
            "env_fps": env.env_fps, "noop_action": env.noop_action,
            "action_space": env_action_space(env), "env_config": config["env"],
        }))
        next_obs_id = 0
        while True:
            command, payload = connection.recv()
            if command == "close":
                return
            if command == "release_observation":
                observation_pool.release(payload["obs_id"])
            elif command == "reset":
                scheduler.reset()
                env.reset(seed=payload["seed"])
                connection.send(("reset", {}))
            elif command == "observe":
                observe_start_ms = monotonic_ms()
                observation = with_action_prefix(env.observe(), scheduler, action_prefix)
                shared = observation_pool.write(obs_id=next_obs_id, observation=observation)
                task = RealtimeObservationTask(
                    obs_id=next_obs_id, observation=shared,
                    submit_env_step=observation.env_step, submit_time_ms=0.0,
                    observation_start_time_ms=observe_start_ms,
                    shared_obs_id=next_obs_id,
                )
                next_obs_id += 1
                connection.send(("observation", {"task": task}))
            elif command == "step":
                result = env.step(payload["action"])
                connection.send(("step", {"result": result}))
            elif command == "episode":
                # The worker now survives resets; collect only outside paced time.
                gc.collect()
                if not _run_episode(
                    connection, config, env, clock, scheduler, observation_pool,
                    video_output_dir=video_output_dir, video_fps=video_fps, **payload,
                ):
                    return
    except BaseException:
        connection.send(("error", {"traceback": traceback.format_exc()}))
    finally:
        if observation_pool is not None:
            observation_pool.close()
        if env is not None:
            env.close()
        if was_enabled:
            gc.enable()


def _run_episode(connection, config, env, clock, action_scheduler, observation_pool,
                 *, episode_id, seed, max_env_steps, video_output_dir, video_fps):
    from latency_bench.logging.video import EpisodeVideoRecorder
    from latency_bench.scheduler.decision import ObservationCadenceDecisionScheduler

    action_scheduler.reset()
    env.reset(seed=seed)
    decision_scheduler = ObservationCadenceDecisionScheduler(clock)
    action_prefix = config["policy"]["action_prefix"]
    video_recorder = None
    if video_output_dir is not None:
        video_recorder = EpisodeVideoRecorder(video_output_dir, fps=video_fps)
        video_recorder.start_episode(episode_id=episode_id, metadata={})
        video_recorder.record_frame(env.render_game_frame())
    episode_start_s = time.perf_counter()
    connection.send(("episode_started", {"episode_start_s": episode_start_s}))
    next_obs_id = 0
    applied_event_count = 0
    release_shared_observation = observation_pool.release
    for env_step in range(int(max_env_steps)):
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
                return False
        wall_elapsed_ms = (time.perf_counter() - episode_start_s) * 1000.0
        wall_clock_lag_ms = max(0.0, wall_elapsed_ms - scheduled_time_ms)

        if _drain_env_action_messages(
            connection=connection,
            action_scheduler=action_scheduler,
            release_observation=release_shared_observation,
        ):
            return False

        if decision_scheduler.should_observe(env_step, scheduled_time_ms):
            observe_start_ms = monotonic_ms()
            observation = with_action_prefix(
                env.observe(), action_scheduler, action_prefix,
                current_time_ms=wall_elapsed_ms,
            )
            obs_id = next_obs_id
            next_obs_id += 1
            shared_observation = observation_pool.write(obs_id=obs_id, observation=observation)
            task = RealtimeObservationTask(
                obs_id=obs_id,
                observation=shared_observation,
                submit_env_step=env_step,
                submit_time_ms=wall_elapsed_ms,
                observation_start_time_ms=observe_start_ms,
                shared_obs_id=obs_id,
            )
            connection.send(("observation", {"task": task}))

        if _drain_env_action_messages(
            connection=connection,
            action_scheduler=action_scheduler,
            release_observation=release_shared_observation,
        ):
            return False

        # Completion timestamps use wall time even when a physical step overruns.
        active_action = action_scheduler.update(env_step, wall_elapsed_ms)
        new_applied_events = action_scheduler.get_applied_events_since(applied_event_count)

        result = env.step(active_action)
        for event in new_applied_events:
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
    # The parent drains every admitted request before acknowledging the boundary.
    while True:
        command, payload = connection.recv()
        if command == "release_observation":
            observation_pool.release(payload["obs_id"])
        elif command == "finish_episode":
            gc.collect()
            connection.send(("episode_finished", {}))
            return True
        elif command == "close":
            return False
