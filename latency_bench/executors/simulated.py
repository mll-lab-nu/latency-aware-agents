from __future__ import annotations

import time
from collections import deque
from copy import deepcopy
from collections.abc import Callable
from dataclasses import dataclass

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import (
    ActionEvent,
    EpisodeMetrics,
    LatencyRecord,
    Observation,
    PolicyOutput,
    StepRecord,
)
from latency_bench.envs.base import EnvAdapter, collection_frame
from latency_bench.latency.sample import LatencySample
from latency_bench.scheduler.simulated_timeline import (
    SimulatedResultTimeline,
    SimulatedWorkerCapacity,
)
from latency_bench.executors._simulated_events import build_simulated_action_event
from latency_bench.executors.base import CollectionObserver, EpisodeExecutor
from latency_bench.executors.realtime.messages import RealtimeObservationTask
from latency_bench.latency.samplers import LatencySampler
from latency_bench.logging.metrics import (
    compute_episode_metrics,
    episode_raw_fact_metadata,
    latency_type_from_source,
    profile_metadata_from_source,
)
from latency_bench.logging.records import build_step_record
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.policy.action_prefix import with_action_prefix
from latency_bench.policy.base import PolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import DecisionScheduler


def _latency_policy_metadata(latency_sample: LatencySample) -> dict[str, float | int]:
    return {
        "latency_ms": float(latency_sample.latency_ms),
        "latency_raw_frames": int(latency_sample.action_ready_raw_frames),
        "worker_service_ms": float(latency_sample.worker_service_ms),
        "worker_service_raw_frames": int(latency_sample.worker_service_raw_frames),
    }


@dataclass
class _PendingInference:
    """An observation whose forward pass is still on an inference worker.

    Everything virtual is fixed at submission (the latency sample decides all
    timestamps); only the action value is outstanding, so wall-clock result
    order cannot leave a trace in the outputs.
    """

    obs_id: int
    action_id: int
    raw_frame: int
    ready_raw_frame: int
    ready_time_ms: float
    latency_sample: LatencySample
    worker_slot: int


class SimulatedLatencyExecutor(EpisodeExecutor):
    """Virtual-clock executor; sampled latency decides every timestamp.

    With ``inference_pool`` set, forward passes run batch-1 on per-GPU worker
    processes (the same pool the realtime executor uses) while the episode
    stream stays serial — mirroring how a measure run's machine serves one
    env with several inference workers. Policy state lives in the workers and
    is reset there at each episode boundary.
    """

    collection_observer: CollectionObserver | None = None

    def __init__(
        self,
        env: EnvAdapter,
        policy: PolicyRunner | None,
        decision_scheduler: DecisionScheduler,
        latency_source: LatencySampler,
        action_scheduler: ActionScheduler,
        clock: EnvClock,
        logger: TrajectoryLogger | None = None,
        video_recorder=None,
        video_episode_metadata: dict[int, dict] | None = None,
        episode_latency_source_factory: Callable[[int], LatencySampler] | None = None,
        simulated_worker_capacity: int | None = None,
        inference_pool=None,
        action_prefix=None,
        harness=None,
    ):
        self.env = env
        self.policy = policy
        self.decision_scheduler = decision_scheduler
        self.latency_source = latency_source
        self.action_scheduler = action_scheduler
        self.clock = clock
        self.logger = logger
        self.video_recorder = video_recorder
        self.video_episode_metadata = video_episode_metadata
        self.episode_latency_source_factory = episode_latency_source_factory
        self.simulated_worker_capacity = simulated_worker_capacity
        self.inference_pool = inference_pool
        self.action_prefix = action_prefix
        self.harness = harness
        self._next_obs_id = 0
        self._next_action_id = 0

    def close(self) -> None:
        try:
            if self.inference_pool is not None:
                self.inference_pool.close()
            else:
                self.policy.close()
        finally:
            if self.harness is not None:
                self.harness.close()
            super().close()

    def run_episode(
        self,
        episode_id: int,
        seed: int | None = None,
        max_env_steps: int = 10000,
    ) -> EpisodeMetrics:
        if self.episode_latency_source_factory is not None:
            self.latency_source = self.episode_latency_source_factory(episode_id)
        self.action_scheduler.reset()
        if self.inference_pool is not None:
            self.inference_pool.drain_until_idle()
            self.inference_pool.reset_state(0)
        else:
            self.policy.reset_state(slot_id=0)
        self.env.reset(seed=seed)
        if self.harness is not None:
            self.harness.start_episode(episode_id)
        if self.collection_observer is not None:
            self.collection_observer.start_episode(episode_id, seed, 0)
        if self.video_recorder is not None:
            self.video_recorder.start_episode(
                episode_id=episode_id,
                metadata=self.video_episode_metadata[episode_id],
            )
            self.video_recorder.record_frame(self.env.render_game_frame())

        step_records: list[StepRecord] = []
        action_events: list[ActionEvent] = []
        latency_values_ms: list[float] = []
        result_timeline = SimulatedResultTimeline(ordering_policy=self.action_scheduler.ordering_policy)
        worker_capacity = SimulatedWorkerCapacity(
            capacity=self.simulated_worker_capacity,
            busy_until_by_worker={},
        )
        submitted_observation_frames = 0
        dropped_observation_count = 0
        latency_type = latency_type_from_source(self.latency_source)
        pool = self.inference_pool
        pending_inferences: deque[_PendingInference] = deque()
        pending_results: dict[int, PolicyOutput] = {}
        submit_queue: deque[RealtimeObservationTask] = deque()
        pooled_latency_records: list[LatencyRecord] = []

        def release_events(frame):
            released, discarded = result_timeline.release_arrived(frame)
            if self.harness is not None:
                for event in sorted(released + discarded, key=lambda event: (event.ready_raw_frame, event.latency.obs_id)):
                    self.harness.returned(event, frame)
            for event in released:
                self.action_scheduler.enqueue(event)
            return bool(released)

        def fuse_ready_results() -> None:
            # A completed newer observation must enter the timeline immediately;
            # waiting for the oldest forward pass would defeat latest_ready.
            while pending_results:
                obs_id, policy_output = pending_results.popitem()
                if self.collection_observer is not None:
                    self.collection_observer.output(episode_id, obs_id=obs_id, output=policy_output)
                item = next(
                    pending_item
                    for pending_item in pending_inferences
                    if pending_item.obs_id == obs_id
                )
                pending_inferences.remove(item)
                latency_record, event = build_simulated_action_event(
                    action_id=item.action_id,
                    obs_id=item.obs_id,
                    policy_output=policy_output,
                    raw_frame=item.raw_frame,
                    ready_raw_frame=item.ready_raw_frame,
                    ready_time_ms=item.ready_time_ms,
                    latency_sample=item.latency_sample,
                    frame_ms=self.clock.frame_ms,
                    latency_type=latency_type,
                    profile_metadata=profile_metadata_from_source(self.latency_source),
                    latency_record_metadata={"worker_id": int(item.worker_slot)},
                )
                result_timeline.attach_event(
                    obs_id=item.obs_id, ready_raw_frame=item.ready_raw_frame, event=event
                )
                action_events.append(event)
                latency_values_ms.append(item.latency_sample.latency_ms)
                if self.logger is not None:
                    pooled_latency_records.append(latency_record)

        def wait_for_events_due(due_raw_frame: int | None) -> None:
            # Virtual time out-waits the wall clock: nothing advances past a
            # ready frame whose action value is still on a worker, so a slow
            # forward pass changes when we wait, never what we compute.
            while True:
                for completion in pool.drain_completed():
                    pending_results[completion.task.obs_id] = completion.policy_output
                while submit_queue and pool.submit(submit_queue[0]):
                    submit_queue.popleft()
                fuse_ready_results()
                if not pending_inferences:
                    return
                if due_raw_frame is not None and all(
                    item.ready_raw_frame > due_raw_frame for item in pending_inferences
                ):
                    return
                time.sleep(0.0005)
        episode_metadata = episode_raw_fact_metadata(
            mode="simulated",
            episode_seed=seed,
            env_fps=self.clock.env_fps,
            obs_fps=self.clock.obs_fps,
            frame_ms=self.clock.frame_ms,
            latency_type=latency_type,
            latency_source=self.latency_source,
        )

        raw_frame = 0
        recent_drop_count = 0
        completed = False
        try:
            while raw_frame < max_env_steps:
                current_time_ms = self.clock.step_to_time_ms(raw_frame)
                worker_capacity.release_ready(raw_frame)
                if pool is not None:
                    wait_for_events_due(raw_frame)
                release_events(raw_frame)
                if self.harness is not None:
                    active_action = self.action_scheduler.update(raw_frame, current_time_ms)
                    self.harness.accepted(self.action_scheduler.latest_applied_event, raw_frame, active_action)
                observation_submitted = False
                observation_dropped = False
                if self.decision_scheduler.should_observe(raw_frame, current_time_ms):
                    observation = self.env.observe()
                    obs_id = self._next_obs_id
                    self._next_obs_id += 1
                    if self.harness is not None:
                        self.harness.observe(obs_id, raw_frame, self.env.render_game_frame())
                    request_allowed = self.harness is None or self.harness.should_request(obs_id, raw_frame)
                    prefix_request_pending = (
                        self.action_prefix is not None
                        and self.action_prefix["mode"] != "none"
                        and result_timeline.pending_observation_count > 0
                    )
                    if request_allowed and worker_capacity.can_submit() and not prefix_request_pending:
                        worker_slot = worker_capacity.assign_worker()
                        latency_context = {
                            "observation": observation,
                            "obs_id": obs_id,
                            "env_step": raw_frame,
                            "raw_frame": raw_frame,
                            "sim_time_ms": current_time_ms,
                            "episode_id": episode_id,
                            "slot_id": 0,
                            "worker_slot": worker_slot,
                            "recent_drop_count": recent_drop_count,
                            "in_flight_count": worker_capacity.in_flight_count,
                            "idle_worker_count": worker_capacity.idle_worker_count,
                        }
                        latency_sample = self.latency_source.sample(latency_context)
                        latency_ms = latency_sample.latency_ms
                        observation = with_action_prefix(
                            observation, self.action_scheduler, self.action_prefix
                        )
                        policy_metadata = dict(observation.metadata)
                        policy_metadata.update(
                            {
                                "obs_id": obs_id,
                                "episode_id": episode_id,
                                "action_noise_seed": seed,
                                "slot_id": 0,
                                **_latency_policy_metadata(latency_sample),
                            }
                        )
                        policy_observation = Observation(
                            data=observation.data,
                            env_step=observation.env_step,
                            sim_time_ms=observation.sim_time_ms,
                            metadata=policy_metadata,
                        )
                        if self.harness is not None:
                            policy_observation = self.harness.request(obs_id, raw_frame)
                        ready_raw_frame = raw_frame + latency_sample.action_ready_raw_frames
                        ready_time_ms = self.clock.step_to_time_ms(ready_raw_frame)
                        worker_capacity.submit(worker_slot, raw_frame + latency_sample.worker_service_raw_frames)
                        action_id = self._next_action_id
                        self._next_action_id += 1
                        if self.collection_observer is not None:
                            self.collection_observer.observation(
                                episode_id, obs_id=obs_id, observation=policy_observation,
                                image=collection_frame(self.env, policy_observation)
                                if self.collection_observer.capture_rgb else None,
                            )
                        if pool is None:
                            policy_output = self.policy.predict(policy_observation)
                            if self.collection_observer is not None:
                                self.collection_observer.output(episode_id, obs_id=obs_id, output=policy_output)
                            latency_record, event = build_simulated_action_event(
                                action_id=action_id,
                                obs_id=obs_id,
                                policy_output=policy_output,
                                raw_frame=raw_frame,
                                ready_raw_frame=ready_raw_frame,
                                ready_time_ms=ready_time_ms,
                                latency_sample=latency_sample,
                                frame_ms=self.clock.frame_ms,
                                latency_type=latency_type,
                                profile_metadata=profile_metadata_from_source(self.latency_source),
                                latency_record_metadata={"worker_id": int(worker_slot)},
                            )
                            result_timeline.submit(obs_id=obs_id, ready_raw_frame=ready_raw_frame, event=event)
                            action_events.append(event)
                            latency_values_ms.append(latency_ms)
                            if self.logger is not None:
                                self.logger.log_latency(latency_record)
                        else:
                            # The observation must be on the coordinator's books at
                            # its submission frame so a newer release can void it
                            # while its forward pass is still in flight.
                            result_timeline.mark_submitted(obs_id=obs_id)
                            task = RealtimeObservationTask(
                                obs_id=obs_id,
                                # A queued request can outlive the environment's reusable buffers.
                                observation=deepcopy(policy_observation),
                                submit_env_step=raw_frame,
                                submit_time_ms=current_time_ms,
                                observation_start_time_ms=time.perf_counter() * 1000.0,
                            )
                            if not pool.submit(task):
                                submit_queue.append(task)
                            pending_inferences.append(
                                _PendingInference(
                                    obs_id=obs_id,
                                    action_id=action_id,
                                    raw_frame=raw_frame,
                                    ready_raw_frame=ready_raw_frame,
                                    ready_time_ms=ready_time_ms,
                                    latency_sample=latency_sample,
                                    worker_slot=worker_slot,
                                )
                            )
                        submitted_observation_frames += 1
                        observation_submitted = True
                        recent_drop_count = 0
                    elif request_allowed:
                        dropped_observation_count += 1
                        observation_dropped = True
                        recent_drop_count += 1
                        if self.harness is not None:
                            self.harness.record("observation_dropped", raw_frame, obs_id=obs_id, reason="virtual_capacity")

                if pool is not None:
                    wait_for_events_due(raw_frame)
                newly_released = release_events(raw_frame)
                if self.harness is None or newly_released:
                    active_action = self.action_scheduler.update(raw_frame, current_time_ms)
                result = self.env.step(active_action)
                if self.harness is not None:
                    self.harness.executed_action(raw_frame, active_action, self.action_scheduler.latest_applied_event, result.reward)
                soft_reset = bool(result.info.get("soft_reset"))
                episode_done = bool(result.done or result.truncated) and not soft_reset
                if self.video_recorder is not None and not episode_done:
                    self.video_recorder.record_frame(self.env.render_game_frame())
                pre_active_info = None
                if self.simulated_worker_capacity is not None:
                    pre_active_info = {
                        "observation_submitted": observation_submitted,
                        "observation_dropped": observation_dropped,
                        "in_flight_count": worker_capacity.in_flight_count,
                        "idle_worker_count": worker_capacity.idle_worker_count,
                    }
                record = build_step_record(
                    episode_id=episode_id,
                    env_step=raw_frame,
                    scheduled_time_ms=current_time_ms,
                    active_action=active_action,
                    reward=result.reward,
                    done=episode_done,
                    info=result.info,
                    active_event=self.action_scheduler.latest_applied_event,
                    frame_ms=self.clock.frame_ms,
                    latency_type=latency_type,
                    pre_active_info=pre_active_info,
                )
                if self.collection_observer is not None:
                    self.collection_observer.step(
                        record, terminated=result.done, truncated=result.truncated,
                    )
                step_records.append(record)
                if self.logger is not None:
                    self.logger.log_step(record)
                if soft_reset:
                    if pool is not None:
                        # Fuse every in-flight forward first so the reset discards
                        # the same parked events the inline path would discard.
                        wait_for_events_due(None)
                    self.action_scheduler.reset()
                    result_timeline.reset()
                    if pool is None:
                        self.policy.reset_state(slot_id=0)
                    else:
                        pool.reset_state(0)
                    worker_capacity.reset()
                    recent_drop_count = 0

                raw_frame += 1
                if episode_done:
                    break

            if pool is not None:
                wait_for_events_due(None)
            completed = True
        finally:
            if self.harness is not None:
                self.harness.finish(raw_frame, interrupted=not completed)
                if not completed:
                    for event in action_events:
                        if not event.applied and not event.dropped:
                            event.dropped = True
                            event.metadata["discard_reason"] = "episode_interrupted"
            if self.logger is not None:
                for record in sorted(pooled_latency_records, key=lambda item: item.obs_id):
                    self.logger.log_latency(record)
                for event in sorted(action_events, key=lambda item: item.action_id):
                    self.logger.log_action_event(event)
        final_released, _ = result_timeline.release_arrived(None)
        for released_event in final_released:
            if self.harness is None:
                self.action_scheduler.enqueue(released_event)
            else:
                released_event.dropped = True
                released_event.metadata["discard_reason"] = "episode_ended_before_arrival"
        if self.simulated_worker_capacity is not None:
            episode_metadata.update(
                {
                    "submitted_observation_frames": submitted_observation_frames,
                    "dropped_observation_count": dropped_observation_count,
                    "simulated_worker_capacity": self.simulated_worker_capacity,
                    "inference_worker_count": self.simulated_worker_capacity,
                    "in_flight_count": worker_capacity.in_flight_count,
                    "idle_worker_count": worker_capacity.idle_worker_count,
                }
            )
        if pool is not None:
            episode_metadata["sim_inference_pool_workers"] = pool.worker_count
            if self.harness is not None:
                episode_metadata["inference_worker_count"] = pool.worker_count
        episode_metadata.update(self.action_scheduler.chunk_metrics())
        metrics = compute_episode_metrics(
            episode_id=episode_id,
            step_records=step_records,
            action_events=action_events,
            latency_values_ms=latency_values_ms,
            metadata=episode_metadata,
            frame_ms=self.clock.frame_ms,
        )
        if self.logger is not None:
            self.logger.log_episode_metrics(metrics)
        if self.video_recorder is not None:
            self.video_recorder.close_episode({"episode_return": metrics.episode_return})
        return metrics
