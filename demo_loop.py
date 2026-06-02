from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import numpy as np
from PIL import Image

from app_settings import DEFAULT_OUTPUT_ROOT
from latency_bench.core.actions import ActionResolver
from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, ActionEvent, LatencyRecord, OrderingPolicy, StepRecord
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import ObservationCadenceDecisionScheduler
from latency_bench.scheduler.persistence import HoldPolicy, OneFrameThenNoopPolicy


@dataclass(frozen=True)
class EnvProtocolConfig:
    env_fps: float
    obs_fps: float
    eval_max_steps: int
    hold_policy: str
    ordering_policy: str
    env_config: dict[str, Any]


REALTIME_ENV_PROTOCOLS: dict[str, EnvProtocolConfig] = {
    "flappy": EnvProtocolConfig(
        env_fps=30.0,
        obs_fps=30.0,
        eval_max_steps=3600,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        env_config={
            "name": "flappy",
            "gym_id": "FlappyBird-v0",
            "render_mode": "rgb_array",
            "observation_mode": "image",
            "use_lidar": False,
            "normalize_obs": True,
            "audio_on": False,
            "frame_stack": 4,
            "obs_resize": [84, 84],
            "use_gpu_render": False,
            "gpu_render_device": "auto",
            "noop_action": "noop",
            "action_map": {"noop": 0, "flap": 1},
            "oneshot_actions": ["flap"],
        },
    ),
    "demon_attack": EnvProtocolConfig(
        env_fps=60.0,
        obs_fps=60.0,
        eval_max_steps=3600,
        hold_policy="hold",
        ordering_policy="latest_ready",
        env_config={
            "name": "demon_attack",
            "env_id": "ALE/DemonAttack-v5",
            "screen_size": 84,
            "noop_max": 30,
            "frame_stack": 4,
            "terminal_on_life_loss": True,
            "clip_reward": False,
            "repeat_action_probability": 0.0,
            "full_action_space": False,
            "expected_action_meanings": ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"],
            "noop_action": 0,
            "action_map": {
                "NOOP": 0,
                "FIRE": 1,
                "RIGHT": 2,
                "LEFT": 3,
                "RIGHTFIRE": 4,
                "LEFTFIRE": 5,
            },
        },
    ),
    "deadly_corridor": EnvProtocolConfig(
        env_fps=35.0,
        obs_fps=35.0,
        eval_max_steps=30000,
        hold_policy="hold",
        ordering_policy="latest_ready",
        env_config={
            "name": "deadly_corridor",
            "env_id": "doom_deadly_corridor",
            "frame_stack": 4,
            "res_w": 128,
            "res_h": 72,
            "wide_aspect_ratio": False,
            "simulator": "cpu",
            "noop_action": [0, 0, 0, 0],
            "action_map": {
                "noop": [0, 0, 0, 0],
                "move_forward": [0, 1, 0, 0],
                "move_backward": [0, 2, 0, 0],
                "move_left": [0, 0, 1, 0],
                "move_right": [0, 0, 2, 0],
                "turn_left": [1, 0, 0, 0],
                "turn_right": [2, 0, 0, 0],
                "attack": [0, 0, 0, 1],
            },
        },
    ),
}


class HumanInputState:
    """Thread-safe keyboard state shared between FastAPI and the realtime loop."""

    def __init__(self):
        self._lock = threading.Lock()
        self._keys: set[str] = set()
        self._updated_at_s = time.perf_counter()

    def set_keys(self, keys) -> None:
        with self._lock:
            self._keys = {str(key) for key in keys}
            self._updated_at_s = time.perf_counter()

    def snapshot(self) -> tuple[set[str], float]:
        with self._lock:
            return set(self._keys), float(self._updated_at_s)


class JsonlProtocolLogger:
    def __init__(self, output_dir: Path, metadata: dict[str, Any]):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.write_metadata(metadata)

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        with self._lock:
            with (self.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, default=_json_default)
                handle.write("\n")

    def log_step(self, record: StepRecord) -> None:
        self._append(
            "steps.jsonl",
            {
                "episode_id": record.episode_id,
                "env_step": record.env_step,
                "sim_time_ms": record.sim_time_ms,
                "active_action": _jsonable_action(record.active_action),
                "reward": record.reward,
                "done": record.done,
                "info": record.info,
            },
        )

    def log_action(self, event: ActionEvent) -> None:
        self._append(
            "actions.jsonl",
            {
                "action_id": event.action_id,
                "obs_env_step": event.obs_env_step,
                "ready_time_ms": event.ready_time_ms,
                "ready_step_float": event.ready_step_float,
                "ready_raw_frame": event.ready_raw_frame,
                "apply_raw_frame": event.apply_raw_frame,
                "applied": event.applied,
                "dropped": event.dropped,
                "action": _jsonable_action(event.action),
                "latency_ms": event.latency.latency_ms,
                "latency_type": event.latency.latency_type,
                "metadata": event.metadata,
            },
        )

    def log_episode(self, payload: dict[str, Any], *, valid_for_eval: bool) -> None:
        filename = "episodes.jsonl" if valid_for_eval else "interrupted_episodes.jsonl"
        self._append(filename, payload)

    def _append(self, filename: str, payload: dict[str, Any]) -> None:
        with self._lock:
            with (self.output_dir / filename).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=_json_default) + "\n")


class DemoLoop:
    """Browser-controlled human realtime loop using the latency_bench protocol."""

    def __init__(
        self,
        env_name: str,
        speed: float = 1.0,
        latency_frames: int = 0,
        human_input: Optional[HumanInputState] = None,
        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
        participant_name: str = "anonymous",
        session_id: str | None = None,
        run_mode: str = "eval",
        episodes_per_latency: int = 20,
        latency_start: int | None = None,
        latency_end: int | None = None,
        max_steps_per_episode: int | None = None,
        seed: int = 0,
    ):
        if env_name not in REALTIME_ENV_PROTOCOLS:
            raise ValueError(f"Unsupported env: {env_name}")
        self.env_name = env_name
        self.protocol = REALTIME_ENV_PROTOCOLS[env_name]
        self._speed = max(0.1, min(4.0, float(speed)))
        self.run_mode = _normalize_run_mode(run_mode)
        self.latency_frames = max(0, int(latency_frames))
        self.latency_start = max(0, int(latency_start if latency_start is not None else self.latency_frames))
        self.latency_end = max(0, int(latency_end if latency_end is not None else self.latency_start))
        if self.latency_end < self.latency_start:
            self.latency_start, self.latency_end = self.latency_end, self.latency_start
        self.episodes_per_latency = max(1, int(episodes_per_latency))
        self.max_steps_per_episode = (
            max(1, int(max_steps_per_episode))
            if max_steps_per_episode is not None
            else self.protocol.eval_max_steps
        )
        self.latencies = self._build_latency_plan()
        self.current_latency_frames = self.latencies[0]
        self.human_input = human_input or HumanInputState()
        self.participant_name = str(participant_name).strip() or "anonymous"
        self.participant_slug = _slugify(self.participant_name)
        self.session_id = str(session_id).strip() if session_id else _make_session_id(self.run_mode)
        self.seed = int(seed)

        self.clock = EnvClock(env_fps=self.protocol.env_fps, obs_fps=self.protocol.obs_fps)
        self.action_resolver = ActionResolver(self.protocol.env_config)
        self.decision_scheduler = ObservationCadenceDecisionScheduler(self.clock)
        self.action_scheduler = ActionScheduler(
            clock=self.clock,
            noop_action=self.action_resolver.noop_action,
            persistence_policy=self._build_persistence_policy(),
            ordering_policy=OrderingPolicy(self.protocol.ordering_policy),
        )
        self.output_root = Path(output_root)
        self.session_root = self._session_root_dir()
        self.env_root = self._env_root_dir()
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.env_root.mkdir(parents=True, exist_ok=True)
        self._write_session_metadata()
        self.logger = self._make_latency_logger(self.current_latency_frames)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[Dict[str, Any]] = None
        self._frame_lock = threading.Lock()
        self._next_action_id = 0

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._write_session_summary(session_status="stopped")

    def set_speed(self, speed: float):
        self._speed = max(0.1, min(4.0, float(speed)))

    def set_human_keys(self, keys) -> None:
        self.human_input.set_keys(keys)

    def get_latest(self) -> Optional[Dict[str, Any]]:
        with self._frame_lock:
            return self._latest

    @property
    def output_dir(self) -> Path:
        return self.logger.output_dir

    @property
    def env_output_dir(self) -> Path:
        return self.env_root

    @property
    def num_episodes(self) -> int | None:
        if self.run_mode == "eval":
            return self.episodes_per_latency * len(self.latencies)
        return None

    def _build_persistence_policy(self):
        if self.protocol.hold_policy == "hold":
            return HoldPolicy(duration_raw_frames=self.clock.obs_stride_raw_frames)
        return OneFrameThenNoopPolicy()

    def _push(self, frame: np.ndarray, stats: Dict[str, Any]):
        img = Image.fromarray(frame.astype(np.uint8))
        w, h = img.size
        scale = 620 / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        with self._frame_lock:
            self._latest = {"frame": b64, "stats": stats}

    def _run(self):
        try:
            env, env_kind = self._make_env()
            self._bind_env_noop_action(env)
            try:
                self._run_realtime_env(env, env_kind=env_kind)
            finally:
                env.close()
        except Exception as exc:
            import traceback

            print(f"[human-realtime] {self.env_name} crashed: {exc}")
            traceback.print_exc()

    def _make_env(self):
        if self.env_name == "flappy":
            from latency_bench.envs.flappy import FlappyBirdEnvAdapter

            env_cfg = self.protocol.env_config
            return FlappyBirdEnvAdapter(
                gym_id=env_cfg["gym_id"],
                env_fps=self.protocol.env_fps,
                noop_action=self.action_resolver.noop_action,
                obs_resize=tuple(env_cfg["obs_resize"]),
                frame_stack=int(env_cfg["frame_stack"]),
                use_gpu_render=bool(env_cfg["use_gpu_render"]),
                gpu_render_device=str(env_cfg["gpu_render_device"]),
                policy_observation_layout="hwc",
                export_env_raw_rgb_frames=False,
            ), "adapter"
        if self.env_name == "demon_attack":
            from latency_bench.envs.demon_attack import DemonAttackEnvAdapter

            env_cfg = self.protocol.env_config
            return DemonAttackEnvAdapter(
                env_id=env_cfg["env_id"],
                env_fps=self.protocol.env_fps,
                noop_action=self.action_resolver.noop_action,
                screen_size=int(env_cfg["screen_size"]),
                noop_max=int(env_cfg["noop_max"]),
                terminal_on_life_loss=bool(env_cfg["terminal_on_life_loss"]),
                clip_reward=bool(env_cfg["clip_reward"]),
                repeat_action_probability=float(env_cfg["repeat_action_probability"]),
                full_action_space=bool(env_cfg["full_action_space"]),
                expected_action_meanings=env_cfg["expected_action_meanings"],
                frame_stack=int(env_cfg["frame_stack"]),
                export_env_raw_rgb_frames=False,
            ), "adapter"
        if self.env_name == "deadly_corridor":
            from latency_bench.envs.deadly_corridor import DeadlyCorridorEnvAdapter

            return DeadlyCorridorEnvAdapter(
                config=self._deadly_corridor_latency_bench_config(),
                noop_action=self.action_resolver.noop_action,
                export_env_raw_rgb_frames=False,
            ), "adapter"
        raise ValueError(f"Unsupported env: {self.env_name}")

    def _bind_env_noop_action(self, env) -> None:
        noop_action = getattr(env, "noop_action", None)
        if isinstance(noop_action, Action):
            self.action_scheduler = ActionScheduler(
                clock=self.clock,
                noop_action=noop_action,
                persistence_policy=self._build_persistence_policy(),
                ordering_policy=OrderingPolicy(self.protocol.ordering_policy),
            )

    def _deadly_corridor_latency_bench_config(self) -> dict[str, Any]:
        return {
            "experiment": {"name": "deadly_corridor_human_realtime", "seed": self.seed},
            "backend": {
                "type": "sample_factory",
                "algo": "APPO",
                "device": "cpu",
                "train_dir": "outputs/human_realtime/sample_factory",
                "restart_behavior": "overwrite",
                "run_mode": "eval",
            },
            "env": {
                **self.protocol.env_config,
                "env_fps": self.protocol.env_fps,
                "obs_fps": self.protocol.obs_fps,
            },
            "latency": {"type": "zero", "add_latency_info": False},
            "scheduler": {
                "hold_policy": self.protocol.hold_policy,
                "ordering_policy": self.protocol.ordering_policy,
            },
            "policy": {"type": "human"},
            "training": {
                "train_for_env_steps": 0,
                "num_workers": 1,
                "num_envs_per_worker": 1,
                "worker_num_splits": 1,
                "num_policies": 1,
                "batch_size": 1,
                "rollout": 1,
                "recurrence": 1,
                "num_epochs": 1,
                "num_batches_per_epoch": 1,
                "async_rl": False,
                "use_rnn": False,
                "normalize_input": True,
                "normalize_returns": True,
            },
            "evaluation": {
                "eval_episodes": 1,
                "eval_parallel_envs": 1,
                "eval_max_steps": self.protocol.eval_max_steps,
                "eval_deterministic": True,
                "eval_raw_reward": True,
            },
            "logging": {"output_dir": str(self.logger.output_dir)},
        }

    def _run_realtime_env(self, env, *, env_kind: str):
        global_episode_id = 0
        for latency_idx, latency_frames in enumerate(self.latencies):
            if self._stop_event.is_set():
                break
            self.current_latency_frames = int(latency_frames)
            self.logger = self._make_latency_logger(self.current_latency_frames)
            episode_id = 0
            while not self._stop_event.is_set() and self._should_run_more_episodes(episode_id):
                global_episode_id = self._run_one_episode(
                    env,
                    env_kind=env_kind,
                    episode_id=episode_id,
                    global_episode_id=global_episode_id,
                    latency_idx=latency_idx,
                )
                episode_id += 1

            if self.run_mode == "freeplay":
                break

        if not self._stop_event.is_set():
            if self.run_mode == "eval":
                self._write_eval_summary()
            self._write_session_summary(session_status="completed")
            with self._frame_lock:
                if self._latest is not None:
                    stats = dict(self._latest.get("stats") or {})
                    stats["run_complete"] = True
                    stats["status"] = "completed"
                    stats["completed_episodes"] = global_episode_id
                    stats["output_dir"] = str(self.output_dir)
                    stats["session_root"] = str(self.session_root)
                    stats["env_root"] = str(self.env_root)
                    stats["session_summary_json"] = str(self.env_root / "session_summary.json")
                    stats["overall_summary_json"] = str(self.session_root / "overall_summary.json")
                    completed_envs = self._completed_eval_envs()
                    stats["completed_envs"] = completed_envs
                    stats["required_envs"] = _required_env_slugs()
                    stats["session_all_envs_complete"] = set(completed_envs) >= set(_required_env_slugs())
                    if self.run_mode == "eval":
                        stats["summary_json"] = str(self.env_root / "summary.json")
                        stats["summary_csv"] = str(self.env_root / "summary.csv")
                    self._latest = {**self._latest, "stats": stats}

    def _run_one_episode(
        self,
        env,
        *,
        env_kind: str,
        episode_id: int,
        global_episode_id: int,
        latency_idx: int,
    ) -> int:
            self.action_scheduler.reset()
            episode_seed = self._episode_seed(episode_id)
            obs = self._reset_env(env, env_kind=env_kind, seed=episode_seed)
            del obs
            total_reward = 0.0
            step_records: list[StepRecord] = []
            action_events: list[ActionEvent] = []
            episode_start_s = time.perf_counter()
            max_wall_clock_lag_ms = 0.0
            env_deadline_miss_count = 0

            completed_episode = False
            interrupt_reason = None

            for env_step in range(self.max_steps_per_episode):
                if self._stop_event.is_set():
                    interrupt_reason = "stop_requested"
                    break
                scheduled_time_ms = self.clock.step_to_time_ms(env_step)
                self._sleep_until(episode_start_s, scheduled_time_ms)
                wall_elapsed_ms = (time.perf_counter() - episode_start_s) * 1000.0
                wall_clock_lag_ms = max(0.0, wall_elapsed_ms - scheduled_time_ms / self._speed)
                max_wall_clock_lag_ms = max(max_wall_clock_lag_ms, wall_clock_lag_ms)
                if wall_clock_lag_ms > self.clock.frame_ms:
                    env_deadline_miss_count += 1

                latest_input_age_ms = None
                if self.decision_scheduler.should_observe(env_step, scheduled_time_ms):
                    event, latest_input_age_ms = self._make_human_action_event(env_step, scheduled_time_ms)
                    self.action_scheduler.enqueue(event)
                    action_events.append(event)

                active_action = self.action_scheduler.update(env_step, scheduled_time_ms)
                reward, done, raw_info = self._step_env(env, env_kind=env_kind, action=active_action)
                total_reward += reward
                record = StepRecord(
                    episode_id=episode_id,
                    env_step=env_step,
                    sim_time_ms=scheduled_time_ms,
                    active_action=active_action,
                    reward=reward,
                    done=done,
                    info={
                        **raw_info,
                        "scheduled_time_ms": scheduled_time_ms,
                        "wall_elapsed_ms": wall_elapsed_ms,
                        "frame_time_ms": self.clock.frame_ms,
                        "wall_clock_lag_ms": wall_clock_lag_ms,
                        "target_env_fps": self.clock.env_fps,
                        "obs_fps": self.clock.obs_fps,
                        "obs_stride_raw_frames": self.clock.obs_stride_raw_frames,
                        "action_source": "human",
                        "active_action_name": active_action.name,
                        "active_action_value": active_action.value,
                        "input_age_ms": latest_input_age_ms,
                    },
                )
                step_records.append(record)
                self.logger.log_step(record)
                for event in action_events:
                    if event.applied and not event.metadata.get("_logged"):
                        self.logger.log_action(event)
                        event.metadata["_logged"] = True

                raw = self._get_render(env, env_kind=env_kind)
                if raw is not None:
                    self._push(
                        raw,
                        self._stats(
                            episode_id=episode_id,
                            global_episode_id=global_episode_id,
                            latency_idx=latency_idx,
                            step=env_step + 1,
                            reward=reward,
                            total_reward=total_reward,
                            active_action=active_action,
                            wall_clock_lag_ms=wall_clock_lag_ms,
                            input_age_ms=latest_input_age_ms,
                        ),
                    )

                if done:
                    completed_episode = True
                    break
            else:
                completed_episode = True

            elapsed_s = max(1e-9, time.perf_counter() - episode_start_s)
            valid_for_eval = bool(completed_episode and not self._stop_event.is_set())
            self.logger.log_episode(
                self._episode_payload(
                    episode_id=episode_id,
                    global_episode_id=global_episode_id,
                    latency_idx=latency_idx,
                    episode_seed=episode_seed,
                    total_reward=total_reward,
                    step_records=step_records,
                    action_events=action_events,
                    elapsed_s=elapsed_s,
                    env_deadline_miss_count=env_deadline_miss_count,
                    max_wall_clock_lag_ms=max_wall_clock_lag_ms,
                    valid_for_eval=valid_for_eval,
                    interrupt_reason=None if valid_for_eval else interrupt_reason or "interrupted",
                ),
                valid_for_eval=valid_for_eval,
            )
            return global_episode_id + 1

    def _sleep_until(self, episode_start_s: float, scheduled_time_ms: float) -> None:
        target_wall_s = episode_start_s + (scheduled_time_ms / 1000.0) / self._speed
        now_s = time.perf_counter()
        if now_s < target_wall_s:
            time.sleep(target_wall_s - now_s)

    def _make_human_action_event(self, env_step: int, scheduled_time_ms: float) -> tuple[ActionEvent, float]:
        keys, updated_at_s = self.human_input.snapshot()
        input_age_ms = max(0.0, (time.perf_counter() - updated_at_s) * 1000.0)
        action = self._decode_human_action(keys)
        ready_raw_frame = int(env_step) + int(self.current_latency_frames)
        ready_time_ms = self.clock.step_to_time_ms(ready_raw_frame)
        event = ActionEvent(
            action_id=self._next_action_id,
            action=action,
            obs_env_step=env_step,
            ready_time_ms=ready_time_ms,
            ready_step_float=float(ready_raw_frame),
            ready_raw_frame=ready_raw_frame,
            latency=LatencyRecord(
                obs_id=env_step,
                latency_ms=self.current_latency_frames * self.clock.frame_ms,
                latency_type="fixed_human_action_latency",
                metadata={
                    "input_age_ms": input_age_ms,
                    "keys": sorted(keys),
                    "latency_raw_frames": self.current_latency_frames,
                },
            ),
            latency_raw_frames=self.current_latency_frames,
            metadata={
                "action_source": "human",
                "keys": sorted(keys),
                "input_age_ms": input_age_ms,
                "latency_raw_frames": self.current_latency_frames,
                "latency_ms": self.current_latency_frames * self.clock.frame_ms,
            },
        )
        self._next_action_id += 1
        return event, input_age_ms

    def _decode_human_action(self, keys: set[str]) -> Action:
        if self.env_name == "flappy":
            flap = "Space" in keys or "ArrowUp" in keys or "KeyW" in keys
            return self.action_resolver.resolve("flap" if flap else "noop")
        if self.env_name == "demon_attack":
            left = "ArrowLeft" in keys or "KeyA" in keys
            right = "ArrowRight" in keys or "KeyD" in keys
            fire = "Space" in keys or "KeyW" in keys or "ArrowUp" in keys
            if left and not right and fire:
                return self.action_resolver.resolve("LEFTFIRE")
            if right and not left and fire:
                return self.action_resolver.resolve("RIGHTFIRE")
            if fire:
                return self.action_resolver.resolve("FIRE")
            if right and not left:
                return self.action_resolver.resolve("RIGHT")
            if left and not right:
                return self.action_resolver.resolve("LEFT")
            return self.action_resolver.resolve("NOOP")
        if self.env_name == "deadly_corridor":
            turn = 0
            move = 0
            strafe = 0
            attack = 0
            if "ArrowLeft" in keys:
                turn = 1
            elif "ArrowRight" in keys:
                turn = 2
            if "KeyW" in keys or "ArrowUp" in keys:
                move = 1
            elif "KeyS" in keys or "ArrowDown" in keys:
                move = 2
            if "KeyA" in keys:
                strafe = 1
            elif "KeyD" in keys:
                strafe = 2
            if "Space" in keys:
                attack = 1
            value = [turn, move, strafe, attack]
            labels = []
            if turn == 1:
                labels.append("TURN_LEFT")
            elif turn == 2:
                labels.append("TURN_RIGHT")
            if move == 1:
                labels.append("MOVE_FORWARD")
            elif move == 2:
                labels.append("MOVE_BACKWARD")
            if strafe == 1:
                labels.append("MOVE_LEFT")
            elif strafe == 2:
                labels.append("MOVE_RIGHT")
            if attack == 1:
                labels.append("ATTACK")
            return Action(value=value, name="+".join(labels) if labels else "NOOP", is_noop=value == [0, 0, 0, 0])
        return self.action_resolver.noop_action

    def _reset_env(self, env, *, env_kind: str, seed: int):
        if env_kind == "adapter":
            return env.reset(seed=seed)
        if env_kind == "vector":
            seed_fn = getattr(env, "seed", None)
            if callable(seed_fn):
                seed_fn(seed)
            return env.reset()
        obs, _info = env.reset(seed=seed)
        return obs

    def _step_env(self, env, *, env_kind: str, action: Action) -> tuple[float, bool, dict[str, Any]]:
        if env_kind == "adapter":
            result = env.step(action)
            return float(result.reward), bool(result.done or result.truncated), dict(result.info or {})
        if env_kind == "vector":
            env_action = self._vector_action_value(env, action)
            _obs, reward, done_vec, info = env.step(env_action)
            info_item = info[0] if isinstance(info, (list, tuple)) and info else {}
            return float(reward[0]), bool(done_vec[0]), dict(info_item or {})
        env_action = self._plain_action_value(env, action)
        _obs, reward, terminated, truncated, info = env.step(env_action)
        return float(reward), bool(terminated or truncated), dict(info or {})

    def _get_render(self, env_or_venv, env_kind: str) -> Optional[np.ndarray]:
        try:
            if env_kind == "adapter":
                return env_or_venv.render_game_frame()
            if env_kind == "vector":
                return env_or_venv.envs[0].unwrapped.render()
            return env_or_venv.unwrapped.render()
        except Exception:
            return None

    def _vector_action_value(self, venv, action: Action):
        value = self._coerce_action_for_space(action, getattr(venv, "action_space", None))
        if isinstance(value, np.ndarray):
            return np.expand_dims(value, axis=0)
        return np.asarray([value], dtype=object if isinstance(value, (list, tuple)) else None)

    def _plain_action_value(self, env, action: Action):
        return self._coerce_action_for_space(action, getattr(env, "action_space", None))

    def _coerce_action_for_space(self, action: Action, action_space):
        value = action.value
        n = getattr(action_space, "n", None)
        if n is not None:
            if isinstance(value, (list, tuple)):
                return _deadly_factorized_to_discrete(value, int(n))
            return int(value)
        if isinstance(value, list):
            return np.asarray(value, dtype=np.int64)
        return value

    def _episode_seed(self, episode_id: int) -> int:
        return self.seed + 104729 + int(episode_id)

    def _stats(
        self,
        *,
        episode_id: int,
        global_episode_id: int,
        latency_idx: int,
        step: int,
        reward: float,
        total_reward: float,
        active_action: Action,
        wall_clock_lag_ms: float,
        input_age_ms: float | None,
    ) -> Dict[str, Any]:
        return {
            "env": self.env_name,
            "env_output_slug": _env_output_slug(self.env_name),
            "run_mode": self.run_mode,
            "episode": episode_id,
            "global_episode": global_episode_id,
            "episode_target": self.episodes_per_latency if self.run_mode == "eval" else None,
            "latency_index": latency_idx,
            "latency_target": len(self.latencies),
            "step": step,
            "reward": round(float(reward), 3),
            "total_reward": round(float(total_reward), 3),
            "speed": round(self._speed, 2),
            "mode": "human_realtime",
            "participant": self.participant_name,
            "session_id": self.session_id,
            "output_dir": str(self.output_dir),
            "session_root": str(self.session_root),
            "env_root": str(self.env_root),
            "target_env_fps": self.clock.env_fps,
            "obs_fps": self.clock.obs_fps,
            "frame_ms": round(self.clock.frame_ms, 3),
            "latency_frames": self.current_latency_frames,
            "latency_ms": round(self.current_latency_frames * self.clock.frame_ms, 3),
            "action": active_action.name,
            "wall_clock_lag_ms": round(float(wall_clock_lag_ms), 3),
            "input_age_ms": None if input_age_ms is None else round(float(input_age_ms), 3),
        }

    def _session_root_dir(self) -> Path:
        return (
            self.output_root
            / self.participant_slug
            / self.session_id
        )

    def _env_root_dir(self) -> Path:
        return self.session_root / _env_output_slug(self.env_name)

    def _latency_output_dir(self, latency_frames: int) -> Path:
        return self.env_root / f"latency_{int(latency_frames)}fr"

    def _make_latency_logger(self, latency_frames: int) -> JsonlProtocolLogger:
        return JsonlProtocolLogger(
            self._latency_output_dir(latency_frames),
            metadata=self._latency_metadata_payload(latency_frames),
        )

    def _write_session_metadata(self) -> None:
        root_payload = self._root_session_metadata_payload()
        with (self.session_root / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(root_payload, handle, indent=2, default=_json_default)
            handle.write("\n")
        env_payload = self._env_metadata_payload()
        with (self.env_root / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(env_payload, handle, indent=2, default=_json_default)
            handle.write("\n")

    def _root_session_metadata_payload(self) -> dict[str, Any]:
        return {
            "participant_name": self.participant_name,
            "participant_slug": self.participant_slug,
            "session_id": self.session_id,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "session_root": str(self.session_root),
            "output_layout": "participant_session_env_latency",
        }

    def _env_metadata_payload(self) -> dict[str, Any]:
        return {
            "participant_name": self.participant_name,
            "participant_slug": self.participant_slug,
            "session_id": self.session_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "env": self.env_name,
            "env_output_slug": _env_output_slug(self.env_name),
            "mode": self.run_mode,
            "episodes_per_latency": self.episodes_per_latency if self.run_mode == "eval" else None,
            "latency_start": self.latency_start if self.run_mode == "eval" else None,
            "latency_end": self.latency_end if self.run_mode == "eval" else None,
            "latencies": self.latencies,
            "seed_base": self.seed,
            "speed": self._speed,
            "session_root": str(self.session_root),
            "env_root": str(self.env_root),
            "protocol": {
                "env_fps": self.clock.env_fps,
                "obs_fps": self.clock.obs_fps,
                "obs_stride_raw_frames": self.clock.obs_stride_raw_frames,
                "eval_max_steps": self.protocol.eval_max_steps,
                "max_steps_per_episode": self.max_steps_per_episode,
                "hold_policy": self.protocol.hold_policy,
                "ordering_policy": self.protocol.ordering_policy,
                "env_config": self.protocol.env_config,
            },
            "episode_seeds": [
                self._episode_seed(i)
                for i in range(self.episodes_per_latency if self.run_mode == "eval" else 0)
            ],
        }

    def _latency_metadata_payload(self, latency_frames: int) -> dict[str, Any]:
        episode_count = self.episodes_per_latency if self.run_mode == "eval" else None
        return {
            "participant_name": self.participant_name,
            "participant_slug": self.participant_slug,
            "session_id": self.session_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "env": self.env_name,
            "mode": self.run_mode,
            "latency_frames": int(latency_frames),
            "latency_ms": int(latency_frames) * self.clock.frame_ms,
            "episodes_per_latency": episode_count,
            "max_steps_per_episode": self.max_steps_per_episode,
            "speed": self._speed,
            "session_root": str(self.session_root),
            "env_root": str(self.env_root),
            "output_dir": str(self._latency_output_dir(latency_frames)),
            "episode_seeds": [
                self._episode_seed(i)
                for i in range(episode_count or 0)
            ],
        }

    def _episode_payload(
        self,
        *,
        episode_id: int,
        global_episode_id: int,
        latency_idx: int,
        episode_seed: int,
        total_reward: float,
        step_records: list[StepRecord],
        action_events: list[ActionEvent],
        elapsed_s: float,
        env_deadline_miss_count: int,
        max_wall_clock_lag_ms: float,
        valid_for_eval: bool,
        interrupt_reason: str | None,
    ) -> dict[str, Any]:
        return {
            "participant_name": self.participant_name,
            "participant_slug": self.participant_slug,
            "session_id": self.session_id,
            "env": self.env_name,
            "env_output_slug": _env_output_slug(self.env_name),
            "run_mode": self.run_mode,
            "episode_id": episode_id,
            "global_episode_id": global_episode_id,
            "latency_index": latency_idx,
            "episode_seed": episode_seed,
            "episode_return": total_reward,
            "survival_steps": len(step_records),
            "num_actions": len(action_events),
            "num_dropped_actions": sum(1 for event in action_events if event.dropped),
            "configured_latency_frames": self.current_latency_frames,
            "configured_latency_ms": self.current_latency_frames * self.clock.frame_ms,
            "target_env_fps": self.clock.env_fps,
            "actual_env_fps": len(step_records) / elapsed_s,
            "obs_fps": self.clock.obs_fps,
            "frame_ms": self.clock.frame_ms,
            "env_deadline_miss_count": env_deadline_miss_count,
            "max_wall_clock_lag_ms": max_wall_clock_lag_ms,
            "episode_status": "completed" if valid_for_eval else "interrupted",
            "valid_for_eval": valid_for_eval,
            "interrupt_reason": interrupt_reason,
            "mode": "human_realtime",
            "output_dir": str(self.output_dir),
            "session_root": str(self.session_root),
            "env_root": str(self.env_root),
        }

    def _build_latency_plan(self) -> list[int]:
        if self.run_mode == "freeplay":
            return [self.latency_frames]
        return list(range(self.latency_start, self.latency_end + 1))

    def _should_run_more_episodes(self, episode_id: int) -> bool:
        if self.run_mode == "freeplay":
            return True
        return episode_id < self.episodes_per_latency

    def _write_eval_summary(self) -> None:
        latencies = [item["metrics"] for item in self._collect_latency_results()]

        summary = {
            "participant_name": self.participant_name,
            "participant_slug": self.participant_slug,
            "session_id": self.session_id,
            "env": self.env_name,
            "env_output_slug": _env_output_slug(self.env_name),
            "mode": self.run_mode,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "episodes_per_latency": self.episodes_per_latency,
            "latency_start": self.latency_start,
            "latency_end": self.latency_end,
            "latencies": latencies,
            "session_root": str(self.session_root),
            "env_root": str(self.env_root),
        }

        with (self.env_root / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=_json_default)
            handle.write("\n")
        self._write_summary_csv(latencies)
        self._write_latency_graphs(latencies)
        self._write_overall_summary()

    def _write_session_summary(self, *, session_status: str) -> None:
        latency_results = self._collect_latency_results()
        payload = {
            "participant_name": self.participant_name,
            "participant_slug": self.participant_slug,
            "session_id": self.session_id,
            "env": self.env_name,
            "env_output_slug": _env_output_slug(self.env_name),
            "mode": self.run_mode,
            "session_status": session_status,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "episodes_per_latency": self.episodes_per_latency if self.run_mode == "eval" else None,
            "latency_start": self.latency_start if self.run_mode == "eval" else None,
            "latency_end": self.latency_end if self.run_mode == "eval" else None,
            "latencies": self.latencies,
            "max_steps_per_episode": self.max_steps_per_episode,
            "speed": self._speed,
            "session_root": str(self.session_root),
            "env_root": str(self.env_root),
            "protocol": {
                "env_fps": self.clock.env_fps,
                "obs_fps": self.clock.obs_fps,
                "obs_stride_raw_frames": self.clock.obs_stride_raw_frames,
                "hold_policy": self.protocol.hold_policy,
                "ordering_policy": self.protocol.ordering_policy,
            },
            "results_by_latency": latency_results,
        }
        with (self.env_root / "session_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=_json_default)
            handle.write("\n")
        self._write_overall_summary()

    def _collect_latency_results(self) -> list[dict[str, Any]]:
        results = []
        for latency_frames in self.latencies:
            output_dir = self._latency_output_dir(latency_frames)
            episodes = _read_jsonl(output_dir / "episodes.jsonl")
            interrupted = _read_jsonl(output_dir / "interrupted_episodes.jsonl")
            valid = [episode for episode in episodes if bool(episode.get("valid_for_eval", True))]
            results.append(
                {
                    "latency_frames": int(latency_frames),
                    "output_dir": str(output_dir),
                    "metrics": self._latency_summary_payload(latency_frames, valid, interrupted),
                    "episodes": valid,
                    "interrupted_episodes": interrupted,
                }
            )
        return results

    def _latency_summary_payload(
        self,
        latency_frames: int,
        valid_episodes: list[dict[str, Any]],
        interrupted_episodes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        returns = [float(item.get("episode_return", 0.0)) for item in valid_episodes]
        survival_steps = [float(item.get("survival_steps", 0.0)) for item in valid_episodes]
        actual_env_fps = [float(item.get("actual_env_fps", 0.0)) for item in valid_episodes]
        deadline_misses = [int(item.get("env_deadline_miss_count", 0)) for item in valid_episodes]
        return {
            "latency_frames": int(latency_frames),
            "latency_ms": int(latency_frames) * self.clock.frame_ms,
            "num_valid_episodes": len(valid_episodes),
            "num_interrupted_episodes": len(interrupted_episodes),
            "expected_episodes": self.episodes_per_latency,
            "mean_return": _mean(returns),
            "std_return": _std(returns),
            "min_return": min(returns) if returns else None,
            "max_return": max(returns) if returns else None,
            "mean_survival_steps": _mean(survival_steps),
            "std_survival_steps": _std(survival_steps),
            "min_survival_steps": min(survival_steps) if survival_steps else None,
            "max_survival_steps": max(survival_steps) if survival_steps else None,
            "mean_actual_env_fps": _mean(actual_env_fps),
            "total_deadline_misses": sum(deadline_misses),
        }

    def _write_summary_csv(self, latencies: list[dict[str, Any]]) -> None:
        rows = []
        for item in latencies:
            rows.append(
                {
                    "participant_name": self.participant_name,
                    "participant_slug": self.participant_slug,
                    "env": self.env_name,
                    "env_output_slug": _env_output_slug(self.env_name),
                    "session_id": self.session_id,
                    **item,
                }
            )
        if not rows:
            return
        with (self.env_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_latency_graphs(self, latencies: list[dict[str, Any]]) -> None:
        points = [(item["latency_frames"], item.get("mean_survival_steps")) for item in latencies]
        _write_line_chart(
            self.env_root / "latency_vs_mean_length.png",
            points,
            title=f"{self.env_name}: latency vs mean length",
            x_label="Latency (frames)",
            y_label="Mean length",
        )
        points = [(item["latency_frames"], item.get("mean_return")) for item in latencies]
        _write_line_chart(
            self.env_root / "latency_vs_mean_reward.png",
            points,
            title=f"{self.env_name}: latency vs mean reward",
            x_label="Latency (frames)",
            y_label="Mean reward",
        )

    def _write_overall_summary(self) -> None:
        envs = []
        for env_dir in sorted(path for path in self.session_root.iterdir() if path.is_dir()):
            summary_path = env_dir / "summary.json"
            session_summary_path = env_dir / "session_summary.json"
            if summary_path.exists():
                payload = _read_json_file(summary_path)
            elif session_summary_path.exists():
                payload = _read_json_file(session_summary_path)
            else:
                continue
            if payload is not None:
                envs.append(
                    {
                        "env_dir": env_dir.name,
                        "env": payload.get("env"),
                        "mode": payload.get("mode"),
                        "summary_path": str(summary_path if summary_path.exists() else session_summary_path),
                        "latencies": payload.get("latencies") or [
                            item.get("metrics") for item in payload.get("results_by_latency", [])
                        ],
                    }
                )
        overall = {
            "participant_name": self.participant_name,
            "participant_slug": self.participant_slug,
            "session_id": self.session_id,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "session_root": str(self.session_root),
            "environments": envs,
        }
        with (self.session_root / "overall_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(overall, handle, indent=2, default=_json_default)
            handle.write("\n")

    def _completed_eval_envs(self) -> list[str]:
        completed = []
        for env_slug in _required_env_slugs():
            summary_path = self.session_root / env_slug / "summary.json"
            if summary_path.exists():
                completed.append(env_slug)
        return completed


def _deadly_factorized_to_discrete(value: Any, n: int) -> int:
    turn, move, strafe, attack = [int(item) for item in list(value)[:4]]
    if attack:
        return min(max(n - 1, 0), n - 1)
    if move == 1:
        return min(1, n - 1)
    if move == 2:
        return min(2, n - 1)
    if strafe == 1 or turn == 1:
        return min(3, n - 1)
    if strafe == 2 or turn == 2:
        return min(4, n - 1)
    return 0


def _jsonable_action(action: Action) -> dict[str, Any]:
    return {
        "value": action.value,
        "name": action.name,
        "is_noop": action.is_noop,
        "is_oneshot": action.is_oneshot,
    }


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip().lower()).strip("_")
    return slug or "anonymous"


def _env_output_slug(env_name: str) -> str:
    if env_name == "deadly_corridor":
        return "deadly"
    return _slugify(env_name)


def _required_env_slugs() -> list[str]:
    return ["flappy", "demon_attack", "deadly"]


def _normalize_run_mode(value: str) -> str:
    mode = str(value or "eval").strip().lower()
    if mode not in {"eval", "freeplay"}:
        raise ValueError(f"Unsupported run mode: {value}")
    return mode


def _make_session_id(run_mode: str = "eval") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = "eval_session" if _normalize_run_mode(run_mode) == "eval" else "freeplay_session"
    return f"{prefix}_{stamp}_{uuid4().hex[:8]}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def _write_line_chart(
    path: Path,
    points: list[tuple[int | float, float | None]],
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    cache_dir = path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    valid_points = sorted((float(x), float(y)) for x, y in points if y is not None)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)

    if valid_points:
        xs = [item[0] for item in valid_points]
        ys = [item[1] for item in valid_points]
        ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=6, color="#1f77b4")
        for x, y in valid_points:
            ax.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
        ax.set_xlim(min(xs) - 0.25, max(xs) + 0.25 if min(xs) != max(xs) else max(xs) + 1.0)
        if min(xs) == max(xs):
            ax.set_xlim(xs[0] - 1.0, xs[0] + 1.0)
        y_min = min(0.0, min(ys))
        y_max = max(1.0, max(ys))
        if y_min == y_max:
            y_max = y_min + 1.0
        ax.set_ylim(y_min, y_max * 1.12 if y_max > 0 else y_max + abs(y_max) * 0.12 + 1.0)
    else:
        ax.text(
            0.5,
            0.5,
            "No valid episodes",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_ylim(0, 1)

    ax.set_title(title, pad=14)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return float(variance ** 0.5)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Action):
        return _jsonable_action(value)
    return str(value)
