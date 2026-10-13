from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any


def append_episode_metrics_event(path: str | Path | None, event: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(event, sort_keys=True)
    with target.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(payload + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class EpisodeMetricsAggregator:
    """Tail episode events and publish aggregate training metrics."""

    def __init__(
        self,
        *,
        path: str | Path,
        rolling_window: int = 100,
        start_at_end: bool = False,
    ):
        self.path = Path(path).expanduser()
        self.rolling_window = rolling_window
        self.completed_episodes = 0
        self.completed_decision_steps = 0
        self.completed_raw_frames = 0
        self.return_sum = 0.0
        self.raw_return_sum = 0.0
        self.length_sum = 0
        self.raw_frames_sum = 0
        self.latency_sum = 0.0
        self.submitted_observation_sum = 0
        self.dropped_observation_sum = 0
        self.return_window: deque[float] = deque(maxlen=self.rolling_window)
        self.raw_return_window: deque[float] = deque(maxlen=self.rolling_window)
        self.length_window: deque[int] = deque(maxlen=self.rolling_window)
        self.raw_frames_window: deque[int] = deque(maxlen=self.rolling_window)
        self.latency_window: deque[int] = deque(maxlen=self.rolling_window)
        self.submitted_observation_window: deque[int] = deque(maxlen=self.rolling_window)
        self.dropped_observation_window: deque[int] = deque(maxlen=self.rolling_window)
        self._offset = (
            self.path.stat().st_size if start_at_end and self.path.exists() else 0
        )
        self._pending_payloads: list[dict[str, Any]] = []
        self._wandb_initialized = False

    def process_event(self, event: dict[str, Any]) -> dict[str, Any]:
        raw_episode_return = float(event.get("raw_return", event["return"]))
        episode_return = raw_episode_return
        episode_length = int(event["length"])
        raw_frames = int(event["raw_frames"])
        latency_raw_frames = event.get("latency_raw_frames")
        latency_raw_frame_value = float(latency_raw_frames) if latency_raw_frames is not None else None
        submitted_observations = int(event["submitted_observation_count"])
        dropped_observations = int(event["dropped_observation_count"])

        self.completed_episodes += 1
        self.completed_decision_steps += episode_length
        self.completed_raw_frames += raw_frames
        self.return_sum += episode_return
        self.raw_return_sum += raw_episode_return
        self.length_sum += episode_length
        self.raw_frames_sum += raw_frames
        if latency_raw_frame_value is not None:
            self.latency_sum += latency_raw_frame_value
        self.submitted_observation_sum += submitted_observations
        self.dropped_observation_sum += dropped_observations

        self.return_window.append(episode_return)
        self.raw_return_window.append(raw_episode_return)
        self.length_window.append(episode_length)
        self.raw_frames_window.append(raw_frames)
        if latency_raw_frames is not None:
            self.latency_window.append(int(latency_raw_frames))
        self.submitted_observation_window.append(submitted_observations)
        self.dropped_observation_window.append(dropped_observations)

        latest_observation_count = submitted_observations + dropped_observations
        rolling_submitted = sum(self.submitted_observation_window)
        rolling_dropped = sum(self.dropped_observation_window)
        cumulative_observation_count = self.submitted_observation_sum + self.dropped_observation_sum

        rolling_raw_return = float(sum(self.raw_return_window) / len(self.raw_return_window))
        rolling_return = float(sum(self.return_window) / len(self.return_window))
        cumulative_return = float(self.return_sum / self.completed_episodes)
        cumulative_raw_return = float(self.raw_return_sum / self.completed_episodes)
        payload = {
            "charts/episodic_return": float(episode_return),
            "charts/episodic_return_raw": float(raw_episode_return),
            "rollout/ep_rew_mean": rolling_return,
            "rollout/ep_raw_rew_mean": rolling_raw_return,
            "episode_train/completed_episodes": int(self.completed_episodes),
            "episode_train/completed_decision_steps": int(self.completed_decision_steps),
            "episode_train/completed_raw_frames": int(self.completed_raw_frames),
            "episode_train/latest_return": float(episode_return),
            "episode_train/latest_raw_return": float(raw_episode_return),
            "episode_train/latest_length": int(episode_length),
            "episode_train/latest_raw_frames": int(raw_frames),
            "episode_train/rolling_mean_return": rolling_return,
            "episode_train/rolling_mean_raw_return": rolling_raw_return,
            "episode_train/rolling_mean_length": float(sum(self.length_window) / len(self.length_window)),
            "episode_train/rolling_mean_raw_frames": float(sum(self.raw_frames_window) / len(self.raw_frames_window)),
            "episode_train/cumulative_mean_return": cumulative_return,
            "episode_train/cumulative_mean_raw_return": cumulative_raw_return,
            "episode_train/cumulative_mean_length": float(self.length_sum / self.completed_episodes),
            "episode_train/cumulative_mean_raw_frames": float(self.raw_frames_sum / self.completed_episodes),
            "episode/length": int(episode_length),
            "episode/return_per_step": (
                float(raw_episode_return) / float(episode_length) if episode_length > 0 else 0.0
            ),
            "episode_train/latest_submitted_observations": submitted_observations,
            "episode_train/latest_dropped_observations": dropped_observations,
            "episode_train/latest_observation_drop_ratio": (
                float(dropped_observations) / float(latest_observation_count)
            ),
            "episode_train/rolling_observation_drop_ratio": (
                float(rolling_dropped) / float(rolling_submitted + rolling_dropped)
            ),
            "episode_train/cumulative_observation_drop_ratio": (
                float(self.dropped_observation_sum) / float(cumulative_observation_count)
            ),
        }
        if event.get("score") is not None:
            payload["episode/score"] = float(event["score"])
        if event.get("pipes_passed") is not None:
            payload["episode/pipes_passed"] = float(event["pipes_passed"])
        for key, value in event.items():
            if str(key).startswith("action/issued_") and isinstance(value, (int, float)):
                payload[str(key)] = float(value)
        if latency_raw_frames is not None:
            payload["episode_train/latest_latency_raw_frames"] = int(latency_raw_frames)
            payload["episode_train/rolling_mean_latency_raw_frames"] = float(
                sum(self.latency_window) / len(self.latency_window)
            )
            payload["episode_train/cumulative_mean_latency_raw_frames"] = float(
                self.latency_sum / self.completed_episodes
            )
        return payload

    def _ensure_wandb(self):
        try:
            import wandb
        except ImportError:
            return None
        if wandb.run is None:
            return None
        if not self._wandb_initialized:
            wandb.define_metric("episode_train/completed_episodes")
            wandb.define_metric("episode_train/*", step_metric="episode_train/completed_episodes")
            self._wandb_initialized = True
        return wandb

    def _flush_pending(self) -> None:
        wandb = self._ensure_wandb()
        if wandb is None or not self._pending_payloads:
            return
        for payload in self._pending_payloads:
            wandb.log(payload)
        self._pending_payloads.clear()

    def poll_once(self) -> int:
        if not self.path.exists():
            self._flush_pending()
            return 0

        lines_processed = 0
        with self.path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            handle.seek(self._offset)
            while True:
                line = handle.readline()
                if not line:
                    break
                self._offset = handle.tell()
                stripped = line.strip()
                if not stripped:
                    continue
                payload = self.process_event(json.loads(stripped))
                self._pending_payloads.append(payload)
                lines_processed += 1
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        self._flush_pending()
        return lines_processed

    def run_forever(self, stop_event: threading.Event, *, poll_interval_s: float = 1.0) -> None:
        while not stop_event.is_set():
            self.poll_once()
            stop_event.wait(poll_interval_s)
        self.poll_once()


def episode_metrics_path_from_cfg(cfg: Any) -> str:
    train_dir = getattr(cfg, "train_dir", None)
    experiment = getattr(cfg, "experiment", None)
    if not train_dir or not experiment:
        return ""
    return str(Path(str(train_dir)).expanduser() / str(experiment) / "episode_metrics.jsonl")


def unlink_stale_episode_metrics_file(path: Any, restart_behavior: Any) -> Path:
    path_obj = Path(path).expanduser()
    if str(restart_behavior).lower() != "resume" and path_obj.exists():
        path_obj.unlink()
    return path_obj


def start_episode_metrics_aggregator(cfg: Any):
    if os.environ.get("LATENCY_BENCH_DISABLE_EPISODE_METRICS_AGGREGATOR") == "1":
        return None, None
    path = cfg.episode_metrics_path or episode_metrics_path_from_cfg(cfg)
    if not path:
        return None, None
    path_obj = unlink_stale_episode_metrics_file(path, cfg.restart_behavior)
    aggregator = EpisodeMetricsAggregator(
        path=path_obj,
        rolling_window=cfg.stats_avg,
        start_at_end=cfg.restart_behavior.lower() == "resume",
    )
    stop_event = threading.Event()
    thread = threading.Thread(
        target=aggregator.run_forever,
        args=(stop_event,),
        kwargs={"poll_interval_s": 0.5},
        daemon=True,
        name="episode-metrics-aggregator",
    )
    thread.start()
    return stop_event, thread


def stop_episode_metrics_aggregator(stop_event, thread) -> None:
    if stop_event is not None:
        stop_event.set()
    if thread is not None:
        thread.join(timeout=5.0)
