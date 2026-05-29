from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from training.sample_factory_common.episode_metrics import append_episode_metrics_event


class BatchedFlappyEpisodeMetrics:
    def __init__(
        self,
        *,
        batch_size: int,
        metrics_path: str,
        worker_index: int,
        vector_index: int,
        env_ids: Sequence[int],
    ):
        self.batch_size = int(batch_size)
        self.metrics_path = str(metrics_path or "")
        self.worker_index = int(worker_index)
        self.vector_index = int(vector_index)
        self.env_ids = [int(item) for item in env_ids]
        self.episode_return = np.zeros(self.batch_size, dtype=np.float64)
        self.episode_raw_return = np.zeros(self.batch_size, dtype=np.float64)
        self.episode_length = np.zeros(self.batch_size, dtype=np.int64)
        self.episode_idx = np.zeros(self.batch_size, dtype=np.int64)

    def reset(self, indices: Sequence[int] | np.ndarray | None = None) -> None:
        target_indices = self._indices(indices)
        for agent_idx in target_indices:
            self.episode_return[agent_idx] = 0.0
            self.episode_raw_return[agent_idx] = 0.0
            self.episode_length[agent_idx] = 0

    def update(
        self,
        *,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        active_mask: np.ndarray,
        scores: np.ndarray,
        queue_latencies: np.ndarray,
    ) -> dict[int, dict[str, Any]]:
        active = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        terminated = np.asarray(terminated, dtype=np.bool_).reshape(-1)
        truncated = np.asarray(truncated, dtype=np.bool_).reshape(-1)
        done = active & np.logical_or(terminated, truncated)

        self.episode_return[active] += rewards[active]
        self.episode_raw_return[active] += rewards[active]
        self.episode_length[active] += 1

        infos: dict[int, dict[str, Any]] = {}
        for agent_idx in np.flatnonzero(done):
            queue_latency = int(queue_latencies[agent_idx])
            extra_stats = {
                "episode/return": float(self.episode_return[agent_idx]),
                "episode/raw_return": float(self.episode_raw_return[agent_idx]),
                "episode/length": int(self.episode_length[agent_idx]),
                "episode/raw_frames": int(self.episode_length[agent_idx]),
                "episode/index": int(self.episode_idx[agent_idx]),
                "episode/queue_latency": queue_latency,
            }
            infos[int(agent_idx)] = {
                "episode_extra_stats": extra_stats,
                "episodic_return": float(self.episode_return[agent_idx]),
                "episodic_length": int(self.episode_length[agent_idx]),
            }
            append_episode_metrics_event(
                self.metrics_path,
                {
                    "episode_idx": int(self.episode_idx[agent_idx]),
                    "return": float(self.episode_return[agent_idx]),
                    "raw_return": float(self.episode_raw_return[agent_idx]),
                    "length": int(self.episode_length[agent_idx]),
                    "raw_frames": int(self.episode_length[agent_idx]),
                    "queue_latency": queue_latency,
                    "game_score": int(scores[agent_idx]),
                    "worker_index": int(self.worker_index),
                    "vector_index": int(self.vector_index),
                    "env_id": int(self.env_ids[agent_idx]),
                },
            )
            self.episode_idx[agent_idx] += 1
        return infos

    def _indices(self, indices: Sequence[int] | np.ndarray | None) -> list[int]:
        if indices is None:
            return list(range(self.batch_size))
        return [int(item) for item in np.asarray(indices).reshape(-1).tolist()]
