from __future__ import annotations

import random
from bisect import bisect_left
from collections import Counter, defaultdict
from typing import Any, Iterator


def build_latency_burst_samples(
    action_records: list[dict[str, Any]],
    episode_metrics_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record, episode_id in _action_records_with_episode_ids(action_records, episode_metrics_records):
        if record["applied"] and not record["dropped"]:
            latency = record["latency"]
            samples.append(
                {
                    "episode_id": episode_id,
                    "sample_index": len(samples),
                    "action_id": int(record["action_id"]),
                    "obs_env_step": int(record["obs_env_step"]),
                    "ready_raw_frame": int(record["ready_raw_frame"]),
                    "apply_raw_frame": int(record["apply_raw_frame"]),
                    "latency_ms": float(latency["latency_ms"]),
                    "latency_raw_frames": int(record["latency_raw_frames"]),
                }
            )
    return samples


def _action_records_with_episode_ids(
    action_records: list[dict[str, Any]],
    episode_metrics_records: list[dict[str, Any]] | None,
) -> Iterator[tuple[dict[str, Any], int]]:
    if episode_metrics_records is None:
        for record in action_records:
            yield record, int(record["metadata"]["episode_id"])
        return

    cursor = 0
    for metric in episode_metrics_records:
        episode_id = int(metric["episode_id"])
        num_actions = int(metric["num_actions"])
        for record in action_records[cursor:cursor + num_actions]:
            yield record, episode_id
        cursor += num_actions
    if cursor != len(action_records):
        raise ValueError(
            "episode metric num_actions sum "
            f"{cursor} does not match action record count {len(action_records)}"
        )


def build_latency_burst_model(samples: list[dict[str, Any]]) -> dict[str, Any]:
    initial_state_counts: Counter[str] = Counter()
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    run_lengths_by_state: dict[str, list[int]] = defaultdict(list)
    latency_ms_by_state: dict[str, list[float]] = defaultdict(list)

    samples_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        samples_by_episode[int(sample["episode_id"])].append(sample)
        latency_ms_by_state[str(int(sample["latency_raw_frames"]))].append(float(sample["latency_ms"]))

    for episode_id in sorted(samples_by_episode):
        episode_samples = sorted(samples_by_episode[episode_id], key=lambda item: int(item["sample_index"]))
        runs = _runs_for_episode(episode_samples)
        initial_state_counts[runs[0][0]] += 1
        for state, run_length in runs:
            run_lengths_by_state[state].append(run_length)
        for index in range(len(runs) - 1):
            transition_counts[runs[index][0]][runs[index + 1][0]] += 1

    return {
        "schema_version": 1,
        "model_type": "semi_markov_latency_frames",
        "reset_scope": "episode",
        "state_key": "latency_raw_frames",
        "initial_state_counts": dict(sorted(initial_state_counts.items())),
        "initial_state_probs": _probs(initial_state_counts),
        "transition_counts": {
            state: dict(sorted(counts.items()))
            for state, counts in sorted(transition_counts.items())
        },
        "transition_probs": {
            state: _probs(counts)
            for state, counts in sorted(transition_counts.items())
        },
        "run_lengths_by_state": {
            state: sorted(lengths)
            for state, lengths in sorted(run_lengths_by_state.items())
        },
        "latency_ms_by_state": {
            state: sorted(values)
            for state, values in sorted(latency_ms_by_state.items())
        },
        "terminal_transition_policy": "initial",
    }


def _runs_for_episode(samples: list[dict[str, Any]]) -> list[tuple[str, int]]:
    runs: list[tuple[str, int]] = []
    current_state = str(int(samples[0]["latency_raw_frames"]))
    current_length = 0
    for sample in samples:
        state = str(int(sample["latency_raw_frames"]))
        if state == current_state:
            current_length += 1
        else:
            runs.append((current_state, current_length))
            current_state = state
            current_length = 1
    runs.append((current_state, current_length))
    return runs


class SemiMarkovLatencyBurstSampler:
    def __init__(self, model: dict[str, Any], seed: int | None = None):
        self.model = model
        self.rng = random.Random(seed)
        self.current_episode_id: int | None = None
        self.current_state: str | None = None
        self.remaining_run_length = 0

    def sample(self, context: dict[str, Any]) -> dict[str, float | int]:
        episode_id = int(context["episode_id"])
        if self.current_episode_id != episode_id:
            self.current_episode_id = episode_id
            self.current_state = self._sample_initial_state()
            self.remaining_run_length = self._sample_run_length(self.current_state)
        elif self.remaining_run_length <= 0:
            self.current_state = self._sample_next_state(self.current_state)
            self.remaining_run_length = self._sample_run_length(self.current_state)

        state = self.current_state
        self.remaining_run_length -= 1
        return {
            "latency_ms": self._sample_latency_ms(state),
            "latency_raw_frames": int(state),
        }

    def _sample_initial_state(self) -> str:
        return _sample_counts(self.rng, self.model["initial_state_counts"])

    def _sample_next_state(self, state: str) -> str:
        transitions = self.model["transition_counts"]
        if state in transitions:
            return _sample_counts(self.rng, transitions[state])
        return self._sample_initial_state()

    def _sample_run_length(self, state: str) -> int:
        return int(self.rng.choice(self.model["run_lengths_by_state"][state]))

    def _sample_latency_ms(self, state: str) -> float:
        return _sample_empirical(self.rng, self.model["latency_ms_by_state"][state])


def _sample_counts(rng: random.Random, counts: dict[str, int]) -> str:
    total = sum(int(value) for value in counts.values())
    draw = rng.randrange(total)
    cumulative = 0
    for state, count in counts.items():
        cumulative += int(count)
        if draw < cumulative:
            return state
    return state


def _probs(counts: Counter[str]) -> dict[str, float]:
    total = sum(int(value) for value in counts.values())
    return {
        state: int(count) / total
        for state, count in sorted(counts.items())
    }


def _sample_empirical(rng: random.Random, values: list[float]) -> float:
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    quantile_levels = [index / (len(sorted_values) - 1) for index in range(len(sorted_values))]
    quantile = rng.random()
    index = bisect_left(quantile_levels, quantile)
    if index == 0:
        return sorted_values[0]
    if index >= len(sorted_values):
        return sorted_values[-1]
    lower_q = quantile_levels[index - 1]
    upper_q = quantile_levels[index]
    lower_latency = sorted_values[index - 1]
    upper_latency = sorted_values[index]
    weight = (quantile - lower_q) / (upper_q - lower_q)
    return lower_latency + weight * (upper_latency - lower_latency)
