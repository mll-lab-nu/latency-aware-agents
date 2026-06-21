from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Callable, Iterator

from latency_bench.utils.stats import percentile_linear


WORKER_SERVICE_STAGE_KEYS = (
    "parent_to_worker_ipc_ms",
    "worker_recv_overhead_ms",
    "worker_to_parent_ipc_ms",
)


# ---------------------------------------------------------------------------
# Bayesian Markov burst model: states keyed on worker-service ms bins.
#
# The Markov state is the worker-service bin, because this is the signal that
# drives realtime worker occupancy and observation drops. The fitted joint
# emission distribution carries both action-ready delay and worker-service time
# so control staleness and worker occupancy stay correlated without replaying
# traces.
# ---------------------------------------------------------------------------


LATENCY_MS_BIN_WIDTH = 1.0
EMISSION_QUANTILE_LEVELS = tuple(index / 100.0 for index in range(101))


def build_bayes_markov_burst_samples(
    action_records: list[dict[str, Any]],
    episode_metrics_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record, episode_id in _action_records_with_episode_ids(action_records, episode_metrics_records):
        action_ready_ms = _action_ready_ms(record)
        worker_service_ms = _worker_service_ms(record)
        samples.append(
            {
                "episode_id": episode_id,
                "sample_index": len(samples),
                "action_id": int(record["action_id"]),
                "obs_env_step": int(record["obs_env_step"]),
                "ready_raw_frame": int(record["ready_raw_frame"]),
                "applied": bool(record["applied"]),
                "dropped": bool(record["dropped"]),
                "action_ready_ms": action_ready_ms,
                "worker_service_ms_bin": _latency_ms_bin(worker_service_ms),
                "worker_service_ms": worker_service_ms,
            }
        )
    return samples


def build_bayes_markov_burst_model(samples: list[dict[str, Any]]) -> dict[str, Any]:
    initial_state_counts: Counter[str] = Counter()
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    run_lengths_by_state: dict[str, list[int]] = defaultdict(list)
    state_counts: Counter[str] = Counter()
    emission_samples_by_state: dict[str, list[dict[str, float]]] = defaultdict(list)

    samples_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        samples_by_episode[int(sample["episode_id"])].append(sample)
        state = _worker_service_state(sample)
        state_counts[state] += 1
        emission_samples_by_state[state].append(
            {
                "action_ready_ms": float(sample["action_ready_ms"]),
                "worker_service_ms": float(sample["worker_service_ms"]),
            }
        )

    for episode_id in sorted(samples_by_episode):
        episode_samples = sorted(samples_by_episode[episode_id], key=lambda item: int(item["sample_index"]))
        states = [_worker_service_state(sample) for sample in episode_samples]
        initial_state_counts[states[0]] += 1
        for index in range(len(states) - 1):
            transition_counts[states[index]][states[index + 1]] += 1
        for state, run_length in _runs_by_key(episode_samples, _worker_service_state):
            run_lengths_by_state[state].append(run_length)

    return {
        "schema_version": 5,
        "model_type": "bayes_markov",
        "reset_scope": "episode",
        "state_key": "worker_service_ms_bin",
        "bin_width_ms": LATENCY_MS_BIN_WIDTH,
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
        "emission_distributions_by_state": {
            state: _fit_joint_emission_distribution(values)
            for state, values in sorted(emission_samples_by_state.items())
        },
        "state_counts": dict(sorted(state_counts.items())),
        "terminal_transition_policy": "initial",
    }


def _worker_service_state(sample: dict[str, Any]) -> str:
    return str(int(sample["worker_service_ms_bin"]))


def _runs_by_key(
    samples: list[dict[str, Any]],
    state_fn: Callable[[dict[str, Any]], str],
) -> list[tuple[str, int]]:
    runs: list[tuple[str, int]] = []
    current_state = state_fn(samples[0])
    current_length = 0
    for sample in samples:
        state = state_fn(sample)
        if state == current_state:
            current_length += 1
        else:
            runs.append((current_state, current_length))
            current_state = state
            current_length = 1
    runs.append((current_state, current_length))
    return runs


def _worker_service_ms(record: dict[str, Any]) -> float:
    profile = record["metadata"]["realtime_profile"]
    return float(record["latency"]["latency_ms"]) + sum(
        float(profile[key])
        for key in WORKER_SERVICE_STAGE_KEYS
    )


def _action_ready_ms(record: dict[str, Any]) -> float:
    frame_ms = float(record["metadata"]["frame_ms"])
    return float(record["ready_time_ms"]) - float(record["obs_env_step"]) * frame_ms


def _latency_ms_bin(latency_ms: float) -> int:
    return int(math.floor(float(latency_ms) / LATENCY_MS_BIN_WIDTH))


def _fit_joint_emission_distribution(samples: list[dict[str, float]]) -> dict[str, Any]:
    action_ready_values = [float(sample["action_ready_ms"]) for sample in samples]
    worker_values = [float(sample["worker_service_ms"]) for sample in samples]
    return {
        "count": len(samples),
        "action_ready_ms": _fit_inverse_cdf(action_ready_values),
        "worker_service_ms": _fit_inverse_cdf(worker_values),
        "spearman_rho": _spearman_rho(action_ready_values, worker_values),
    }


def _fit_inverse_cdf(values: list[float]) -> dict[str, Any]:
    return {
        "distribution_type": "inverse_cdf",
        "quantile_levels": list(EMISSION_QUANTILE_LEVELS),
        "latency_ms": [
            percentile_linear(values, level * 100.0)
            for level in EMISSION_QUANTILE_LEVELS
        ],
    }


def _spearman_rho(xs: list[float], ys: list[float]) -> float:
    return _pearson(_ranks(xs), _ranks(ys))


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted((float(value), index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for index in range(cursor, end):
            ranks[ordered[index][1]] = rank
        cursor = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    var_x = sum(value * value for value in centered_x)
    var_y = sum(value * value for value in centered_y)
    if var_x == 0.0 or var_y == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(centered_x, centered_y)) / math.sqrt(var_x * var_y)


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


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


def _probs(counts: Counter[str]) -> dict[str, float]:
    total = sum(int(value) for value in counts.values())
    return {
        state: int(count) / total
        for state, count in sorted(counts.items())
    }
