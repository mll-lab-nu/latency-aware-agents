from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Iterator

from latency_bench.core.clock import ms_to_raw_frames


WORKER_SERVICE_STAGE_KEYS = (
    "parent_to_worker_ipc_ms",
    "worker_recv_overhead_ms",
    "worker_to_parent_ipc_ms",
)


# ---------------------------------------------------------------------------
# Bayesian Markov burst model: states keyed on worker_service_raw_frames.
#
# The Markov state is the worker-occupancy length (in frames), because that is
# what drives observation drops in the simulated timeline: a long worker_service
# keeps the worker busy and forces consecutive obs drops. The chain advances once
# per *submitted* observation (in episode order), and emits the joint
# (action_ready, worker_service) record so control staleness and occupancy stay
# correlated. Drops are left emergent in the simulator and are NOT conditioned on.
# ---------------------------------------------------------------------------


def build_bayes_markov_burst_samples(
    action_records: list[dict[str, Any]],
    episode_metrics_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    frame_ms_by_episode = {
        int(record["episode_id"]): float(record["metadata"]["frame_ms"])
        for record in episode_metrics_records
    }
    samples: list[dict[str, Any]] = []
    for record, episode_id in _action_records_with_episode_ids(action_records, episode_metrics_records):
        frame_ms = frame_ms_by_episode[episode_id]
        action_ready_raw_frames = int(record["ready_raw_frame"]) - int(record["obs_env_step"])
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
                "action_ready_raw_frames": action_ready_raw_frames,
                "worker_service_ms": worker_service_ms,
                "worker_service_raw_frames": ms_to_raw_frames(worker_service_ms, env_fps=1000.0 / frame_ms),
            }
        )
    return samples


def build_bayes_markov_burst_model(samples: list[dict[str, Any]]) -> dict[str, Any]:
    initial_state_counts: Counter[str] = Counter()
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    run_lengths_by_state: dict[str, list[int]] = defaultdict(list)
    state_counts: Counter[str] = Counter()
    emissions_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)

    samples_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        samples_by_episode[int(sample["episode_id"])].append(sample)
        state = _worker_service_state(sample)
        state_counts[state] += 1
        emissions_by_state[state].append(
            {
                "action_ready_raw_frames": int(sample["action_ready_raw_frames"]),
                "worker_service_raw_frames": int(sample["worker_service_raw_frames"]),
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
        "schema_version": 3,
        "model_type": "bayes_markov",
        "reset_scope": "episode",
        "state_key": "worker_service_raw_frames",
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
        "emissions_by_state": {
            state: values
            for state, values in sorted(emissions_by_state.items())
        },
        "state_counts": dict(sorted(state_counts.items())),
        "terminal_transition_policy": "initial",
    }


def _worker_service_state(sample: dict[str, Any]) -> str:
    return str(int(sample["worker_service_raw_frames"]))


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
