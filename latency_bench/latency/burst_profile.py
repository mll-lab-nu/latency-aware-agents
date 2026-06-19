from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterator

from latency_bench.core.clock import ms_to_raw_frames


WORKER_SERVICE_STAGE_KEYS = (
    "parent_to_worker_ipc_ms",
    "worker_recv_overhead_ms",
    "worker_to_parent_ipc_ms",
)


# ---------------------------------------------------------------------------
# Run-length semi-Markov (burst model v1): states keyed on latency_raw_frames
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Queue-conditioned (burst model v2): iid draw keyed on recent_drop_count_capped
# ---------------------------------------------------------------------------


def build_queue_conditioned_burst_samples(
    action_records: list[dict[str, Any]],
    episode_metrics_records: list[dict[str, Any]],
    step_records: list[dict[str, Any]],
    recent_drop_count_cap: int = 2,
) -> list[dict[str, Any]]:
    frame_ms_by_episode = {
        int(record["episode_id"]): float(record["metadata"]["frame_ms"])
        for record in episode_metrics_records
    }
    recent_drop_count_by_episode_step = _recent_drop_count_by_episode_step(
        step_records=step_records,
        cap=recent_drop_count_cap,
    )
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
                "apply_raw_frame": record["apply_raw_frame"],
                "applied": bool(record["applied"]),
                "dropped": bool(record["dropped"]),
                "recent_drop_count_capped": recent_drop_count_by_episode_step[episode_id][int(record["obs_env_step"])],
                "action_ready_ms": float(action_ready_raw_frames) * frame_ms,
                "action_ready_raw_frames": action_ready_raw_frames,
                "worker_service_ms": worker_service_ms,
                "worker_service_raw_frames": ms_to_raw_frames(worker_service_ms, env_fps=1000.0 / frame_ms),
                "inference_latency_ms": float(record["latency"]["latency_ms"]),
            }
        )
    return samples


def build_queue_conditioned_burst_model(samples: list[dict[str, Any]]) -> dict[str, Any]:
    state_counts: Counter[str] = Counter()
    samples_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for sample in samples:
        state = str(int(sample["recent_drop_count_capped"]))
        state_counts[state] += 1
        samples_by_state[state].append(
            {
                "action_ready_ms": float(sample["action_ready_ms"]),
                "action_ready_raw_frames": int(sample["action_ready_raw_frames"]),
                "worker_service_ms": float(sample["worker_service_ms"]),
                "worker_service_raw_frames": int(sample["worker_service_raw_frames"]),
                "inference_latency_ms": float(sample["inference_latency_ms"]),
                "applied": bool(sample["applied"]),
                "dropped": bool(sample["dropped"]),
            }
        )

    return {
        "schema_version": 2,
        "model_type": "queue_conditioned",
        "reset_scope": "episode",
        "state_key": "recent_drop_count_capped",
        "recent_drop_count_cap": max(int(state) for state in state_counts),
        "state_counts": dict(sorted(state_counts.items())),
        "samples_by_state": {
            state: values
            for state, values in sorted(samples_by_state.items())
        },
    }


def _recent_drop_count_by_episode_step(
    *,
    step_records: list[dict[str, Any]],
    cap: int,
) -> dict[int, dict[int, int]]:
    steps_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in step_records:
        steps_by_episode[int(record["episode_id"])].append(record)

    recent_drop_count_by_episode_step: dict[int, dict[int, int]] = defaultdict(dict)
    for episode_id, records in steps_by_episode.items():
        recent_drop_count = 0
        for record in sorted(records, key=lambda item: int(item["env_step"])):
            info = record["info"]
            if "observation_dropped" in info and info["observation_dropped"]:
                recent_drop_count += 1
            elif "observation_submitted" in info and info["observation_submitted"]:
                recent_drop_count_by_episode_step[episode_id][int(record["env_step"])] = min(
                    recent_drop_count,
                    cap,
                )
                recent_drop_count = 0
    return recent_drop_count_by_episode_step


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
