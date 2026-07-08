from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterator

from latency_bench.utils.stats import percentile_linear


WORKER_SERVICE_STAGE_KEYS = (
    "parent_to_worker_ipc_ms",
    "worker_recv_overhead_ms",
    "worker_to_parent_ipc_ms",
)


# ---------------------------------------------------------------------------
# Hidden-regime burst model: a two-state (calm/burst) hidden Markov chain over
# worker-service latency, with excursions modeled as objects inside bursts.
#
# Real latency spikes arrive in clusters with normal-valued samples in
# between, so a chain keyed on the latency value itself forgets it is inside a
# burst as soon as one normal sample lands. The hidden regime carries that
# memory. Fitting is a deterministic labeling pass (no EM): samples above
# ``SPIKE_MEDIAN_MULTIPLIER x median`` are spikes, spikes within
# ``BURST_MERGE_GAP_RECORDS`` of each other belong to one burst, and every
# record from a burst's first to last spike (including the interleaved normal
# ones) is labeled burst.
#
# The regime is machine state, not episode state: a degraded machine stays
# degraded through env resets (observed as one slow condition spanning tens of
# consecutive episodes). Labeling therefore treats the whole sample stream as
# one machine timeline — bursts, excursions, and transitions bridge episode
# boundaries — and the sampler carries the regime across episode resets.
#
# Within a burst, each maximal run of consecutive above-threshold samples is
# one excursion, described by two continuous latent parameters: its duration
# in draws and its level (mean worker-service time). Drawing each spike iid
# from a pooled distribution can produce isolated spikes but never a sustained
# plateau (a 16-draw all-elevated run has probability p^16), while real
# devices show both shapes; the fitted joint (duration, level) distribution
# carries whatever mass the device actually exhibits — tall-and-short,
# mild-and-long, or anything else — without any discrete spike/plateau
# taxonomy or setup-specific boundary. The fitted joint emission distribution
# per regime carries both action-ready delay and worker-service time so
# control staleness and worker occupancy stay correlated.
# ---------------------------------------------------------------------------


LATENCY_MS_BIN_WIDTH = 1.0
EMISSION_QUANTILE_LEVELS = tuple(index / 100.0 for index in range(101))
SPIKE_MEDIAN_MULTIPLIER = 1.25
BURST_MERGE_GAP_RECORDS = 30


def build_burst_samples(
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


def build_hidden_regime_burst_model(samples: list[dict[str, Any]]) -> dict[str, Any]:
    spike_threshold_ms = SPIKE_MEDIAN_MULTIPLIER * median(
        float(sample["worker_service_ms"]) for sample in samples
    )

    ordered_samples = sorted(
        samples, key=lambda item: (int(item["episode_id"]), int(item["sample_index"]))
    )
    regimes = _label_regimes(ordered_samples, spike_threshold_ms)

    transition_counts = {"calm": {"calm": 0, "burst": 0}, "burst": {"calm": 0, "burst": 0}}
    regime_step_counts = {"calm": 0, "burst": 0}
    burst_dwell_lengths: list[int] = []
    emission_samples_by_regime: dict[str, list[dict[str, float]]] = {"calm": [], "burst": []}
    excursion_runs: list[list[dict[str, Any]]] = []
    excursion_gaps: list[int] = []
    pending_gap: int | None = None

    active_run: list[dict[str, Any]] = []
    for sample, regime in zip(ordered_samples, regimes):
        regime_step_counts[regime] += 1
        if float(sample["worker_service_ms"]) > spike_threshold_ms:
            # A gap is the sub-threshold stretch between two excursions of
            # the SAME burst (>= 1 by maximality); it carries the fitted
            # within-burst spike density.
            if pending_gap:
                excursion_gaps.append(pending_gap)
            pending_gap = None
            active_run.append(sample)
            continue
        if active_run:
            excursion_runs.append(active_run)
            active_run = []
            pending_gap = 0
        if pending_gap is not None:
            pending_gap = pending_gap + 1 if regime == "burst" else None
        emission_samples_by_regime[regime].append(
            {
                "action_ready_ms": float(sample["action_ready_ms"]),
                "worker_service_ms": float(sample["worker_service_ms"]),
            }
        )
    if active_run:
        excursion_runs.append(active_run)
    for index in range(len(regimes) - 1):
        transition_counts[regimes[index]][regimes[index + 1]] += 1
    length = 0
    for regime in regimes + ["calm"]:
        if regime == "burst":
            length += 1
        elif length:
            burst_dwell_lengths.append(length)
            length = 0

    return {
        "schema_version": 8,
        "model_type": "hidden_regime",
        "reset_scope": "session",
        "spike_threshold_ms": spike_threshold_ms,
        "spike_median_multiplier": SPIKE_MEDIAN_MULTIPLIER,
        "burst_merge_gap_records": BURST_MERGE_GAP_RECORDS,
        "regime_transition_counts": transition_counts,
        "regime_step_counts": regime_step_counts,
        "burst_dwell_lengths": sorted(burst_dwell_lengths),
        # Burst dwell is a fitted continuous duration, sampled fresh at each
        # burst entry. A geometric (per-draw exit) dwell rejects the observed
        # mixture of draw-scale blips and minutes-long degradations.
        "burst_dwell_distribution": (
            _fit_inverse_cdf([float(length) for length in burst_dwell_lengths])
            if burst_dwell_lengths
            else None
        ),
        "emission_distributions_by_regime": {
            regime: (_fit_joint_emission_distribution(values) if values else None)
            for regime, values in emission_samples_by_regime.items()
        },
        "excursions": _fit_excursion_model(excursion_runs, excursion_gaps),
    }


def _fit_excursion_model(
    excursion_runs: list[list[dict[str, Any]]],
    excursion_gaps: list[int],
) -> dict[str, Any] | None:
    if not excursion_runs:
        return None
    durations = [float(len(run)) for run in excursion_runs]
    levels = [
        sum(float(sample["worker_service_ms"]) for sample in run) / len(run)
        for run in excursion_runs
    ]
    ready_deltas = [
        float(sample["action_ready_ms"]) - float(sample["worker_service_ms"])
        for run in excursion_runs
        for sample in run
    ]
    residual_square_sum = 0.0
    residual_dof = 0
    for run, level in zip(excursion_runs, levels):
        if len(run) < 2:
            continue
        residual_square_sum += sum(
            (float(sample["worker_service_ms"]) - level) ** 2 for sample in run
        )
        residual_dof += len(run) - 1
    return {
        "count": len(excursion_runs),
        "duration_draws": _fit_inverse_cdf(durations),
        "level_ms": _fit_inverse_cdf(levels),
        "duration_level_spearman_rho": _spearman_rho(durations, levels),
        "within_excursion_sigma_ms": (
            math.sqrt(residual_square_sum / residual_dof) if residual_dof else 0.0
        ),
        "ready_delta_ms": _fit_inverse_cdf(ready_deltas),
        "gap_draws": (
            _fit_inverse_cdf([float(gap) for gap in excursion_gaps])
            if excursion_gaps
            else None
        ),
    }


def _label_regimes(stream_samples: list[dict[str, Any]], spike_threshold_ms: float) -> list[str]:
    regimes = ["calm"] * len(stream_samples)
    last_spike_index: int | None = None
    for index, sample in enumerate(stream_samples):
        if float(sample["worker_service_ms"]) <= spike_threshold_ms:
            continue
        if last_spike_index is not None and index - last_spike_index <= BURST_MERGE_GAP_RECORDS:
            start = last_spike_index
        else:
            start = index
        regimes[start:index + 1] = ["burst"] * (index + 1 - start)
        last_spike_index = index
    return regimes


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


