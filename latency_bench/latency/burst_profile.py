from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterator


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
# Separate measurement sessions remain separate timelines: they share fitted
# distributions but contribute no synthetic boundary transition or dwell.
#
# Within a burst, structure is fitted per WORKER, not on the merged stream.
# Real degradations are worker states: one slow GPU serving every other
# request makes the merged stream alternate spike/normal, so merged-stream
# excursions fit the interleaving artifact (durations ~1) instead of the
# constant-slow worker underneath — and a sampler replaying that artifact
# spreads the slow draws iid across workers, so no single worker saturates
# and the real failure mode (one worker at utilization > 1 dropping half the
# observations) never reproduces. Inside each dwell, workers are ranked by
# severity (in-dwell spike rate — an order statistic, no boundary), and each
# rank pools its own excursion structure and sub-threshold emission across
# dwells: rank-1 carries the degraded worker's near-constant slowness, lower
# ranks carry whatever mild elevation the remaining workers actually showed.
# A binary "participating worker set" cannot express the observed continuum
# (100%/0% single-worker events, 89%/7% degradations, 48%/35% host-wide
# wobble) without a classification boundary; ranks express it directly.
#
# How severe a degradation is belongs to the DWELL, not to any sub-stretch of
# it: observed dwells range from constant-slow (every serve above threshold)
# to mild wobble (a third of serves), and pooling their internal run/gap
# structure regresses every generated dwell to the average severity — the
# constant-slow shape that saturates a worker then never reproduces. Each
# rank therefore carries two per-dwell continuous latents, both drawn once at
# dwell entry: its severity (that rank's above-threshold fraction of serves
# in the dwell) and its level (the dwell's mean above-threshold service).
# Within the dwell each serve is then above threshold with probability
# severity, emitting level plus a draw from the fitted asymmetric residual
# distribution (spike values hug their dwell level with a long right tail; a
# symmetric sigma would leak mass across the frame-boundary below). Run-length
# closure on all fitted devices shows no within-dwell autocorrelation beyond
# what severity carries, so the serve order is memoryless by evidence, not
# assumption; plateaus are dwells whose severity sits near one, not a model
# class. Severity is NOT independent of dwell length: on every fitted device
# the host-wide events are short blips while the long degradations belong to
# a single worker, so lower ranks' severities fall sharply with dwell length
# (sampling them independently hands a long degradation's healthy worker a
# blip's severity and over-generates the fitted in-dwell spike mass 1.4-2x on
# every device). Each rank therefore carries a fitted dwell-length/severity
# Spearman rho, applied through the same Gaussian-copula draw the emission
# joints use. Severity–level and cross-rank severity dependencies tested
# absent, so those latents sample independently. The fitted joint emission
# distribution per regime carries both action-ready delay and worker-service
# time so control staleness and worker occupancy stay correlated.
# ---------------------------------------------------------------------------


LATENCY_MS_BIN_WIDTH = 1.0
# A uniform 1% grid smears its outermost bins: with p99=45ms and max=98ms,
# linear interpolation spreads a full 1% of mass evenly over 45-98ms where
# the device really puts 0.06% — fabricated tail latency that crosses frame
# deadlines. Densified tail levels keep the fitted CDF faithful out to the
# extremes; the tables are self-describing, so samplers need no change.
EMISSION_QUANTILE_LEVELS = tuple(sorted(
    {index / 100.0 for index in range(101)}
    | {index / 1000.0 for index in range(1, 10)}
    | {(990 + index) / 1000.0 for index in range(1, 10)}
    | {0.0001, 0.0005, 0.9995, 0.9999}
))
SPIKE_MEDIAN_MULTIPLIER = 1.25
BURST_MERGE_GAP_RECORDS = 30


def build_burst_samples(
    action_records: list[dict[str, Any]],
    episode_metrics_records: list[dict[str, Any]],
    *,
    measure_session_id: int | None = None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record, episode_id in _action_records_with_episode_ids(action_records, episode_metrics_records):
        action_ready_ms = _action_ready_ms(record)
        worker_service_ms = _worker_service_ms(record)
        sample = {
            "episode_id": episode_id,
            "sample_index": len(samples),
            "action_id": int(record["action_id"]),
            "obs_env_step": int(record["obs_env_step"]),
            "ready_raw_frame": int(record["ready_raw_frame"]),
            "applied": bool(record["applied"]),
            "dropped": bool(record["dropped"]),
            "worker_id": int(record["latency"]["metadata"]["worker_id"]),
            "action_ready_ms": action_ready_ms,
            "worker_service_ms_bin": _latency_ms_bin(worker_service_ms),
            "worker_service_ms": worker_service_ms,
        }
        if measure_session_id is not None:
            sample["measure_session_id"] = measure_session_id
        samples.append(sample)
    return samples


def build_hidden_regime_burst_model(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return build_hidden_regime_burst_model_from_sessions([samples])


def build_hidden_regime_burst_model_from_sessions(
    session_samples: list[list[dict[str, Any]]],
    *,
    reset_scope: str = "session",
) -> dict[str, Any]:
    """Fit one model while preserving independent measurement boundaries."""
    pooled_samples = [sample for samples in session_samples for sample in samples]
    spike_threshold_ms = SPIKE_MEDIAN_MULTIPLIER * median(
        float(sample["worker_service_ms"]) for sample in pooled_samples
    )

    ordered_sessions = [
        sorted(
            samples,
            key=lambda item: (int(item["episode_id"]), int(item["sample_index"])),
        )
        for samples in session_samples
    ]
    session_regimes = [
        _label_regimes(ordered_samples, spike_threshold_ms)
        for ordered_samples in ordered_sessions
    ]

    transition_counts = {"calm": {"calm": 0, "burst": 0}, "burst": {"calm": 0, "burst": 0}}
    regime_step_counts = {"calm": 0, "burst": 0}
    calm_emission_samples: list[dict[str, float]] = []
    session_dwells: list[list[tuple[int, int]]] = []
    for ordered_samples, regimes in zip(ordered_sessions, session_regimes):
        for sample, regime in zip(ordered_samples, regimes):
            regime_step_counts[regime] += 1
            if regime == "calm":
                calm_emission_samples.append(
                    {
                        "action_ready_ms": float(sample["action_ready_ms"]),
                        "worker_service_ms": float(sample["worker_service_ms"]),
                    }
                )
        for index in range(len(regimes) - 1):
            transition_counts[regimes[index]][regimes[index + 1]] += 1
        dwells: list[tuple[int, int]] = []
        start: int | None = None
        for index, regime in enumerate(regimes + ["calm"]):
            if regime == "burst" and start is None:
                start = index
            elif regime != "burst" and start is not None:
                dwells.append((start, index - 1))
                start = None
        session_dwells.append(dwells)
    burst_dwell_lengths = [
        end - begin + 1
        for dwells in session_dwells
        for begin, end in dwells
        if reset_scope == "session" or begin > 0
    ]

    rank_processes, ready_deltas = _fit_rank_processes_from_sessions(
        list(zip(ordered_sessions, session_dwells)), spike_threshold_ms
    )

    initial_regime_counts = {"calm": 0, "burst": 0}
    initial_burst_residual_lengths: list[int] = []
    if reset_scope == "episode":
        for regimes in session_regimes:
            initial_regime_counts[regimes[0]] += 1
            if regimes[0] == "burst":
                residual_length = next(
                    (index for index, regime in enumerate(regimes) if regime != "burst"),
                    len(regimes),
                )
                initial_burst_residual_lengths.append(residual_length)

    model = {
        "schema_version": 9 if reset_scope == "session" else 10,
        "model_type": "hidden_regime",
        "reset_scope": reset_scope,
        "spike_threshold_ms": spike_threshold_ms,
        "spike_median_multiplier": SPIKE_MEDIAN_MULTIPLIER,
        "burst_merge_gap_records": BURST_MERGE_GAP_RECORDS,
        "worker_count": len({int(sample["worker_id"]) for sample in pooled_samples}),
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
            "calm": (
                _fit_joint_emission_distribution(calm_emission_samples)
                if calm_emission_samples
                else None
            ),
        },
        "burst_rank_processes": rank_processes,
        "ready_delta_ms": _fit_inverse_cdf(ready_deltas) if ready_deltas else None,
    }
    if reset_scope == "episode":
        model["initial_regime_counts"] = initial_regime_counts
        model["initial_burst_residual_dwell_distribution"] = (
            _fit_inverse_cdf([float(length) for length in initial_burst_residual_lengths])
            if initial_burst_residual_lengths
            else None
        )
    return model


def _fit_rank_processes_from_sessions(
    sessions: list[tuple[list[dict[str, Any]], list[tuple[int, int]]]],
    spike_threshold_ms: float,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Pool each severity rank's per-dwell latents across dwells.

    Ranking is per dwell: workers present in the dwell ordered by in-dwell
    spike rate (ties by spike count, then worker id). Rank r's pool therefore
    only collects from dwells where at least r + 1 workers served — a 1-draw
    blip contributes to rank 0 alone. Each dwell contributes one severity
    (and, when it spiked, one level plus that dwell's spike residuals) to its
    rank's pools.
    """
    rank_severities: list[list[float]] = []
    rank_dwell_lengths: list[list[float]] = []
    rank_levels: list[list[float]] = []
    rank_residuals: list[list[float]] = []
    rank_emissions: list[list[dict[str, float]]] = []
    rank_draw_counts: list[int] = []
    rank_spike_counts: list[int] = []
    ready_deltas: list[float] = []

    for ordered_samples, dwells in sessions:
        for begin, end in dwells:
            by_worker: dict[int, list[dict[str, Any]]] = {}
            for sample in ordered_samples[begin:end + 1]:
                by_worker.setdefault(int(sample["worker_id"]), []).append(sample)

            def _severity(worker: int) -> tuple[float, int, int]:
                stream = by_worker[worker]
                spikes = sum(
                    1 for item in stream if float(item["worker_service_ms"]) > spike_threshold_ms
                )
                return (-spikes / len(stream), -spikes, worker)

            for rank, worker in enumerate(sorted(by_worker, key=_severity)):
                while len(rank_severities) <= rank:
                    rank_severities.append([])
                    rank_dwell_lengths.append([])
                    rank_levels.append([])
                    rank_residuals.append([])
                    rank_emissions.append([])
                    rank_draw_counts.append(0)
                    rank_spike_counts.append(0)
                stream = by_worker[worker]
                spikes = [
                    sample for sample in stream
                    if float(sample["worker_service_ms"]) > spike_threshold_ms
                ]
                rank_draw_counts[rank] += len(stream)
                rank_spike_counts[rank] += len(spikes)
                rank_severities[rank].append(len(spikes) / len(stream))
                rank_dwell_lengths[rank].append(float(end - begin + 1))
                if spikes:
                    level = sum(float(s["worker_service_ms"]) for s in spikes) / len(spikes)
                    rank_levels[rank].append(level)
                    rank_residuals[rank].extend(
                        float(s["worker_service_ms"]) - level for s in spikes
                    )
                    ready_deltas.extend(
                        float(s["action_ready_ms"]) - float(s["worker_service_ms"])
                        for s in spikes
                    )
                rank_emissions[rank].extend(
                    {
                        "action_ready_ms": float(sample["action_ready_ms"]),
                        "worker_service_ms": float(sample["worker_service_ms"]),
                    }
                    for sample in stream
                    if float(sample["worker_service_ms"]) <= spike_threshold_ms
                )

    rank_processes = [
        {
            "draw_count": rank_draw_counts[rank],
            "spike_count": rank_spike_counts[rank],
            "severity": _fit_inverse_cdf(rank_severities[rank]),
            "dwell_length_spearman_rho": _spearman_rho(
                rank_dwell_lengths[rank], rank_severities[rank]
            ),
            "level_ms": (
                _fit_inverse_cdf(rank_levels[rank]) if rank_levels[rank] else None
            ),
            "level_residual_ms": (
                _fit_inverse_cdf(rank_residuals[rank]) if rank_residuals[rank] else None
            ),
            "emission": (
                _fit_joint_emission_distribution(rank_emissions[rank])
                if rank_emissions[rank]
                else None
            ),
        }
        for rank in range(len(rank_severities))
    ]
    return rank_processes, ready_deltas


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
    # Samples sit at Hazen plotting positions ((i + 0.5) / n) rather than on
    # the [0, 1] endpoints: endpoint pinning gives the extreme order statistic
    # essentially zero quantile mass, which for a small heavy-tailed sample
    # (45 burst dwells, one 2631-draw degradation) smears the observed extreme
    # into fabricated mid-range values and shrinks the resampled mean by a
    # third. With positions clamped past (n - 0.5) / n, each sample keeps its
    # 1/n share and the maximum stays a real plateau.
    ordered = sorted(values)
    n = len(ordered)

    def _inverse(level: float) -> float:
        position = level * n - 0.5
        if position <= 0.0:
            return ordered[0]
        if position >= n - 1.0:
            return ordered[-1]
        index = int(position)
        fraction = position - index
        return ordered[index] + fraction * (ordered[index + 1] - ordered[index])

    return {
        "distribution_type": "inverse_cdf",
        "quantile_levels": list(EMISSION_QUANTILE_LEVELS),
        "latency_ms": [_inverse(level) for level in EMISSION_QUANTILE_LEVELS],
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
