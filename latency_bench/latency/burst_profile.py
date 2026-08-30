from __future__ import annotations

import math
from statistics import median
from typing import Any

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
# rank pools its own excursion structure across dwells: rank-1 carries the
# degraded worker's near-constant slowness, lower ranks carry whatever mild
# elevation the remaining workers actually showed. Calm emissions are fitted
# independently per worker slot below, so they are not duplicated here.
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
# distribution per regime carries both observation-to-action latency and worker-service
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
    attempt_records: list[dict[str, Any]],
    *,
    measure_session_id: int | None = None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in sorted(attempt_records, key=lambda item: item["obs_id"]):
        if not record["submitted"]:
            continue
        worker_service_time_ms = record["worker_service_time_ms"]
        observation_to_action_latency_ms = record["observation_to_action_latency_ms"]
        sample = {
            "episode_id": record["episode_id"],
            "sample_index": len(samples),
            "obs_id": record["obs_id"],
            "obs_env_step": record["obs_env_step"],
            "worker_id": record["worker_id"],
            "observation_to_action_latency_ms": observation_to_action_latency_ms,
            "worker_service_time_ms_bin": _latency_ms_bin(worker_service_time_ms),
            "worker_service_time_ms": worker_service_time_ms,
        }
        if measure_session_id is not None:
            sample["measure_session_id"] = measure_session_id
        samples.append(sample)
    return samples


def build_hidden_regime_burst_model(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return build_hidden_regime_burst_model_from_sessions([samples])


def build_hidden_regime_burst_model_from_sessions(
    session_samples: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Fit one model while preserving independent measurement boundaries."""
    pooled_samples = [sample for samples in session_samples for sample in samples]
    spike_thresholds_by_slot = _worker_slot_spike_thresholds(session_samples)

    ordered_sessions = [
        sorted(
            samples,
            key=lambda item: item["sample_index"],
        )
        for samples in session_samples
    ]
    session_regimes = [
        _label_shared_regimes(ordered_samples, spike_thresholds_by_slot)
        for ordered_samples in ordered_sessions
    ]

    transition_counts = {"calm": {"calm": 0, "burst": 0}, "burst": {"calm": 0, "burst": 0}}
    regime_step_counts = {"calm": 0, "burst": 0}
    session_dwells: list[list[tuple[int, int]]] = []
    for ordered_samples, regimes in zip(ordered_sessions, session_regimes):
        for sample, regime in zip(ordered_samples, regimes):
            regime_step_counts[regime] += 1
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
    ]

    rank_processes, pre_worker_times, slot_rank_templates = _fit_rank_processes_from_sessions(
        list(zip(ordered_sessions, session_dwells)), spike_thresholds_by_slot
    )

    model = {
        "schema_version": 12,
        "model_type": "hidden_regime",
        "reset_scope": "session",
        "spike_threshold_ms_by_worker_slot": spike_thresholds_by_slot,
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
        "burst_rank_processes": rank_processes,
        "burst_slot_rank_templates": slot_rank_templates,
        "pre_worker_time_ms": (
            _fit_inverse_cdf(pre_worker_times)
            if rank_processes and pre_worker_times
            else None
        ),
    }
    return model


def build_worker_slot_distributions_from_sessions(
    session_samples: list[list[dict[str, Any]]],
    *,
    spike_threshold_ms_by_worker_slot: dict[str, float],
) -> dict[str, Any]:
    """Fit per-slot joint distributions for IID and temporal replay.

    ``all`` contains every observed sample for a slot. ``steady`` contains
    samples in the session-level calm regime; samples inside a burst stay out
    of the temporal baseline even when their service time is below threshold.
    """
    slot_samples: dict[int, dict[str, list[dict[str, float]]]] = {}
    for samples in session_samples:
        ordered_samples = sorted(
            samples,
            key=lambda item: item["sample_index"],
        )
        samples_by_slot: dict[int, list[dict[str, Any]]] = {}
        for sample in ordered_samples:
            samples_by_slot.setdefault(sample["worker_id"], []).append(sample)
        for slot, slot_sample_records in samples_by_slot.items():
            regimes = _label_regimes(
                slot_sample_records,
                spike_threshold_ms_by_worker_slot[str(slot)],
            )
            all_samples = slot_samples.setdefault(slot, {"all": [], "steady": []})
            for sample, regime in zip(slot_sample_records, regimes):
                emission = {
                    "observation_to_action_latency_ms": sample["observation_to_action_latency_ms"],
                    "worker_service_time_ms": sample["worker_service_time_ms"],
                }
                all_samples["all"].append(emission)
                if regime == "calm":
                    all_samples["steady"].append(emission)

    return {
        "schema_version": 3,
        "worker_slots": {
            str(slot): {
                "all": _fit_joint_emission_distribution(values["all"]),
                "steady": (
                    _fit_joint_emission_distribution(values["steady"])
                    if values["steady"]
                    else None
                ),
            }
            for slot, values in sorted(slot_samples.items())
        },
    }


def _fit_rank_processes_from_sessions(
    sessions: list[tuple[list[dict[str, Any]], list[tuple[int, int]]]],
    spike_threshold_ms_by_worker_slot: dict[str, float],
) -> tuple[list[dict[str, Any]], list[float], list[dict[str, Any]]]:
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
    rank_draw_counts: list[int] = []
    rank_spike_counts: list[int] = []
    pre_worker_times = [
        sample["observation_to_action_latency_ms"] - sample["worker_service_time_ms"]
        for samples, _dwells in sessions
        for sample in samples
    ]
    slot_rank_templates: list[dict[str, Any]] = []

    for ordered_samples, dwells in sessions:
        for begin, end in dwells:
            by_worker: dict[int, list[dict[str, Any]]] = {}
            for sample in ordered_samples[begin:end + 1]:
                by_worker.setdefault(sample["worker_id"], []).append(sample)

            def _severity(worker: int) -> tuple[float, int, int]:
                stream = by_worker[worker]
                spikes = sum(
                    1
                    for item in stream
                    if item["worker_service_time_ms"]
                    > spike_threshold_ms_by_worker_slot[str(worker)]
                )
                return (-spikes / len(stream), -spikes, worker)

            ranked_workers = sorted(by_worker, key=_severity)
            slot_rank_templates.append(
                {
                    "dwell_length": end - begin + 1,
                    "slot_to_rank": {
                        str(worker): rank for rank, worker in enumerate(ranked_workers)
                    },
                }
            )
            for rank, worker in enumerate(ranked_workers):
                while len(rank_severities) <= rank:
                    rank_severities.append([])
                    rank_dwell_lengths.append([])
                    rank_levels.append([])
                    rank_residuals.append([])
                    rank_draw_counts.append(0)
                    rank_spike_counts.append(0)
                stream = by_worker[worker]
                spikes = [
                    sample for sample in stream
                    if sample["worker_service_time_ms"]
                    > spike_threshold_ms_by_worker_slot[str(worker)]
                ]
                rank_draw_counts[rank] += len(stream)
                rank_spike_counts[rank] += len(spikes)
                rank_severities[rank].append(len(spikes) / len(stream))
                rank_dwell_lengths[rank].append(end - begin + 1)
                if spikes:
                    level = sum(s["worker_service_time_ms"] for s in spikes) / len(spikes)
                    rank_levels[rank].append(level)
                    rank_residuals[rank].extend(
                        s["worker_service_time_ms"] - level for s in spikes
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
        }
        for rank in range(len(rank_severities))
    ]
    return rank_processes, pre_worker_times, slot_rank_templates


def _worker_slot_spike_thresholds(
    session_samples: list[list[dict[str, Any]]],
) -> dict[str, float]:
    """Keep a prolonged degraded episode from redefining a slot's baseline."""
    values_by_slot_episode: dict[int, dict[tuple[int, int], list[float]]] = {}
    for session_id, samples in enumerate(session_samples):
        for sample in samples:
            episodes = values_by_slot_episode.setdefault(sample["worker_id"], {})
            episodes.setdefault((session_id, sample["episode_id"]), []).append(
                float(sample["worker_service_time_ms"])
            )
    return {
        str(slot): SPIKE_MEDIAN_MULTIPLIER * min(
            median(values) for values in episodes.values()
        )
        for slot, episodes in sorted(values_by_slot_episode.items())
    }


def _label_shared_regimes(
    ordered_samples: list[dict[str, Any]],
    spike_thresholds_by_slot: dict[str, float],
) -> list[str]:
    """Union each slot's burst dwells into one shared machine timeline."""
    shared_regimes = ["calm"] * len(ordered_samples)
    samples_by_slot: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for global_index, sample in enumerate(ordered_samples):
        samples_by_slot.setdefault(sample["worker_id"], []).append(
            (global_index, sample)
        )
    for worker_slot, entries in samples_by_slot.items():
        local_regimes = _label_regimes(
            [sample for _, sample in entries],
            spike_thresholds_by_slot[str(worker_slot)],
        )
        start: int | None = None
        for local_index, regime in enumerate(local_regimes + ["calm"]):
            if regime == "burst" and start is None:
                start = local_index
            elif regime != "burst" and start is not None:
                first_global_index = entries[start][0]
                last_global_index = entries[local_index - 1][0]
                shared_regimes[first_global_index:last_global_index + 1] = [
                    "burst"
                ] * (last_global_index - first_global_index + 1)
                start = None
    return shared_regimes


def _label_regimes(stream_samples: list[dict[str, Any]], spike_threshold_ms: float) -> list[str]:
    regimes = ["calm"] * len(stream_samples)
    last_spike_index: int | None = None
    for index, sample in enumerate(stream_samples):
        if float(sample["worker_service_time_ms"]) <= spike_threshold_ms:
            continue
        if last_spike_index is not None and index - last_spike_index <= BURST_MERGE_GAP_RECORDS:
            start = last_spike_index
        else:
            start = index
        regimes[start:index + 1] = ["burst"] * (index + 1 - start)
        last_spike_index = index
    return regimes


def _latency_ms_bin(latency_ms: float) -> int:
    return int(math.floor(float(latency_ms) / LATENCY_MS_BIN_WIDTH))


def _fit_joint_emission_distribution(samples: list[dict[str, float]]) -> dict[str, Any]:
    observation_to_action_values = [
        float(sample["observation_to_action_latency_ms"]) for sample in samples
    ]
    worker_values = [float(sample["worker_service_time_ms"]) for sample in samples]
    return {
        "count": len(samples),
        "observation_to_action_latency_ms": _fit_inverse_cdf(
            observation_to_action_values
        ),
        "worker_service_time_ms": _fit_inverse_cdf(worker_values),
        "spearman_rho": _spearman_rho(observation_to_action_values, worker_values),
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
