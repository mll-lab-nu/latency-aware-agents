"""Sample-based scores for ordered windows and session-level timing diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance

from scripts.experiments.temporal_profiles.compare import burst_lengths, session_acf

HORIZONS = (8, 32, 128)


def windows(values, length, count, seed):
    """Windows stay inside one session; overlaps are allowed, not independent units."""
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(values) - length + 1, size=min(count, len(values) - length + 1), replace=False)
    return values[starts[:, None] + np.arange(length)]


def window_scores(observed, generated, center, scale):
    """Fair ensemble energy / variogram scores; lower is better.

    Each row of generated must come from an independent generated session.
    Energy uses off-diagonal ensemble pairs. Variogram (p=.5) subtracts the
    finite-ensemble variance term. These standard corrections are unbiased for
    IID members; the study's length-stratified sessions make that an approximation.
    Scale is learned from fitting data only.
    """
    y = (observed - center) / scale
    x = (generated - center) / scale
    m, length = x.shape
    if m < 2:
        raise ValueError("At least two independent generated windows are required")
    energy = (cdist(y, x).mean(axis=1) - .5 * pdist(x).mean()) / np.sqrt(length)
    variogram = np.zeros(len(y))
    weight_sum = 0.
    for lag in (1, 2, 4, 8, 16, 32, 64):
        if lag >= length:
            continue
        xdiff = np.sqrt(np.abs(x[:, lag:] - x[:, :-lag]))
        ydiff = np.sqrt(np.abs(y[:, lag:] - y[:, :-lag]))
        expected = xdiff.mean(axis=0)
        correction = xdiff.var(axis=0, ddof=1) / m
        variogram += (((ydiff - expected) ** 2 - correction).sum(axis=1)) / lag
        weight_sum += (length - lag) / lag
    return {"energy": float(energy.mean()), "variogram": float((variogram / weight_sum).mean())}


def evaluate_windows(sessions, ensembles, center, scale, seed, horizons=HORIZONS):
    rows = []
    for session in sessions:
        row = {"run_id": session["run_id"]}
        for length in horizons:
            observed = windows(session["pairs"][:, 0], length, 64, seed + session["run_id"] + length)
            for metric, value in window_scores(observed, ensembles[length], center, scale).items():
                row[f"{metric}_{length}"] = value
        row["window_score"] = float(np.mean([row[f"{metric}_{length}"]
            for metric in ("energy", "variogram") for length in horizons]))
        rows.append(row)
    return rows


def timing_diagnostics(values, reference, slow_cutoff, frame_ms):
    slow = values > slow_cutoff
    slow_spells = burst_lengths(values, slow_cutoff)
    frames = np.ceil(values / frame_ms - 1e-10)
    late = frames > 1
    late_spells = burst_lengths(frames, 1)
    max_lag = min(64, len(values) - 1, len(reference) - 1)
    acf = session_acf([values], max_lag)
    real_acf = session_acf([reference], max_lag)
    result = {
        "mean_ms": float(values.mean()), "p95_ms": float(np.quantile(values, .95)),
        "p99_ms": float(np.quantile(values, .99)),
        "distribution_error_ms": float(wasserstein_distance(values, reference)),
        "slow_fraction": float(slow.mean()),
        "slow_pair_fraction": float(np.mean(slow[:-1] & slow[1:])),
        "slow_spells_per_1000": float(1000 * len(slow_spells) / len(values)),
        "max_slow_spell": int(slow_spells.max()) if len(slow_spells) else 0,
        "lag1": float(acf[0]) if np.isfinite(acf[0]) else None,
        "acf_error": float(np.sqrt(np.mean((acf - real_acf) ** 2)))
        if np.isfinite(acf).all() and np.isfinite(real_acf).all() else None,
        "miss_1_frame": float(late.mean()), "miss_2_frames": float(np.mean(frames > 2)),
        "adjacent_misses": float(np.mean(late[:-1] & late[1:])),
        "max_miss_streak": int(late_spells.max()) if len(late_spells) else 0,
    }
    for length in (8, 32, 128):
        # Full nonoverlapping count windows; no bridging session boundaries.
        count = len(values) // length
        actual_count = len(reference) // length
        if count == 0 or actual_count == 0:
            result[f"slow_count_error_{length}"] = None
            continue
        generated_counts = slow[:count * length].reshape(count, length).sum(axis=1)
        real_counts = (reference[:actual_count * length] > slow_cutoff).reshape(actual_count, length).sum(axis=1)
        result[f"slow_count_error_{length}"] = float(wasserstein_distance(generated_counts, real_counts))
    return result


def bootstrap_indices(repetitions, runs, draws=2000, seed=20261108):
    rng = np.random.default_rng(seed)
    return (rng.integers(repetitions, size=(draws, repetitions)),
            rng.integers(runs, size=(draws, runs)))


def interval(matrix, indices):
    """Crossed bootstrap over independent real runs and generation repetitions."""
    repetitions, runs = indices
    samples = matrix[repetitions[:, :, None], runs[:, None, :]].mean(axis=(1, 2))
    return {"mean": float(matrix.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(),
            "generation_sd": float(matrix.mean(axis=1).std(ddof=1))}
