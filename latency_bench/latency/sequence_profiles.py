"""Fit alternative timing generators from independent, ordered sessions.

Both alternatives currently model one inference worker (slot 0). Fitting never
joins measurement sessions or treats rejected observations as timing samples.
"""
from __future__ import annotations

import numpy as np


def fit_hmm(sessions, n_components=2, max_iter=150):
    """Full Gaussian HMM on standardized log busy time and log overhead.

    Returns the estimator, standardization and portable JSON parameters. The
    estimator is only needed during fitting; runtime replay needs just NumPy.
    """
    from hmmlearn.hmm import GaussianHMM
    from threadpoolctl import threadpool_limits

    if n_components < 1 or max_iter < 2:
        raise ValueError("HMM needs at least one state and two fitting iterations")
    pairs = np.concatenate([s["pairs"] for s in sessions])
    if not np.isfinite(pairs).all() or np.any(pairs[:, 1] <= 0) or np.any(pairs[:, 0] <= pairs[:, 1]):
        raise ValueError("Log-overhead HMM requires finite action latency > worker busy time > 0")
    features = np.log(np.column_stack((pairs[:, 1], pairs[:, 0] - pairs[:, 1])))
    center, scale = features.mean(axis=0), features.std(axis=0)
    scale = np.where(scale == 0, 1., scale)
    features = (features - center) / scale
    lengths = [len(s["pairs"]) for s in sessions]
    trials, candidates = [], []
    with threadpool_limits(limits=1):
        for seed in (7, 42, 2026):
            model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=max_iter,
                                tol=0.01, random_state=seed, implementation="log")
            model.fit(features, lengths=lengths)
            score = float(model.score(features, lengths=lengths))
            if not np.isfinite(score):
                raise ValueError(f"Nonfinite HMM fit likelihood for restart {seed}")
            history = list(model.monitor_.history)
            gain = float(history[-1] - history[-2])
            trials.append({"seed": seed, "fit_log_likelihood": score,
                           "iterations": model.monitor_.iter, "final_likelihood_gain": gain,
                           "reached_tolerance": abs(gain) < model.tol})
            candidates.append(model)
    best = int(np.argmax([t["fit_log_likelihood"] for t in trials]))
    model = candidates[best]
    parameters = {"n_components": n_components,
                  "features": ["log(worker_busy_ms)", "log(action_latency_ms - worker_busy_ms)"],
                  "center": center.tolist(), "scale": scale.tolist(),
                  "startprob": model.startprob_.tolist(), "transmat": model.transmat_.tolist(),
                  "means": model.means_.tolist(), "covars": model.covars_.tolist(),
                  "trials": trials, "selected_seed": trials[best]["seed"]}
    return model, center, scale, parameters


def build_sequence_model(session_samples, *, method, hmm_states=2, hmm_max_iterations=1000,
                         block_mean_length=32):
    sessions = []
    for samples in session_samples:
        if not samples:
            continue
        if {sample["worker_id"] for sample in samples} != {0}:
            raise ValueError("HMM/block profiles currently require one inference worker, slot 0")
        pairs = np.array([[s["observation_to_action_latency_ms"], s["worker_service_time_ms"]]
                          for s in samples], dtype=float)
        if (not np.isfinite(pairs).all() or np.any(pairs < 0)
                or np.any(pairs[:, 0] < pairs[:, 1])):
            raise ValueError("Timing pairs must be finite with action latency >= busy time >= 0")
        sessions.append({"pairs": pairs})
    if not sessions:
        raise ValueError("Cannot fit a profile without admitted timing samples")
    common = {"schema_version": 1, "model_type": method, "reset_scope": "session",
              "worker_slots": [0], "sequence_unit": "admitted_observation",
              "fit_session_lengths": [len(s["pairs"]) for s in sessions],
              "observed_max_latency_ms": float(max(s["pairs"][:, 0].max() for s in sessions))}
    if method == "hmm":
        _, _, _, parameters = fit_hmm(sessions, hmm_states, hmm_max_iterations)
        return {**common, **parameters, "emission_support": "unbounded_positive"}
    if method == "block":
        if not np.isfinite(block_mean_length) or block_mean_length < 1:
            raise ValueError("Block mean length must be finite and >= 1")
        return {**common, "mean_block_length": float(block_mean_length),
                "boundary_policy": "circular_within_session",
                "pair_fields": ["observation_to_action_latency_ms", "worker_service_time_ms"],
                "sessions": [s["pairs"].tolist() for s in sessions]}
    raise ValueError(f"Unknown sequence profile method: {method}")
