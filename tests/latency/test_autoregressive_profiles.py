"""Statistical and replay contracts for the AR timing baselines."""
from copy import deepcopy

import numpy as np
import pytest

from latency_bench.core.clock import EnvClock
from latency_bench.latency.autoregressive import fit_autoregressive
from latency_bench.latency.sequence_samplers import AutoregressiveSampler


def known_model(method):
    return {"model_type": method, "intercept": [0., 0.], "ar": [.7, .5],
            "ma": [.35, .2] if method == "arma" else [0., 0.],
            "innovation_covariance": [[.09, .015], [.015, .04]],
            "center": [4., 1.], "scale": [1., 1.]}


def draw(model, seed, n):
    sampler = AutoregressiveSampler(model, clock=EnvClock(env_fps=30, obs_fps=30), seed=seed)
    samples = [sampler.sample({"worker_slot": 0, "episode_id": i // 100}) for i in range(n)]
    return np.array([[s.latency_ms, s.worker_service_ms] for s in samples])


@pytest.mark.parametrize("method", ["ar1", "arma"])
def test_fit_recovers_known_process_and_restarts_at_session_boundaries(method):
    model = known_model(method)
    sessions = [{"pairs": draw(model, seed, 1500)} for seed in (7, 82, 401)]
    fitted = fit_autoregressive(sessions, method=method)
    np.testing.assert_allclose(fitted["ar"], model["ar"], atol=.08)
    np.testing.assert_allclose(fitted["ma"], model["ma"], atol=.1)
    reversed_fit = fit_autoregressive(list(reversed(sessions)), method=method)
    np.testing.assert_allclose(reversed_fit["ar"], fitted["ar"], atol=.002)
    np.testing.assert_allclose(reversed_fit["ma"], fitted["ma"], atol=.002)
    assert all(fit["trials"][fit["selected_restart"]]["converged"] for fit in fitted["fits"])
    pairs = draw({**model, **fitted}, 42, 1000)
    assert np.all(pairs[:, 0] > pairs[:, 1]) and np.all(pairs[:, 1] > 0)


def test_sampler_initializes_stationary_and_deepcopies_stream_without_episode_reset():
    model = known_model("arma")
    clock = EnvClock(env_fps=10, obs_fps=10)
    sampler = AutoregressiveSampler(model, clock=clock, seed=42)
    clone = deepcopy(sampler)
    for i in range(30):
        assert sampler.sample({"worker_slot": 0, "episode_id": i}) == clone.sample({"worker_slot": 0, "episode_id": 0})
    # Stationary initial draws must have the same variance as the infinite MA
    # expansion, rather than beginning every generated session at its mean.
    values = np.array([AutoregressiveSampler(model, clock=clock, seed=i).value for i in range(3000)])
    ar, ma = np.array(model["ar"]), np.array(model["ma"])
    expected = np.array(model["innovation_covariance"]) * (1 + np.outer(ar + ma, ar + ma) / (1 - np.outer(ar, ar)))
    np.testing.assert_allclose(np.cov(values.T), expected, atol=.025)


def test_fit_rejects_degenerate_or_too_short_series():
    with pytest.raises(ValueError, match=">=32"):
        fit_autoregressive([{"pairs": np.array([[10., 8.]] * 10)}], method="arma")
    with pytest.raises(ValueError, match="nonconstant"):
        fit_autoregressive([{"pairs": np.array([[10., 8.]] * 100)}], method="ar1")
