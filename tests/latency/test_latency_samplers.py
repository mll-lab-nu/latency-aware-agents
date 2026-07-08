from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import LatencyRecord
from latency_bench.latency.burst_profile import build_hidden_regime_burst_model
from latency_bench.latency.profile import (
    build_inverse_cdf_distribution,
    save_latency_distribution,
    save_profile_manifest,
    write_latency_samples,
)
from latency_bench.latency.samplers import (
    FixedSampler,
    HiddenRegimeSampler,
    NormalSampler,
    UniformSampler,
    ZeroSampler,
    build_sampler,
)
from latency_bench.latency.trace import load_latency_trace, save_latency_trace


def _clock() -> EnvClock:
    return EnvClock(env_fps=25.0, obs_fps=25.0)


def _profile_sampler(profile_dir: Path, *, seed=0, sampler: str = "profile_default"):
    return build_sampler(
        {"type": "profile_sample", "profile_path": str(profile_dir / "profile.json"), "sampler": sampler},
        clock=_clock(),
        seed=seed,
    )


def _write_latency_profile(
    tmp_path: Path,
    samples: list[float] | None = None,
    *,
    trace_path: str = "latencies.jsonl",
    write_trace: bool = True,
    write_burst_model: bool = True,
    worker_service_sequence: list[tuple[int, int]] | None = None,
) -> Path:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    samples = [10.0, 20.0, 30.0] if samples is None else samples
    if write_trace:
        write_latency_samples(samples, profile_dir / trace_path)
    save_latency_distribution(
        build_inverse_cdf_distribution(samples),
        profile_dir / "latency_distribution.json",
    )
    worker_service_sequence = worker_service_sequence or [(1, 1), (2, 2), (3, 3)]
    model_samples = [
        {
            "episode_id": 0,
            "sample_index": index,
            "action_ready_ms": float(action_ready) * 40.0,
            "worker_service_ms": float(worker_service) * 40.0,
            "worker_service_ms_bin": int(float(worker_service) * 40.0),
        }
        for index, (worker_service, action_ready) in enumerate(worker_service_sequence)
    ]
    if write_burst_model:
        (profile_dir / "latency_burst_model.json").write_text(
            json.dumps(build_hidden_regime_burst_model(model_samples), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "profile_id": "L_test",
        "policy_id": "pi_test",
        "env_id": "env_test",
        "target_device": "test_gpu",
        "env_fps": 25.0,
        "frame_ms": 40.0,
        "n_samples": len(samples),
        "trace_path": trace_path,
        "distribution_path": "latency_distribution.json",
        "summary": {"mean_ms": 20.0, "std_ms": 8.0, "min_ms": 10.0, "max_ms": 30.0},
    }
    if write_burst_model:
        manifest["latency_method"] = "temporal"
        manifest["sample_model_type"] = "hidden_regime"
        manifest["burst_model_path"] = "latency_burst_model.json"
    save_profile_manifest(manifest, profile_dir / "profile.json")
    return profile_dir


def _constant_inverse_cdf(value: float) -> dict:
    return {
        "distribution_type": "inverse_cdf",
        "quantile_levels": [0.0, 1.0],
        "latency_ms": [value, value],
    }


def _constant_joint_emission(worker_service_ms: float, action_ready_ms: float | None = None) -> dict:
    return {
        "count": 10,
        "action_ready_ms": _constant_inverse_cdf(
            worker_service_ms if action_ready_ms is None else action_ready_ms
        ),
        "worker_service_ms": _constant_inverse_cdf(worker_service_ms),
        "spearman_rho": 0.0,
    }


def _constant_excursions(
    *,
    duration: float,
    level_ms: float,
    ready_delta_ms: float = 0.0,
    sigma_ms: float = 0.0,
) -> dict:
    return {
        "count": 5,
        "duration_draws": _constant_inverse_cdf(duration),
        "level_ms": _constant_inverse_cdf(level_ms),
        "duration_level_spearman_rho": 0.0,
        "within_excursion_sigma_ms": sigma_ms,
        "ready_delta_ms": _constant_inverse_cdf(ready_delta_ms),
        "start_counts": {"start": 1000, "hold": 0},
    }


def _hand_model(
    *,
    excursions: dict | None,
    burst_emission: dict | None,
    enter_burst_count: int = 1000,
) -> dict:
    return {
        "schema_version": 7,
        "model_type": "hidden_regime",
        "reset_scope": "episode",
        "spike_threshold_ms": 50.0,
        "regime_transition_counts": {
            "calm": {"calm": 1000 - enter_burst_count, "burst": enter_burst_count},
            "burst": {"calm": 1000, "burst": 0},
        },
        "regime_step_counts": {"calm": 1000, "burst": 1000},
        "burst_dwell_lengths": [],
        "emission_distributions_by_regime": {
            "calm": _constant_joint_emission(40.0),
            "burst": burst_emission,
        },
        "excursions": excursions,
    }


def test_zero_sampler_returns_zero():
    assert ZeroSampler(clock=_clock()).sample().latency_ms == 0.0


def test_fixed_sampler_returns_constant():
    sampler = FixedSampler(42.5, clock=_clock())
    assert sampler.sample().latency_ms == 42.5
    assert sampler.sample().latency_ms == 42.5


def test_normal_sampler_is_deterministic_with_seed():
    first = NormalSampler(100.0, 20.0, clock=_clock(), seed=0)
    second = NormalSampler(100.0, 20.0, clock=_clock(), seed=0)
    assert [first.sample().latency_ms for _ in range(5)] == [second.sample().latency_ms for _ in range(5)]


def test_normal_sampler_clamps_to_min_and_max():
    low = NormalSampler(5.0, 0.0, clock=_clock(), min_latency_ms=10.0)
    high = NormalSampler(100.0, 0.0, clock=_clock(), max_latency_ms=50.0)
    assert low.sample().latency_ms == 10.0
    assert high.sample().latency_ms == 50.0


def test_uniform_sampler_is_deterministic_within_bounds():
    first = UniformSampler(10.0, 20.0, clock=_clock(), seed=0)
    second = UniformSampler(10.0, 20.0, clock=_clock(), seed=0)
    first_values = [first.sample().latency_ms for _ in range(10)]
    assert first_values == [second.sample().latency_ms for _ in range(10)]
    assert all(10.0 <= value <= 20.0 for value in first_values)


def test_fixed_from_profile_uses_summary_mean(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    sampler = build_sampler(
        {"method": "fixed", "profile_path": str(profile_dir / "profile.json")},
        clock=_clock(),
        seed=None,
    )
    assert sampler.sample().latency_ms == 20.0


def test_normal_from_profile_uses_summary_mean_and_std(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    sampler = build_sampler(
        {"method": "normal", "profile_path": str(profile_dir / "profile.json"), "seed": 0},
        clock=_clock(),
        seed=0,
    )
    assert isinstance(sampler, NormalSampler)
    assert sampler.mean_latency_ms == 20.0
    assert sampler.std_latency_ms == 8.0


def test_per_episode_modifier_freezes_draw_per_episode():
    sampler = build_sampler(
        {"type": "mixed_normal_distribution", "mean_latency_ms": 100.0, "std_latency_ms": 20.0,
         "min_latency_ms": 0.0, "max_latency_ms": 200.0, "seed": 0},
        clock=_clock(),
        seed=0,
    )
    ep0 = [sampler.sample({"episode_id": 0}).latency_ms for _ in range(5)]
    ep1 = [sampler.sample({"episode_id": 1}).latency_ms for _ in range(5)]
    assert len(set(ep0)) == 1
    assert len(set(ep1)) == 1
    assert ep0[0] != ep1[0]
    assert sampler.latency_type == "mixed_normal_distribution"


def test_latency_trace_load_and_save(tmp_path: Path):
    trace_path = tmp_path / "saved.jsonl"
    save_latency_trace([LatencyRecord(obs_id=0, latency_ms=12.0)], trace_path)
    assert load_latency_trace(trace_path) == [12.0]


def test_profile_default_routes_to_hidden_regime(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    sampler = _profile_sampler(profile_dir, seed=0)
    assert isinstance(sampler, HiddenRegimeSampler)
    draws = [sampler.sample({"episode_id": 0}) for _ in range(5)]
    assert all(s.worker_service_raw_frames in {1, 2, 3} for s in draws)
    assert all(s.action_ready_raw_frames in {1, 2, 3} for s in draws)
    assert sampler.profile_metadata["profile_id"] == "L_test"
    assert sampler.profile_metadata["target_device"] == "test_gpu"


def test_legacy_bayes_markov_sampler_config_routes_to_temporal(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    sampler = _profile_sampler(profile_dir, seed=0, sampler="bayes_markov")
    assert isinstance(sampler, HiddenRegimeSampler)
    draws = [sampler.sample({"episode_id": 0}) for _ in range(5)]
    assert all(s.worker_service_raw_frames in {1, 2, 3} for s in draws)
    assert all(s.action_ready_raw_frames in {1, 2, 3} for s in draws)


def test_temporal_rejects_stale_burst_model_type(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    model = json.loads((profile_dir / "latency_burst_model.json").read_text(encoding="utf-8"))
    model["model_type"] = "bayes_markov"
    (profile_dir / "latency_burst_model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="rebuild the profile"):
        _profile_sampler(profile_dir, seed=0)


def test_profile_sampler_is_deterministic_with_seed(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    first = _profile_sampler(profile_dir, seed=0)
    second = _profile_sampler(profile_dir, seed=0)
    assert [first.sample({"episode_id": 0}) for _ in range(20)] == [
        second.sample({"episode_id": 0}) for _ in range(20)
    ]


def test_hidden_regime_sampler_excursion_emits_duration_consecutive_draws():
    # Sparse burst entries (~1/26) with a fixed (D=4, L=100) excursion model:
    # every above-threshold run must be exactly 4 consecutive draws at the
    # level, with action_ready = level + ready_delta, even though the outer
    # chain leaves burst (exit prob ~1) long before the excursion finishes.
    model = _hand_model(
        excursions=_constant_excursions(duration=4.0, level_ms=100.0, ready_delta_ms=7.0),
        burst_emission=None,
        enter_burst_count=40,
    )
    sampler = HiddenRegimeSampler(model, clock=_clock(), seed=3)
    draws = [sampler.sample({"episode_id": 0}) for _ in range(200)]

    runs: list[int] = []
    run = 0
    for draw in draws:
        if draw.worker_service_ms > 50.0:
            assert draw.worker_service_ms == pytest.approx(100.0)
            assert draw.latency_ms == pytest.approx(107.0)
            run += 1
        else:
            assert draw.worker_service_ms == pytest.approx(40.0)
            if run:
                runs.append(run)
                run = 0
    # A trailing run cut by the 200-draw window is dropped, not asserted on.
    assert runs
    assert set(runs) == {4}


def test_hidden_regime_sampler_back_to_back_excursions_never_fuse():
    # Chain in burst nearly every draw and start prob ~1: without the
    # maximality separator two excursions would fuse into one longer observed
    # run. Every above-threshold run must still be exactly the fitted D=4.
    model = _hand_model(
        excursions=_constant_excursions(duration=4.0, level_ms=100.0),
        burst_emission=_constant_joint_emission(45.0),
        enter_burst_count=1000,
    )
    sampler = HiddenRegimeSampler(model, clock=_clock(), seed=5)
    values = [sampler.sample({"episode_id": 0}).worker_service_ms for _ in range(400)]

    runs: list[int] = []
    run = 0
    for value in values:
        if value > 50.0:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    assert runs
    assert set(runs) == {4}


def test_hidden_regime_sampler_episode_reset_cancels_in_flight_excursion():
    model = _hand_model(
        excursions=_constant_excursions(duration=6.0, level_ms=100.0),
        burst_emission=None,
    )
    sampler = HiddenRegimeSampler(model, clock=_clock(), seed=0)
    draws = [sampler.sample({"episode_id": 0}).worker_service_ms for _ in range(3)]
    # Draw 1 is the forced-calm reset draw; the chain then enters burst
    # (enter prob ~1) and starts the 6-draw excursion.
    assert draws == pytest.approx([40.0, 100.0, 100.0])
    assert sampler.sample({"episode_id": 1}).worker_service_ms == pytest.approx(40.0)


def test_hidden_regime_sampler_without_excursions_never_exceeds_threshold():
    # A device whose burst records were all sub-threshold interleaved normals:
    # burst emission exists but there are no excursions to start.
    model = _hand_model(
        excursions=None,
        burst_emission=_constant_joint_emission(48.0),
    )
    sampler = HiddenRegimeSampler(model, clock=_clock(), seed=7)
    values = {sampler.sample({"episode_id": 0}).worker_service_ms for _ in range(200)}
    assert values == {40.0, 48.0}


def test_hidden_regime_spikeless_profile_is_constant(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path, worker_service_sequence=[(2, 2), (2, 2)])
    sampler = _profile_sampler(profile_dir, seed=1)
    draws = [sampler.sample({"episode_id": 0}) for _ in range(4)]
    assert [s.worker_service_raw_frames for s in draws] == [2, 2, 2, 2]
    assert [s.action_ready_raw_frames for s in draws] == [2, 2, 2, 2]


def test_profile_can_force_iid_inverse_cdf_sampler(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    sampler = _profile_sampler(profile_dir, seed=0, sampler="iid_inverse_cdf")
    samples = [sampler.sample({"episode_id": 0}) for _ in range(3)]
    assert [s.latency_ms for s in samples] == pytest.approx([
        26.888437030500963,
        25.15908805880605,
        18.4114316166169,
    ])


def test_profile_default_without_burst_model_raises(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path, write_burst_model=False)
    with pytest.raises(ValueError, match="profile_default"):
        _profile_sampler(profile_dir, seed=0)


def test_profile_accepts_profile_directory(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    sampler = build_sampler(
        {"type": "profile_sample", "profile_path": str(profile_dir)},
        clock=_clock(),
        seed=0,
    )
    assert sampler.sample({"episode_id": 0}).worker_service_raw_frames in {1, 2, 3}


def test_profile_rejects_missing_profile_path():
    with pytest.raises(FileNotFoundError):
        build_sampler({"type": "profile_sample", "profile_path": "/nonexistent/profile.json"}, clock=_clock(), seed=0)


def test_profile_rejects_missing_burst_model_file(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    (profile_dir / "latency_burst_model.json").unlink()
    with pytest.raises(FileNotFoundError):
        _profile_sampler(profile_dir, sampler="bayes_markov")
