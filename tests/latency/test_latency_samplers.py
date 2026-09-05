from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_bench.core.clock import EnvClock
from latency_bench.latency.burst_profile import (
    build_hidden_regime_burst_model,
    build_worker_slot_distributions_from_sessions,
)
from latency_bench.latency.profile import (
    save_latency_distribution,
    save_profile_manifest,
)
from latency_bench.latency.samplers import (
    FixedSampler,
    HiddenRegimeSampler,
    NormalSampler,
    UniformSampler,
    ZeroSampler,
    build_episode_sampler_factory,
    build_sampler,
)


def _clock() -> EnvClock:
    return EnvClock(env_fps=25.0, obs_fps=25.0)


def _profile_sampler(profile_dir: Path, *, seed=0, method: str = "temporal"):
    return build_sampler(
        {"method": method, "profile_path": str(profile_dir / "profile.json"), "seed": seed},
        clock=_clock(),
    )


def _write_latency_profile(
    tmp_path: Path,
    samples: list[float] | None = None,
    *,
    write_burst_model: bool = True,
    worker_service_sequence: list[tuple[int, int]] | None = None,
) -> Path:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    samples = [10.0, 20.0, 30.0] if samples is None else samples
    worker_service_sequence = worker_service_sequence or [(1, 1), (2, 2), (3, 3)]
    model_samples = [
        {
            "episode_id": 0,
            "sample_index": index,
            "worker_id": 0,
            "observation_to_action_latency_ms": float(action_ready) * 40.0,
            "worker_service_time_ms": float(worker_service) * 40.0,
            "worker_service_time_ms_bin": int(float(worker_service) * 40.0),
        }
        for index, (worker_service, action_ready) in enumerate(worker_service_sequence)
    ]
    burst_model = build_hidden_regime_burst_model(model_samples) if write_burst_model else None
    distribution = build_worker_slot_distributions_from_sessions(
        [model_samples],
        spike_threshold_ms_by_worker_slot=(
            burst_model["spike_threshold_ms_by_worker_slot"]
            if burst_model is not None
            else {"0": float("inf")}
        ),
    )
    save_latency_distribution(distribution, profile_dir / "latency_distribution.json")
    if write_burst_model:
        (profile_dir / "latency_burst_model.json").write_text(
            json.dumps(burst_model, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "model_id": "model_test",
        "gpu_class": "1x-test_gpu",
        "workload_id": "workload_test",
        "instance_id": "instance_test",
        "source_run_id": "run_test",
        "env_fps": 25.0,
        "frame_ms": 40.0,
        "n_samples": len(samples),
        "distribution_path": "latency_distribution.json",
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


def _constant_joint_emission(
    worker_service_time_ms: float,
    observation_to_action_latency_ms: float | None = None,
) -> dict:
    return {
        "count": 10,
        "observation_to_action_latency_ms": _constant_inverse_cdf(
            worker_service_time_ms
            if observation_to_action_latency_ms is None
            else observation_to_action_latency_ms
        ),
        "worker_service_time_ms": _constant_inverse_cdf(worker_service_time_ms),
        "spearman_rho": 0.0,
    }


def _rank_process(
    *,
    severity: float,
    level_ms: float | None,
) -> dict:
    return {
        "draw_count": 1000,
        "spike_count": int(1000 * severity),
        "severity": _constant_inverse_cdf(severity),
        "dwell_length_spearman_rho": 0.0,
        "level_ms": None if level_ms is None else _constant_inverse_cdf(level_ms),
        "level_residual_ms": None if level_ms is None else _constant_inverse_cdf(0.0),
    }


def _hand_model(
    *,
    rank_process: dict,
    enter_burst_count: int = 1000,
    dwell: float = 1.0,
    pre_worker_time_ms: float = 0.0,
) -> dict:
    return {
        "schema_version": 12,
        "model_type": "hidden_regime",
        "reset_scope": "session",
        "spike_threshold_ms_by_worker_slot": {"0": 50.0},
        "worker_count": 1,
        "regime_transition_counts": {
            "calm": {"calm": 1000 - enter_burst_count, "burst": enter_burst_count},
            "burst": {"calm": 1000, "burst": 0},
        },
        "regime_step_counts": {"calm": 1000, "burst": 1000},
        "burst_dwell_lengths": [],
        "burst_dwell_distribution": _constant_inverse_cdf(dwell),
        "burst_rank_processes": [rank_process],
        "burst_slot_rank_templates": [
            {"dwell_length": int(dwell), "slot_to_rank": {"0": 0}}
        ],
        "pre_worker_time_ms": _constant_inverse_cdf(pre_worker_time_ms),
    }


def _hand_distributions(*, steady_ms: float = 40.0) -> dict:
    return {
        "schema_version": 3,
        "worker_slots": {
            "0": {
                "all": _constant_joint_emission(steady_ms),
                "steady": _constant_joint_emission(steady_ms),
            }
        },
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


def test_per_episode_modifier_freezes_draw_per_episode():
    sampler = build_sampler(
        {"method": "normal", "per_episode": True, "mean_latency_ms": 100.0, "std_latency_ms": 20.0,
         "min_latency_ms": 0.0, "max_latency_ms": 200.0, "seed": 0},
        clock=_clock(),
    )
    ep0 = [sampler.sample({"episode_id": 0, "worker_slot": 0}).latency_ms for _ in range(5)]
    ep1 = [sampler.sample({"episode_id": 1, "worker_slot": 0}).latency_ms for _ in range(5)]
    assert len(set(ep0)) == 1
    assert len(set(ep1)) == 1
    assert ep0[0] != ep1[0]
    assert sampler.latency_type == "mixed_normal_distribution"


def test_temporal_method_builds_hidden_regime_sampler(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    sampler = _profile_sampler(profile_dir, seed=0)
    assert isinstance(sampler, HiddenRegimeSampler)
    draws = [sampler.sample({"episode_id": 0, "worker_slot": 0}) for _ in range(5)]
    assert all(s.worker_service_raw_frames in {1, 2, 3} for s in draws)
    assert all(s.action_ready_raw_frames in {1, 2, 3} for s in draws)
    assert sampler.profile_metadata["model_id"] == "model_test"
    assert sampler.profile_metadata["gpu_class"] == "1x-test_gpu"


def test_temporal_rejects_stale_burst_model_type(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    model = json.loads((profile_dir / "latency_burst_model.json").read_text(encoding="utf-8"))
    model["model_type"] = "bayes_markov"
    (profile_dir / "latency_burst_model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="rebuild the profile"):
        _profile_sampler(profile_dir, seed=0)


def test_temporal_rejects_old_burst_schema(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    profile_path = profile_dir / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["workload_id"] = "isolated"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    model_path = profile_dir / "latency_burst_model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["schema_version"] = 11
    model_path.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported temporal burst model"):
        _profile_sampler(profile_dir, seed=0)


def test_profile_sampler_is_deterministic_with_seed(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    first = _profile_sampler(profile_dir, seed=0)
    second = _profile_sampler(profile_dir, seed=0)
    assert [first.sample({"episode_id": 0, "worker_slot": 0}) for _ in range(20)] == [
        second.sample({"episode_id": 0, "worker_slot": 0}) for _ in range(20)
    ]


def test_hidden_regime_sampler_severity_one_covers_the_whole_dwell():
    # Sparse burst entries (~1/26) with severity 1 at level 100 and a 4-draw
    # dwell: every above-threshold run must be exactly the dwell's 4 draws at
    # the level, with action_ready = level + ready_delta.
    model = _hand_model(
        rank_process=_rank_process(severity=1.0, level_ms=100.0),
        enter_burst_count=40,
        dwell=4.0,
        pre_worker_time_ms=7.0,
    )
    sampler = HiddenRegimeSampler(model, _hand_distributions(), clock=_clock(), seed=3)
    draws = [sampler.sample({"episode_id": 0, "worker_slot": 0}) for _ in range(200)]

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


def test_hidden_regime_sampler_dwell_end_ends_the_slowness():
    # Severity is a dwell property and dies with it: a 1-draw dwell emits one
    # spike and the very next draw is calm again.
    model = _hand_model(
        rank_process=_rank_process(severity=1.0, level_ms=100.0),
        enter_burst_count=40,
        dwell=1.0,
    )
    sampler = HiddenRegimeSampler(model, _hand_distributions(), clock=_clock(), seed=3)

    runs: list[int] = []
    run = 0
    for _ in range(400):
        draw = sampler.sample({"episode_id": 0, "worker_slot": 0})
        if draw.worker_service_ms > 50.0:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    assert runs
    assert set(runs) == {1}


def test_hidden_regime_sampler_partial_severity_mixes_spikes_and_steady():
    # Severity 0.5 inside a permanent dwell: serves split between the level
    # and the slot's steady emission, at roughly the fitted fraction.
    model = _hand_model(
        rank_process=_rank_process(
            severity=0.5, level_ms=100.0
        ),
        enter_burst_count=1000,
        dwell=1000.0,
    )
    sampler = HiddenRegimeSampler(model, _hand_distributions(), clock=_clock(), seed=5)
    values = [sampler.sample({"episode_id": 0, "worker_slot": 0}).worker_service_ms for _ in range(2000)]

    spike_share = sum(1 for value in values if value == pytest.approx(100.0)) / len(values)
    assert {round(value, 6) for value in values} == {40.0, 100.0}
    assert 0.45 < spike_share < 0.55


def test_hidden_regime_sampler_episode_change_preserves_dwell_slowness():
    # The regime is machine state: an env reset does not interrupt a slow
    # condition, so the 6-draw dwell entered in episode 0 keeps emitting its
    # severity-one level through the episode boundary and only then falls
    # back to calm.
    model = _hand_model(
        rank_process=_rank_process(severity=1.0, level_ms=100.0),
        dwell=6.0,
    )
    sampler = HiddenRegimeSampler(model, _hand_distributions(), clock=_clock(), seed=0)
    draws = [sampler.sample({"episode_id": 0, "worker_slot": 0}).worker_service_ms for _ in range(3)]
    # Draw 1 is the sampler's very first draw (chain starts calm); the chain
    # then enters the 6-draw dwell.
    assert draws == pytest.approx([40.0, 100.0, 100.0])
    # The remaining four dwell draws keep emitting in episode 1.
    continued = [sampler.sample({"episode_id": 1, "worker_slot": 0}).worker_service_ms for _ in range(4)]
    assert continued == pytest.approx([100.0, 100.0, 100.0, 100.0])


def test_hidden_regime_sampler_episode_change_preserves_burst_regime():
    # A long fitted dwell: once the chain is in burst it must stay there
    # across an episode boundary instead of resetting to calm.
    model = _hand_model(
        rank_process=_rank_process(severity=0.0, level_ms=None),
        dwell=1000.0,
    )
    sampler = HiddenRegimeSampler(model, _hand_distributions(), clock=_clock(), seed=0)
    values = [sampler.sample({"episode_id": 0, "worker_slot": 0}).worker_service_ms for _ in range(2)]
    assert values == pytest.approx([40.0, 40.0])
    assert sampler.sample({"episode_id": 1, "worker_slot": 0}).worker_service_ms == pytest.approx(40.0)


def test_hidden_regime_sampler_draws_entry_probability_once_per_episode(monkeypatch):
    # Each sampler belongs to one replica, so episode transitions are observed
    # in that replica's episode order.
    model = _hand_model(
        rank_process=_rank_process(severity=1.0, level_ms=100.0),
        enter_burst_count=1000,
        dwell=1.0,
    )
    sampler = HiddenRegimeSampler(model, _hand_distributions(), clock=_clock(), seed=3)
    draws: list[tuple[str, str]] = []

    def counting(source: str, target: str) -> float:
        draws.append((source, target))
        return 0.0

    monkeypatch.setattr(sampler, "_episode_transition_prob", counting)
    for episode_id in range(5):
        for _ in range(50):
            sampler.sample({"episode_id": episode_id, "worker_slot": 0})
    assert draws == [("calm", "burst")] * 5


def test_hidden_regime_sampler_zero_severity_never_exceeds_threshold():
    # A device whose burst records were all sub-threshold interleaved normals:
    # severity zero with an emission, so no spike is ever emitted.
    model = _hand_model(
        rank_process=_rank_process(severity=0.0, level_ms=None),
    )
    sampler = HiddenRegimeSampler(model, _hand_distributions(), clock=_clock(), seed=7)
    values = {sampler.sample({"episode_id": 0, "worker_slot": 0}).worker_service_ms for _ in range(200)}
    assert values == {40.0}


def test_hidden_regime_spikeless_profile_is_constant(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path, worker_service_sequence=[(2, 2), (2, 2)])
    sampler = _profile_sampler(profile_dir, seed=1)
    draws = [sampler.sample({"episode_id": 0, "worker_slot": 0}) for _ in range(4)]
    assert [s.worker_service_raw_frames for s in draws] == [2, 2, 2, 2]
    assert [s.action_ready_raw_frames for s in draws] == [2, 2, 2, 2]


def test_iid_method_uses_inverse_cdf_distribution(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    sampler = _profile_sampler(profile_dir, seed=0, method="iid")
    samples = [sampler.sample({"episode_id": 0, "worker_slot": 0}) for _ in range(3)]
    assert all(40.0 <= sample.worker_service_ms <= 120.0 for sample in samples)
    assert all(abs(sample.latency_ms - sample.worker_service_ms) < 0.1 for sample in samples)


def test_iid_episode_factory_derives_reproducible_independent_streams(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    factory = build_episode_sampler_factory(
        {
            "method": "iid",
            "profile_path": profile_dir,
            "seed": 7,
        },
        _clock(),
    )

    first = factory(3)
    repeat = factory(3)
    other = factory(4)
    first_values = [first.sample({"episode_id": 3, "worker_slot": 0}).latency_ms for _ in range(10)]
    repeat_values = [repeat.sample({"episode_id": 3, "worker_slot": 0}).latency_ms for _ in range(10)]
    other_values = [other.sample({"episode_id": 4, "worker_slot": 0}).latency_ms for _ in range(10)]

    assert first_values == repeat_values
    assert first_values != other_values
    assert first.latency_type == "profile_sample"
    assert first.profile_metadata == {
        "model_id": "model_test",
        "gpu_class": "1x-test_gpu",
        "workload_id": "workload_test",
        "instance_id": "instance_test",
        "source_run_id": "run_test",
    }


def test_temporal_method_requires_burst_model_path(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path, write_burst_model=False)
    with pytest.raises(KeyError, match="burst_model_path"):
        _profile_sampler(profile_dir, seed=0)


def test_profile_accepts_profile_directory(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    sampler = build_sampler(
        {"method": "temporal", "profile_path": str(profile_dir)},
        clock=_clock(),
    )
    assert sampler.sample({"episode_id": 0, "worker_slot": 0}).worker_service_raw_frames in {1, 2, 3}


def test_profile_rejects_missing_profile_path():
    with pytest.raises(FileNotFoundError):
        build_sampler({"method": "temporal", "profile_path": "/nonexistent/profile.json"}, clock=_clock())


def test_profile_rejects_missing_burst_model_file(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    (profile_dir / "latency_burst_model.json").unlink()
    with pytest.raises(FileNotFoundError):
        _profile_sampler(profile_dir)
