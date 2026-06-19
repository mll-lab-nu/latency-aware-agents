from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import LatencyRecord
from latency_bench.latency.burst_profile import build_latency_burst_model
from latency_bench.latency.profile import (
    build_inverse_cdf_distribution,
    save_latency_distribution,
    save_profile_manifest,
    write_latency_samples,
)
from latency_bench.latency.samplers import (
    FixedSampler,
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


def _write_v2_latency_profile(tmp_path: Path) -> Path:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    save_latency_distribution(
        build_inverse_cdf_distribution([10.0, 90.0]),
        profile_dir / "latency_distribution.json",
    )
    (profile_dir / "latency_burst_model.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model_type": "queue_conditioned",
                "state_key": "recent_drop_count_capped",
                "recent_drop_count_cap": 1,
                "state_counts": {"0": 1, "1": 1},
                "samples_by_state": {
                    "0": [
                        {
                            "action_ready_raw_frames": 1,
                            "worker_service_raw_frames": 2,
                            "worker_service_ms": 20.0,
                            "inference_latency_ms": 10.0,
                            "applied": True,
                            "dropped": False,
                        }
                    ],
                    "1": [
                        {
                            "action_ready_raw_frames": 4,
                            "worker_service_raw_frames": 5,
                            "worker_service_ms": 110.0,
                            "inference_latency_ms": 90.0,
                            "applied": True,
                            "dropped": False,
                        }
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    save_profile_manifest(
        {
            "profile_id": "L_test",
            "policy_id": "pi_test",
            "env_id": "env_test",
            "target_device": "test_gpu",
            "env_fps": 25.0,
            "frame_ms": 40.0,
            "n_samples": 2,
            "trace_path": "latencies.jsonl",
            "distribution_path": "latency_distribution.json",
            "summary": {},
            "latency_method": "temporal",
            "sample_model_type": "queue_conditioned",
            "burst_model_path": "latency_burst_model.json",
            "burst_state_key": "recent_drop_count_capped",
        },
        profile_dir / "profile.json",
    )
    return profile_dir


def _write_latency_profile(
    tmp_path: Path,
    samples: list[float] | None = None,
    *,
    trace_path: str = "latencies.jsonl",
    write_trace: bool = True,
    sample_model_type: str | None = "semi_markov_latency_frames",
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
    burst_samples = [
        {
            "episode_id": 0,
            "sample_index": index,
            "latency_ms": float(sample),
            "latency_raw_frames": index + 1,
        }
        for index, sample in enumerate(samples)
    ]
    (profile_dir / "latency_burst_model.json").write_text(
        json.dumps(build_latency_burst_model(burst_samples), indent=2, sort_keys=True) + "\n",
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
    if sample_model_type is not None:
        manifest["latency_method"] = "temporal"
        manifest["sample_model_type"] = sample_model_type
        manifest["burst_model_path"] = "latency_burst_model.json"
        manifest["burst_state_key"] = "latency_raw_frames"
    save_profile_manifest(manifest, profile_dir / "profile.json")
    return profile_dir


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


def test_profile_sampler_is_deterministic_with_seed(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    first = _profile_sampler(profile_dir, seed=0)
    second = _profile_sampler(profile_dir, seed=0)
    assert [first.sample({"episode_id": 0}) for _ in range(20)] == [
        second.sample({"episode_id": 0}) for _ in range(20)
    ]
    assert first.profile_metadata["profile_id"] == "L_test"
    assert first.profile_metadata["target_device"] == "test_gpu"


def test_profile_default_resolves_to_temporal_run_length(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    sampler = _profile_sampler(profile_dir, seed=1)
    samples = [sampler.sample({"episode_id": 0}) for _ in range(3)]
    assert [s.latency_raw_frames for s in samples] == [1, 2, 3]
    assert [s.latency_ms for s in samples] == [10.0, 20.0, 30.0]


def test_profile_default_conditions_v2_burst_model_on_recent_drop_count(tmp_path: Path):
    profile_dir = _write_v2_latency_profile(tmp_path)
    sampler = _profile_sampler(profile_dir, seed=0)
    clean = sampler.sample({"episode_id": 0, "recent_drop_count": 0})
    after_drop = sampler.sample({"episode_id": 0, "recent_drop_count": 1})
    assert (clean.action_ready_raw_frames, clean.worker_service_raw_frames, clean.worker_service_ms) == (1, 2, 20.0)
    assert (after_drop.action_ready_raw_frames, after_drop.worker_service_raw_frames, after_drop.worker_service_ms) == (4, 5, 110.0)


def test_profile_can_force_iid_inverse_cdf_sampler(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    sampler = _profile_sampler(profile_dir, seed=0, sampler="iid_inverse_cdf")
    samples = [sampler.sample({"episode_id": 0}) for _ in range(3)]
    assert [s.latency_ms for s in samples] == pytest.approx([
        26.888437030500963,
        25.15908805880605,
        18.4114316166169,
    ])


def test_legacy_profile_without_burst_model_defaults_to_iid(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path, sample_model_type=None)
    sampler = _profile_sampler(profile_dir, seed=0)
    assert sampler.sample({"episode_id": 0}).latency_ms == pytest.approx(26.888437030500963)


def test_profile_accepts_profile_directory(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    sampler = build_sampler(
        {"type": "profile_sample", "profile_path": str(profile_dir)},
        clock=_clock(),
        seed=0,
    )
    assert sampler.sample({"episode_id": 0}).latency_raw_frames in {1, 2, 3}


def test_profile_rejects_missing_profile_path():
    with pytest.raises(FileNotFoundError):
        build_sampler({"type": "profile_sample", "profile_path": "/nonexistent/profile.json"}, clock=_clock(), seed=0)


def test_profile_rejects_missing_burst_model_file(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    (profile_dir / "latency_burst_model.json").unlink()
    with pytest.raises(FileNotFoundError):
        _profile_sampler(profile_dir, sampler="semi_markov_latency_frames")
