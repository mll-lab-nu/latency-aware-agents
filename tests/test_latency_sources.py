from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_bench.core.types import LatencyRecord
from latency_bench.latency.sources import (
    FixedLatencySource,
    MixedNormalDistributionLatencySource,
    MixedUniformDistributionLatencySource,
    NormalDistributionLatencySource,
    ProfileSampleLatencySource,
    UniformDistributionLatencySource,
    ZeroLatencySource,
)
from latency_bench.latency.burst_profile import build_latency_burst_model
from latency_bench.latency.profile import build_inverse_cdf_distribution, save_latency_distribution, save_profile_manifest, write_latency_samples
from latency_bench.latency.trace import load_latency_trace, save_latency_trace


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
            "summary": {},
        }
    if sample_model_type is not None:
        manifest["sample_model_type"] = sample_model_type
        manifest["burst_model_path"] = "latency_burst_model.json"
        manifest["burst_state_key"] = "latency_raw_frames"
    save_profile_manifest(manifest, profile_dir / "profile.json")
    return profile_dir


def test_zero_latency_source_returns_zero():
    assert ZeroLatencySource().next_latency_ms() == 0.0


def test_fixed_latency_source_returns_constant():
    source = FixedLatencySource(42.5)

    assert source.next_latency_ms() == 42.5
    assert source.next_latency_ms() == 42.5


def test_fixed_latency_source_preserves_contract_value_without_range_validation():
    source = FixedLatencySource(-4.5)

    assert source.fixed_latency_ms == -4.5
    assert source.next_latency_ms() == -4.5


def test_normal_distribution_samples_deterministically_with_seed():
    first = NormalDistributionLatencySource(mean_latency_ms=100.0, std_latency_ms=20.0, seed=0)
    second = NormalDistributionLatencySource(mean_latency_ms=100.0, std_latency_ms=20.0, seed=0)

    assert [first.next_latency_ms() for _ in range(5)] == [second.next_latency_ms() for _ in range(5)]


def test_normal_distribution_clamps_to_zero_by_default():
    source = NormalDistributionLatencySource(mean_latency_ms=-10.0, std_latency_ms=0.0)

    assert source.next_latency_ms() == 0.0


def test_normal_distribution_respects_min_and_max_bounds():
    low = NormalDistributionLatencySource(mean_latency_ms=5.0, std_latency_ms=0.0, min_latency_ms=10.0)
    high = NormalDistributionLatencySource(mean_latency_ms=100.0, std_latency_ms=0.0, max_latency_ms=50.0)

    assert low.next_latency_ms() == 10.0
    assert high.next_latency_ms() == 50.0


def test_normal_distribution_preserves_contract_values_without_range_validation():
    source = NormalDistributionLatencySource(
        mean_latency_ms=100.0,
        std_latency_ms=-1.0,
        min_latency_ms=-5.0,
        max_latency_ms=1.0,
    )

    assert source.mean_latency_ms == 100.0
    assert source.std_latency_ms == -1.0
    assert source.min_latency_ms == -5.0
    assert source.max_latency_ms == 1.0


def test_uniform_distribution_samples_deterministically_with_seed_and_bounds():
    first = UniformDistributionLatencySource(min_latency_ms=10.0, max_latency_ms=20.0, seed=0)
    second = UniformDistributionLatencySource(min_latency_ms=10.0, max_latency_ms=20.0, seed=0)

    first_values = [first.next_latency_ms() for _ in range(10)]
    second_values = [second.next_latency_ms() for _ in range(10)]

    assert first_values == second_values
    assert all(10.0 <= value <= 20.0 for value in first_values)


def test_uniform_distribution_preserves_contract_values_without_range_validation():
    source = UniformDistributionLatencySource(min_latency_ms=20.0, max_latency_ms=10.0)

    assert source.min_latency_ms == 20.0
    assert source.max_latency_ms == 10.0


def test_mixed_distribution_sources_sample_once_per_episode():
    normal = MixedNormalDistributionLatencySource(
        mean_latency_ms=100.0,
        std_latency_ms=20.0,
        seed=0,
        min_latency_ms=0.0,
        max_latency_ms=200.0,
    )
    uniform = MixedUniformDistributionLatencySource(
        min_latency_ms=10.0,
        max_latency_ms=20.0,
        seed=0,
    )

    normal_ep0 = [normal.next_latency_ms({"episode_id": 0}) for _ in range(5)]
    normal_ep1 = [normal.next_latency_ms({"episode_id": 1}) for _ in range(5)]
    uniform_ep0 = [uniform.next_latency_ms({"episode_id": 0}) for _ in range(5)]
    uniform_ep1 = [uniform.next_latency_ms({"episode_id": 1}) for _ in range(5)]

    assert len(set(normal_ep0)) == 1
    assert len(set(normal_ep1)) == 1
    assert normal_ep0[0] != normal_ep1[0]
    assert len(set(uniform_ep0)) == 1
    assert len(set(uniform_ep1)) == 1
    assert uniform_ep0[0] != uniform_ep1[0]


def test_latency_trace_load_and_save(tmp_path: Path):
    trace_path = tmp_path / "saved.jsonl"
    save_latency_trace(
        [LatencyRecord(obs_id=0, latency_ms=12.0)],
        trace_path,
    )

    assert load_latency_trace(trace_path) == [12.0]


def test_profile_sample_samples_deterministically_with_seed(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    first = ProfileSampleLatencySource(profile_dir / "profile.json", seed=0)
    second = ProfileSampleLatencySource(profile_dir / "profile.json", seed=0)

    assert [first.next_latency_sample({"episode_id": 0}) for _ in range(20)] == [
        second.next_latency_sample({"episode_id": 0}) for _ in range(20)
    ]
    assert first.profile_id == "L_test"
    assert first.target_device == "test_gpu"
    assert first.last_metadata["profile_id"] == "L_test"


def test_profile_sample_returns_raw_frames_from_burst_model(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    source = ProfileSampleLatencySource(profile_dir / "profile.json", seed=1)

    samples = [source.next_latency_sample({"episode_id": 0}) for _ in range(3)]

    assert [sample["latency_raw_frames"] for sample in samples] == [1, 2, 3]
    assert [sample["latency_ms"] for sample in samples] == [10.0, 20.0, 30.0]


def test_profile_sample_can_force_iid_inverse_cdf_sampler(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    source = ProfileSampleLatencySource(profile_dir / "profile.json", seed=0, sampler="iid_inverse_cdf")

    samples = [source.next_latency_sample({"episode_id": 0}) for _ in range(3)]

    assert [sample["latency_ms"] for sample in samples] == pytest.approx([
        26.888437030500963,
        25.15908805880605,
        18.4114316166169,
    ])
    assert all("latency_raw_frames" not in sample for sample in samples)


def test_profile_sample_legacy_profile_defaults_to_iid_inverse_cdf(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path, sample_model_type=None)
    source = ProfileSampleLatencySource(profile_dir / "profile.json", seed=0)

    sample = source.next_latency_sample({"episode_id": 0})

    assert sample["latency_ms"] == pytest.approx(26.888437030500963)
    assert "latency_raw_frames" not in sample


def test_profile_sample_accepts_profile_directory(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    source = ProfileSampleLatencySource(profile_dir, seed=0)

    assert source.profile_path == profile_dir / "profile.json"
    assert source.next_latency_raw_frames({"episode_id": 0}) in {1, 2, 3}


def test_profile_sample_rejects_missing_profile_path():
    with pytest.raises(ValueError, match="profile_path is required"):
        ProfileSampleLatencySource(None)


def test_profile_sample_rejects_missing_burst_model_file(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    (profile_dir / "latency_burst_model.json").unlink()

    with pytest.raises(FileNotFoundError, match="latency_burst_model.json"):
        ProfileSampleLatencySource(profile_dir / "profile.json", sampler="semi_markov_latency_frames")


def test_profile_sample_passes_seed_through_to_rng(tmp_path: Path):
    profile_dir = _write_latency_profile(tmp_path)
    source = ProfileSampleLatencySource(profile_dir / "profile.json", seed=True)

    assert source.next_latency_raw_frames({"episode_id": 0}) in {1, 2, 3}
