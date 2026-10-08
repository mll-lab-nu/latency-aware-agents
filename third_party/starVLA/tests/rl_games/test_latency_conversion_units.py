import json

import pytest

from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_flappy_dataset import (
    build_latency_prompt_map,
    latency_id_from_row,
    source_timing_from_args,
)


def test_raw_frame_latency_is_preserved() -> None:
    assert latency_id_from_row(
        {"latency_raw_frames": 6},
        latency_column="latency_raw_frames",
        target_latency_unit="raw_frames",
        obs_stride_raw_frames=4,
    ) == 6


def test_prompt_map_includes_raw_frame_latency_field() -> None:
    prompt_map = build_latency_prompt_map(
        [{"prompt": "latency six", "latency_raw_frames": 6, "latency_ms": 100.0}],
        latency_column="latency_raw_frames",
        target_latency_unit="raw_frames",
        obs_stride_raw_frames=4,
    )

    assert prompt_map["6"]["latency"] == 6
    assert prompt_map["6"]["latency_raw_frames"] == 6


def test_prompt_map_derives_raw_frames_from_observation_step_latency() -> None:
    prompt_map = build_latency_prompt_map(
        [{"prompt": "latency two", "latency_raw_frames": 8, "latency_ms": 133.3}],
        latency_column="latency_raw_frames",
        target_latency_unit="observation_steps",
        obs_stride_raw_frames=4,
    )

    assert prompt_map["2"]["latency"] == 2
    assert prompt_map["2"]["latency_raw_frames"] == 8


def test_raw_frame_latency_converts_to_observation_steps() -> None:
    assert latency_id_from_row(
        {"latency_raw_frames": 8},
        latency_column="latency_raw_frames",
        target_latency_unit="observation_steps",
        obs_stride_raw_frames=4,
    ) == 2


def test_observation_step_conversion_requires_divisibility() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        latency_id_from_row(
            {"latency_raw_frames": 6},
            latency_column="latency_raw_frames",
            target_latency_unit="observation_steps",
            obs_stride_raw_frames=4,
        )


def test_latency_column_is_exact_and_none_disables_latency() -> None:
    row = {"latency": 2, "latency_raw_frames": 8}
    assert latency_id_from_row(
        row,
        latency_column="latency",
        target_latency_unit="raw_frames",
        obs_stride_raw_frames=4,
    ) == 2
    assert latency_id_from_row(
        row,
        latency_column=None,
        target_latency_unit="raw_frames",
        obs_stride_raw_frames=4,
        default_latency=0,
    ) == 0


def test_explicit_latency_column_must_exist() -> None:
    with pytest.raises(KeyError, match="latency_raw_frames"):
        latency_id_from_row(
            {"latency": 2},
            latency_column="latency_raw_frames",
            target_latency_unit="raw_frames",
            obs_stride_raw_frames=4,
        )


def test_source_metadata_preserves_float_fps(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "rows_unit": "decision_step",
                "obs_fps": 8.75,
                "obs_stride_raw_frames": 4,
            }
        ),
        encoding="utf-8",
    )
    assert source_timing_from_args(
        source_metadata=str(metadata_path),
        source_fps=None,
        obs_stride_raw_frames=None,
    ) == (8.75, 4, "decision_step")
