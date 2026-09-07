import json
from pathlib import Path

import pytest
from PIL import Image

from latency_bench.analysis.utils import write_latency_distribution_plot
from latency_bench.latency.profile_builder import build_profile_from_traces


def _write_session(
    root: Path,
    session_id: int,
    service_ms: list[float],
    *,
    dropped_indices: tuple[int, ...] = (),
) -> dict:
    session = root / str(session_id)
    session.mkdir(parents=True)
    attempts = session / "observation_attempts.jsonl"
    attempts.write_text(
        "".join(
            json.dumps(
                {
                    "obs_id": index,
                    "episode_id": 0,
                    "obs_env_step": index,
                    "submitted": index not in dropped_indices,
                    **({
                        "worker_id": index % 2,
                        "worker_service_time_ms": value,
                        "observation_to_action_latency_ms": value + 3.0,
                    } if index not in dropped_indices else {}),
                }
            )
            + "\n"
            for index, value in enumerate(service_ms)
        ),
        encoding="utf-8",
    )
    return {
        "measure_session_id": session_id,
        "observation_attempts": attempts,
    }


def test_profile_builder_emits_per_slot_bundle_and_preserves_session_boundaries(
    tmp_path: Path,
):
    sessions = [
        _write_session(
            tmp_path / "sessions",
            0,
            [10.0, 11.0, 50.0, 12.0, 13.0],
            dropped_indices=(4,),
        ),
        _write_session(tmp_path / "sessions", 1, [10.0, 11.0, 60.0, 12.0]),
    ]
    output = tmp_path / "profile"

    profile = build_profile_from_traces(
        sessions=sessions,
        output_dir=output,
        model_id="openvla",
        gpu_class="2x-rtx4090",
        workload_id="flappy",
        instance_id="instance_abc",
        source_run_id="run_1",
        env_fps=25.0,
        metadata={"session_ids": [0, 1]},
    )

    assert {path.name for path in output.iterdir()} == {
        "profile.json",
        "latency_distribution.json",
        "latency_burst_model.json",
        "latency_profile.png",
    }
    assert profile["visualization_path"] == "latency_profile.png"
    assert (output / profile["visualization_path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert profile["source_run_id"] == "run_1"
    assert profile["n_observation_attempts"] == 9
    assert profile["n_admitted_observations"] == 8
    assert profile["summary"]["n_samples"] == 8
    assert profile["n_capacity_drops"] == 1
    assert set(profile["per_slot_summary"]) == {"0", "1"}
    assert (
        profile["per_slot_summary"]["0"]["mean_worker_service_time_ms"]
        > profile["per_slot_summary"]["1"]["mean_worker_service_time_ms"]
    )
    assert profile["per_slot_summary"]["0"]["admitted_count"] == 4
    burst = json.loads((output / "latency_burst_model.json").read_text())
    assert burst["burst_dwell_lengths"] == [1, 1]
    assert len(burst["burst_slot_rank_templates"]) == 2

    with pytest.raises(FileExistsError):
        build_profile_from_traces(
            sessions=sessions,
            output_dir=output,
            model_id="openvla",
            gpu_class="2x-rtx4090",
            workload_id="flappy",
            instance_id="instance_abc",
            source_run_id="run_1",
            env_fps=25.0,
            metadata={"session_ids": [0, 1]},
        )


def test_latency_plot_renders_constant_profile(tmp_path):
    output = tmp_path / "latency_profile.png"

    write_latency_distribution_plot([10.0, 10.0], output)

    image = Image.open(output).convert("RGB")
    colors = image.getcolors(image.width * image.height)
    assert any(color == (165, 71, 98) for _, color in colors)
