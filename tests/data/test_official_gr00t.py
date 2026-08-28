from __future__ import annotations

import json

import pyarrow.parquet as pq

from helpers import save_test_image


def test_official_gr00t_converter_infers_raw_state_from_canonical_rows(tmp_path):
    from latency_bench.data.action_layouts import gymnasium_continuous_action_spec
    from latency_bench.data.official_gr00t import (
        convert_canonical_rollout_to_official_gr00t,
    )
    from latency_bench.data.hf_parquet import write_parquet_splits

    image_path = tmp_path / "image.png"
    save_test_image(image_path, 3)
    labels = [f"joint_{index}" for index in range(29)]
    action_spec = gymnasium_continuous_action_spec(labels)
    raw_export = tmp_path / "raw"
    train_state = [float(index) for index in range(93)]
    val_state = [float(index + 100) for index in range(93)]
    write_parquet_splits(
        raw_export,
        [
            {
                "image": str(image_path),
                "env_name": "g1_sonic_run",
                "episode_idx": 0,
                "decision_step": 0,
                "split": "train",
                "seed": 0,
                "prompt": "run",
                "latency_raw_frames": 0,
                "latency_ms": 0.0,
                "raw_reward": 1.0,
                "action": [float(index) for index in range(29)],
                "action_text": "run",
                "state": train_state,
            }
        ],
        [
            {
                "image": str(image_path),
                "env_name": "g1_sonic_run",
                "episode_idx": 1,
                "decision_step": 0,
                "split": "val",
                "seed": 1,
                "prompt": "run",
                "latency_raw_frames": 1,
                "latency_ms": 20.0,
                "raw_reward": 2.0,
                "action": [float(index + 1) for index in range(29)],
                "action_text": "run",
                "state": val_state,
            }
        ],
        action_spec=action_spec,
    )
    (raw_export / "metadata.json").write_text(
        json.dumps(
            {
                "env_name": "g1_sonic_run",
                "integration_name": "gymnasium",
                "action_layout": "gymnasium_continuous_v1",
                "action_labels": labels,
                "reward_field": "raw_reward",
                "rows_unit": "decision_step",
                "obs_fps": 50.0,
                "obs_stride_raw_frames": 1,
                "gymnasium_task": {
                    "task_name": "g1_sonic_run",
                    "env_id": "LatencyBench/G1SonicRun-v0",
                    "make_kwargs": {},
                    "registration_imports": [],
                    "base_prompt": "run",
                    "env_fps": 50.0,
                    "obs_fps": 50.0,
                    "frame_stack": 1,
                    "action_space": {"type": "box", "labels": labels},
                },
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "official"
    manifest = convert_canonical_rollout_to_official_gr00t(
        raw_export,
        output_dir,
        force=True,
    )

    row = pq.read_table(
        output_dir / "data/chunk-000/episode_000000.parquet"
    ).to_pylist()[0]
    assert manifest["state_dim"] == 93
    assert manifest["action_dim"] == 29
    assert row["observation.state"] == train_state
    assert row["action"] == [float(index) for index in range(29)]
    assert (
        output_dir
        / "videos/chunk-000/observation.image/episode_000000.mp4"
    ).is_file()
    val_output_dir = tmp_path / "official__val"
    val_row = pq.read_table(
        val_output_dir / "data/chunk-000/episode_000000.parquet"
    ).to_pylist()[0]
    assert val_row["observation.state"] == val_state
    assert val_row["action"] == [float(index + 1) for index in range(29)]
    assert (
        val_output_dir
        / "videos/chunk-000/observation.image/episode_000000.mp4"
    ).is_file()
    assert json.loads((output_dir / "meta/info.json").read_text())["features"][
        "observation.image"
    ]["dtype"] == "video"
    assert json.loads((output_dir / "meta/info.json").read_text())["features"][
        "observation.image"
    ]["info"]["video.codec"] == "h264"
    modality = json.loads((output_dir / "meta/modality.json").read_text())
    assert modality["state"]["state"]["end"] == 93
    assert modality["action"]["action"]["end"] == 29
    assert modality["video"] == {
        "ego_view": {"original_key": "observation.image"}
    }
    assert modality["annotation"] == {
        "human.task_description": {"original_key": "task_index"}
    }
    assert json.loads((output_dir / "manifest.json").read_text())[
        "official_embodiment_tag"
    ] == "NEW_EMBODIMENT"
    assert json.loads((tmp_path / "official__val/manifest.json").read_text())[
        "official_action_horizon"
    ] == 40
    assert not (output_dir / "latency_prompt_map.json").exists()
    assert not (tmp_path / "_generated_mixtures").exists()
