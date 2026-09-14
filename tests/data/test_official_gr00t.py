from __future__ import annotations

import json

import pyarrow.parquet as pq
import pytest

from helpers import save_test_image
from latency_bench.envs.humanoidbench_sonic_contract import (
    G1_GR00T_ACTION_DIM,
)


def test_official_gr00t_converter_infers_raw_state_from_canonical_rows(tmp_path):
    from latency_bench.data.action_layouts import G1_SONIC_GR00T_ACTION_SPEC
    from latency_bench.data.official_gr00t import (
        convert_canonical_rollout_to_official_gr00t,
    )
    from latency_bench.data.hf_parquet import write_parquet_splits

    image_path = tmp_path / "image.png"
    save_test_image(image_path, 3)
    labels = list(G1_SONIC_GR00T_ACTION_SPEC.labels)
    action_spec = G1_SONIC_GR00T_ACTION_SPEC
    raw_export = tmp_path / "raw"
    state_dim = 198
    train_state = [float(index) for index in range(state_dim)]
    val_state = [float(index + 100) for index in range(state_dim)]
    write_parquet_splits(
        raw_export,
        [
            {
                "image": str(image_path),
                "env_name": "g1_sonic_balance_simple",
                "episode_idx": 0,
                "decision_step": 0,
                "split": "train",
                "seed": 0,
                "prompt": "Keep balance.",
                "latency_raw_frames": 0,
                "latency_ms": 0.0,
                "raw_reward": 1.0,
                "action": [float(index) for index in range(G1_GR00T_ACTION_DIM)],
                "action_text": "Keep balance.",
                "state": train_state,
            }
        ],
        [
            {
                "image": str(image_path),
                "env_name": "g1_sonic_balance_simple",
                "episode_idx": 1,
                "decision_step": 0,
                "split": "val",
                "seed": 1,
                "prompt": "Keep balance.",
                "latency_raw_frames": 1,
                "latency_ms": 20.0,
                "raw_reward": 2.0,
                "action": [float(index + 1) for index in range(G1_GR00T_ACTION_DIM)],
                "action_text": "Keep balance.",
                "state": val_state,
            }
        ],
        action_spec=action_spec,
    )
    (raw_export / "metadata.json").write_text(
        json.dumps(
            {
                "env_name": "g1_sonic_balance_simple",
                "integration_name": "gymnasium",
                "action_layout": G1_SONIC_GR00T_ACTION_SPEC.layout,
                "action_labels": labels,
                "reward_field": "raw_reward",
                "rows_unit": "decision_step",
                "obs_fps": 50.0,
                "obs_stride_raw_frames": 1,
                "gymnasium_task": {
                    "task_name": "g1_sonic_balance_simple",
                    "env_id": "LatencyBench/G1SonicRun-v0",
                    "make_kwargs": {},
                    "registration_imports": [],
                    "base_prompt": "Keep balance.",
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
    assert manifest["state_dim"] == state_dim
    assert manifest["action_dim"] == G1_GR00T_ACTION_DIM
    assert row["observation.state"] == train_state
    assert row["action"] == [float(index) for index in range(G1_GR00T_ACTION_DIM)]
    assert (
        output_dir
        / "videos/chunk-000/observation.image/episode_000000.mp4"
    ).is_file()
    val_output_dir = tmp_path / "official__val"
    val_row = pq.read_table(
        val_output_dir / "data/chunk-000/episode_000000.parquet"
    ).to_pylist()[0]
    assert val_row["observation.state"] == val_state
    assert val_row["action"] == [
        float(index + 1) for index in range(G1_GR00T_ACTION_DIM)
    ]
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
    assert modality["state"]["state"]["end"] == state_dim
    assert modality["action"]["action"]["end"] == G1_GR00T_ACTION_DIM
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


def test_official_gr00t_converter_rejects_a_non_g1_action_layout(tmp_path):
    from latency_bench.data.official_gr00t import (
        convert_canonical_rollout_to_official_gr00t,
    )

    raw_export = tmp_path / "raw"
    raw_export.mkdir()
    (raw_export / "metadata.json").write_text(
        json.dumps(
            {
                "action_layout": "gymnasium_continuous_v1",
                "action_labels": [
                    f"joint_{index}" for index in range(G1_GR00T_ACTION_DIM)
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="g1_sonic_gr00t_v2"):
        convert_canonical_rollout_to_official_gr00t(
            raw_export,
            tmp_path / "official",
        )


def test_official_gr00t_converter_rejects_misordered_action_labels(tmp_path):
    from latency_bench.data.action_layouts import G1_SONIC_GR00T_ACTION_SPEC
    from latency_bench.data.official_gr00t import (
        convert_canonical_rollout_to_official_gr00t,
    )

    raw_export = tmp_path / "raw"
    raw_export.mkdir()
    (raw_export / "metadata.json").write_text(
        json.dumps(
            {
                "action_layout": G1_SONIC_GR00T_ACTION_SPEC.layout,
                "action_labels": list(reversed(G1_SONIC_GR00T_ACTION_SPEC.labels)),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="action labels"):
        convert_canonical_rollout_to_official_gr00t(
            raw_export,
            tmp_path / "official",
        )


def test_official_gr00t_dataset_converter_supports_parkour_contract(tmp_path):
    from latency_bench.data.action_layouts import (
        GO2_PARKOUR_TERRAIN_LATENT_YAW_ACTION_SPEC,
    )
    from latency_bench.data.official_gr00t import (
        convert_canonical_rollout_to_official_gr00t_dataset,
    )
    from latency_bench.data.hf_parquet import write_parquet_splits

    image_path = tmp_path / "image.png"
    save_test_image(image_path, 5)
    action_spec = GO2_PARKOUR_TERRAIN_LATENT_YAW_ACTION_SPEC
    raw_export = tmp_path / "raw"
    state = [float(index) for index in range(53)]
    action = [float(index) for index in range(34)]
    row = {
        "image": str(image_path),
        "env_name": "extreme_parkour_go2",
        "episode_idx": 0,
        "decision_step": 0,
        "split": "train",
        "seed": 0,
        "prompt": "Traverse the parkour course.",
        "latency_raw_frames": 0,
        "latency_ms": 0.0,
        "raw_reward": 1.0,
        "action": action,
        "action_text": str(action),
        "state": state,
    }
    write_parquet_splits(raw_export, [row], [], action_spec=action_spec)
    (raw_export / "metadata.json").write_text(
        json.dumps(
            {
                "env_name": "extreme_parkour_go2",
                "integration_name": "isaaclab_parkour",
                "action_layout": "go2_parkour_terrain_latent_yaw_v1",
                "action_labels": list(action_spec.labels),
                "state_labels": [f"state_{i}" for i in range(53)],
                "reward_field": "raw_reward",
                "rows_unit": "decision_step",
                "obs_fps": 10.0,
                "obs_stride_raw_frames": 5,
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "official"
    manifest = convert_canonical_rollout_to_official_gr00t_dataset(
        raw_export,
        output_dir,
        action_dim=34,
        action_horizon=1,
        modality_config="latency_bench/policy/official_gr00t_parkour_modality.py",
        embodiment_tag="NEW_EMBODIMENT",
        action_layout="go2_parkour_terrain_latent_yaw_v1",
        action_labels=action_spec.labels,
        force=True,
    )

    assert manifest["action_dim"] == 34
    assert manifest["state_dim"] == 53
    assert manifest["official_action_horizon"] == 1
    assert manifest["official_modality_config"].endswith(
        "official_gr00t_parkour_modality.py"
    )
    modality = json.loads((output_dir / "meta/modality.json").read_text())
    assert modality["state"]["state"]["end"] == 53
    assert modality["action"]["action"]["end"] == 34
