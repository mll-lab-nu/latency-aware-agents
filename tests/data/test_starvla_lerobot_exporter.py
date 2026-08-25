from __future__ import annotations

import json

import pyarrow.parquet as pq
import pytest

from helpers import save_test_image


def test_starvla_lerobot_exporter_reads_canonical_rollout_schema(tmp_path):
    from latency_bench.data.action_layouts import action_spec_for_env
    from latency_bench.data.exporters.starvla_lerobot import convert_canonical_rollout_to_lerobot
    from latency_bench.data.hf_parquet import write_parquet_splits

    image_path = tmp_path / "image.png"
    save_test_image(image_path, 9)
    raw_export = tmp_path / "raw"
    action_spec = action_spec_for_env("deadly_corridor")
    row = {
        "image": str(image_path),
        "env_name": "deadly_corridor",
        "episode_idx": 3,
        "decision_step": 0,
        "split": "train",
        "seed": 14,
        "prompt": "pick action",
        "latency_raw_frames": 2,
        "latency_ms": 57.14,
        "raw_reward": 1.0,
        "action_id": 11,
        "action_text": "MOVE_FORWARD + MOVE_RIGHT + ATTACK",
    }
    write_parquet_splits(raw_export, [row], [], action_spec=action_spec)
    (raw_export / "metadata.json").write_text(
        json.dumps(
            {
                "env_name": "deadly_corridor",
                "action_layout": "deadly_corridor_joint_54",
                "reward_field": "raw_reward",
                "action_labels": list(action_spec.labels),
                "rows_unit": "decision_step",
                "obs_fps": 8.75,
                "obs_stride_raw_frames": 4,
            }
        ),
        encoding="utf-8",
    )

    manifest = convert_canonical_rollout_to_lerobot(raw_export, tmp_path / "lerobot", force=True)

    table = pq.read_table(tmp_path / "lerobot" / "data" / "chunk-000" / "episode_000000.parquet")
    assert manifest["action_layout"] == "deadly_corridor_joint_54"
    assert manifest["frames"] == 1
    expected_action = [0.0] * 54
    expected_action[11] = 1.0
    assert table.to_pylist()[0]["action"] == expected_action
    assert table.to_pylist()[0]["done"] is True
    assert table.column_names == [
        "observation.image",
        "observation.state",
        "action",
        "timestamp",
        "episode_index",
        "frame_index",
        "task_index",
        "done",
        "reward",
        "density_weight",
    ]
    assert not (tmp_path / "lerobot" / "meta" / "modality.json").exists()
    assert not (tmp_path / "lerobot" / "latency_prompt_map.json").exists()
    assert not (tmp_path / "lerobot__val" / "manifest.json").exists()


@pytest.mark.parametrize("obs_fps", [15.0, 8.75])
def test_starvla_lerobot_exporter_preserves_decision_rows_at_obs_cadence(tmp_path, obs_fps):
    from latency_bench.data.action_layouts import action_spec_for_env
    from latency_bench.data.exporters.starvla_lerobot import convert_canonical_rollout_to_lerobot
    from latency_bench.data.hf_parquet import write_parquet_splits

    image_path = tmp_path / "image.png"
    save_test_image(image_path, 9)
    raw_export = tmp_path / "raw"
    action_spec = action_spec_for_env("deadly_corridor")
    # Canonical rows are already observation decisions; stride only describes their
    # spacing on the environment's raw-frame clock.
    train_rows = [
        {
            "image": str(image_path),
            "env_name": "deadly_corridor",
            "episode_idx": 0,
            "decision_step": step,
            "split": "train",
            "seed": 7,
            "prompt": "pick action",
            "latency_raw_frames": 2,
            "latency_ms": 57.14,
            "raw_reward": 1.0,
            "action_id": 11,
            "action_text": "MOVE_FORWARD + MOVE_RIGHT + ATTACK",
        }
        for step in range(8)
    ]
    write_parquet_splits(raw_export, train_rows, [], action_spec=action_spec)
    (raw_export / "metadata.json").write_text(
        json.dumps(
            {
                "env_name": "deadly_corridor",
                "action_layout": "deadly_corridor_joint_54",
                "reward_field": "raw_reward",
                "action_labels": list(action_spec.labels),
                "rows_unit": "decision_step",
                "obs_fps": obs_fps,
                "obs_stride_raw_frames": 4,
            }
        ),
        encoding="utf-8",
    )

    manifest = convert_canonical_rollout_to_lerobot(raw_export, tmp_path / "lerobot", force=True)

    table = pq.read_table(tmp_path / "lerobot" / "data" / "chunk-000" / "episode_000000.parquet")
    rows = table.to_pylist()
    assert manifest["frames"] == 8
    assert manifest["obs_fps"] == obs_fps
    assert [r["frame_index"] for r in rows] == list(range(8))
    assert [r["episode_index"] for r in rows] == [0] * 8
    assert [r["timestamp"] for r in rows] == pytest.approx([step / obs_fps for step in range(8)])
    info = json.loads((tmp_path / "lerobot" / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["fps"] == obs_fps
    assert rows[-1]["done"] is True


def test_starvla_lerobot_exporter_requires_rows_unit_metadata(tmp_path):
    from latency_bench.data.exporters.starvla_lerobot import convert_canonical_rollout_to_lerobot

    raw_export = tmp_path / "raw"
    raw_export.mkdir()
    (raw_export / "metadata.json").write_text(
        json.dumps(
            {
                "env_name": "flappy",
                "action_layout": "flappy_discrete",
                "reward_field": "raw_reward",
                "action_labels": ["NOOP", "FLAP"],
                "obs_fps": 30.0,
                "obs_stride_raw_frames": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="rows_unit"):
        convert_canonical_rollout_to_lerobot(raw_export, tmp_path / "lerobot", force=True)


def test_starvla_lerobot_exporter_rejects_non_decision_rows(tmp_path):
    from latency_bench.data.exporters.starvla_lerobot import convert_canonical_rollout_to_lerobot

    raw_export = tmp_path / "raw"
    raw_export.mkdir()
    (raw_export / "metadata.json").write_text(
        json.dumps(
            {
                "env_name": "flappy",
                "action_layout": "flappy_discrete",
                "reward_field": "raw_reward",
                "action_labels": ["NOOP", "FLAP"],
                "rows_unit": "raw_frame",
                "obs_fps": 30.0,
                "obs_stride_raw_frames": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rows_unit must be 'decision_step'"):
        convert_canonical_rollout_to_lerobot(raw_export, tmp_path / "lerobot", force=True)


def test_starvla_lerobot_exporter_does_not_accept_latency_alias(tmp_path):
    from datasets import Dataset, Features, Image as HFImage, Sequence as HFSequence, Value
    from latency_bench.data.exporters.starvla_lerobot import convert_canonical_rollout_to_lerobot

    raw_export = tmp_path / "raw"
    raw_export.mkdir()
    image_path = tmp_path / "image.png"
    save_test_image(image_path, 9)
    features = Features(
        {
            "image": HFImage(),
            "env_name": Value("string"),
            "episode_idx": Value("int64"),
            "decision_step": Value("int64"),
            "split": Value("string"),
            "seed": Value("int64"),
            "prompt": Value("string"),
            "latency": Value("int64"),
            "latency_ms": Value("float64"),
            "raw_reward": Value("float64"),
            "action_id": Value("int64"),
            "action": HFSequence(Value("float32")),
            "action_text": Value("string"),
        }
    )
    Dataset.from_dict(
        {
            "image": [str(image_path)],
            "env_name": ["flappy"],
            "episode_idx": [0],
            "decision_step": [0],
            "split": ["train"],
            "seed": [11],
            "prompt": ["pick action"],
            "latency": [2],
            "latency_ms": [66.0],
            "raw_reward": [1.0],
            "action_id": [1],
            "action": [[0.0, 1.0]],
            "action_text": ["FLAP"],
        },
        features=features,
    ).to_parquet(raw_export / "train.parquet")
    Dataset.from_dict(
        {
            "image": [str(image_path)],
            "env_name": ["flappy"],
            "episode_idx": [1],
            "decision_step": [0],
            "split": ["val"],
            "seed": [12],
            "prompt": ["pick action"],
            "latency": [2],
            "latency_ms": [66.0],
            "raw_reward": [1.0],
            "action_id": [1],
            "action": [[0.0, 1.0]],
            "action_text": ["FLAP"],
        },
        features=features,
    ).to_parquet(raw_export / "val.parquet")
    (raw_export / "metadata.json").write_text(
        json.dumps(
            {
                "env_name": "flappy",
                "action_layout": "flappy_discrete",
                "reward_field": "raw_reward",
                "action_labels": ["NOOP", "FLAP"],
                "rows_unit": "decision_step",
                "obs_fps": 30.0,
                "obs_stride_raw_frames": 1,
            }
        ),
        encoding="utf-8",
    )

    try:
        convert_canonical_rollout_to_lerobot(raw_export, tmp_path / "lerobot", force=True)
    except KeyError as exc:
        assert "latency_raw_frames" in str(exc)
    else:
        raise AssertionError("expected missing latency_raw_frames to fail")


def test_starvla_lerobot_exporter_preserves_high_dimensional_continuous_actions(
    tmp_path,
):
    from latency_bench.data.action_layouts import gymnasium_continuous_action_spec
    from latency_bench.data.exporters.starvla_lerobot import (
        convert_canonical_rollout_to_lerobot,
    )
    from latency_bench.data.hf_parquet import write_parquet_splits

    image_path = tmp_path / "image.png"
    save_test_image(image_path, 7)
    action_labels = [f"actuator_{index}" for index in range(275)]
    action_spec = gymnasium_continuous_action_spec(action_labels)
    actions = [
        [0.0] * 275,
        [index / 275.0 for index in range(275)],
    ]
    state = [float(index) for index in range(177)]
    rows = [
        {
            "image": str(image_path),
            "env_name": "myochallenge_table_tennis_p2",
            "episode_idx": 0,
            "decision_step": step,
            "split": "train",
            "seed": 4,
            "prompt": "Return the ball.",
            "latency_raw_frames": 1,
            "latency_ms": 10.0,
            "raw_reward": 0.0,
            "action": action,
            "state": state,
            "action_text": "continuous_control",
        }
        for step, action in enumerate(actions)
    ]
    raw_export = tmp_path / "raw"
    write_parquet_splits(
        raw_export,
        rows,
        [],
        action_spec=action_spec,
        include_state=True,
    )
    (raw_export / "metadata.json").write_text(
        json.dumps(
            {
                "env_name": "myochallenge_table_tennis_p2",
                "integration_name": "gymnasium",
                "schema_version": 3,
                "include_state": True,
                "action_layout": "gymnasium_continuous_v1",
                "reward_field": "raw_reward",
                "action_labels": action_labels,
                "gymnasium_task": {
                    "task_name": "myochallenge_table_tennis_p2",
                    "env_id": "myoChallengeTableTennisP2-v0",
                    "make_kwargs": {"render_mode": "rgb_array"},
                    "registration_imports": ["myosuite"],
                    "action_space": {
                        "type": "box",
                        "labels": action_labels,
                        "low": [-1.0] * 275,
                        "high": [1.0] * 275,
                        "dtype": "float32",
                    },
                    "noop_action": [0.0] * 275,
                    "base_prompt": "Return the ball.",
                    "env_fps": 100.0,
                    "obs_fps": 100.0,
                    "frame_stack": 1,
                },
                "rows_unit": "decision_step",
                "obs_fps": 100.0,
                "obs_stride_raw_frames": 1,
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "lerobot"
    stale_modality = output_dir / "meta/modality.json"
    stale_prompt_map = output_dir / "latency_prompt_map.json"
    stale_val_modality = tmp_path / "lerobot__val/meta/modality.json"
    for stale_path in (stale_modality, stale_prompt_map, stale_val_modality):
        stale_path.parent.mkdir(parents=True, exist_ok=True)
        stale_path.write_text("{}", encoding="utf-8")
    stale_mixture = tmp_path / "_generated_mixtures/lerobot.json"
    stale_mixture.parent.mkdir()
    stale_mixture.write_text("{}", encoding="utf-8")
    manifest = convert_canonical_rollout_to_lerobot(raw_export, output_dir)
    validation_manifest = json.loads(
        (tmp_path / "lerobot__val/manifest.json").read_text(encoding="utf-8")
    )

    table = pq.read_table(output_dir / "data/chunk-000/episode_000000.parquet")
    rows_out = table.to_pylist()
    info = json.loads((output_dir / "meta/info.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "lerobot_v2_image_parquet"
    assert manifest["action_dim"] == 275
    assert manifest["active_action_dim"] == 275
    assert manifest["state_dim"] == 177
    assert manifest["carrier_action_labels"] == action_labels
    assert rows_out[0]["action"] == [0.0] * 275
    assert rows_out[1]["action"] == pytest.approx(actions[1])
    assert rows_out[0]["observation.state"] == state
    assert info["features"]["action"]["shape"] == [275]
    assert info["features"]["action"]["names"] == action_labels
    assert info["features"]["observation.state"]["shape"] == [177]
    assert [row["timestamp"] for row in rows_out] == pytest.approx([0.0, 0.01])
    assert stale_modality.exists()
    assert stale_val_modality.exists()
    assert not stale_prompt_map.exists()
    assert not stale_mixture.exists()
    for output_manifest in (manifest, validation_manifest):
        assert "robot_type" not in output_manifest
        assert "latency_prompt_map_path" not in output_manifest
        assert "custom_mixtures_path" not in output_manifest


def test_starvla_lerobot_exporter_rejects_unknown_gymnasium_action_layout(tmp_path):
    from latency_bench.data.exporters.starvla_lerobot import (
        convert_canonical_rollout_to_lerobot,
    )

    raw_export = tmp_path / "raw"
    raw_export.mkdir()
    (raw_export / "metadata.json").write_text(
        json.dumps(
            {
                "env_name": "unknown_task",
                "integration_name": "gymnasium",
                "action_layout": "gymnasium_unknown_v1",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported Gymnasium action_layout"):
        convert_canonical_rollout_to_lerobot(raw_export, tmp_path / "lerobot")


def test_starvla_lerobot_exporter_rejects_unknown_gymnasium_schema_version(tmp_path):
    from latency_bench.data.exporters.starvla_lerobot import (
        convert_canonical_rollout_to_lerobot,
    )

    raw_export = tmp_path / "raw"
    raw_export.mkdir()
    (raw_export / "metadata.json").write_text(
        json.dumps(
            {
                "env_name": "unknown_task",
                "integration_name": "gymnasium",
                "schema_version": 99,
                "action_layout": "gymnasium_continuous_v1",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported Gymnasium schema_version=99"):
        convert_canonical_rollout_to_lerobot(raw_export, tmp_path / "lerobot")


def test_starvla_lerobot_exporter_requires_include_state_for_schema_v3(tmp_path):
    from latency_bench.data.exporters.starvla_lerobot import (
        convert_canonical_rollout_to_lerobot,
    )

    raw_export = tmp_path / "raw"
    raw_export.mkdir()
    (raw_export / "metadata.json").write_text(
        json.dumps(
            {
                "env_name": "unknown_task",
                "integration_name": "gymnasium",
                "schema_version": 3,
                "action_layout": "gymnasium_continuous_v1",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="include_state"):
        convert_canonical_rollout_to_lerobot(raw_export, tmp_path / "lerobot")
