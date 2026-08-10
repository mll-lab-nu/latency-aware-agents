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
