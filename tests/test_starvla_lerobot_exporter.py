from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _save_image(path: Path) -> None:
    from PIL import Image

    Image.fromarray(np.full((2, 2, 3), 9, dtype=np.uint8)).save(path)


def test_starvla_lerobot_exporter_reads_canonical_rollout_schema(tmp_path):
    from latency_bench.data.action_layouts import action_spec_for_env
    from latency_bench.data.exporters.starvla_lerobot import convert_canonical_rollout_to_lerobot
    from latency_bench.data.hf_parquet import write_parquet_splits

    image_path = tmp_path / "image.png"
    _save_image(image_path)
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


def test_starvla_lerobot_exporter_does_not_accept_latency_alias(tmp_path):
    from datasets import Dataset, Features, Image as HFImage, Sequence as HFSequence, Value
    from latency_bench.data.exporters.starvla_lerobot import convert_canonical_rollout_to_lerobot

    raw_export = tmp_path / "raw"
    raw_export.mkdir()
    image_path = tmp_path / "image.png"
    _save_image(image_path)
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
