from __future__ import annotations

import io
from pathlib import Path
import json

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from helpers import REPO_ROOT, save_test_image


def test_load_rows_preserves_nested_values_and_order_across_batches(tmp_path):
    from latency_bench.integrations.starvla.lerobot_export import _load_rows

    rows = [
        {
            "image": {"bytes": f"image-{index}".encode(), "path": None},
            "episode_idx": index // 1000,
            "decision_step": index % 1000,
            "state": [float(index), 0.5],
            "issued_command_chunk": [[float(index), float(step)] for step in range(8)],
            "action_prefix_mask": [False, False, True, True, False, False, False, False],
            "action_loss_mask": [False, False, True, True, True, True, True, True],
        }
        for index in range(2053)
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "train.parquet", row_group_size=257)

    assert _load_rows(tmp_path, "train") == rows


def test_balance_preparation_preserves_raw_state_masks_and_training_statistics(tmp_path):
    """Long-lived task handoff: Balance owns its annotation, masks and frozen statistics."""
    from latency_bench.data.hf_parquet import embed_image_rows
    from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot
    from latency_bench.integrations.starvla.task_contracts import prepare_lerobot, task_spec

    spec = task_spec("balance_profile_h8")
    labels = [f"joint_{index}" for index in range(spec.action_dim)]
    image = tmp_path / "image.png"
    save_test_image(image, 7)
    raw = tmp_path / "raw"
    rows = [{
        "image": str(image), "env_name": spec.name, "episode_idx": episode,
        "decision_step": 0, "split": "train" if episode < 2 else "val", "seed": episode,
        "prompt": "Keep balance.", "latency_raw_frames": 2, "latency_ms": 40.0,
        "raw_reward": 1.0, "state": [value] * spec.state_dim,
        "action": [value / 10] * spec.action_dim, "action_text": "balance",
        "issued_command_chunk": [[value / 10] * spec.action_dim] * spec.action_horizon,
        "action_prefix_mask": [True, True] + [False] * 6,
        "action_loss_mask": [False, False] + [True] * 6,
    } for episode, value in enumerate([2.0, 4.0, 6.0])]
    # Balance's recipe writes chunk/mask columns outside the single-action schema.
    raw.mkdir()
    train = pa.Table.from_pylist(embed_image_rows(rows[:2]))
    pq.write_table(train, raw / "train.parquet")
    pq.write_table(pa.Table.from_pylist(embed_image_rows(rows[2:]), schema=train.schema), raw / "val.parquet")
    (raw / "metadata.json").write_text(json.dumps({
        "env_name": spec.name, "integration_name": "gymnasium", "action_layout": "gymnasium_continuous_v1",
        "action_labels": labels, "reward_field": "raw_reward", "rows_unit": "decision_step",
        "obs_fps": spec.fps, "obs_stride_raw_frames": 1,
        "gymnasium_task": {"task_name": spec.name, "action_space": {"type": "box", "labels": labels}},
        "label_provenance": {"semantics": "teacher_issued_command"},
    }))
    converted = tmp_path / "converted"
    convert_canonical_rollout_to_lerobot(raw, converted, write_starvla_handoff=False)
    output = tmp_path / "prepared"
    prepare_lerobot("balance_profile_h8", converted, tmp_path / "converted__val", output)
    train_contract = json.loads((output / "task_contract.json").read_text())
    for name, expected_rows in (("prepared", rows[:2]), ("prepared__val", rows[2:])):
        directory = tmp_path / name
        actual = [row for path in sorted((directory / "data").rglob("*.parquet"))
                  for row in pq.read_table(path).to_pylist()]
        for source, target in zip(expected_rows, actual, strict=True):
            assert target["observation.state"] == source["state"]
            for key in ("action_prefix_mask", "action_loss_mask"):
                assert target[key] == source[key]
            np.testing.assert_allclose(target["issued_command_chunk"], source["issued_command_chunk"])
        modality = json.loads((directory / "meta/modality.json").read_text())
        assert modality["annotation"] == {"human.task_description": {"original_key": "task_index"}}
        contract = json.loads((directory / "task_contract.json").read_text())
        assert contract["normalization"] == train_contract["normalization"]
        assert contract["normalization"]["state"]["min"] == [2.0] * spec.state_dim
        assert contract["normalization"]["state"]["max"] == [4.0] * spec.state_dim



def test_starvla_lerobot_exporter_reads_canonical_rollout_schema(tmp_path):
    from latency_bench.data.action_layouts import action_spec_for_env
    from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot
    from helpers import write_rollout_fixture

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
    write_rollout_fixture(raw_export, [row], [], action_spec=action_spec)
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


@pytest.mark.parametrize("task,state_dim", [("deadly_corridor", 120), ("atlantis", 56), ("asterix", 136)])
def test_history_export_preserves_state_and_task_action_contract(tmp_path, task, state_dim):
    from latency_bench.data.action_layouts import action_spec_for_env
    from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot
    from helpers import write_rollout_fixture
    from latency_bench.core.config import load_config
    from latency_bench.envs.factory import resolve_env_config
    from latency_bench.data.action_layouts import gymnasium_discrete_action_spec
    from latency_bench.envs.tasks.contracts import gymnasium_task_contract

    if task == "deadly_corridor":
        action_spec = action_spec_for_env(task)
        action_id = 2 * 18 + 1 * 6 + 2 * 2 + 1
        expected_action = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
        task_metadata = {}
    else:
        env = load_config(f"configs/base/{task}.yaml")["env"]
        env["frame_stack"] = 4
        action_spec = gymnasium_discrete_action_spec(env["action_order"])
        action_id = len(action_spec.labels) - 1
        expected_action = [0.0] * action_id + [1.0]
        resolve_env_config({"env": env})
        task_metadata = {"integration_name": "gymnasium", "gymnasium_task": gymnasium_task_contract(env)}

    image_path = tmp_path / "image.png"
    save_test_image(image_path, 9)
    raw_export = tmp_path / "raw"
    row = {
        "image": str(image_path), "env_name": task,
        "episode_idx": 0, "decision_step": 0, "split": "train", "seed": 42,
        "prompt": "Choose an action.", "latency_raw_frames": 3,
        "latency_ms": 85.7, "raw_reward": 0.0,
        "action_id": action_id,
        "action_text": action_spec.labels[action_id],
        "state": [0.0] * (state_dim - 1) + [1.0],
    }
    if task != "deadly_corridor":
        row["action"] = expected_action
    write_rollout_fixture(raw_export, [row], [], action_spec=action_spec)
    (raw_export / "metadata.json").write_text(json.dumps({
        "env_name": task, "action_layout": action_spec.layout,
        "action_labels": list(action_spec.labels),
        "reward_field": "raw_reward", "rows_unit": "decision_step",
        "obs_fps": 8.75, "obs_stride_raw_frames": 4,
        "action_history_decisions": 8,
        **task_metadata,
    }))

    manifest = convert_canonical_rollout_to_lerobot(raw_export, tmp_path / "lerobot")
    converted = pq.read_table(tmp_path / "lerobot/data/chunk-000/episode_000000.parquet").to_pylist()[0]

    assert converted["observation.state"] == row["state"]
    assert converted["action"] == expected_action
    assert manifest["action_layout"] == ("multibinary_7" if task == "deadly_corridor" else action_spec.layout)
    assert manifest["state_normalization"] == {"min": [0.0] * state_dim, "max": [1.0] * state_dim}
    for key, value in task_metadata.items():
        assert manifest[key] == ({**value, "frame_stack": 1} if key == "gymnasium_task" else value)
    modality = json.loads((tmp_path / "lerobot/meta/modality.json").read_text())
    assert modality["annotation"] == {
        "human.action.task_description": {"original_key": "task_index"}
    }


@pytest.mark.parametrize("obs_fps", [15.0, 8.75])
def test_starvla_lerobot_exporter_preserves_decision_rows_at_obs_cadence(tmp_path, obs_fps):
    from latency_bench.data.action_layouts import action_spec_for_env
    from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot
    from helpers import write_rollout_fixture

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
    write_rollout_fixture(raw_export, train_rows, [], action_spec=action_spec)
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
    from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot

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
    from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot

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


@pytest.mark.parametrize("required_field", ["latency_raw_frames", "action"])
def test_starvla_lerobot_exporter_requires_canonical_row_fields(tmp_path, required_field):
    """Long-lived conversion boundary: aliases and action IDs do not replace required fields."""
    from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot

    raw_export = tmp_path / "raw"
    raw_export.mkdir()
    image_path = tmp_path / "image.png"
    save_test_image(image_path, 9)
    row = {
        "image": {"bytes": image_path.read_bytes(), "path": None},
        "env_name": "flappy", "episode_idx": 0, "decision_step": 0,
        "seed": 11, "prompt": "pick action", "latency_raw_frames": 2, "latency": 2,
        "latency_ms": 66.0, "raw_reward": 1.0,
        "action_id": 1, "action": [0.0, 1.0], "action_text": "FLAP",
    }
    del row[required_field]
    for split in ("train", "val"):
        pq.write_table(pa.Table.from_pylist([row | {"split": split}]), raw_export / f"{split}.parquet")
    (raw_export / "metadata.json").write_text(json.dumps({
        "env_name": "flappy", "action_layout": "flappy_discrete", "reward_field": "raw_reward",
        "action_labels": ["NOOP", "FLAP"], "rows_unit": "decision_step",
        "obs_fps": 30.0, "obs_stride_raw_frames": 1,
    }))
    with pytest.raises(KeyError, match=required_field):
        convert_canonical_rollout_to_lerobot(raw_export, tmp_path / "lerobot", force=True)


def test_starvla_lerobot_exporter_preserves_high_dimensional_continuous_actions(
    tmp_path,
):
    from latency_bench.data.action_layouts import gymnasium_continuous_action_spec
    from latency_bench.integrations.starvla.lerobot_export import (
        convert_canonical_rollout_to_lerobot,
    )
    from helpers import write_rollout_fixture

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
            "env_name": "synthetic_continuous",
            "episode_idx": 0,
            "decision_step": step,
            "split": "train",
            "seed": 4,
            "prompt": "Control the actuators.",
            "latency_raw_frames": 1,
            "latency_ms": 10.0,
            "raw_reward": 0.0,
            "action": action,
            "state": [value + step for value in state],
            "action_text": "continuous_control",
        }
        for step, action in enumerate(actions)
    ]
    raw_export = tmp_path / "raw"
    write_rollout_fixture(
        raw_export,
        rows,
        [],
        action_spec=action_spec,
    )
    (raw_export / "metadata.json").write_text(
        json.dumps(
            {
                "env_name": "synthetic_continuous",
                "integration_name": "gymnasium",
                "action_layout": "gymnasium_continuous_v1",
                "reward_field": "raw_reward",
                "action_labels": action_labels,
                "gymnasium_task": {
                    "task_name": "synthetic_continuous",
                    "env_id": "SyntheticContinuous-v0",
                    "make_kwargs": {"render_mode": "rgb_array"},
                    "registration_imports": [],
                    "action_space": {
                        "type": "box",
                        "labels": action_labels,
                        "low": [-1.0] * 275,
                        "high": [1.0] * 275,
                        "dtype": "float32",
                    },
                    "noop_action": [0.0] * 275,
                    "base_prompt": "Control the actuators.",
                    "env_fps": 100.0,
                    "obs_fps": 100.0,
                    "frame_stack": 1,
                    "state_space": {
                        "labels": [f"state_{index}" for index in range(177)]
                    },
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
    modality = json.loads(
        (output_dir / "meta/modality.json").read_text(encoding="utf-8")
    )
    assert manifest["format"] == "lerobot_v2_image_parquet"
    assert manifest["action_dim"] == 275
    assert manifest["active_action_dim"] == 275
    assert manifest["state_dim"] == 177
    assert manifest["carrier_action_labels"] == action_labels
    assert manifest["action_carrier"] == "native"
    assert manifest["robot_type"] == "rl_games_gymnasium"
    assert rows_out[0]["action"] == [0.0] * 275
    assert rows_out[1]["action"] == pytest.approx(actions[1])
    assert rows_out[0]["observation.state"] == state
    assert rows_out[1]["observation.state"] == [value + 1 for value in state]
    assert info["features"]["action"]["shape"] == [275]
    assert info["features"]["action"]["names"] == action_labels
    assert info["features"]["observation.state"]["shape"] == [177]
    assert modality["state"]["game_state"]["original_key"] == "observation.state"
    assert modality["action"]["button"]["original_key"] == "action"
    assert [row["timestamp"] for row in rows_out] == pytest.approx([0.0, 0.01])
    assert stale_modality.exists()
    assert stale_val_modality.exists()
    assert stale_prompt_map.exists()
    assert stale_mixture.exists()
    for output_manifest in (manifest, validation_manifest):
        assert output_manifest["robot_type"] == "rl_games_gymnasium"
        assert output_manifest["latency_prompt_map_path"]
        assert output_manifest["custom_mixtures_path"]


@pytest.mark.parametrize(
    ("action_carrier", "action_dim"),
    [("bridge", 7), ("native", 6)],
)
def test_starvla_lerobot_exporter_writes_handoff_for_generic_continuous_state(
    tmp_path,
    action_carrier,
    action_dim,
):
    from latency_bench.data.action_layouts import gymnasium_continuous_action_spec
    from latency_bench.integrations.starvla.lerobot_export import (
        convert_canonical_rollout_to_lerobot,
    )
    from helpers import write_rollout_fixture

    image_path = tmp_path / "image.png"
    save_test_image(image_path, 7)
    action_labels = [f"torque_{index}" for index in range(6)]
    state_labels = [f"state_{index}" for index in range(17)]
    raw_export = tmp_path / "raw"
    write_rollout_fixture(
        raw_export,
        [
            {
                "image": str(image_path),
                "env_name": "continuous_state_task",
                "episode_idx": 0,
                "decision_step": 0,
                "split": "train",
                "seed": 4,
                "prompt": "Walk forward.",
                "latency_raw_frames": 0,
                "latency_ms": 0.0,
                "raw_reward": 1.0,
                "state": [float(index) for index in range(17)],
                "action": [0.1] * 6,
                "action_text": "continuous_control",
            },
            {
                "image": str(image_path),
                "env_name": "continuous_state_task",
                "episode_idx": 0,
                "decision_step": 1,
                "split": "train",
                "seed": 4,
                "prompt": "Walk forward.",
                "latency_raw_frames": 0,
                "latency_ms": 0.0,
                "raw_reward": 1.0,
                "state": [float(index + 1) for index in range(17)],
                "action": [0.2] * 6,
                "action_text": "continuous_control",
            },
        ],
        [],
        action_spec=gymnasium_continuous_action_spec(action_labels),
    )
    (raw_export / "metadata.json").write_text(
        json.dumps(
            {
                "env_name": "continuous_state_task",
                "integration_name": "gymnasium",
                "action_layout": "gymnasium_continuous_v1",
                "reward_field": "raw_reward",
                "action_labels": action_labels,
                "gymnasium_task": {
                    "task_name": "continuous_state_task",
                    "env_id": "Walker2d-v4",
                    "make_kwargs": {"render_mode": "rgb_array"},
                    "registration_imports": [],
                    "action_space": {
                        "type": "box",
                        "labels": action_labels,
                        "low": [-1.0] * 6,
                        "high": [1.0] * 6,
                        "dtype": "float32",
                        "openvla_carrier_dim": 7,
                        "openvla_padding": [0.0],
                    },
                    "state_space": {"labels": state_labels},
                    "noop_action": [0.0] * 6,
                    "base_prompt": "Walk forward.",
                    "env_fps": 125.0,
                    "obs_fps": 125.0,
                    "frame_stack": 1,
                },
                "rows_unit": "decision_step",
                "obs_fps": 125.0,
                "obs_stride_raw_frames": 1,
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / f"continuous_state__{action_carrier}"
    manifest = convert_canonical_rollout_to_lerobot(
        raw_export,
        output_dir,
        force=True,
        action_carrier=action_carrier,
    )

    table = pq.read_table(output_dir / "data/chunk-000/episode_000000.parquet")
    modality = json.loads((output_dir / "meta/modality.json").read_text())
    assert manifest["format"] == "lerobot_v2_image_parquet"
    assert manifest["action_dim"] == action_dim
    assert manifest["active_action_dim"] == 6
    assert manifest["state_dim"] == 17
    assert manifest["robot_type"] == "rl_games_gymnasium"
    if action_carrier == "bridge":
        assert manifest["gymnasium_task"]["action_space"]["openvla_carrier_dim"] == 7
    assert table.to_pylist()[0]["action"] == pytest.approx(
        [0.1] * 6 + ([0.0] if action_carrier == "bridge" else [])
    )
    assert table.to_pylist()[0]["observation.state"] == pytest.approx(list(range(17)))
    assert table.to_pylist()[1]["observation.state"] == pytest.approx(list(range(1, 18)))
    assert modality["state"]["game_state"]["end"] == 17
    assert (output_dir / "latency_prompt_map.json").exists()
    assert (
        tmp_path / f"_generated_mixtures/continuous_state__{action_carrier}.json"
    ).exists()
    assert manifest["state_labels"] == state_labels
    assert manifest["state_normalization"] == {
        "type": "min_max", "min": [float(index) for index in range(17)],
        "max": [float(index + 1) for index in range(17)],
    }
    info = json.loads((output_dir / "meta/info.json").read_text())
    assert info["features"]["observation.state"]["names"] == state_labels


def test_starvla_lerobot_exporter_rejects_unknown_gymnasium_action_layout(tmp_path):
    from latency_bench.integrations.starvla.lerobot_export import (
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
                "action_labels": ["action"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported Gymnasium action_layout"):
        convert_canonical_rollout_to_lerobot(raw_export, tmp_path / "lerobot")


ACTION_LABELS = ["PUSH_LEFT", "COAST", "PUSH_RIGHT"]


def _canonical_row(
    image_path: Path,
    *,
    action_id: int = 1,
    task_name: str = "mountain_car",
    decision_step: int = 0,
    context_images: list[Path] | None = None,
    state: list[float] | None = None,
) -> dict:
    action = [0.0] * len(ACTION_LABELS)
    action[action_id] = 1.0
    row = {
        "image": str(image_path),
        "env_name": task_name,
        "episode_idx": 0,
        "decision_step": decision_step,
        "split": "train",
        "seed": 4,
        "prompt": "Drive the car up the hill.",
        "latency_raw_frames": 1,
        "latency_ms": 33.33,
        "raw_reward": -1.0,
        "action_id": action_id,
        "action": action,
        "action_text": ACTION_LABELS[action_id],
    }
    if context_images is not None:
        row["context_images"] = [str(path) for path in context_images]
    if state is not None:
        row["state"] = state
    return row


def _write_generic_export(
    raw_export: Path,
    train_rows: list[dict],
    *,
    task_name: str = "mountain_car",
    integration_name: str = "gymnasium",
    include_state: bool = False,
    state_labels: list[str] | None = None,
) -> None:
    from latency_bench.data.action_layouts import gymnasium_discrete_action_spec
    from helpers import write_rollout_fixture

    action_spec = gymnasium_discrete_action_spec(ACTION_LABELS)
    write_rollout_fixture(
        raw_export,
        train_rows,
        [],
        action_spec=action_spec,
    )
    metadata = {
        "env_name": task_name,
        "integration_name": integration_name,
        "schema_version": 2,
        "action_layout": "gymnasium_discrete_v1",
        "reward_field": "raw_reward",
        "action_labels": list(action_spec.labels),
        "gymnasium_task": {
            "task_name": task_name,
            "env_id": "MountainCar-v0",
            "make_kwargs": {"render_mode": "rgb_array"},
            "registration_imports": [],
            "action_labels": list(action_spec.labels),
            "action_values": [0, 1, 2],
            "noop_action_id": 1,
            "base_prompt": "Drive the car up the hill.",
            "env_fps": 30.0,
            "obs_fps": 10.0,
            "frame_stack": 1,
        },
        "rows_unit": "decision_step",
        "obs_fps": 10.0,
        "obs_stride_raw_frames": 3,
    }
    if include_state:
        labels = state_labels or [f"state_{index}" for index in range(len(train_rows[0]["state"]))]
        metadata["gymnasium_task"]["state_space"] = {"labels": labels}
    elif state_labels is not None:
        metadata["gymnasium_task"]["state_space"] = {"labels": state_labels}
    (raw_export / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def test_starvla_lerobot_converter_writes_handoff_contract(tmp_path):
    from latency_bench.integrations.starvla.lerobot_export import (
        convert_canonical_rollout_to_lerobot,
    )

    image_path = tmp_path / "image.png"
    save_test_image(image_path, 7)
    raw_export = tmp_path / "raw"
    # The generic task name may match a built-in environment name.
    rows = [_canonical_row(image_path, task_name="flappy", decision_step=step) for step in range(2)]
    _write_generic_export(raw_export, rows, task_name="flappy")

    output_dir = tmp_path / "lerobot"
    manifest = convert_canonical_rollout_to_lerobot(raw_export, output_dir, force=True)

    table = pq.read_table(output_dir / "data/chunk-000/episode_000000.parquet")
    assert table["density_weight"].to_pylist() == [1.0, 1.0]
    row = table.to_pylist()[0]
    info = json.loads((output_dir / "meta/info.json").read_text(encoding="utf-8"))
    modality = json.loads(
        (output_dir / "meta/modality.json").read_text(encoding="utf-8")
    )
    prompt_map = json.loads(
        (output_dir / "latency_prompt_map.json").read_text(encoding="utf-8")
    )
    mixtures = json.loads(
        (tmp_path / "_generated_mixtures/lerobot.json").read_text(encoding="utf-8")
    )
    validation_manifest = json.loads(
        (tmp_path / "lerobot__val/manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["action_dim"] == 3
    assert manifest["active_action_dim"] == 3
    assert manifest["action_carrier"] == "native"
    assert manifest["robot_type"] == "rl_games_gymnasium"
    assert manifest["integration_name"] == "gymnasium"
    assert manifest["task_name"] == "flappy"
    assert manifest["gymnasium_task"]["action_values"] == [0, 1, 2]
    assert manifest["gymnasium_task"]["noop_action_id"] == 1
    assert validation_manifest["integration_name"] == "gymnasium"
    assert validation_manifest["task_name"] == "flappy"
    assert validation_manifest["gymnasium_task"] == manifest["gymnasium_task"]
    assert validation_manifest["active_action_dim"] == 3
    assert validation_manifest["training_dataset_name"] == "lerobot"
    assert row["action"] == [0.0, 1.0, 0.0]
    assert row["observation.state"] == [0.0]
    assert info["chunks_size"] == 1000
    assert info["data_path"] == (
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    assert info["video_path"] == (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    assert info["features"]["observation.image"] == {
        "dtype": "image",
        "shape": [2, 2, 3],
        "names": ["height", "width", "channel"],
        "video_info": {"video.fps": 10.0},
    }
    assert info["features"]["observation.state"] == {
        "dtype": "float32",
        "shape": [1],
        "names": ["state"],
    }
    assert info["features"]["action"] == {
        "dtype": "float32",
        "shape": [3],
        "names": ACTION_LABELS,
    }
    assert info["features"]["latency"] == {"dtype": "int64", "shape": [1]}
    assert modality == {
        "state": {
            "game_state": {
                "start": 0,
                "end": 1,
                "dtype": "float32",
                "absolute": True,
                "original_key": "observation.state",
            }
        },
        "action": {
            "button": {
                "start": 0,
                "end": 3,
                "dtype": "float32",
                "absolute": True,
                "original_key": "action",
            }
        },
        "video": {"image": {"original_key": "observation.image"}},
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }
    assert prompt_map == {
        "1": {
            "prompt": "Drive the car up the hill.",
            "latency_raw_frames": 1,
            "latency_ms": 33.33,
        }
    }
    assert mixtures == {
        "lerobot": [["lerobot", 1.0, "rl_games_gymnasium"]],
        "lerobot__val": [["lerobot__val", 1.0, "rl_games_gymnasium"]],
    }


def test_generic_loader_normalizes_raw_state_across_training_sources(tmp_path, monkeypatch):
    """Long-lived converter/mixture contract: preserve raw units and the Gymnasium loader transform."""
    pytest.importorskip("accelerate")
    monkeypatch.syspath_prepend(str(REPO_ROOT / "third_party/starVLA"))
    from omegaconf import OmegaConf
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    from latency_bench.data.action_layouts import gymnasium_discrete_action_spec
    from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot
    from helpers import write_rollout_fixture

    manifest = None
    for name, train_values in (("first", [2.0, 4.0]), ("second", [6.0, 8.0])):
        image = tmp_path / f"{name}.png"
        save_test_image(image, 7)
        raw = tmp_path / f"raw_{name}"
        train_rows = [_canonical_row(image, decision_step=step, state=[value])
                      for step, value in enumerate(train_values)]
        val_rows = [{**_canonical_row(image, decision_step=step, state=[value]),
                     "episode_idx": 1, "split": "val"}
                    for step, value in enumerate([3.0, 10.0])]
        _write_generic_export(raw, train_rows, include_state=True)
        write_rollout_fixture(raw, train_rows, val_rows,
                              action_spec=gymnasium_discrete_action_spec(ACTION_LABELS))
        manifest = convert_canonical_rollout_to_lerobot(raw, tmp_path / name)
        for dataset_name, expected in ((name, train_values), (f"{name}__val", [3.0, 10.0])):
            rows = pq.read_table(tmp_path / dataset_name / "data/chunk-000/episode_000000.parquet").to_pylist()
            assert [row["observation.state"] for row in rows] == [[value] for value in expected]
        assert manifest["state_normalization"] == {
            "type": "min_max", "min": [train_values[0]], "max": [train_values[1]],
        }

    mixtures = tmp_path / "mixtures.json"
    mixtures.write_text(json.dumps({
        "train": [["first", 1.0, "rl_games_gymnasium"], ["second", 1.0, "rl_games_gymnasium"]],
        "validation": [["first__val", 1.0, "rl_games_gymnasium"]],
    }))
    cfg = OmegaConf.create({
        "data_root_dir": str(tmp_path), "data_mix": "train", "custom_mixtures_path": str(mixtures),
        "video_backend": "pyav", "include_state": True, "shuffle": False,
        "gymnasium_task_contract": manifest["gymnasium_task"], "active_action_dim": 3,
    })
    for mix, expected in (("train", [-1.0, -1 / 3, 1 / 3, 1.0]), ("validation", [-1.0, 1.0])):
        cfg.data_mix = mix
        dataset = get_vla_dataset(cfg, mode="eval")
        samples = [dataset[index] for index in range(len(dataset))]
        np.testing.assert_allclose(sorted(float(sample["state"][0, 0]) for sample in samples),
                                   expected, atol=1e-6)
        assert all(sample["lang"] == "Drive the car up the hill." for sample in samples)



def test_starvla_lerobot_bridge_carrier_pads_action_and_preserves_state(tmp_path):
    from latency_bench.integrations.starvla.lerobot_export import (
        convert_canonical_rollout_to_lerobot,
    )

    image_path = tmp_path / "image.png"
    save_test_image(image_path, 7)
    raw_export = tmp_path / "raw"
    state = [float(index) for index in range(17)]
    _write_generic_export(
        raw_export,
        [
            _canonical_row(image_path, action_id=2, state=state),
            _canonical_row(
                image_path,
                action_id=2,
                decision_step=1,
                state=[value + 1.0 for value in state],
            ),
        ],
        include_state=True,
        state_labels=[f"state_{index}" for index in range(17)],
    )

    output_dir = tmp_path / "lerobot"
    manifest = convert_canonical_rollout_to_lerobot(
        raw_export,
        output_dir,
        force=True,
        action_carrier="bridge",
    )

    row = pq.read_table(
        output_dir / "data/chunk-000/episode_000000.parquet"
    ).to_pylist()[0]
    info = json.loads((output_dir / "meta/info.json").read_text(encoding="utf-8"))

    assert manifest["action_dim"] == 7
    assert manifest["active_action_dim"] == 3
    assert manifest["carrier_action_labels"] == [
        *ACTION_LABELS,
        "BRIDGE_PAD_3",
        "BRIDGE_PAD_4",
        "BRIDGE_PAD_5",
        "BRIDGE_PAD_6",
    ]
    assert row["action"] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert row["observation.state"] == state
    assert info["features"]["action"]["shape"] == [7]
    assert info["features"]["observation.state"] == {
        "dtype": "float32",
        "shape": [17],
        "names": [f"state_{index}" for index in range(17)],
    }


def test_starvla_lerobot_bridge_carrier_rejects_more_than_seven_actions(tmp_path):
    from latency_bench.integrations.starvla.lerobot_export import (
        convert_canonical_rollout_to_lerobot,
    )

    raw_export = tmp_path / "raw"
    raw_export.mkdir()
    action_labels = [f"ACTION_{index}" for index in range(8)]
    (raw_export / "metadata.json").write_text(
        json.dumps(
            {
                "env_name": "eight_action_task",
                "integration_name": "gymnasium",
                "schema_version": 2,
                "action_layout": "gymnasium_discrete_v1",
                "action_labels": action_labels,
                "gymnasium_task": {
                    "task_name": "eight_action_task",
                    "env_id": "EightAction-v0",
                    "make_kwargs": {"render_mode": "rgb_array"},
                    "registration_imports": [],
                    "action_labels": action_labels,
                    "action_values": list(range(8)),
                    "noop_action_id": 0,
                    "base_prompt": "Choose one action.",
                    "env_fps": 30.0,
                    "obs_fps": 10.0,
                    "frame_stack": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="supports at most 7 actions"):
        convert_canonical_rollout_to_lerobot(
            raw_export,
            tmp_path / "lerobot",
            action_carrier="bridge",
        )


def test_starvla_lerobot_preserves_context_images_oldest_to_newest(tmp_path):
    from latency_bench.integrations.starvla.lerobot_export import (
        convert_canonical_rollout_to_lerobot,
    )

    image_path = tmp_path / "current.png"
    context_paths = [tmp_path / f"context_{index}.png" for index in range(3)]
    save_test_image(image_path, 40)
    for context_path, fill in zip(context_paths, [10, 20, 30], strict=True):
        save_test_image(context_path, fill)
    raw_export = tmp_path / "raw"
    _write_generic_export(
        raw_export,
        [_canonical_row(image_path, context_images=context_paths)],
    )

    output_dir = tmp_path / "lerobot"
    convert_canonical_rollout_to_lerobot(raw_export, output_dir, force=True)

    row = pq.read_table(
        output_dir / "data/chunk-000/episode_000000.parquet"
    ).to_pylist()[0]
    info = json.loads((output_dir / "meta/info.json").read_text(encoding="utf-8"))
    modality = json.loads(
        (output_dir / "meta/modality.json").read_text(encoding="utf-8")
    )
    fills = [
        int(np.asarray(Image.open(io.BytesIO(entry["bytes"])).convert("RGB"))[0, 0, 0])
        for entry in row["observation.context_images"]
    ]

    assert fills == [10, 20, 30]
    assert info["features"]["observation.context_images"] == {
        "dtype": "image_sequence",
        "shape": [3, 2, 2, 3],
        "names": ["time", "height", "width", "channel"],
        "video_info": {"video.fps": 10.0},
    }
    assert modality["video"] == {"image": {"original_key": "observation.image"}}


@pytest.fixture
def normalization_run(tmp_path, monkeypatch):
    """Two raw training sources and a distinct validation range for normalization contracts."""
    pytest.importorskip("accelerate")
    monkeypatch.syspath_prepend(str(REPO_ROOT / "third_party/starVLA"))
    from accelerate import Accelerator
    from omegaconf import OmegaConf
    from helpers import write_rollout_fixture
    from latency_bench.data.action_layouts import gymnasium_discrete_action_spec
    from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot

    image = tmp_path / "image.png"
    save_test_image(image, 7)
    for name, values in (("first", [2.0, 4.0]), ("second", [6.0, 8.0])):
        raw = tmp_path / f"raw_{name}"
        train = [_canonical_row(image, decision_step=i, state=[value, 9.0]) for i, value in enumerate(values)]
        val = [{**_canonical_row(image, decision_step=i, state=[value, 9.0]), "episode_idx": 1, "split": "val"}
               for i, value in enumerate([3.0, 10.0])]
        _write_generic_export(raw, train, include_state=True)
        write_rollout_fixture(raw, train, val, action_spec=gymnasium_discrete_action_spec(ACTION_LABELS))
        manifest = convert_canonical_rollout_to_lerobot(raw, tmp_path / name)
    mixtures = tmp_path / "mixtures.json"
    mixtures.write_text(json.dumps({
        "training_mix": [["first", 1.0, "rl_games_gymnasium"], ["second", 1.0, "rl_games_gymnasium"]],
        "validation_mix": [["first__val", 1.0, "rl_games_gymnasium"]],
    }))
    run = tmp_path / "run"
    run.mkdir()
    cfg = OmegaConf.create({
        "output_dir": str(run), "trainer": {"is_resume": False, "pretrained_checkpoint": None},
        "framework": {"action_model": {"state_dim": 2, "action_dim": 3, "action_horizon": 1}},
        "datasets": {"vla_data": {
            "dataset_py": "lerobot_datasets", "data_root_dir": str(tmp_path),
            "data_mix": "training_mix", "eval_data_mix": "validation_mix",
            "custom_mixtures_path": str(mixtures), "gymnasium_task_contract": manifest["gymnasium_task"],
            "active_action_dim": 3, "video_backend": "pyav", "include_state": True,
            "shuffle": False, "per_device_batch_size": 2, "num_workers": 0,
            "pin_memory": False, "persistent_workers": False,
        }},
    })
    return cfg, Accelerator(cpu=True)


def test_training_validation_and_policy_wrappers_share_statistics(normalization_run, tmp_path, monkeypatch):
    """Long-lived integration: each model sees one training-derived transform in every path."""
    import torch
    import yaml
    from omegaconf import OmegaConf
    from starVLA.training.train_starvla import prepare_data
    from starVLA.model.framework.base_framework import baseframework
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from latency_bench.policy.starvla import LiveStarVlaWrapper

    cfg, accelerator = normalization_run
    train, val = prepare_data(cfg, accelerator, Path(cfg.output_dir))
    np.testing.assert_allclose([item["state"][0] for item in next(iter(val))], [[-2 / 3, 0], [5 / 3, 0]], atol=1e-6)
    train_statistics = json.loads((Path(cfg.output_dir) / "dataset_statistics.json").read_text())
    eval_statistics = json.loads((Path(cfg.output_dir) / "dataset_statistics_eval.json").read_text())
    tag = next(iter(train_statistics))
    assert train_statistics[tag]["state"] == eval_statistics[tag]["state"]
    assert train_statistics[tag]["action"] == eval_statistics[tag]["action"]

    class Framework(torch.nn.Module):
        def predict_action(self, examples, **kwargs):
            self.examples = examples
            return {"normalized_actions": np.tile([[[0.0, 1.0, 0.0]]], (len(examples), 1, 1))}

    checkpoint = Path(cfg.output_dir) / "checkpoints/steps_9_pytorch_model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights are supplied by the injected framework")
    model_cfg = OmegaConf.to_container(cfg, resolve=True)
    (Path(cfg.output_dir) / "config.yaml").write_text(yaml.safe_dump(model_cfg))
    disk_framework, live_framework = Framework(), Framework()
    monkeypatch.setattr(baseframework, "from_pretrained", lambda *args, **kwargs: disk_framework)
    disk = PolicyServerWrapper(str(checkpoint), device="cpu", action_output_mode="rl_games",
                               rl_games_env_name="gymnasium", rl_games_action_env_dim=3)
    live = LiveStarVlaWrapper(framework=live_framework, model_cfg=model_cfg, env_name="gymnasium",
                              rl_games_action_env_dim=3, normalization_statistics=train_statistics)
    for raw in ([3.0, 9.0], [10.0, 9.0]):
        examples = [{"state": np.asarray([raw], dtype=np.float32), "image": [], "lang": "Drive."}]
        disk_output = disk.predict_action(examples)
        live_output = live.predict_action(examples)
        expected = [[(raw[0] - 2) / 3 - 1, 0]]
        np.testing.assert_allclose(disk_framework.examples[0]["state"], expected, atol=1e-6)
        np.testing.assert_allclose(live_framework.examples[0]["state"], expected, atol=1e-6)
        np.testing.assert_array_equal(disk_output["actions"], live_output["actions"])


@pytest.mark.parametrize("mode", ["same_run", "new_directory", "new_training"])
def test_resume_freezes_training_statistics(normalization_run, tmp_path, mode):
    """Long-lived resume contract: saved statistics survive changed training data."""
    from starVLA.training.train_starvla import prepare_data

    cfg, accelerator = normalization_run
    from omegaconf import OmegaConf
    original = Path(cfg.output_dir)
    prepare_data(cfg, accelerator, original)
    (original / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    saved = (original / "dataset_statistics.json").read_bytes()
    checkpoint = original / "checkpoints/steps_9_pytorch_model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    for name in ("first", "second"):
        dataset = tmp_path / name
        path = dataset / "data/chunk-000/episode_000000.parquet"
        table = pq.read_table(path)
        values = [[row["observation.state"][0] + 100, 9.0] for row in table.to_pylist()]
        table = table.set_column(table.schema.get_field_index("observation.state"), "observation.state",
                                 pa.array(values, type=table["observation.state"].type))
        pq.write_table(table, path)
        (dataset / "meta/stats_gr00t.json").unlink()
    expected_checkpoint = checkpoint
    if mode == "same_run":
        expected_checkpoint = checkpoint.parent / "steps_9_state"
        expected_checkpoint.mkdir()
        (expected_checkpoint / "model.safetensors").write_bytes(b"full training state")
    target = original if mode == "same_run" else tmp_path / mode
    target.mkdir(exist_ok=True)
    cfg.output_dir = str(target)
    cfg.trainer.is_resume = mode != "new_training"
    cfg.trainer.pretrained_checkpoint = str(checkpoint) if mode != "same_run" else None
    train, val = prepare_data(cfg, accelerator, target)
    actual = (target / "dataset_statistics.json").read_bytes()
    if mode == "new_training":
        payload = json.loads(actual)
        assert next(iter(payload.values()))["state"]["min"] == [102.0, 9.0]
    else:
        assert actual == saved
        assert cfg.trainer.pretrained_checkpoint == str(expected_checkpoint)
        assert cfg.trainer.resume_step == 9
        np.testing.assert_allclose(next(iter(train))[0]["state"], [[(102.0 - 2) / 3 - 1, 0]], atol=1e-5)
        np.testing.assert_allclose(next(iter(val))[0]["state"], [[-2 / 3, 0]], atol=1e-6)
    assert (original / "dataset_statistics.json").read_bytes() == saved


@pytest.mark.parametrize("corruption", ["missing", "wrong_dimension", "changed_transform"])
def test_resume_rejects_unusable_training_statistics(normalization_run, tmp_path, corruption):
    """Long-lived artifact boundary: failed statistics restore cannot overwrite the source."""
    from starVLA.training.train_starvla import prepare_data

    cfg, accelerator = normalization_run
    from omegaconf import OmegaConf
    run = Path(cfg.output_dir)
    prepare_data(cfg, accelerator, run)
    (run / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    checkpoint = run / "checkpoints/steps_9_pytorch_model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    path = run / "dataset_statistics.json"
    if corruption == "missing":
        path.unlink()
        expected_error = FileNotFoundError
    elif corruption == "changed_transform":
        mixtures = Path(cfg.datasets.vla_data.custom_mixtures_path)
        payload = json.loads(mixtures.read_text())
        for entry in payload["training_mix"]:
            entry[2] = "rl_games_flappy"
        mixtures.write_text(json.dumps(payload))
        expected_error = ValueError
    else:
        payload = json.loads(path.read_text())
        next(iter(payload.values()))["state"]["min"] = [[2.0, 9.0], [2.0, 9.0]]
        path.write_text(json.dumps(payload))
        expected_error = ValueError
    before = path.read_bytes() if path.exists() else None
    cfg.trainer.is_resume = True
    with pytest.raises(expected_error):
        prepare_data(cfg, accelerator, run)
    assert (path.read_bytes() if path.exists() else None) == before
