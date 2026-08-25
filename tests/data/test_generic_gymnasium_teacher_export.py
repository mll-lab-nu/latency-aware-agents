from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from helpers import save_test_image


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
    from latency_bench.data.hf_parquet import write_parquet_splits

    action_spec = gymnasium_discrete_action_spec(ACTION_LABELS)
    write_parquet_splits(
        raw_export,
        train_rows,
        [],
        action_spec=action_spec,
        include_state=include_state,
    )
    metadata = {
        "env_name": task_name,
        "integration_name": integration_name,
        "schema_version": 3 if include_state else 2,
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
        metadata["include_state"] = True
    if state_labels is not None:
        metadata["gymnasium_task"]["state_space"] = {"labels": state_labels}
    (raw_export / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def _generic_flat_cfg() -> dict:
    return {
        "algo": "APPO",
        "env": "latency_gymnasium",
        "experiment": "mountain_car_fixed_latency",
        "latency_type": "fixed",
        "gym_task_name": "mountain_car",
        "gym_env_id": "MountainCar-v0",
        "gym_make_kwargs_json": json.dumps({"render_mode": "rgb_array"}),
        "gym_registration_imports_json": json.dumps([]),
        "gym_action_labels_json": json.dumps(["PUSH_LEFT", "COAST", "PUSH_RIGHT"]),
        "gym_action_values_json": json.dumps([0, 1, 2]),
        "gym_noop_action_id": 1,
        "gym_state_labels_json": "",
        "gym_base_prompt": "Drive the car up the hill.",
        "env_fps": 30.0,
        "obs_fps": 10.0,
        "frame_stack": 1,
        "hold_policy": "repeat_last",
        "ordering_policy": "fifo",
    }


def test_generic_checkpoint_builds_dynamic_task_contract_without_global_registration(
    tmp_path,
):
    from latency_bench.data.action_layouts import ACTION_SPECS_BY_ENV
    from latency_bench.data.prompts import BASE_PROMPTS
    from latency_bench.data.sf_teacher_config import load_rollout_training_config

    original_action_spec_keys = tuple(ACTION_SPECS_BY_ENV)
    original_prompt_keys = tuple(BASE_PROMPTS)
    experiment_dir = tmp_path / "mountain_car_fixed_latency"
    experiment_dir.mkdir()
    (experiment_dir / "config.json").write_text(
        json.dumps(_generic_flat_cfg()), encoding="utf-8"
    )

    descriptor, metadata = load_rollout_training_config(experiment_dir)

    assert descriptor["env_name"] == "mountain_car"
    assert descriptor["integration_name"] == "gymnasium"
    assert descriptor["action_spec"].layout == "gymnasium_discrete_v1"
    assert descriptor["action_spec"].labels == ("PUSH_LEFT", "COAST", "PUSH_RIGHT")
    assert descriptor["base_prompt"] == "Drive the car up the hill."
    assert metadata["gymnasium_task"] == {
        "task_name": "mountain_car",
        "env_id": "MountainCar-v0",
        "make_kwargs": {"render_mode": "rgb_array"},
        "registration_imports": [],
        "action_labels": ["PUSH_LEFT", "COAST", "PUSH_RIGHT"],
        "action_values": [0, 1, 2],
        "noop_action_id": 1,
        "base_prompt": "Drive the car up the hill.",
        "env_fps": 30.0,
        "obs_fps": 10.0,
        "frame_stack": 1,
    }
    assert tuple(ACTION_SPECS_BY_ENV) == original_action_spec_keys
    assert tuple(BASE_PROMPTS) == original_prompt_keys


def test_generic_teacher_row_uses_explicit_action_spec_and_prompt():
    from latency_bench.data.action_layouts import gymnasium_discrete_action_spec
    from latency_bench.data.sf_teacher_rows import teacher_dataset_row

    action_spec = gymnasium_discrete_action_spec(["PUSH_LEFT", "COAST", "PUSH_RIGHT"])
    row = teacher_dataset_row(
        env_name="mountain_car",
        spec={"episode_idx": 2, "split": "train", "seed": 19},
        decision_step=4,
        image="frame.png",
        raw_action=2,
        state=np.asarray([1.0, 2.0], dtype=np.float64),
        info={"latency_raw_frames": 3, "latency_ms": 100.0},
        reward=-1.0,
        env_fps=30.0,
        obs_fps=10.0,
        action_spec=action_spec,
        base_prompt="Drive the car up the hill.",
    )

    assert row["env_name"] == "mountain_car"
    assert row["action_id"] == 2
    assert row["action"] == [0.0, 0.0, 1.0]
    assert row["action_text"] == "PUSH_RIGHT"
    assert row["state"] == [1.0, 2.0]
    assert row["prompt"].startswith("Drive the car up the hill.")
    assert "Current action latency is 3 raw frames (100.00 ms)." in row["prompt"]


def test_gymnasium_replay_exports_pre_action_native_state_with_stacked_observations(
    monkeypatch,
    tmp_path,
):
    import torch

    from latency_bench.data import sf_teacher_rollout
    from latency_bench.data.action_layouts import gymnasium_discrete_action_spec
    from latency_bench.data.hf_parquet import iter_episode_rows
    from latency_bench.data.sf_teacher_drivers import RolloutStepResult

    policy_observations = []
    step_index = 0

    class Driver:
        slot_count = 1

        def reset_slots(self, reset_plan):
            assert reset_plan == {0: 7}
            return {
                0: (
                    {"obs": np.asarray([[0.0, 0.0]], dtype=np.float32)},
                    {
                        "gymnasium_state": np.asarray([0.0], dtype=np.float32),
                        "env_raw_rgb_frame_stack": np.zeros(
                            (2, 2, 2, 3), dtype=np.uint8
                        ),
                    },
                )
            }

        def close(self):
            pass

    def step_teacher_policy(
        runtime,
        driver,
        *,
        active_slots,
        active_obs,
        rnn_states,
        deterministic,
        preserve_leading_batch,
    ):
        nonlocal step_index
        policy_observations.append(active_obs[0]["obs"].copy())
        action_id = (2, 0)[step_index]
        next_state = float(step_index + 1)
        result = RolloutStepResult(
            next_obs={
                "obs": np.asarray(
                    [[next_state - 1.0, next_state]], dtype=np.float32
                )
            },
            reward=0.0,
            terminated=step_index == 1,
            truncated=False,
            info={
                "gymnasium_state": np.asarray([next_state], dtype=np.float32),
                "env_raw_rgb_frame_stack": np.full(
                    (2, 2, 2, 3), step_index + 1, dtype=np.uint8
                ),
                "latency_raw_frames": 0,
                "latency_ms": 0.0,
            },
        )
        step_index += 1
        return [action_id], torch.zeros_like(rnn_states), {0: result}

    monkeypatch.setattr(
        sf_teacher_rollout,
        "_step_teacher_policy",
        step_teacher_policy,
    )
    output_dir = tmp_path / "rollout"
    sf_teacher_rollout._replay_accepted_specs(
        {"rnn_size": 1, "device": torch.device("cpu")},
        Driver(),
        cfg=SimpleNamespace(env_fps=30.0, obs_fps=30.0, frame_stack=2),
        env_name="mountain_car",
        output_dir=output_dir,
        accepted_specs=[{"episode_idx": 0, "split": "val", "seed": 7}],
        completed_episode_indices=set(),
        max_steps_per_episode=3,
        deterministic=True,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
        action_spec=gymnasium_discrete_action_spec(ACTION_LABELS),
        base_prompt="Drive the car up the hill.",
        integration_name="gymnasium",
    )

    rows = list(iter_episode_rows(output_dir, "val"))
    assert [(row["state"], row["action_id"]) for row in rows] == [
        ([0.0], 2),
        ([1.0], 0),
    ]
    assert [observation.tolist() for observation in policy_observations] == [
        [[0.0, 0.0]],
        [[0.0, 1.0]],
    ]


def test_generic_vector_observations_keep_sample_factory_batch_dimension():
    from latency_bench.data.sf_teacher_drivers import batch_observations

    observations = [
        {"obs": np.asarray([[slot, slot + 1]], dtype=np.float32)} for slot in (1, 3)
    ]

    batch = batch_observations(observations, preserve_leading_batch=True)

    assert batch["obs"].shape == (2, 2)
    assert batch["obs"].tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_generic_rollout_runtime_selects_gymnasium_integration_and_replays_checkpoint_args(
    monkeypatch,
    tmp_path,
):
    from sample_factory.cfg import arguments as sf_arguments

    from latency_bench.data import sf_teacher_runtime

    calls: list[object] = []
    parsed_cfg = SimpleNamespace(policy_index=0)
    module = SimpleNamespace(
        register_gymnasium_envs=lambda: calls.append("registered"),
        SAMPLE_FACTORY_CONFIG_PARSER=SimpleNamespace(
            parse_eval=lambda argv: calls.append(argv) or parsed_cfg
        ),
    )
    monkeypatch.setattr(
        sf_teacher_runtime,
        "import_integration_module",
        lambda config: calls.append(config) or module,
    )
    monkeypatch.setattr(sf_arguments, "load_from_checkpoint", lambda cfg: cfg)
    flat_cfg = _generic_flat_cfg()

    result_module, result_cfg = sf_teacher_runtime.build_sample_factory_rollout_cfg(
        {
            "env_name": "mountain_car",
            "integration_name": "gymnasium",
            "checkpoint_experiment_dir": str(tmp_path / "teacher"),
            "checkpoint_path": None,
            "flat_cfg": flat_cfg,
            "device_override": None,
            "latency_override": None,
        }
    )

    argv = calls[2]
    assert result_module is module
    assert result_cfg is parsed_cfg
    assert calls[0] == {"env": {"name": "gymnasium"}}
    assert calls[1] == "registered"
    assert argv[argv.index("--gym-task-name") + 1] == "mountain_car"
    assert (
        argv[argv.index("--gym-action-labels-json") + 1]
        == flat_cfg["gym_action_labels_json"]
    )
    assert (
        argv[argv.index("--gym-action-values-json") + 1]
        == flat_cfg["gym_action_values_json"]
    )
    assert argv[argv.index("--gym-noop-action-id") + 1] == "1"
    assert json.loads(argv[argv.index("--gym-action-space-json") + 1]) == {
        "type": "discrete",
        "labels": ["PUSH_LEFT", "COAST", "PUSH_RIGHT"],
        "values": [0, 1, 2],
    }
    assert json.loads(argv[argv.index("--gym-noop-action-json") + 1]) == "COAST"


def test_generic_rollout_runtime_forwards_only_continuous_action_contract(monkeypatch, tmp_path):
    from sample_factory.cfg import arguments as sf_arguments
    from latency_bench.data import sf_teacher_runtime

    calls: list[object] = []
    parsed_cfg = SimpleNamespace(policy_index=0)
    module = SimpleNamespace(
        register_gymnasium_envs=lambda: calls.append("registered"),
        SAMPLE_FACTORY_CONFIG_PARSER=SimpleNamespace(
            parse_eval=lambda argv: calls.append(argv) or parsed_cfg
        ),
    )
    monkeypatch.setattr(sf_teacher_runtime, "import_integration_module", lambda config: module)
    monkeypatch.setattr(sf_arguments, "load_from_checkpoint", lambda cfg: cfg)
    flat_cfg = _generic_flat_cfg()
    flat_cfg["gym_action_space_json"] = json.dumps(
        {
            "type": "box",
            "labels": ["a0", "a1"],
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
            "dtype": "float32",
        }
    )
    flat_cfg["gym_noop_action_json"] = json.dumps([0.0, 0.0])
    del flat_cfg["gym_action_labels_json"]
    del flat_cfg["gym_action_values_json"]
    del flat_cfg["gym_noop_action_id"]

    sf_teacher_runtime.build_sample_factory_rollout_cfg(
        {
            "env_name": "continuous_task",
            "integration_name": "gymnasium",
            "checkpoint_experiment_dir": str(tmp_path / "teacher"),
            "checkpoint_path": None,
            "flat_cfg": flat_cfg,
            "device_override": None,
            "latency_override": None,
        }
    )

    argv = calls[1]
    assert json.loads(argv[argv.index("--gym-action-space-json") + 1])["type"] == "box"
    assert json.loads(argv[argv.index("--gym-noop-action-json") + 1]) == [0.0, 0.0]
    assert "--gym-action-labels-json" not in argv
    assert "--gym-action-values-json" not in argv
    assert "--gym-noop-action-id" not in argv


def test_teacher_export_cli_leaves_generic_task_name_to_checkpoint_inference():
    from scripts.rollout_data.export_sf_teacher_dataset import build_parser

    args = build_parser().parse_args(
        [
            "--checkpoint-root",
            "checkpoint",
            "--output-dir",
            "out",
            "--max-episodes",
            "1",
            "--max-steps-per-episode",
            "10",
        ]
    )

    assert args.env_name == ""


def test_teacher_export_cli_accepts_strict_episode_return_gate():
    from scripts.rollout_data.export_sf_teacher_dataset import build_parser

    args = build_parser().parse_args(
        [
            "--checkpoint-root",
            "checkpoint",
            "--output-dir",
            "out",
            "--max-episodes",
            "100",
            "--max-steps-per-episode",
            "1000",
            "--episode-return-gt",
            "3000",
        ]
    )

    assert args.episode_return_gt == pytest.approx(3000.0)


@pytest.mark.parametrize(
    ("episode_return", "accepted"),
    [(2999.999, False), (3000.0, False), (3000.001, True)],
)
def test_generic_episode_return_gate_is_strictly_greater(episode_return, accepted):
    from latency_bench.data.sf_teacher_rollout import _passes_episode_return_gate

    assert (
        _passes_episode_return_gate(
            {"episode_raw_return": episode_return},
            episode_return_field="episode_raw_return",
            episode_return_gt=3000.0,
        )
        is accepted
    )


def test_generic_teacher_export_rejects_task_identity_override(monkeypatch, tmp_path):
    from latency_bench.data.action_layouts import gymnasium_discrete_action_spec
    from latency_bench.data import sf_teacher_export

    monkeypatch.setattr(
        sf_teacher_export,
        "load_rollout_training_config",
        lambda *args, **kwargs: (
            {
                "env_name": "mountain_car",
                "integration_name": "gymnasium",
                "action_spec": gymnasium_discrete_action_spec(ACTION_LABELS),
                "base_prompt": "Drive the car up the hill.",
            },
            {},
        ),
    )

    with pytest.raises(ValueError, match="fixed by the teacher checkpoint"):
        sf_teacher_export.export_sample_factory_teacher_dataset(
            checkpoint_root=tmp_path / "checkpoint",
            output_dir=tmp_path / "output",
            latency_config_path=None,
            env_name="renamed_task",
            max_episodes=1,
            max_steps_per_episode=10,
            val_fraction=0.1,
            split_seed=0,
            num_envs=1,
            replay_num_envs=1,
            device="",
            deterministic=True,
            export_image_size=None,
            image_writer_workers=1,
            row_buffer_size=1,
            push_to_hub=False,
            hf_repo_id="",
            hf_private=False,
            hf_token="",
            hf_max_shard_size="500MB",
        )


def test_starvla_lerobot_converter_writes_native_handoff_contract(tmp_path):
    from latency_bench.data.exporters.starvla_lerobot import (
        convert_canonical_rollout_to_lerobot,
    )

    image_path = tmp_path / "image.png"
    save_test_image(image_path, 7)
    raw_export = tmp_path / "raw"
    _write_generic_export(raw_export, [_canonical_row(image_path)])

    output_dir = tmp_path / "lerobot"
    manifest = convert_canonical_rollout_to_lerobot(raw_export, output_dir, force=True)

    table = pq.read_table(output_dir / "data/chunk-000/episode_000000.parquet")
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
    assert manifest["robot_type"] == "rl_games_gymnasium_native"
    assert manifest["integration_name"] == "gymnasium"
    assert manifest["task_name"] == "mountain_car"
    assert manifest["gymnasium_task"]["action_values"] == [0, 1, 2]
    assert manifest["gymnasium_task"]["noop_action_id"] == 1
    assert validation_manifest["integration_name"] == "gymnasium"
    assert validation_manifest["task_name"] == "mountain_car"
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
            "native": {
                "start": 0,
                "end": 1,
                "dtype": "float32",
                "absolute": True,
                "original_key": "observation.state",
            }
        },
        "action": {
            "native": {
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
        "lerobot": [["lerobot", 1.0, "rl_games_gymnasium_native"]],
        "lerobot__val": [["lerobot__val", 1.0, "rl_games_gymnasium_native"]],
    }

def test_starvla_lerobot_converter_uses_explicit_gymnasium_state(tmp_path):
    from latency_bench.data.exporters.starvla_lerobot import (
        convert_canonical_rollout_to_lerobot,
    )

    image_path = tmp_path / "image.png"
    save_test_image(image_path, 7)
    raw_export = tmp_path / "raw"
    rows = [
        _canonical_row(image_path, decision_step=0, state=[-2.0, 1.0]),
        _canonical_row(image_path, decision_step=1, state=[2.0, 3.0]),
    ]
    _write_generic_export(
        raw_export,
        rows,
        task_name="inverted_pendulum_rgb_state",
        include_state=True,
        state_labels=["cart_position", "pole_angle"],
    )

    output_dir = tmp_path / "lerobot"
    manifest = convert_canonical_rollout_to_lerobot(raw_export, output_dir, force=True)
    converted = pq.read_table(
        output_dir / "data/chunk-000/episode_000000.parquet"
    ).to_pylist()

    assert [row["observation.state"] for row in converted] == [
        [-2.0, 1.0],
        [2.0, 3.0],
    ]
    assert manifest["include_state"] is True
    assert manifest["state_dim"] == 2
    assert "state_labels" not in manifest
    info = json.loads((output_dir / "meta/info.json").read_text(encoding="utf-8"))
    assert info["features"]["observation.state"]["names"] == [
        "cart_position",
        "pole_angle",
    ]


def test_starvla_lerobot_bridge_carrier_pads_action_and_preserves_native_state(tmp_path):
    from latency_bench.data.exporters.starvla_lerobot import (
        convert_canonical_rollout_to_lerobot,
    )

    image_path = tmp_path / "image.png"
    save_test_image(image_path, 7)
    raw_export = tmp_path / "raw"
    state = [float(index) for index in range(17)]
    _write_generic_export(
        raw_export,
        [_canonical_row(image_path, action_id=2, state=state)],
        include_state=True,
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
    from latency_bench.data.exporters.starvla_lerobot import (
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
    from latency_bench.data.exporters.starvla_lerobot import (
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


def test_generic_task_name_matching_legacy_env_keeps_generic_handoff_semantics(
    tmp_path,
):
    from latency_bench.data.exporters.starvla_lerobot import (
        convert_canonical_rollout_to_lerobot,
    )

    image_path = tmp_path / "image.png"
    save_test_image(image_path, 7)
    rows = [
        _canonical_row(image_path, task_name="flappy", decision_step=decision_step)
        for decision_step in range(2)
    ]
    raw_export = tmp_path / "raw"
    _write_generic_export(raw_export, rows, task_name="flappy")

    output_dir = tmp_path / "lerobot"
    manifest = convert_canonical_rollout_to_lerobot(raw_export, output_dir, force=True)

    table = pq.read_table(output_dir / "data/chunk-000/episode_000000.parquet")
    mixtures = json.loads(
        (tmp_path / "_generated_mixtures/lerobot.json").read_text(encoding="utf-8")
    )
    assert manifest["robot_type"] == "rl_games_gymnasium_native"
    assert table["density_weight"].to_pylist() == [1.0, 1.0]
    assert mixtures["lerobot"] == [["lerobot", 1.0, "rl_games_gymnasium_native"]]
