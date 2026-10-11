from __future__ import annotations

import json
from types import SimpleNamespace
import numpy as np
import pytest


ACTION_LABELS = ["PUSH_LEFT", "COAST", "PUSH_RIGHT"]


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
        "gym_action_space_json": json.dumps(
            {
                "type": "discrete",
                "labels": ["PUSH_LEFT", "COAST", "PUSH_RIGHT"],
                "values": [0, 1, 2],
            }
        ),
        "gym_action_labels_json": json.dumps(["PUSH_LEFT", "COAST", "PUSH_RIGHT"]),
        "gym_action_values_json": json.dumps([0, 1, 2]),
        "gym_noop_action_id": 1,
        "gym_noop_action_json": json.dumps("COAST"),
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
    from latency_bench.integrations.sample_factory.teacher_config import load_rollout_training_config

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


def test_generic_checkpoint_applies_rollout_overrides(tmp_path):
    from latency_bench.integrations.sample_factory.teacher_config import load_rollout_training_config

    experiment_dir = tmp_path / "mountain_car_fixed_latency"
    experiment_dir.mkdir()
    flat_cfg = _generic_flat_cfg()
    (experiment_dir / "config.json").write_text(
        json.dumps(flat_cfg), encoding="utf-8"
    )

    descriptor, metadata = load_rollout_training_config(
        experiment_dir,
        gym_camera_name="head_camera",
        gym_make_kwargs_override={"noop_max": 30, "fire_reset": True},
    )

    make_kwargs = json.loads(descriptor["flat_cfg"]["gym_make_kwargs_json"])
    assert make_kwargs["camera_name"] == "head_camera"
    assert make_kwargs["render_mode"] == "rgb_array"
    assert make_kwargs["noop_max"] == 30
    assert make_kwargs["fire_reset"] is True
    assert metadata["gymnasium_task"]["make_kwargs"]["camera_name"] == "head_camera"
    assert metadata["gymnasium_task"]["make_kwargs"]["render_mode"] == "rgb_array"
    assert metadata["gymnasium_task"]["make_kwargs"]["noop_max"] == 30
    assert metadata["gymnasium_task"]["make_kwargs"]["fire_reset"] is True


def test_gymnasium_replay_exports_pre_action_state_with_stacked_observations(
    monkeypatch,
    tmp_path,
):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    import torch

    from latency_bench.data.action_layouts import gymnasium_discrete_action_spec
    from latency_bench.data.hf_parquet import iter_episode_rows
    from latency_bench.integrations.sample_factory.teacher_drivers import RolloutStepResult

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
        teacher_runtime, "step_teacher_policy",
        step_teacher_policy,
    )
    output_dir = tmp_path / "rollout"
    _replay_existing_specs(monkeypatch,
        {"rnn_size": 1, "device": torch.device("cpu")},
        Driver(),
        cfg=SimpleNamespace(
            env_fps=30.0,
            obs_fps=30.0,
            frame_stack=2,
            gym_state_labels_json='["state"]',
        ),
        env_name="mountain_car",
        output_dir=output_dir,
        accepted_specs=[{"episode_idx": 0, "split": "val", "seed": 7, "episode_length": 2}],
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


def test_g1_balance_replay_exports_generic_state_and_absolute_action_contract(
    monkeypatch,
    tmp_path,
):
    from latency_bench.integrations.sample_factory import teacher_runtime as teacher_runtime
    import torch

    g1_task = "g1_sonic_balance_simple"

    from latency_bench.data.action_layouts import G1_SONIC_ACTION_SPEC
    from latency_bench.data.hf_parquet import iter_episode_rows
    from latency_bench.integrations.sample_factory.teacher_drivers import RolloutStepResult
    action_dim = len(G1_SONIC_ACTION_SPEC.labels)
    state_dim = 198

    step_index = 0

    class Driver:
        slot_count = 1
        control_env = SimpleNamespace(unwrapped=SimpleNamespace(
            _control_mode="raw_body_target",
            g1_absolute_request_from_raw_action=lambda action: 10.0 + 2.0 * action,
        ))

        def reset_slots(self, reset_plan):
            return {
                0: (
                    {"obs": np.zeros((1, 2), dtype=np.float32)},
                    {
                        "gymnasium_state": np.full(
                            state_dim, 10.0, dtype=np.float32
                        ),
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
        next_state = float(step_index + 11)
        action_target = np.full(
            action_dim, step_index + 0.25, dtype=np.float32
        )
        result = RolloutStepResult(
            next_obs={"obs": np.zeros((1, 2), dtype=np.float32)},
            reward=0.0,
            terminated=step_index == 1,
            truncated=False,
            info={
                "gymnasium_state": np.full(
                    state_dim, next_state, dtype=np.float32
                ),
                "env_raw_rgb_frame_stack": np.zeros((2, 2, 2, 3), dtype=np.uint8),
                "latency_raw_frames": 0,
                "latency_ms": 0.0,
            },
        )
        teacher_action = np.full(
            action_dim, -0.5 + 0.25 * step_index, dtype=np.float32
        )
        if g1_task == "g1_sonic_balance_simple":
            result.info.update(
                command_admitted=True,
                observation_dropped=False,
                applied_action=action_target,
                applied_source_raw_frame=step_index,
                applied_source_obs_id=step_index,
                raw_done=False,
            )
        step_index += 1
        return [teacher_action], torch.zeros_like(rnn_states), {0: result}

    monkeypatch.setattr(
        teacher_runtime, "step_teacher_policy",
        step_teacher_policy,
    )
    output_dir = tmp_path / "rollout"
    _replay_existing_specs(monkeypatch,
        {"rnn_size": 1, "device": torch.device("cpu")},
        Driver(),
        cfg=SimpleNamespace(
            env_fps=50.0,
            obs_fps=50.0,
            frame_stack=1,
            gym_state_labels_json=json.dumps([f"state_{index}" for index in range(state_dim)]),
        ),
        env_name=g1_task,
        output_dir=output_dir,
        accepted_specs=[{"episode_idx": 0, "split": "val", "seed": 7, "episode_length": 2}],
        max_steps_per_episode=3,
        deterministic=True,
        export_image_size=None,
        image_writer_workers=0,
        row_buffer_size=1,
        action_spec=G1_SONIC_ACTION_SPEC,
        base_prompt="Keep balance.",
        integration_name="gymnasium",
    )

    rows = list(iter_episode_rows(output_dir, "val"))
    assert [row["state"] for row in rows] == [
        [10.0] * state_dim,
        [11.0] * state_dim,
    ]
    expected_actions = [9.0, 9.5]
    assert [row["action"] for row in rows] == [
        [value] * action_dim for value in expected_actions
    ]
    assert [row["raw_done"] for row in rows] == [False, True]
    assert [row["applied_action"] for row in rows] == [
        [0.25] * action_dim,
        [1.25] * action_dim,
    ]
    assert [row["prompt"] for row in rows] == ["Keep balance."] * 2


def test_g1_checkpoint_metadata_uses_absolute_action_labels():
    from latency_bench.data.action_layouts import G1_SONIC_ACTION_SPEC
    from latency_bench.integrations.sample_factory.teacher_config import _gymnasium_task_metadata
    from latency_bench.envs.tasks.humanoidbench.contract import (
        G1_BODY_JOINT_NAMES,
        G1_HAND_JOINT_NAMES,
        MUJOCO_INDEX_FOR_POLICY,
    )

    flat_cfg = _generic_flat_cfg()
    flat_cfg["gym_task_name"] = "g1_sonic_balance_simple"
    action_dim = len(G1_SONIC_ACTION_SPEC.labels)
    flat_cfg["gym_action_space_json"] = json.dumps(
        {
            "type": "box",
            "labels": [
                f"residual_{index}" for index in range(action_dim)
            ],
            "low": [-1.0] * action_dim,
            "high": [1.0] * action_dim,
            "dtype": "float32",
        }
    )
    flat_cfg["gym_noop_action_json"] = json.dumps([0.0] * action_dim)

    metadata = _gymnasium_task_metadata(flat_cfg)

    assert G1_SONIC_ACTION_SPEC.labels == tuple(
        G1_BODY_JOINT_NAMES[index] for index in MUJOCO_INDEX_FOR_POLICY
    ) + G1_HAND_JOINT_NAMES
    assert metadata["action_space"] == {
        "type": "box",
        "labels": list(G1_SONIC_ACTION_SPEC.labels),
        "dtype": "float32",
    }


def test_generic_vector_observations_keep_sample_factory_batch_dimension():
    from latency_bench.integrations.sample_factory.teacher_drivers import batch_observations

    observations = [
        {"obs": np.asarray([[slot, slot + 1]], dtype=np.float32)} for slot in (1, 3)
    ]

    batch = batch_observations(observations, preserve_leading_batch=True)

    assert batch["obs"].shape == (2, 2)
    assert batch["obs"].tolist() == [[1.0, 2.0], [3.0, 4.0]]


@pytest.mark.parametrize("action_type", ["discrete", "box"])
def test_generic_rollout_runtime_replays_checkpoint_action_contract(monkeypatch, tmp_path, action_type):
    from sample_factory.cfg import arguments as sf_arguments
    from latency_bench.integrations.sample_factory import teacher_runtime

    registered = []
    parser_args = []
    integration_configs = []
    parsed_cfg = SimpleNamespace(policy_index=0)
    module = SimpleNamespace(
        register_gymnasium_envs=lambda: registered.append(True),
        SAMPLE_FACTORY_CONFIG_PARSER=SimpleNamespace(
            parse_eval=lambda argv: parser_args.extend(argv) or parsed_cfg
        ),
    )
    monkeypatch.setattr(
        teacher_runtime, "import_module",
        lambda module_name: integration_configs.append(module_name) or module,
    )
    monkeypatch.setattr(sf_arguments, "load_from_checkpoint", lambda cfg: cfg)
    flat_cfg = _generic_flat_cfg()
    if action_type == "box":
        flat_cfg["gym_action_space_json"] = json.dumps({
            "type": "box", "labels": ["a0", "a1"], "low": [-1.0, -1.0],
            "high": [1.0, 1.0], "dtype": "float32",
        })
        flat_cfg["gym_noop_action_json"] = json.dumps([0.0, 0.0])
        for key in ("gym_action_labels_json", "gym_action_values_json", "gym_noop_action_id"):
            del flat_cfg[key]

    result_module, result_cfg = teacher_runtime.build_sample_factory_rollout_cfg({
        "env_name": "mountain_car", "integration_name": "gymnasium",
        "checkpoint_experiment_dir": str(tmp_path / "teacher"), "checkpoint_path": None,
        "flat_cfg": flat_cfg, "device_override": None, "latency_override": None,
    })

    assert result_module is module
    assert result_cfg is parsed_cfg
    assert registered
    assert integration_configs == ["training.gymnasium_sf.integration"]
    assert parser_args[parser_args.index("--gym-task-name") + 1] == "mountain_car"
    assert json.loads(parser_args[parser_args.index("--gym-action-space-json") + 1]) == json.loads(flat_cfg["gym_action_space_json"])
    assert json.loads(parser_args[parser_args.index("--gym-noop-action-json") + 1]) == json.loads(flat_cfg["gym_noop_action_json"])
    if action_type == "discrete":
        assert json.loads(parser_args[parser_args.index("--gym-action-labels-json") + 1]) == ACTION_LABELS
        assert json.loads(parser_args[parser_args.index("--gym-action-values-json") + 1]) == [0, 1, 2]
        assert parser_args[parser_args.index("--gym-noop-action-id") + 1] == "1"


def test_teacher_export_cli_infers_task_and_accepts_return_gate():
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

    assert args.env_name == ""
    assert args.episode_return_gt == pytest.approx(3000.0)


def test_generic_teacher_export_rejects_task_identity_override(monkeypatch, tmp_path):
    from latency_bench.data.action_layouts import gymnasium_discrete_action_spec
    from latency_bench.integrations.sample_factory import teacher_export as sf_teacher_export

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


def _replay_existing_specs(monkeypatch, runtime, driver, *, cfg, env_name, output_dir,
                           accepted_specs, action_spec, base_prompt,
                           integration_name, **options):
    """Exercise accepted replay through the public SF-to-common collection adapter."""
    from latency_bench.data import rollout_probe
    from latency_bench.integrations.sample_factory import teacher_runtime, teacher_rollout
    cfg.seed = 7
    runtime["checkpoint"] = {"train_step": 0}
    monkeypatch.setattr(teacher_runtime, "build_rollout_driver_runtime",
                        lambda *a, **kw: (cfg, runtime, driver, cfg.frame_stack))
    state = rollout_probe.new_probe_state()
    state["accepted_specs"] = accepted_specs
    rollout_probe.write_probe_state(output_dir, state)
    return teacher_rollout.collect_teacher_dataset(
        {"env_name": env_name, "integration_name": integration_name,
         "action_spec": action_spec, "base_prompt": base_prompt, "flat_cfg": {}},
        output_dir=output_dir, max_episodes=len(accepted_specs), val_fraction=1.0,
        split_seed=0, num_envs=1, filter_preset="none", **options,
    )


def test_box_teacher_command_is_not_replaced_by_delayed_applied_action():
    """Long-lived SF boundary: return the issued native command for dataset labels."""
    import torch
    from gymnasium.spaces import Box
    from latency_bench.integrations.sample_factory.teacher_runtime import step_teacher_policy
    from latency_bench.integrations.sample_factory.teacher_drivers import RolloutStepResult

    class Actor:
        def __call__(self, obs, states):
            return {"actions": torch.tensor([[0.75, -0.25]]), "new_rnn_states": states}

    class Driver:
        action_space = Box(-1, 1, (2,), dtype=np.float32)

        def step(self, slots, actions):
            assert actions == [[0.75, -0.25]]
            # The native environment can reuse its input buffer after accepting it.
            actions[0][0] = 0.0
            return {0: RolloutStepResult(next_obs=None, reward=0.0, terminated=True, truncated=False,
                                         info={"applied_action": [-0.5, 0.5]})}

    issued, _, result = step_teacher_policy(
        {"actor_critic": Actor(), "device": torch.device("cpu"), "env_info": object(),
         "prepare_and_normalize_obs": lambda actor, obs: obs,
         "argmax_actions": None, "preprocess_actions": lambda env_info, actions: actions.numpy(),
         "unsqueeze_tensor": lambda value, dim: value.unsqueeze(dim)},
        Driver(), active_slots=[0], active_obs=[{"obs": np.zeros(2)}],
        rnn_states=torch.zeros((1, 1)), deterministic=False,
    )
    assert issued == [[0.75, -0.25]]
    assert result[0].info["applied_action"] == [-0.5, 0.5]
