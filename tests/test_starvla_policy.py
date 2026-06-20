from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from latency_bench.core.actions import ActionResolver
from latency_bench.core.config import load_config
from latency_bench.core.types import Observation
from latency_bench.policy.factory import build_policy
from latency_bench.data.prompts import BASE_PROMPTS

REPO_ROOT = Path(__file__).resolve().parents[1]
FLAPPY_REALTIME_STARVLA_CONFIGS = (
    "configs/experiments/flappy/realtime/starvla_openvla_fix_latency_2_15fps_l2_2gpu_warmup200_3ep.yaml",
    "configs/experiments/flappy/realtime/starvla_openvla_fix_latency_2_16fps_l2_2gpu_warmup200_3ep.yaml",
    "configs/experiments/flappy/realtime/starvla_openvla_fix_latency_2_8fps_l1_2gpu_warmup200_3ep.yaml",
)
FLAPPY_SIMULATED_STARVLA_CONFIGS = (
    "configs/experiments/flappy/simulated/starvla_openvla_fix_latency_2_15fps_l2_warmup200_3ep.yaml",
    "configs/experiments/flappy/simulated/starvla_openvla_fix_latency_2_16fps_l2_warmup200_3ep.yaml",
    "configs/experiments/flappy/simulated/starvla_openvla_fix_latency_2_8fps_l1_warmup200_3ep.yaml",
)


class FakePolicyServerWrapper:
    instances: list["FakePolicyServerWrapper"] = []

    def __init__(
        self,
        ckpt_path,
        device="cuda",
        use_bf16=False,
        unnorm_key=None,
        action_output_mode=None,
        rl_games_env_name=None,
    ):
        self.ckpt_path = ckpt_path
        self.device = device
        self.use_bf16 = use_bf16
        self.unnorm_key = unnorm_key
        self.action_output_mode = action_output_mode
        self.rl_games_env_name = rl_games_env_name
        self.calls = []
        self._model_cfg = {
            "datasets": {
                "vla_data": {
                    "include_state": True,
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 7,
                },
            },
        }
        FakePolicyServerWrapper.instances.append(self)

    def predict_action(self, examples, unnorm_key=None, **kwargs):
        self.calls.append(
            {
                "examples": examples,
                "unnorm_key": unnorm_key,
                "kwargs": kwargs,
            }
        )
        return {
            "actions": np.asarray(
                [
                    [[1]],
                    [[0]],
                ][: len(examples)],
                dtype=np.int64,
            ),
            "raw_action_scores": np.asarray(
                [
                    [[0.1, 0.9]],
                    [[0.8, 0.2]],
                ][: len(examples)],
                dtype=np.float32,
            ),
            "action_output_type": "rl_games_discrete_id",
        }


def _config(
    tmp_path: Path,
    *,
    latency_prompt_map_path: str | None = None,
) -> dict:
    policy = {
        "type": "starvla",
        "checkpoint_path": str(tmp_path / "checkpoint"),
        "device": "cuda:0",
        "unnorm_key": "flappy",
        "actions": ["noop", "flap"],
    }
    if latency_prompt_map_path is not None:
        policy["latency_prompt_map_path"] = latency_prompt_map_path
    return {
        "env": {
            "name": "flappy",
            "noop_action": "noop",
            "action_map": {"noop": 0, "flap": 1},
            "oneshot_actions": ["flap"],
        },
        "policy": policy,
    }


def test_starvla_passes_all_raw_frames_and_base_prompt(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: FakePolicyServerWrapper)
    config = _config(tmp_path)
    policy = build_policy(config, ActionResolver(config["env"]))
    frame_stack = np.zeros((2, 3, 6), dtype=np.uint8)
    frame_stack[:, :, :3] = 11
    frame_stack[:, :, 3:] = 22

    output = policy.predict(
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
            metadata={"env_raw_rgb_frame_stack": frame_stack},
        )
    )

    wrapper = FakePolicyServerWrapper.instances[0]
    assert wrapper.use_bf16 is True
    assert wrapper.unnorm_key == "flappy"
    assert wrapper.action_output_mode == "rl_games"
    assert wrapper.rl_games_env_name == "flappy"
    assert wrapper.calls[0]["unnorm_key"] == "flappy"
    example = wrapper.calls[0]["examples"][0]
    assert example["lang"] == BASE_PROMPTS["flappy"]
    assert len(example["image"]) == 2
    assert np.asarray(example["image"][0]).mean() == 11
    assert np.asarray(example["image"][1]).mean() == 22
    assert example["state"].shape == (1, 7)
    assert output.metadata["prompt_source"] == "base"
    assert output.action.name == "flap"
    assert output.action.value == 1


def test_starvla_policy_uses_wrapper_default_unnorm_key(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: FakePolicyServerWrapper)
    config = _config(tmp_path)
    del config["policy"]["unnorm_key"]
    policy = build_policy(config, ActionResolver(config["env"]))

    output = policy.predict(
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
            metadata={"env_raw_rgb_frame_stack": np.zeros((2, 3, 3), dtype=np.uint8)},
        )
    )

    wrapper = FakePolicyServerWrapper.instances[0]
    assert wrapper.unnorm_key is None
    assert wrapper.calls[0]["unnorm_key"] is None
    assert output.action.name == "flap"


def test_starvla_policy_uses_saved_checkpoint_no_state_config(monkeypatch, tmp_path: Path):
    import sys

    from omegaconf import OmegaConf

    from latency_bench.policy import starvla

    starvla_root = str(starvla.STARVLA_ROOT)
    if starvla_root not in sys.path:
        sys.path.insert(0, starvla_root)
    from starVLA.model.framework.share_tools import apply_config_compat

    class SavedCheckpointPolicyServerWrapper(FakePolicyServerWrapper):
        def __init__(
            self,
            ckpt_path,
            device="cuda",
            use_bf16=False,
            unnorm_key=None,
            action_output_mode=None,
            rl_games_env_name=None,
        ):
            super().__init__(
                ckpt_path=ckpt_path,
                device=device,
                use_bf16=use_bf16,
                unnorm_key=unnorm_key,
                action_output_mode=action_output_mode,
                rl_games_env_name=rl_games_env_name,
            )
            cfg = OmegaConf.create(
                {
                    "datasets": {
                        "vla_data": {
                            "data_mix": "flappy_train__bridge",
                            "dataset_py": "lerobot_datasets",
                            "eval_data_mix": "flappy_train__bridge__val",
                            "obs_image_size": None,
                            "per_device_batch_size": 32,
                        },
                    },
                    "framework": {
                        "name": "QwenOFT",
                        "action_model": {
                            "action_dim": 7,
                            "action_env_dim": 2,
                            "action_hidden_dim": 2560,
                            "action_horizon": 1,
                            "action_model_type": "MLP",
                            "loss_type": "discrete_ce",
                        },
                        "qwenvl": {
                            "attn_implementation": "flash_attention_2",
                            "enable_gradient_checkpointing": True,
                        },
                    },
                    "rl_games": {
                        "task": "flappy",
                    },
                }
            )
            apply_config_compat(cfg)
            self._model_cfg = OmegaConf.to_container(cfg, resolve=True)

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: SavedCheckpointPolicyServerWrapper)
    config = _config(tmp_path)
    policy = build_policy(config, ActionResolver(config["env"]))

    output = policy.predict(
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
            metadata={"env_raw_rgb_frame_stack": np.zeros((2, 3, 3), dtype=np.uint8)},
        )
    )

    wrapper = FakePolicyServerWrapper.instances[0]
    example = wrapper.calls[0]["examples"][0]
    assert "state" not in example
    assert output.action.name == "flap"


def test_starvla_policy_uses_legacy_train_data_include_state_config(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    class Pi05CheckpointPolicyServerWrapper(FakePolicyServerWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._model_cfg = {
                "train_data": {
                    "include_state": True,
                },
                "datasets": {
                    "vla_data": {
                        "data_mix": "deadly_corridor_train__bridge",
                    },
                },
                "framework": {
                    "action_model": {
                        "state_dim": 7,
                    },
                },
            }

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: Pi05CheckpointPolicyServerWrapper)
    config = _config(tmp_path)
    policy = build_policy(config, ActionResolver(config["env"]))

    policy.predict(
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
            metadata={"env_raw_rgb_frame_stack": np.zeros((2, 3, 3), dtype=np.uint8)},
        )
    )

    example = FakePolicyServerWrapper.instances[0].calls[0]["examples"][0]
    assert example["state"].shape == (1, 7)


def test_starvla_policy_uses_pi05_model_family_include_state_config(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    class Pi05CheckpointPolicyServerWrapper(FakePolicyServerWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._model_cfg = {
                "model": "pi05",
                "rl_games": {
                    "model_alias": "pi-0.5",
                },
                "datasets": {
                    "vla_data": {
                        "data_mix": "deadly_corridor_train__bridge",
                    },
                },
                "framework": {
                    "name": "QwenPI_v3",
                    "action_model": {
                        "state_dim": 7,
                    },
                },
            }

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: Pi05CheckpointPolicyServerWrapper)
    config = _config(tmp_path)
    policy = build_policy(config, ActionResolver(config["env"]))

    policy.predict(
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
            metadata={"env_raw_rgb_frame_stack": np.zeros((2, 3, 3), dtype=np.uint8)},
        )
    )

    example = FakePolicyServerWrapper.instances[0].calls[0]["examples"][0]
    assert example["state"].shape == (1, 7)


def test_starvla_policy_uses_openvla_model_family_no_state_config(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    class OpenVlaCheckpointPolicyServerWrapper(FakePolicyServerWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._model_cfg = {
                "model": "openvla",
                "rl_games": {
                    "model_alias": "openvla",
                },
                "datasets": {
                    "vla_data": {
                        "data_mix": "deadly_corridor_train__bridge",
                    },
                },
                "framework": {
                    "name": "QwenOFT",
                    "action_model": {
                        "state_dim": 1,
                    },
                },
            }

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: OpenVlaCheckpointPolicyServerWrapper)
    config = _config(tmp_path)
    policy = build_policy(config, ActionResolver(config["env"]))

    policy.predict(
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
            metadata={"env_raw_rgb_frame_stack": np.zeros((2, 3, 3), dtype=np.uint8)},
        )
    )

    example = FakePolicyServerWrapper.instances[0].calls[0]["examples"][0]
    assert "state" not in example


def test_starvla_passes_single_raw_frame_and_base_prompt(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: FakePolicyServerWrapper)
    config = _config(tmp_path)
    policy = build_policy(config, ActionResolver(config["env"]))

    policy.predict(
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
            metadata={"env_raw_rgb_frame_stack": np.full((1, 12, 16, 3), 77, dtype=np.uint8)},
        )
    )

    example = FakePolicyServerWrapper.instances[0].calls[0]["examples"][0]
    assert example["lang"] == BASE_PROMPTS["flappy"]
    assert len(example["image"]) == 1
    assert np.asarray(example["image"][0]).shape == (12, 16, 3)
    assert np.asarray(example["image"][0]).mean() == 77


def test_starvla_omits_state_when_checkpoint_excludes_state(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    class NoStatePolicyServerWrapper(FakePolicyServerWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._model_cfg = {
                "datasets": {
                    "vla_data": {
                        "include_state": False,
                    },
                },
                "framework": {
                    "action_model": {},
                },
            }

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: NoStatePolicyServerWrapper)
    config = _config(tmp_path)
    policy = build_policy(config, ActionResolver(config["env"]))

    policy.predict(
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
            metadata={"env_raw_rgb_frame_stack": np.zeros((2, 3, 6), dtype=np.uint8)},
        )
    )

    example = FakePolicyServerWrapper.instances[0].calls[0]["examples"][0]
    assert "state" not in example


def test_starvla_requires_state_dim_when_checkpoint_includes_state(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    class MissingStateDimWrapper(FakePolicyServerWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._model_cfg = {
                "datasets": {
                    "vla_data": {
                        "include_state": True,
                    },
                },
                "framework": {
                    "action_model": {},
                },
            }

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: MissingStateDimWrapper)
    config = _config(tmp_path)
    with pytest.raises(KeyError, match="state_dim"):
        build_policy(config, ActionResolver(config["env"]))


def test_starvla_latency_prompt_map_does_not_change_image_count(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: FakePolicyServerWrapper)
    prompt_map_path = tmp_path / "latency_prompt_map.json"
    prompt_map_path.write_text(
        json.dumps({"3": {"latency_raw_frames": 3, "latency_ms": 100.0, "prompt": "latency prompt 3"}}),
        encoding="utf-8",
    )
    config = _config(tmp_path, latency_prompt_map_path=str(prompt_map_path))
    policy = build_policy(config, ActionResolver(config["env"]))
    frame_stack = np.zeros((6, 8, 9), dtype=np.uint8)
    frame_stack[:3, :, :] = 33
    frame_stack[3:, :, :] = 44

    output = policy.predict(
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
            metadata={
                "latency_raw_frames": 3,
                "env_raw_rgb_frame_stack": frame_stack,
            },
        )
    )

    wrapper = FakePolicyServerWrapper.instances[0]
    example = wrapper.calls[0]["examples"][0]
    assert example["lang"] == "latency prompt 3"
    assert len(example["image"]) == 2
    assert np.asarray(example["image"][0]).mean() == 33
    assert np.asarray(example["image"][1]).mean() == 44
    assert output.metadata["prompt_source"] == "latency_prompt_map"
    assert output.metadata["unnorm_key"] == "flappy"


def test_starvla_latency_prompt_map_uses_raw_rgb_frame_stack_metadata(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: FakePolicyServerWrapper)
    prompt_map_path = tmp_path / "latency_prompt_map.json"
    prompt_map_path.write_text(
        json.dumps({"2": {"latency_raw_frames": 2, "latency_ms": 33.0, "prompt": "latency prompt 2"}}),
        encoding="utf-8",
    )
    config = _config(tmp_path, latency_prompt_map_path=str(prompt_map_path))
    policy = build_policy(config, ActionResolver(config["env"]))
    policy_observation_stack = np.zeros((4, 8, 8), dtype=np.uint8)
    raw_rgb_stack = np.zeros((4, 12, 16, 3), dtype=np.uint8)
    raw_rgb_stack[-1] = 77

    policy.predict(
        Observation(
            data=policy_observation_stack,
            env_step=0,
            sim_time_ms=0.0,
            metadata={
                "latency_raw_frames": 2,
                "env_raw_rgb_frame_stack": raw_rgb_stack,
            },
        )
    )

    example = FakePolicyServerWrapper.instances[0].calls[0]["examples"][0]
    assert len(example["image"]) == 4
    assert np.asarray(example["image"][-1]).shape == (12, 16, 3)
    assert np.asarray(example["image"][-1]).mean() == 77


def test_flappy_realtime_starvla_configs_use_base_prompt_without_latency_prompt_map():
    for config_path in FLAPPY_REALTIME_STARVLA_CONFIGS:
        config = load_config(REPO_ROOT / config_path)

        assert config["executor"]["mode"] == "realtime"
        assert config["policy"]["type"] == "starvla"
        assert "latency_prompt_map_path" not in config["policy"]
        assert "latency_prompt_raw_frames" not in config["policy"]


def test_flappy_simulated_starvla_configs_keep_latency_prompt_map():
    for config_path in FLAPPY_SIMULATED_STARVLA_CONFIGS:
        config = load_config(REPO_ROOT / config_path)

        assert config["executor"]["mode"] == "simulated"
        assert config["policy"]["type"] == "starvla"
        assert "latency_prompt_map_path" in config["policy"]


def test_starvla_requires_raw_rgb_frame_stack_metadata(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: FakePolicyServerWrapper)
    config = _config(tmp_path)
    policy = build_policy(config, ActionResolver(config["env"]))

    with pytest.raises(KeyError, match="env_raw_rgb_frame_stack"):
        policy.predict(Observation(data=np.zeros((2, 3, 3), dtype=np.uint8), env_step=0, sim_time_ms=0.0))


def test_deadly_corridor_factorized_starvla_vector_decodes_to_tuple_action():
    from latency_bench.policy.starvla import decode_starvla_action

    vector = np.asarray([0.0, 0.2, 0.9, 0.8, 0.1, 0.2, 0.1, 0.7, 0.0, 0.3, 0.9])

    action, metadata = decode_starvla_action(
        vector=vector,
        env_name="deadly_corridor",
        action_by_raw_id={},
    )

    assert action.value == [2, 0, 1, 1]
    assert action.name == "TURN_RIGHT+MOVE_LEFT+ATTACK"
    assert metadata["action_layout"] == "deadly_corridor_factorized_11"

    with pytest.raises(ValueError, match="expected 7, 11, or 54"):
        decode_starvla_action(
            vector=np.zeros(8, dtype=np.float32),
            env_name="deadly_corridor",
            action_by_raw_id={},
        )


def test_demon_attack_decodes_only_six_supported_actions():
    from latency_bench.policy.starvla import decode_starvla_action

    vector = np.asarray([0.0, 0.2, 0.9, 0.1, 0.4, 0.3, 10.0], dtype=np.float32)

    action, metadata = decode_starvla_action(
        vector=vector,
        env_name="demon_attack",
        action_by_raw_id={},
    )

    assert action.value == 2
    assert action.name == "RIGHT"
    assert metadata["raw_action_id"] == 2


def test_starvla_policy_uses_decoded_actions_not_raw_scores(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    class ConflictingPolicyServerWrapper(FakePolicyServerWrapper):
        def predict_action(self, examples, unnorm_key=None, **kwargs):
            self.calls.append(
                {
                    "examples": examples,
                    "unnorm_key": unnorm_key,
                    "kwargs": kwargs,
                }
            )
            return {
                "actions": np.asarray([[[1]]], dtype=np.int64),
                "raw_action_scores": np.asarray([[[0.9, 0.1]]], dtype=np.float32),
                "action_output_type": "rl_games_discrete_id",
            }

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: ConflictingPolicyServerWrapper)
    config = _config(tmp_path)
    policy = build_policy(config, ActionResolver(config["env"]))

    output = policy.predict(
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
            metadata={"env_raw_rgb_frame_stack": np.zeros((2, 3, 3), dtype=np.uint8)},
        )
    )

    assert output.action.name == "flap"
    assert output.raw_output == 1
    assert output.metadata["raw_action_scores"] == [0.8999999761581421, 0.10000000149011612]


def test_deadly_corridor_joint_fifty_four_starvla_vector_decodes_to_tuple_action():
    from latency_bench.policy.starvla import decode_starvla_action

    vector = np.zeros(54, dtype=np.float32)
    vector[11] = 1.0

    action, metadata = decode_starvla_action(
        vector=vector,
        env_name="deadly_corridor",
        action_by_raw_id={},
    )

    assert action.value == [0, 1, 2, 1]
    assert action.name == "MOVE_FORWARD+MOVE_RIGHT+ATTACK"
    assert metadata["raw_action_id"] == 11
    assert metadata["decoded_action_tuple"] == [0, 1, 2, 1]
    assert metadata["turn_label"] == "TURN_NOOP"
    assert metadata["move_label"] == "MOVE_FORWARD"
    assert metadata["strafe_label"] == "MOVE_RIGHT"
    assert metadata["attack_label"] == "ATTACK"
    assert metadata["action_layout"] == "deadly_corridor_joint_54"


def test_predict_batch_calls_starvla_once(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: FakePolicyServerWrapper)
    config = _config(tmp_path)
    policy = build_policy(config, ActionResolver(config["env"]))
    observations = [
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
            metadata={"env_raw_rgb_frame_stack": np.zeros((2, 3, 3), dtype=np.uint8)},
        ),
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=1,
            sim_time_ms=33.3,
            metadata={"env_raw_rgb_frame_stack": np.zeros((2, 3, 3), dtype=np.uint8)},
        ),
    ]

    outputs = policy.predict_batch(observations)

    wrapper = FakePolicyServerWrapper.instances[0]
    assert len(wrapper.calls) == 1
    assert len(wrapper.calls[0]["examples"]) == 2
    assert [output.action.name for output in outputs] == ["flap", "noop"]


def test_starvla_default_policy_metadata_omits_input_frame_hashes(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: FakePolicyServerWrapper)
    config = _config(tmp_path)
    policy = build_policy(config, ActionResolver(config["env"]))

    output = policy.predict(
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
            metadata={"env_raw_rgb_frame_stack": np.zeros((2, 3, 3), dtype=np.uint8)},
        )
    )

    assert "input_frame_sha256" not in output.metadata
    assert "input_frame_shape" not in output.metadata
    assert output.metadata["input_frame_count"] == 1
