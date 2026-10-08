from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.rl_games.scripts import launch_train
from starVLA.training.rl_games.alias import MODEL_ALIAS_TO_FRAMEWORK, apply_model_alias


REPO_ROOT = Path(__file__).resolve().parents[2]


def _namespace(mapping: dict) -> SimpleNamespace:
    values = {}
    for key, value in mapping.items():
        values[key] = _namespace(value) if isinstance(value, dict) else value
    return SimpleNamespace(**values)


def test_wan_oft_alias_resolves_to_wan_oft_framework() -> None:
    cfg = _namespace({
        "rl_games": {"model_alias": "wan_oft"},
        "framework": {"name": "QwenOFT"},
    })

    apply_model_alias(cfg)

    assert MODEL_ALIAS_TO_FRAMEWORK["wan_oft"] == "WanOFT"
    assert cfg.framework.name == "WanOFT"


def test_wan_oft_flappy_config_preserves_released_checkpoint_shapes() -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="wan_oft",
        env="flappy",
        init="wan_oft_libero",
        mode="single",
        overrides=[],
    )

    assert cfg.framework.name == "WanOFT"
    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.action_env_dim == 2
    assert cfg.framework.action_model.state_dim == 7
    assert cfg.framework.action_model.action_horizon == 8
    assert cfg.framework.action_model.future_action_window_size == 7
    assert cfg.framework.action_model.past_action_window_size == 0
    assert cfg.framework.action_model.loss_type == "discrete_ce"
    assert cfg.dataset.single_converted_name == "flappy_train__bridge"
    assert cfg.dataset.converted_name == "flappy_train__bridge"
    assert cfg.datasets.vla_data.data_mix == "flappy_train__bridge"
    assert cfg.datasets.vla_data.include_state is True
    assert list(cfg.datasets.vla_data.observation_indices) == [-3, -2, -1, 0]
    assert list(cfg.datasets.vla_data.state_indices) == [0]
    assert list(cfg.datasets.vla_data.action_indices) == list(range(8))
    assert cfg.datasets.vla_data.pack_image_sequence is True
    assert cfg.datasets.vla_data.image_sequence_length == 4
    assert cfg.datasets.vla_data.context_images_column == "observation.context_images"
    assert cfg.base_model.repo_id == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert cfg.initialization.checkpoint_hf_repo_id == "StarVLA/WM4A-Wan2d2-OFT-LIBERO-4in1"
    assert cfg.initialization.checkpoint_filename == "checkpoints/steps_60000_pytorch_model.pt"
    assert cfg.framework.world_model.num_frames is None


def test_wan_oft_demon_attack_config_uses_bridge_dataset_and_six_active_actions() -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="wan_oft",
        env="demon_attack",
        init="wan_oft_libero",
        mode="single",
        overrides=[],
    )

    assert cfg.framework.name == "WanOFT"
    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.action_env_dim == 6
    assert cfg.framework.action_model.state_dim == 7
    assert cfg.framework.action_model.action_horizon == 8
    assert cfg.framework.action_model.future_action_window_size == 7
    assert cfg.framework.action_model.loss_type == "discrete_ce"
    assert cfg.dataset.single_converted_name == "demon_attack_train__bridge"
    assert cfg.dataset.converted_name == "demon_attack_train__bridge"
    assert cfg.datasets.vla_data.data_mix == "demon_attack_train__bridge"
    assert cfg.datasets.vla_data.pack_image_sequence is True
    assert cfg.datasets.vla_data.context_images_column == "observation.context_images"


def test_wan_oft_deadly_corridor_context_config_composes_and_forwards(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="wan_oft",
        env="deadly_corridor",
        init="wan_oft_libero",
        mode="single",
        overrides=[
            "dataset.converted_name=deadly_corridor_train__bridge",
            "datasets.vla_data.data_mix=deadly_corridor_train__bridge",
            "datasets.vla_data.image_sequence_length=5",
            "datasets.vla_data.observation_indices=[-4,-3,-2,-1,0]",
            "framework.world_model.num_frames=5",
            "rl_games.deadly_corridor_loss_type=current_multibinary_bce",
            "rl_games.env_eval.deadly.action_layout=multibinary_7",
            "rl_games.env_eval.deadly.multibinary_threshold=0.0",
        ],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "wan_base"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert cfg.framework.name == "WanOFT"
    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.action_env_dim == 7
    assert cfg.framework.action_model.loss_type == "current_multibinary_bce"
    assert list(cfg.datasets.vla_data.observation_indices) == [-4, -3, -2, -1, 0]
    assert cfg.datasets.vla_data.image_sequence_length == 5
    assert cfg.framework.world_model.num_frames == 5
    assert cfg.rl_games.env_eval.deadly.multibinary_threshold == 0.0
    assert "++framework.action_model.loss_type=current_multibinary_bce" in cmd
    assert "++datasets.vla_data.observation_indices=[-4,-3,-2,-1,0]" in cmd
    assert "++framework.world_model.num_frames=5" in cmd
    assert "++rl_games.env_eval.deadly.multibinary_threshold=0.0" in cmd


def test_launch_train_forwards_world_model_path_only_for_wan_oft(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="wan_oft",
        env="flappy",
        init="wan_oft_libero",
        mode="single",
        overrides=[],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "wan_base"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert f"++framework.world_model.base_wm={tmp_path / 'wan_base'}" in cmd
    assert f"++framework.qwenvl.base_vlm={tmp_path / 'wan_base'}" in cmd

    openvla_cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="flappy",
        init="bridge",
        mode="single",
        overrides=[],
    )
    openvla_cmd = launch_train.build_trainer_command(openvla_cfg, setup, tmp_path, "results/Checkpoints")

    assert not any(item.startswith("++framework.world_model.base_wm=") for item in openvla_cmd)


def test_launch_train_forwards_explicit_wan_oft_bridge_data_mix(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="wan_oft",
        env="flappy",
        init="wan_oft_libero",
        mode="single",
        overrides=[
            "datasets.vla_data.data_mix=flappy_train__bridge",
            "datasets.vla_data.eval_data_mix=flappy_train__bridge__val",
        ],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "wan_base"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "++datasets.vla_data.data_mix=flappy_train__bridge" in cmd
    assert "++datasets.vla_data.eval_data_mix=flappy_train__bridge__val" in cmd


def test_launch_train_forwards_wan_world_model_num_frames(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="wan_oft",
        env="flappy",
        init="wan_oft_libero",
        mode="single",
        overrides=["framework.world_model.num_frames=5"],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "wan_base"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert cfg.framework.world_model.num_frames == 5
    assert "++framework.world_model.num_frames=5" in cmd


def test_wan_temporal_latent_frame_count_exposes_four_frame_boundary() -> None:
    pytest.importorskip("torch")
    wan_module = importlib.import_module("starVLA.model.modules.world_model.Wan2")

    assert wan_module.wan_temporal_latent_frame_count(4) == 1
    assert wan_module.wan_temporal_latent_frame_count(5) == 2


def test_train_starvla_uses_device_aware_distributed_barrier() -> None:
    trainer_text = (REPO_ROOT / "starVLA" / "training" / "train_starvla.py").read_text(encoding="utf-8")

    assert "def _distributed_barrier()" in trainer_text
    assert "dist.barrier(device_ids=[device_idx])" in trainer_text
    assert trainer_text.count("dist.barrier()") == 1
    assert trainer_text.count("_distributed_barrier()") == 5


def test_rl_games_modality_indices_can_be_overridden_for_wan_oft_clip() -> None:
    temporal_clip = importlib.import_module("starVLA.training.rl_games.temporal_clip")

    resolved = temporal_clip.resolve_modality_indices(
        default_observation_indices=[0],
        default_state_indices=[0],
        default_action_indices=[0],
        data_cfg={
            "observation_indices": [-3, -2, -1, 0],
            "state_indices": [0],
            "action_indices": list(range(8)),
        },
    )

    assert resolved.observation_indices == [-3, -2, -1, 0]
    assert resolved.language_indices == [0]
    assert resolved.state_indices == [0]
    assert resolved.action_indices == list(range(8))


def test_pack_sample_preserves_configured_temporal_clip() -> None:
    temporal_clip = importlib.import_module("starVLA.training.rl_games.temporal_clip")

    frames = np.stack([
        np.full((4, 4, 3), fill_value=value, dtype=np.uint8)
        for value in (10, 20, 30, 40)
    ])

    images = temporal_clip.pack_image_sequence(
        frames=frames,
        pack_image_sequence=True,
        image_sequence_length=4,
        resize_size=(224, 224),
    )

    assert len(images) == 4
    assert [int(np.asarray(image.resize((4, 4)))[0, 0, 0]) for image in images] == [10, 20, 30, 40]


def test_wan_oft_implements_discrete_ce_action_loss() -> None:
    wan_oft_source = (REPO_ROOT / "starVLA" / "model" / "framework" / "WM4A" / "WanOFT.py").read_text(
        encoding="utf-8"
    )

    assert "discrete_ce" in wan_oft_source
    assert "F.cross_entropy" in wan_oft_source


def test_wan_oft_supports_current_action_discrete_ce_loss() -> None:
    wan_oft_source = (REPO_ROOT / "starVLA" / "model" / "framework" / "WM4A" / "WanOFT.py").read_text(
        encoding="utf-8"
    )

    assert "current_discrete_ce" in wan_oft_source
    assert "pred_actions[:, 0, :effective_dim]" in wan_oft_source
    assert "actions_target[:, 0, :effective_dim]" in wan_oft_source


def test_wan_oft_current_plus_future_ce_adds_weighted_future_supervision() -> None:
    torch = pytest.importorskip("torch")
    wan_oft_module = importlib.import_module("starVLA.model.framework.WM4A.WanOFT")
    model = object.__new__(wan_oft_module.Wan_OFT)
    model.action_horizon = 3
    model.action_env_dim = 2
    model.action_loss_type = "current_plus_future_discrete_ce"
    model.config = _namespace({
        "framework": {
            "action_model": {
                "class_weights": None,
                "future_loss_weight": 0.25,
            }
        }
    })

    pred_actions = torch.tensor([[[2.0, -1.0], [-1.0, 2.0], [3.0, -2.0]]])
    actions_target = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])

    loss = model._compute_action_loss(pred_actions, actions_target)

    current_loss = torch.nn.functional.cross_entropy(pred_actions[:, 0, :], torch.tensor([0]))
    future_loss = torch.nn.functional.cross_entropy(
        pred_actions[:, 1:, :].reshape(-1, 2),
        torch.tensor([1, 1]),
    )
    expected = current_loss + 0.25 * future_loss
    assert torch.allclose(loss, expected)


def test_wan_oft_current_multibinary_bce_supervises_only_current_action() -> None:
    torch = pytest.importorskip("torch")
    wan_oft_module = importlib.import_module("starVLA.model.framework.WM4A.WanOFT")
    model = object.__new__(wan_oft_module.Wan_OFT)
    model.action_horizon = 2
    model.action_env_dim = 3
    model.action_loss_type = "current_multibinary_bce"

    pred_actions = torch.tensor([[[1.0, -1.0, 0.5], [20.0, 20.0, 20.0]]])
    actions_target = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])

    loss = model._compute_action_loss(pred_actions, actions_target)

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_actions[:, 0, :],
        actions_target[:, 0, :],
    )
    assert torch.allclose(loss, expected)


def test_wan_oft_source_exposes_current_plus_future_ce_loss() -> None:
    wan_oft_source = (REPO_ROOT / "starVLA" / "model" / "framework" / "WM4A" / "WanOFT.py").read_text(
        encoding="utf-8"
    )

    assert "current_plus_future_discrete_ce" in wan_oft_source
    assert "future_loss_weight" in wan_oft_source
    assert "future_logits = pred_actions[:, 1:, :effective_dim]" in wan_oft_source


def test_wan_oft_training_forward_resizes_images_with_configured_obs_size() -> None:
    wan_oft_source = (REPO_ROOT / "starVLA" / "model" / "framework" / "WM4A" / "WanOFT.py").read_text(
        encoding="utf-8"
    )
    forward_source = wan_oft_source.split("    def forward(", 1)[1].split("    @torch.inference_mode()", 1)[0]

    assert "train_obs_image_size = getattr(self.config.datasets.vla_data, \"obs_image_size\", None)" in forward_source
    assert "batch_images = resize_images(batch_images, target_size=train_obs_image_size)" in forward_source
    assert forward_source.index("batch_images = resize_images") < forward_source.index("self.backbone.build_inputs")


def test_wan_oft_discrete_ce_supports_configured_class_weights() -> None:
    wan_oft_source = (REPO_ROOT / "starVLA" / "model" / "framework" / "WM4A" / "WanOFT.py").read_text(
        encoding="utf-8"
    )

    assert "class_weights" in wan_oft_source
    assert "_action_class_weight_tensor" in wan_oft_source
    assert "weight=class_weights" in wan_oft_source


def test_wan_oft_default_action_query_source_is_mean_pooling() -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="wan_oft",
        env="flappy",
        init="wan_oft_libero",
        mode="single",
        overrides=[],
    )

    assert cfg.framework.action_model.action_query_source == "mean"


def test_wan_oft_mean_action_query_projection_aligns_hidden_dtype_to_projector() -> None:
    wan_oft_source = (REPO_ROOT / "starVLA" / "model" / "framework" / "WM4A" / "WanOFT.py").read_text(
        encoding="utf-8"
    )
    pool_source = wan_oft_source.split("    def _pool_to_action_queries(", 1)[1].split("    def forward(", 1)[0]

    assert "self.action_query_proj.weight.dtype" in pool_source
    assert "hidden_states.mean(dim=1).to(dtype=target_dtype)" in pool_source


def test_token_attention_action_query_projector_uses_full_token_sequence() -> None:
    torch = pytest.importorskip("torch")
    wan_oft_module = importlib.import_module("starVLA.model.framework.WM4A.WanOFT")
    projector = wan_oft_module.TokenAttentionActionQueryProjector(
        hidden_dim=16,
        chunk_len=3,
        num_heads=4,
    )
    hidden_states = torch.randn(2, 5, 16, requires_grad=True)

    action_queries = projector(hidden_states)
    action_queries.square().sum().backward()

    assert action_queries.shape == (2, 3, 16)
    assert hidden_states.grad is not None
    assert torch.count_nonzero(hidden_states.grad).item() > 0
