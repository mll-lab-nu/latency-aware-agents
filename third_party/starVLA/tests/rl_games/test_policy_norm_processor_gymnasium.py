import pytest

from starVLA.dataloader.gr00t_lerobot.registry import resolve_robot_type


@pytest.mark.parametrize(
    ("task_name", "data_mix", "robot_type"),
    [
        ("flappy", "flappy_train", "rl_games_flappy"),
        ("flappy", "flappy_train__val", "rl_games_flappy"),
        ("demon_attack", "demon_attack_train", "rl_games_demon_attack"),
        ("deadly_corridor", "deadly_corridor_train", "rl_games_deadly_corridor"),
    ],
)
def test_legacy_rl_games_resolves_from_static_mixture(
    task_name: str,
    data_mix: str,
    robot_type: str,
) -> None:
    config = {
        "rl_games": {"task": task_name},
        "datasets": {"vla_data": {"data_mix": data_mix}},
    }

    assert resolve_robot_type(config) == robot_type


def test_unknown_static_mixture_still_fails() -> None:
    config = {
        "rl_games": {"task": "flappy"},
        "datasets": {"vla_data": {"data_mix": "unregistered_custom_mix"}},
    }

    with pytest.raises(KeyError, match="unregistered_custom_mix"):
        resolve_robot_type(config)


def test_history_vla_bundle_resolves_explicit_robot_type() -> None:
    config = {"datasets": {"vla_data": {"robot_type": "rl_games_flappy"}}}

    assert resolve_robot_type(config) == "rl_games_flappy"


@pytest.mark.parametrize("robot_type,expected_state,expected_actions", [
    ("rl_games_asterix", [[2.0, 4.0]], [[[0.0, 0.5, 1.0]]]),
    ("rl_games_gymnasium_native", [[1.0, 1.0]], [[[0.0, 1.0, 2.0]]]),
])
def test_checkpoint_and_live_wrappers_use_the_training_transform(
    tmp_path, monkeypatch, robot_type, expected_state, expected_actions,
):
    """Long-lived model contract: environment identity cannot replace the training DataConfig."""
    import json
    import numpy as np
    import torch
    import yaml
    from latency_bench.policy.starvla import LiveStarVlaWrapper
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from starVLA.model.framework.base_framework import baseframework

    def stats(dim):
        return {"min": [-20.0] * dim, "max": [20.0] * dim,
                "mean": [0.0] * dim, "std": [1.0] * dim,
                "q01": [-2.0] * dim, "q99": [2.0] * dim}

    cfg = {"datasets": {"vla_data": {"robot_type": robot_type}},
           "framework": {"action_model": {"action_horizon": 1}}}
    statistics = {"new_embodiment": {"state": stats(2), "action": stats(3)}}
    checkpoint = tmp_path / "checkpoints/steps_1_pytorch_model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"injected model")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_path / "dataset_statistics.json").write_text(json.dumps(statistics))

    class Framework(torch.nn.Module):
        def predict_action(self, examples, **kwargs):
            self.examples = examples
            return {"normalized_actions": np.asarray([[[0.0, 0.5, 1.0]]])}

    disk_model, live_model = Framework(), Framework()
    monkeypatch.setattr(baseframework, "from_pretrained", lambda *args, **kwargs: disk_model)
    disk = PolicyServerWrapper(str(checkpoint), device="cpu", action_output_mode="rl_games",
                               rl_games_env_name="gymnasium", rl_games_action_env_dim=3,
                               rl_games_gymnasium_action_space_type="box")
    live = LiveStarVlaWrapper(framework=live_model, model_cfg=cfg, env_name="gymnasium",
                              normalization_statistics=statistics, rl_games_action_env_dim=3,
                              gymnasium_action_space_type="box")
    examples = [{"image": [], "lang": "Move.", "state": np.asarray([[2.0, 4.0]], dtype=np.float32)}]
    for wrapper, model in ((disk, disk_model), (live, live_model)):
        result = wrapper.predict_action(examples)
        np.testing.assert_allclose(model.examples[0]["state"], expected_state)
        np.testing.assert_allclose(result["actions"], expected_actions)
