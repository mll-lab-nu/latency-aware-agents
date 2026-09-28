from scripts.starvla.model_config import make_model_config


def test_make_model_config_sets_h1_continuous_task_contract() -> None:
    for model in ("qwenoft", "qwenpi_v3", "qwengr00t", "wanoft"):
        config = make_model_config(
            model,
            state_dim=7,
            action_dim=7,
            backbone_path="/models/qwen",
        )
        action_model = config["action_model"]
        assert config["qwenvl"]["enable_gradient_checkpointing"] is True
        assert action_model["action_horizon"] == 1
        assert action_model["future_action_window_size"] == 0
        assert action_model["past_action_window_size"] == 0
        assert action_model["state_dim"] == 7
        assert action_model["action_dim"] == 7
        assert action_model["action_env_dim"] == 7

    assert make_model_config(
        "qwenoft", state_dim=198, action_dim=43, backbone_path="/models/qwen"
    )["action_model"]["state_encoding"] == "continuous_projector"
    assert "state_encoding" not in make_model_config(
        "qwengr00t", state_dim=7, action_dim=7, backbone_path="/models/qwen"
    )["action_model"]
    assert make_model_config(
        "wanoft", state_dim=7, action_dim=7, backbone_path="/models/wan"
    )["world_model"] == {"base_wm": "/models/wan"}
