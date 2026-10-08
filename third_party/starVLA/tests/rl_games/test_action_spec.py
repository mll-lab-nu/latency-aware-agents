from pathlib import Path
from types import SimpleNamespace

import pytest

from starVLA.training.rl_games.action_spec import apply_action_spec


REPO_ROOT = Path(__file__).resolve().parents[2]


def _cfg(task, model_alias="openvla", init_mode="scratch", action_carrier="native", action_dim=2):
    return SimpleNamespace(
        rl_games=SimpleNamespace(
            task=task,
            model_alias=model_alias,
            initialization_mode=init_mode,
            action_carrier=action_carrier,
            env_eval=SimpleNamespace(deadly=SimpleNamespace(action_layout="multibinary_7")),
        ),
        framework=SimpleNamespace(
            action_model=SimpleNamespace(
                action_dim=action_dim,
                action_env_dim=action_dim,
                action_horizon=16,
                future_action_window_size=15,
                past_action_window_size=2,
            )
        ),
    )


def test_openvla_bridge_flappy_uses_7d_carrier_with_2d_loss_surface():
    cfg = _cfg("flappy", init_mode="bridge", action_carrier="bridge", action_dim=2)

    apply_action_spec(cfg)

    action_cfg = cfg.framework.action_model
    assert action_cfg.action_dim == 7
    assert action_cfg.action_env_dim == 2
    assert action_cfg.action_horizon == 1
    assert action_cfg.future_action_window_size == 0
    assert action_cfg.past_action_window_size == 0


def test_wan_oft_bridge_flappy_preserves_checkpoint_action_chunk():
    cfg = _cfg("flappy", model_alias="wan_oft", init_mode="bridge", action_carrier="bridge", action_dim=7)

    apply_action_spec(cfg)

    action_cfg = cfg.framework.action_model
    assert action_cfg.action_dim == 7
    assert action_cfg.action_env_dim == 2
    assert action_cfg.action_horizon == 16
    assert action_cfg.future_action_window_size == 15
    assert action_cfg.past_action_window_size == 2


def test_action_carrier_bridge_is_sufficient_for_demon_attack():
    cfg = _cfg("demon_attack", init_mode="scratch", action_carrier="bridge", action_dim=7)

    apply_action_spec(cfg)

    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.action_env_dim == 6


def test_action_carrier_bridge_is_sufficient_for_air_raid():
    cfg = _cfg("air_raid", init_mode="scratch", action_carrier="bridge", action_dim=7)

    apply_action_spec(cfg)

    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.action_env_dim == 6


def test_action_carrier_bridge_is_sufficient_for_defend_the_line():
    cfg = _cfg("defend_the_line", init_mode="scratch", action_carrier="bridge", action_dim=7)

    apply_action_spec(cfg)

    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.action_env_dim == 6
def test_openvla_native_gymnasium_uses_explicit_action_env_dim():
    cfg = _cfg("gymnasium", action_dim=2)
    cfg.framework.action_model.action_env_dim = 9

    apply_action_spec(cfg)

    assert cfg.framework.action_model.action_dim == 9
    assert cfg.framework.action_model.action_env_dim == 9


def test_openvla_native_gymnasium_does_not_fall_back_to_action_dim():
    cfg = _cfg("gymnasium", action_dim=9)
    del cfg.framework.action_model.action_env_dim

    with pytest.raises(AttributeError, match="action_env_dim"):
        apply_action_spec(cfg)


def test_pi0_bridge_forces_model_7d_carrier_and_masks_deadly_to_7d():
    cfg = _cfg("deadly_corridor", model_alias="pi0", init_mode="bridge", action_carrier="bridge", action_dim=32)

    apply_action_spec(cfg)

    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.action_env_dim == 7


def test_wan_oft_deadly_corridor_applies_current_multibinary_loss():
    cfg = _cfg(
        "deadly_corridor",
        model_alias="wan_oft",
        init_mode="bridge",
        action_carrier="bridge",
        action_dim=7,
    )
    cfg.rl_games.deadly_corridor_loss_type = "current_multibinary_bce"

    apply_action_spec(cfg)

    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.action_env_dim == 7
    assert cfg.framework.action_model.loss_type == "current_multibinary_bce"


def test_bridge_rejects_deadly_factorized_11_layout():
    cfg = _cfg("deadly_corridor", init_mode="bridge", action_carrier="bridge", action_dim=11)
    cfg.rl_games.env_eval.deadly.action_layout = "factorized_11"

    with pytest.raises(ValueError, match="7D action carrier"):
        apply_action_spec(cfg)


def test_backbone_bridge_factorized11_stays_native_11d():
    cfg = _cfg(
        "deadly_corridor",
        model_alias="pi-0.5",
        init_mode="backbone_bridge_factorized11",
        action_carrier="native",
        action_dim=32,
    )
    cfg.rl_games.env_eval.deadly.action_layout = "factorized_11"

    apply_action_spec(cfg)

    assert cfg.framework.action_model.action_dim == 32
    assert cfg.framework.action_model.action_env_dim == 11


def test_qwen_pi_legacy_dit_qwen_action_head_preset_is_available():
    source = (REPO_ROOT / "starVLA/model/modules/action_model/GR00T_ActionHeader.py").read_text(
        encoding="utf-8"
    )

    assert '"DiT-Qwen": {"input_embedding_dim": 2048, "attention_head_dim": 64, "num_attention_heads": 32}' in source


def test_qwen_pi_legacy_action_head_uses_action_hidden_width():
    source = (REPO_ROOT / "starVLA/model/modules/action_model/GR00T_ActionHeader.py").read_text(
        encoding="utf-8"
    )

    assert "self.hidden_size = int(config.get(\"action_hidden_dim\", config.hidden_size))" in source
    assert "self.action_hidden_dim = int(config.get(\"action_hidden_dim\", self.model.config.output_dim))" in source
    assert "input_dim=self.action_hidden_dim" in source
    assert "return_all_hidden_states=self._uses_hidden_action_decoder" in source
    assert "model_output = all_hidden_states[-1] if self._uses_hidden_action_decoder else projected_output" in source
