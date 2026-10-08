from __future__ import annotations

from omegaconf import OmegaConf

from starVLA.model.framework.share_tools import apply_config_compat


def test_config_compat_defaults_missing_profile_timing_to_disabled() -> None:
    cfg = OmegaConf.create({"trainer": {}})

    apply_config_compat(cfg)

    assert cfg.trainer.profile_timing.enabled is False
    assert cfg.trainer.profile_timing.log_interval == 10


def test_config_compat_preserves_explicit_profile_timing() -> None:
    cfg = OmegaConf.create({"trainer": {"profile_timing": {"enabled": True, "log_interval": 3}}})

    apply_config_compat(cfg)

    assert cfg.trainer.profile_timing.enabled is True
    assert cfg.trainer.profile_timing.log_interval == 3
