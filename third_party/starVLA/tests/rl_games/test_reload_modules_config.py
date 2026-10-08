from __future__ import annotations

import logging
import sys
import types

from omegaconf import OmegaConf


class _StubRichHandler(logging.Handler):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()

    def emit(self, record: logging.LogRecord) -> None:
        return None


rich_module = types.ModuleType("rich")
rich_logging_module = types.ModuleType("rich.logging")
rich_logging_module.RichHandler = _StubRichHandler
rich_module.logging = rich_logging_module
sys.modules.setdefault("rich", rich_module)
sys.modules.setdefault("rich.logging", rich_logging_module)

from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, reload_module_paths


def test_reload_module_paths_accepts_comma_separated_string() -> None:
    assert reload_module_paths("backbone,action_model") == ["backbone", "action_model"]


def test_reload_module_paths_accepts_access_tracked_list_config() -> None:
    reload_modules = AccessTrackedConfig(OmegaConf.create(["backbone", "action_model"]))

    assert reload_module_paths(reload_modules) == ["backbone", "action_model"]
