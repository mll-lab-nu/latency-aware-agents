from __future__ import annotations

from pathlib import Path
from typing import Any

from training.common.config_adapter import load_training_config


def load_training_config_generic(path: str | Path) -> dict[str, Any]:
    return load_training_config(Path(path))
