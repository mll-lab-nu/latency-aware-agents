from __future__ import annotations

from pathlib import Path
from typing import Any

from training.sample_factory_common.config_adapter import _load_training_config_unvalidated


def load_training_config_generic(path: str | Path) -> dict[str, Any]:
    return _load_training_config_unvalidated(Path(path))
