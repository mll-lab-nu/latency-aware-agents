from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FLAPPY_GYMNASIUM_ROOT = REPO_ROOT / "flappy-bird-gymnasium"


def ensure_flappy_gymnasium_on_path() -> None:
    if FLAPPY_GYMNASIUM_ROOT.exists() and str(FLAPPY_GYMNASIUM_ROOT) not in sys.path:
        sys.path.insert(0, str(FLAPPY_GYMNASIUM_ROOT))
