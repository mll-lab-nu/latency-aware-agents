"""Replay sampling used by the Balance and MIKASA resident DAgger runs."""

from __future__ import annotations

import json
from pathlib import Path


def write_dagger_mixture(
    bootstrap: Path, deltas: list[Path], robot_type: str, destination: Path
) -> None:
    """Keep bootstrap and the latest delta prominent as rounds accumulate."""
    train = [[str(bootstrap / "lerobot"), 1.0, robot_type]]
    if deltas:
        if len(deltas) > 1:
            train.extend(
                [str(delta / "lerobot"), 0.5 / (len(deltas) - 1), robot_type]
                for delta in deltas[:-1]
            )
        train.append([str(deltas[-1] / "lerobot"), 1.0, robot_type])
    payload = {
        "train": train,
        "validation": [[str(bootstrap / "lerobot__val"), 1.0, robot_type]],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
