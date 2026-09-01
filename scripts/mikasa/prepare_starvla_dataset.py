#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODALITY = {
    "state": {
        "proprio": {
            "start": 0,
            "end": 7,
            "dtype": "float32",
            "absolute": True,
            "original_key": "observation.state",
        }
    },
    "action": {
        "eef": {
            "start": 0,
            "end": 7,
            "dtype": "float32",
            "absolute": False,
            "original_key": "action",
        }
    },
    "video": {
        "top": {"original_key": "observation.images.top"},
        "wrist": {"original_key": "observation.images.wrist"},
    },
    "annotation": {
        "human.task_description": {"original_key": "task_index"},
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-name", required=True)
    parser.add_argument("--val-name", required=True)
    args = parser.parse_args()

    for name in (args.train_name, args.val_name):
        modality_path = args.data_root / name / "meta" / "modality.json"
        modality_path.write_text(json.dumps(MODALITY, indent=2) + "\n", encoding="utf-8")

    mixture_path = args.data_root / "mikasa_intercept_grab_fast_h1.json"
    mixture_path.write_text(
        json.dumps(
            {
                args.train_name: [[args.train_name, 1.0, "mikasa_franka_h1"]],
                args.val_name: [[args.val_name, 1.0, "mikasa_franka_h1"]],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(mixture_path)


if __name__ == "__main__":
    main()
