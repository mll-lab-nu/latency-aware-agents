#!/usr/bin/env python3
"""Export one fixed-latency InterceptGrabFast small PPO policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil






def _latency_spec(latency: int) -> dict[str, object]:
    if latency == 0:
        return {"method": "zero"}
    return {"method": "fixed", "fixed_latency_ms": latency * 50.0}


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _copy_tree_payload(*, latency: int, output_root: Path, export_dir: Path) -> dict[str, object]:
    train_dir = output_root / f"fixed_l{latency}"
    eval_dir = output_root / f"fixed_l{latency}_eval"
    required = {
        "checkpoint": train_dir / "best_policy.pt",
        "training_config": train_dir / "config.yaml",
        "best_evaluation": train_dir / "best_evaluation.json",
        "final_evaluation": eval_dir / "evaluation.json",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot export; missing required files:\n" + "\n".join(missing))

    export_dir.mkdir(parents=True, exist_ok=True)
    for child in export_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    _link_or_copy(required["checkpoint"], export_dir / "checkpoint.pt")
    shutil.copy2(required["training_config"], export_dir / "training_config.yaml")
    shutil.copy2(required["best_evaluation"], export_dir / "best_evaluation.json")
    shutil.copy2(required["final_evaluation"], export_dir / "evaluation" / "evaluation.json")

    manifest = {
        "task": "mikasa_intercept_grab_fast",
        "environment": "InterceptGrabFast-VLA-v0",
        "policy": "small PPO",
        "latency": latency,
        "latency_config": _latency_spec(latency),
        "control_hz": 20,
        "training_timesteps": 150000000,
        "source": {
            "train_dir": str(train_dir),
            "eval_dir": str(eval_dir),
        },
    }
    (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (export_dir / "README.md").write_text(
        "\n".join(
            [
                f"# InterceptGrabFast fixed-latency L{latency} small PPO policy",
                "",
                "This artifact contains the selected native PPO teacher checkpoint for",
                "`InterceptGrabFast-VLA-v0` under the fixed-latency small-policy sweep.",
                "",
                f"- Latency: L{latency}",
                f"- Latency config: `{json.dumps(_latency_spec(latency), sort_keys=True)}`",
                "- Control/observation rate: 20 Hz",
                "- Training budget: 150,000,000 environment timesteps",
                "",
                "Files:",
                "",
                "- `checkpoint.pt`: selected `best_policy.pt`, selected inference weights",
                "- `training_config.yaml`: resolved training config",
                "- `best_evaluation.json`: periodic-training best checkpoint evidence",
                "- `evaluation/evaluation.json`: independent post-training evaluation",
                "- `manifest.json`: provenance and source paths",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency", type=int, choices=range(5), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/mikasa_fixed_latency_ppo"))
    args = parser.parse_args()
    export_dir = args.output_root / "inference" / f"fixed_l{args.latency}"
    _copy_tree_payload(latency=args.latency, output_root=args.output_root, export_dir=export_dir)
    print(f"MODEL_PATH={export_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
