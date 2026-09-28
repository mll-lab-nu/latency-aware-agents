"""Collect the fixed teacher bootstrap datasets used by StarVLA H1 runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.data.starvla_tasks import task_spec, write_task_contract


FORMAL_EPISODES = {
    "mikasa_intercept_grab_fast": (250, 25),
}


def _run(command: Sequence[str], *, cwd: Path = REPO_ROOT) -> None:
    env = os.environ.copy()
    pythonpath = [
        str(REPO_ROOT),
        str(REPO_ROOT / "third_party/starVLA"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    subprocess.run(list(command), cwd=cwd, env=env, check=True)


def _counts(task: str, smoke: bool) -> tuple[int, int]:
    return (2, 1) if smoke else FORMAL_EPISODES[task]


def _prepare_balance_bootstrap(raw_dir: Path, output_dir: Path) -> None:
    """Convert Balance teacher rows and freeze the bootstrap task contract."""
    from latency_bench.data.exporters.starvla_lerobot import (
        convert_canonical_rollout_to_lerobot,
    )
    from scripts.starvla.prepare_dataset import prepare_lerobot

    converted = output_dir.with_name(f"_{output_dir.name}_converted")
    convert_canonical_rollout_to_lerobot(
        raw_dir,
        converted,
        force=True,
        action_carrier="native",
        normalize_state=False,
        write_starvla_handoff=False,
    )
    converted_val = converted.with_name(f"{converted.name}__val")
    prepare_lerobot(
        "balance",
        converted,
        converted_val if converted_val.exists() else None,
        output_dir,
    )
    shutil.rmtree(converted)
    if converted_val.exists():
        shutil.rmtree(converted_val)


def _collect_mikasa(
    *,
    teacher_checkpoint: Path,
    output_root: Path,
    task_python: str,
    train_episodes: int,
    val_episodes: int,
    keep_failed: bool,
    device: str,
) -> Path:
    del device
    raw_dir = output_root / "raw_mikasa"
    npz_root = raw_dir / "data_npz"
    rlds_root = raw_dir / "data_rlds"
    env_id = "InterceptGrabFast-VLA-v0"
    builder_python = str(
        REPO_ROOT
        / "third_party/MIKASA-Robo/utils/convert_npz_to_rlds/rlds_dataset_builder/.venv/bin/python"
    )
    converter_python = str(
        REPO_ROOT
        / "third_party/MIKASA-Robo/utils/convert_rlds_to_lerobot/.venv/bin/python"
    )
    collect = [
        task_python,
        str(
            REPO_ROOT
            / "third_party/MIKASA-Robo/mikasa_robo_suite/vla/dataset_collectors/"
            / "get_mikasa_robo_datasets.py"
        ),
        "--env-id",
        env_id,
        "--checkpoint",
        str(teacher_checkpoint),
        "--path-to-save-data",
        str(raw_dir),
        "--num-train-data",
        str(train_episodes + val_episodes),
    ]
    if keep_failed:
        collect.append("--keep-failed")
    _run(collect)
    _run(
        [
            builder_python,
            str(
                REPO_ROOT
                / "third_party/MIKASA-Robo/utils/convert_npz_to_rlds/"
                / "convert_npz_task_to_rlds.py"
            ),
            "--task",
            env_id,
            "--repo-root",
            str(REPO_ROOT / "third_party/MIKASA-Robo"),
            "--npz-root",
            str(npz_root),
            "--rlds-root",
            str(rlds_root),
        ]
    )
    temporary_root = raw_dir / "converted_lerobot"
    dataset_name = "intercept_grab_fast_vla_v0"
    _run(
        [
            converter_python,
            str(
                REPO_ROOT
                / "third_party/MIKASA-Robo/utils/convert_rlds_to_lerobot/"
                / "convert_rlds_to_lerobot.py"
            ),
            "--task",
            env_id,
            "--repo-root",
            str(REPO_ROOT / "third_party/MIKASA-Robo"),
            "--rlds-root",
            str(rlds_root),
            "--lerobot-root",
            str(temporary_root),
            "--repo-id-template",
            "local/{task}_h1_train",
            "--fps",
            "20",
            "--split",
            f"train[:{train_episodes}]",
        ]
    )
    _run(
        [
            converter_python,
            str(
                REPO_ROOT
                / "third_party/MIKASA-Robo/utils/convert_rlds_to_lerobot/"
                / "convert_rlds_to_lerobot.py"
            ),
            "--task",
            env_id,
            "--repo-root",
            str(REPO_ROOT / "third_party/MIKASA-Robo"),
            "--rlds-root",
            str(rlds_root),
            "--lerobot-root",
            str(temporary_root),
            "--repo-id-template",
            "local/{task}_h1_val",
            "--fps",
            "20",
            "--split",
            f"train[{train_episodes}:{train_episodes + val_episodes}]",
        ]
    )
    source_train = temporary_root / "local" / f"{dataset_name}_h1_train"
    source_val = temporary_root / "local" / f"{dataset_name}_h1_val"
    train_dir = output_root / "lerobot"
    val_dir = output_root / "lerobot__val"
    train_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_train), train_dir)
    shutil.move(str(source_val), val_dir)
    _write_mikasa_contract(train_dir, val_dir)
    return raw_dir


def _write_mikasa_contract(train_dir: Path, val_dir: Path) -> None:
    spec = task_spec("mikasa")
    for dataset_dir in (train_dir, val_dir):
        (dataset_dir / "meta" / "modality.json").write_text(
            json.dumps(
                {
                    "state": {
                        "proprio": {
                            "start": 0,
                            "end": spec.state_dim,
                            "dtype": "float32",
                            "absolute": True,
                            "original_key": "observation.state",
                        }
                    },
                    "action": {
                        "target": {
                            "start": 0,
                            "end": spec.action_dim,
                            "dtype": "float32",
                            "absolute": False,
                            "original_key": "action",
                        }
                    },
                    "video": {
                        key.removeprefix("video."): {
                            "original_key": f"observation.images.{key.removeprefix('video.')}"
                        }
                        for key in spec.video_keys
                    },
                    "annotation": {
                        "human.task_description": {"original_key": "task_index"}
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    write_task_contract(train_dir, spec=spec)
    write_task_contract(val_dir, spec=spec, normalization_source=train_dir)
    manifest = {
        "format": "latency_bench_starvla_h1_v1",
        "task": spec.name,
        "robot_type": spec.robot_type,
        "state_dim": spec.state_dim,
        "action_dim": spec.action_dim,
        "action_horizon": 1,
        "vla_fps": spec.fps,
        "dagger": False,
    }
    for path in (train_dir / "manifest.json", val_dir / "manifest.json"):
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def collect_bootstrap(
    task: str,
    teacher_checkpoint: str | Path,
    output_root: str | Path,
    task_python: str | Path,
    smoke: bool = False,
    device: str = "cuda:0",
) -> tuple[Path, Path]:
    """Collect and convert one fixed-teacher StarVLA H1 bootstrap dataset."""
    checkpoint = Path(teacher_checkpoint).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_episodes, val_episodes = _counts(task, smoke)
    python = str(task_python)
    keep_failed = smoke
    collector = {
        "mikasa_intercept_grab_fast": _collect_mikasa,
    }[task]
    collector(
        teacher_checkpoint=checkpoint,
        output_root=output,
        task_python=python,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        keep_failed=keep_failed,
        device=device,
    )
    return output / "lerobot", output / "lerobot__val"


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=sorted(FORMAL_EPISODES))
    parser.add_argument("teacher_checkpoint", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--task-python", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    train, val = collect_bootstrap(
        args.task,
        args.teacher_checkpoint,
        args.output_root,
        args.task_python,
        smoke=args.smoke,
        device=args.device,
    )
    print(json.dumps({"train": str(train), "val": str(val)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
