"""Probe MyoChallenge Table Tennis P2 and emit its benchmark contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scaffold_myochallenge import (
    build_myochallenge_env_config,
    build_myochallenge_probe_report,
    write_myochallenge_outputs,
)
from latency_bench.envs.tasks.myochallenge_table_tennis import (
    GYMNASIUM_MYOCHALLENGE_TABLE_TENNIS_P2_ENV_ID,
    MYOCHALLENGE_TABLE_TENNIS_P2_BASE_ENV_ID,
    MYOCHALLENGE_TABLE_TENNIS_P2_CAMERA,
    MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE,
)


MYOCHALLENGE_TABLE_TENNIS_TASK_NAME = "myochallenge_table_tennis_p2"
MYOCHALLENGE_TABLE_TENNIS_REGISTRATION_IMPORT = (
    "latency_bench.envs.tasks.myochallenge_table_tennis"
)
MYOCHALLENGE_TABLE_TENNIS_PROMPT = (
    "Return the incoming ping-pong ball with the paddle so it contacts the "
    "paddle exactly once and lands on the opponent side without touching the "
    "agent side. Predict continuous muscle and actuator controls in the "
    "native MuJoCo actuator order."
)
MYOCHALLENGE_TABLE_TENNIS_ACTION_DIM = 275


def build_myochallenge_table_tennis_probe_report(
    *,
    base_env_id: str = MYOCHALLENGE_TABLE_TENNIS_P2_BASE_ENV_ID,
    base_make_kwargs: dict[str, Any],
    camera_name: str = MYOCHALLENGE_TABLE_TENNIS_P2_CAMERA,
    render_width: int = MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[0],
    render_height: int = MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[1],
) -> dict[str, Any]:
    """Probe the pinned Table Tennis wrapper with its approved zero vector."""
    return build_myochallenge_probe_report(
        benchmark_env_id=GYMNASIUM_MYOCHALLENGE_TABLE_TENNIS_P2_ENV_ID,
        base_env_id=base_env_id,
        registration_imports=[MYOCHALLENGE_TABLE_TENNIS_REGISTRATION_IMPORT],
        base_make_kwargs=base_make_kwargs,
        camera_name=camera_name,
        render_width=render_width,
        render_height=render_height,
        noop_action=np.zeros(MYOCHALLENGE_TABLE_TENNIS_ACTION_DIM, dtype=np.float32),
    )


def build_myochallenge_table_tennis_env_config(
    *,
    report: dict[str, Any],
    base_env_id: str,
    base_make_kwargs: dict[str, Any],
    camera_name: str,
    render_width: int,
    render_height: int,
) -> dict[str, Any]:
    """Build the frozen Table Tennis ``env`` mapping from one API probe."""
    return {
        "env": build_myochallenge_env_config(
            report=report,
            task_name=MYOCHALLENGE_TABLE_TENNIS_TASK_NAME,
            benchmark_env_id=GYMNASIUM_MYOCHALLENGE_TABLE_TENNIS_P2_ENV_ID,
            base_env_id=base_env_id,
            registration_imports=[MYOCHALLENGE_TABLE_TENNIS_REGISTRATION_IMPORT],
            base_make_kwargs=base_make_kwargs,
            camera_name=camera_name,
            render_width=render_width,
            render_height=render_height,
            base_prompt=MYOCHALLENGE_TABLE_TENNIS_PROMPT,
            env_fps=report["timing"]["native_env_fps"],
            obs_fps=report["timing"]["native_env_fps"],
            frame_stack=1,
        )
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-env-id", default=MYOCHALLENGE_TABLE_TENNIS_P2_BASE_ENV_ID)
    parser.add_argument("--base-make-kwargs-json", default="{}")
    parser.add_argument("--camera-name", default=MYOCHALLENGE_TABLE_TENNIS_P2_CAMERA)
    parser.add_argument("--render-width", type=int, default=MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[0])
    parser.add_argument("--render-height", type=int, default=MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[1])
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--env-yaml-path", type=Path)
    args = parser.parse_args(argv)

    base_make_kwargs = json.loads(args.base_make_kwargs_json)
    report = build_myochallenge_table_tennis_probe_report(
        base_env_id=args.base_env_id,
        base_make_kwargs=base_make_kwargs,
        camera_name=args.camera_name,
        render_width=args.render_width,
        render_height=args.render_height,
    )
    env_config = build_myochallenge_table_tennis_env_config(
        report=report,
        base_env_id=args.base_env_id,
        base_make_kwargs=base_make_kwargs,
        camera_name=args.camera_name,
        render_width=args.render_width,
        render_height=args.render_height,
    )
    write_myochallenge_outputs(
        report=report,
        env_config=env_config["env"],
        report_path=args.report_path,
        env_yaml_path=args.env_yaml_path,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
