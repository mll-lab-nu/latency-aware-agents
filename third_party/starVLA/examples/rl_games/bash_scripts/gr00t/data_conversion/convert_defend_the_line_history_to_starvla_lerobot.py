#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.bash_scripts.gr00t.data_conversion import (
    convert_atari_history_to_starvla_lerobot as history_converter,
    convert_defend_the_line_to_starvla_lerobot as defend_converter,
)


DEFAULT_DATASET_NAME = "latency-sensitive-bench/benchmark-datasets"
DEFAULT_DATASET_CONFIG_NAME = "defend_the_line_fixed_latency_0_1000ep_7k2steps"
DEFAULT_OUTPUT_DIR = Path(
    "data/defend_the_line_fix_latency_0_1000ep_context5/defend_the_line_train__bridge"
)
SOURCE_OBSERVATION_FPS = 8.75
SOURCE_ENV_FPS = 35.0
SOURCE_ENV_FRAMESKIP = 4


def convert_hub_dataset(
    dataset_name: str,
    dataset_config_name: str,
    output_dir: Path,
    cache_dir: str | None,
    max_episodes: int | None,
    force: bool,
    action_carrier: str,
    image_sequence_length: int,
    context_images_output_column: str,
    batch_size: int,
    *,
    dataset_source_prefix: str = "zero-latency/defend-the-line",
    dataset_revision: str = "4d8be25dc6fccfe244c2fa6be4b771ee448f2c2a",
) -> dict[str, object]:
    return history_converter.convert_hub_dataset(
        dataset_name,
        dataset_config_name,
        output_dir,
        dataset_source_subdir=f"{dataset_source_prefix}/{dataset_config_name}",
        dataset_revision=dataset_revision,
        source_env_name="defend_the_line",
        display_name="Defend the Line",
        base_converter=defend_converter.base,
        cache_dir=cache_dir,
        max_episodes=max_episodes,
        force=force,
        action_carrier=action_carrier,
        image_sequence_length=image_sequence_length,
        context_images_output_column=context_images_output_column,
        batch_size=batch_size,
        source_observation_fps=SOURCE_OBSERVATION_FPS,
        source_env_fps=SOURCE_ENV_FPS,
        source_env_frameskip=SOURCE_ENV_FRAMESKIP,
        conversion_context=defend_converter._defend_the_line_constants,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert row-history Defend the Line rollouts into WanOFT context-image LeRobot format."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config-name", default=DEFAULT_DATASET_CONFIG_NAME)
    parser.add_argument("--dataset-source-prefix", default="zero-latency/defend-the-line")
    parser.add_argument("--dataset-revision", default="4d8be25dc6fccfe244c2fa6be4b771ee448f2c2a")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--action-carrier", choices=["native", "bridge"], default="bridge")
    parser.add_argument("--image-sequence-length", type=int, default=5)
    parser.add_argument(
        "--context-images-output-column",
        default=defend_converter.base.DEFAULT_CONTEXT_IMAGES_OUTPUT_COLUMN,
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    manifest = convert_hub_dataset(
        args.dataset_name,
        args.dataset_config_name,
        args.output_dir,
        args.cache_dir,
        args.max_episodes,
        args.force,
        args.action_carrier,
        args.image_sequence_length,
        args.context_images_output_column,
        args.batch_size,
        dataset_source_prefix=args.dataset_source_prefix,
        dataset_revision=args.dataset_revision,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
