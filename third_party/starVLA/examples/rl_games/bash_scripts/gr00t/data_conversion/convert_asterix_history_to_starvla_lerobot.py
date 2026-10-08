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
    convert_asterix_to_starvla_lerobot as asterix_converter,
    convert_atari_history_to_starvla_lerobot as history_converter,
)


DEFAULT_DATASET_NAME = "latency-sensitive-bench/benchmark-datasets"
SOURCE_OBSERVATION_FPS = 15.0
SOURCE_ENV_FPS = 60.0
SOURCE_ENV_FRAMESKIP = 4


def convert_hub_dataset(
    dataset_name: str,
    dataset_config_name: str,
    output_dir: Path,
    cache_dir: str | None,
    max_episodes: int | None,
    force: bool,
    action_carrier: str,
    action_layout: str | None,
    image_sequence_length: int,
    context_images_output_column: str,
    batch_size: int,
    *,
    dataset_source_prefix: str,
    dataset_revision: str,
) -> dict[str, object]:
    normalized_action_layout = asterix_converter._normalize_action_layout(
        action_layout,
        action_carrier,
    )
    return history_converter.convert_hub_dataset(
        dataset_name,
        dataset_config_name,
        output_dir,
        dataset_source_subdir=f"{dataset_source_prefix}/{dataset_config_name}",
        dataset_revision=dataset_revision,
        source_env_name="asterix",
        display_name="Asterix",
        base_converter=asterix_converter.base,
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
        conversion_context=lambda: asterix_converter._asterix_constants(normalized_action_layout),
        action_layout=normalized_action_layout,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert row-history Asterix rollouts into WanOFT context-image LeRobot format."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config-name", required=True)
    parser.add_argument("--dataset-source-prefix", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--action-carrier", choices=["native", "bridge"], default="bridge")
    parser.add_argument("--action-layout", default="factorized_6")
    parser.add_argument("--image-sequence-length", type=int, default=5)
    parser.add_argument(
        "--context-images-output-column",
        default=asterix_converter.base.DEFAULT_CONTEXT_IMAGES_OUTPUT_COLUMN,
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
        args.action_layout,
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
