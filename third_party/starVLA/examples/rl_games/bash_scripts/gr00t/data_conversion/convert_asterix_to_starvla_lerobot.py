#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from examples.rl_games.bash_scripts.gr00t.data_conversion import convert_demon_attack_to_starvla_lerobot as base
from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_flappy_dataset import source_timing_from_args


ACTION_LABELS = [
    "noop",
    "up",
    "right",
    "left",
    "down",
    "upright",
    "upleft",
    "downright",
    "downleft",
]

ACTION_LAYOUT_DISCRETE_9 = "discrete_9"
ACTION_LAYOUT_FACTORIZED_6 = "factorized_6"

FACTORIZED_6_ACTION_LABELS = [
    "vertical_none",
    "vertical_up",
    "vertical_down",
    "horizontal_none",
    "horizontal_right",
    "horizontal_left",
]

FACTORIZED_6_BY_ACTION_ID = {
    0: (0, 3),  # noop
    1: (1, 3),  # up
    2: (0, 4),  # right
    3: (0, 5),  # left
    4: (2, 3),  # down
    5: (1, 4),  # upright
    6: (1, 5),  # upleft
    7: (2, 4),  # downright
    8: (2, 5),  # downleft
}


def _normalize_action_layout(action_layout: str | None, action_carrier: str) -> str:
    layout = str(action_layout or "").strip().lower()
    if not layout:
        return ACTION_LAYOUT_FACTORIZED_6 if action_carrier == "bridge" else ACTION_LAYOUT_DISCRETE_9
    aliases = {
        "discrete_9": ACTION_LAYOUT_DISCRETE_9,
        "joint_9": ACTION_LAYOUT_DISCRETE_9,
        "asterix_discrete_9": ACTION_LAYOUT_DISCRETE_9,
        "factorized_6": ACTION_LAYOUT_FACTORIZED_6,
        "factorized6": ACTION_LAYOUT_FACTORIZED_6,
        "asterix_factorized_6": ACTION_LAYOUT_FACTORIZED_6,
        "asterix_factorized6": ACTION_LAYOUT_FACTORIZED_6,
    }
    if layout not in aliases:
        supported = "|".join(sorted(aliases))
        raise ValueError(f"Unsupported Asterix action_layout={action_layout!r}; expected one of: {supported}")
    normalized = aliases[layout]
    if action_carrier == "bridge" and normalized != ACTION_LAYOUT_FACTORIZED_6:
        raise ValueError("Asterix Bridge requires action_layout=factorized_6 so it fits the 7D carrier.")
    return normalized


def _factorized_one_hot(action_id: int, action_dim: int | None = None) -> list[float]:
    action_id = int(action_id)
    if action_id not in FACTORIZED_6_BY_ACTION_ID:
        raise ValueError(f"Invalid Asterix action id={action_id}; expected 0..8")
    dim = int(action_dim if action_dim is not None else len(FACTORIZED_6_ACTION_LABELS))
    if dim < len(FACTORIZED_6_ACTION_LABELS):
        raise ValueError(f"Asterix factorized action_dim must be >= 6, got {dim}")
    values = [0.0] * dim
    vertical_idx, horizontal_idx = FACTORIZED_6_BY_ACTION_ID[action_id]
    values[vertical_idx] = 1.0
    values[horizontal_idx] = 1.0
    return values


@contextmanager
def _asterix_constants(action_layout: str) -> Iterator[None]:
    old_action_labels = base.ACTION_LABELS
    old_action_dim = base.ACTION_DIM
    old_one_hot = base._one_hot
    try:
        if action_layout == ACTION_LAYOUT_FACTORIZED_6:
            base.ACTION_LABELS = list(FACTORIZED_6_ACTION_LABELS)
            base.ACTION_DIM = len(FACTORIZED_6_ACTION_LABELS)
            base._one_hot = _factorized_one_hot
        else:
            base.ACTION_LABELS = list(ACTION_LABELS)
            base.ACTION_DIM = len(ACTION_LABELS)
        yield
    finally:
        base.ACTION_LABELS = old_action_labels
        base.ACTION_DIM = old_action_dim
        base._one_hot = old_one_hot


def _annotate_manifest(manifest_path: Path, action_layout: str) -> None:
    if not manifest_path.exists():
        return
    manifest = base.json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["action_layout"] = action_layout
    manifest_path.write_text(base.json.dumps(manifest, indent=2), encoding="utf-8")


def convert_dataset(*args, action_layout: str | None = None, **kwargs):
    action_carrier = str(kwargs.get("action_carrier", "native")).strip().lower()
    action_layout = _normalize_action_layout(action_layout, action_carrier)
    output_dir = Path(args[1] if len(args) >= 2 else kwargs["output_dir"])
    with _asterix_constants(action_layout):
        manifest = base.convert_dataset(*args, **kwargs)
    manifest["action_layout"] = action_layout
    _annotate_manifest(output_dir / "manifest.json", action_layout)
    validation_dataset_name = manifest.get("validation_dataset_name")
    if validation_dataset_name:
        _annotate_manifest(output_dir.with_name(str(validation_dataset_name)) / "manifest.json", action_layout)
    return manifest


_BASE_CONVERT_PARAMS = list(inspect.signature(base.convert_dataset).parameters.values())
_ACTION_LAYOUT_PARAM = inspect.Parameter(
    "action_layout",
    inspect.Parameter.KEYWORD_ONLY,
    default=None,
    annotation=str | None,
)
_INSERT_AT = next(
    (idx + 1 for idx, param in enumerate(_BASE_CONVERT_PARAMS) if param.name == "action_carrier"),
    len(_BASE_CONVERT_PARAMS),
)
convert_dataset.__signature__ = inspect.Signature(
    parameters=[
        *_BASE_CONVERT_PARAMS[:_INSERT_AT],
        _ACTION_LAYOUT_PARAM,
        *_BASE_CONVERT_PARAMS[_INSERT_AT:],
    ],
    return_annotation=dict[str, object],
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", "--dataset_name", required=True)
    parser.add_argument("--dataset-config-name", "--dataset_config_name", default=None)
    parser.add_argument("--dataset-source-subdir", "--dataset_source_subdir", default=None)
    parser.add_argument("--output-dir", "--output_dir", required=True)
    parser.add_argument("--cache-dir", "--cache_dir", default=None)
    parser.add_argument("--max-episodes", "--max_episodes", type=int, default=None)
    parser.add_argument("--latency-filter", "--latency_filter", default=None)
    parser.add_argument("--episodes-per-latency", "--episodes_per_latency", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--action-carrier", "--action_carrier", choices=["native", "bridge"], default="native")
    parser.add_argument("--action-layout", "--action_layout", default=None)
    parser.add_argument("--context-images-column", "--context_images_column", default=None)
    parser.add_argument(
        "--context-images-output-column",
        "--context_images_output_column",
        default=base.DEFAULT_CONTEXT_IMAGES_OUTPUT_COLUMN,
    )
    parser.add_argument("--image-sequence-length", "--image_sequence_length", type=int, default=4)
    parser.add_argument("--source-metadata")
    parser.add_argument("--source-fps", type=float)
    parser.add_argument("--obs-stride-raw-frames", type=int)
    parser.add_argument("--source-latency-column", choices=["latency", "latency_raw_frames"])
    parser.add_argument(
        "--target-latency-unit",
        choices=["raw_frames", "observation_steps"],
        required=True,
    )
    args = parser.parse_args()
    fps, obs_stride_raw_frames, source_rows_unit = source_timing_from_args(
        source_metadata=args.source_metadata,
        source_fps=args.source_fps,
        obs_stride_raw_frames=args.obs_stride_raw_frames,
    )
    latency_filter = None
    if args.latency_filter:
        latency_filter = [int(item.strip()) for item in str(args.latency_filter).split(",") if item.strip()]

    manifest = convert_dataset(
        args.dataset_name,
        Path(args.output_dir),
        cache_dir=args.cache_dir,
        dataset_config_name=args.dataset_config_name,
        dataset_source_subdir=args.dataset_source_subdir,
        max_episodes=args.max_episodes,
        force=args.force,
        require_latency_prompt_map=False,
        latency_filter=latency_filter,
        episodes_per_latency=args.episodes_per_latency,
        action_carrier=args.action_carrier,
        action_layout=args.action_layout,
        context_images_column=args.context_images_column,
        context_images_output_column=args.context_images_output_column,
        image_sequence_length=args.image_sequence_length,
        fps=fps,
        obs_stride_raw_frames=obs_stride_raw_frames,
        source_latency_column=args.source_latency_column,
        target_latency_unit=args.target_latency_unit,
        source_rows_unit=source_rows_unit,
    )
    print(base.json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
