from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from itertools import pairwise
from pathlib import Path
from typing import TypedDict

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from starVLA.training.rl_games.temporal_clip import decode_context_image_sequence

CURRENT_ACTION_LABEL = "current_action"
TIME_TO_NEXT_FLAP_LABEL = "time_to_next_flap"
TIME_SINCE_LAST_FLAP_LABEL = "time_since_last_flap"
LATENCY_ID_LABEL = "latency_id"

PROBE_LABEL_NAMES = (
    CURRENT_ACTION_LABEL,
    TIME_TO_NEXT_FLAP_LABEL,
    TIME_SINCE_LAST_FLAP_LABEL,
    LATENCY_ID_LABEL,
)

CONTEXT_IMAGES_COLUMN = "observation.context_images"
CURRENT_IMAGE_COLUMN = "observation.image"
STATE_COLUMN = "observation.state"
LATENCY_PROMPT_MARKER = " Current action latency is "

REQUIRED_COLUMNS = (
    CONTEXT_IMAGES_COLUMN,
    CURRENT_IMAGE_COLUMN,
    STATE_COLUMN,
    "episode_index",
    "frame_index",
    "decision_step",
    "task_index",
    "action_id",
    "latency",
)


class ProbeLabels(TypedDict):
    current_action: int
    time_to_next_flap: int
    time_since_last_flap: int
    latency_id: int


class ProbeExample(TypedDict):
    frames: tuple[Image.Image, ...]
    prompt: str
    state: np.ndarray
    episode_index: int
    frame_index: int
    decision_step: int
    labels: ProbeLabels


def select_episode_paths(dataset_dir: Path, max_episodes: int, seed: int) -> list[Path]:
    if max_episodes <= 0:
        raise ValueError(f"max_episodes must be positive, got {max_episodes}")
    episode_paths = sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No LeRobot episode parquet files found under {dataset_dir}")
    if max_episodes > len(episode_paths):
        raise ValueError(
            f"Requested max_episodes={max_episodes}, but dataset {dataset_dir} contains only "
            f"{len(episode_paths)} episodes"
        )
    rng = np.random.default_rng(seed)
    selected_indices = sorted(int(index) for index in rng.choice(len(episode_paths), size=max_episodes, replace=False))
    return [episode_paths[index] for index in selected_indices]


def load_task_prompts(dataset_dir: Path) -> dict[int, str]:
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot task metadata: {tasks_path}")
    prompts: dict[int, str] = {}
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if "task_index" not in row or "task" not in row:
                raise ValueError(f"Invalid task row at {tasks_path}:{line_number}: {row}")
            task_index = int(row["task_index"])
            if task_index in prompts:
                raise ValueError(f"Duplicate task_index={task_index} in {tasks_path}")
            prompts[task_index] = str(row["task"])
    if not prompts:
        raise ValueError(f"No task prompts found in {tasks_path}")
    return prompts


def bucket_temporal_distance(distance: int | None, maximum_exact_distance: int) -> int:
    if maximum_exact_distance <= 0:
        raise ValueError(f"maximum_exact_distance must be positive, got {maximum_exact_distance}")
    if distance is None:
        return maximum_exact_distance
    if distance < 0:
        raise ValueError(f"Temporal distance must be non-negative, got {distance}")
    return min(int(distance), maximum_exact_distance)


def build_flap_timing_labels(
    decision_steps: Sequence[int],
    action_ids: Sequence[int],
    flap_action_id: int,
    maximum_exact_distance: int,
) -> tuple[list[int], list[int]]:
    if len(decision_steps) != len(action_ids):
        raise ValueError(
            f"decision_steps and action_ids must have equal length, got {len(decision_steps)} and {len(action_ids)}"
        )
    if not decision_steps:
        raise ValueError("Cannot build timing labels for an empty episode")
    normalized_steps = [int(step) for step in decision_steps]
    if any(right <= left for left, right in pairwise(normalized_steps)):
        raise ValueError(f"decision_steps must be strictly increasing, got {normalized_steps}")

    next_distances: list[int] = [maximum_exact_distance] * len(normalized_steps)
    next_flap_step: int | None = None
    for index in range(len(normalized_steps) - 1, -1, -1):
        if int(action_ids[index]) == flap_action_id:
            next_flap_step = normalized_steps[index]
        distance = None if next_flap_step is None else next_flap_step - normalized_steps[index]
        next_distances[index] = bucket_temporal_distance(distance, maximum_exact_distance)

    previous_distances: list[int] = [maximum_exact_distance] * len(normalized_steps)
    previous_flap_step: int | None = None
    for index, decision_step in enumerate(normalized_steps):
        if int(action_ids[index]) == flap_action_id:
            previous_flap_step = decision_step
        distance = None if previous_flap_step is None else decision_step - previous_flap_step
        previous_distances[index] = bucket_temporal_distance(distance, maximum_exact_distance)

    return next_distances, previous_distances


def _episode_rows(episode_path: Path) -> list[dict[str, object]]:
    available_columns = set(pq.read_schema(episode_path).names)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - available_columns)
    if missing_columns:
        raise ValueError(
            f"Probe dataset episode {episode_path} is missing required columns {missing_columns}. "
            "Re-run the Flappy conversion with the probing-aware converter."
        )
    rows = pq.read_table(episode_path, columns=list(REQUIRED_COLUMNS)).to_pylist()
    if not rows:
        raise ValueError(f"Probe dataset episode is empty: {episode_path}")
    return rows


def iter_episode_examples(
    episode_path: Path,
    task_prompts: dict[int, str],
    image_sequence_length: int,
    maximum_exact_distance: int,
    flap_action_id: int,
) -> Iterator[ProbeExample]:
    if image_sequence_length != 5:
        raise ValueError(f"WanOFT temporal probing requires image_sequence_length=5, got {image_sequence_length}")
    rows = _episode_rows(episode_path)
    episode_indices = {int(row["episode_index"]) for row in rows}
    if len(episode_indices) != 1:
        raise ValueError(f"Expected one episode_index in {episode_path}, got {sorted(episode_indices)}")

    decision_steps = [int(row["decision_step"]) for row in rows]
    action_ids = [int(row["action_id"]) for row in rows]
    next_flap, since_flap = build_flap_timing_labels(
        decision_steps=decision_steps,
        action_ids=action_ids,
        flap_action_id=flap_action_id,
        maximum_exact_distance=maximum_exact_distance,
    )

    for row_index, row in enumerate(rows):
        task_index = int(row["task_index"])
        if task_index not in task_prompts:
            raise ValueError(f"Episode {episode_path} references missing task_index={task_index}")
        frame_array = decode_context_image_sequence(
            context_entry=row[CONTEXT_IMAGES_COLUMN],
            current_entry=row[CURRENT_IMAGE_COLUMN],
            image_sequence_length=image_sequence_length,
            dataset_path=episode_path.parent.parent.parent,
        )
        frames = tuple(Image.fromarray(frame).convert("RGB") for frame in frame_array)
        state = np.asarray(row[STATE_COLUMN], dtype=np.float32)
        if state.ndim != 1:
            raise ValueError(f"Expected one-dimensional state in {episode_path}, got shape={state.shape}")
        action_id = action_ids[row_index]
        if action_id not in (0, flap_action_id):
            raise ValueError(f"Flappy action_id must be 0 or {flap_action_id}, got {action_id} in {episode_path}")
        yield ProbeExample(
            frames=frames,
            prompt=task_prompts[task_index],
            state=state[None, :],
            episode_index=int(row["episode_index"]),
            frame_index=int(row["frame_index"]),
            decision_step=decision_steps[row_index],
            labels=ProbeLabels(
                current_action=action_id,
                time_to_next_flap=next_flap[row_index],
                time_since_last_flap=since_flap[row_index],
                latency_id=int(row["latency"]),
            ),
        )


def iter_probe_batches(
    episode_paths: Sequence[Path],
    task_prompts: dict[int, str],
    image_sequence_length: int,
    maximum_exact_distance: int,
    flap_action_id: int,
    batch_size: int,
) -> Iterator[list[ProbeExample]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    batch: list[ProbeExample] = []
    for episode_path in episode_paths:
        for example in iter_episode_examples(
            episode_path=episode_path,
            task_prompts=task_prompts,
            image_sequence_length=image_sequence_length,
            maximum_exact_distance=maximum_exact_distance,
            flap_action_id=flap_action_id,
        ):
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def normal_examples(examples: Sequence[ProbeExample], seed: int) -> list[ProbeExample]:
    del seed
    return [ProbeExample(**example) for example in examples]


def shuffled_frame_examples(examples: Sequence[ProbeExample], seed: int) -> list[ProbeExample]:
    transformed: list[ProbeExample] = []
    for example in examples:
        sample_seed = np.random.SeedSequence([seed, example["episode_index"], example["frame_index"]])
        permutation = np.random.default_rng(sample_seed).permutation(len(example["frames"]))
        frames = tuple(example["frames"][int(index)] for index in permutation)
        transformed.append(ProbeExample(**{**example, "frames": frames}))
    return transformed


def repeated_last_frame_examples(examples: Sequence[ProbeExample], seed: int) -> list[ProbeExample]:
    del seed
    transformed: list[ProbeExample] = []
    for example in examples:
        last_frame = example["frames"][-1]
        frames = tuple(last_frame for _ in example["frames"])
        transformed.append(ProbeExample(**{**example, "frames": frames}))
    return transformed


def latency_neutral_prompt(prompt: str) -> str:
    head, marker, _tail = prompt.partition(LATENCY_PROMPT_MARKER)
    return head.rstrip() if marker else prompt


def latency_neutral_prompt_examples(examples: Sequence[ProbeExample], seed: int) -> list[ProbeExample]:
    del seed
    return [
        ProbeExample(**{**example, "prompt": latency_neutral_prompt(example["prompt"])})
        for example in examples
    ]
