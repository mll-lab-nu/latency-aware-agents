"""Materialize selected decisions and their image context for one episode."""
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from latency_bench.data.rollout_image_io import RolloutImageWriter

def episode_image_dir(output_dir: Path, split: str, episode_idx: int) -> Path:
    """Build the image directory for one rollout episode.

    Args:
        output_dir: Rollout export directory.
        split: Row split or attempt namespace.
        episode_idx: Episode or attempt index.

    Returns:
        Directory containing decision-step PNG files.
    """
    return output_dir / "images" / split / f"episode_{episode_idx:06d}"


def finalize_attempt_rows(
    rows: list[dict[str, Any]],
    *,
    final_dir: Path,
    episode_idx: int,
    split: str,
) -> list[dict[str, Any]]:
    """Rewrite accepted attempt rows to their final dataset identity.

    Args:
        rows: Rows collected under the attempt namespace.
        final_dir: Final split image directory.
        episode_idx: Final accepted episode index.
        split: Final dataset split.

    Returns:
        Final row copies with updated split, episode index, and image path.
    """
    finalized_rows = []
    for row in rows:
        finalized = dict(row)
        finalized["episode_idx"] = episode_idx
        finalized["split"] = split
        finalized["image"] = str(final_dir / f"decision_step_{int(row['decision_step']):06d}.png")
        finalized_rows.append(finalized)
    return finalized_rows


def _context_decision_steps(
    decision_step: int,
    context_window: int,
    available_decision_steps: Sequence[int],
) -> list[int]:
    """Build historical decision steps from the current contiguous replay segment.

    Args:
        decision_step: Current decision step.
        context_window: Total image window length. Values 0 and 1 keep only the
            current ``image``; values above 1 add historical ``context_images``.
        available_decision_steps: Materialized decision steps ordered from oldest to newest.

    Returns:
        Historical decision steps ordered from oldest to newest.
    """
    current_index = available_decision_steps.index(decision_step)
    segment_start_index = current_index
    while segment_start_index > 0:
        current_step = available_decision_steps[segment_start_index]
        previous_step = available_decision_steps[segment_start_index - 1]
        if previous_step != current_step - 1:
            break
        segment_start_index -= 1

    context_steps: list[int] = []
    cursor = current_index - 1
    while cursor >= segment_start_index and len(context_steps) < context_window - 1:
        context_steps.append(available_decision_steps[cursor])
        cursor -= 1

    first_segment_step = available_decision_steps[segment_start_index]
    while len(context_steps) < context_window - 1:
        context_steps.append(first_segment_step)

    return list(reversed(context_steps))


def _image_path_for_decision_step(row: Mapping[str, Any], decision_step: int) -> Path:
    return Path(str(row["image"])).with_name(f"decision_step_{int(decision_step):06d}.png")


def write_filtered_episode_images(
    image_writer: RolloutImageWriter,
    rows: list[dict[str, Any]],
    frames_by_decision_step: Mapping[int, np.ndarray],
    *,
    context_window: int,
) -> list[dict[str, Any]]:
    """Write current and optional historical images for filtered rollout rows.

    Args:
        image_writer: Image writer shared by the rollout run.
        rows: Final rows selected for materialization.
        frames_by_decision_step: Captured frames keyed by decision step.
        context_window: Total image window length. Values 0 and 1 keep only the
            current ``image``; values above 1 add historical ``context_images``.

    Returns:
        Row copies with image paths written to disk.
    """
    if not rows:
        return []
    available_decision_steps = sorted(frames_by_decision_step)
    required_steps: set[int] = set()
    for row in rows:
        decision_step = int(row["decision_step"])
        required_steps.add(decision_step)
        if int(context_window) > 1:
            required_steps.update(
                _context_decision_steps(
                    decision_step,
                    int(context_window),
                    available_decision_steps,
                )
            )

    written_paths_by_step: dict[int, str] = {}
    first_row = rows[0]
    for decision_step in sorted(required_steps):
        image_path = _image_path_for_decision_step(first_row, decision_step)
        written_paths_by_step[decision_step] = image_writer.write(frames_by_decision_step[decision_step], image_path)
    image_writer.flush()

    written_rows = []
    for row in rows:
        written = dict(row)
        decision_step = int(written["decision_step"])
        written["image"] = written_paths_by_step[decision_step]
        if int(context_window) > 1:
            written["context_images"] = [
                written_paths_by_step[context_step]
                for context_step in _context_decision_steps(
                    decision_step,
                    int(context_window),
                    available_decision_steps,
                )
            ]
        written_rows.append(written)
    return written_rows
