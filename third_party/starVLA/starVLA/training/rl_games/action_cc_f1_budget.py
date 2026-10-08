from collections.abc import Sequence

EpisodeFrames = list[tuple[int, int]]
EpisodeRecord = tuple[str, int, str, int | None, EpisodeFrames]
FrameRecord = tuple[int, int, str, int, str, int | None]


def _validate_budget_inputs(
    num_procs: int,
    rank: int,
    shared_frame_budget: int,
    per_latency_frame_budget: int | None,
) -> None:
    if num_procs <= 0:
        raise ValueError(f"num_procs must be positive, got {num_procs}.")
    if rank < 0 or rank >= num_procs:
        raise ValueError(f"rank must be in [0, {num_procs}), got {rank}.")
    if shared_frame_budget <= 0:
        raise ValueError(f"shared_frame_budget must be positive, got {shared_frame_budget}.")
    if per_latency_frame_budget is not None and per_latency_frame_budget <= 0:
        raise ValueError(f"per_latency_frame_budget must be positive when set, got {per_latency_frame_budget}.")


def _append_frame_records(
    frame_records: list[FrameRecord],
    ep_task: str,
    dataset_index: int,
    episode_key: str,
    episode_latency_value: int | None,
    frames: Sequence[tuple[int, int]],
) -> None:
    for flat_idx, base_idx in frames:
        frame_records.append(
            (
                int(dataset_index),
                int(flat_idx),
                str(episode_key),
                int(base_idx),
                str(ep_task),
                episode_latency_value,
            )
        )


def _expected_latency_values(episodes: Sequence[EpisodeRecord]) -> list[int]:
    return sorted({int(ep[3]) for ep in episodes if ep[3] is not None})


def _expected_task_latency_values(episodes: Sequence[EpisodeRecord]) -> list[tuple[str, int]]:
    return sorted({(str(ep[0]), int(ep[3])) for ep in episodes if ep[3] is not None})


def build_frame_records(
    episodes: Sequence[EpisodeRecord],
    num_procs: int,
    rank: int,
    shared_frame_budget: int,
    per_latency_frame_budget: int | None,
    cross_task_mode: bool,
) -> list[FrameRecord]:
    _validate_budget_inputs(
        num_procs=num_procs,
        rank=rank,
        shared_frame_budget=shared_frame_budget,
        per_latency_frame_budget=per_latency_frame_budget,
    )
    expected_latencies = _expected_latency_values(episodes)
    expected_task_latencies = _expected_task_latency_values(episodes) if cross_task_mode else []
    use_per_latency_budget = per_latency_frame_budget is not None and bool(expected_latencies)

    frame_records: list[FrameRecord] = []
    shared_remaining = int(shared_frame_budget)
    latency_frame_counts: dict[int, int] = {}
    task_latency_frame_counts: dict[tuple[str, int], int] = {}

    for ep_idx, episode in enumerate(episodes):
        if ep_idx % num_procs != rank:
            continue
        ep_task, dataset_index, episode_key, latency, frames = episode
        if not frames:
            continue

        if not use_per_latency_budget:
            if shared_remaining <= 0:
                break
            selected_frames = frames[:shared_remaining]
            _append_frame_records(
                frame_records=frame_records,
                ep_task=ep_task,
                dataset_index=dataset_index,
                episode_key=episode_key,
                episode_latency_value=latency,
                frames=selected_frames,
            )
            shared_remaining -= len(selected_frames)
            if shared_remaining <= 0:
                break
            continue

        if latency is None:
            raise ValueError(
                "per_latency_frame_budget requires every assigned CC-F1 episode to have a latency value. "
                f"Missing latency for episode_key={episode_key!r}, task={ep_task!r}."
            )

        latency_value = int(latency)
        if cross_task_mode:
            task_latency_key = (str(ep_task), latency_value)
            remaining = int(per_latency_frame_budget) - task_latency_frame_counts.get(task_latency_key, 0)
        else:
            task_latency_key = None
            remaining = int(per_latency_frame_budget) - latency_frame_counts.get(latency_value, 0)
        if remaining <= 0:
            continue

        selected_frames = frames[:remaining]
        _append_frame_records(
            frame_records=frame_records,
            ep_task=ep_task,
            dataset_index=dataset_index,
            episode_key=episode_key,
            episode_latency_value=latency_value,
            frames=selected_frames,
        )

        if task_latency_key is not None:
            task_latency_frame_counts[task_latency_key] = task_latency_frame_counts.get(task_latency_key, 0) + len(
                selected_frames
            )
            all_task_latencies_reached = all(
                task_latency_frame_counts.get(value, 0) >= int(per_latency_frame_budget)
                for value in expected_task_latencies
            )
            if all_task_latencies_reached:
                break
        else:
            latency_frame_counts[latency_value] = latency_frame_counts.get(latency_value, 0) + len(selected_frames)
            all_latencies_reached = all(
                latency_frame_counts.get(value, 0) >= int(per_latency_frame_budget)
                for value in expected_latencies
            )
            if all_latencies_reached:
                break

    return frame_records
