def should_run_step_interval_event(completed_steps: int, interval: int, gradients_synced: bool) -> bool:
    if interval <= 0:
        raise ValueError(f"Step interval must be positive, got {interval}.")
    return gradients_synced and completed_steps > 0 and completed_steps % interval == 0


def should_run_optional_step_interval_event(completed_steps: int, interval: int, gradients_synced: bool) -> bool:
    if interval <= 0:
        return False
    return should_run_step_interval_event(
        completed_steps=completed_steps,
        interval=interval,
        gradients_synced=gradients_synced,
    )


def calculate_epoch_progress(completed_steps: int, total_batch_size: int, dataset_size: int) -> float:
    if completed_steps < 0:
        raise ValueError(f"completed_steps must be non-negative, got {completed_steps}.")
    if total_batch_size <= 0:
        raise ValueError(f"total_batch_size must be positive, got {total_batch_size}.")
    if dataset_size <= 0:
        raise ValueError(f"dataset_size must be positive, got {dataset_size}.")
    return completed_steps * total_batch_size / dataset_size
