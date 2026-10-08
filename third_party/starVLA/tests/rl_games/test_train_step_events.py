import pytest

from starVLA.training.train_step_events import (
    calculate_epoch_progress,
    should_run_optional_step_interval_event,
    should_run_step_interval_event,
)


def test_step_interval_event_waits_for_gradient_sync() -> None:
    assert not should_run_step_interval_event(completed_steps=100, interval=100, gradients_synced=False)


def test_step_interval_event_runs_on_synced_interval_step() -> None:
    assert should_run_step_interval_event(completed_steps=100, interval=100, gradients_synced=True)


def test_step_interval_event_skips_zero_and_non_interval_steps() -> None:
    assert not should_run_step_interval_event(completed_steps=0, interval=100, gradients_synced=True)
    assert not should_run_step_interval_event(completed_steps=99, interval=100, gradients_synced=True)


def test_optional_step_interval_event_treats_non_positive_interval_as_disabled() -> None:
    assert not should_run_optional_step_interval_event(completed_steps=100, interval=0, gradients_synced=True)
    assert not should_run_optional_step_interval_event(completed_steps=100, interval=-1, gradients_synced=True)


def test_epoch_progress_uses_effective_global_batch_size() -> None:
    epoch_progress = calculate_epoch_progress(
        completed_steps=2500,
        total_batch_size=512,
        dataset_size=369008,
    )

    assert epoch_progress == pytest.approx(3.4688, abs=0.0001)


def test_epoch_progress_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="dataset_size must be positive"):
        calculate_epoch_progress(completed_steps=1, total_batch_size=512, dataset_size=0)
