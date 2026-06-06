from __future__ import annotations

from http.client import RemoteDisconnected

from urllib3.exceptions import ProtocolError

from scripts.push_sf_teacher_dataset import (
    _is_retryable_hf_error,
    _run_with_hf_retry,
)


def test_remote_disconnected_protocol_error_is_retryable() -> None:
    exc = ProtocolError(
        "Connection aborted.",
        RemoteDisconnected("Remote end closed connection without response"),
    )

    assert _is_retryable_hf_error(exc)


def test_retry_repeats_after_remote_disconnect(monkeypatch) -> None:
    attempts = 0

    def operation() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ProtocolError(
                "Connection aborted.",
                RemoteDisconnected("Remote end closed connection without response"),
            )

    monkeypatch.setattr("scripts.push_sf_teacher_dataset.time.sleep", lambda _: None)

    _run_with_hf_retry(
        operation_name="push_to_hub",
        context="organization/dataset",
        operation=operation,
        attempts=2,
        initial_delay_seconds=1.0,
        max_delay_seconds=1.0,
    )

    assert attempts == 2
