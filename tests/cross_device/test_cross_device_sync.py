from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from latency_bench.cross_device import protocol, sync


class _DownloadApi:
    def __init__(self, existing: set[str]) -> None:
        self.existing = existing

    def file_exists(self, *, filename: str, **_kwargs: object) -> bool:
        return filename in self.existing


def test_download_raw_selection_stages_before_replacing_exact_local_leaves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = Path("gpu/instance/workload")
    remote = Path("raw") / target
    snapshot = tmp_path / "snapshot"
    source = snapshot / remote
    source.mkdir(parents=True)
    (source / "calibration.json").write_text("new calibration", encoding="utf-8")
    (source / "hardware.json").write_text("new hardware", encoding="utf-8")
    (source / "calibration.DONE").touch()
    session = source / "sessions/0"
    session.mkdir(parents=True)
    (session / "result.jsonl").write_text("new result", encoding="utf-8")
    (session / "DONE").touch()

    local = tmp_path / "local/raw" / target
    (local / "sessions/0").mkdir(parents=True)
    (local / "sessions/0/stale.txt").write_text("stale", encoding="utf-8")
    (local / "sessions/0/DONE").touch()
    (local / "sessions/7").mkdir(parents=True)
    (local / "sessions/7/keep.txt").write_text("keep", encoding="utf-8")
    (local / "calibration.json").write_text("old calibration", encoding="utf-8")
    (local / "hardware.json").write_text("old hardware", encoding="utf-8")
    (local / "calibration.DONE").touch()

    markers = {
        f"{remote.as_posix()}/calibration.DONE",
        f"{remote.as_posix()}/sessions/0/DONE",
    }
    api = _DownloadApi(markers)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: api, snapshot_download=lambda **_kwargs: str(snapshot)),
    )

    sync.download_raw_selection(
        {"hf_repo": "org/repo", "hf_repo_type": "dataset"},
        target,
        local,
        [0],
        overwrite=True,
    )

    assert (local / "calibration.json").read_text(encoding="utf-8") == "new calibration"
    assert (local / "sessions/0/result.jsonl").read_text(encoding="utf-8") == "new result"
    assert not (local / "sessions/0/stale.txt").exists()
    assert (local / "sessions/7/keep.txt").read_text(encoding="utf-8") == "keep"


def test_download_raw_selection_does_not_replace_when_remote_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = Path("gpu/instance/workload")
    remote = Path("raw") / target
    local = tmp_path / "raw" / target
    local.mkdir(parents=True)
    (local / "calibration.json").write_text("original", encoding="utf-8")
    api = _DownloadApi({f"{remote.as_posix()}/calibration.DONE"})
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: api, snapshot_download=lambda **_kwargs: "unused"),
    )

    with pytest.raises(FileNotFoundError, match="sessions/0/DONE"):
        sync.download_raw_selection(
            {"hf_repo": "org/repo", "hf_repo_type": "dataset"},
            target,
            local,
            [0],
            overwrite=True,
        )

    assert (local / "calibration.json").read_text(encoding="utf-8") == "original"


def test_publish_overwrite_deletes_only_leaf_payload_then_uploads_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class Api:
        def file_exists(self, **_kwargs: object) -> bool:
            raise AssertionError("overwrite must not use append-only DONE detection")

        def upload_folder(self, **kwargs: object) -> None:
            calls.append(("folder", kwargs))

        def upload_file(self, **kwargs: object) -> None:
            calls.append(("file", kwargs))

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api))
    leaf = tmp_path / "profile"
    leaf.mkdir()
    (leaf / "profile.json").write_text("{}", encoding="utf-8")
    (leaf / "DONE").touch()

    sync.publish_directory_leaf(
        {"hf_repo": "org/repo", "hf_repo_type": "dataset"},
        leaf,
        Path("profiles/gpu/instance/workload/fit-0"),
        overwrite=True,
    )

    assert [name for name, _ in calls] == ["folder", "file"]
    assert calls[0][1]["path_in_repo"] == "profiles/gpu/instance/workload/fit-0"
    assert calls[0][1]["ignore_patterns"] == ["DONE"]
    assert calls[0][1]["delete_patterns"] == ["**"]
    assert calls[1][1]["path_in_repo"].endswith("/DONE")


def test_protocol_dispatches_raw_download_overwrite(monkeypatch: pytest.MonkeyPatch) -> None:
    workload = {"artifact_root": "results/cross_device"}
    calls = []
    monkeypatch.setattr(protocol, "load_workload", lambda _path: workload)
    monkeypatch.setattr(
        protocol,
        "download_raw",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    protocol.main(
        [
            "download",
            "raw",
            "--target-ref",
            "gpu/instance/openvla-flappy-isolated-30s",
            "--sessions",
            "0",
            "--overwrite",
        ]
    )

    assert calls == [
        (
            (
                workload,
                Path("gpu/instance/openvla-flappy-isolated-30s"),
                [0],
            ),
            {"overwrite": True},
        )
    ]


def test_protocol_dispatches_profile_publish_overwrite(monkeypatch: pytest.MonkeyPatch) -> None:
    workload = {"artifact_root": "results/cross_device"}
    calls = []
    monkeypatch.setattr(protocol, "load_workload", lambda _path: workload)
    monkeypatch.setattr(
        protocol,
        "publish_profile",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    protocol.main(
        [
            "publish",
            "profile",
            "--profile-ref",
            "gpu/instance/openvla-flappy-isolated-30s/fit-0",
            "--overwrite",
        ]
    )

    assert calls == [
        (
            (
                workload,
                Path("gpu/instance/openvla-flappy-isolated-30s/fit-0"),
            ),
            {"resume": False, "overwrite": True},
        )
    ]


def test_protocol_rejects_resume_with_overwrite() -> None:
    with pytest.raises(SystemExit):
        protocol.main(
            [
                "publish",
                "profile",
                "--profile-ref",
                "gpu/instance/workload/fit-0",
                "--resume",
                "--overwrite",
            ]
        )
