from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


def test_profile_cache_is_local_first(tmp_path: Path):
    from scripts.cross_device.profile_cache import ensure_profile_cached

    profile_path = tmp_path / "profiles" / "dry-policy" / "flappy" / "L_A100" / "profile.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("{}\n", encoding="utf-8")

    calls: list[tuple[str, Path, str, str]] = []

    def download_profile_folder(*, repo_id: str, local_dir: Path, remote_dir: str, repo_type: str, revision: str | None) -> None:
        calls.append((repo_id, local_dir, remote_dir, repo_type))

    resolved = ensure_profile_cached(
        profile_path=profile_path,
        profiles_dir=tmp_path / "profiles",
        hf_repo="latency-sensitive-bench/profiles",
        download_profile_folder=download_profile_folder,
    )

    assert resolved == profile_path
    assert calls == []


def test_profile_cache_downloads_missing_profile(tmp_path: Path):
    from scripts.cross_device.profile_cache import ensure_profile_cached

    profile_path = tmp_path / "profiles" / "dry-policy" / "flappy" / "L_A100" / "profile.json"
    calls: list[tuple[str, Path, str, str]] = []

    def download_profile_folder(*, repo_id: str, local_dir: Path, remote_dir: str, repo_type: str, revision: str | None) -> None:
        calls.append((repo_id, local_dir, remote_dir, repo_type))
        local_dir.mkdir(parents=True)
        (local_dir / "profile.json").write_text("{}\n", encoding="utf-8")

    resolved = ensure_profile_cached(
        profile_path=profile_path,
        profiles_dir=tmp_path / "profiles",
        hf_repo="latency-sensitive-bench/profiles",
        download_profile_folder=download_profile_folder,
    )

    assert resolved == profile_path
    assert profile_path.exists()
    assert calls == [
        (
            "latency-sensitive-bench/profiles",
            profile_path.parent,
            "dry-policy/flappy/L_A100",
            "dataset",
        )
    ]


def test_profile_cache_downloads_whole_profile_folder_from_hf(tmp_path: Path, monkeypatch):
    from scripts.cross_device.profile_cache import download_profile_folder_from_hf

    remote_dir = "dry-policy/flappy/L_A100"
    snapshot_root = tmp_path / "snapshot"
    remote_profile_dir = snapshot_root / remote_dir
    remote_profile_dir.mkdir(parents=True)
    (remote_profile_dir / "profile.json").write_text("{}\n", encoding="utf-8")
    (remote_profile_dir / "latency_distribution.json").write_text("{}\n", encoding="utf-8")
    (remote_profile_dir / "future_artifact.json").write_text("{}\n", encoding="utf-8")

    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot_root)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))

    local_dir = tmp_path / "profiles" / remote_dir
    download_profile_folder_from_hf(
        repo_id="latency-sensitive-bench/profiles",
        local_dir=local_dir,
        remote_dir=remote_dir,
        repo_type="dataset",
        revision="main",
    )

    assert calls == [
        {
            "repo_id": "latency-sensitive-bench/profiles",
            "repo_type": "dataset",
            "revision": "main",
            "allow_patterns": "dry-policy/flappy/L_A100/*",
        }
    ]
    assert (local_dir / "profile.json").exists()
    assert (local_dir / "latency_distribution.json").exists()
    assert (local_dir / "future_artifact.json").exists()
