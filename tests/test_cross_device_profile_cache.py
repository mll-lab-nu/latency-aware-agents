from __future__ import annotations

from pathlib import Path


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
