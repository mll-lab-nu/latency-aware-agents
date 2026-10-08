#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DEST_REPO_ID = "latency-sensitive-bench/inverted_pendulum_200ep_2"
MIXED_SOURCE_REPO = "latency-sensitive-bench/inverted_pendulum_200ep"
OG_SOURCE_REPO = "latency-sensitive-bench/Standard-Pipeline"
OG_SOURCE_REVISION = "7d088338043f96511df41c74bc540e62d27aa417"
OG_SOURCE_SUBDIR = "inverted_pendulum/zero_latency/demonstrations/raw"

L0_DEST_SUBDIR = "inverted_pendulum_fixed_latency_0_100ep_1ksteps_og"
L2_SUBDIR = "inverted_pendulum_fixed_latency_2_1000ep_7k2steps"
L4_SUBDIR = "inverted_pendulum_fixed_latency_4_1000ep_7k2steps"


def _download_file(
    *,
    repo_id: str,
    filename: str,
    dest: Path,
    revision: str | None = None,
    token: str | None = None,
) -> None:
    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        revision=revision,
        token=token,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, dest)


def _copy_prefix(
    *,
    api: HfApi,
    repo_id: str,
    prefix: str,
    stage_dir: Path,
    revision: str | None = None,
    token: str | None = None,
) -> list[str]:
    copied: list[str] = []
    prefix = prefix.strip("/")
    for filename in api.list_repo_files(repo_id, repo_type="dataset", revision=revision, token=token):
        if not filename.startswith(f"{prefix}/"):
            continue
        _download_file(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=token,
            dest=stage_dir / filename,
        )
        copied.append(filename)
    if not copied:
        raise FileNotFoundError(f"No files found under {repo_id}/{prefix}")
    return copied


def _copy_og_l0(*, stage_dir: Path, token: str | None = None) -> list[str]:
    copied: list[str] = []
    for basename in ("train.parquet", "val.parquet", "metadata.json"):
        source = f"{OG_SOURCE_SUBDIR}/{basename}"
        dest = stage_dir / L0_DEST_SUBDIR / basename
        _download_file(
            repo_id=OG_SOURCE_REPO,
            filename=source,
            revision=OG_SOURCE_REVISION,
            token=token,
            dest=dest,
        )
        copied.append(str(dest.relative_to(stage_dir)))
    return copied


def _write_dataset_card(stage_dir: Path) -> None:
    source_subdir = ",".join([L0_DEST_SUBDIR, L2_SUBDIR, L4_SUBDIR])
    card = f"""---
configs:
- config_name: {L0_DEST_SUBDIR}
  data_files:
  - split: train
    path: {L0_DEST_SUBDIR}/train.parquet
  - split: validation
    path: {L0_DEST_SUBDIR}/val.parquet
- config_name: {L2_SUBDIR}
  data_files:
  - split: train
    path: {L2_SUBDIR}/train-*.parquet
  - split: validation
    path: {L2_SUBDIR}/val-*.parquet
- config_name: {L4_SUBDIR}
  data_files:
  - split: train
    path: {L4_SUBDIR}/train-*.parquet
  - split: validation
    path: {L4_SUBDIR}/val-*.parquet
---

# Inverted Pendulum Mixed Latency 024, OG L0

This dataset repackages inverted-pendulum rollouts for mixed-latency StarVLA/OpenVLA training.

- Latency 0 comes from `{OG_SOURCE_REPO}@{OG_SOURCE_REVISION}/{OG_SOURCE_SUBDIR}`.
- Latency 2 comes from `{MIXED_SOURCE_REPO}/{L2_SUBDIR}`.
- Latency 4 comes from `{MIXED_SOURCE_REPO}/{L4_SUBDIR}`.

Suggested StarVLA settings:

```bash
dataset.source_hf="{DEST_REPO_ID}"
dataset.source_subdir="{source_subdir}"
dataset.latency_filter=[0,2,4]
dataset.target_latency_unit=observation_steps
```
"""
    (stage_dir / "README.md").write_text(card, encoding="utf-8")


def _write_manifest(stage_dir: Path) -> None:
    manifest = {
        "repo_id": DEST_REPO_ID,
        "latency_0": {
            "source_repo": OG_SOURCE_REPO,
            "source_revision": OG_SOURCE_REVISION,
            "source_subdir": OG_SOURCE_SUBDIR,
            "dest_subdir": L0_DEST_SUBDIR,
        },
        "latency_2": {
            "source_repo": MIXED_SOURCE_REPO,
            "source_subdir": L2_SUBDIR,
            "dest_subdir": L2_SUBDIR,
        },
        "latency_4": {
            "source_repo": MIXED_SOURCE_REPO,
            "source_subdir": L4_SUBDIR,
            "dest_subdir": L4_SUBDIR,
        },
        "starvla_source_subdir": ",".join([L0_DEST_SUBDIR, L2_SUBDIR, L4_SUBDIR]),
    }
    (stage_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_dataset(stage_dir: Path, *, token: str | None = None) -> None:
    api = HfApi()
    stage_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stage] copying OG latency 0 from {OG_SOURCE_REPO}@{OG_SOURCE_REVISION}")
    for filename in _copy_og_l0(stage_dir=stage_dir, token=token):
        print(f"  + {filename}")

    for subdir in (L2_SUBDIR, L4_SUBDIR):
        print(f"[stage] copying {subdir} from {MIXED_SOURCE_REPO}")
        for filename in _copy_prefix(
            api=api,
            repo_id=MIXED_SOURCE_REPO,
            prefix=subdir,
            stage_dir=stage_dir,
            token=token,
        ):
            print(f"  + {filename}")

    _write_dataset_card(stage_dir)
    _write_manifest(stage_dir)


def upload_dataset(stage_dir: Path, *, repo_id: str, private: bool, token: str | None = None) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(stage_dir),
        commit_message="Create inverted pendulum mixed latency dataset with OG L0",
        token=token,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEST_REPO_ID)
    parser.add_argument("--stage-dir", default="/tmp/inverted_pendulum_200ep_2")
    parser.add_argument("--token", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--keep-stage", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage_dir = Path(args.stage_dir).expanduser()
    if stage_dir.exists() and not args.keep_stage:
        shutil.rmtree(stage_dir)

    build_dataset(stage_dir, token=args.token)
    print(f"[stage] ready at {stage_dir}")
    print(f"[stage] StarVLA source_subdir: {L0_DEST_SUBDIR},{L2_SUBDIR},{L4_SUBDIR}")

    if args.stage_only:
        return

    print(f"[upload] pushing to {args.repo_id}")
    upload_dataset(stage_dir, repo_id=args.repo_id, private=args.private, token=args.token)
    print(f"[done] https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
