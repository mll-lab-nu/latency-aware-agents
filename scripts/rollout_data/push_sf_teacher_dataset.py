from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import datasets
import huggingface_hub
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

from latency_bench.utils.hf import resolve_token


@dataclass(frozen=True)
class DatasetPushSpec:
    export_dir: Path
    repo_id: str


UploadMode = Literal["direct", "datasets"]


def validate_export_dir(spec: DatasetPushSpec) -> None:
    missing = [
        name
        for name in ("train.parquet", "val.parquet", "metadata.json")
        if not (spec.export_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing required export files under {spec.export_dir}: {', '.join(missing)}"
        )


def _disable_progress_bars() -> None:
    datasets.disable_progress_bars()
    disable_hf_progress_bars = getattr(huggingface_hub, "disable_progress_bars", None)
    if disable_hf_progress_bars is not None:
        disable_hf_progress_bars()


def push_existing_export_direct(
    *,
    spec: DatasetPushSpec,
    private: bool,
    token: str | None,
) -> None:
    validate_export_dir(spec)
    api = HfApi()
    create_repo_kwargs = {
        "repo_id": spec.repo_id,
        "repo_type": "dataset",
        "private": private,
        "exist_ok": True,
    }
    upload_folder_kwargs = {
        "folder_path": str(spec.export_dir),
        "repo_id": spec.repo_id,
        "repo_type": "dataset",
        "allow_patterns": ["train.parquet", "val.parquet", "metadata.json"],
        "commit_message": "Upload teacher dataset export",
    }
    if token:
        create_repo_kwargs["token"] = token
        upload_folder_kwargs["token"] = token
    api.create_repo(**create_repo_kwargs)
    api.upload_folder(**upload_folder_kwargs)


def push_existing_export(
    *,
    spec: DatasetPushSpec,
    private: bool,
    token: str | None,
    max_shard_size: str,
) -> None:
    validate_export_dir(spec)
    _disable_progress_bars()
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_parquet(str(spec.export_dir / "train.parquet")),
            "val": Dataset.from_parquet(str(spec.export_dir / "val.parquet")),
        }
    )
    push_kwargs = {
        "repo_id": spec.repo_id,
        "private": private,
        "max_shard_size": max_shard_size,
    }
    if token:
        push_kwargs["token"] = token
    dataset_dict.push_to_hub(**push_kwargs)
    upload_metadata_kwargs = {
        "path_or_fileobj": str(spec.export_dir / "metadata.json"),
        "path_in_repo": "metadata.json",
        "repo_id": spec.repo_id,
        "repo_type": "dataset",
    }
    if token:
        upload_metadata_kwargs["token"] = token
    HfApi().upload_file(**upload_metadata_kwargs)


def push_export(
    *,
    spec: DatasetPushSpec,
    private: bool,
    token: str | None,
    max_shard_size: str,
    upload_mode: UploadMode,
) -> None:
    if upload_mode == "direct":
        push_existing_export_direct(spec=spec, private=private, token=token)
        return
    if upload_mode == "datasets":
        push_existing_export(spec=spec, private=private, token=token, max_shard_size=max_shard_size)
        return
    raise ValueError(f"Unsupported upload mode: {upload_mode}")


def push_specs(
    *,
    specs: Sequence[DatasetPushSpec],
    private: bool,
    token: str | None,
    max_shard_size: str,
    upload_mode: UploadMode,
    workers: int,
) -> None:
    if workers == 1:
        for spec in specs:
            push_export(
                spec=spec,
                private=private,
                token=token,
                max_shard_size=max_shard_size,
                upload_mode=upload_mode,
            )
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                push_export,
                spec=spec,
                private=private,
                token=token,
                max_shard_size=max_shard_size,
                upload_mode=upload_mode,
            )
            for spec in specs
        ]
        for future in futures:
            future.result()


def _parse_dataset_specs(raw_specs: Sequence[Sequence[str]]) -> list[DatasetPushSpec]:
    return [
        DatasetPushSpec(export_dir=Path(raw_spec[0]).expanduser().resolve(), repo_id=raw_spec[1].strip())
        for raw_spec in raw_specs
    ]


def _parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).replace(",", " ").split() if item.strip()]


def _build_fixed_latency_specs(
    *,
    root: str | Path,
    env: str,
    latencies: Sequence[int],
    repo_template: str,
) -> list[DatasetPushSpec]:
    root_path = Path(root).expanduser().resolve()
    return [
        DatasetPushSpec(
            export_dir=root_path / f"{env}_fix_latency_{latency}",
            repo_id=repo_template.format(env=env, latency=latency),
        )
        for latency in latencies
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Push existing Sample Factory teacher parquet exports to Hugging Face.")
    parser.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        metavar=("EXPORT_DIR", "HF_REPO_ID"),
        default=[],
        help="Local export directory and destination HF dataset repo id. Repeat to push multiple datasets.",
    )
    parser.add_argument("--fixed-latency-root", default="", help="Root containing fixed-latency export directories.")
    parser.add_argument("--fixed-latency-env", default="", help="Environment name used in fixed-latency export directory names.")
    parser.add_argument("--fixed-latencies", default="", help="Comma or space separated fixed latency values.")
    parser.add_argument(
        "--fixed-latency-repo-template",
        default="",
        help="HF repo template for fixed-latency exports. Supports {env} and {latency}.",
    )
    parser.add_argument("--upload-mode", choices=["direct", "datasets"], default="direct")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--hf-token", default="", help="Optional HF token. Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")
    parser.add_argument("--hf-max-shard-size", default="500MB")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    specs = _parse_dataset_specs(args.dataset)
    if args.fixed_latencies:
        specs.extend(
            _build_fixed_latency_specs(
                root=args.fixed_latency_root,
                env=args.fixed_latency_env,
                latencies=_parse_int_csv(args.fixed_latencies),
                repo_template=args.fixed_latency_repo_template,
            )
        )
    token = resolve_token(args.hf_token)
    push_specs(
        specs=specs,
        private=args.hf_private,
        token=token,
        max_shard_size=args.hf_max_shard_size,
        upload_mode=args.upload_mode,
        workers=args.workers,
    )
    for spec in specs:
        print(f"[push] pushed {spec.export_dir} to Hugging Face repo {spec.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
