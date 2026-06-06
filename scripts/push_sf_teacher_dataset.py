from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import time
from dataclasses import dataclass
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from urllib3.exceptions import ProtocolError


@dataclass(frozen=True)
class DatasetPushSpec:
    export_dir: Path
    repo_id: str


UploadMode = Literal["direct", "datasets"]
HF_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
HF_RETRY_ATTEMPTS = 5
HF_RETRY_INITIAL_DELAY_SECONDS = 10.0
HF_RETRY_MAX_DELAY_SECONDS = 120.0


def validate_export_dir(spec: DatasetPushSpec) -> None:
    missing = [
        name
        for name in ("train.parquet", "val.parquet", "metadata.json")
        if not (spec.export_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required export files under "
            f"{spec.export_dir}: {', '.join(missing)}"
        )


def _load_dataset_dict(export_dir: Path) -> Any:
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise RuntimeError(
            "Pushing teacher datasets requires the 'datasets' package and a "
            "parquet backend such as 'pyarrow'."
        ) from exc

    return DatasetDict(
        {
            "train": Dataset.from_parquet(str(export_dir / "train.parquet")),
            "val": Dataset.from_parquet(str(export_dir / "val.parquet")),
        }
    )


def _disable_progress_bars() -> None:
    try:
        import datasets
    except ImportError:
        datasets = None
    if datasets is not None:
        disable_fn = getattr(datasets, "disable_progress_bars", None)
        if callable(disable_fn):
            disable_fn()

    try:
        import huggingface_hub
    except ImportError:
        huggingface_hub = None
    if huggingface_hub is not None:
        disable_fn = getattr(huggingface_hub, "disable_progress_bars", None)
        if callable(disable_fn):
            disable_fn()


def _hf_http_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is None:
        return None
    return int(status_code)


def _is_retryable_protocol_error(exc: ProtocolError) -> bool:
    return any(isinstance(arg, RemoteDisconnected) for arg in exc.args)


def _is_retryable_hf_error(exc: Exception) -> bool:
    status_code = _hf_http_status_code(exc)
    if status_code in HF_RETRYABLE_STATUS_CODES:
        return True
    if isinstance(exc, ProtocolError):
        return _is_retryable_protocol_error(exc)
    return False


def _run_with_hf_retry(
    *,
    operation_name: str,
    context: str,
    operation: Callable[[], Any],
    attempts: int,
    initial_delay_seconds: float,
    max_delay_seconds: float,
) -> None:
    delay_seconds = float(initial_delay_seconds)
    attempt = 1
    while True:
        try:
            operation()
            return
        except Exception as exc:
            if not _is_retryable_hf_error(exc) or attempt >= attempts:
                raise
            status_code = _hf_http_status_code(exc)
            print(
                "[retry] "
                f"operation={operation_name} context={context} "
                f"status_code={status_code} attempt={attempt}/{attempts} "
                f"delay_seconds={delay_seconds:.1f}"
            )
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2.0, float(max_delay_seconds))
            attempt += 1


def _upload_metadata(
    *,
    export_dir: Path,
    repo_id: str,
    token: str | None,
) -> None:
    _disable_progress_bars()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Uploading metadata.json requires the 'huggingface_hub' package."
        ) from exc

    upload_kwargs: dict[str, Any] = {
        "path_or_fileobj": str(export_dir / "metadata.json"),
        "path_in_repo": "metadata.json",
        "repo_id": repo_id,
        "repo_type": "dataset",
    }
    if token:
        upload_kwargs["token"] = token
    _run_with_hf_retry(
        operation_name="upload_file",
        context=f"{repo_id}/metadata.json",
        operation=lambda: HfApi().upload_file(**upload_kwargs),
        attempts=HF_RETRY_ATTEMPTS,
        initial_delay_seconds=HF_RETRY_INITIAL_DELAY_SECONDS,
        max_delay_seconds=HF_RETRY_MAX_DELAY_SECONDS,
    )


def _build_hf_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Uploading teacher datasets requires the 'huggingface_hub' package."
        ) from exc
    return HfApi()


def _token_kwargs(token: str | None) -> dict[str, str]:
    if token:
        return {"token": token}
    return {}


def push_existing_export_direct(
    *,
    spec: DatasetPushSpec,
    private: bool,
    token: str | None,
) -> None:
    validate_export_dir(spec)
    api = _build_hf_api()
    _run_with_hf_retry(
        operation_name="create_repo",
        context=spec.repo_id,
        operation=lambda: api.create_repo(
            repo_id=spec.repo_id,
            repo_type="dataset",
            private=bool(private),
            exist_ok=True,
            **_token_kwargs(token),
        ),
        attempts=HF_RETRY_ATTEMPTS,
        initial_delay_seconds=HF_RETRY_INITIAL_DELAY_SECONDS,
        max_delay_seconds=HF_RETRY_MAX_DELAY_SECONDS,
    )
    _run_with_hf_retry(
        operation_name="upload_folder",
        context=spec.repo_id,
        operation=lambda: api.upload_folder(
            folder_path=str(spec.export_dir),
            repo_id=spec.repo_id,
            repo_type="dataset",
            allow_patterns=["train.parquet", "val.parquet", "metadata.json"],
            commit_message="Upload teacher dataset export",
            **_token_kwargs(token),
        ),
        attempts=HF_RETRY_ATTEMPTS,
        initial_delay_seconds=HF_RETRY_INITIAL_DELAY_SECONDS,
        max_delay_seconds=HF_RETRY_MAX_DELAY_SECONDS,
    )


def push_existing_export(
    *,
    spec: DatasetPushSpec,
    private: bool,
    token: str | None,
    max_shard_size: str,
) -> None:
    validate_export_dir(spec)
    _disable_progress_bars()
    dataset_dict = _load_dataset_dict(spec.export_dir)
    push_kwargs: dict[str, Any] = {
        "repo_id": spec.repo_id,
        "private": bool(private),
        "max_shard_size": str(max_shard_size),
    }
    if token:
        push_kwargs["token"] = token
    _run_with_hf_retry(
        operation_name="push_to_hub",
        context=spec.repo_id,
        operation=lambda: dataset_dict.push_to_hub(**push_kwargs),
        attempts=HF_RETRY_ATTEMPTS,
        initial_delay_seconds=HF_RETRY_INITIAL_DELAY_SECONDS,
        max_delay_seconds=HF_RETRY_MAX_DELAY_SECONDS,
    )
    _upload_metadata(export_dir=spec.export_dir, repo_id=spec.repo_id, token=token)


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
        push_existing_export(
            spec=spec,
            private=private,
            token=token,
            max_shard_size=max_shard_size,
        )
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
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    if worker_count == 1:
        for spec in specs:
            push_export(
                spec=spec,
                private=private,
                token=token,
                max_shard_size=max_shard_size,
                upload_mode=upload_mode,
            )
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
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
    specs: list[DatasetPushSpec] = []
    for raw_spec in raw_specs:
        if len(raw_spec) != 2:
            raise ValueError("--dataset expects exactly two values: EXPORT_DIR REPO_ID")
        export_dir = Path(raw_spec[0]).expanduser().resolve()
        repo_id = str(raw_spec[1]).strip()
        if not repo_id:
            raise ValueError("--dataset repo id must be non-empty")
        specs.append(DatasetPushSpec(export_dir=export_dir, repo_id=repo_id))
    if not specs:
        raise ValueError("At least one --dataset EXPORT_DIR REPO_ID pair is required")
    return specs


def _parse_int_csv(value: str) -> list[int]:
    raw = str(value).strip()
    if not raw:
        return []
    latencies: list[int] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        latency = int(stripped)
        if latency < 0:
            raise ValueError(f"Latency values must be non-negative, got {latency}")
        latencies.append(latency)
    return latencies


def _build_fixed_latency_specs(
    *,
    root: Path,
    env_name: str,
    latencies: Sequence[int],
    repo_template: str,
) -> list[DatasetPushSpec]:
    env = str(env_name).strip()
    if not env:
        raise ValueError("--fixed-latency-env must be non-empty")
    template = str(repo_template).strip()
    if not template:
        raise ValueError("--fixed-latency-repo-template must be non-empty")
    specs: list[DatasetPushSpec] = []
    for latency in latencies:
        latency_value = int(latency)
        if latency_value < 0:
            raise ValueError(f"Latency values must be non-negative, got {latency_value}")
        specs.append(
            DatasetPushSpec(
                export_dir=Path(root).expanduser().resolve() / f"{env}_fix_latency_{latency_value}",
                repo_id=template.format(env=env, latency=latency_value),
            )
        )
    return specs


def _resolve_token(explicit_token: str) -> str | None:
    token = str(explicit_token).strip()
    if token:
        return token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Push existing Sample Factory teacher parquet exports to Hugging Face."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("EXPORT_DIR", "REPO_ID"),
        help="Local export directory and destination HF dataset repo id. Repeat to push multiple datasets.",
    )
    parser.add_argument(
        "--fixed-latency-root",
        type=str,
        default="",
        help="Root containing fixed-latency export folders named ENV_fix_latency_LATENCY.",
    )
    parser.add_argument(
        "--fixed-latency-env",
        type=str,
        default="",
        help="Environment name for fixed-latency folder discovery, e.g. demon_attack.",
    )
    parser.add_argument(
        "--fixed-latencies",
        type=str,
        default="",
        help="Comma-separated latency values for fixed-latency exports, e.g. 0,1,2.",
    )
    parser.add_argument(
        "--fixed-latency-repo-template",
        type=str,
        default="",
        help="HF repo template for fixed-latency exports. Supports {env} and {latency}.",
    )
    parser.add_argument(
        "--upload-mode",
        type=str,
        choices=["direct", "datasets"],
        default="datasets",
        help=(
            "datasets preserves the original Dataset.from_parquet + push_to_hub path; "
            "direct uploads existing parquet files without rebuilding them."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of dataset repos to push concurrently.",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Create or update the HF dataset repos as private.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="Optional HF token. If omitted, HF_TOKEN or HUGGINGFACE_HUB_TOKEN is used.",
    )
    parser.add_argument(
        "--hf-max-shard-size",
        type=str,
        default="500MB",
        help="Shard size forwarded to datasets.push_to_hub.",
    )
    return parser


def main(argv: Sequence[str] | None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    specs: list[DatasetPushSpec] = []
    if args.dataset:
        specs.extend(_parse_dataset_specs(args.dataset))
    fixed_latencies = _parse_int_csv(str(args.fixed_latencies))
    if fixed_latencies:
        if not str(args.fixed_latency_root).strip():
            raise ValueError("--fixed-latency-root is required with --fixed-latencies")
        if not str(args.fixed_latency_env).strip():
            raise ValueError("--fixed-latency-env is required with --fixed-latencies")
        if not str(args.fixed_latency_repo_template).strip():
            raise ValueError("--fixed-latency-repo-template is required with --fixed-latencies")
        specs.extend(
            _build_fixed_latency_specs(
                root=Path(str(args.fixed_latency_root)),
                env_name=str(args.fixed_latency_env),
                latencies=fixed_latencies,
                repo_template=str(args.fixed_latency_repo_template),
            )
        )
    if not specs:
        raise ValueError("At least one --dataset pair or --fixed-latencies value is required")
    token = _resolve_token(str(args.hf_token))
    push_specs(
        specs=specs,
        private=bool(args.hf_private),
        token=token,
        max_shard_size=str(args.hf_max_shard_size),
        upload_mode=args.upload_mode,
        workers=int(args.workers),
    )
    for spec in specs:
        print(f"[push] pushed {spec.export_dir} to Hugging Face repo {spec.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
