#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa

try:
    from datasets.exceptions import DatasetGenerationError
except Exception:
    class DatasetGenerationError(Exception):
        pass


_LATENCY_SUBDIR_RE = re.compile(r"((?:fix|fixed)_latency_)\d+(_)")
READ_SCHEMA_EXCEPTIONS = (ValueError, KeyError, FileNotFoundError, pa.ArrowInvalid, DatasetGenerationError)


def latency_subdir_for(template: str, latency: int) -> str | None:
    new_value, count = _LATENCY_SUBDIR_RE.subn(rf"\g<1>{latency}\g<2>", template, count=1)
    if count == 1 and new_value.startswith(("zero-latency/", "latency-aware/")):
        condition = "zero-latency" if latency == 0 else "latency-aware"
        new_value = condition + "/" + new_value.split("/", 1)[1]
    return new_value if count == 1 else None


def resolve_latency_subdirs(template: str | None, latencies: Iterable[int] | None) -> list[str | None]:
    if template is None or not latencies:
        return [template]
    if "," in str(template):
        return [item.strip() for item in str(template).split(",") if item.strip()]
    unique_latencies = sorted({int(value) for value in latencies})
    resolved = [latency_subdir_for(template, latency) for latency in unique_latencies]
    if any(value is None for value in resolved):
        # template doesn't match the known fix_latency_<N>_ pattern; don't partially apply
        return [template]
    return resolved


def concatenate_latency_parts(parts: list) -> Any:
    if len(parts) == 1:
        return parts[0]
    from datasets import concatenate_datasets

    try:
        return concatenate_datasets(parts)
    except Exception:
        try:
            casted = [parts[0]] + [part.cast(parts[0].features) for part in parts[1:]]
            return concatenate_datasets(casted)
        except Exception as exc:
            raise ValueError(f"could not concatenate per-latency dataset parts: {exc}") from exc


EXPECTED_ACTIONS = ["NOOP", "FLAP"]
EXPECTED_PROMPT = (
    "You are playing Flappy Bird. "
    "Pass through the pipe gaps and stay alive. "
    "Choose the action: NOOP, FLAP."
)


def _local_parquet_files(dataset_name: str, dataset_source_subdir: str | None = None) -> list[str] | None:
    dataset_path = Path(dataset_name).expanduser()
    if not dataset_path.exists():
        return None
    if dataset_source_subdir not in (None, ""):
        dataset_path = dataset_path / str(dataset_source_subdir)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"dataset_source_subdir={dataset_source_subdir!r} does not exist under {dataset_name!r}"
            )
    if dataset_path.is_file():
        if dataset_path.suffix != ".parquet":
            raise ValueError(f"dataset_name={dataset_name!r} exists but is not a parquet file")
        return [str(dataset_path)]

    parquet_files = sorted(dataset_path.rglob("*.parquet"))
    if len(parquet_files) == 0:
        raise FileNotFoundError(f"dataset_name={dataset_name!r} exists but contains no parquet files")

    train_files = [
        parquet_file
        for parquet_file in parquet_files
        if any("train" in part.lower() for part in parquet_file.relative_to(dataset_path).parts)
    ]
    return [str(parquet_file) for parquet_file in (train_files or parquet_files)]


def _load_hf_dataset(
    dataset_name: str,
    dataset_config_name: str | None,
    dataset_source_subdir: str | None,
    *,
    split: str,
    cache_dir: str | None,
    columns: list[str] | None = None,
):
    from datasets import load_dataset

    load_kwargs = {
        "split": split,
        "cache_dir": cache_dir,
        "columns": columns,
        # Some hosted RL-game datasets record the validation split as `val` in
        # dataset_info but generate it as `validation`. Disable split-count
        # verification so train loading is not blocked by that naming mismatch.
        "verification_mode": "no_checks",
    }
    if dataset_config_name not in (None, ""):
        return load_dataset(dataset_name, dataset_config_name, **load_kwargs)
    if dataset_source_subdir not in (None, ""):
        source_subdir = str(dataset_source_subdir)
        try:
            return load_dataset(dataset_name, source_subdir, **load_kwargs)
        except (ValueError, FileNotFoundError):
            load_kwargs["data_dir"] = source_subdir
            return load_dataset(dataset_name, **load_kwargs)
    return load_dataset(dataset_name, **load_kwargs)


def _load_train_split(
    dataset_name: str,
    cache_dir: str | None,
    columns: list[str] | None = None,
    dataset_config_name: str | None = None,
    dataset_source_subdir: str | None = None,
    latencies: Iterable[int] | None = None,
):
    from datasets import load_dataset

    def _filter_internal_split(ds):
        if "split" in ds.column_names:
            return ds.filter(lambda row: str(row["split"]).lower() == "train")
        return ds

    def _load_remote_parquet(subdir: str | None):
        if subdir in (None, "") or Path(dataset_name).expanduser().exists():
            return None
        remote_file = f"hf://datasets/{dataset_name}/{str(subdir).strip('/')}/train.parquet"
        load_columns = list(columns) if columns is not None else None
        if load_columns is not None and "split" not in load_columns:
            load_columns.append("split")
        try:
            ds = load_dataset("parquet", data_files=[remote_file], split="train", cache_dir=cache_dir, columns=load_columns)
        except READ_SCHEMA_EXCEPTIONS:
            if columns is None:
                return None
            try:
                ds = load_dataset("parquet", data_files=[remote_file], split="train", cache_dir=cache_dir, columns=columns)
            except READ_SCHEMA_EXCEPTIONS:
                return None
        return _filter_internal_split(ds)

    def _load_one(subdir: str | None):
        local_files = _local_parquet_files(dataset_name, subdir)
        if local_files is not None:
            load_columns = list(columns) if columns is not None else None
            if load_columns is not None and "split" not in load_columns:
                load_columns.append("split")
            try:
                ds = load_dataset("parquet", data_files=local_files, split="train", cache_dir=cache_dir, columns=load_columns)
            except READ_SCHEMA_EXCEPTIONS:
                if columns is None:
                    raise
                ds = load_dataset("parquet", data_files=local_files, split="train", cache_dir=cache_dir, columns=columns)
            return _filter_internal_split(ds)

        remote_ds = _load_remote_parquet(subdir)
        if remote_ds is not None:
            return remote_ds

        try:
            ds = _load_hf_dataset(
                dataset_name, dataset_config_name, subdir,
                split="train", cache_dir=cache_dir, columns=columns,
            )
            return _filter_internal_split(ds)
        except READ_SCHEMA_EXCEPTIONS:
            load_columns = list(columns or [])
            if "split" not in load_columns:
                load_columns.append("split")
            ds_all = _load_hf_dataset(
                dataset_name, dataset_config_name, subdir,
                split="train", cache_dir=cache_dir, columns=load_columns,
            )
            return ds_all.filter(lambda row: str(row["split"]).lower() == "train")

    subdirs = resolve_latency_subdirs(dataset_source_subdir, latencies)
    parts = [_load_one(subdir) for subdir in subdirs]
    return concatenate_latency_parts(parts)


def latency_id_from_row(
    row: dict[str, Any],
    *,
    latency_column: str | None,
    target_latency_unit: str,
    obs_stride_raw_frames: int,
    default_latency: int | None = None,
) -> int | None:
    if target_latency_unit not in {"raw_frames", "observation_steps"}:
        raise ValueError(f"unsupported target_latency_unit={target_latency_unit!r}")
    if latency_column is None:
        return default_latency
    value = row[latency_column]
    if value is None:
        return default_latency
    if latency_column != "latency_raw_frames" or target_latency_unit == "raw_frames":
        return value
    if value % obs_stride_raw_frames != 0:
        raise ValueError(
            f"latency_raw_frames={value} is not divisible by "
            f"obs_stride_raw_frames={obs_stride_raw_frames}"
        )
    return value // obs_stride_raw_frames


def source_timing_from_args(
    *,
    source_metadata: str | None,
    source_fps: float | None,
    obs_stride_raw_frames: int | None,
) -> tuple[float, int, str]:
    if source_metadata is not None:
        if source_fps is not None or obs_stride_raw_frames is not None:
            raise ValueError("source_metadata cannot be combined with explicit source timing")
        metadata = json.loads(Path(source_metadata).read_text(encoding="utf-8"))
        if metadata["rows_unit"] != "decision_step":
            raise ValueError(
                f"source metadata rows_unit must be decision_step, got {metadata['rows_unit']!r}"
            )
        return (
            metadata["obs_fps"],
            metadata["obs_stride_raw_frames"],
            metadata["rows_unit"],
        )
    if source_fps is None or obs_stride_raw_frames is None:
        raise ValueError(
            "pass --source-metadata or both --source-fps and --obs-stride-raw-frames"
        )
    return source_fps, obs_stride_raw_frames, "decision_step"


def build_latency_prompt_map(
    rows: Iterable[dict[str, Any]],
    *,
    latency_column: str,
    target_latency_unit: str,
    obs_stride_raw_frames: int,
) -> dict[str, dict[str, Any]]:
    by_latency: dict[int, dict[str, Any]] = {}
    for row in rows:
        if "split" in row and str(row["split"]).lower() != "train":
            continue
        if "prompt" not in row:
            raise KeyError(f"row is missing latency/prompt columns; available columns: {sorted(row.keys())}")
        latency = latency_id_from_row(
            row,
            latency_column=latency_column,
            target_latency_unit=target_latency_unit,
            obs_stride_raw_frames=obs_stride_raw_frames,
        )
        if latency is None:
            raise KeyError(f"row is missing latency/prompt columns; available columns: {sorted(row.keys())}")
        if latency_column == "latency_raw_frames":
            latency_raw_frames = int(row[latency_column])
        elif "latency_raw_frames" in row and row["latency_raw_frames"] is not None:
            latency_raw_frames = int(row["latency_raw_frames"])
        elif target_latency_unit == "raw_frames":
            latency_raw_frames = int(latency)
        else:
            latency_raw_frames = int(latency) * int(obs_stride_raw_frames)
        prompt = str(row["prompt"])
        latency_ms = row.get("latency_ms")
        current = by_latency.get(latency)
        if current is None:
            by_latency[latency] = {
                "latency": latency,
                "latency_raw_frames": latency_raw_frames,
                "latency_ms": latency_ms,
                "prompt": prompt,
            }
            continue
        if (
            current["prompt"] != prompt
            or current.get("latency_ms") != latency_ms
            or int(current["latency_raw_frames"]) != latency_raw_frames
        ):
            raise ValueError(f"inconsistent prompt/latency_ms values for latency={latency}")
    return {str(k): by_latency[k] for k in sorted(by_latency)}


def verify_dataset(
    dataset_name: str,
    *,
    rows: int = 200,
    cache_dir: str | None = None,
    dataset_config_name: str | None = None,
    dataset_source_subdir: str | None = None,
    strict: bool = False,
    allow_mixed_latency_prompts: bool = False,
    latencies: list[int] | None = None,
    source_latency_column: str,
    target_latency_unit: str,
    obs_stride_raw_frames: int,
) -> bool:
    try:
        for columns in (
            ["prompt", "action_id", "action_text", source_latency_column, "latency_ms"],
            ["prompt", "action_id", "action_text"],
            ["prompt", "action_id"],
            None,
        ):
            try:
                ds = _load_train_split(
                    dataset_name,
                    cache_dir,
                    columns=columns,
                    dataset_config_name=dataset_config_name,
                    dataset_source_subdir=dataset_source_subdir,
                    latencies=latencies,
                )
                break
            except Exception:
                if columns is None:
                    raise
    except Exception as exc:
        print(f"ERROR: could not load dataset {dataset_name}: {exc}")
        if strict:
            raise
        return False

    if len(ds) == 0:
        print("ERROR: dataset has zero train rows.")
        if strict:
            raise ValueError("dataset has zero train rows")
        return False

    sample_n = min(rows, len(ds))
    ok = True

    prompts = {str(ds[i]["prompt"]) for i in range(sample_n)}
    if any(not prompt.strip() for prompt in prompts):
        print("ERROR: prompt must be a non-empty string.")
        ok = False
    if allow_mixed_latency_prompts:
        try:
            mapping = build_latency_prompt_map(
                ds,
                latency_column=source_latency_column,
                target_latency_unit=target_latency_unit,
                obs_stride_raw_frames=obs_stride_raw_frames,
            )
            if latencies and len({int(v) for v in latencies}) > 1 and len(mapping) <= 1:
                raise ValueError(f"expected more than one latency prompt, got {len(mapping)}")
            print("Latency prompt map:")
            print(json.dumps(mapping, indent=2))
        except Exception as exc:
            print(f"ERROR: invalid mixed-latency prompt mapping: {exc}")
            ok = False

    action_id_to_text: dict[int, set[str]] = defaultdict(set)
    seen_ids = set()
    has_action_text = "action_text" in ds.column_names
    for i in range(sample_n):
        action_id = int(ds[i]["action_id"])
        seen_ids.add(action_id)
        if has_action_text:
            action_id_to_text[action_id].add(str(ds[i]["action_text"]))

    if not seen_ids or min(seen_ids) < 0 or max(seen_ids) >= len(EXPECTED_ACTIONS):
        print(f"ERROR: action_id values must be in [0, {len(EXPECTED_ACTIONS) - 1}], saw {sorted(seen_ids)}")
        ok = False

    if has_action_text:
        for action_id, texts in sorted(action_id_to_text.items()):
            expected = EXPECTED_ACTIONS[action_id] if action_id < len(EXPECTED_ACTIONS) else None
            if texts != {expected}:
                print(f"ERROR: action_id={action_id} maps to {sorted(texts)}, expected {expected!r}")
                ok = False

    if ok:
        print(f"Flappy dataset verification passed for {dataset_name} ({sample_n} sampled rows).")
    elif strict:
        raise ValueError("Flappy dataset verification failed")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", "--dataset_name", required=True)
    parser.add_argument("--dataset-config-name", "--dataset_config_name", default=None)
    parser.add_argument("--dataset-source-subdir", "--dataset_source_subdir", default=None)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--cache-dir", "--cache_dir", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-mixed-latency-prompts", "--allow_mixed_latency_prompts", action="store_true")
    parser.add_argument("--source-latency-column", choices=["latency", "latency_raw_frames"], required=True)
    parser.add_argument("--target-latency-unit", choices=["raw_frames", "observation_steps"], required=True)
    parser.add_argument("--obs-stride-raw-frames", type=int, required=True)
    args = parser.parse_args()

    try:
        ok = verify_dataset(
            args.dataset_name,
            rows=args.rows,
            cache_dir=args.cache_dir,
            dataset_config_name=args.dataset_config_name,
            dataset_source_subdir=args.dataset_source_subdir,
            strict=args.strict,
            allow_mixed_latency_prompts=args.allow_mixed_latency_prompts,
            source_latency_column=args.source_latency_column,
            target_latency_unit=args.target_latency_unit,
            obs_stride_raw_frames=args.obs_stride_raw_frames,
        )
    except Exception:
        if args.strict:
            raise
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
