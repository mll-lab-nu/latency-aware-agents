#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_flappy_dataset import resolve_latency_subdirs


STEP_FILE_RE = re.compile(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$")
STEP_STATE_RE = re.compile(r"steps_(\d+)_state$")
DEBUG_DATASET_RE = re.compile(r"^(?P<base>.+)(?P<debug>__debug(?:_[A-Za-z0-9_-]+)?_\d+ep)$")
INITIALIZATION_SOURCE_MODES = {"bridge", "pre-trained", "pretrained", "backbone_bridge_factorized11"}


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _safe_path_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "checkpoint"


def _parse_latency_episode_map(value: Any) -> dict[int, int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        return {int(k): int(v) for k, v in value.items()}
    if isinstance(value, str):
        pairs: dict[int, int] = {}
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(
                    "--episodes-per-latency-by-latency must use 'latency:episodes' items, "
                    f"got {value!r}"
                )
            latency, episodes = item.split(":", 1)
            pairs[int(latency.strip())] = int(episodes.strip())
        return pairs or None
    raise TypeError(f"Unsupported episodes_per_latency_by_latency={value!r}")


def _format_latency_episode_map(value: dict[int, int] | None) -> str | None:
    if not value:
        return None
    return ",".join(f"{int(latency)}:{int(value[latency])}" for latency in sorted(value))


def _safe_token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_") or "value"


def _fixed_dataset_source(dataset_name, revision, source_subdir, latencies, cache_dir):
    """Materialize one fixed HF dataset revision for existing local converters."""
    if revision is None:
        return dataset_name

    subdirs = resolve_latency_subdirs(source_subdir, latencies)
    patterns = [f"{subdir}/**" for subdir in subdirs] if source_subdir is not None else None
    return snapshot_download(
        dataset_name, repo_type="dataset", revision=revision,
        cache_dir=cache_dir, allow_patterns=patterns, endpoint="https://huggingface.co",
    )


def _record_fixed_dataset_source(dataset_path, source, revision):
    """Preserve the HF origin when conversion reads its local fixed snapshot."""
    if revision is None:
        return
    path = dataset_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source"] = source
    manifest["source_revision"] = revision
    path.write_text(json.dumps(manifest, indent=2))


def _latency_suffix(latencies: list[int] | None) -> str:
    if not latencies:
        return "all"
    return "lat" + "_".join(str(int(value)) for value in latencies)


def _derived_dataset_name(base_name: str, *, latency_filter: list[int] | None, episodes_per_latency: int | None, max_episodes: int | None) -> str:
    pieces = [str(base_name), _latency_suffix(latency_filter)]
    if episodes_per_latency is not None:
        pieces.append(f"{int(episodes_per_latency)}ep_per_lat")
    elif max_episodes is not None:
        pieces.append(f"{int(max_episodes)}ep")
    return "__".join(_safe_token(piece) for piece in pieces)


def _has_files(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True
    return any(path.iterdir())


def _dataset_ready(dataset_dir: Path) -> bool:
    required = [
        dataset_dir / "meta/modality.json",
        dataset_dir / "meta/info.json",
        dataset_dir / "meta/episodes.jsonl",
        dataset_dir / "meta/tasks.jsonl",
    ]
    return all(path.exists() for path in required) and any(dataset_dir.glob("data/*/*.parquet"))


def _load_source_latency_prompt_map(
    dataset_name: str,
    *,
    cache_dir: str | None = None,
    dataset_config_name: str | None = None,
    dataset_source_subdir: str | None = None,
    source_latency_column: str,
    target_latency_unit: str,
    obs_stride_raw_frames: int,
    latencies: list[int] | None = None,
) -> dict[str, dict[str, Any]]:
    from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_flappy_dataset import (
        _load_train_split,
        build_latency_prompt_map,
    )

    column_variants: tuple[list[str], ...] = (
        ["prompt", source_latency_column, "latency_ms", "split"],
        ["prompt", source_latency_column, "latency_ms"],
    )
    last_exc: Exception | None = None
    ds = None
    for columns in column_variants:
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
        except Exception as exc:
            last_exc = exc
            ds = None
    if ds is None:
        raise ValueError(
            "could not load source dataset latency prompts "
            f"(dataset={dataset_name!r}, config={dataset_config_name!r}, "
            f"source_subdir={dataset_source_subdir!r}, latencies={latencies!r})"
        ) from last_exc
    return build_latency_prompt_map(
        ds,
        latency_column=source_latency_column,
        target_latency_unit=target_latency_unit,
        obs_stride_raw_frames=obs_stride_raw_frames,
    )


def _write_prompt_map(path: Path, prompt_map: dict[str, dict[str, Any]], *, latency_filter: list[int] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if latency_filter:
        allowed = {int(value) for value in latency_filter}
        prompt_map = {str(k): v for k, v in prompt_map.items() if int(k) in allowed}
    path.write_text(json.dumps(prompt_map, indent=2), encoding="utf-8")
    return path


def _prompt_map_has_raw_frame_entries(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        mapping = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(mapping, dict) or not mapping:
        return False
    return all(
        isinstance(entry, dict) and "latency_raw_frames" in entry
        for entry in mapping.values()
    )


def _read_prompt_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        mapping = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(mapping, dict):
        return {}
    return {str(key): value for key, value in mapping.items() if isinstance(value, dict)}


def _prompt_map_entry_count(path: Path) -> int:
    return len(_read_prompt_map(path))


def _extract_latency_ms_from_prompt(prompt: str) -> float | None:
    match = re.search(r"\(([0-9]+(?:\.[0-9]+)?)\s*ms\)", prompt)
    if match is None:
        return None
    return float(match.group(1))


def _read_local_task_prompts(dataset_dir: Path) -> dict[int, str]:
    tasks_path = dataset_dir / "meta/tasks.jsonl"
    if not tasks_path.exists():
        return {}
    prompts: dict[int, str] = {}
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        task_index = int(record.get("task_index", len(prompts)))
        prompt = record.get("task", record.get("prompt"))
        if prompt is not None:
            prompts[task_index] = str(prompt)
    return prompts


def _latency_raw_frames_from_target_latency(
    latency: int,
    *,
    target_latency_unit: str,
    obs_stride_raw_frames: int,
) -> int:
    if target_latency_unit == "raw_frames":
        return int(latency)
    if target_latency_unit == "observation_steps":
        return int(latency) * int(obs_stride_raw_frames)
    raise ValueError(f"unsupported target_latency_unit={target_latency_unit!r}")


def _load_local_latency_prompt_map(
    dataset_dir: Path,
    *,
    target_latency_unit: str,
    obs_stride_raw_frames: int,
) -> dict[str, dict[str, Any]]:
    task_prompts = _read_local_task_prompts(dataset_dir)
    if not task_prompts:
        return {}

    try:
        import pyarrow.parquet as pq
    except Exception:
        return {}

    prompt_map: dict[int, dict[str, Any]] = {}
    for parquet_path in sorted(dataset_dir.glob("data/*/*.parquet")):
        try:
            parquet = pq.ParquetFile(parquet_path)
            schema_names = set(parquet.schema_arrow.names)
        except Exception:
            continue
        if "latency" not in schema_names or "task_index" not in schema_names:
            continue
        for batch in parquet.iter_batches(columns=["latency", "task_index"], batch_size=65536):
            for row in batch.to_pylist():
                latency = int(row["latency"])
                prompt = task_prompts.get(int(row["task_index"]))
                if prompt is None:
                    continue
                latency_raw_frames = _latency_raw_frames_from_target_latency(
                    latency,
                    target_latency_unit=target_latency_unit,
                    obs_stride_raw_frames=obs_stride_raw_frames,
                )
                entry = {
                    "latency": latency,
                    "latency_raw_frames": latency_raw_frames,
                    "latency_ms": _extract_latency_ms_from_prompt(prompt),
                    "prompt": prompt,
                }
                existing = prompt_map.get(latency)
                if existing is None:
                    prompt_map[latency] = entry
                elif existing["prompt"] != prompt or int(existing["latency_raw_frames"]) != latency_raw_frames:
                    raise ValueError(
                        f"local dataset has inconsistent prompt map rows for latency={latency}: "
                        f"{existing!r} vs {entry!r}"
                    )
    return {str(k): prompt_map[k] for k in sorted(prompt_map)}


def _normalize_prompt_map_entries(
    prompt_map: dict[str, dict[str, Any]],
    *,
    target_latency_unit: str,
    obs_stride_raw_frames: int,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in prompt_map.items():
        if not isinstance(value, dict) or "prompt" not in value:
            continue
        latency = int(value.get("latency", key))
        entry = dict(value)
        entry["latency"] = latency
        if entry.get("latency_raw_frames") is None:
            entry["latency_raw_frames"] = _latency_raw_frames_from_target_latency(
                latency,
                target_latency_unit=target_latency_unit,
                obs_stride_raw_frames=obs_stride_raw_frames,
            )
        else:
            entry["latency_raw_frames"] = int(entry["latency_raw_frames"])
        if "latency_ms" not in entry:
            entry["latency_ms"] = _extract_latency_ms_from_prompt(str(entry["prompt"]))
        normalized[str(latency)] = entry
    return normalized


def _write_sorted_prompt_map(path: Path, prompt_map: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({str(k): prompt_map[str(k)] for k in sorted(int(k) for k in prompt_map)}, indent=2),
        encoding="utf-8",
    )


def _find_latest_local_checkpoint(checkpoint_dir: Path) -> tuple[Path | None, int, str | None]:
    if not checkpoint_dir.exists():
        return None, 0, None
    candidates: list[tuple[int, int, Path, str]] = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir():
            match = STEP_STATE_RE.match(item.name)
            if match:
                candidates.append((int(match.group(1)), 1, item, "state"))
        elif item.is_file():
            match = STEP_FILE_RE.match(item.name)
            if match:
                candidates.append((int(match.group(1)), 0, item, "model"))
    if not candidates:
        return None, 0, None
    candidates.sort(key=lambda item: (item[0], item[1]))
    step, _, path, kind = candidates[-1]
    return path, step, kind


def _read_best_checkpoint_step(checkpoint_dir: Path) -> int:
    metadata_path = checkpoint_dir / "best_model_metadata.json"
    if not metadata_path.exists():
        return 0
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    for key in ("best_step", "step"):
        if metadata.get(key) is not None:
            try:
                return int(metadata[key])
            except Exception:
                return 0
    return 0


def _find_local_best_checkpoint(checkpoint_dir: Path) -> tuple[Path | None, int, str | None]:
    best_state = checkpoint_dir / "best_state"
    if not best_state.is_dir():
        return None, 0, None
    return best_state, _read_best_checkpoint_step(checkpoint_dir), "state"


def _candidate_resume_paths(checkpoint: str, checkpoint_dir: Path) -> list[Path]:
    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.is_absolute():
        return [checkpoint_path]

    run_dir = checkpoint_dir.parent
    return [
        run_dir / checkpoint_path,
        checkpoint_dir / checkpoint_path.name,
        checkpoint_dir / checkpoint_path,
    ]


def _resolve_existing_resume_path(checkpoint: str, checkpoint_dir: Path) -> Path:
    candidates = _candidate_resume_paths(checkpoint, checkpoint_dir)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    checked = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(f"explicit resume checkpoint does not exist: {checkpoint}. Checked: {checked}")


def _resolve_explicit_resume_checkpoint(checkpoint: str, checkpoint_dir: Path) -> tuple[Path, int, str]:
    checkpoint_path = _resolve_existing_resume_path(checkpoint, checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"explicit resume checkpoint does not exist: {checkpoint_path}")
    if checkpoint_path.is_dir():
        state_match = STEP_STATE_RE.match(checkpoint_path.name)
        if state_match:
            return checkpoint_path, int(state_match.group(1)), "state"
        if checkpoint_path.name == "best_state":
            return checkpoint_path, _read_best_checkpoint_step(checkpoint_path.parent), "state"
        raise ValueError(
            f"unsupported explicit resume checkpoint directory: {checkpoint_path}. "
            "Expected steps_<N>_state/ or best_state/."
        )
    if checkpoint_path.is_file():
        file_match = STEP_FILE_RE.match(checkpoint_path.name)
        if file_match:
            return checkpoint_path, int(file_match.group(1)), "model"
        if checkpoint_path.parent.name.endswith("_state"):
            state_match = STEP_STATE_RE.match(checkpoint_path.parent.name)
            if state_match and checkpoint_path.name in {"model.safetensors", "pytorch_model.bin"}:
                return checkpoint_path.parent, int(state_match.group(1)), "state"
        if checkpoint_path.parent.name == "best_state" and checkpoint_path.name in {"model.safetensors", "pytorch_model.bin"}:
            return checkpoint_path.parent, _read_best_checkpoint_step(checkpoint_path.parent.parent), "state"
    raise ValueError(
        f"unsupported explicit resume checkpoint: {checkpoint_path}. "
        "Expected steps_<N>_state/, steps_<N>_pytorch_model.pt, steps_<N>_model.safetensors, "
        "best_state/, or a model file inside a state directory."
    )


def _download_latest_hf_checkpoint(repo_id: str, checkpoint_dir: Path) -> tuple[Path | None, int, str | None, str | None]:
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except Exception as exc:
        return None, 0, None, f"huggingface_hub import failed: {exc}"

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        files = HfApi(token=hf_token).list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as exc:
        return None, 0, None, f"could not list HF checkpoint repo {repo_id}: {exc}"

    candidates: list[tuple[int, int, str, str]] = []
    seen_state_dirs = set()
    for file_path in files:
        first_part = file_path.split("/", 1)[0]
        state_match = STEP_STATE_RE.match(first_part)
        if state_match:
            if first_part not in seen_state_dirs:
                candidates.append((int(state_match.group(1)), 1, first_part, "state"))
                seen_state_dirs.add(first_part)
            continue
        file_match = STEP_FILE_RE.match(os.path.basename(file_path))
        if file_match:
            candidates.append((int(file_match.group(1)), 0, file_path, "model"))
    if not candidates:
        return None, 0, None, None

    candidates.sort(key=lambda item: (item[0], item[1]))
    step, _, chosen, kind = candidates[-1]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        if kind == "state":
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                allow_patterns=[f"{chosen}/**"],
                local_dir=str(checkpoint_dir),
                local_dir_use_symlinks=False,
                token=hf_token,
            )
            local_path = checkpoint_dir / chosen
        else:
            local_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                filename=chosen,
                local_dir=str(checkpoint_dir),
                local_dir_use_symlinks=False,
                token=hf_token,
            )
    except Exception as exc:
        return None, 0, None, f"could not download HF checkpoint {chosen}: {exc}"
    return Path(local_path), step, kind, None


def _download_hf_best_checkpoint(repo_id: str, checkpoint_dir: Path) -> tuple[Path | None, int, str | None, str | None]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception as exc:
        return None, 0, None, f"huggingface_hub import failed: {exc}"

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        files = HfApi(token=hf_token).list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as exc:
        return None, 0, None, f"could not list HF checkpoint repo {repo_id}: {exc}"

    if not any(file_path.startswith("best_state/") for file_path in files):
        return None, 0, None, None

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=["best_state/**", "best_model_metadata.json"],
            local_dir=str(checkpoint_dir),
            local_dir_use_symlinks=False,
            token=hf_token,
        )
    except Exception as exc:
        return None, 0, None, f"could not download HF best checkpoint from {repo_id}: {exc}"

    best_state = checkpoint_dir / "best_state"
    if not best_state.is_dir():
        return None, 0, None, f"HF best checkpoint from {repo_id} did not contain best_state/"
    return best_state, _read_best_checkpoint_step(checkpoint_dir), "state", None


def _download_hf_checkpoint_file(repo_id: str, filename: str, checkpoint_dir: Path) -> tuple[Path | None, int, str | None]:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        return None, 0, f"huggingface_hub import failed: {exc}"

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=filename,
            local_dir=str(checkpoint_dir),
            local_dir_use_symlinks=False,
            token=hf_token,
        )
    except Exception as exc:
        return None, 0, f"could not download HF checkpoint {filename} from {repo_id}: {exc}"

    file_match = STEP_FILE_RE.match(os.path.basename(filename))
    step = int(file_match.group(1)) if file_match else 0
    return Path(local_path), step, None


def _resolve_local_initialization_checkpoint(local_dir: str, filename: str) -> tuple[Path | None, int]:
    if not local_dir or not filename:
        return None, 0
    checkpoint_path = (Path(local_dir).expanduser() / filename).resolve()
    if not checkpoint_path.exists():
        return None, 0
    file_match = STEP_FILE_RE.match(checkpoint_path.name)
    step = int(file_match.group(1)) if file_match else 0
    return checkpoint_path, step


def _is_missing_hf_repo_error(error: str | None) -> bool:
    if not error:
        return False
    return "Repository Not Found" in error or "404 Client Error" in error


def _ensure_base_model(model: str, base_model_dir: Path, base_model_repo_id: str | None) -> dict[str, Any]:
    info = {
        "base_model_dir": str(base_model_dir),
        "base_model_repo_id": base_model_repo_id,
        "base_model_downloaded": False,
    }
    if _has_files(base_model_dir):
        return info
    if not base_model_repo_id:
        raise ValueError(f"base model directory is missing and no repo id was provided: {base_model_dir}")

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required to download base model weights") from exc

    base_model_dir.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    snapshot_download(
        repo_id=base_model_repo_id,
        repo_type="model",
        local_dir=str(base_model_dir),
        token=hf_token,
    )
    info["base_model_downloaded"] = True
    return info


def _starvla_runtime_cache_paths(dataset_dir: Path) -> dict[str, Path]:
    return {
        "dataset_statistics": dataset_dir / "dataset_statistics.json",
        "stats": dataset_dir / "meta" / "stats_gr00t.json",
        "steps": dataset_dir / "meta" / "steps_data_index.pkl",
    }


def _starvla_runtime_cache_ready(dataset_dir: Path) -> bool:
    paths = _starvla_runtime_cache_paths(dataset_dir)
    return all(path.exists() for path in paths.values())


def _read_starvla_runtime_cache_summary(data_root_dir: Path, data_mix: str) -> dict[str, Any]:
    from starVLA.dataloader.gr00t_lerobot.registry import (
        ROBOT_TYPE_TO_EMBODIMENT_TAG,
        get_dataset_named_mixture,
    )

    total_steps = 0
    total_trajectories = 0
    first_stats: dict[str, Any] | None = None
    for dataset_name, _, robot_type in get_dataset_named_mixture(data_mix):
        paths = _starvla_runtime_cache_paths(data_root_dir / dataset_name)
        with paths["steps"].open("rb") as stream:
            steps_cache = pickle.load(stream)
        total_steps += steps_cache["total_steps"]
        total_trajectories += steps_cache["num_trajectories"]
        if first_stats is None:
            first_stats = {
                "dataset_stats_path": str(paths["dataset_statistics"]),
                "dataset_robot_type": robot_type,
                "dataset_embodiment_tag": str(ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]),
            }
    assert first_stats is not None
    return {
        **first_stats,
        "dataset_num_steps": total_steps,
        "dataset_num_trajectories": total_trajectories,
    }


def _materialize_starvla_runtime_cache(data_root_dir: Path, data_mix: str) -> dict[str, Any]:
    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
    from starVLA.dataloader.gr00t_lerobot.registry import (
        ROBOT_TYPE_TO_EMBODIMENT_TAG,
        get_dataset_named_mixture,
    )

    mixture = get_dataset_named_mixture(data_mix)
    total_steps = 0
    total_trajectories = 0
    first_stats: dict[str, Any] | None = None
    for dataset_name, _, robot_type in mixture:
        data_cfg = {
            "include_state": False,
            "video_backend": "torchvision_av",
            "lerobot_version": "v2.0",
        }
        if str(robot_type) in {"rl_games_gymnasium", "rl_games_gymnasium_discrete", "rl_games_gymnasium_native"}:
            manifest = json.loads((data_root_dir / dataset_name / "manifest.json").read_text(encoding="utf-8"))
            data_cfg["gymnasium_task_contract"] = manifest["gymnasium_task"]
            data_cfg["active_action_dim"] = manifest["active_action_dim"]
        dataset = make_LeRobotSingleDataset(
            data_root_dir=data_root_dir,
            data_name=dataset_name,
            robot_type=robot_type,
            data_cfg=data_cfg,
        )
        stats_path = data_root_dir / dataset_name / "dataset_statistics.json"
        dataset._save_dataset_statistics_(stats_path)
        total_steps += len(dataset)
        total_trajectories += len(dataset.trajectory_ids)
        if first_stats is None:
            first_stats = {
                "dataset_stats_path": str(stats_path),
                "dataset_robot_type": robot_type,
                "dataset_embodiment_tag": str(ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]),
            }
    assert first_stats is not None
    return {
        **first_stats,
        "dataset_num_steps": total_steps,
        "dataset_num_trajectories": total_trajectories,
    }


def _validate_starvla_dataset(data_root_dir: Path, data_mix: str) -> dict[str, Any]:
    from starVLA.dataloader.gr00t_lerobot.registry import get_dataset_named_mixture

    mixture = get_dataset_named_mixture(data_mix)
    if all(_starvla_runtime_cache_ready(data_root_dir / dataset_name) for dataset_name, _, _ in mixture):
        return _read_starvla_runtime_cache_summary(data_root_dir=data_root_dir, data_mix=data_mix)
    return _materialize_starvla_runtime_cache(data_root_dir=data_root_dir, data_mix=data_mix)


def _initialization_mode(args) -> str:
    return str(getattr(args, "initialization_mode", None) or "scratch").lower()


def _action_carrier(args) -> str:
    configured = str(getattr(args, "action_carrier", "") or "").lower()
    if configured in {"native", "bridge"}:
        return configured
    if _initialization_mode(args) in {"pre-trained", "pretrained", "bridge"}:
        return "bridge"
    return "native"


def _carrier_dataset_name(data_mix: str, action_carrier: str) -> str:
    if action_carrier != "bridge" or "__bridge" in data_mix:
        return data_mix
    match = DEBUG_DATASET_RE.match(data_mix)
    if match:
        return f"{match.group('base')}__bridge{match.group('debug')}"
    return f"{data_mix}__bridge"


def _gymnasium_task_contract(args) -> dict[str, Any] | None:
    value = getattr(args, "gymnasium_task_contract", None)
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        return json.loads(json.dumps(value))
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def _single_task_robot_type(env_name: str) -> str:
    mapping = {
        "flappy": "rl_games_flappy",
        "demon_attack": "rl_games_demon_attack",
        "defend_the_line": "rl_games_defend_the_line",
        "deadly_corridor": "rl_games_deadly_corridor",
        "asterix": "rl_games_asterix",
        "atlantis": "rl_games_atlantis",
        "gymnasium": "rl_games_gymnasium",
    }
    if env_name not in mapping:
        raise ValueError(f"Unsupported single-task env for generated mixture registration: {env_name!r}")
    return mapping[env_name]


def _write_single_task_mixture_file(
    *,
    data_root_dir: Path,
    data_mix: str,
    eval_data_mix: str,
    robot_type: str,
) -> Path:
    from starVLA.dataloader.gr00t_lerobot.registry import load_custom_mixtures

    payload = {
        data_mix: [[data_mix, 1.0, robot_type]],
        eval_data_mix: [[eval_data_mix, 1.0, robot_type]],
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    custom_mixtures_path = data_root_dir / "_generated_mixtures" / f"single_{digest}.json"
    custom_mixtures_path.parent.mkdir(parents=True, exist_ok=True)
    custom_mixtures_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    load_custom_mixtures(custom_mixtures_path)
    return custom_mixtures_path


def _ensure_rl_games_lerobot_dataset(
    args,
    *,
    convert_dataset,
    verify_dataset,
    env_name: str,
    env_fps: float,
    obs_fps: float,
) -> dict[str, Any]:
    data_root_dir = Path(args.dataset_local_dir).expanduser().resolve()
    action_carrier = _action_carrier(args)
    action_layout = str(
        getattr(args, "action_layout", "")
        or getattr(args, "deadly_action_layout", "")
        or ""
    )
    source_dataset = str(getattr(args, "source_dataset_hf", "") or "")
    source_config_name = getattr(args, "source_dataset_config_name", None)
    source_config_name = None if source_config_name in (None, "") else str(source_config_name)
    source_subdir = args.source_dataset_subdir
    source_revision = getattr(args, "source_dataset_revision", None)
    required_latencies = getattr(args, "latency_filter", None)
    source_for_reading = _fixed_dataset_source(
        source_dataset, source_revision, source_subdir, required_latencies, args.dataset_cache_dir,
    )
    data_mix = _carrier_dataset_name(args.converted_dataset_name, action_carrier)
    dataset_dir = data_root_dir / data_mix
    eval_data_mix = f"{data_mix}__val"
    eval_dataset_dir = data_root_dir / eval_data_mix
    custom_mixtures_path = _write_single_task_mixture_file(
        data_root_dir=data_root_dir,
        data_mix=data_mix,
        eval_data_mix=eval_data_mix,
        robot_type=_single_task_robot_type(env_name),
    )
    force = _str2bool(args.setup_force) or _str2bool(args.dataset_force_download)
    mixed_latency = args.mode == "mixed_latency" or str(args.latency_mode or "").lower() == "mixed"
    source_latency_column = "latency_raw_frames"
    target_latency_unit = args.target_latency_unit
    obs_stride_raw_frames = round(env_fps / obs_fps)
    prompt_map = dataset_dir / "latency_prompt_map.json"
    episodes_per_latency_by_latency = _parse_latency_episode_map(
        getattr(args, "episodes_per_latency_by_latency", None)
    )

    def _manifest_mismatch_reasons(dataset_path: Path) -> list[str]:

        reasons: list[str] = []
        manifest_path = dataset_path / "manifest.json"
        if not manifest_path.exists():
            return reasons
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return [f"{manifest_path} is unreadable: {exc}"]
        if str(manifest.get("action_carrier", "native")) != action_carrier:
            reasons.append(
                f"action_carrier mismatch: manifest={manifest.get('action_carrier', 'native')!r} expected={action_carrier!r}"
            )
        if source_revision is not None and ("source_revision" not in manifest or manifest["source_revision"] != source_revision):
            reasons.append("source_revision mismatch")
        if (manifest.get("source_config") or None) != source_config_name:
            reasons.append(
                f"source_config mismatch: manifest={(manifest.get('source_config') or None)!r} expected={source_config_name!r}"
            )
        if (manifest.get("source_subdir") or None) != source_subdir:
            reasons.append(
                f"source_subdir mismatch: manifest={(manifest.get('source_subdir') or None)!r} expected={source_subdir!r}"
            )
        if action_layout and ("action_layout" not in manifest or str(manifest["action_layout"]) != action_layout):
            reasons.append(
                f"action_layout mismatch: manifest={manifest.get('action_layout')!r} expected={action_layout!r}"
            )
        if mixed_latency and manifest.get("latency_metadata") is not True:
            reasons.append("latency_metadata missing/false for mixed-latency run")
        if manifest.get("fps") != obs_fps:
            reasons.append(f"fps mismatch: manifest={manifest.get('fps')!r} expected={obs_fps!r}")
        if manifest.get("obs_stride_raw_frames") != obs_stride_raw_frames:
            reasons.append(
                f"obs_stride_raw_frames mismatch: manifest={manifest.get('obs_stride_raw_frames')!r} expected={obs_stride_raw_frames!r}"
            )
        if manifest.get("target_latency_unit") != target_latency_unit:
            reasons.append(
                f"target_latency_unit mismatch: manifest={manifest.get('target_latency_unit')!r} expected={target_latency_unit!r}"
            )
        if manifest.get("source_latency_column") != source_latency_column:
            reasons.append(
                f"source_latency_column mismatch: manifest={manifest.get('source_latency_column')!r} expected={source_latency_column!r}"
            )
        expected_latency_filter = getattr(args, "latency_filter", None)
        if expected_latency_filter is not None:
            manifest_latency_filter = manifest.get("latency_filter")
            if manifest_latency_filter != [int(value) for value in expected_latency_filter]:
                reasons.append(
                    f"latency_filter mismatch: manifest={manifest_latency_filter!r} "
                    f"expected={[int(value) for value in expected_latency_filter]!r}"
                )
            expected_subdirs = [str(s) for s in resolve_latency_subdirs(source_subdir, expected_latency_filter)]
            if manifest.get("latency_subdirs") != expected_subdirs:
                reasons.append(
                    f"latency_subdirs mismatch: manifest={manifest.get('latency_subdirs')!r} expected={expected_subdirs!r}"
                )
        expected_episodes_per_latency = getattr(args, "episodes_per_latency", None)
        if expected_episodes_per_latency is not None:
            if manifest.get("episodes_per_latency") != int(expected_episodes_per_latency):
                reasons.append(
                    f"episodes_per_latency mismatch: manifest={manifest.get('episodes_per_latency')!r} "
                    f"expected={int(expected_episodes_per_latency)!r}"
                )
        if episodes_per_latency_by_latency is not None:
            expected_map = {str(k): int(v) for k, v in sorted(episodes_per_latency_by_latency.items())}
            if manifest.get("episodes_per_latency_by_latency") != expected_map:
                reasons.append(
                    "episodes_per_latency_by_latency mismatch: "
                    f"manifest={manifest.get('episodes_per_latency_by_latency')!r} expected={expected_map!r}"
                )
        expected_gymnasium_task = _gymnasium_task_contract(args)
        if expected_gymnasium_task is not None and manifest.get("gymnasium_task") != expected_gymnasium_task:
            reasons.append("gymnasium_task contract mismatch")
        expected_max_episodes = getattr(args, "max_episodes", None)
        if expected_max_episodes is not None:
            if manifest.get("max_episodes") != int(expected_max_episodes):
                reasons.append(
                    f"max_episodes mismatch: manifest={manifest.get('max_episodes')!r} expected={int(expected_max_episodes)!r}"
                )
        expected_max_steps_per_episode = getattr(args, "max_steps_per_episode", None)
        expected_max_steps_value = (
            int(expected_max_steps_per_episode)
            if expected_max_steps_per_episode not in (None, "")
            else None
        )
        if manifest.get("max_steps_per_episode") != expected_max_steps_value:
            reasons.append(
                "max_steps_per_episode mismatch: "
                f"manifest={manifest.get('max_steps_per_episode')!r} expected={expected_max_steps_value!r}"
            )
        return reasons

    def _manifest_matches(dataset_path: Path) -> bool:
        return not _manifest_mismatch_reasons(dataset_path)

    def _mixed_prompt_map_ready() -> bool:
        if not mixed_latency:
            return True
        if not prompt_map.exists():
            return _dataset_ready(dataset_dir)
        try:
            mapping = json.loads(prompt_map.read_text(encoding="utf-8"))
        except Exception:
            return _dataset_ready(dataset_dir)
        if len(mapping) > 1 and _prompt_map_has_raw_frame_entries(prompt_map):
            return True
        return _dataset_ready(dataset_dir)

    def _source_prompt_map(latencies: list[int] | None = None) -> dict[str, dict[str, Any]]:
        if not source_dataset:
            return {}
        prompt_source_for_reading = _fixed_dataset_source(
            source_dataset, source_revision, source_subdir, latencies, args.dataset_cache_dir,
        )
        return _load_source_latency_prompt_map(
            prompt_source_for_reading,
            cache_dir=getattr(args, "dataset_cache_dir", None),
            dataset_config_name=source_config_name,
            dataset_source_subdir=source_subdir,
            latencies=latencies,
            source_latency_column=source_latency_column,
            target_latency_unit=target_latency_unit,
            obs_stride_raw_frames=obs_stride_raw_frames,
        )

    def _remake_prompt_map_if_needed(*, required_latencies: list[int] | None = None) -> None:
        existing_map = _normalize_prompt_map_entries(
            _read_prompt_map(prompt_map),
            target_latency_unit=target_latency_unit,
            obs_stride_raw_frames=obs_stride_raw_frames,
        )
        required_keys = {str(int(value)) for value in (required_latencies or [])}
        prompt_map_required = mixed_latency or bool(required_keys) or prompt_map.exists()
        if not prompt_map_required:
            return
        needs_remake = (
            not _prompt_map_has_raw_frame_entries(prompt_map)
            or (mixed_latency and _prompt_map_entry_count(prompt_map) <= 1)
            or not required_keys.issubset(existing_map)
        )
        if not needs_remake:
            return

        local_map = _load_local_latency_prompt_map(
            dataset_dir,
            target_latency_unit=target_latency_unit,
            obs_stride_raw_frames=obs_stride_raw_frames,
        )
        merged = {
            **existing_map,
            **_normalize_prompt_map_entries(
                local_map,
                target_latency_unit=target_latency_unit,
                obs_stride_raw_frames=obs_stride_raw_frames,
            ),
        }
        missing_required = [int(value) for value in sorted(required_keys, key=int) if value not in merged]
        if (
            (missing_required or (mixed_latency and len(merged) <= 1))
            and source_dataset
        ):
            merged = {
                **merged,
                **_normalize_prompt_map_entries(
                    _source_prompt_map(missing_required or None),
                    target_latency_unit=target_latency_unit,
                    obs_stride_raw_frames=obs_stride_raw_frames,
                ),
            }

        if not merged:
            if required_keys or mixed_latency:
                raise ValueError(
                    f"could not remake latency prompt map from local dataset: {prompt_map}. "
                    "Expected converted parquet files with latency/task_index columns and meta/tasks.jsonl."
                )
            return
        if mixed_latency and len(merged) <= 1:
            raise ValueError(
                f"remade latency prompt map has only {len(merged)} entry: {prompt_map}. "
                "Mixed-latency training/eval needs more than one latency prompt."
            )
        missing_required = [int(value) for value in sorted(required_keys, key=int) if value not in merged]
        if missing_required:
            raise ValueError(
                f"could not add required eval latencies {missing_required} to {prompt_map}; "
                f"available latencies are {sorted(int(key) for key in merged)}"
            )
        _write_sorted_prompt_map(prompt_map, merged)

    rebuild_reasons: list[str] = []
    if force:
        rebuild_reasons.append("dataset.setup_force or dataset.force_download is true")
    if not _dataset_ready(dataset_dir):
        rebuild_reasons.append(f"train dataset is not ready: {dataset_dir}")
    if not _dataset_ready(eval_dataset_dir):
        rebuild_reasons.append(f"eval dataset is not ready: {eval_dataset_dir}")
    for reason in _manifest_mismatch_reasons(dataset_dir):
        rebuild_reasons.append(f"train manifest: {reason}")
    for reason in _manifest_mismatch_reasons(eval_dataset_dir):
        rebuild_reasons.append(f"eval manifest: {reason}")
    if not _mixed_prompt_map_ready():
        rebuild_reasons.append(f"mixed latency prompt map is not ready: {prompt_map}")
    rebuild = bool(rebuild_reasons)
    converted = False
    if rebuild:
        print("[setup] rebuilding converted dataset because:", file=sys.stderr, flush=True)
        for reason in rebuild_reasons:
            print(f"[setup]   - {reason}", file=sys.stderr, flush=True)
        if not args.source_dataset_hf:
            raise ValueError(
                f"{dataset_dir} is not ready; pass --source-dataset-hf so setup can verify and convert it"
            )
        if not _str2bool(getattr(args, "skip_verification", "false")):
            verify_kwargs = {
                "rows": args.verify_rows,
                "cache_dir": args.dataset_cache_dir,
                "strict": True,
                "allow_mixed_latency_prompts": mixed_latency,
                "source_latency_column": source_latency_column,
                "target_latency_unit": target_latency_unit,
                "obs_stride_raw_frames": obs_stride_raw_frames,
            }
            if "dataset_config_name" in inspect.signature(verify_dataset).parameters:
                verify_kwargs["dataset_config_name"] = source_config_name
            if "dataset_source_subdir" in inspect.signature(verify_dataset).parameters:
                verify_kwargs["dataset_source_subdir"] = source_subdir
            if action_layout and "action_layout" in inspect.signature(verify_dataset).parameters:
                verify_kwargs["action_layout"] = action_layout
            if "latencies" in inspect.signature(verify_dataset).parameters:
                verify_kwargs["latencies"] = getattr(args, "latency_filter", None)
            if "gymnasium_task_contract" in inspect.signature(verify_dataset).parameters:
                verify_kwargs["gymnasium_task_contract"] = _gymnasium_task_contract(args)
            verify_dataset(source_for_reading, **verify_kwargs)
        convert_kwargs = {
            "cache_dir": args.dataset_cache_dir,
            "max_episodes": args.max_episodes,
            "force": rebuild,
            "require_latency_prompt_map": mixed_latency,
            "fps": obs_fps,
            "obs_stride_raw_frames": obs_stride_raw_frames,
            "source_latency_column": source_latency_column,
            "target_latency_unit": target_latency_unit,
            "source_rows_unit": "decision_step",
        }
        if "dataset_config_name" in inspect.signature(convert_dataset).parameters:
            convert_kwargs["dataset_config_name"] = source_config_name
        if "dataset_source_subdir" in inspect.signature(convert_dataset).parameters:
            convert_kwargs["dataset_source_subdir"] = source_subdir
        if "latency_filter" in inspect.signature(convert_dataset).parameters:
            convert_kwargs["latency_filter"] = getattr(args, "latency_filter", None)
        if "episodes_per_latency" in inspect.signature(convert_dataset).parameters:
            convert_kwargs["episodes_per_latency"] = getattr(args, "episodes_per_latency", None)
        if "episodes_per_latency_by_latency" in inspect.signature(convert_dataset).parameters:
            convert_kwargs["episodes_per_latency_by_latency"] = episodes_per_latency_by_latency
        if "max_steps_per_episode" in inspect.signature(convert_dataset).parameters:
            convert_kwargs["max_steps_per_episode"] = getattr(args, "max_steps_per_episode", None)
        if "action_carrier" in inspect.signature(convert_dataset).parameters:
            convert_kwargs["action_carrier"] = action_carrier
        if action_layout and "action_layout" in inspect.signature(convert_dataset).parameters:
            convert_kwargs["action_layout"] = action_layout
        if "gymnasium_task_contract" in inspect.signature(convert_dataset).parameters:
            convert_kwargs["gymnasium_task_contract"] = _gymnasium_task_contract(args)
        convert_dataset(source_for_reading, dataset_dir, **convert_kwargs)
        _record_fixed_dataset_source(dataset_dir, source_dataset, source_revision)
        _record_fixed_dataset_source(eval_dataset_dir, source_dataset, source_revision)
        converted = True
        if mixed_latency and not _mixed_prompt_map_ready():
            raise ValueError(
                f"mixed-latency dataset conversion did not create a usable prompt map: {prompt_map}. "
                "Check that the selected training episodes contain latency/prompt columns for more than one latency."
            )
        validation = _materialize_starvla_runtime_cache(data_root_dir=data_root_dir, data_mix=data_mix)
        eval_validation = _materialize_starvla_runtime_cache(data_root_dir=data_root_dir, data_mix=eval_data_mix)
    else:
        validation = _validate_starvla_dataset(data_root_dir=data_root_dir, data_mix=data_mix)
        eval_validation = _validate_starvla_dataset(data_root_dir=data_root_dir, data_mix=eval_data_mix)

    # If training used a latency_filter the embedded prompt map only has the filtered
    # latencies. Load the missing latency subdirs from HF (already cached for training
    # latencies; downloads only the needed eval-only ones) and fill in the map so that
    # post-train eval at non-training latencies gets the correct dataset prompt.
    latency_filter = getattr(args, "latency_filter", None)
    eval_latencies = getattr(args, "eval_latencies", None)
    all_needed = sorted({
        *(int(v) for v in (latency_filter or [])),
        *(int(v) for v in (eval_latencies or [])),
    })
    _remake_prompt_map_if_needed(required_latencies=all_needed)

    return {
        "dataset_ready": True,
        "dataset_converted": converted,
        "dataset_local_dir": str(data_root_dir),
        "dataset_dir": str(dataset_dir),
        "data_mix": data_mix,
        "eval_data_mix": eval_data_mix,
        "custom_mixtures_path": str(custom_mixtures_path),
        "eval_dataset_dir": str(eval_dataset_dir),
        "action_carrier": action_carrier,
        "latency_prompt_map_path": str(prompt_map) if prompt_map.exists() else None,
        **validation,
        "eval_dataset_stats_path": eval_validation["dataset_stats_path"],
        "eval_dataset_num_steps": eval_validation["dataset_num_steps"],
        "eval_dataset_num_trajectories": eval_validation["dataset_num_trajectories"],
    }


def _ensure_flappy_dataset(args) -> dict[str, Any]:
    from examples.rl_games.bash_scripts.gr00t.data_conversion.convert_flappy_to_starvla_lerobot import convert_dataset
    from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_flappy_dataset import verify_dataset

    return _ensure_rl_games_lerobot_dataset(
        args,
        convert_dataset=convert_dataset,
        verify_dataset=verify_dataset,
        env_name="flappy",
        env_fps=30.0,
        obs_fps=30.0,
    )


def _ensure_demon_attack_dataset(args) -> dict[str, Any]:
    from examples.rl_games.bash_scripts.gr00t.data_conversion.convert_demon_attack_to_starvla_lerobot import convert_dataset
    from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_demon_attack_dataset import verify_dataset

    return _ensure_rl_games_lerobot_dataset(
        args,
        convert_dataset=convert_dataset,
        verify_dataset=verify_dataset,
        env_name="demon_attack",
        env_fps=60.0,
        obs_fps=15.0,
    )


def _ensure_defend_the_line_dataset(args) -> dict[str, Any]:
    from examples.rl_games.bash_scripts.gr00t.data_conversion.convert_defend_the_line_to_starvla_lerobot import convert_dataset
    from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_defend_the_line_dataset import verify_dataset

    return _ensure_rl_games_lerobot_dataset(
        args,
        convert_dataset=convert_dataset,
        verify_dataset=verify_dataset,
        env_name="defend_the_line",
        env_fps=35.0,
        obs_fps=8.75,
    )


def _ensure_asterix_dataset(args) -> dict[str, Any]:
    from examples.rl_games.bash_scripts.gr00t.data_conversion.convert_asterix_to_starvla_lerobot import convert_dataset
    from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_asterix_dataset import verify_dataset

    return _ensure_rl_games_lerobot_dataset(
        args,
        convert_dataset=convert_dataset,
        verify_dataset=verify_dataset,
        env_name="asterix",
        env_fps=60.0,
        obs_fps=15.0,
    )


def _ensure_atlantis_dataset(args) -> dict[str, Any]:
    from examples.rl_games.bash_scripts.gr00t.data_conversion.convert_atlantis_to_starvla_lerobot import convert_dataset
    from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_atlantis_dataset import verify_dataset

    return _ensure_rl_games_lerobot_dataset(
        args,
        convert_dataset=convert_dataset,
        verify_dataset=verify_dataset,
        env_name="atlantis",
        env_fps=60.0,
        obs_fps=15.0,
    )


def _ensure_gymnasium_dataset(args) -> dict[str, Any]:
    from examples.rl_games.bash_scripts.gr00t.data_conversion.convert_gymnasium_to_starvla_lerobot import convert_dataset
    from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_gymnasium_dataset import verify_dataset

    contract = _gymnasium_task_contract(args)
    if contract is None:
        raise ValueError("rl_games.gymnasium.task_contract is required to setup a Gymnasium StarVLA dataset")
    env_fps = float(contract.get("env_fps", 25.0))
    obs_fps = float(contract.get("obs_fps", env_fps))
    return _ensure_rl_games_lerobot_dataset(
        args,
        convert_dataset=convert_dataset,
        verify_dataset=verify_dataset,
        env_name="gymnasium",
        env_fps=env_fps,
        obs_fps=obs_fps,
    )


def _task_converter_and_verifier(task: str):
    if task == "flappy":
        from examples.rl_games.bash_scripts.gr00t.data_conversion.convert_flappy_to_starvla_lerobot import convert_dataset
        from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_flappy_dataset import verify_dataset
        return convert_dataset, verify_dataset, "rl_games_flappy"
    if task == "demon_attack":
        from examples.rl_games.bash_scripts.gr00t.data_conversion.convert_demon_attack_to_starvla_lerobot import convert_dataset
        from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_demon_attack_dataset import verify_dataset
        return convert_dataset, verify_dataset, "rl_games_demon_attack"
    if task == "defend_the_line":
        from examples.rl_games.bash_scripts.gr00t.data_conversion.convert_defend_the_line_to_starvla_lerobot import convert_dataset
        from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_defend_the_line_dataset import verify_dataset
        return convert_dataset, verify_dataset, "rl_games_defend_the_line"
    if task == "deadly_corridor":
        from examples.rl_games.bash_scripts.gr00t.data_conversion.convert_deadly_corridor_to_starvla_lerobot import convert_dataset
        from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_deadly_corridor_dataset import verify_dataset
        return convert_dataset, verify_dataset, "rl_games_deadly_corridor"
    if task == "asterix":
        from examples.rl_games.bash_scripts.gr00t.data_conversion.convert_asterix_to_starvla_lerobot import convert_dataset
        from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_asterix_dataset import verify_dataset
        return convert_dataset, verify_dataset, "rl_games_asterix"
    if task == "atlantis":
        from examples.rl_games.bash_scripts.gr00t.data_conversion.convert_atlantis_to_starvla_lerobot import convert_dataset
        from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_atlantis_dataset import verify_dataset
        return convert_dataset, verify_dataset, "rl_games_atlantis"
    raise ValueError(
        "cross-task rl_games currently supports only flappy, demon_attack, "
        f"defend_the_line, deadly_corridor, asterix, and atlantis, got {task!r}"
    )


def _get_task_value(task_cfg: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in task_cfg and task_cfg[name] not in (None, ""):
            return task_cfg[name]
    return default


def _ensure_cross_task_datasets(args) -> dict[str, Any]:
    from starVLA.dataloader.gr00t_lerobot.registry import load_custom_mixtures

    cross_cfg = getattr(args, "cross_task", None) or {}
    train_tasks = cross_cfg.get("train_tasks") if isinstance(cross_cfg, dict) else None
    if not train_tasks:
        raise ValueError("rl_games.cross_task.train_tasks must contain at least flappy and demon_attack entries")

    data_root_dir = Path(args.dataset_local_dir).expanduser().resolve()
    action_carrier = _action_carrier(args)
    if action_carrier != "bridge":
        raise ValueError("cross-task OpenVLA RL-games currently requires action_carrier=bridge")

    mixture_entries: list[list[Any]] = []
    val_mixture_entries: list[list[Any]] = []
    prompt_maps: dict[str, str] = {}
    converted_names: dict[str, str] = {}
    converted = False
    force = _str2bool(args.setup_force) or _str2bool(args.dataset_force_download)

    for raw_task_cfg in train_tasks:
        task_cfg = dict(raw_task_cfg)
        task_name = str(_get_task_value(task_cfg, "name", "task"))
        convert_dataset, verify_dataset, robot_type = _task_converter_and_verifier(task_name)

        train_source_value = _get_task_value(task_cfg, "train_source_hf", "source_hf")
        prompt_source_value = _get_task_value(task_cfg, "prompt_source_hf", default=train_source_value)
        train_config_value = _get_task_value(task_cfg, "train_config_name", "source_config_name", "config_name", default=None)
        prompt_config_value = _get_task_value(task_cfg, "prompt_config_name", default=train_config_value)
        train_subdir_value = _get_task_value(task_cfg, "train_source_subdir", "source_subdir", default=None)
        prompt_subdir_value = _get_task_value(task_cfg, "prompt_source_subdir", default=train_subdir_value)
        train_source = str(train_source_value or "")
        prompt_source = str(prompt_source_value or "")
        train_config_name = None if train_config_value in (None, "") else str(train_config_value)
        prompt_config_name = None if prompt_config_value in (None, "") else str(prompt_config_value)
        train_subdir = train_subdir_value
        prompt_subdir = prompt_subdir_value
        if not train_source:
            raise ValueError(f"cross-task train task {task_name} is missing train_source_hf/source_hf")
        if not prompt_source:
            raise ValueError(f"cross-task train task {task_name} is missing prompt_source_hf")

        latency_filter = _get_task_value(task_cfg, "train_latency_filter", "latency_filter", default=None)
        latency_filter = [int(value) for value in latency_filter] if latency_filter not in (None, "") else None
        eval_latency_filter = _get_task_value(task_cfg, "eval_latency_filter", default=latency_filter)
        eval_latency_filter = (
            [int(value) for value in eval_latency_filter]
            if eval_latency_filter not in (None, "")
            else None
        )
        required_latencies = sorted({int(v) for v in (latency_filter or []) + (eval_latency_filter or [])}) or None
        episodes_per_latency = _get_task_value(task_cfg, "episodes_per_latency", default=None)
        episodes_per_latency = int(episodes_per_latency) if episodes_per_latency not in (None, "") else None
        eval_episodes_per_latency = _get_task_value(task_cfg, "eval_episodes_per_latency", default=episodes_per_latency)
        eval_episodes_per_latency = (
            int(eval_episodes_per_latency)
            if eval_episodes_per_latency not in (None, "")
            else None
        )
        max_episodes = _get_task_value(task_cfg, "max_episodes", default=None)
        max_episodes = int(max_episodes) if max_episodes not in (None, "") else None
        weight = float(_get_task_value(task_cfg, "weight", default=1.0))
        action_layout = _get_task_value(task_cfg, "action_layout", default=None)
        action_layout = None if action_layout in (None, "") else str(action_layout)
        task_timing = {
            "flappy": (30.0, 1),
            "demon_attack": (15.0, 4),
            "defend_the_line": (8.75, 4),
            "deadly_corridor": (8.75, 4),
            "asterix": (15.0, 4),
            "atlantis": (15.0, 4),
        }[task_name]
        task_fps, task_obs_stride_raw_frames = task_timing
        task_target_latency_unit = (
            "raw_frames" if task_name in {"asterix", "atlantis", "defend_the_line"} else "observation_steps"
        )

        base_converted_name = str(_get_task_value(task_cfg, "converted_name", default=f"{task_name}_cross_task_train"))
        converted_base = _derived_dataset_name(
            base_converted_name,
            latency_filter=latency_filter,
            episodes_per_latency=episodes_per_latency,
            max_episodes=max_episodes,
        )
        data_mix = _carrier_dataset_name(converted_base, action_carrier)
        dataset_dir = data_root_dir / data_mix
        eval_data_mix = f"{data_mix}__val"
        eval_dataset_dir = data_root_dir / eval_data_mix

        train_revision = _get_task_value(task_cfg, "train_revision", default=None)
        prompt_revision = _get_task_value(task_cfg, "prompt_revision", default=None)
        train_for_reading = _fixed_dataset_source(
            train_source, train_revision, train_subdir, required_latencies, args.dataset_cache_dir,
        )
        prompt_for_reading = _fixed_dataset_source(
            prompt_source, prompt_revision, prompt_subdir, required_latencies, args.dataset_cache_dir,
        )
        prompt_map = _load_source_latency_prompt_map(
            prompt_for_reading,
            cache_dir=args.dataset_cache_dir,
            dataset_config_name=prompt_config_name,
            dataset_source_subdir=prompt_subdir,
            source_latency_column="latency_raw_frames",
            target_latency_unit=task_target_latency_unit,
            obs_stride_raw_frames=task_obs_stride_raw_frames,
            latencies=required_latencies,
        )
        prompt_dir = data_root_dir / "_prompt_maps" / data_mix
        eval_prompt_map_path = _write_prompt_map(prompt_dir / "eval_latency_prompt_map.json", prompt_map)
        train_prompt_map_path = _write_prompt_map(
            prompt_dir / "train_latency_prompt_map.json",
            prompt_map,
            latency_filter=latency_filter,
        )

        def _manifest_matches(
            dataset_path: Path,
            *,
            expected_latency_filter: list[int] | None,
            expected_episodes_per_latency: int | None,
        ) -> bool:

            manifest_path = dataset_path / "manifest.json"
            if not manifest_path.exists():
                return False
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                return False
            expected_subdirs = [str(s) for s in resolve_latency_subdirs(train_subdir, expected_latency_filter)]
            return (
                str(manifest.get("source", "")) == train_source
                and (train_revision is None or ("source_revision" in manifest and manifest["source_revision"] == train_revision))
                and (prompt_revision is None or ("prompt_source_revision" in manifest and manifest["prompt_source_revision"] == prompt_revision))
                and str(manifest.get("prompt_source", "")) == prompt_source
                and (manifest.get("source_config") or None) == train_config_name
                and (manifest.get("prompt_source_config") or None) == prompt_config_name
                and (manifest.get("source_subdir") or None) == train_subdir
                and (manifest.get("prompt_source_subdir") or None) == prompt_subdir
                and manifest.get("latency_subdirs") == expected_subdirs
                and str(manifest.get("action_carrier", "")) == action_carrier
                and (not action_layout or str(manifest.get("action_layout", "")) == action_layout)
                and manifest.get("latency_filter") == expected_latency_filter
                and manifest.get("episodes_per_latency") == expected_episodes_per_latency
                and manifest.get("max_episodes") == max_episodes
                and manifest.get("fps") == task_fps
                and manifest.get("obs_stride_raw_frames") == task_obs_stride_raw_frames
                and manifest.get("target_latency_unit") == task_target_latency_unit
                and manifest.get("source_latency_column") == "latency_raw_frames"
            )

        rebuild = (
            force
            or not _dataset_ready(dataset_dir)
            or not _dataset_ready(eval_dataset_dir)
            or not _manifest_matches(
                dataset_dir,
                expected_latency_filter=latency_filter,
                expected_episodes_per_latency=episodes_per_latency,
            )
            or not _manifest_matches(
                eval_dataset_dir,
                expected_latency_filter=eval_latency_filter,
                expected_episodes_per_latency=eval_episodes_per_latency,
            )
        )
        if rebuild:
            allow_mixed = bool(latency_filter or eval_latency_filter) and train_source == prompt_source
            verify_kwargs = {
                "rows": args.verify_rows,
                "cache_dir": args.dataset_cache_dir,
                "dataset_config_name": train_config_name,
                "dataset_source_subdir": train_subdir,
                "strict": True,
                "allow_mixed_latency_prompts": allow_mixed,
                "source_latency_column": "latency_raw_frames",
                "target_latency_unit": task_target_latency_unit,
                "obs_stride_raw_frames": task_obs_stride_raw_frames,
            }
            if action_layout and "action_layout" in inspect.signature(verify_dataset).parameters:
                verify_kwargs["action_layout"] = action_layout
            if "latencies" in inspect.signature(verify_dataset).parameters:
                verify_kwargs["latencies"] = required_latencies
            verify_dataset(train_for_reading, **verify_kwargs)

            convert_kwargs = {
                "cache_dir": args.dataset_cache_dir,
                "dataset_config_name": train_config_name,
                "dataset_source_subdir": train_subdir,
                "max_episodes": max_episodes,
                "force": rebuild,
                "require_latency_prompt_map": bool(latency_filter or eval_latency_filter),
                "latency_filter": latency_filter,
                "train_latency_filter": latency_filter,
                "eval_latency_filter": eval_latency_filter,
                "episodes_per_latency": episodes_per_latency,
                "train_episodes_per_latency": episodes_per_latency,
                "eval_episodes_per_latency": eval_episodes_per_latency,
                "prompt_map_override": prompt_map,
                "default_latency": 0,
                "action_carrier": action_carrier,
                "fps": task_fps,
                "obs_stride_raw_frames": task_obs_stride_raw_frames,
                "source_latency_column": "latency_raw_frames",
                "target_latency_unit": task_target_latency_unit,
                "source_rows_unit": "decision_step",
            }
            if action_layout:
                convert_kwargs["action_layout"] = action_layout
            supported_convert_kwargs = inspect.signature(convert_dataset).parameters
            convert_kwargs = {
                key: value
                for key, value in convert_kwargs.items()
                if key in supported_convert_kwargs
            }
            convert_dataset(train_for_reading, dataset_dir, **convert_kwargs)
            for manifest_path in (dataset_dir / "manifest.json", eval_dataset_dir / "manifest.json"):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["source"] = train_source
                manifest["source_revision"] = train_revision
                manifest["prompt_source"] = prompt_source
                manifest["prompt_source_revision"] = prompt_revision
                manifest["prompt_source_config"] = prompt_config_name
                manifest["prompt_source_subdir"] = prompt_subdir
                manifest["eval_prompt_map_path"] = str(eval_prompt_map_path)
                manifest["train_prompt_map_path"] = str(train_prompt_map_path)
                manifest["train_latency_filter"] = latency_filter
                manifest["eval_latency_filter"] = eval_latency_filter
                manifest["train_episodes_per_latency"] = episodes_per_latency
                manifest["eval_episodes_per_latency"] = eval_episodes_per_latency
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            # The per-task derived names (e.g. ..._lat0_2_4__40ep_per_lat__bridge)
            # are not in any static data_config, and the combined cross__ mixture
            # is only registered after this loop. Register each as a trivial
            # single-dataset mixture so the per-task materialize below can resolve
            # it via get_dataset_named_mixture.
            from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES

            DATASET_NAMED_MIXTURES[data_mix] = [[data_mix, weight, robot_type]]
            DATASET_NAMED_MIXTURES[eval_data_mix] = [[eval_data_mix, weight, robot_type]]
            _materialize_starvla_runtime_cache(data_root_dir=data_root_dir, data_mix=data_mix)
            _materialize_starvla_runtime_cache(data_root_dir=data_root_dir, data_mix=eval_data_mix)
            converted = True

        mixture_entries.append([data_mix, weight, robot_type])
        val_mixture_entries.append([eval_data_mix, weight, robot_type])
        prompt_maps[task_name] = str(eval_prompt_map_path)
        converted_names[task_name] = data_mix

    mixture_name = "cross__" + "__".join(converted_names[task] for task in sorted(converted_names))
    mixture_name = _safe_token(mixture_name)
    eval_mixture_name = f"{mixture_name}__val"
    mixture_file_name = f"cross_{hashlib.sha1(mixture_name.encode('utf-8')).hexdigest()[:16]}.json"
    custom_mixtures_path = data_root_dir / "_generated_mixtures" / mixture_file_name
    custom_mixtures_path.parent.mkdir(parents=True, exist_ok=True)
    custom_mixtures_path.write_text(
        json.dumps({mixture_name: mixture_entries, eval_mixture_name: val_mixture_entries}, indent=2),
        encoding="utf-8",
    )
    load_custom_mixtures(custom_mixtures_path)
    validation = _validate_starvla_dataset(data_root_dir=data_root_dir, data_mix=mixture_name)
    eval_validation = _validate_starvla_dataset(data_root_dir=data_root_dir, data_mix=eval_mixture_name)
    return {
        "dataset_ready": True,
        "dataset_converted": converted,
        "dataset_local_dir": str(data_root_dir),
        "dataset_dir": str(data_root_dir),
        "data_mix": mixture_name,
        "eval_data_mix": eval_mixture_name,
        "custom_mixtures_path": str(custom_mixtures_path),
        "cross_task_prompt_maps": prompt_maps,
        "cross_task_datasets": converted_names,
        "action_carrier": action_carrier,
        **validation,
        "eval_dataset_stats_path": eval_validation["dataset_stats_path"],
        "eval_dataset_num_steps": eval_validation["dataset_num_steps"],
        "eval_dataset_num_trajectories": eval_validation["dataset_num_trajectories"],
    }


def _ensure_deadly_corridor_dataset(args) -> dict[str, Any]:
    from examples.rl_games.bash_scripts.gr00t.data_conversion.convert_deadly_corridor_to_starvla_lerobot import convert_dataset
    from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_deadly_corridor_dataset import verify_dataset

    return _ensure_rl_games_lerobot_dataset(
        args,
        convert_dataset=convert_dataset,
        verify_dataset=verify_dataset,
        env_name="deadly_corridor",
        env_fps=35.0,
        obs_fps=8.75,
    )


def setup_assets(args) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": args.model,
        "env": args.env,
        "mode": args.mode,
        "initialization_mode": _initialization_mode(args),
        "action_carrier": _action_carrier(args),
    }

    supported_models = {"openvla", "pi0", "pi05", "gr00t"}
    if args.model in supported_models and str(getattr(args, "env", "")) == "cross_task":
        result.update(_ensure_cross_task_datasets(args))
    elif args.model in supported_models and args.env == "flappy":
        result.update(_ensure_flappy_dataset(args))
    elif args.model in supported_models and args.env == "demon_attack":
        result.update(_ensure_demon_attack_dataset(args))
    elif args.model in supported_models and args.env == "defend_the_line":
        result.update(_ensure_defend_the_line_dataset(args))
    elif args.model in supported_models and args.env == "asterix":
        result.update(_ensure_asterix_dataset(args))
    elif args.model in supported_models and args.env == "atlantis":
        result.update(_ensure_atlantis_dataset(args))
    elif args.model in supported_models and args.env == "deadly_corridor":
        result.update(_ensure_deadly_corridor_dataset(args))
    elif args.model in supported_models and args.env == "gymnasium":
        result.update(_ensure_gymnasium_dataset(args))
    else:
        data_root_dir = Path(args.dataset_local_dir).expanduser().resolve()
        result.update({
            "dataset_ready": _has_files(data_root_dir),
            "dataset_local_dir": str(data_root_dir),
            "data_mix": None,
            "eval_data_mix": None,
            "latency_prompt_map_path": None,
        })

    base_model_dir = Path(args.base_model_dir).expanduser().resolve()
    result.update(_ensure_base_model(args.model, base_model_dir, args.base_model_repo_id))

    checkpoint_dir = Path(args.checkpoint_local_dir).expanduser().resolve()
    explicit_checkpoint = str(getattr(args, "checkpoint", "") or "")
    if explicit_checkpoint:
        resume_checkpoint, resume_step, resume_kind = _resolve_explicit_resume_checkpoint(explicit_checkpoint, checkpoint_dir)
        result.update({
            "resume_found": True,
            "resume_source": "explicit",
            "resume_kind": resume_kind,
            "resume_checkpoint": str(resume_checkpoint),
            "resume_step": resume_step,
            "checkpoint_local_dir": str(checkpoint_dir),
        })
        return result

    local_ckpt, local_step, local_kind = (None, 0, None)
    if args.checkpoint_load in {"auto", "local"}:
        local_ckpt, local_step, local_kind = _find_latest_local_checkpoint(checkpoint_dir)
        if local_ckpt is not None and args.checkpoint_load == "local":
            result.update({
                "resume_found": True,
                "resume_source": "local",
                "resume_kind": local_kind,
                "resume_checkpoint": str(local_ckpt),
                "resume_step": local_step,
                "checkpoint_local_dir": str(checkpoint_dir),
            })
            return result

    hf_repo_id = args.checkpoint_hf_repo_id or args.hf_repo_id
    if args.checkpoint_load in {"auto", "hf"} and hf_repo_id:
        hf_ckpt, hf_step, hf_kind, hf_error = _download_latest_hf_checkpoint(hf_repo_id, checkpoint_dir)
        hf_is_better = (
            args.checkpoint_load == "hf"
            or local_ckpt is None
            or hf_step > local_step
            or (hf_step == local_step and hf_kind == "state" and local_kind != "state")
        )
        if hf_ckpt is not None and hf_is_better:
            result.update({
                "resume_found": True,
                "resume_source": "hf",
                "resume_kind": hf_kind,
                "resume_checkpoint": str(hf_ckpt),
                "resume_step": hf_step,
                "checkpoint_local_dir": str(checkpoint_dir),
            })
            return result
        if local_ckpt is not None:
            result.update({
                "resume_found": True,
                "resume_source": "local",
                "resume_kind": local_kind,
                "resume_checkpoint": str(local_ckpt),
                "resume_step": local_step,
                "checkpoint_local_dir": str(checkpoint_dir),
            })
            if hf_ckpt is not None:
                result["checkpoint_hf_status"] = (
                    f"local checkpoint step {local_step} is newer than or equal to HF step {hf_step}; using local"
                )
            return result
        if hf_error:
            sync_repo_id = str(getattr(args, "checkpoint_sync_repo_id", "") or "")
            sync_enabled = _str2bool(getattr(args, "checkpoint_sync_enabled", False))
            if (
                args.checkpoint_load == "auto"
                and sync_enabled
                and sync_repo_id == hf_repo_id
            ):
                result["checkpoint_hf_status"] = (
                    f"HF resume repo {hf_repo_id} was not available during auto-resume; "
                    "starting from local/base model and using it as the checkpoint sync destination"
                )
            else:
                result["checkpoint_hf_warning"] = hf_error

    if local_ckpt is not None:
        result.update({
            "resume_found": True,
            "resume_source": "local",
            "resume_kind": local_kind,
            "resume_checkpoint": str(local_ckpt),
            "resume_step": local_step,
            "checkpoint_local_dir": str(checkpoint_dir),
        })
        return result

    initialization_hf_repo_id = str(getattr(args, "initialization_hf_repo_id", "") or "")
    initialization_local_dir = str(getattr(args, "initialization_local_dir", "") or "")
    initialization_checkpoint_filename = str(getattr(args, "initialization_checkpoint_filename", "") or "")
    has_initialization_source = bool(initialization_local_dir or initialization_hf_repo_id)
    if has_initialization_source and _initialization_mode(args) in INITIALIZATION_SOURCE_MODES:
        local_init_ckpt, local_init_step = _resolve_local_initialization_checkpoint(
            initialization_local_dir,
            initialization_checkpoint_filename,
        )
        if local_init_ckpt is not None:
            result.update({
                "resume_found": False,
                "resume_source": None,
                "resume_kind": None,
                "resume_checkpoint": None,
                "resume_step": 0,
                "checkpoint_local_dir": str(checkpoint_dir),
                "pretrained_checkpoint": str(local_init_ckpt),
                "initialization_source": "local",
                "initialization_local_dir": str(Path(initialization_local_dir).expanduser().resolve()),
                "initialization_hf_repo_id": initialization_hf_repo_id or None,
                "initialization_checkpoint_filename": initialization_checkpoint_filename or None,
                "initialization_step": local_init_step,
            })
            return result

        if not initialization_hf_repo_id:
            raise FileNotFoundError(
                f"Bridge initialization checkpoint not found under initialization_local_dir="
                f"{initialization_local_dir!r} with filename={initialization_checkpoint_filename!r}, "
                "and no initialization_hf_repo_id was provided for download."
            )

        init_dir = checkpoint_dir / "_initialization" / _safe_path_name(initialization_hf_repo_id)
        if initialization_checkpoint_filename:
            init_ckpt, init_step, init_error = _download_hf_checkpoint_file(
                initialization_hf_repo_id,
                initialization_checkpoint_filename,
                init_dir,
            )
            init_kind = "model" if init_ckpt is not None else None
        else:
            init_ckpt, init_step, init_kind, init_error = _download_latest_hf_checkpoint(initialization_hf_repo_id, init_dir)
        if init_ckpt is None:
            raise RuntimeError(
                f"Could not download bridge initialization checkpoint from {initialization_hf_repo_id}: "
                f"{init_error or 'no checkpoint files found'}"
            )
        if init_kind == "state":
            raise ValueError(
                f"Bridge initialization expected model weights, but {initialization_hf_repo_id} resolved "
                f"to a full training-state directory: {init_ckpt}"
            )
        result.update({
            "resume_found": False,
            "resume_source": None,
            "resume_kind": None,
            "resume_checkpoint": None,
            "resume_step": 0,
            "checkpoint_local_dir": str(checkpoint_dir),
            "pretrained_checkpoint": str(init_ckpt),
            "initialization_source": "hf",
            "initialization_local_dir": initialization_local_dir or None,
            "initialization_hf_repo_id": initialization_hf_repo_id,
            "initialization_checkpoint_filename": initialization_checkpoint_filename or None,
            "initialization_step": init_step,
        })
        return result

    result.update({
        "resume_found": False,
        "resume_source": None,
        "resume_kind": None,
        "resume_checkpoint": None,
        "resume_step": 0,
        "checkpoint_local_dir": str(checkpoint_dir),
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--initialization-mode", default="")
    parser.add_argument("--action-carrier", choices=["", "native", "bridge"], default="")
    parser.add_argument("--action-layout", default="")
    parser.add_argument("--deadly-action-layout", default="")
    parser.add_argument("--latency-mode", default="")
    parser.add_argument("--source-dataset-hf", default="")
    parser.add_argument("--source-dataset-revision", default=None)
    parser.add_argument("--source-dataset-config-name", default=None)
    parser.add_argument("--source-dataset-subdir", default=None)
    parser.add_argument("--dataset-local-dir", required=True)
    parser.add_argument("--converted-dataset-name", default="flappy_train")
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--dataset-force-download", default="false")
    parser.add_argument("--setup-force", default="false")
    parser.add_argument("--skip-verification", default="false")
    parser.add_argument(
        "--target-latency-unit",
        choices=["raw_frames", "observation_steps"],
        default="observation_steps",
    )
    parser.add_argument("--verify-rows", type=int, default=200)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--episodes-per-latency", type=int, default=None)
    parser.add_argument("--episodes-per-latency-by-latency", default=None)
    parser.add_argument("--latency-filter", default=None)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--base-model-repo-id", default=None)
    parser.add_argument("--checkpoint-local-dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint-load", choices=["auto", "none", "local", "hf"], default="auto")
    parser.add_argument("--checkpoint-hf-repo-id", default="")
    parser.add_argument("--checkpoint-save-best-model", default="true")
    parser.add_argument("--initialization-local-dir", default="")
    parser.add_argument("--initialization-hf-repo-id", default="")
    parser.add_argument("--initialization-checkpoint-filename", default="")
    parser.add_argument("--gymnasium-task-contract", default=None)
    parser.add_argument("--checkpoint-sync-enabled", default="false")
    parser.add_argument("--checkpoint-sync-repo-id", default="")
    parser.add_argument("--hf-repo-id", default="")
    args = parser.parse_args()
    if args.dataset_cache_dir == "":
        args.dataset_cache_dir = None
    if args.base_model_repo_id == "":
        args.base_model_repo_id = None
    if isinstance(args.latency_filter, str) and args.latency_filter:
        args.latency_filter = [int(item) for item in args.latency_filter.split(",") if item.strip()]
    elif args.latency_filter == "":
        args.latency_filter = None
    args.episodes_per_latency_by_latency = _parse_latency_episode_map(args.episodes_per_latency_by_latency)

    with contextlib.redirect_stdout(sys.stderr):
        result = setup_assets(args)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
