"""Contracts shared by the StarVLA dataset and task policy paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class StarVLATaskSpec:
    """Static dimensions and auxiliary columns for one latency-bench task."""

    name: str
    robot_type: str
    state_dim: int
    action_dim: int
    fps: float
    video_keys: tuple[str, ...]
    auxiliary_shapes: tuple[tuple[str, tuple[int, ...]], ...] = ()

    action_horizon: int = 1

    @property
    def action_key(self) -> str:
        return "issued_command_chunk" if self.action_horizon > 1 else "action"

    def auxiliary_shape_map(self) -> dict[str, tuple[int, ...]]:
        return dict(self.auxiliary_shapes)


TASK_SPECS: dict[str, StarVLATaskSpec] = {
    "balance_profile_h16": StarVLATaskSpec(
        name="humanoidbench_balance_simple",
        robot_type="latency_balance_profile_h16",
        auxiliary_shapes=(
            ("action_prefix_mask", (16,)),
            ("action_loss_mask", (16,)),
        ),
        action_horizon=16,
        state_dim=902,
        action_dim=43,
        fps=50.0,
        video_keys=("video.ego_view",),
    ),
    "balance_profile_h8": StarVLATaskSpec(
        name="humanoidbench_balance_simple",
        robot_type="latency_balance_profile_h8",
        auxiliary_shapes=(
            ("action_prefix_mask", (8,)),
            ("action_loss_mask", (8,)),
        ),
        action_horizon=8,
        state_dim=550,
        action_dim=43,
        fps=50.0,
        video_keys=("video.ego_view",),
    ),
    "balance": StarVLATaskSpec(
        name="humanoidbench_balance_simple",
        robot_type="latency_balance_h8",
        auxiliary_shapes=(("action_loss_mask", (8,)),),
        action_horizon=8,
        state_dim=198,
        action_dim=43,
        fps=50.0,
        video_keys=("video.ego_view",),
    ),
    "parkour": StarVLATaskSpec(
        name="extreme_parkour_go2",
        robot_type="latency_parkour_h1",
        state_dim=53,
        action_dim=34,
        fps=10.0,
        video_keys=("video.ego_view",),
        auxiliary_shapes=(
            ("actor_observation", (5, 753)),
            ("termination", (5,)),
        ),
    ),
    "mikasa": StarVLATaskSpec(
        name="mikasa_intercept_grab_fast",
        robot_type="latency_mikasa_h1",
        state_dim=7,
        action_dim=7,
        fps=20.0,
        video_keys=("video.top", "video.wrist"),
    ),
    "mikasa_h8": StarVLATaskSpec(
        name="mikasa_intercept_grab_fast",
        robot_type="latency_mikasa_h8",
        state_dim=7,
        action_dim=7,
        fps=20.0,
        video_keys=("video.top", "video.wrist"),
        auxiliary_shapes=(("action_loss_mask", (8,)),),
        action_horizon=8,
    ),
    "mikasa_conditioned_h8": StarVLATaskSpec(
        name="mikasa_intercept_grab_fast",
        robot_type="latency_mikasa_conditioned_h8",
        state_dim=63,
        action_dim=7,
        fps=20.0,
        video_keys=("video.top", "video.wrist"),
        auxiliary_shapes=(("action_prefix_mask", (8,)), ("action_loss_mask", (8,))),
        action_horizon=8,
    ),
    "haic": StarVLATaskSpec(
        name="haic_pull_cart",
        robot_type="latency_haic_h1",
        state_dim=605,
        action_dim=256,
        fps=50.0,
        video_keys=("video.ego_view",),
        auxiliary_shapes=(
            ("actor_input", (605,)),
            ("teacher_action", (23,)),
            ("termination", (1,)),
        ),
    ),
}


def task_spec(task: str) -> StarVLATaskSpec:
    return TASK_SPECS[task]


def _numeric_stats(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _train_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for path in sorted((dataset_dir / "data").glob("*/*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _normalization_from_rows(rows: list[dict[str, Any]], spec: StarVLATaskSpec) -> dict[str, dict[str, list[float]]]:
    states = np.asarray([row["observation.state"] for row in rows], dtype=np.float32)
    actions = np.asarray([row[spec.action_key] for row in rows], dtype=np.float32)
    auxiliary = spec.auxiliary_shape_map()
    if "action_loss_mask" in auxiliary:
        valid = np.asarray([row["action_loss_mask"] for row in rows], dtype=bool)
        actions = actions[valid]
    state_stats = _numeric_stats(states)
    if spec.robot_type == "latency_mikasa_conditioned_h8":
        # Action slots retain their physical scale even when constant in bootstrap data.
        state_stats["min"][7:] = [-1.0] * 56
        state_stats["max"][7:] = [1.0] * 56
    return {"state": state_stats, "action": _numeric_stats(actions.reshape(-1, spec.action_dim))}


def _read_prompt(dataset_dir: Path) -> str | None:
    tasks_jsonl = dataset_dir / "meta" / "tasks.jsonl"
    if tasks_jsonl.is_file():
        for line in tasks_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                return row["task"]

    tasks_parquet = dataset_dir / "meta" / "tasks.parquet"
    if tasks_parquet.is_file():
        import pyarrow.parquet as pq

        row = pq.read_table(tasks_parquet).to_pylist()[0]
        if "task" in row:
            return row["task"]
        return row["__index_level_0__"]
    return None


def _feature_names(dataset_dir: Path, key: str) -> list[str] | None:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    feature = info["features"][key]
    names = feature.get("names")
    return names


def _normalization_source_contract(source: Path | str) -> dict[str, Any]:
    source = Path(source)
    path = source if source.name == "task_contract.json" else source / "task_contract.json"
    return json.loads(path.read_text(encoding="utf-8"))


def write_task_contract(
    dataset_dir: Path | str,
    *,
    spec: StarVLATaskSpec,
    normalization_source: Path | str | None = None,
) -> Path:
    """Write the task contract and StarVLA statistics for a prepared dataset."""
    dataset_dir = Path(dataset_dir)
    rows = _train_rows(dataset_dir)
    normalization = (
        _normalization_from_rows(rows, spec)
        if normalization_source is None
        else _normalization_source_contract(normalization_source)["normalization"]
    )
    version = (
        3 if spec.robot_type in ("latency_balance_profile_h8", "latency_balance_profile_h16",
                                "latency_mikasa_conditioned_h8")
        else 2 if spec.name == TASK_SPECS["balance"].name
        else 1
    )
    contract: dict[str, Any] = {
        "format": f"latency_bench_starvla_h{spec.action_horizon}_v{version}",
        "task": spec.name,
        "robot_type": spec.robot_type,
        "state_dim": spec.state_dim,
        "action_dim": spec.action_dim,
        "action_horizon": spec.action_horizon,
        "fps": spec.fps,
        "state_key": "observation.state",
        "action_key": spec.action_key,
        "video_keys": list(spec.video_keys),
        "auxiliary": {
            key: {"shape": list(shape), "column": key}
            for key, shape in spec.auxiliary_shape_map().items()
        },
        "normalization": normalization,
    }
    if spec.name == TASK_SPECS["balance"].name:
        info = json.loads((dataset_dir / "meta/info.json").read_text())
        contract["label_provenance"] = info["label_provenance"]
    prompt = _read_prompt(dataset_dir)
    if prompt is not None:
        contract["prompt"] = prompt
    state_labels = _feature_names(dataset_dir, "observation.state")
    action_labels = _feature_names(dataset_dir, spec.action_key)
    if state_labels is not None:
        contract["state_labels"] = state_labels
    if action_labels is not None:
        contract["action_labels"] = action_labels

    path = dataset_dir / "task_contract.json"
    path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    _write_stats(dataset_dir, normalization, spec.action_key)
    return path


def _write_stats(dataset_dir: Path, normalization: Mapping[str, Mapping[str, list[float]]], action_key: str) -> None:
    stats = {
        "observation.state": dict(normalization["state"]),
        action_key: dict(normalization["action"]),
    }
    payload = {
        "__format_version": 2,
        "__cache_config": {"mode": "abs"},
        "statistics": stats,
    }
    (dataset_dir / "meta" / "stats_gr00t.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def read_task_contract(dataset_dir: Path | str) -> dict[str, Any]:
    return json.loads((Path(dataset_dir) / "task_contract.json").read_text(encoding="utf-8"))


def normalize(values: Any, stats: Mapping[str, list[float]]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    minimum = np.asarray(stats["min"], dtype=np.float32)
    maximum = np.asarray(stats["max"], dtype=np.float32)
    result = np.zeros_like(values, dtype=np.float32)
    mask = minimum != maximum
    result[..., mask] = 2.0 * (values[..., mask] - minimum[mask]) / (
        maximum[mask] - minimum[mask]
    ) - 1.0
    return np.clip(result, -1.0, 1.0)


def denormalize(values: Any, stats: Mapping[str, list[float]]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    minimum = np.asarray(stats["min"], dtype=np.float32)
    maximum = np.asarray(stats["max"], dtype=np.float32)
    return (values + 1.0) * 0.5 * (maximum - minimum) + minimum


__all__ = [
    "TASK_SPECS",
    "StarVLATaskSpec",
    "denormalize",
    "normalize",
    "read_task_contract",
    "task_spec",
    "write_task_contract",
]
