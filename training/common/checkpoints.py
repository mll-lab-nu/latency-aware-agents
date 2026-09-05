from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch


_SAMPLE_FACTORY_CHECKPOINT_LOADER_PATCHED = False
_SAMPLE_FACTORY_CHECKPOINT_RETENTION_PATCHED = False
_SAMPLE_FACTORY_LEARNER_INIT_PATCHED = False
_EVAL_LEASE_DIR = ".latency_bench_eval_leases"
_EVAL_RETENTION_DIR = ".latency_bench_eval_checkpoints"


def atomic_torch_save(payload: Any, path: str | os.PathLike[str]) -> None:
    """Replace a single-writer checkpoint only after serialization succeeds."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_rng_state() -> dict[str, Any]:
    """Snapshot native learner random streams without starting CUDA for CPU runs."""
    return {
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore learner streams after constructing and loading model state."""
    torch.set_rng_state(state["torch_rng"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])


def experiment_dir(train_dir: str | os.PathLike[str], experiment_name: str) -> Path:
    name = str(experiment_name).strip()
    if not name:
        raise ValueError("experiment.name must be a non-empty string")
    return Path(train_dir).expanduser() / name


def checkpoint_dir(
    train_dir: str | os.PathLike[str],
    experiment_name: str,
    *,
    policy_id: int = 0,
) -> Path:
    return experiment_dir(train_dir, experiment_name) / f"checkpoint_p{int(policy_id)}"


def format_checkpoint_name(*, train_step: int, env_steps: int) -> str:
    return f"checkpoint_{int(train_step):09d}_{int(env_steps):012d}.pth"


def find_checkpoint(
    train_dir: str | os.PathLike[str],
    experiment_name: str,
    *,
    kind: str = "latest",
    policy_id: int = 0,
) -> Path:
    normalized_kind = str(kind).strip().lower()
    if normalized_kind == "latest":
        pattern = "checkpoint_*.pth"
    elif normalized_kind == "best":
        pattern = "best_*.pth"
    else:
        raise ValueError("checkpoint kind must be 'latest' or 'best'")

    root = checkpoint_dir(train_dir, experiment_name, policy_id=policy_id)
    checkpoints = sorted(root.glob(pattern))
    if not checkpoints:
        raise FileNotFoundError(
            f"No {normalized_kind} checkpoint matching {pattern!r} under {root}"
        )
    return checkpoints[-1]


def checkpoint_paths(root: str | os.PathLike[str], pattern: str = "checkpoint_*.pth") -> list[Path]:
    return sorted(Path(root).expanduser().glob(pattern))


def checkpoint_mtimes(root: str | os.PathLike[str], pattern: str = "checkpoint_*.pth") -> dict[Path, int]:
    return {path: path.stat().st_mtime_ns for path in checkpoint_paths(root, pattern)}


def read_checkpoint_backend_type(path: str | os.PathLike[str]) -> str | None:
    checkpoint = trusted_torch_load(path, torch.device("cpu"))
    if not isinstance(checkpoint, dict):
        return None
    backend_type: Any = checkpoint.get("backend_type")
    if backend_type is None:
        return None
    return str(backend_type)


@dataclass(frozen=True)
class CheckpointEvalLease:
    checkpoint_path: Path
    marker_path: Path


def trusted_torch_load(path: str | os.PathLike[str], device: torch.device | str) -> dict[str, Any]:
    load_path = str(Path(path).expanduser())
    try:
        return torch.load(load_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(load_path, map_location=device)


def init_learner_process_with_checkpoint_io_patch(sf_context: Any, learner_worker: Any):
    from sample_factory.algo.learning.learner_worker import init_learner_process

    patch_sample_factory_checkpoint_loader()
    return init_learner_process(sf_context, learner_worker)


def patch_sample_factory_checkpoint_loader() -> None:
    global _SAMPLE_FACTORY_CHECKPOINT_LOADER_PATCHED
    global _SAMPLE_FACTORY_CHECKPOINT_RETENTION_PATCHED
    global _SAMPLE_FACTORY_LEARNER_INIT_PATCHED

    import sample_factory.algo.learning.learner as sf_learner

    if not _SAMPLE_FACTORY_CHECKPOINT_LOADER_PATCHED:

        def load_checkpoint(checkpoints, device):
            if len(checkpoints) <= 0:
                sf_learner.log.warning("No checkpoints found")
                return None

            latest_checkpoint = checkpoints[-1]
            num_attempts = 3
            for attempt in range(num_attempts):
                try:
                    sf_learner.log.warning("Loading state from checkpoint %s...", latest_checkpoint)
                    return trusted_torch_load(latest_checkpoint, device)
                except Exception:
                    sf_learner.log.exception(f"Could not load from checkpoint, attempt {attempt}")
            return None

        sf_learner.Learner.load_checkpoint = staticmethod(load_checkpoint)
        _SAMPLE_FACTORY_CHECKPOINT_LOADER_PATCHED = True

    if not _SAMPLE_FACTORY_CHECKPOINT_RETENTION_PATCHED:

        def _save_impl(self, name_prefix, name_suffix, keep_checkpoints, verbose=True) -> bool:
            if not self.is_initialized:
                return False

            checkpoint = self._get_checkpoint_dict()
            assert checkpoint is not None

            root = self.checkpoint_dir(self.cfg, self.policy_id)
            tmp_filepath = os.path.join(root, f"{name_prefix}_temp")
            checkpoint_name = f"{name_prefix}_{self.train_step:09d}_{self.env_steps}{name_suffix}.pth"
            filepath = os.path.join(root, checkpoint_name)
            if verbose:
                sf_learner.log.info("Saving %s...", filepath)

            torch.save(checkpoint, tmp_filepath)
            os.rename(tmp_filepath, filepath)
            prune_unleased_sample_factory_checkpoints(
                root,
                name_prefix,
                keep_checkpoints=keep_checkpoints,
                verbose=verbose,
                logger=sf_learner.log,
            )
            return True

        sf_learner.Learner._save_impl = _save_impl
        _SAMPLE_FACTORY_CHECKPOINT_RETENTION_PATCHED = True

    if not _SAMPLE_FACTORY_LEARNER_INIT_PATCHED:
        import sample_factory.algo.runners.runner_parallel as sf_runner_parallel

        sf_runner_parallel.init_learner_process = init_learner_process_with_checkpoint_io_patch
        _SAMPLE_FACTORY_LEARNER_INIT_PATCHED = True


def load_sample_factory_checkpoint_from_paths(
    checkpoints: list[str] | list[Path],
    device: torch.device,
) -> dict[str, Any]:
    if not checkpoints:
        raise RuntimeError("No checkpoints found to load")
    last_error: Exception | None = None
    for checkpoint_path in reversed([str(Path(item).expanduser()) for item in checkpoints]):
        try:
            return trusted_torch_load(checkpoint_path, device)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    if last_error is not None:
        raise RuntimeError("Could not load checkpoint") from last_error
    raise RuntimeError("Could not load checkpoint")


def acquire_checkpoint_eval_lease(
    checkpoint_path: str | Path,
    *,
    metadata: dict[str, Any],
) -> CheckpointEvalLease:
    path = Path(checkpoint_path).expanduser()
    lease_dir = path.parent / _EVAL_LEASE_DIR
    lease_dir.mkdir(parents=True, exist_ok=True)
    marker_path = lease_dir / f"{path.name}.{uuid4().hex}.json"
    payload = {
        "checkpoint_path": str(path),
        "runner_pid": os.getpid(),
        **metadata,
    }
    marker_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return CheckpointEvalLease(checkpoint_path=path, marker_path=marker_path)


def update_checkpoint_eval_lease(lease: CheckpointEvalLease, metadata: dict[str, Any]) -> None:
    payload = json.loads(lease.marker_path.read_text(encoding="utf-8"))
    payload.update(metadata)
    lease.marker_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def release_checkpoint_eval_lease(lease: CheckpointEvalLease) -> None:
    lease.marker_path.unlink(missing_ok=True)


def mark_checkpoint_for_eval_retention(
    checkpoint_path: str | Path,
    *,
    metadata: dict[str, Any],
) -> Path:
    path = Path(checkpoint_path).expanduser()
    retention_dir = path.parent / _EVAL_RETENTION_DIR
    retention_dir.mkdir(parents=True, exist_ok=True)
    marker_path = retention_dir / f"{path.name}.json"
    payload = {
        "checkpoint_path": str(path),
        "runner_pid": os.getpid(),
        **metadata,
    }
    marker_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return marker_path


def prune_unleased_sample_factory_checkpoints(
    checkpoint_dir: str | Path,
    name_prefix: str,
    *,
    keep_checkpoints: int,
    verbose: bool = True,
    logger: Any | None = None,
) -> None:
    root = Path(checkpoint_dir).expanduser()
    while True:
        checkpoints = checkpoint_paths(root, f"{name_prefix}_*.pth")
        protected_names = _protected_checkpoint_names(root)
        unprotected_checkpoints = [path for path in checkpoints if path.name not in protected_names]
        if len(unprotected_checkpoints) <= int(keep_checkpoints):
            return
        oldest_unleased = unprotected_checkpoints[0]
        if verbose and logger is not None:
            logger.debug("Removing %s", str(oldest_unleased))
        oldest_unleased.unlink()


def _leased_checkpoint_names(checkpoint_dir: Path) -> set[str]:
    lease_dir = checkpoint_dir / _EVAL_LEASE_DIR
    return {marker.name.rsplit(".", 2)[0] for marker in lease_dir.glob("*.json")}


def _retained_checkpoint_names(checkpoint_dir: Path) -> set[str]:
    retention_dir = checkpoint_dir / _EVAL_RETENTION_DIR
    return {marker.name.removesuffix(".json") for marker in retention_dir.glob("*.json")}


def _protected_checkpoint_names(checkpoint_dir: Path) -> set[str]:
    return _leased_checkpoint_names(checkpoint_dir) | _retained_checkpoint_names(checkpoint_dir)
