from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch


_SAMPLE_FACTORY_CHECKPOINT_LOADER_PATCHED = False
_SAMPLE_FACTORY_CHECKPOINT_RETENTION_PATCHED = False
_SAMPLE_FACTORY_LEARNER_INIT_PATCHED = False
_EVAL_LEASE_DIR = ".latency_bench_eval_leases"


@dataclass(frozen=True)
class CheckpointEvalLease:
    checkpoint_path: Path
    marker_path: Path


def trusted_torch_load(path: str | Path, device: torch.device) -> dict[str, Any]:
    load_path = str(Path(path).expanduser())
    try:
        return torch.load(load_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(load_path, map_location=device)


def init_learner_process_with_checkpoint_io_patch(sf_context: Any, learner_worker: Any):
    from sample_factory.algo.learning.learner_worker import init_learner_process

    # Sample Factory uses multiprocessing spawn for learner workers, so parent
    # monkeypatches are not inherited by the child process.
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

            checkpoint_dir = self.checkpoint_dir(self.cfg, self.policy_id)
            tmp_filepath = os.path.join(checkpoint_dir, f"{name_prefix}_temp")
            checkpoint_name = f"{name_prefix}_{self.train_step:09d}_{self.env_steps}{name_suffix}.pth"
            filepath = os.path.join(checkpoint_dir, checkpoint_name)
            if verbose:
                sf_learner.log.info("Saving %s...", filepath)

            torch.save(checkpoint, tmp_filepath)
            os.rename(tmp_filepath, filepath)
            prune_unleased_sample_factory_checkpoints(
                checkpoint_dir,
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


def load_sample_factory_checkpoint_from_paths(checkpoints: list[str] | list[Path], device: torch.device) -> dict[str, Any]:
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


def prune_unleased_sample_factory_checkpoints(
    checkpoint_dir: str | Path,
    name_prefix: str,
    *,
    keep_checkpoints: int,
    verbose: bool = True,
    logger: Any | None = None,
) -> None:
    root = Path(checkpoint_dir).expanduser()
    while len(checkpoints := sorted(root.glob(f"{name_prefix}_*.pth"))) > int(keep_checkpoints):
        leased_names = _leased_checkpoint_names(root)
        oldest_unleased = next((path for path in checkpoints if path.name not in leased_names), None)
        if oldest_unleased is None:
            return
        if verbose and logger is not None:
            logger.debug("Removing %s", str(oldest_unleased))
        oldest_unleased.unlink()


def _leased_checkpoint_names(checkpoint_dir: Path) -> set[str]:
    lease_dir = checkpoint_dir / _EVAL_LEASE_DIR
    return {marker.name.rsplit(".", 2)[0] for marker in lease_dir.glob("*.json")}
