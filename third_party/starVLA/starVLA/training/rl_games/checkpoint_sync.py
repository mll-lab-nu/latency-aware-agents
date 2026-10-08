from __future__ import annotations

import os
import re
import shutil
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


logger = logging.getLogger(__name__)
STEP_FILE_RE = re.compile(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$")
STEP_STATE_RE = re.compile(r"steps_(\d+)_state$")


@dataclass
class CheckpointRecord:
    step: int
    state_path: Optional[str] = None
    model_path: Optional[str] = None


class CheckpointSyncManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self._saved: List[CheckpointRecord] = []
        self._sync_enabled = bool(
            getattr(getattr(getattr(cfg, "checkpoint", {}), "sync", {}), "enabled", False)
        )
        self._hf_repo_id = getattr(getattr(getattr(cfg, "checkpoint", {}), "sync", {}), "repo_id", None)
        self._hf_keep_last_n = int(getattr(getattr(getattr(cfg, "checkpoint", {}), "sync", {}), "keep_last_n", 0))
        self._local_keep_last_n = int(getattr(getattr(getattr(cfg, "checkpoint", {}), "local", {}), "keep_last_n", 0))
        self._hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    def register_local_checkpoint(
        self,
        step: int,
        state_path: str | None = None,
        model_path: str | None = None,
    ) -> None:
        checkpoint_dir = self._checkpoint_dir(state_path=state_path, model_path=model_path)
        if checkpoint_dir is not None:
            self._saved = self._discover_local_checkpoints(checkpoint_dir)
        else:
            self._saved.append(CheckpointRecord(step=step, state_path=state_path, model_path=model_path))
        self._saved.sort(key=lambda record: record.step)
        self._prune_local_checkpoints()
        if not self._sync_enabled:
            logger.info("HF checkpoint sync disabled; skipping upload for step %s", step)
            return
        if not self._hf_repo_id:
            logger.warning("HF checkpoint sync enabled but checkpoint.sync.repo_id is empty; skipping upload")
            return
        if not self._hf_token:
            logger.warning(
                "HF checkpoint sync enabled for %s but HF_TOKEN/HUGGINGFACE_HUB_TOKEN is not set; skipping upload",
                self._hf_repo_id,
            )
            return
        self._sync_to_hf(state_path=state_path, model_path=model_path)
        self._prune_hf_checkpoints()

    @staticmethod
    def _checkpoint_dir(state_path: str | None, model_path: str | None) -> Path | None:
        for path in (state_path, model_path):
            if not path:
                continue
            return Path(path).expanduser().resolve().parent
        return None

    @staticmethod
    def _discover_local_checkpoints(checkpoint_dir: Path) -> List[CheckpointRecord]:
        records_by_step: dict[int, CheckpointRecord] = {}
        if not checkpoint_dir.exists():
            return []
        for item in checkpoint_dir.iterdir():
            if item.is_dir():
                match = STEP_STATE_RE.match(item.name)
                if match:
                    step = int(match.group(1))
                    record = records_by_step.setdefault(step, CheckpointRecord(step=step))
                    record.state_path = str(item)
                continue
            if item.is_file():
                match = STEP_FILE_RE.match(item.name)
                if match:
                    step = int(match.group(1))
                    record = records_by_step.setdefault(step, CheckpointRecord(step=step))
                    record.model_path = str(item)
        return sorted(records_by_step.values(), key=lambda record: record.step)

    def sync_eval_result(self, eval_path: str | None, stage: str, step: int) -> None:
        if not eval_path or not os.path.isfile(eval_path):
            return
        if not self._sync_enabled:
            logger.info("HF checkpoint sync disabled; skipping eval result upload for step %s", step)
            return
        if not self._hf_repo_id:
            logger.warning("HF checkpoint sync enabled but checkpoint.sync.repo_id is empty; skipping eval result upload")
            return
        if not self._hf_token:
            logger.warning(
                "HF checkpoint sync enabled for %s but HF_TOKEN/HUGGINGFACE_HUB_TOKEN is not set; skipping eval result upload",
                self._hf_repo_id,
            )
            return
        self._sync_eval_to_hf(eval_path=eval_path, stage=stage, step=step)

    def sync_best_checkpoint(
        self,
        state_path: str,
        metadata_path: str,
        config_paths: list[str] | None = None,
    ) -> None:
        if not self._sync_enabled:
            logger.info("HF checkpoint sync disabled; skipping best checkpoint upload")
            return
        if not self._hf_repo_id:
            logger.warning("HF checkpoint sync enabled but checkpoint.sync.repo_id is empty; skipping best checkpoint upload")
            return
        if not self._hf_token:
            logger.warning(
                "HF checkpoint sync enabled for %s but HF_TOKEN/HUGGINGFACE_HUB_TOKEN is not set; skipping best checkpoint upload",
                self._hf_repo_id,
            )
            return
        self._sync_best_to_hf(
            state_path=state_path,
            metadata_path=metadata_path,
            config_paths=config_paths or [],
        )

    def _prune_local_checkpoints(self) -> None:
        if self._local_keep_last_n <= 0:
            return
        while len(self._saved) > self._local_keep_last_n:
            old = self._saved.pop(0)
            for path in (old.state_path, old.model_path):
                if not path or not os.path.exists(path):
                    continue
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except OSError:
                    # Non-fatal: keep training even if cleanup fails.
                    pass

    def _sync_to_hf(self, state_path: str | None = None, model_path: str | None = None) -> None:
        # Lazy-import and soft-fail to avoid hard dependency for local-only users.
        try:
            from huggingface_hub import HfApi, upload_file, upload_folder
        except Exception as exc:
            logger.warning("HF checkpoint sync skipped: could not import huggingface_hub: %s", exc)
            return
        try:
            api = HfApi(token=self._hf_token)
            api.create_repo(repo_id=self._hf_repo_id, repo_type="model", exist_ok=True)
            if state_path and os.path.isdir(state_path):
                logger.info("Uploading checkpoint state folder to HF: %s -> %s", state_path, self._hf_repo_id)
                upload_folder(
                    folder_path=state_path,
                    path_in_repo=os.path.basename(state_path),
                    repo_id=self._hf_repo_id,
                    repo_type="model",
                    token=self._hf_token,
                )
            if model_path and os.path.isfile(model_path):
                logger.info("Uploading checkpoint model file to HF: %s -> %s", model_path, self._hf_repo_id)
                upload_file(
                    path_or_fileobj=model_path,
                    path_in_repo=os.path.basename(model_path),
                    repo_id=self._hf_repo_id,
                    repo_type="model",
                    token=self._hf_token,
                )
            logger.info("HF checkpoint sync completed for repo %s", self._hf_repo_id)
        except Exception as exc:
            # Non-fatal by design.
            logger.warning("HF checkpoint sync failed for repo %s: %s", self._hf_repo_id, exc)
            return

    def _sync_eval_to_hf(self, eval_path: str, stage: str, step: int) -> None:
        try:
            from huggingface_hub import HfApi, upload_file
        except Exception as exc:
            logger.warning("HF eval sync skipped: could not import huggingface_hub: %s", exc)
            return
        try:
            api = HfApi(token=self._hf_token)
            api.create_repo(repo_id=self._hf_repo_id, repo_type="model", exist_ok=True)
            path_in_repo = f"eval/{stage}/step_{int(step)}.json"
            logger.info("Uploading eval result to HF: %s -> %s/%s", eval_path, self._hf_repo_id, path_in_repo)
            upload_file(
                path_or_fileobj=eval_path,
                path_in_repo=path_in_repo,
                repo_id=self._hf_repo_id,
                repo_type="model",
                token=self._hf_token,
            )
        except Exception as exc:
            logger.warning("HF eval sync failed for repo %s: %s", self._hf_repo_id, exc)
            return

    def _sync_best_to_hf(self, state_path: str, metadata_path: str, config_paths: list[str]) -> None:
        try:
            from huggingface_hub import HfApi, upload_file, upload_folder
        except Exception as exc:
            logger.warning("HF best checkpoint sync skipped: could not import huggingface_hub: %s", exc)
            return
        try:
            api = HfApi(token=self._hf_token)
            api.create_repo(repo_id=self._hf_repo_id, repo_type="model", exist_ok=True)

            try:
                files = api.list_repo_files(repo_id=self._hf_repo_id, repo_type="model")
            except Exception:
                files = []
            for file_path in sorted(files):
                if file_path.startswith("best_state/") or file_path == "best_model_metadata.json":
                    api.delete_file(
                        path_in_repo=file_path,
                        repo_id=self._hf_repo_id,
                        repo_type="model",
                        token=self._hf_token,
                    )

            logger.info("Uploading best checkpoint state folder to HF: %s -> %s/best_state", state_path, self._hf_repo_id)
            upload_folder(
                folder_path=state_path,
                path_in_repo="best_state",
                repo_id=self._hf_repo_id,
                repo_type="model",
                token=self._hf_token,
            )

            if os.path.isfile(metadata_path):
                upload_file(
                    path_or_fileobj=metadata_path,
                    path_in_repo="best_model_metadata.json",
                    repo_id=self._hf_repo_id,
                    repo_type="model",
                    token=self._hf_token,
                )

            for config_path in config_paths:
                if not config_path or not os.path.isfile(config_path):
                    continue
                upload_file(
                    path_or_fileobj=config_path,
                    path_in_repo=os.path.basename(config_path),
                    repo_id=self._hf_repo_id,
                    repo_type="model",
                    token=self._hf_token,
                )
            logger.info("HF best checkpoint sync completed for repo %s", self._hf_repo_id)
        except Exception as exc:
            logger.warning("HF best checkpoint sync failed for repo %s: %s", self._hf_repo_id, exc)
            return

    def _prune_hf_checkpoints(self) -> None:
        if self._hf_keep_last_n <= 0:
            return
        try:
            from huggingface_hub import HfApi
        except Exception as exc:
            logger.warning("HF checkpoint pruning skipped: could not import huggingface_hub: %s", exc)
            return
        try:
            api = HfApi(token=self._hf_token)
            files = api.list_repo_files(repo_id=self._hf_repo_id, repo_type="model")
            paths_by_step = {}
            for file_path in files:
                first_part = file_path.split("/", 1)[0]
                if first_part.startswith("steps_") and first_part.endswith("_state"):
                    try:
                        step = int(first_part.split("steps_")[1].split("_state")[0])
                    except Exception:
                        continue
                    paths_by_step.setdefault(step, set()).add(file_path)
                    continue
                if os.path.basename(file_path).startswith("steps_"):
                    try:
                        step = int(os.path.basename(file_path).split("steps_")[1].split("_")[0])
                    except Exception:
                        continue
                    paths_by_step.setdefault(step, set()).add(file_path)
            steps = sorted(paths_by_step)
            if len(steps) <= self._hf_keep_last_n:
                return
            for step in steps[: len(steps) - self._hf_keep_last_n]:
                for delete_path in sorted(paths_by_step[step]):
                    api.delete_file(
                        path_in_repo=delete_path,
                        repo_id=self._hf_repo_id,
                        repo_type="model",
                        token=self._hf_token,
                    )
        except Exception as exc:
            logger.warning("HF checkpoint pruning failed for repo %s: %s", self._hf_repo_id, exc)
            return
