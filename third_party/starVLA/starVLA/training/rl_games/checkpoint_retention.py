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


class CheckpointRetention:
    """Retain the configured number of complete local training checkpoints."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._saved: List[CheckpointRecord] = []
        self._local_keep_last_n = int(getattr(getattr(getattr(cfg, "checkpoint", {}), "local", {}), "keep_last_n", 0))

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
