from pathlib import Path
import importlib.util
import logging
import sys
import types

import pytest


SHARE_TOOLS_PATH = Path(__file__).resolve().parents[2] / "starVLA" / "model" / "framework" / "share_tools.py"
rich_module = types.ModuleType("rich")
rich_logging_module = types.ModuleType("rich.logging")


class _FakeRichHandler(logging.StreamHandler):
    def __init__(self, *args, **kwargs):
        super().__init__()


rich_logging_module.RichHandler = _FakeRichHandler
rich_module.logging = rich_logging_module
sys.modules.setdefault("rich", rich_module)
sys.modules.setdefault("rich.logging", rich_logging_module)
SPEC = importlib.util.spec_from_file_location("share_tools_under_test", SHARE_TOOLS_PATH)
assert SPEC is not None
share_tools = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(share_tools)
resolve_pretrained_checkpoint_path = share_tools.resolve_pretrained_checkpoint_path


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    return run_dir


def test_resolve_pretrained_checkpoint_accepts_standalone_pt(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    checkpoint = run_dir / "checkpoints" / "steps_5_pytorch_model.pt"
    checkpoint.write_text("weights", encoding="utf-8")

    checkpoint_with_parent_segment = checkpoint.parent / ".." / "checkpoints" / checkpoint.name
    resolved_checkpoint, resolved_run_dir = resolve_pretrained_checkpoint_path(checkpoint_with_parent_segment)

    assert resolved_checkpoint == checkpoint
    assert resolved_run_dir == run_dir


def test_resolve_pretrained_checkpoint_preserves_hf_snapshot_symlink(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    blob = tmp_path / "blobs" / "checkpoint"
    blob.parent.mkdir()
    blob.write_text("weights", encoding="utf-8")
    checkpoint = run_dir / "checkpoints" / "steps_5_pytorch_model.pt"
    checkpoint.symlink_to(blob)

    resolved_checkpoint, resolved_run_dir = resolve_pretrained_checkpoint_path(checkpoint)

    assert resolved_checkpoint == checkpoint
    assert resolved_run_dir == run_dir


def test_resolve_pretrained_checkpoint_accepts_state_directory(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    state_dir = run_dir / "checkpoints" / "steps_5_state"
    state_dir.mkdir()
    checkpoint = state_dir / "model.safetensors"
    checkpoint.write_text("weights", encoding="utf-8")

    resolved_checkpoint, resolved_run_dir = resolve_pretrained_checkpoint_path(state_dir)

    assert resolved_checkpoint == checkpoint
    assert resolved_run_dir == run_dir


def test_resolve_pretrained_checkpoint_accepts_file_inside_state_directory(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    state_dir = run_dir / "checkpoints" / "steps_5_state"
    state_dir.mkdir()
    checkpoint = state_dir / "model.safetensors"
    checkpoint.write_text("weights", encoding="utf-8")

    resolved_checkpoint, resolved_run_dir = resolve_pretrained_checkpoint_path(checkpoint)

    assert resolved_checkpoint == checkpoint
    assert resolved_run_dir == run_dir


def test_resolve_pretrained_checkpoint_rejects_state_directory_without_model_file(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    state_dir = run_dir / "checkpoints" / "steps_5_state"
    state_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        resolve_pretrained_checkpoint_path(state_dir)
