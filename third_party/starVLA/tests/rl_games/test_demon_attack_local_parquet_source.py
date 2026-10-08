from __future__ import annotations

import importlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from PIL import Image


def _png(value: int) -> bytes:
    image = Image.fromarray(np.full((2, 2, 3), fill_value=value, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_entry(value: int) -> dict[str, Any]:
    return {"bytes": _png(value), "path": None}


class FakeDatasetImage:
    def __init__(self, decode: bool) -> None:
        self.decode = decode


class FakeDatasetSequence:
    def __init__(self, feature: Any) -> None:
        self.feature = feature


def _optional_dependency_stubs() -> dict[str, ModuleType]:
    datasets = ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: None
    datasets.Image = FakeDatasetImage
    datasets.Sequence = FakeDatasetSequence

    pyarrow = ModuleType("pyarrow")
    pyarrow_parquet = ModuleType("pyarrow.parquet")
    return {
        "datasets": datasets,
        "pyarrow": pyarrow,
        "pyarrow.parquet": pyarrow_parquet,
    }


@pytest.fixture()
def convert_demon_attack(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module_name = "examples.rl_games.bash_scripts.gr00t.data_conversion.convert_demon_attack_to_starvla_lerobot"
    sys.modules.pop(module_name, None)
    for dependency_name, module in _optional_dependency_stubs().items():
        monkeypatch.setitem(sys.modules, dependency_name, module)

    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


def test_convert_demon_attack_resolves_local_parquet_directory(
    tmp_path: Path,
    convert_demon_attack: ModuleType,
) -> None:
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    train_dir.mkdir()
    validation_dir.mkdir()
    (train_dir / "part-000.parquet").touch()
    (validation_dir / "part-000.parquet").touch()

    train_files = convert_demon_attack._local_parquet_files(str(tmp_path), "train")
    validation_files = convert_demon_attack._local_parquet_files(str(tmp_path), "validation")

    assert train_files == [str(train_dir / "part-000.parquet")]
    assert validation_files == [str(validation_dir / "part-000.parquet")]


def test_convert_demon_attack_returns_absolute_local_parquet_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    convert_demon_attack: ModuleType,
) -> None:
    dataset_dir = tmp_path / "dataset"
    train_dir = dataset_dir / "train"
    train_dir.mkdir(parents=True)
    parquet_file = train_dir / "part-000.parquet"
    parquet_file.touch()
    monkeypatch.chdir(tmp_path)

    train_files = convert_demon_attack._local_parquet_files("dataset", "train")

    assert train_files == [str(parquet_file)]


def test_convert_demon_attack_resolves_clean_v1_column_aliases(
    monkeypatch: pytest.MonkeyPatch,
    convert_demon_attack: ModuleType,
) -> None:
    monkeypatch.setattr(
        convert_demon_attack,
        "_local_parquet_columns",
        lambda dataset_name, split, dataset_source_subdir=None: {
            "episode_idx",
            "decision_step",
            "action_id",
            "raw_reward",
            "prompt",
            "latency_raw_frames",
            "latency_ms",
        },
    )

    columns = convert_demon_attack._resolve_demon_attack_columns("demon_attack_clean_v1", "train", want_latency=True)

    assert columns.frame == "decision_step"
    assert columns.reward == "raw_reward"
    assert columns.done is None
    assert columns.latency == "latency_raw_frames"
    assert columns.latency_ms == "latency_ms"


def test_convert_demon_attack_marks_last_frame_done_when_done_column_is_absent(
    convert_demon_attack: ModuleType,
) -> None:
    assert convert_demon_attack._row_done({"done": True}, "done", frame_idx=0, episode_length=3) is True
    assert convert_demon_attack._row_done({}, None, frame_idx=0, episode_length=3) is False
    assert convert_demon_attack._row_done({}, None, frame_idx=2, episode_length=3) is True


def test_convert_demon_attack_preserves_context_images_in_temporal_order(
    convert_demon_attack: ModuleType,
) -> None:
    row = {
        "context_images": [_image_entry(10), _image_entry(20), _image_entry(30), _image_entry(40)],
        "image": _image_entry(50),
    }

    sequence = convert_demon_attack._context_images_from_context(
        row,
        context_images_column="context_images",
        image_sequence_length=5,
    )

    assert len(sequence) == 4
    assert [
        int(np.asarray(Image.open(io.BytesIO(entry["bytes"])).convert("RGB"))[0, 0, 0])
        for entry in sequence
    ] == [10, 20, 30, 40]


def test_convert_demon_attack_rejects_context_image_count_mismatch(
    convert_demon_attack: ModuleType,
) -> None:
    row = {
        "context_images": [_image_entry(10), _image_entry(20), _image_entry(30)],
        "image": _image_entry(40),
    }

    with pytest.raises(ValueError, match="Expected 4 context image"):
        convert_demon_attack._context_images_from_context(
            row,
            context_images_column="context_images",
            image_sequence_length=5,
        )


def test_convert_demon_attack_writes_loader_compatible_context_image_metadata(
    convert_demon_attack: ModuleType,
    tmp_path: Path,
) -> None:
    convert_demon_attack._write_metadata(
        tmp_path,
        episode_lengths=[3],
        task_prompts=["play"],
        action_dim=7,
        action_labels=["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE", "PAD6"],
        state_dim=7,
        state_labels=["s0", "s1", "s2", "s3", "s4", "s5", "s6"],
        context_images_output_column="observation.context_images",
        image_sequence_length=5,
        image_shape=[84, 84, 3],
        fps=15.0,
    )

    modality = json.loads((tmp_path / "meta" / "modality.json").read_text())
    info = json.loads((tmp_path / "meta" / "info.json").read_text())

    assert "context_images" not in modality["video"]
    assert info["features"]["observation.context_images"] == {
        "dtype": "image_sequence",
        "shape": [4, 84, 84, 3],
        "names": ["time", "height", "width", "channel"],
        "video_info": {"video.fps": 15.0},
    }
