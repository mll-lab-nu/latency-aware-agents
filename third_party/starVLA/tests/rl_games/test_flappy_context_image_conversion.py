from __future__ import annotations

import importlib
import io
import json
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


@pytest.fixture()
def flappy_converter(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module_name = "examples.rl_games.bash_scripts.gr00t.data_conversion.convert_flappy_to_starvla_lerobot"
    sys.modules.pop(module_name, None)
    datasets = ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: None
    datasets.Image = FakeDatasetImage
    datasets.Sequence = FakeDatasetSequence
    pyarrow = ModuleType("pyarrow")
    pyarrow_parquet = ModuleType("pyarrow.parquet")
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "pyarrow", pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", pyarrow_parquet)
    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


def test_flappy_converter_preserves_context_images_in_temporal_order(flappy_converter: ModuleType) -> None:
    row = {
        "context_images": [_image_entry(10), _image_entry(20), _image_entry(30)],
        "image": _image_entry(40),
    }

    sequence = flappy_converter._context_images_from_context(
        row,
        context_images_column="context_images",
        image_sequence_length=4,
    )

    assert len(sequence) == 3
    assert [
        int(np.asarray(Image.open(io.BytesIO(entry["bytes"])).convert("RGB"))[0, 0, 0])
        for entry in sequence
    ] == [10, 20, 30]


def test_flappy_converter_rejects_context_image_count_mismatch(flappy_converter: ModuleType) -> None:
    row = {
        "context_images": [_image_entry(10), _image_entry(20)],
        "image": _image_entry(30),
    }

    with pytest.raises(ValueError, match="Expected 3 context image"):
        flappy_converter._context_images_from_context(
            row,
            context_images_column="context_images",
            image_sequence_length=4,
        )


def test_flappy_converter_writes_loader_compatible_context_image_metadata(
    flappy_converter: ModuleType,
    tmp_path: Any,
) -> None:
    flappy_converter._write_metadata(
        tmp_path,
        episode_lengths=[3],
        task_prompts=["play"],
        action_dim=7,
        action_labels=["NOOP", "FLAP", "PAD2", "PAD3", "PAD4", "PAD5", "PAD6"],
        state_dim=7,
        state_labels=["s0", "s1", "s2", "s3", "s4", "s5", "s6"],
        context_images_output_column="observation.context_images",
        image_sequence_length=5,
        image_shape=[84, 84, 3],
        fps=30.0,
    )

    modality = json.loads((tmp_path / "meta" / "modality.json").read_text())
    info = json.loads((tmp_path / "meta" / "info.json").read_text())

    assert "context_images" not in modality["video"]
    assert info["features"]["observation.context_images"] == {
        "dtype": "image_sequence",
        "shape": [4, 84, 84, 3],
        "names": ["time", "height", "width", "channel"],
        "video_info": {"video.fps": 30.0},
    }


def test_temporal_clip_decodes_context_images_with_current_image_in_temporal_order() -> None:
    temporal_clip = importlib.import_module("starVLA.training.rl_games.temporal_clip")
    context_images = [_image_entry(10), _image_entry(20), _image_entry(30)]
    current_image = _image_entry(40)

    frames = temporal_clip.decode_context_image_sequence(
        context_entry=context_images,
        current_entry=current_image,
        image_sequence_length=4,
        dataset_path=None,
    )

    assert frames.shape == (4, 2, 2, 3)
    assert [int(frames[index, 0, 0, 0]) for index in range(4)] == [10, 20, 30, 40]
