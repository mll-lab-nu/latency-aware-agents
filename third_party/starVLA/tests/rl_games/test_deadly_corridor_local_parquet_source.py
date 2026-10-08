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


def _optional_dependency_stubs() -> dict[str, ModuleType]:
    datasets = ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: None

    pyarrow = ModuleType("pyarrow")
    pyarrow_parquet = ModuleType("pyarrow.parquet")
    return {
        "datasets": datasets,
        "pyarrow": pyarrow,
        "pyarrow.parquet": pyarrow_parquet,
    }


@pytest.fixture()
def convert_deadly_corridor(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module_name = "examples.rl_games.bash_scripts.gr00t.data_conversion.convert_deadly_corridor_to_starvla_lerobot"
    sys.modules.pop(module_name, None)
    for dependency_name, module in _optional_dependency_stubs().items():
        monkeypatch.setitem(sys.modules, dependency_name, module)

    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


def test_convert_deadly_corridor_resolves_local_parquet_directory(
    tmp_path: Path,
    convert_deadly_corridor: ModuleType,
) -> None:
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    train_dir.mkdir()
    validation_dir.mkdir()
    (train_dir / "part-000.parquet").touch()
    (validation_dir / "part-000.parquet").touch()

    train_files = convert_deadly_corridor._local_parquet_files(str(tmp_path), "train")
    validation_files = convert_deadly_corridor._local_parquet_files(str(tmp_path), "validation")

    assert train_files == [str(train_dir.resolve() / "part-000.parquet")]
    assert validation_files == [str(validation_dir.resolve() / "part-000.parquet")]


def test_convert_deadly_corridor_returns_absolute_local_parquet_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    convert_deadly_corridor: ModuleType,
) -> None:
    dataset_dir = tmp_path / "dataset"
    train_dir = dataset_dir / "train"
    train_dir.mkdir(parents=True)
    parquet_file = train_dir / "part-000.parquet"
    parquet_file.touch()
    monkeypatch.chdir(tmp_path)

    train_files = convert_deadly_corridor._local_parquet_files("dataset", "train")

    assert train_files == [str(parquet_file)]


def test_convert_deadly_corridor_resolves_clean_v1_column_aliases(
    monkeypatch: pytest.MonkeyPatch,
    convert_deadly_corridor: ModuleType,
) -> None:
    monkeypatch.setattr(
        convert_deadly_corridor,
        "_local_parquet_columns",
        lambda dataset_name, split, dataset_source_subdir=None: {
            "episode_idx",
            "decision_step",
            "action",
            "raw_reward",
            "prompt",
            "latency_raw_frames",
            "latency_ms",
        },
    )

    columns = convert_deadly_corridor._resolve_deadly_corridor_columns("deadly_corridor_clean_v1", "train", want_latency=True)

    assert columns.frame == "decision_step"
    assert columns.reward == "raw_reward"
    assert columns.done is None
    assert columns.latency == "latency_raw_frames"
    assert columns.latency_ms == "latency_ms"


def test_convert_deadly_corridor_marks_last_frame_done_when_done_column_is_absent(
    convert_deadly_corridor: ModuleType,
) -> None:
    assert convert_deadly_corridor._row_done({"done": True}, "done", frame_idx=0, episode_length=3) is True
    assert convert_deadly_corridor._row_done({}, None, frame_idx=0, episode_length=3) is False
    assert convert_deadly_corridor._row_done({}, None, frame_idx=2, episode_length=3) is True


def test_convert_deadly_corridor_decodes_joint_54_action_id_into_semantic_buttons(
    convert_deadly_corridor: ModuleType,
) -> None:
    action = convert_deadly_corridor._source_action_vector(
        {"action_id": 11},
        action_layout="multibinary_7",
        source_action_layout="deadly_corridor_joint_54",
    )

    assert action == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]


def test_convert_deadly_corridor_rejects_invalid_joint_54_action_id(
    convert_deadly_corridor: ModuleType,
) -> None:
    with pytest.raises(ValueError, match=r"must be in \[0, 53\]"):
        convert_deadly_corridor._source_action_vector(
            {"action_id": 54},
            action_layout="multibinary_7",
            source_action_layout="deadly_corridor_joint_54",
        )


def test_convert_deadly_corridor_preserves_context_images_in_temporal_order(
    convert_deadly_corridor: ModuleType,
) -> None:
    row = {
        "context_images": [_image_entry(10), _image_entry(20), _image_entry(30), _image_entry(40)],
        "image": _image_entry(50),
    }

    sequence = convert_deadly_corridor._context_images_from_context(
        row,
        context_images_column="context_images",
        image_sequence_length=5,
    )

    assert [
        int(np.asarray(Image.open(io.BytesIO(entry["bytes"])).convert("RGB"))[0, 0, 0])
        for entry in sequence
    ] == [10, 20, 30, 40]


def test_convert_deadly_corridor_writes_context_image_metadata(
    convert_deadly_corridor: ModuleType,
    tmp_path: Path,
) -> None:
    convert_deadly_corridor._write_metadata(
        tmp_path,
        episode_lengths=[3],
        task_prompts=["play"],
        action_dim=7,
        action_labels=list(convert_deadly_corridor.ACTION_LABELS),
        state_dim=7,
        state_labels=[f"s{index}" for index in range(7)],
        image_shape=[2, 2, 3],
        context_images_output_column="observation.context_images",
        image_sequence_length=5,
        fps=35.0,
    )

    info = json.loads((tmp_path / "meta" / "info.json").read_text(encoding="utf-8"))

    assert info["features"]["observation.context_images"] == {
        "dtype": "image_sequence",
        "shape": [4, 2, 2, 3],
        "names": ["time", "height", "width", "channel"],
        "video_info": {"video.fps": 35.0},
    }


def test_convert_deadly_corridor_canonical_rollout_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("datasets")
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    module_name = "examples.rl_games.bash_scripts.gr00t.data_conversion.convert_deadly_corridor_to_starvla_lerobot"
    sys.modules.pop(module_name, None)
    converter = importlib.import_module(module_name)

    image_type = pyarrow.struct([("bytes", pyarrow.binary()), ("path", pyarrow.string())])
    context_type = pyarrow.list_(image_type)
    raw_dir = tmp_path / "raw"
    for split_name in ("train", "validation"):
        split_dir = raw_dir / split_name
        split_dir.mkdir(parents=True)
        table = pyarrow.table(
            {
                "image": pyarrow.array([_image_entry(50)], type=image_type),
                "context_images": pyarrow.array(
                    [[_image_entry(10), _image_entry(20), _image_entry(30), _image_entry(40)]],
                    type=context_type,
                ),
                "episode_idx": pyarrow.array([0], type=pyarrow.int64()),
                "decision_step": pyarrow.array([0], type=pyarrow.int64()),
                "split": pyarrow.array([split_name], type=pyarrow.string()),
                "prompt": pyarrow.array(["play"], type=pyarrow.string()),
                "latency_raw_frames": pyarrow.array([8], type=pyarrow.int64()),
                "latency_ms": pyarrow.array([228.5714], type=pyarrow.float64()),
                "raw_reward": pyarrow.array([1.0], type=pyarrow.float32()),
                "action_id": pyarrow.array([11], type=pyarrow.int64()),
                "action_text": pyarrow.array(
                    ["MOVE_FORWARD + MOVE_RIGHT + ATTACK"],
                    type=pyarrow.string(),
                ),
            }
        )
        parquet.write_table(table, split_dir / "part-000.parquet")

    output_dir = tmp_path / "converted" / "deadly_corridor_train__bridge"
    manifest = converter.convert_dataset(
        str(raw_dir),
        output_dir,
        force=False,
        action_carrier="bridge",
        action_layout="multibinary_7",
        source_action_layout="deadly_corridor_joint_54",
        context_images_column="context_images",
        image_sequence_length=5,
        fps=35.0,
        obs_stride_raw_frames=4,
        source_latency_column="latency_raw_frames",
        target_latency_unit="raw_frames",
        source_rows_unit="decision_step",
    )

    output_table = parquet.read_table(output_dir / "data/chunk-000/episode_000000.parquet")
    output_info = json.loads((output_dir / "meta/info.json").read_text(encoding="utf-8"))
    assert output_table["action"].to_pylist() == [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]]
    assert len(output_table["observation.context_images"].to_pylist()[0]) == 4
    assert output_info["features"]["observation.image"]["shape"] == [2, 2, 3]
    assert output_info["features"]["observation.context_images"]["shape"] == [4, 2, 2, 3]
    assert manifest["source_action_layout"] == "deadly_corridor_joint_54"
    assert manifest["image_sequence_length"] == 5
