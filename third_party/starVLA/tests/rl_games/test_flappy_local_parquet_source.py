from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import ModuleType

import pytest


class FakeDatasetImage:
    def __init__(self, decode: bool) -> None:
        self.decode = decode


class FakeDatasetSequence:
    def __init__(self, feature: object) -> None:
        self.feature = feature


class FakeListFeature:
    def __init__(self, feature: object) -> None:
        self.feature = feature


def _optional_dependency_stubs() -> dict[str, ModuleType]:
    datasets = ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: None
    datasets.Image = FakeDatasetImage
    datasets.Sequence = FakeDatasetSequence

    numpy = ModuleType("numpy")
    pyarrow = ModuleType("pyarrow")
    pyarrow.ArrowInvalid = ValueError
    pyarrow_parquet = ModuleType("pyarrow.parquet")
    pil = ModuleType("PIL")
    pil_image = ModuleType("PIL.Image")
    pil.Image = pil_image
    return {
        "datasets": datasets,
        "numpy": numpy,
        "pyarrow": pyarrow,
        "pyarrow.parquet": pyarrow_parquet,
        "PIL": pil,
        "PIL.Image": pil_image,
    }


@pytest.fixture()
def flappy_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    module_names = (
        "examples.rl_games.bash_scripts.gr00t.data_conversion.convert_flappy_to_starvla_lerobot",
        "examples.rl_games.bash_scripts.gr00t.data_conversion.verify_flappy_dataset",
    )
    for module_name in module_names:
        sys.modules.pop(module_name, None)
    for module_name, module in _optional_dependency_stubs().items():
        monkeypatch.setitem(sys.modules, module_name, module)

    convert_flappy = importlib.import_module(
        "examples.rl_games.bash_scripts.gr00t.data_conversion.convert_flappy_to_starvla_lerobot"
    )
    verify_flappy = importlib.import_module("examples.rl_games.bash_scripts.gr00t.data_conversion.verify_flappy_dataset")
    yield convert_flappy, verify_flappy
    for module_name in module_names:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize(
    ("template", "latency", "expected"),
    [
        ("zero-latency/flappy/flappy_200ep/flappy_fix_latency_0_200ep", 3,
         "latency-aware/flappy/flappy_200ep/flappy_fix_latency_3_200ep"),
        ("latency-aware/flappy/flappy_200ep/flappy_fix_latency_3_200ep", 0,
         "zero-latency/flappy/flappy_200ep/flappy_fix_latency_0_200ep"),
    ],
)
def test_mixed_latency_partitions_use_the_canonical_training_namespace(flappy_modules, template, latency, expected):
    """Retain the mixed-training contract when L0 and delayed data have different owners' directories."""
    _, verifier = flappy_modules
    assert verifier.latency_subdir_for(template, latency) == expected


def test_convert_flappy_resolves_local_parquet_directory(
    tmp_path: Path,
    flappy_modules: tuple[ModuleType, ModuleType],
) -> None:
    convert_flappy, _ = flappy_modules
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    train_dir.mkdir()
    validation_dir.mkdir()
    (train_dir / "part-000.parquet").touch()
    (validation_dir / "part-000.parquet").touch()

    train_files = convert_flappy._local_parquet_files(str(tmp_path), "train")
    validation_files = convert_flappy._local_parquet_files(str(tmp_path), "validation")

    assert train_files == [str(train_dir / "part-000.parquet")]
    assert validation_files == [str(validation_dir / "part-000.parquet")]


def test_convert_flappy_skips_split_filter_for_split_specific_local_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flappy_modules: tuple[ModuleType, ModuleType],
) -> None:
    convert_flappy, _ = flappy_modules
    train_file = tmp_path / "train.parquet"
    train_file.touch()

    class FakeDataset:
        column_names = ["split", "image"]

        def filter(self, fn):
            raise AssertionError("split-specific local files should not run row-level split filtering")

    fake_dataset = FakeDataset()

    def fake_load_dataset(*args, **kwargs):
        return fake_dataset

    monkeypatch.setattr(convert_flappy, "load_dataset", fake_load_dataset)

    loaded = convert_flappy._load_split(str(tmp_path), "train")

    assert loaded is fake_dataset


def test_convert_flappy_casts_image_columns_to_encoded_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flappy_modules: tuple[ModuleType, ModuleType],
) -> None:
    convert_flappy, _ = flappy_modules
    train_file = tmp_path / "train.parquet"
    train_file.touch()

    class FakeDataset:
        column_names = ["image", "context_images"]
        features = {
            "image": FakeDatasetImage(decode=True),
            "context_images": FakeListFeature(FakeDatasetImage(decode=True)),
        }

        def __init__(self) -> None:
            self.cast_calls: list[tuple[str, object]] = []

        def cast_column(self, column: str, feature: object):
            self.cast_calls.append((column, feature))
            return self

    fake_dataset = FakeDataset()

    def fake_load_dataset(*args, **kwargs):
        return fake_dataset

    monkeypatch.setattr(convert_flappy, "load_dataset", fake_load_dataset)

    loaded = convert_flappy._load_split(
        str(tmp_path),
        "train",
        image_columns=["image", "context_images"],
    )

    assert loaded is fake_dataset
    assert [column for column, _ in fake_dataset.cast_calls] == ["image", "context_images"]
    assert isinstance(fake_dataset.cast_calls[0][1], FakeDatasetImage)
    assert fake_dataset.cast_calls[0][1].decode is False
    assert isinstance(fake_dataset.cast_calls[1][1], FakeDatasetSequence)
    assert isinstance(fake_dataset.cast_calls[1][1].feature, FakeDatasetImage)
    assert fake_dataset.cast_calls[1][1].feature.decode is False


def test_verify_flappy_resolves_local_parquet_directory(
    tmp_path: Path,
    flappy_modules: tuple[ModuleType, ModuleType],
) -> None:
    _, verify_flappy = flappy_modules
    parquet_file = tmp_path / "part-000.parquet"
    parquet_file.touch()

    files = verify_flappy._local_parquet_files(str(tmp_path))

    assert files == [str(parquet_file)]


def test_verify_flappy_prefers_train_parquet_files(
    tmp_path: Path,
    flappy_modules: tuple[ModuleType, ModuleType],
) -> None:
    _, verify_flappy = flappy_modules
    train_file = tmp_path / "train-00000.parquet"
    validation_file = tmp_path / "validation-00000.parquet"
    train_file.touch()
    validation_file.touch()

    files = verify_flappy._local_parquet_files(str(tmp_path))

    assert files == [str(train_file)]


def test_convert_flappy_resolves_clean_v1_column_aliases(
    monkeypatch: pytest.MonkeyPatch,
    flappy_modules: tuple[ModuleType, ModuleType],
) -> None:
    convert_flappy, _ = flappy_modules
    monkeypatch.setattr(
        convert_flappy,
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

    columns = convert_flappy._resolve_flappy_columns("flappy_clean_v1", "train", want_latency=True)

    assert columns.frame == "decision_step"
    assert columns.reward == "raw_reward"
    assert columns.done is None
    assert columns.latency == "latency_raw_frames"
    assert columns.latency_ms == "latency_ms"


def test_convert_flappy_marks_last_frame_done_when_done_column_is_absent(
    flappy_modules: tuple[ModuleType, ModuleType],
) -> None:
    convert_flappy, _ = flappy_modules

    assert convert_flappy._row_done({"done": True}, "done", frame_idx=0, episode_length=3) is True
    assert convert_flappy._row_done({}, None, frame_idx=0, episode_length=3) is False
    assert convert_flappy._row_done({}, None, frame_idx=2, episode_length=3) is True


def test_convert_flappy_hf_loader_passes_dataset_config_name(
    monkeypatch: pytest.MonkeyPatch,
    flappy_modules: tuple[ModuleType, ModuleType],
) -> None:
    convert_flappy, _ = flappy_modules
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return "dataset"

    monkeypatch.setattr(convert_flappy, "load_dataset", fake_load_dataset)

    result = convert_flappy._load_hf_dataset(
        "example/flappy-configs",
        "flappy_clean_v1",
        None,
        split="train",
        cache_dir="/tmp/cache",
        columns=["prompt"],
    )

    assert result == "dataset"
    assert calls == [
        (
            ("example/flappy-configs", "flappy_clean_v1"),
            {
                "split": "train",
                "cache_dir": "/tmp/cache",
                "columns": ["prompt"],
                "verification_mode": "no_checks",
            },
        )
    ]


def test_convert_flappy_hf_loader_passes_dataset_source_subdir(
    monkeypatch: pytest.MonkeyPatch,
    flappy_modules: tuple[ModuleType, ModuleType],
) -> None:
    convert_flappy, _ = flappy_modules
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return "dataset"

    monkeypatch.setattr(convert_flappy, "load_dataset", fake_load_dataset)

    result = convert_flappy._load_hf_dataset(
        "latency-sensitive-bench/flappy_200ep",
        None,
        "flappy_fix_latency_0_200ep",
        split="train",
        cache_dir="/tmp/cache",
        columns=["prompt"],
    )

    assert result == "dataset"
    assert calls == [
        (
            ("latency-sensitive-bench/flappy_200ep",),
            {
                "split": "train",
                "cache_dir": "/tmp/cache",
                "columns": ["prompt"],
                "verification_mode": "no_checks",
                "data_dir": "flappy_fix_latency_0_200ep",
            },
        )
    ]


def test_convert_flappy_index_split_retries_canonical_hf_columns(
    monkeypatch: pytest.MonkeyPatch,
    flappy_modules: tuple[ModuleType, ModuleType],
) -> None:
    convert_flappy, _ = flappy_modules
    calls = []

    class FakeDataset:
        column_names = [
            "episode_idx",
            "decision_step",
            "action_id",
            "raw_reward",
            "prompt",
            "latency_raw_frames",
            "latency_ms",
        ]

        def filter(self, fn):
            return self

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        if "t" in kwargs.get("columns", []):
            raise RuntimeError("No match for FieldRef.Name(t)")
        return FakeDataset()

    monkeypatch.setattr(convert_flappy, "load_dataset", fake_load_dataset)

    ds, columns = convert_flappy._load_index_split(
        "latency-sensitive-bench/flappy_200ep",
        "train",
        cache_dir="/tmp/cache",
        want_latency=True,
    )

    assert isinstance(ds, FakeDataset)
    assert columns.frame == "decision_step"
    assert columns.reward == "raw_reward"
    assert columns.done is None
    assert columns.latency == "latency_raw_frames"
    assert calls[0][1]["columns"] == [
        "episode_idx",
        "t",
        "action_id",
        "reward",
        "prompt",
        "done",
        "latency",
        "latency_ms",
    ]
    assert calls[1][1]["columns"] == [
        "episode_idx",
        "decision_step",
        "action_id",
        "raw_reward",
        "prompt",
        "latency_raw_frames",
        "latency_ms",
    ]


def test_convert_flappy_zero_latency_index_split_retries_canonical_hf_columns(
    monkeypatch: pytest.MonkeyPatch,
    flappy_modules: tuple[ModuleType, ModuleType],
) -> None:
    convert_flappy, _ = flappy_modules
    calls = []

    class FakeDataset:
        column_names = [
            "episode_idx",
            "decision_step",
            "action_id",
            "raw_reward",
            "prompt",
        ]

        def filter(self, fn):
            return self

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        columns = kwargs.get("columns") or []
        missing = [column for column in columns if column not in FakeDataset.column_names]
        if missing:
            raise RuntimeError(f"No match for FieldRef.Name({missing[0]})")
        return FakeDataset()

    monkeypatch.setattr(convert_flappy, "load_dataset", fake_load_dataset)

    ds, columns = convert_flappy._load_index_split(
        "latency-sensitive-bench/flappy_200ep",
        "train",
        cache_dir="/tmp/cache",
        want_latency=False,
    )

    assert isinstance(ds, FakeDataset)
    assert columns.frame == "decision_step"
    assert columns.reward == "raw_reward"
    assert columns.latency is None
    assert calls[-1][1]["columns"] == ["episode_idx", "decision_step", "action_id", "raw_reward", "prompt"]
