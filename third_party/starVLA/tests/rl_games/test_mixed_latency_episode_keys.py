from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


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

    numpy = ModuleType("numpy")
    pyarrow = ModuleType("pyarrow")
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
def converters(monkeypatch: pytest.MonkeyPatch) -> list[ModuleType]:
    module_names = [
        "examples.rl_games.bash_scripts.gr00t.data_conversion.convert_flappy_to_starvla_lerobot",
        "examples.rl_games.bash_scripts.gr00t.data_conversion.convert_demon_attack_to_starvla_lerobot",
        "examples.rl_games.bash_scripts.gr00t.data_conversion.convert_deadly_corridor_to_starvla_lerobot",
        "examples.rl_games.bash_scripts.gr00t.data_conversion.verify_flappy_dataset",
    ]
    for module_name in module_names:
        sys.modules.pop(module_name, None)
    for module_name, module in _optional_dependency_stubs().items():
        monkeypatch.setitem(sys.modules, module_name, module)

    loaded = [importlib.import_module(module_name) for module_name in module_names[:3]]
    yield loaded
    for module_name in module_names:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("converter_index", [0, 1, 2])
def test_converters_preserve_raw_frame_latency(
    converters: list[ModuleType],
    converter_index: int,
) -> None:
    converter = converters[converter_index]

    assert converter._row_latency(
        {"latency_raw_frames": 6},
        latency_column="latency_raw_frames",
        target_latency_unit="raw_frames",
        obs_stride_raw_frames=4,
        default_latency=None,
    ) == 6


@pytest.mark.parametrize("converter_index", [0, 1, 2])
def test_converters_preserve_legacy_observation_step_ids(
    converters: list[ModuleType],
    converter_index: int,
) -> None:
    converter = converters[converter_index]

    assert converter._row_latency(
        {"latency_raw_frames": 8},
        latency_column="latency_raw_frames",
        target_latency_unit="observation_steps",
        obs_stride_raw_frames=4,
        default_latency=None,
    ) == 2


@pytest.mark.parametrize("converter_index", [0, 1, 2])
def test_full_dataset_fallback_retains_explicit_latency_column(
    monkeypatch: pytest.MonkeyPatch,
    converters: list[ModuleType],
    converter_index: int,
) -> None:
    converter = converters[converter_index]

    class FakeDataset:
        column_names = [
            "episode_idx",
            "decision_step",
            "action_id",
            "raw_reward",
            "prompt",
            "done",
            "latency",
            "latency_raw_frames",
            "latency_ms",
        ]

    def fake_load_split(*args: Any, **kwargs: Any) -> FakeDataset:
        if "columns" in kwargs:
            raise RuntimeError("projected loading unavailable")
        return FakeDataset()

    monkeypatch.setattr(converter, "_load_split", fake_load_split)

    _, columns = converter._load_index_split(
        "fake/dataset",
        "train",
        cache_dir=None,
        want_latency=True,
        source_latency_column="latency_raw_frames",
    )

    assert columns.latency == "latency_raw_frames"


def test_mixed_latency_variants_are_distinct_episode_keys(converters: list[ModuleType]) -> None:
    for converter in converters:
        first = converter._episode_key(0, 0)
        second = converter._episode_key(0, 2)

        assert first != second
        assert sorted([second, first], key=converter._episode_sort_key) == [first, second]
        assert converter._select_episode_ids(
            [first, second],
            {first: 0, second: 2},
            max_episodes=None,
            require_latency_prompt_map=True,
            episodes_per_latency=1,
        ) == [first, second]


def test_demon_attack_index_split_retries_canonical_hf_columns(
    monkeypatch: pytest.MonkeyPatch,
    converters: list[ModuleType],
) -> None:
    demon = converters[1]
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
        columns = kwargs.get("columns") or []
        missing = [column for column in columns if column not in FakeDataset.column_names]
        if missing:
            raise RuntimeError(f"No match for FieldRef.Name({missing[0]})")
        return FakeDataset()

    monkeypatch.setattr(demon, "load_dataset", fake_load_dataset)

    ds, columns = demon._load_index_split(
        "talha1503/demon_attack_mixed_latency_parquet",
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
    assert calls[-1][1]["columns"] == [
        "episode_idx",
        "decision_step",
        "action_id",
        "raw_reward",
        "prompt",
        "latency_raw_frames",
        "latency_ms",
    ]


def test_demon_attack_zero_latency_index_split_retries_canonical_hf_columns(
    monkeypatch: pytest.MonkeyPatch,
    converters: list[ModuleType],
) -> None:
    demon = converters[1]
    calls = []

    class FakeDataset:
        column_names = ["episode_idx", "decision_step", "action_id", "raw_reward", "prompt"]

        def filter(self, fn):
            return self

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        columns = kwargs.get("columns") or []
        missing = [column for column in columns if column not in FakeDataset.column_names]
        if missing:
            raise RuntimeError(f"No match for FieldRef.Name({missing[0]})")
        return FakeDataset()

    monkeypatch.setattr(demon, "load_dataset", fake_load_dataset)

    ds, columns = demon._load_index_split(
        "talha1503/demon_attack_zero_latency_parquet",
        "train",
        cache_dir="/tmp/cache",
        want_latency=False,
    )

    assert isinstance(ds, FakeDataset)
    assert columns.frame == "decision_step"
    assert columns.reward == "raw_reward"
    assert columns.done is None
    assert calls[-1][1]["columns"] == ["episode_idx", "decision_step", "action_id", "raw_reward", "prompt"]


def test_demon_attack_zero_latency_conversion_writes_latency_column(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    converters: list[ModuleType],
) -> None:
    demon = converters[1]
    captured_rows: list[dict[str, Any]] = []
    rows = [
        {
            "episode_idx": 0,
            "decision_step": 0,
            "action_id": 1,
            "raw_reward": 0.5,
            "prompt": "play demon attack",
            "image": object(),
        }
    ]

    class FakeDataset:
        column_names = ["episode_idx", "decision_step", "action_id", "raw_reward", "prompt", "image"]

        def __init__(self, dataset_rows: list[dict[str, Any]]) -> None:
            self.dataset_rows = dataset_rows

        def __len__(self) -> int:
            return len(self.dataset_rows)

        def __iter__(self):
            return iter(self.dataset_rows)

        def filter(self, fn):
            return self

        def select(self, indices: list[int]):
            return FakeDataset([self.dataset_rows[index] for index in indices])

        def cast_column(self, name, feature):
            return self

    def fake_load_dataset(*args, **kwargs):
        columns = kwargs.get("columns")
        if columns is None:
            return FakeDataset(rows)
        missing = [column for column in columns if column not in FakeDataset.column_names]
        if missing:
            raise RuntimeError(f"No match for FieldRef.Name({missing[0]})")
        return FakeDataset([{column: row[column] for column in columns} for row in rows])

    def fake_write_episode(
        path, episode_rows, *, action_dim: int, state_dim: int,
        context_images_output_column=None,
    ) -> None:
        captured_rows.extend(episode_rows)

    monkeypatch.setattr(demon, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(demon, "_png_bytes", lambda image: b"image")
    monkeypatch.setattr(demon, "_write_episode", fake_write_episode)

    demon.convert_dataset(
        "fake/demon_attack_zero_latency",
        tmp_path / "demon_train",
        cache_dir="/tmp/cache",
        dataset_config_name=None,
        dataset_source_subdir=None,
        max_episodes=1,
        force=False,
        require_latency_prompt_map=False,
        latency_filter=None,
        episodes_per_latency=None,
        prompt_map_override=None,
        default_latency=None,
        action_carrier="native",
        fps=15.0,
        obs_stride_raw_frames=4,
        source_latency_column=None,
        target_latency_unit="raw_frames",
        source_rows_unit="decision_step",
    )

    assert captured_rows
    assert all(row["latency"] == 0 for row in captured_rows)


def test_deadly_corridor_index_split_retries_canonical_hf_columns(
    monkeypatch: pytest.MonkeyPatch,
    converters: list[ModuleType],
) -> None:
    deadly = converters[2]
    calls = []

    class FakeDataset:
        column_names = [
            "episode_idx",
            "decision_step",
            "prompt",
            "latency_raw_frames",
            "latency_ms",
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

    monkeypatch.setattr(deadly, "load_dataset", fake_load_dataset)

    ds, columns = deadly._load_index_split(
        "talha1503/deadly_corridor_mixed_latency_parquet",
        "train",
        cache_dir="/tmp/cache",
        want_latency=True,
    )

    assert isinstance(ds, FakeDataset)
    assert columns.frame == "decision_step"
    assert columns.latency == "latency_raw_frames"
    assert calls[-1][1]["columns"] == [
        "episode_idx",
        "decision_step",
        "prompt",
        "latency_raw_frames",
        "latency_ms",
    ]


def test_deadly_corridor_zero_latency_index_split_retries_canonical_hf_columns(
    monkeypatch: pytest.MonkeyPatch,
    converters: list[ModuleType],
) -> None:
    deadly = converters[2]
    calls = []

    class FakeDataset:
        column_names = ["episode_idx", "decision_step", "prompt"]

        def filter(self, fn):
            return self

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        columns = kwargs.get("columns") or []
        missing = [column for column in columns if column not in FakeDataset.column_names]
        if missing:
            raise RuntimeError(f"No match for FieldRef.Name({missing[0]})")
        return FakeDataset()

    monkeypatch.setattr(deadly, "load_dataset", fake_load_dataset)

    ds, columns = deadly._load_index_split(
        "talha1503/deadly_corridor_zero_latency_parquet",
        "train",
        cache_dir="/tmp/cache",
        want_latency=False,
    )

    assert isinstance(ds, FakeDataset)
    assert columns.frame == "decision_step"
    assert calls[-1][1]["columns"] == ["episode_idx", "decision_step", "prompt"]


def test_deadly_corridor_zero_latency_conversion_writes_latency_column(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    converters: list[ModuleType],
) -> None:
    deadly = converters[2]
    captured_rows: list[dict[str, Any]] = []
    rows = [
        {
            "episode_idx": 0,
            "decision_step": 0,
            "raw_reward": 0.5,
            "prompt": "play deadly corridor",
            "image": object(),
            "action_text": "NOOP",
        }
    ]

    class FakeDataset:
        column_names = ["episode_idx", "decision_step", "raw_reward", "prompt", "image", "action_text"]

        def __init__(self, dataset_rows: list[dict[str, Any]]) -> None:
            self.dataset_rows = dataset_rows

        def __len__(self) -> int:
            return len(self.dataset_rows)

        def __iter__(self):
            return iter(self.dataset_rows)

        def filter(self, fn):
            return self

        def select(self, indices: list[int]):
            return FakeDataset([self.dataset_rows[index] for index in indices])

    def fake_load_dataset(*args, **kwargs):
        columns = kwargs.get("columns")
        if columns is None:
            return FakeDataset(rows)
        missing = [column for column in columns if column not in FakeDataset.column_names]
        if missing:
            raise RuntimeError(f"No match for FieldRef.Name({missing[0]})")
        return FakeDataset([{column: row[column] for column in columns} for row in rows])

    def fake_write_episode(path, episode_rows, *, action_dim: int, state_dim: int) -> None:
        captured_rows.extend(episode_rows)

    monkeypatch.setattr(deadly, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(deadly, "_png_bytes", lambda image: b"image")
    monkeypatch.setattr(deadly, "_write_episode", fake_write_episode)

    deadly.convert_dataset(
        "fake/deadly_corridor_zero_latency",
        tmp_path / "deadly_train",
        cache_dir="/tmp/cache",
        dataset_config_name=None,
        dataset_source_subdir=None,
        max_episodes=1,
        force=False,
        require_latency_prompt_map=False,
        latency_filter=None,
        episodes_per_latency=None,
        action_carrier="native",
        fps=8.75,
        obs_stride_raw_frames=4,
        source_latency_column=None,
        target_latency_unit="raw_frames",
        source_rows_unit="decision_step",
        action_layout=deadly.ACTION_LAYOUT_MULTIBINARY_7,
    )

    assert captured_rows
    assert all(row["latency"] == 0 for row in captured_rows)
