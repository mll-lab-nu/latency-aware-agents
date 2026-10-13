"""
Centralized registry that auto-discovers benchmark-specific data configs from
``examples/*/train_files/data_registry/`` and merges them with the base
registries defined in this package.

Three registries are maintained:

* ``DATASET_NAMED_MIXTURES``       – mixture_name → [(dataset, weight, robot_type)]
* ``ROBOT_TYPE_CONFIG_MAP``       – robot_type → DataConfig instance
* ``ROBOT_TYPE_TO_EMBODIMENT_TAG`` – robot_type → EmbodimentTag


Usage::

    from starVLA.dataloader.gr00t_lerobot.registry import (
        ROBOT_TYPE_CONFIG_MAP,
        ROBOT_TYPE_TO_EMBODIMENT_TAG,
        DATASET_NAMED_MIXTURES,
    )
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import re
import sys
from pathlib import Path

# Base registries (kept as fallback / seed values)
from starVLA.dataloader.gr00t_lerobot.data_config import (
    ROBOT_TYPE_CONFIG_MAP as _BASE_CONFIG_MAP,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import (
    ROBOT_TYPE_TO_EMBODIMENT_TAG as _BASE_EMBODIMENT_MAP,
    EmbodimentTag,  # noqa: F401  – re-export for convenience
)
from starVLA.dataloader.gr00t_lerobot.mixtures import (
    DATASET_NAMED_MIXTURES as _BASE_MIXTURES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mutable copies – will be extended by discovered modules
# ---------------------------------------------------------------------------
ROBOT_TYPE_CONFIG_MAP: dict = dict(_BASE_CONFIG_MAP)
ROBOT_TYPE_TO_EMBODIMENT_TAG: dict = dict(_BASE_EMBODIMENT_MAP)
DATASET_NAMED_MIXTURES: dict = dict(_BASE_MIXTURES)
_DEBUG_DATASET_RE = re.compile(r"^(?P<base>.+)__debug(?:_[A-Za-z0-9_-]+)?_(?P<episodes>\d+)ep$")

# ---------------------------------------------------------------------------
# Discovery logic
# ---------------------------------------------------------------------------
_REGISTRY_DIR_NAME = "data_registry"
_DISCOVERED = False


def _find_registry_dirs() -> list[Path]:
    """Return all ``examples/*/train_files/data_registry/`` directories."""
    # Walk up from this file to the repo root
    # registry.py is at starVLA/dataloader/gr00t_lerobot/registry.py
    #   parents: [0]=gr00t_lerobot, [1]=dataloader, [2]=starVLA(pkg), [3]=repo root
    repo_root = Path(__file__).resolve().parents[3]
    examples_dir = repo_root / "examples"
    if not examples_dir.is_dir():
        return []
    dirs: list[Path] = []
    for bench_dir in sorted(examples_dir.iterdir()):
        registry_dir = bench_dir / "train_files" / _REGISTRY_DIR_NAME
        if registry_dir.is_dir():
            dirs.append(registry_dir)
    return dirs


def _load_module_from_path(module_name: str, file_path: Path):
    """Import a Python file as a module with the given name."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def discover_and_merge() -> None:
    """Scan ``examples/*/train_files/data_registry/`` and merge into global registries."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    for registry_dir in _find_registry_dirs():
        bench_name = registry_dir.parents[1].name  # examples/<BenchName>/train_files/data_registry
        prefix = f"_data_registry_{bench_name}"

        # --- data_config.py (may contain all three registries) ---
        cfg_file = registry_dir / "data_config.py"
        if cfg_file.is_file():
            mod = _load_module_from_path(f"{prefix}.data_config", cfg_file)
            if mod:
                if hasattr(mod, "ROBOT_TYPE_CONFIG_MAP"):
                    ROBOT_TYPE_CONFIG_MAP.update(mod.ROBOT_TYPE_CONFIG_MAP)
                    logger.info(f"[registry] Loaded data_config from {bench_name}: {list(mod.ROBOT_TYPE_CONFIG_MAP.keys())}")
                if hasattr(mod, "ROBOT_TYPE_TO_EMBODIMENT_TAG"):
                    ROBOT_TYPE_TO_EMBODIMENT_TAG.update(mod.ROBOT_TYPE_TO_EMBODIMENT_TAG)
                    logger.info(f"[registry] Loaded embodiment_tags from {bench_name} (data_config): {list(mod.ROBOT_TYPE_TO_EMBODIMENT_TAG.keys())}")
                if hasattr(mod, "DATASET_NAMED_MIXTURES"):
                    DATASET_NAMED_MIXTURES.update(mod.DATASET_NAMED_MIXTURES)
                    logger.info(f"[registry] Loaded mixtures from {bench_name} (data_config): {list(mod.DATASET_NAMED_MIXTURES.keys())}")


def get_dataset_named_mixture(data_mix: str):
    """Return a mixture spec, including debug-subset variants of known mixtures."""
    if data_mix in DATASET_NAMED_MIXTURES:
        return DATASET_NAMED_MIXTURES[data_mix]

    base_data_mix = str(data_mix)
    if base_data_mix.endswith("__val"):
        base_data_mix = base_data_mix.removesuffix("__val")

    if base_data_mix in DATASET_NAMED_MIXTURES:
        return [
            (data_mix, weight, robot_type)
            for _, weight, robot_type in DATASET_NAMED_MIXTURES[base_data_mix]
        ]

    match = _DEBUG_DATASET_RE.match(base_data_mix)
    if match:
        base_mix = match.group("base")
        if base_mix in DATASET_NAMED_MIXTURES:
            return [
                (data_mix, weight, robot_type)
                for _, weight, robot_type in DATASET_NAMED_MIXTURES[base_mix]
            ]

    raise KeyError(data_mix)


def load_custom_mixtures(path: str | Path | None) -> None:
    """Load generated mixture specs from a JSON file.

    The expected JSON shape is:
        {"mixture_name": [["dataset_name", weight, "robot_type"], ...]}
    """
    if path in (None, ""):
        return
    mixture_path = Path(path).expanduser()
    if not mixture_path.exists():
        raise FileNotFoundError(f"custom mixture file does not exist: {mixture_path}")
    payload = json.loads(mixture_path.read_text(encoding="utf-8"))
    for name, entries in payload.items():
        DATASET_NAMED_MIXTURES[str(name)] = [
            (str(dataset_name), float(weight), str(robot_type))
            for dataset_name, weight, robot_type in entries
        ]
    logger.info(f"[registry] Loaded custom mixtures from {mixture_path}: {list(payload.keys())}")


def resolve_robot_type(
    model_cfg: dict,
    unnorm_key: str | None = None,
) -> str:
    """Look up the training robot_type from the saved cfg.

    Convention used by starVLA train scripts:
      ``cfg.datasets.vla_data.data_mix`` is a key in
      ``DATASET_NAMED_MIXTURES`` whose value is a list of
      ``(dataset_name, weight, robot_type)`` tuples.

    When a data_mix contains entries from multiple robot types (e.g.
    ``bridge_rt_1`` covers ``oxe_bridge`` + ``oxe_rt1``), ``unnorm_key``
    is used to identify which embodiment is requested.  In those mixtures
    the ``robot_type`` field of each entry **matches** the top-level key in
    ``dataset_statistics.json``, so ``unnorm_key`` serves as the selector.
    Inference bundles can store the resolved robot_type directly when their
    training mixture is not installed on the inference host.
    """
    vla_data = model_cfg["datasets"]["vla_data"]
    if "robot_type" in vla_data:
        return vla_data["robot_type"]
    try:
        data_mix = vla_data["data_mix"]
    except (KeyError, TypeError) as e:
        raise KeyError(
            "ckpt config.yaml is missing `datasets.vla_data.data_mix`; "
            "cannot resolve training-time robot_type."
        ) from e

    if "custom_mixtures_path" in vla_data:
        load_custom_mixtures(vla_data["custom_mixtures_path"])
    mixture = get_dataset_named_mixture(data_mix)
    robot_types = sorted({entry[2] for entry in mixture})

    if len(robot_types) == 1:
        return robot_types[0]

    # Multiple robot types in the mixture.
    # Use unnorm_key as a direct selector: for multi-robot mixtures the
    # dataset_statistics.json top-level keys equal the robot_type values.
    if unnorm_key is not None and unnorm_key in robot_types:
        return unnorm_key

    raise ValueError(
        f"data_mix={data_mix!r} contains multiple robot_types {robot_types}. "
        "Pass `unnorm_key` matching one of them to disambiguate "
        f"(e.g. unnorm_key={robot_types[0]!r})."
    )


# Run discovery on first import
discover_and_merge()
