from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import yaml


TRAIN_NAME = "local/intercept_grab_fast_vla_v0_h1_train"
VAL_NAME = "local/intercept_grab_fast_vla_v0_h1_val"


def test_prepare_starvla_dataset_writes_exact_h1_modalities(tmp_path: Path):
    for name in (TRAIN_NAME, VAL_NAME):
        (tmp_path / name / "meta").mkdir(parents=True)

    subprocess.run(
        [
            sys.executable,
            "scripts/mikasa/prepare_starvla_dataset.py",
            "--data-root",
            str(tmp_path),
            "--train-name",
            TRAIN_NAME,
            "--val-name",
            VAL_NAME,
        ],
        check=True,
    )

    modality = json.loads((tmp_path / TRAIN_NAME / "meta/modality.json").read_text())
    assert modality["video"]["top"]["original_key"] == "observation.images.top"
    assert modality["video"]["wrist"]["original_key"] == "observation.images.wrist"
    assert modality["state"]["proprio"]["end"] == 7
    assert modality["action"]["eef"]["end"] == 7

    mixtures = json.loads((tmp_path / "mikasa_intercept_grab_fast_h1.json").read_text())
    assert mixtures[TRAIN_NAME] == [[TRAIN_NAME, 1.0, "mikasa_franka_h1"]]
    assert mixtures[VAL_NAME] == [[VAL_NAME, 1.0, "mikasa_franka_h1"]]


def test_starvla_exposes_training_and_deployment_contracts():
    starvla_root = Path("third_party/starVLA")
    config = yaml.safe_load(
        (starvla_root / "examples/MIKASA/train_files/starvla_qwenoft_intercept_grab_fast_h1.yaml").read_text()
    )

    assert config["datasets"]["vla_data"]["lerobot_version"] == "v3.0"
    assert config["framework"]["action_model"]["action_horizon"] == 1

    module = ast.parse(
        (starvla_root / "examples/MIKASA/train_files/data_registry/data_config.py").read_text()
    )
    data_config = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "MikasaFrankaH1DataConfig"
    )
    class_attributes = {
        target.id: ast.unparse(node.value)
        for node in data_config.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert class_attributes["embodiment_tag"] == "EmbodimentTag.NEW_EMBODIMENT"

    mixtures = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "DATASET_NAMED_MIXTURES" for target in node.targets)
    )
    assert mixtures[TRAIN_NAME] == [(TRAIN_NAME, 1.0, "mikasa_franka_h1")]
