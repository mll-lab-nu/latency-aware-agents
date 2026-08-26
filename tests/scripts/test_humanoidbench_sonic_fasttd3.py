from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import TensorProto, helper

import scripts.experiments.humanoidbench_sonic_fasttd3 as sonic
from latency_bench.envs import humanoidbench_sonic_assets as assets
from scripts.experiment_runner.job_manifest import load_manifest


def test_generate_g1_10m_writes_raw_and_sonic_jobs(tmp_path, monkeypatch):
    decoder_path = tmp_path / "decoder.onnx"
    decoder_path.write_bytes(b"decoder")
    asset_root = tmp_path / "sonic_g1"
    scene_paths = {task: tmp_path / f"{task}.xml" for task in sonic.G1_10M_TASKS}
    for task, path in scene_paths.items():
        spec = sonic.G1_SONIC_ENV_SPECS[task]
        target = asset_root / spec["scene_filename"].removeprefix("sonic_g1/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.name.encode())
    monkeypatch.setattr(
        sonic,
        "prepare_g1_sonic_assets",
        lambda _root: (asset_root, decoder_path),
    )
    planner_path = tmp_path / "planner.onnx"
    encoder_path = tmp_path / "encoder.onnx"
    monkeypatch.setattr(
        sonic,
        "prepare_g1_sonic_reference_assets",
        lambda _root: (planner_path, encoder_path),
    )

    def write_tokens(_planner, _encoder, output_path, horizon):
        np.save(output_path, np.zeros((horizon + 1, 64), dtype=np.float32))

    monkeypatch.setattr(sonic, "write_reference_tokens", write_tokens)
    monkeypatch.setattr(
        sonic, "_keyframe_joint_targets", lambda _path, names: [0.0] * len(names)
    )

    summary = sonic.generate_g1_10m(
        output_root=tmp_path / "results",
        train_for_env_steps=128,
    )

    rows = load_manifest(summary["manifest"])
    assert len(rows) == 9
    assert summary["experiment_family"] == "G1-SONIC-10M"
    assert summary["wandb_group"] == "G1-SONIC-10M"
    assert {row["sweep_name"] for row in rows} == {
        "raw43",
        "sonic78",
        "sonic64_residual",
    }
    for row in rows:
        train_config = yaml.safe_load(Path(row["train_config"]).read_text())
        eval_config = yaml.safe_load(Path(row["eval_config"]).read_text())
        assert train_config["logging"]["wandb_name"] == (
            f"results_{train_config['experiment']['name']}"
        )
        assert row["wandb_name"] == train_config["logging"]["wandb_name"]
        variant = row["sweep_name"]
        make_kwargs = train_config["env"]["make_kwargs"]
        if variant == "raw43":
            assert len(train_config["env"]["action_space"]["low"]) == 43
            assert len(train_config["env"]["noop_action"]) == 43
            assert (
                train_config["env"]["action_space"]["labels"]
                == sonic.G1_RAW_ACTION_LABELS
            )
            assert "fasttd3_sonic_decoder_path" not in train_config["training"]
            assert "sonic_decoder_path" not in make_kwargs
        elif variant == "sonic78":
            assert len(train_config["env"]["action_space"]["low"]) == 78
            assert len(train_config["env"]["noop_action"]) == 78
            assert (
                train_config["env"]["action_space"]["labels"]
                == sonic.G1_SONIC_ACTION_LABELS
            )
            assert train_config["training"]["fasttd3_sonic_decoder_path"] == str(
                decoder_path
            )
            assert make_kwargs["sonic_decoder_mode"] == "inference_worker"
            assert eval_config["env"]["make_kwargs"]["sonic_decoder_mode"] == (
                "environment"
            )
        else:
            assert row["env"] == "g1_sonic_hurdle"
            assert len(train_config["env"]["action_space"]["low"]) == 64
            assert len(train_config["env"]["noop_action"]) == 64
            assert (
                train_config["env"]["action_space"]["labels"]
                == sonic.G1_SONIC_RESIDUAL_ACTION_LABELS
            )
            assert train_config["training"]["fasttd3_sonic_decoder_path"] == str(
                decoder_path
            )
            assert make_kwargs["sonic_decoder_mode"] == "inference_worker"
            assert eval_config["env"]["make_kwargs"]["sonic_decoder_mode"] == (
                "environment"
            )
            assert (
                make_kwargs["sonic_base_token_path"]
                == eval_config["env"]["make_kwargs"]["sonic_base_token_path"]
            )
        assert train_config["training"]["train_for_env_steps"] == 128
        assert train_config["backend"]["restart_behavior"] == "restart"
        assert train_config["evaluation"]["eval_interval_steps"] is None
        assert eval_config["evaluation"]["eval_episodes"] == 10


def test_prepare_g1_sonic_assets_returns_explicit_pinned_paths(tmp_path, monkeypatch):
    snapshot = tmp_path / "assets"
    (snapshot / "sonic_g1").mkdir(parents=True)
    decoder = tmp_path / "decoder" / "model_decoder.onnx"
    decoder.parent.mkdir()
    graph = helper.make_graph(
        [],
        "decoder",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 994])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 29])],
    )
    onnx.save(helper.make_model(graph), decoder)
    monkeypatch.setattr(assets, "snapshot_download", lambda **_kwargs: snapshot)
    monkeypatch.setattr(assets, "hf_hub_download", lambda **_kwargs: decoder)

    asset_root, dynamic_decoder = assets.prepare_g1_sonic_assets(tmp_path / "cache")

    assert asset_root == snapshot / "sonic_g1"
    assert dynamic_decoder == decoder.with_name("model_decoder_dynamic.onnx")
    model = onnx.load(dynamic_decoder)
    assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    assert model.graph.output[0].type.tensor_type.shape.dim[0].dim_param == "batch"
