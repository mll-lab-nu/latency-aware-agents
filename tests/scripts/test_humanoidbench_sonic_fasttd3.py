from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import TensorProto, helper

from latency_bench.envs import humanoidbench_sonic_assets as assets
from scripts.experiments.humanoidbench import fasttd3 as sonic
from scripts.experiment_runner.job_manifest import load_manifest


def test_generate_g1_experiment_writes_29d_one_million_step_job(tmp_path, monkeypatch):
    decoder_path = tmp_path / "decoder.onnx"
    decoder_path.write_bytes(b"decoder")
    asset_root = tmp_path / "sonic_g1"
    spec = sonic.G1_SONIC_ENV_SPECS[sonic.HURDLE_TASK]
    scene_path = asset_root / spec["scene_filename"].removeprefix("sonic_g1/")
    scene_path.parent.mkdir(parents=True)
    scene_path.write_bytes(b"scene")
    monkeypatch.setattr(
        sonic,
        "prepare_g1_sonic_assets",
        lambda _root: (asset_root, decoder_path),
    )
    encoder_path = tmp_path / "encoder.onnx"
    monkeypatch.setattr(
        sonic,
        "prepare_g1_sonic_encoder",
        lambda _root: encoder_path,
    )
    planner_path = tmp_path / "planner.onnx"
    monkeypatch.setattr(
        sonic,
        "prepare_g1_sonic_planner",
        lambda _root: planner_path,
    )

    monkeypatch.setattr(
        sonic,
        "prepare_g1_sonic_skill_references",
        lambda _root: {"obstacle_jump": tmp_path / "jump.csv"},
    )

    def write_bundle(_planner, _skills, _scene, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            run_encoder_qpos=np.zeros((47, 36), dtype=np.float32),
            jump_encoder_qpos=np.zeros((47, 36), dtype=np.float32),
            run_reference_qpos=np.zeros((2, 36), dtype=np.float32),
            jump_reference_qpos=np.zeros((2, 36), dtype=np.float32),
            run_reference_foot_contacts=np.zeros((2, 2), dtype=bool),
        )

    monkeypatch.setattr(sonic, "_write_reference_bundle", write_bundle)
    summary = sonic.generate_g1_experiment(output_root=tmp_path / "results")

    rows = load_manifest(summary["manifest"])
    assert len(rows) == 1
    assert summary["experiment_family"] == sonic.G1_EXPERIMENT_FAMILY
    assert summary["task"] == "g1_sonic_hurdle"
    assert summary["variant"] == sonic.SONIC_VARIANT
    assert summary["hurdle_height"] == sonic.HURDLE_HEIGHT
    assert summary["train_for_env_steps"] == sonic.TRAIN_FOR_ENV_STEPS
    assert summary["eval_interval_steps"] == sonic.EVAL_INTERVAL_STEPS
    row = rows[0]
    train_config = yaml.safe_load(Path(row["train_config"]).read_text())
    eval_config = yaml.safe_load(Path(row["eval_config"]).read_text())
    assert row["sweep_name"] == sonic.SONIC_VARIANT
    assert sonic.SONIC_VARIANT in train_config["logging"]["wandb_name"]
    assert row["wandb_name"] == train_config["logging"]["wandb_name"]
    make_kwargs = train_config["env"]["make_kwargs"]
    action_space = train_config["env"]["action_space"]
    assert len(action_space["low"]) == len(sonic.G1_BODY_JOINT_NAMES) == 29
    assert action_space["low"] == [-1] * 29
    assert action_space["high"] == [1] * 29
    assert len(train_config["env"]["noop_action"]) == 29
    assert action_space["labels"] == [
        f"{name}_residual" for name in sonic.G1_BODY_JOINT_NAMES
    ]
    assert train_config["training"]["fasttd3_sonic_decoder_path"] == str(
        decoder_path
    )
    assert make_kwargs["sonic_encoder_path"] == str(encoder_path)
    assert make_kwargs["sonic_decoder_mode"] == "inference_worker"
    assert make_kwargs["hurdle_height"] == sonic.HURDLE_HEIGHT
    assert make_kwargs["hurdle_count"] == 1
    assert make_kwargs["hurdle_horizon"] == 400
    assert train_config["training"]["gamma"] == 0.999
    assert train_config["training"]["fasttd3_v_min"] == -300.0
    assert train_config["training"]["fasttd3_v_max"] == 300.0
    assert train_config["training"]["fasttd3_actor_action_l2"] == 1.0
    assert eval_config["env"]["make_kwargs"]["sonic_decoder_mode"] == "environment"
    assert eval_config["env"]["make_kwargs"]["hurdle_height"] == sonic.HURDLE_HEIGHT
    assert (
        make_kwargs["sonic_reference_bundle_path"]
        == eval_config["env"]["make_kwargs"]["sonic_reference_bundle_path"]
    )
    assert train_config["training"]["train_for_env_steps"] == 1_000_000
    assert train_config["backend"]["restart_behavior"] == "restart"
    assert train_config["evaluation"]["eval_interval_steps"] == 100_000
    assert train_config["evaluation"]["eval_suites"] == {"fixed": [[0]]}
    assert eval_config["policy"]["checkpoint_kind"] == "latest"
    assert eval_config["evaluation"]["eval_episodes"] == 50
    assert eval_config["evaluation"]["eval_max_steps"] == 400
    assert eval_config["logging"]["save_action_records"] is True
    assert train_config["experiment"]["name"] == row["run_name"]
    assert row["checkpoint_dir"] == str(
        Path(train_config["backend"]["train_dir"])
        / row["run_name"]
        / "checkpoint_p0"
    )


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
