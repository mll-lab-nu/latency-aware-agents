"""Reference Gymnasium teacher/VLA configuration contracts."""
import json

import pytest


def test_air_raid_gr00t_configs_keep_appo_h1_and_realtime_modes(tmp_path):
    from scripts.gym_adapt.gr00t import ROOT, evaluation_config, teacher_config, vla_config

    source = ROOT / "configs/examples/gymnasium/air_raid/small_model_train.yaml"
    teacher = teacher_config("air_raid", tmp_path, None, source)
    assert teacher["experiment"]["name"] == "air_raid_zero_latency_appo_h1"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(json.dumps({
        "robot_type": "rl_games_gymnasium", "action_dim": 6, "active_action_dim": 6,
        "uses_state": False, "state_dim": 6, "gymnasium_task": teacher["env"],
    }))
    config = vla_config("air_raid", dataset, tmp_path / "backbone", tmp_path / "vla",
                        num_gpus=1, micro_batch=8, condition="zero_latency", wandb_entity="test")
    assert config["framework"]["action_model"]["action_horizon"] == 1
    assert config["datasets"]["vla_data"]["action_type"] == "discrete"
    assert config["datasets"]["vla_data"]["include_state"] is False
    assert config["trainer"]["max_train_steps"] == 5000
    for mode in ("zero", "iid", "measured"):
        evaluation = evaluation_config("air_raid", tmp_path, tmp_path / "backbone", tmp_path,
            mode=mode, profile=tmp_path / "profile.json", episodes=20, profiling=False, prompt_key=0)
        assert evaluation["latency"]["method"] == mode
        assert evaluation["executor"]["mode"] == ("realtime" if mode == "measured" else "simulated")
        assert evaluation["evaluation"]["eval_max_steps"] == 3600
        assert evaluation["policy"]["task_manifest_path"] == str(tmp_path / "manifest.json")


@pytest.mark.parametrize("task,state_dim,action_dim", [
    ("inverted_pendulum", 4, 5), ("ant", 27, 8), ("half_cheetah", 17, 6),
    ("hopper", 11, 3), ("humanoid", 376, 17), ("walker2d", 17, 6),
])
def test_appo_gr00t_h1_manifest_and_four_gpu_config(tmp_path, task, state_dim, action_dim):
    from scripts.gym_adapt.gr00t import teacher_config, vla_config, evaluation_config
    from latency_bench.envs.tasks.contracts import gymnasium_task_contract
    teacher = teacher_config(task, tmp_path, None)
    contract = gymnasium_task_contract(teacher["env"])
    assert contract["env_fps"] == contract["obs_fps"] == 5
    assert teacher["backend"]["algo"] == "APPO"
    assert len(contract["state_space"]["labels"]) == state_dim
    if task == "inverted_pendulum":
        assert contract["env_id"] == "LatencyBench/InvertedPendulumDiscrete-v0"
        assert len(contract["action_labels"]) == 5
    else:
        assert len(contract["action_space"]["labels"]) == action_dim
    manifest = {"robot_type": "rl_games_gymnasium", "action_dim": action_dim,
                "active_action_dim": action_dim, "uses_state": True,
                "state_dim": state_dim, "gymnasium_task": contract}
    if task == "hopper":
        manifest["gymnasium_task"] = {
            **contract,
            "task_name": "hopper_rgb_state",
            "env_id": "LatencyBench/HopperRgbState-v0",
            "registration_imports": ["latency_bench.envs.gymnasium_hopper_rgb_state"],
        }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    config = vla_config(task, tmp_path, tmp_path, tmp_path / "out",
                       num_gpus=4, micro_batch=16, condition="zero_latency", wandb_entity="test")
    model = config["framework"]["action_model"]
    assert (model["state_dim"], model["action_dim"], model["action_horizon"]) == (state_dim, action_dim, 1)
    assert model["future_action_window_size"] == 0
    assert config["trainer"]["gradient_accumulation_steps"] == 1
    assert config["trainer"]["pretrained_checkpoint"] is None
    assert config["trainer"]["max_train_steps"] == (2000 if task == "inverted_pendulum" else 5000)
    assert config["datasets"]["vla_data"]["action_type"] == ("discrete" if task == "inverted_pendulum" else "continuous")
    for profiling in (False, True):
        evaluation = evaluation_config(task, tmp_path, tmp_path, tmp_path / "eval",
            mode="zero", profile=None, episodes=20, profiling=profiling, prompt_key=0)
        assert gymnasium_task_contract(evaluation["env"]) == contract
        assert evaluation["evaluation"]["eval_max_steps"] == (900 if profiling else 1000)
    if task == "inverted_pendulum":
        import yaml
        from scripts.gym_adapt.gr00t import export_bundle
        trained = tmp_path / "trained"
        trained.mkdir()
        (trained / "config.full.yaml").write_text(yaml.safe_dump(config))
        (trained / "dataset_statistics.json").write_text("{}")
        checkpoint = trained / "steps_2000_pytorch_model.pt"
        checkpoint.write_bytes(b"test weights")
        (tmp_path / "latency_prompt_map.json").write_text("{}")
        source = tmp_path / "source.json"
        source.write_text("{}")
        bundle = tmp_path / "bundle"
        export_bundle(task, trained, checkpoint, tmp_path, bundle, source_manifest=source)
        inference = yaml.safe_load((bundle / "config.yaml").read_text())
        assert inference["datasets"]["vla_data"]["include_state"] is True
        assert inference["datasets"]["vla_data"]["gymnasium_task_contract"] == contract
        assert inference["framework"]["action_model"]["state_dim"] == 4
        assert inference["framework"]["action_model"]["action_dim"] == 5
        assert json.loads((bundle / "manifest.json").read_text()) == manifest
