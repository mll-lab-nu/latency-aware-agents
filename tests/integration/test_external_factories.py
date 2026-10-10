"""Long-lived custom factory contract across CLI and real subprocess boundaries."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from helpers import REPO_ROOT


@pytest.fixture
def user_project(tmp_path):
    project = tmp_path / "user_project"
    project.mkdir()
    shutil.copyfile(
        Path(__file__).parent / "fixtures/external_components.py",
        project / "outside_lagen.py",
    )
    return project


def _run_user_project(project, mode, policy_factory="make_policy", *, collection=False,
                      parallel_envs=1, inference_pool=False, continuous=False, include_state=False):
    output = project / "output"
    config = {
        "experiment": {"name": "outside_project", "seed": 7},
        "executor": {"mode": mode, "inference_devices": ["cpu"]},
        "env": {
            "name": "custom", "factory": "outside_lagen:make_env",
            "env_fps": 20, "obs_fps": 20, "noop_action": 0,
        },
        "policy": {"type": "custom", "factory": f"outside_lagen:{policy_factory}"},
        "latency": {"method": "measured" if mode == "realtime" else "zero"},
        "scheduler": {"hold_policy": "hold", "ordering_policy": "latest_ready"},
        "evaluation": {"eval_episodes": 2, "eval_max_steps": 4},
        "logging": {
            "output_dir": str(output), "save_step_records": True,
            "save_action_records": True, "save_latency_records": True,
            "realtime_pipeline_profile": mode == "realtime",
        },
        "factory_audit_path": str(project / "audit.jsonl"),
    }
    config_path = project / "eval.yaml"
    if collection:
        config["env"]["factory"] = "outside_lagen:make_collection_env"
        config["policy"]["factory"] = "outside_lagen:make_collection_policy"
        config["env"].update(task_name="external_counter", base_prompt="Choose the next count.",
                             action_order=["zero", "one", "two", "three", "four"],
                             action_map={"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4},
                             noop_action="zero")
        config["collection"] = {"student_observation": {"image_key": None if parallel_envs > 1 else "rgb"}, "val_fraction": 0.5}
        if continuous:
            for key in ("action_order", "action_map"):
                del config["env"][key]
            config["env"].update(
                noop_action=[0.0],
                action_space={"type": "box", "shape": [1], "dtype": "float32",
                              "low": [-5.0], "high": [5.0], "labels": ["command"]},
            )
        if include_state:
            config["env"]["state_space"] = {"labels": ["count"]}
            config["collection"]["student_observation"].update(
                state_key="student_state", state_source="data",
            )
        config["latency"] = {"method": "fixed", "fixed_latency_ms": 100} if mode == "simulated" else {"method": "zero"}
        config["evaluation"]["eval_parallel_envs"] = parallel_envs
        config["executor"]["simulated_inference_pool"] = inference_pool
        if inference_pool:
            config["executor"]["inference_batch_size"] = parallel_envs
    config_path.write_text(yaml.safe_dump(config))
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(project), str(REPO_ROOT)]),
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "MPLBACKEND": "Agg", "LATENCY_BENCH_ENV_BACKEND_START_METHOD": "spawn",
    }
    command = [sys.executable, "-m", "latency_bench.run", "--config", str(config_path)]
    if collection:
        command = [sys.executable, "-c",
                   "from latency_bench.core.config import load_config; "
                   "from latency_bench.data.rollout import collect_rollouts; "
                   "import sys; collect_rollouts(load_config(sys.argv[1]))", str(config_path)]
    result = subprocess.run(
        command,
        cwd=project, env=env, capture_output=True, text=True, timeout=60,
    )
    assert (project / "audit.jsonl").exists(), result.stdout + result.stderr
    records = [json.loads(line) for line in (project / "audit.jsonl").read_text().splitlines()]
    return result, output, records


@pytest.mark.parametrize("mode,parallel_envs,inference_pool", [
    ("paused", 1, False), ("simulated", 1, False), ("simulated", 2, False),
    ("simulated", 1, True), ("simulated", 2, True),
])
def test_external_collection_preserves_inputs_across_execution_modes(
    user_project, mode, parallel_envs, inference_pool,
):
    import io
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot

    result, output, records = _run_user_project(
        user_project, mode, collection=True, parallel_envs=parallel_envs, inference_pool=inference_pool,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    all_rows = []
    for split in ("train", "val"):
        rows = pq.read_table(output / f"{split}.parquet").to_pylist()
        assert rows
        for row in rows:
            rgb = np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])))
            assert (rgb == row["decision_step"]).all()
            assert row["action_id"] == row["decision_step"] + 1
        all_rows.extend(rows)
    assert {row["episode_idx"] for row in all_rows} == {0, 1}
    assert any(row["event"] == "env_closed" for row in records)
    converted = convert_canonical_rollout_to_lerobot(output, output.with_name("converted"), action_carrier="native")
    assert converted["frames"] == 4


@pytest.mark.parametrize("mode", ["simulated", "realtime"])
def test_external_factories_run_from_user_project(user_project, mode):
    result, output, records = _run_user_project(user_project, mode)
    assert result.returncode == 0, result.stdout + result.stderr
    env_pid = next(row["pid"] for row in records if row["event"] == "env_created")
    policy_pid = next(row["pid"] for row in records if row["event"] == "policy_created")
    assert (env_pid != policy_pid) == (mode == "realtime")
    assert any(row["event"] == "env_closed" and row["pid"] == env_pid for row in records)
    metrics = [json.loads(line) for line in (output / "episode_metrics.jsonl").read_text().splitlines()]
    assert [row["episode_return"] for row in metrics] == [4.0, 4.0]
    if mode != "realtime":
        steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
        for episode in range(2):
            assert [row["active_action"]["value"] for row in steps if row["episode_id"] == episode] == [1, 2, 3, 4]
    resolved = yaml.safe_load((output / "resolved_config.yaml").read_text())
    assert resolved["env"]["factory"] == "outside_lagen:make_env"
    assert resolved["policy"]["factory"] == "outside_lagen:make_policy"


def test_realtime_constructor_failure_is_reported_and_environment_closed(user_project):
    result, _, records = _run_user_project(user_project, "realtime", policy_factory="fail_policy")
    assert result.returncode != 0
    assert "user policy construction failed" in result.stderr
    assert any(row["event"] == "env_closed" for row in records)


@pytest.mark.parametrize("mode,continuous,include_state", [
    ("paused", False, False), ("simulated", False, True),
    ("simulated", True, False), ("paused", True, True),
])
def test_collected_dataset_loads_starvla_training_batch(
    user_project, monkeypatch, mode, continuous, include_state,
):
    """Long-lived collection/conversion/loader contract for external task authors."""
    pytest.importorskip("accelerate")
    import numpy as np
    import pyarrow.parquet as pq
    from accelerate import Accelerator
    from omegaconf import OmegaConf
    from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot

    monkeypatch.syspath_prepend(str(REPO_ROOT / "third_party/starVLA"))
    from starVLA.dataloader import build_dataloader

    result, raw, _ = _run_user_project(
        user_project, mode, collection=True, continuous=continuous, include_state=include_state,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = convert_canonical_rollout_to_lerobot(raw, user_project / "lerobot")
    # Use the training entrypoint, including generated mixtures and its collator.
    Accelerator(cpu=True)
    cfg = OmegaConf.create({"datasets": {"vla_data": {
        "data_root_dir": str(user_project), "data_mix": "lerobot",
        "custom_mixtures_path": manifest["custom_mixtures_path"],
        "gymnasium_task_contract": manifest["gymnasium_task"],
        "active_action_dim": manifest["active_action_dim"], "include_state": include_state,
        "video_backend": "pyav", "per_device_batch_size": 2,
        "num_workers": 0, "pin_memory": False, "persistent_workers": False,
    }}})
    episode_ids = []
    for split, data_mix in (("train", "lerobot"), ("val", "lerobot__val")):
        rows = pq.read_table(raw / f"{split}.parquet").to_pylist()
        episode_ids.append({row["episode_idx"] for row in rows})
        loader = build_dataloader(cfg, dataset_py="lerobot_datasets", data_mix=data_mix,
                                  mode="train" if split == "train" else "eval",
                                  save_statistics_filename=None)
        batch = next(iter(loader))
        assert len(batch) == 2
        for sample in batch:
            pixel = int(np.asarray(sample["image"][0])[0, 0, 0])
            row = next(row for row in rows if row["decision_step"] == pixel)
            np.testing.assert_array_equal(sample["action"], [row["action"]])
            assert sample["lang"] == row["prompt"]
            assert sample["action_env_dim"] == (1 if continuous else 5)
            assert ("state" in sample) == include_state
            if include_state:
                np.testing.assert_allclose(sample["state"], [[pixel * 2 / 3 - 1]], atol=1e-6)
    assert episode_ids[0].isdisjoint(episode_ids[1])
