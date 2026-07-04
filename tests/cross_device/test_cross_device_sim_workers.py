from __future__ import annotations

import json
from pathlib import Path

from latency_bench.cross_device.models import ProtocolTask
from latency_bench.cross_device import sim_workers


def _task(tmp_path, *, n_episodes: int = 5) -> ProtocolTask:
    return ProtocolTask(
        stage="sim_same",
        experiment_id="exp2_same_device_calibration",
        experiment_ids=["exp2_same_device_calibration"],
        panel="P2",
        row={
            "policy": "OpenVLA",
            "env": "flappy",
            "policy_type": "vla",
            "model_type": "openvla",
            "env_name": "flappy",
            "runtime_env": "openvla",
        },
        target_device={"id": "L_L40S", "display_name": "L40S", "profile_id": "l40s"},
        replay_device=None,
        seed=7,
        seed_set=[7],
        n_episodes=n_episodes,
        checkpoint_kind="best",
        output_dir=tmp_path / "runs" / "sim_same",
        generated_config_path=tmp_path / "configs" / "sim_same.yaml",
        target_profile_path=tmp_path / "profiles" / "profile.json",
        profile_output_dir=None,
        instance_id="instance_abc",
        instance_origin="manual_override",
        target_profile_instance_id="instance_abc",
        runtime_env_alias="openvla",
        runtime_conda_env="openvla",
    )


def test_build_sim_worker_plan_splits_episodes_across_visible_devices(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    task = _task(tmp_path, n_episodes=5)

    plan = sim_workers.build_sim_worker_plan(task, {"sim_worker_count": "auto"})

    assert [item["cuda_visible_devices"] for item in plan] == ["0", "1", "2"]
    assert [(item["episode_start"], item["episode_count"]) for item in plan] == [(0, 2), (2, 2), (4, 1)]
    assert [Path(item["output_dir"]).name for item in plan] == ["worker_0", "worker_1", "worker_2"]
    assert all("--episode-start" in item["command"] for item in plan)
    assert all("--episode-count" in item["command"] for item in plan)


def test_build_sim_worker_plan_uses_single_process_for_single_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert sim_workers.build_sim_worker_plan(_task(tmp_path), {"sim_worker_count": "auto"}) == []


def test_worker_eval_command_redirects_output_and_episode_window(tmp_path):
    task = _task(tmp_path)
    output_dir = tmp_path / "worker"

    cmd = sim_workers.worker_eval_command(task, output_dir, 2, 3)

    assert cmd[cmd.index("--output-dir") + 1] == str(output_dir)
    assert cmd[-4:] == ["--episode-start", "2", "--episode-count", "3"]


def test_merge_sim_worker_episode_metrics_preserves_episode_order(tmp_path):
    plan = []
    for worker_index, episode_ids in enumerate(([2, 4], [1, 3])):
        output_dir = tmp_path / f"worker_{worker_index}"
        output_dir.mkdir()
        with (output_dir / "episode_metrics.jsonl").open("w", encoding="utf-8") as f:
            for episode_id in episode_ids:
                f.write(json.dumps({"episode_id": episode_id, "metadata": {"source": worker_index}}) + "\n")
        plan.append({"worker_index": worker_index, "output_dir": str(output_dir)})

    sim_workers.merge_sim_worker_episode_metrics(tmp_path / "merged", plan)

    records = [
        json.loads(line)
        for line in ((tmp_path / "merged" / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines())
    ]
    assert [record["episode_id"] for record in records] == [1, 2, 3, 4]
    assert [record["metadata"]["sim_worker_index"] for record in records] == [1, 0, 1, 0]
    assert all(record["metadata"]["output_dir"] == str(tmp_path / "merged") for record in records)
