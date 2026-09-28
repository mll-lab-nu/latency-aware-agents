import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.starvla import backfill_wandb_eval as backfill
from scripts.starvla import train_task


def _summary():
    returns = [float(index) for index in range(50)]
    lengths = [10] * 50
    return {
        "latency": 0,
        "latency_type": "zero",
        "suite_name": "fixed_0",
        "mean_return": 24.5,
        "std_return": 14.43,
        "mean_length": 10.0,
        "returns": returns,
        "lengths": lengths,
        "task_metrics": {"success": {"mean": 0.0, "std": 0.0, "values": [0.0] * 50}},
    }


def _fake_evaluation(work_dir: Path) -> Path:
    evaluation = work_dir / "evals/final"
    evaluation.mkdir(parents=True)
    (evaluation / "eval_config.yaml").write_text("evaluation:\n  eval_episodes: 50\n")
    (evaluation / "stdout.log").write_text("ok\n")
    (evaluation / "episode_metrics.jsonl").write_text("{}\n")
    (evaluation / "queue_eval_latency_0.json").write_text(json.dumps(_summary()))
    return evaluation


def test_download_model_uses_profile_artifact_and_work_local_hf_caches(tmp_path, monkeypatch):
    captured = {}
    job = {"task": "mikasa", "workload": "mikasa_intercept_grab_fast", "model": "qwenoft"}
    profile = {
        "hf_repo": "latency-sensitive-bench/profiles",
        "model_artifact": {
            "hf_repo": backfill.HF_REPO,
            "hf_path": "zero-latency/task/vla/model",
        },
    }
    monkeypatch.setattr(backfill, "load_profile_spec", lambda *_args: profile)

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        captured["hub_cache_env"] = kwargs["env"]["HF_HUB_CACHE"]
        captured["xet_cache_env"] = kwargs["env"]["HF_XET_CACHE"]

    monkeypatch.setattr(backfill.subprocess, "run", fake_run)
    before_hub = backfill.os.environ.get("HF_HUB_CACHE")
    before_xet = backfill.os.environ.get("HF_XET_CACHE")
    bundle = backfill.download_model(job, tmp_path / "work")

    assert captured["command"] == [
        "hf", "download", backfill.HF_REPO, "--repo-type", "model",
        "--revision", backfill.HF_REVISION, "--include", "zero-latency/task/vla/model/**",
        "--local-dir", str(tmp_path / "work/download"),
    ]
    assert captured["check"] is True
    assert captured["hub_cache_env"] == str(tmp_path / "work/.hf-cache/hub")
    assert captured["xet_cache_env"] == str(tmp_path / "work/.hf-cache/xet")
    assert bundle == tmp_path / "work/download/zero-latency/task/vla/model"
    assert backfill.os.environ.get("HF_HUB_CACHE") == before_hub
    assert backfill.os.environ.get("HF_XET_CACHE") == before_xet


@pytest.mark.parametrize("task", ["mikasa", "balance"])
def test_train_task_eval_config_accepts_fifty_episode_override(tmp_path, monkeypatch, task):
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("WANDB_RUN_GROUP", "test-group")
    args = backfill._eval_args(
        {"task": task, "model": "qwenoft"}, tmp_path,
        task_python=tmp_path / "task-python", vla_python=tmp_path / "vla-python",
        inference_device="cuda:1", simulator_device="cuda:0",
    )
    assert args.eval_seed == train_task.TASK_SETTINGS[task]["seed"]
    args.eval_seed = 1234
    config = train_task.eval_config(
        args, tmp_path / "bundle", tmp_path / "eval",
        eval_episodes=50,
    )
    assert config["evaluation"]["eval_episodes"] == 50
    assert config["experiment"]["seed"] == 1234


def test_job_table_contains_the_eight_fixed_backfill_targets():
    assert len(backfill.BACKFILL_JOBS) == 8
    assert [(job["task"], job["model"], job["step"]) for job in backfill.BACKFILL_JOBS] == [
        ("mikasa", "qwengr00t", 5000),
        ("mikasa", "qwenoft", 5000),
        ("mikasa", "qwenpi_v3", 5000),
        ("mikasa", "wanoft", 5000),
        ("balance", "qwengr00t", 15000),
        ("balance", "qwenoft", 15000),
        ("balance", "qwenpi_v3", 15000),
        ("balance", "wanoft", 15000),
    ]
    assert all(job["wandb_id"] for job in backfill.BACKFILL_JOBS)


def test_run_all_processes_in_order_and_cleans_each_work_dir(tmp_path, monkeypatch):
    jobs = backfill.BACKFILL_JOBS[:2]
    events = []
    logged = {}

    class Run:
        def __init__(self):
            self.summary = {}

        def log(self, payload):
            events.append(("log", payload["global_step"]))
            logged[self.run_id] = payload

        def finish(self):
            events.append(("finish", self.run_id))

    class Api:
        def run(self, path):
            events.append(("confirm", path))
            run_id = path.rsplit("/", 1)[-1]
            return SimpleNamespace(summary=logged[run_id])

    class Wandb:
        @staticmethod
        def define_metric(*args, **kwargs):
            events.append(("define_metric", args, kwargs))

    Wandb.Api = Api

    def init(**kwargs):
        events.append(("init", kwargs))
        run = Run()
        run.run_id = kwargs["id"]
        return run

    Wandb.init = staticmethod(init)

    def fake_download(job, work_dir):
        events.append(("download", job["model"]))
        bundle = work_dir / "download/bundle"
        bundle.mkdir(parents=True)
        return bundle

    def fake_evaluate(job, bundle, work_dir, **kwargs):
        events.append(("evaluate", job["model"], kwargs["task_python"], kwargs["vla_python"]))
        return _fake_evaluation(work_dir)

    monkeypatch.setattr(backfill, "BACKFILL_JOBS", jobs)
    monkeypatch.setattr(backfill, "download_model", fake_download)
    monkeypatch.setattr(backfill, "evaluate_model", fake_evaluate)
    monkeypatch.setattr(backfill, "wandb", Wandb)
    backfill.run_all(
        work_root=tmp_path / "work", result_root=tmp_path / "results",
        task_python=tmp_path / "task-python", vla_python=tmp_path / "vla-python",
    )

    assert [event[1] for event in events if event[0] == "download"] == ["qwengr00t", "qwenoft"]
    assert [event[1] for event in events if event[0] == "evaluate"] == ["qwengr00t", "qwenoft"]
    assert not (tmp_path / "work/mikasa/qwengr00t").exists()
    assert not (tmp_path / "work/mikasa/qwenoft").exists()
    for job in jobs:
        result = tmp_path / "results/mikasa" / job["model"]
        assert (result / "queue_eval_latency_0.json").is_file()
        assert json.loads((result / "DONE.json").read_text())["episodes"] == 50
    init_kwargs = [event[1] for event in events if event[0] == "init"]
    assert all(kwargs["resume"] == "must" and "name" not in kwargs for kwargs in init_kwargs)
    assert [event[0] for event in events if event[0] in ("download", "evaluate", "confirm")] == [
        "download", "evaluate", "confirm", "download", "evaluate", "confirm"
    ]


def test_run_all_stops_and_keeps_current_work_on_failure(tmp_path, monkeypatch):
    jobs = backfill.BACKFILL_JOBS[:3]
    calls = []

    def fake_download(job, work_dir):
        calls.append(("download", job["model"]))
        bundle = work_dir / "download/bundle"
        bundle.mkdir(parents=True)
        return bundle

    def fake_evaluate(job, bundle, work_dir, **kwargs):
        calls.append(("evaluate", job["model"]))
        if job["model"] == "qwenoft":
            raise RuntimeError("evaluation failed")
        return _fake_evaluation(work_dir)

    class Run:
        def __init__(self):
            self.summary = {}

        def log(self, payload):
            self.payload = payload

        def finish(self):
            pass

    class Api:
        def run(self, path):
            return SimpleNamespace(summary={
                "eval_corrected/fixed/latency_0/completed_episodes": 50,
            })

    class Wandb:
        @staticmethod
        def define_metric(*args, **kwargs):
            pass

    Wandb.Api = Api
    Wandb.init = staticmethod(lambda **kwargs: Run())

    monkeypatch.setattr(backfill, "BACKFILL_JOBS", jobs)
    monkeypatch.setattr(backfill, "download_model", fake_download)
    monkeypatch.setattr(backfill, "evaluate_model", fake_evaluate)
    monkeypatch.setattr(backfill, "wandb", Wandb)

    with pytest.raises(RuntimeError, match="evaluation failed"):
        backfill.run_all(
            work_root=tmp_path / "work", result_root=tmp_path / "results",
            task_python=tmp_path / "task-python", vla_python=tmp_path / "vla-python",
        )

    assert calls == [
        ("download", "qwengr00t"), ("evaluate", "qwengr00t"),
        ("download", "qwenoft"), ("evaluate", "qwenoft"),
    ]
    assert (tmp_path / "results/mikasa/qwengr00t/DONE.json").is_file()
    assert (tmp_path / "work/mikasa/qwenoft").is_dir()
    assert not (tmp_path / "work/mikasa/qwengr00t").exists()
    assert not (tmp_path / "results/mikasa/qwenpi_v3").exists()
