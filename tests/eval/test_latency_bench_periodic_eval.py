from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from helpers import REPO_ROOT
from latency_bench.policy.model_source import model_file




def _assert_train_eval_artifacts_disabled(config: dict) -> None:
    # Train-time eval should not persist per-step/action/latency trajectory artifacts.
    assert config["logging"]["save_step_records"] is False
    assert config["logging"]["save_action_records"] is False
    assert config["logging"]["save_latency_records"] is False


def _training_config(tmp_path: Path) -> dict:
    return {
        "executor": {"mode": "simulated"},
        "policy": {"type": "sample_factory", "deterministic": True},
        "experiment": {
            "name": "flappy_sf_l0_seed0",
            "seed": 0,
        },
        "backend": {
            "type": "sample_factory",
            "algo": "APPO",
            "device": "gpu",
            "train_dir": str(tmp_path / "checkpoints"),
            "run_mode": "train",
        },
        "env": {
            "name": "flappy",
            "env_fps": 30,
            "obs_fps": 30,
            "frame_stack": 4,
            "obs_resize": [84, 84],
            "use_gpu_render": True,
        },
        "latency": {
            "method": "zero",
            "add_latency_info": False,
        },
        "scheduler": {
            "hold_policy": "one_frame_then_noop",
            "ordering_policy": "latest_ready",
        },
        "training": {
            "train_for_env_steps": 10_000,
        },
        "evaluation": {
            "eval_interval_steps": 1_000,
            "episodes": 3,
            "suites": [{'name': 'fixed_0', 'latency': {'method': 'zero'}}, {'name': 'fixed_1', 'latency': {'method': 'fixed', 'fixed_latency_ms': 33.333333333333336}}],
        },
        "logging": {
            "output_dir": str(tmp_path / "train_outputs"),
            "save_step_records": True,
            "save_action_records": True,
            "save_latency_records": True,
        },
    }


def test_build_latency_bench_config_from_training_config_uses_base_env_config(tmp_path: Path):
    from training.common.latency_bench_periodic_eval import (
        build_latency_bench_config_from_training_config,
    )

    checkpoint_path = (
        tmp_path
        / "checkpoints"
        / "flappy_sf_l0_seed0"
        / "checkpoint_p0"
        / "checkpoint_000000001_1000.pth"
    )

    training_config = _training_config(tmp_path)
    training_config["executor"] = {
        "mode": "simulated",
        "simulated_worker_capacity": 1,
    }
    config = build_latency_bench_config_from_training_config(
        training_config,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "eval_outputs",
    )

    assert config["executor"]["mode"] == "simulated"
    assert config["executor"]["simulated_worker_capacity"] == 1
    assert config["backend"]["run_mode"] == "eval"
    assert config["backend"]["train_dir"] == str(tmp_path / "checkpoints")
    assert config["policy"]["type"] == "sample_factory"
    assert model_file(config["policy"], "checkpoint") == str(checkpoint_path)
    assert config["policy"]["deterministic"] is True
    assert config["experiment"]["name"] == "flappy_sf_l0_seed0"
    assert config["experiment"]["seed"] == 0
    assert config["evaluation"]["episodes"] == 3
    assert config["evaluation"]["suites"] == training_config["evaluation"]["suites"]
    assert config["env"]["action_map"] == {"noop": 0, "flap": 1}
    assert config["evaluation"]["eval_raw_reward"] is False
    assert config["logging"]["output_dir"] == str(tmp_path / "eval_outputs")


def test_periodic_observer_saves_checkpoint_and_defers_latency_bench_job(monkeypatch, tmp_path: Path):
    import training.common.latency_bench_periodic_eval as periodic_eval

    checkpoint_dir = tmp_path / "checkpoints" / "flappy_sf_l0_seed0" / "checkpoint_p0"
    checkpoint_dir.mkdir(parents=True)
    calls = []

    class SavePeriodic:
        def emit(self):
            path = checkpoint_dir / "checkpoint_000000002_000000001000.pth"
            path.write_text("checkpoint\n", encoding="utf-8")

    class Writer:
        def __init__(self):
            self.scalars = []

        def add_scalar(self, tag, scalar, step):
            self.scalars.append((tag, scalar, step))

        def flush(self):
            pass

    writer = Writer()
    runner = SimpleNamespace(
        env_steps={0: 1_000},
        save_periodic=SavePeriodic(),
        writers={0: writer},
    )

    def fake_run_eval_suites(config, progress=False, extra_metadata=None):
        raise AssertionError("periodic latency_bench eval should run outside the training event loop")

    class Process:
        exitcode = None

        def join(self):
            self.exitcode = 0

    def fake_launch(config, metadata, result_path):
        calls.append((config, metadata, result_path))
        return Process()

    monkeypatch.setattr(periodic_eval, "run_eval_suites_from_config", fake_run_eval_suites)
    monkeypatch.setattr(periodic_eval, "_launch_latency_bench_eval_process", fake_launch)

    observer = periodic_eval.LatencyBenchPeriodicEvalObserver(_training_config(tmp_path))
    observer.on_training_step(runner, training_iteration_since_resume=1)

    assert calls == []
    assert observer.active_job is None
    assert len(observer.pending_jobs) == 1
    request = observer.pending_jobs[0]
    config = request.config
    metadata = request.metadata
    result_path = request.result_path
    assert model_file(config["policy"], "checkpoint").endswith("checkpoint_000000002_000000001000.pth")
    assert config["logging"]["output_dir"] == str(
        tmp_path / "train_outputs" / "latency_bench_train_eval" / "step_1000"
    )
    assert metadata == {
        "eval_phase": "train_periodic",
        "train_env_steps": 1000,
        "checkpoint_path": model_file(config["policy"], "checkpoint"),
    }
    assert result_path == tmp_path / "train_outputs" / "latency_bench_train_eval" / "step_1000" / "eval_result.json"
    assert writer.scalars == []


def test_checkpoint_only_observer_keeps_milestone_without_running_eval(tmp_path: Path):
    from training.common.checkpoints import prune_unleased_sample_factory_checkpoints
    from training.common.latency_bench_periodic_eval import LatencyBenchPeriodicEvalObserver, periodic_eval_enabled

    config = _training_config(tmp_path)
    del config["evaluation"]["eval_interval_steps"]
    config["evaluation"]["checkpoint_interval_steps"] = 1_000
    checkpoint_dir = tmp_path / "checkpoints" / "flappy_sf_l0_seed0" / "checkpoint_p0"
    checkpoint_dir.mkdir(parents=True)
    milestone = checkpoint_dir / "checkpoint_000000002_000000001000.pth"

    class SavePeriodic:
        def emit(self):
            milestone.write_text("checkpoint\n", encoding="utf-8")

    runner = SimpleNamespace(env_steps={0: 1_000}, save_periodic=SavePeriodic())
    assert periodic_eval_enabled(config)
    observer = LatencyBenchPeriodicEvalObserver(config)
    observer.on_training_step(runner, training_iteration_since_resume=1)
    assert list(observer.pending_jobs) == []
    newer = checkpoint_dir / "checkpoint_000000003_000000002000.pth"
    newer.write_text("newer\n", encoding="utf-8")
    prune_unleased_sample_factory_checkpoints(checkpoint_dir, "checkpoint", keep_checkpoints=1, verbose=False)
    assert milestone.exists()


def test_fasttd3_periodic_eval_uses_optimizer_step_and_keeps_raw_frames(monkeypatch, tmp_path: Path):
    import training.common.latency_bench_periodic_eval as periodic_eval

    checkpoint = tmp_path / "checkpoint_000000100_500.pth"
    checkpoint.write_text("checkpoint\n", encoding="utf-8")
    config = _training_config(tmp_path)
    config["backend"]["algo"] = "FAST_TD3"
    config["evaluation"]["eval_interval_steps"] = 100
    runner = SimpleNamespace(env_steps={0: 500}, train_steps={0: 100})
    observer = periodic_eval.LatencyBenchPeriodicEvalObserver(config)
    monkeypatch.setattr(observer, "_save_checkpoint_for_eval", lambda runner: checkpoint)

    observer.on_training_step(runner, training_iteration_since_resume=1)

    request = observer.pending_jobs[0]
    assert request.step == 100
    assert request.raw_frames == 500
    assert request.metadata == {
        "eval_phase": "train_periodic",
        "train_env_steps": 500,
        "checkpoint_path": str(checkpoint),
        "train_optimizer_steps": 100,
    }
    assert request.result_path.parent.name == "step_100"


def test_periodic_observer_runs_deferred_eval_fifo_after_training(monkeypatch, tmp_path: Path):
    import training.common.latency_bench_periodic_eval as periodic_eval

    checkpoint_dir = tmp_path / "checkpoints" / "flappy_sf_l0_seed0" / "checkpoint_p0"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_index = 0
    launches = []

    class SavePeriodic:
        def emit(self):
            nonlocal checkpoint_index
            checkpoint_index += 1
            path = checkpoint_dir / f"checkpoint_000000002_{checkpoint_index * 1000}.pth"
            path.write_text("checkpoint\n", encoding="utf-8")

    class Writer:
        def add_scalar(self, tag, scalar, step):
            raise AssertionError("latency_bench eval should not create per-latency TensorBoard scalars")

        def flush(self):
            pass

    class Process:
        def __init__(self, result_path):
            self.exitcode = None
            self.result_path = result_path

        def finish(self, summaries):
            self.result_path.parent.mkdir(parents=True, exist_ok=True)
            self.result_path.write_text(json.dumps({"summaries": summaries}), encoding="utf-8")
            self.exitcode = 0

        def join(self):
            self.finish(
                [
                    {
                        "suite_name": "fixed_0_to_1",
                        "latency_type": "fixed_sweep",
                        "latency": 0,
                        "mean_return": 12.0,
                        "std_return": 1.0,
                        "mean_length": 30.0,
                        "returns": [11.0, 12.0, 13.0],
                    }
                ]
            )

    def fake_launch(config, metadata, result_path):
        process = Process(result_path)
        launches.append((config, metadata, result_path, process))
        return process

    monkeypatch.setattr(periodic_eval, "_launch_latency_bench_eval_process", fake_launch)
    runner = SimpleNamespace(
        env_steps={0: 1_000},
        save_periodic=SavePeriodic(),
        writers={0: Writer()},
    )

    observer = periodic_eval.LatencyBenchPeriodicEvalObserver(_training_config(tmp_path))
    observer.on_training_step(runner, training_iteration_since_resume=1)

    runner.env_steps = {0: 2_000}
    observer.on_training_step(runner, training_iteration_since_resume=2)

    assert launches == []
    assert len(observer.pending_jobs) == 2

    observer.run_deferred_eval(runner)

    assert [launch[1]["train_env_steps"] for launch in launches] == [1_000, 2_000]
    assert observer.active_job is None
    assert len(observer.pending_jobs) == 0
    assert observer.last_eval_step == 2_000


def test_checkpoint_retention_skips_leased_checkpoint(tmp_path: Path):
    from training.common.checkpoints import (
        acquire_checkpoint_eval_lease,
        release_checkpoint_eval_lease,
        prune_unleased_sample_factory_checkpoints,
    )

    checkpoint_dir = tmp_path / "checkpoint_p0"
    checkpoint_dir.mkdir()
    first = checkpoint_dir / "checkpoint_000000001_1000.pth"
    second = checkpoint_dir / "checkpoint_000000002_2000.pth"
    third = checkpoint_dir / "checkpoint_000000003_3000.pth"
    for path in (first, second, third):
        path.write_text(path.name, encoding="utf-8")

    lease = acquire_checkpoint_eval_lease(
        first,
        metadata={"eval_phase": "train_periodic", "train_env_steps": 1000},
    )
    prune_unleased_sample_factory_checkpoints(
        checkpoint_dir,
        "checkpoint",
        keep_checkpoints=1,
        verbose=False,
    )

    assert first.exists()
    assert not second.exists()
    assert third.exists()

    release_checkpoint_eval_lease(lease)
    prune_unleased_sample_factory_checkpoints(
        checkpoint_dir,
        "checkpoint",
        keep_checkpoints=1,
        verbose=False,
    )

    assert not first.exists()
    assert third.exists()


def test_checkpoint_retention_keeps_deferred_eval_checkpoint_after_eval_finishes(tmp_path: Path):
    from training.common.checkpoints import (
        mark_checkpoint_for_eval_retention,
        prune_unleased_sample_factory_checkpoints,
    )

    checkpoint_dir = tmp_path / "checkpoint_p0"
    checkpoint_dir.mkdir()
    first = checkpoint_dir / "checkpoint_000000001_1000.pth"
    second = checkpoint_dir / "checkpoint_000000002_2000.pth"
    third = checkpoint_dir / "checkpoint_000000003_3000.pth"
    for path in (first, second, third):
        path.write_text(path.name, encoding="utf-8")

    mark_checkpoint_for_eval_retention(
        first,
        metadata={"eval_phase": "train_periodic", "train_env_steps": 1000},
    )
    prune_unleased_sample_factory_checkpoints(
        checkpoint_dir,
        "checkpoint",
        keep_checkpoints=1,
        verbose=False,
    )

    assert first.exists()
    assert not second.exists()
    assert third.exists()


def test_deferred_observer_releases_checkpoint_lease_after_eval_success(monkeypatch, tmp_path: Path):
    import training.common.latency_bench_periodic_eval as periodic_eval

    checkpoint_dir = tmp_path / "checkpoints" / "flappy_sf_l0_seed0" / "checkpoint_p0"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "checkpoint_000000002_000000001000.pth"
    launches = []

    class SavePeriodic:
        def emit(self):
            checkpoint_path.write_text("checkpoint\n", encoding="utf-8")

    class Writer:
        def add_scalar(self, tag, scalar, step):
            raise AssertionError("latency_bench eval should write W&B scalars directly")

        def flush(self):
            pass

    class Process:
        def __init__(self, result_path):
            self.exitcode = None
            self.result_path = result_path

        def finish(self):
            self.result_path.parent.mkdir(parents=True, exist_ok=True)
            self.result_path.write_text(
                json.dumps(
                    {
                        "summaries": [
                            {
                                "suite_name": "fixed_0_to_1",
                                "latency_type": "fixed_sweep",
                                "latency": 0,
                                "mean_return": 12.0,
                                "std_return": 1.0,
                                "mean_length": 30.0,
                                "returns": [12.0],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.exitcode = 0

        def join(self):
            self.finish()

    def fake_launch(config, metadata, result_path):
        process = Process(result_path)
        launches.append((config, metadata, result_path, process))
        return process

    monkeypatch.setattr(periodic_eval, "_launch_latency_bench_eval_process", fake_launch)
    runner = SimpleNamespace(
        env_steps={0: 1_000},
        save_periodic=SavePeriodic(),
        writers={0: Writer()},
    )

    observer = periodic_eval.LatencyBenchPeriodicEvalObserver(_training_config(tmp_path))
    observer.on_training_step(runner, training_iteration_since_resume=1)

    lease_dir = checkpoint_dir / ".latency_bench_eval_leases"
    retention_dir = checkpoint_dir / ".latency_bench_eval_checkpoints"
    assert list(lease_dir.glob("checkpoint_000000002_000000001000.pth.*.json")) == []
    assert len(list(retention_dir.glob("checkpoint_000000002_000000001000.pth.json"))) == 1

    assert observer.run_deferred_eval(runner) is True

    assert list(lease_dir.glob("checkpoint_000000002_000000001000.pth.*.json")) == []


def test_deferred_observer_reports_failure_without_stopping_training(monkeypatch, tmp_path: Path):
    import training.common.latency_bench_periodic_eval as periodic_eval

    checkpoint_dir = tmp_path / "checkpoints" / "flappy_sf_l0_seed0" / "checkpoint_p0"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "checkpoint_000000002_000000001000.pth"
    launches = []

    class SavePeriodic:
        def emit(self):
            checkpoint_path.write_text("checkpoint\n", encoding="utf-8")

    class Writer:
        def add_scalar(self, tag, scalar, step):
            pass

        def flush(self):
            pass

    class Process:
        exitcode = None

        def join(self):
            self.exitcode = 1

    def fake_launch(config, metadata, result_path):
        process = Process()
        launches.append((config, metadata, result_path, process))
        return process

    class Runner:
        def __init__(self):
            self.env_steps = {0: 1_000}
            self.save_periodic = SavePeriodic()
            self.writers = {0: Writer()}
            self.stop_calls = []

        def _stop_training(self, failed=False):
            self.stop_calls.append(failed)

    monkeypatch.setattr(periodic_eval, "_launch_latency_bench_eval_process", fake_launch)
    runner = Runner()
    observer = periodic_eval.LatencyBenchPeriodicEvalObserver(_training_config(tmp_path))
    observer.on_training_step(runner, training_iteration_since_resume=1)

    lease_dir = checkpoint_dir / ".latency_bench_eval_leases"
    assert list(lease_dir.glob("checkpoint_000000002_000000001000.pth.*.json")) == []

    assert observer.run_deferred_eval(runner) is False

    assert runner.stop_calls == []
    assert observer.active_job is None
    assert len(observer.pending_jobs) == 0
    assert list(lease_dir.glob("checkpoint_000000002_000000001000.pth.*.json")) == []


def test_log_eval_summaries_writes_plain_wandb_scalars(monkeypatch):
    import training.common.latency_bench_periodic_eval as periodic_eval

    log_calls = []

    fake_wandb = SimpleNamespace(
        run=object(),
        log=lambda payload, step=None: log_calls.append((payload, step)),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    class Writer:
        def add_scalar(self, tag, scalar, step):
            raise AssertionError("latency_bench eval should write W&B scalars directly")

        def flush(self):
            pass

    runner = SimpleNamespace(writers={0: Writer()})

    periodic_eval._log_eval_summaries(
        runner,
        [
            {
                "suite_name": "fixed_0_to_4",
                "latency_type": "fixed_sweep",
                "latency": 0,
                "mean_return": 10.0,
                "std_return": 1.0,
                "mean_length": 30.0,
                "returns": [9.0, 10.0, 11.0],
            },
            {
                "suite_name": "fixed_0_to_4",
                "latency_type": "fixed_sweep",
                "latency": 1,
                "mean_return": 11.0,
                "std_return": 1.1,
                "mean_length": 31.0,
                "returns": [10.0, 12.0],
            },
            {
                "suite_name": "normal_mean_2_std_1_min_0_max_15",
                "latency_type": "normal_distribution",
                "latency": "normal_mean_2_std_1_min_0_max_15",
                "mean_return": 12.0,
                "std_return": 1.2,
                "mean_length": 32.0,
                "returns": [10.0, 11.0, 12.0, 13.0, 14.0],
            },
        ],
        step=1_000,
        raw_frames=5_000,
        uses_optimizer_steps=True,
    )

    assert log_calls[-1][1] is None
    payload = log_calls[-1][0]
    assert payload == {
        "global_step": 1_000,
        "eval_corrected/global_steps": 1_000,
        "eval_corrected/raw_frames": 5_000,
        "eval_corrected/fixed/latency_0/mean_return": 10.0,
        "eval_corrected/fixed/latency_0/std_return": 1.0,
        "eval_corrected/fixed/latency_0/mean_length": 30.0,
        "eval_corrected/fixed/latency_0/completed_episodes": 3.0,
        "eval_corrected/fixed/latency_1/mean_return": 11.0,
        "eval_corrected/fixed/latency_1/std_return": 1.1,
        "eval_corrected/fixed/latency_1/mean_length": 31.0,
        "eval_corrected/fixed/latency_1/completed_episodes": 2.0,
        "eval_corrected/normal/normal_mean_2_std_1_min_0_max_15/mean_return": 12.0,
        "eval_corrected/normal/normal_mean_2_std_1_min_0_max_15/std_return": 1.2,
        "eval_corrected/normal/normal_mean_2_std_1_min_0_max_15/mean_length": 32.0,
        "eval_corrected/normal/normal_mean_2_std_1_min_0_max_15/completed_episodes": 5.0,
    }


def test_latency_bench_eval_process_main_writes_result_file(monkeypatch, tmp_path: Path):
    import training.common.latency_bench_periodic_eval as periodic_eval

    result_path = tmp_path / "job" / "eval_result.json"

    def fake_run_eval_suites(config, progress=False, extra_metadata=None):
        return [
            {
                "suite_name": config["suite_name"],
                "latency_type": extra_metadata["latency_type"],
                "latency": 0,
                "mean_return": 12.0,
                "std_return": 1.0,
                "mean_length": 30.0,
            }
        ]

    monkeypatch.setattr(periodic_eval, "run_eval_suites_from_config", fake_run_eval_suites)

    config = _training_config(tmp_path)
    config["suite_name"] = "fixed_0_to_1"
    periodic_eval._latency_bench_eval_process_main(
        config,
        {"latency_type": "fixed_sweep"},
        str(result_path),
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "summaries": [
            {
                "suite_name": "fixed_0_to_1",
                "latency_type": "fixed_sweep",
                "latency": 0,
                "mean_return": 12.0,
                "std_return": 1.0,
                "mean_length": 30.0,
            }
        ]
    }


def test_latency_bench_eval_process_main_runs_current_condition_without_suites(monkeypatch, tmp_path: Path):
    import training.common.latency_bench_periodic_eval as periodic_eval

    result_path = tmp_path / "job" / "eval_result.json"
    metrics = [object()]
    summary = {"latency": "temporal", "mean_length": 321.0}
    calls = []
    monkeypatch.setattr(
        periodic_eval,
        "run_from_config",
        lambda config, extra_metadata=None: calls.append((config, extra_metadata)) or metrics,
    )
    monkeypatch.setattr(
        periodic_eval,
        "_summary_payloads_for_printing",
        lambda config, values: [summary] if values is metrics else [],
    )

    config = _training_config(tmp_path)
    config["evaluation"] = {"episodes": 8}
    metadata = {"eval_phase": "train_periodic"}
    periodic_eval._latency_bench_eval_process_main(config, metadata, str(result_path))

    assert calls == [(config, metadata)]
    assert json.loads(result_path.read_text(encoding="utf-8")) == {"summaries": [summary]}


def test_periodic_observer_uses_updated_checkpoint_when_sample_factory_overwrites_same_name(tmp_path: Path):
    from training.common.latency_bench_periodic_eval import LatencyBenchPeriodicEvalObserver

    checkpoint_dir = tmp_path / "checkpoints" / "flappy_sf_l0_seed0" / "checkpoint_p0"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "checkpoint_000000002_1000.pth"
    checkpoint_path.write_text("old\n", encoding="utf-8")
    os.utime(checkpoint_path, (1.0, 1.0))

    class SavePeriodic:
        def emit(self):
            checkpoint_path.write_text("new\n", encoding="utf-8")
            os.utime(checkpoint_path, (10.0, 10.0))

    runner = SimpleNamespace(save_periodic=SavePeriodic())
    observer = LatencyBenchPeriodicEvalObserver(_training_config(tmp_path), checkpoint_timeout_sec=0.2)

    assert observer._save_checkpoint_for_eval(runner) == checkpoint_path


def test_run_sample_factory_training_registers_periodic_observer(monkeypatch, tmp_path: Path):
    import sample_factory.algo.utils.misc as sf_misc
    import sample_factory.train as sf_train
    import training.common.latency_bench_periodic_eval as periodic_eval

    from training.common.latency_bench_periodic_eval import (
        LatencyBenchPeriodicEvalObserver,
        run_sample_factory_training,
    )

    observers = []
    calls = []

    class Runner:
        def register_observer(self, observer):
            observers.append(observer)

        def init(self):
            return sf_misc.ExperimentStatus.SUCCESS

        def run(self):
            return sf_misc.ExperimentStatus.SUCCESS

    runner = Runner()
    monkeypatch.setattr(periodic_eval, "patch_sample_factory_checkpoint_loader", lambda: calls.append("patch"), raising=False)
    monkeypatch.setattr(sf_train, "make_runner", lambda cfg: calls.append("make_runner") or (cfg, runner))
    monkeypatch.setattr(LatencyBenchPeriodicEvalObserver, "run_final_eval", lambda self, runner: True)

    status = run_sample_factory_training(SimpleNamespace(name="cfg"), _training_config(tmp_path))

    assert status == sf_misc.ExperimentStatus.SUCCESS
    assert calls == ["patch", "make_runner"]
    assert len(observers) == 1
    assert isinstance(observers[0], LatencyBenchPeriodicEvalObserver)


def test_run_sample_factory_training_runs_final_latency_bench_after_runner_finishes(monkeypatch, tmp_path: Path):
    import sample_factory.algo.utils.misc as sf_misc
    import sample_factory.train as sf_train
    import training.common.latency_bench_periodic_eval as periodic_eval

    from training.common.latency_bench_periodic_eval import run_sample_factory_training

    checkpoint_dir = tmp_path / "checkpoints" / "flappy_sf_l0_seed0" / "checkpoint_p0"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "checkpoint_000000002_1234.pth"
    checkpoint_path.write_text("checkpoint\n", encoding="utf-8")
    calls = []

    class Writer:
        def add_scalar(self, tag, scalar, step):
            pass

        def flush(self):
            pass

    class SavePeriodic:
        def emit(self):
            raise AssertionError("final latency_bench eval should use Sample Factory's final checkpoint")

    class Runner:
        def __init__(self):
            self.env_steps = {0: 1_234}
            self.save_periodic = SavePeriodic()
            self.writers = {0: Writer()}

        def register_observer(self, observer):
            self.observer = observer

        def init(self):
            return sf_misc.ExperimentStatus.SUCCESS

        def run(self):
            return sf_misc.ExperimentStatus.SUCCESS

    runner = Runner()
    monkeypatch.setattr(sf_train, "make_runner", lambda cfg: (cfg, runner))

    class Process:
        exitcode = 0

        def join(self):
            pass

    def fake_launch(config, metadata, result_path):
        calls.append((config, metadata, result_path))
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "summaries": [
                        {
                            "suite_name": "fixed_0_to_1",
                            "latency_type": "fixed_sweep",
                            "latency": 0,
                            "mean_return": 12.0,
                            "std_return": 1.0,
                            "mean_length": 30.0,
                            "returns": [12.0],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return Process()

    monkeypatch.setattr(periodic_eval, "_launch_latency_bench_eval_process", fake_launch)

    status = run_sample_factory_training(SimpleNamespace(name="cfg"), _training_config(tmp_path))

    assert status == sf_misc.ExperimentStatus.SUCCESS
    assert len(calls) == 1
    config, metadata, result_path = calls[0]
    assert model_file(config["policy"], "checkpoint") == str(checkpoint_path)
    assert config["logging"]["output_dir"] == str(
        tmp_path / "train_outputs" / "latency_bench_train_eval" / "step_1234_final"
    )
    _assert_train_eval_artifacts_disabled(config)
    assert result_path == tmp_path / "train_outputs" / "latency_bench_train_eval" / "step_1234_final" / "eval_result.json"
    assert metadata == {
        "eval_phase": "train_final",
        "train_env_steps": 1234,
        "checkpoint_path": str(checkpoint_path),
    }


def test_run_sample_factory_training_returns_failure_when_deferred_eval_fails(monkeypatch, tmp_path: Path):
    import sample_factory.algo.utils.misc as sf_misc
    import sample_factory.train as sf_train
    import training.common.latency_bench_periodic_eval as periodic_eval

    from training.common.latency_bench_periodic_eval import run_sample_factory_training

    checkpoint_dir = tmp_path / "checkpoints" / "flappy_sf_l0_seed0" / "checkpoint_p0"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "checkpoint_000000002_1000.pth"

    class Writer:
        def add_scalar(self, tag, scalar, step):
            pass

        def flush(self):
            pass

    class SavePeriodic:
        def emit(self):
            checkpoint_path.write_text("checkpoint\n", encoding="utf-8")

    class Runner:
        def __init__(self):
            self.env_steps = {0: 1_000}
            self.save_periodic = SavePeriodic()
            self.writers = {0: Writer()}

        def register_observer(self, observer):
            self.observer = observer

        def init(self):
            return sf_misc.ExperimentStatus.SUCCESS

        def run(self):
            self.observer.on_training_step(self, training_iteration_since_resume=1)
            return sf_misc.ExperimentStatus.SUCCESS

    class Process:
        exitcode = 1

        def join(self):
            pass

    runner = Runner()
    monkeypatch.setattr(sf_train, "make_runner", lambda cfg: (cfg, runner))
    monkeypatch.setattr(periodic_eval, "_launch_latency_bench_eval_process", lambda config, metadata, result_path: Process())

    status = run_sample_factory_training(SimpleNamespace(name="cfg"), _training_config(tmp_path))

    assert status == sf_misc.ExperimentStatus.FAILURE


def test_run_sample_factory_training_skips_final_eval_after_runner_failure(monkeypatch, tmp_path: Path):
    import sample_factory.algo.utils.misc as sf_misc
    import sample_factory.train as sf_train
    import training.common.latency_bench_periodic_eval as periodic_eval

    from training.common.latency_bench_periodic_eval import (
        LatencyBenchPeriodicEvalObserver,
        run_sample_factory_training,
    )

    class Runner:
        def register_observer(self, observer):
            self.observer = observer

        def init(self):
            return sf_misc.ExperimentStatus.SUCCESS

        def run(self):
            return sf_misc.ExperimentStatus.FAILURE

    runner = Runner()
    monkeypatch.setattr(sf_train, "make_runner", lambda cfg: (cfg, runner))
    monkeypatch.setattr(
        LatencyBenchPeriodicEvalObserver,
        "run_final_eval",
        lambda self, runner: (_ for _ in ()).throw(AssertionError("final eval should be skipped after failure")),
    )

    status = run_sample_factory_training(SimpleNamespace(name="cfg"), _training_config(tmp_path))

    assert status == sf_misc.ExperimentStatus.FAILURE


def test_run_final_eval_runs_deferred_periodic_fifo_before_starting_final(monkeypatch, tmp_path: Path):
    import training.common.latency_bench_periodic_eval as periodic_eval

    checkpoint_dir = tmp_path / "checkpoints" / "flappy_sf_l0_seed0" / "checkpoint_p0"
    checkpoint_dir.mkdir(parents=True)
    periodic_checkpoint = checkpoint_dir / "checkpoint_000000002_1000.pth"
    final_checkpoint = checkpoint_dir / "checkpoint_000000003_1234.pth"
    final_checkpoint.write_text("final\n", encoding="utf-8")
    launches = []

    class Writer:
        def add_scalar(self, tag, scalar, step):
            raise AssertionError("latency_bench eval should not create per-latency TensorBoard scalars")

        def flush(self):
            pass

    class SavePeriodic:
        count = 0

        def emit(self):
            self.count += 1
            path = periodic_checkpoint.with_name(f"checkpoint_000000002_{self.count * 1000}.pth")
            path.write_text("periodic\n", encoding="utf-8")

    class Process:
        def __init__(self, result_path):
            self.exitcode = None
            self.result_path = result_path

        def finish(self):
            self.result_path.parent.mkdir(parents=True, exist_ok=True)
            self.result_path.write_text(
                json.dumps(
                    {
                        "summaries": [
                            {
                                "suite_name": "fixed_0_to_1",
                                "latency_type": "fixed_sweep",
                                "latency": 0,
                                "mean_return": 12.0,
                                "std_return": 1.0,
                                "mean_length": 30.0,
                                "returns": [12.0],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.exitcode = 0

        def join(self):
            self.finish()

    def fake_launch(config, metadata, result_path):
        process = Process(result_path)
        launches.append((metadata["eval_phase"], config, metadata, result_path, process))
        if metadata["eval_phase"] == "train_final":
            process.finish()
        return process

    monkeypatch.setattr(periodic_eval, "_launch_latency_bench_eval_process", fake_launch)
    runner = SimpleNamespace(
        env_steps={0: 1_000},
        save_periodic=SavePeriodic(),
        writers={0: Writer()},
    )

    observer = periodic_eval.LatencyBenchPeriodicEvalObserver(_training_config(tmp_path))
    observer.on_training_step(runner, training_iteration_since_resume=1)
    runner.env_steps = {0: 2_000}
    observer.on_training_step(runner, training_iteration_since_resume=2)
    assert launches == []
    runner.env_steps = {0: 1_234}
    assert observer.run_final_eval(runner) is True

    assert [launch[0] for launch in launches] == ["train_periodic", "train_periodic", "train_final"]
    assert observer.active_job is None
    assert observer.last_eval_step == 1_234


def test_run_final_eval_does_not_repeat_terminal_periodic_step(tmp_path: Path):
    import training.common.latency_bench_periodic_eval as periodic_eval

    observer = periodic_eval.LatencyBenchPeriodicEvalObserver(_training_config(tmp_path))
    observer.last_eval_step = 1_000
    runner = SimpleNamespace(env_steps={0: 1_000})

    assert observer.run_final_eval(runner) is True
    assert len(observer.pending_jobs) == 0


def test_run_sample_factory_training_uses_native_run_rl_without_interval(monkeypatch, tmp_path: Path):
    import sample_factory.train as sf_train
    import training.common.latency_bench_periodic_eval as periodic_eval

    from training.common.latency_bench_periodic_eval import run_sample_factory_training

    config = _training_config(tmp_path)
    config["evaluation"].pop("eval_interval_steps")
    calls = []
    monkeypatch.setattr(periodic_eval, "patch_sample_factory_checkpoint_loader", lambda: calls.append("patch"), raising=False)
    monkeypatch.setattr(sf_train, "run_rl", lambda cfg: calls.append(("run_rl", cfg)) or 23)

    status = run_sample_factory_training(SimpleNamespace(name="cfg"), config)

    assert status == 23
    assert calls == ["patch", ("run_rl", SimpleNamespace(name="cfg"))]


def test_base_configs_include_periodic_latency_bench_eval_defaults():
    from latency_bench.core.config import load_config
    from latency_bench.eval.config import parse_eval_suites
    from training.common.latency_bench_periodic_eval import periodic_eval_enabled

    for env_name in ("flappy", "demon_attack", "deadly_corridor"):
        config = load_config(REPO_ROOT / f"configs/base/{env_name}.yaml")
        suites = parse_eval_suites(config)

        assert periodic_eval_enabled(config) is True
        assert "eval_interval_steps" in config["evaluation"]
        assert suites
