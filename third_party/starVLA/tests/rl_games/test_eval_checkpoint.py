import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


EVAL_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2] / "examples" / "rl_games" / "scripts" / "eval_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("eval_checkpoint_under_test", EVAL_CHECKPOINT_PATH)
assert SPEC is not None
eval_checkpoint = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(eval_checkpoint)

from starVLA.training.rl_games import build_rl_games_eval_runner


def _cfg(*, backend: str = "latency_bench") -> SimpleNamespace:
    return SimpleNamespace(
        rl_games=SimpleNamespace(
            env_eval=SimpleNamespace(eval_backend=backend),
        ),
        datasets=SimpleNamespace(
            vla_data=SimpleNamespace(image_mode="stitch", num_obs_frames=4),
        ),
        framework=SimpleNamespace(
            kv_memory=SimpleNamespace(enabled=False, packed_train=False),
            qwenvl=SimpleNamespace(attn_implementation="flash_attention_2"),
        ),
    )


def test_runner_factory_uses_latency_bench_backend(monkeypatch) -> None:
    class FakeLatencyBenchRunner:
        def __init__(self, *, cfg, output_dir):
            self.cfg = cfg
            self.output_dir = output_dir

    integration_module = types.ModuleType(
        "latency_bench.integrations.starvla_rl_games_eval_runner"
    )
    integration_module.LatencyBenchRlGamesEvalRunner = FakeLatencyBenchRunner
    monkeypatch.setitem(
        sys.modules,
        "latency_bench.integrations.starvla_rl_games_eval_runner",
        integration_module,
    )

    cfg = _cfg()
    runner = build_rl_games_eval_runner(cfg, "/tmp/eval")

    assert isinstance(runner, FakeLatencyBenchRunner)
    assert runner.cfg is cfg
    assert runner.output_dir == "/tmp/eval"


def test_runner_factory_uses_eval_core_only_when_explicit(monkeypatch) -> None:
    class FakeEvalCoreRunner:
        def __init__(self, *, cfg, output_dir):
            self.cfg = cfg
            self.output_dir = output_dir

    import starVLA.training.rl_games as rl_games

    monkeypatch.setattr(rl_games, "RlGamesEvalRunner", FakeEvalCoreRunner, raising=False)

    cfg = _cfg(backend="eval_core")
    runner = build_rl_games_eval_runner(cfg, "/tmp/eval")

    assert isinstance(runner, FakeEvalCoreRunner)
    assert runner.cfg is cfg
    assert runner.output_dir == "/tmp/eval"


def test_runner_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown rl_games.env_eval.eval_backend: other"):
        build_rl_games_eval_runner(_cfg(backend="other"), "/tmp/eval")


def test_runner_factory_requires_backend() -> None:
    cfg = SimpleNamespace(rl_games=SimpleNamespace(env_eval=SimpleNamespace()))

    with pytest.raises(AttributeError):
        build_rl_games_eval_runner(cfg, "/tmp/eval")


def test_independent_wandb_run_records_eval_provenance(monkeypatch, tmp_path) -> None:
    init_calls = []

    class FakeRun:
        def __init__(self):
            self.logs = []
            self.finished = False

        def log(self, payload, **kwargs):
            self.logs.append((payload, kwargs))

        def finish(self):
            self.finished = True

    run = FakeRun()
    wandb_module = types.ModuleType("wandb")

    def init(**kwargs):
        init_calls.append(kwargs)
        return run

    wandb_module.init = init
    monkeypatch.setitem(sys.modules, "wandb", wandb_module)

    args = SimpleNamespace(
        wandb_project="starVLA_rl_games",
        wandb_entity="zihanwang-ai-northwestern-university",
        wandb_run_id=None,
        wandb_run_name="run__post_train__step_4000__latency_bench",
        stage="post_train",
    )
    result = SimpleNamespace(aggregate={}, per_latency={})

    eval_checkpoint._log_eval_to_wandb(
        args=args,
        cfg=_cfg(),
        eval_runner=SimpleNamespace(),
        result=result,
        run_dir=tmp_path,
        step=4000,
    )

    assert init_calls == [
        {
            "project": "starVLA_rl_games",
            "entity": "zihanwang-ai-northwestern-university",
            "dir": str(tmp_path / "wandb"),
            "config": {
                "eval_backend": "latency_bench",
                "eval_runner": "types.SimpleNamespace",
                "checkpoint_step": 4000,
                "image_mode": "stitch",
                "num_obs_frames": 4,
                "kv_memory_enabled": False,
                "attn_implementation": "flash_attention_2",
                "kv_memory_packed_train": False,
            },
            "name": "run__post_train__step_4000__latency_bench",
            "group": "rl-games-eval",
        }
    ]
    assert run.logs[0][1] == {"step": 4000}
    assert run.finished


def test_resumed_wandb_run_records_eval_provenance(monkeypatch, tmp_path) -> None:
    init_calls = []

    class FakeRun:
        def __init__(self):
            self.logs = []
            self.finished = False

        def log(self, payload, **kwargs):
            self.logs.append((payload, kwargs))

        def finish(self):
            self.finished = True

    run = FakeRun()
    wandb_module = types.ModuleType("wandb")

    def init(**kwargs):
        init_calls.append(kwargs)
        return run

    wandb_module.init = init
    monkeypatch.setitem(sys.modules, "wandb", wandb_module)

    args = SimpleNamespace(
        wandb_project="starVLA_rl_games",
        wandb_entity="zihanwang-ai-northwestern-university",
        wandb_run_id="existing-run",
        wandb_run_name=None,
        stage="post_train",
    )

    eval_checkpoint._log_eval_to_wandb(
        args=args,
        cfg=_cfg(),
        eval_runner=SimpleNamespace(),
        result=SimpleNamespace(aggregate={}, per_latency={}),
        run_dir=tmp_path,
        step=4000,
    )

    eval_checkpoint._log_eval_to_wandb(
        args=args,
        cfg=_cfg(backend="eval_core"),
        eval_runner=SimpleNamespace(),
        result=SimpleNamespace(aggregate={}, per_latency={}),
        run_dir=tmp_path,
        step=5000,
    )

    assert init_calls == [
        {
            "project": "starVLA_rl_games",
            "entity": "zihanwang-ai-northwestern-university",
            "dir": str(tmp_path / "wandb"),
            "id": "existing-run",
            "resume": "must",
        },
        {
            "project": "starVLA_rl_games",
            "entity": "zihanwang-ai-northwestern-university",
            "dir": str(tmp_path / "wandb"),
            "id": "existing-run",
            "resume": "must",
        },
    ]
    assert run.logs[0][0]["eval_provenance/eval_backend"] == "latency_bench"
    assert run.logs[0][0]["eval_provenance/checkpoint_step"] == 4000
    assert run.logs[1][0]["eval_provenance/eval_backend"] == "eval_core"
    assert run.logs[1][0]["eval_provenance/checkpoint_step"] == 5000
    assert run.finished
