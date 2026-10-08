from types import SimpleNamespace


def test_configure_torch_dynamo_recompile_limits_raises_default_limits(monkeypatch):
    from starVLA.training import train_starvla

    config = train_starvla.torch._dynamo.config
    monkeypatch.setattr(config, "recompile_limit", 8)
    monkeypatch.setattr(config, "accumulated_recompile_limit", 256)

    train_starvla._configure_torch_dynamo_recompile_limits()

    assert config.recompile_limit == 64
    assert config.accumulated_recompile_limit == 1024


def test_configure_torch_dynamo_recompile_limits_preserves_higher_limits(monkeypatch):
    from starVLA.training import train_starvla

    config = train_starvla.torch._dynamo.config
    monkeypatch.setattr(config, "recompile_limit", 128)
    monkeypatch.setattr(config, "accumulated_recompile_limit", 2048)

    train_starvla._configure_torch_dynamo_recompile_limits()

    assert config.recompile_limit == 128
    assert config.accumulated_recompile_limit == 2048


class _ModeTrackingModel:
    def __init__(self):
        self.training = True
        self.reset_memory_calls = []

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def reset_memory(self, slot_id=None):
        self.reset_memory_calls.append(slot_id)


class _Accelerator:
    is_main_process = True
    process_index = 0
    num_processes = 1

    def __init__(self):
        self.wait_calls = 0

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self):
        self.wait_calls += 1


class _LiveEvalRunner:
    def __init__(self, *, raises=False, record_dynamo_disabled=None):
        self.raises = raises
        self.record_dynamo_disabled = record_dynamo_disabled
        self.model_modes = []
        self.run_kwargs = []

    def run(self, *, model, step, stage, **kwargs):
        self.model_modes.append(model.training)
        self.run_kwargs.append(kwargs)
        if self.record_dynamo_disabled is not None:
            self.record_dynamo_disabled()
        if self.raises:
            raise RuntimeError("eval failed")
        from starVLA.training.rl_games.eval_core import EvalResult

        return EvalResult(per_latency={}, aggregate={"mean_reward": 3.0}, path=f"/tmp/{stage}-{step}.json")

    def merge_results(self, partial_results, *, step, stage):
        from starVLA.training.rl_games.eval_core import EvalResult

        return EvalResult(per_latency={}, aggregate={"mean_reward": 4.0}, path=f"/tmp/merged-{stage}-{step}.json")

    def save(self, *, result, step, stage):
        self.saved = (result, step, stage)


def _trainer(runner):
    from starVLA.training.train_starvla import VLATrainer

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.model = _ModeTrackingModel()
    trainer.accelerator = _Accelerator()
    trainer.completed_steps = 250
    trainer._rl_games_eval_runner = runner
    trainer.config = SimpleNamespace(
        rl_games=SimpleNamespace(
            env_eval=SimpleNamespace(distributed_mode="none"),
        ),
    )
    return trainer


def test_live_rl_games_eval_runs_with_model_in_eval_mode_and_restores_train_mode(monkeypatch):
    from starVLA.training import train_starvla

    empty_cache_calls = []
    monkeypatch.setattr(train_starvla.torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))
    runner = _LiveEvalRunner()
    trainer = _trainer(runner)

    result = trainer._run_rl_games_eval_with_model_mode(stage="mid_train")

    assert runner.model_modes == [False]
    assert result.aggregate["mean_reward"] == 3.0
    assert trainer.model.reset_memory_calls == [None]
    assert empty_cache_calls == []
    assert trainer.model.training is True


def test_live_rl_games_eval_restores_train_mode_when_runner_raises(monkeypatch):
    from starVLA.training import train_starvla

    empty_cache_calls = []
    monkeypatch.setattr(train_starvla.torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))
    runner = _LiveEvalRunner(raises=True)
    trainer = _trainer(runner)

    try:
        trainer._run_rl_games_eval_with_model_mode(stage="mid_train")
    except RuntimeError as exc:
        assert str(exc) == "eval failed"

    assert runner.model_modes == [False]
    assert trainer.model.reset_memory_calls == [None]
    assert empty_cache_calls == []
    assert trainer.model.training is True


def test_live_rl_games_eval_runs_with_torch_dynamo_disabled(monkeypatch):
    from starVLA.training import train_starvla

    state = {"depth": 0, "observed": []}

    def disable(fn):
        def wrapped(*args, **kwargs):
            state["depth"] += 1
            try:
                return fn(*args, **kwargs)
            finally:
                state["depth"] -= 1

        return wrapped

    monkeypatch.setattr(train_starvla.torch._dynamo, "disable", disable)

    runner = _LiveEvalRunner(record_dynamo_disabled=lambda: state["observed"].append(state["depth"] > 0))
    trainer = _trainer(runner)

    trainer._run_rl_games_eval_with_model_mode(stage="mid_train")

    assert state["observed"] == [True]


def test_distributed_live_rl_games_eval_passes_rank_shard_arguments():
    runner = _LiveEvalRunner()
    trainer = _trainer(runner)
    trainer.accelerator.process_index = 1
    trainer.accelerator.num_processes = 2

    result = trainer._run_distributed_rl_games_eval(stage="mid_train")

    assert runner.model_modes == [False]
    assert runner.run_kwargs == [{"shard_rank": 1, "shard_count": 2, "save": False}]
    assert result.aggregate["mean_reward"] == 4.0
    assert runner.saved[1:] == (250, "mid_train")
    assert trainer.accelerator.wait_calls == 1
    assert trainer.model.training is True
