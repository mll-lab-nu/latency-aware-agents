from __future__ import annotations

from types import SimpleNamespace

import training.train as train_entry


def _flappy_config() -> dict:
    return {
        "experiment": {"name": "flappy_l0_seed0"},
        "backend": {"type": "flappy_ppo"},
        "env": {"name": "flappy"},
    }


def test_training_train_dispatches_flappy_ppo_backend(monkeypatch):
    calls = []

    def fake_run_from_config(config):
        calls.append(config)
        return 23

    import training.flappy_ppo.config_backend as flappy_backend

    monkeypatch.setattr(flappy_backend, "run_from_config", fake_run_from_config)

    assert train_entry.run_from_config(_flappy_config()) == 23
    assert calls == [_flappy_config()]


def test_training_train_keeps_sample_factory_integration_dispatch(monkeypatch):
    cfg = {
        "experiment": {"name": "demon_attack_sf_l0"},
        "backend": {"type": "sample_factory"},
        "env": {"name": "demon_attack"},
        "latency": {},
        "training": {},
    }
    module = SimpleNamespace(run_from_config=lambda config: 17)

    monkeypatch.setattr(train_entry, "import_integration_module", lambda config: module)

    assert train_entry.run_from_config(cfg) == 17
