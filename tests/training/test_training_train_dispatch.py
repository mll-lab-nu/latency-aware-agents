from __future__ import annotations

from types import SimpleNamespace

import training.train as train_entry


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
