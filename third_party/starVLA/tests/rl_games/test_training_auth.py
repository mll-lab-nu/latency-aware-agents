"""Long-lived authentication opt-in contracts for the reference training runtime."""

import sys
from types import SimpleNamespace

import pytest

from starVLA.training.rl_games.auth import login_training_services


@pytest.mark.parametrize("project", [None, "user-project"])
def test_training_auth_requires_explicit_project_and_credential_file(tmp_path, monkeypatch, project):
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "WANDB_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "user-key")
    (tmp_path / ".env").write_text("WANDB_API_KEY=other-key\nHF_TOKEN=unexpected-token\n")
    calls = []
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(login=lambda **kwargs: calls.append(kwargs)))
    result = login_training_services({"wandb_project": project}, workspace_dir=tmp_path, repo_root=tmp_path)
    assert result.env_file is None
    assert result.hf_configured is False
    assert calls == ([] if project is None else [{"key": "user-key", "relogin": True}])
