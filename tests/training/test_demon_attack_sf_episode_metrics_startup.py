from __future__ import annotations

from types import SimpleNamespace


def _patch_run_from_config_dependencies(monkeypatch, integration, cfg):
    monkeypatch.setattr(integration, "register_demon_attack_envs", lambda: None)
    monkeypatch.setattr(integration, "build_cli_args_from_config", lambda _config: [])
    monkeypatch.setattr(
        integration,
        "SAMPLE_FACTORY_CONFIG_PARSER",
        SimpleNamespace(parse_train=lambda _args: cfg),
    )
    monkeypatch.setattr(integration, "apply_runtime_logging_config", lambda _cfg, _logging_cfg: None)
    monkeypatch.setattr(integration, "seed_everything", lambda _seed: None, raising=False)
    monkeypatch.setattr(integration, "patch_sample_factory_wandb", lambda: None)
    monkeypatch.setattr(
        integration,
        "ensure_sample_factory_raw_reward_console_stats_installed",
        lambda *, env_name: None,
    )
    monkeypatch.setattr(integration, "run_sample_factory_training", lambda _cfg, _config: 0)


def test_demon_attack_gpu_training_uses_sample_factory_episode_metrics_not_direct_wandb(
    monkeypatch,
    tmp_path,
):
    from training.demon_attack_sf import integration

    metrics_path = tmp_path / "episode_metrics.jsonl"
    metrics_path.write_text("old\n", encoding="utf-8")
    cfg = SimpleNamespace(
        mode="train",
        simulator="gpu",
        episode_metrics_path=str(metrics_path),
        restart_behavior="overwrite",
        seed=0,
    )
    _patch_run_from_config_dependencies(monkeypatch, integration, cfg)
    started = []
    monkeypatch.setattr(
        integration,
        "start_episode_metrics_aggregator",
        lambda _cfg: started.append(_cfg) or (None, None),
    )
    stopped = []
    monkeypatch.setattr(
        integration,
        "stop_episode_metrics_aggregator",
        lambda stop_event, thread: stopped.append((stop_event, thread)),
    )

    integration.run_from_config({"logging": {}})

    assert started == []
    assert not metrics_path.exists()
    assert stopped == [(None, None)]


def test_demon_attack_gpu_training_resume_keeps_episode_metrics_file(monkeypatch, tmp_path):
    from training.demon_attack_sf import integration

    metrics_path = tmp_path / "episode_metrics.jsonl"
    metrics_path.write_text("old\n", encoding="utf-8")
    cfg = SimpleNamespace(
        mode="train",
        simulator="gpu",
        episode_metrics_path=str(metrics_path),
        restart_behavior="resume",
        seed=0,
    )
    _patch_run_from_config_dependencies(monkeypatch, integration, cfg)
    started = []
    monkeypatch.setattr(
        integration,
        "start_episode_metrics_aggregator",
        lambda _cfg: started.append(_cfg) or (None, None),
    )
    monkeypatch.setattr(integration, "stop_episode_metrics_aggregator", lambda stop_event, thread: None)

    integration.run_from_config({"logging": {}})

    assert started == []
    assert metrics_path.read_text(encoding="utf-8") == "old\n"


def test_demon_attack_cpu_training_still_starts_direct_episode_metrics_aggregator(monkeypatch, tmp_path):
    from training.demon_attack_sf import integration

    cfg = SimpleNamespace(
        mode="train",
        simulator="cpu",
        episode_metrics_path=str(tmp_path / "episode_metrics.jsonl"),
        restart_behavior="overwrite",
        seed=0,
    )
    _patch_run_from_config_dependencies(monkeypatch, integration, cfg)
    started = []
    monkeypatch.setattr(
        integration,
        "start_episode_metrics_aggregator",
        lambda _cfg: started.append(_cfg) or ("stop", "thread"),
    )
    stopped = []
    monkeypatch.setattr(
        integration,
        "stop_episode_metrics_aggregator",
        lambda stop_event, thread: stopped.append((stop_event, thread)),
    )

    integration.run_from_config({"logging": {}})

    assert started == [cfg]
    assert stopped == [("stop", "thread")]
