__all__ = ["apply_model_alias", "apply_action_spec", "validate_rl_games_config", "sync_kv_memory_obs_window", "CheckpointSyncManager", "RlGamesEvalRunner", "build_rl_games_eval_runner"]


def build_rl_games_eval_runner(cfg, output_dir: str):
    """Build the eval runner selected by the resolved rl_games config."""
    backend = cfg.rl_games.env_eval.eval_backend
    if backend == "eval_core":
        from . import RlGamesEvalRunner

        return RlGamesEvalRunner(cfg=cfg, output_dir=output_dir)
    if backend == "latency_bench":
        from latency_bench.integrations.starvla_rl_games_eval_runner import (
            LatencyBenchRlGamesEvalRunner,
        )

        return LatencyBenchRlGamesEvalRunner(cfg=cfg, output_dir=output_dir)
    raise ValueError(f"Unknown rl_games.env_eval.eval_backend: {backend}")


def __getattr__(name):
    if name == "apply_model_alias":
        from .alias import apply_model_alias

        return apply_model_alias
    if name == "apply_action_spec":
        from .action_spec import apply_action_spec

        return apply_action_spec
    if name == "validate_rl_games_config":
        from .config_validation import validate_rl_games_config

        return validate_rl_games_config
    if name == "sync_kv_memory_obs_window":
        from .config_validation import sync_kv_memory_obs_window

        return sync_kv_memory_obs_window
    if name == "CheckpointSyncManager":
        from .checkpoint_sync import CheckpointSyncManager

        return CheckpointSyncManager
    if name == "RlGamesEvalRunner":
        from .eval_core import RlGamesEvalRunner

        return RlGamesEvalRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
