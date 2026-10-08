import pytest

from deployment.model_server.policy_norm_processor import _resolve_robot_type


@pytest.mark.parametrize(
    ("task_name", "data_mix", "robot_type"),
    [
        ("flappy", "flappy_train", "rl_games_flappy"),
        ("demon_attack", "demon_attack_train", "rl_games_demon_attack"),
        ("deadly_corridor", "deadly_corridor_train", "rl_games_deadly_corridor"),
    ],
)
def test_legacy_rl_games_resolves_from_static_mixture(
    task_name: str,
    data_mix: str,
    robot_type: str,
) -> None:
    config = {
        "rl_games": {"task": task_name},
        "datasets": {"vla_data": {"data_mix": data_mix}},
    }

    assert _resolve_robot_type(config) == robot_type


def test_unknown_static_mixture_still_fails() -> None:
    config = {
        "rl_games": {"task": "flappy"},
        "datasets": {"vla_data": {"data_mix": "unregistered_custom_mix"}},
    }

    with pytest.raises(KeyError, match="unregistered_custom_mix"):
        _resolve_robot_type(config)


def test_history_vla_bundle_resolves_explicit_robot_type() -> None:
    config = {"datasets": {"vla_data": {"robot_type": "rl_games_flappy"}}}

    assert _resolve_robot_type(config) == "rl_games_flappy"
