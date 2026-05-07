from __future__ import annotations

from latency_bench.core.actions import ActionResolver


def _flappy_env_config() -> dict:
    return {
        "name": "flappy",
        "fps_env": 30,
        "noop_action": "noop",
        "action_map": {
            "noop": 0,
            "flap": 1,
        },
        "oneshot_actions": ["flap"],
    }


def test_noop_resolves_to_noop_action():
    resolver = ActionResolver(_flappy_env_config())

    action = resolver.resolve("noop")

    assert action.value == 0
    assert action.name == "noop"
    assert action.is_noop
    assert not action.is_oneshot


def test_flap_resolves_to_mapped_oneshot_action():
    resolver = ActionResolver(_flappy_env_config())

    action = resolver.resolve("flap")

    assert action.value == 1
    assert action.name == "flap"
    assert not action.is_noop
    assert action.is_oneshot


def test_policy_actions_can_be_resolved_in_batches():
    resolver = ActionResolver(_flappy_env_config())

    actions = resolver.resolve_many(["noop", "flap"])

    assert [action.value for action in actions] == [0, 1]
    assert [action.name for action in actions] == ["noop", "flap"]


def test_missing_action_map_does_not_invent_flap():
    resolver = ActionResolver({"name": "dummy", "fps_env": 30, "noop_action": "noop"})

    assert resolver.default_action_refs() == ["noop"]
    assert resolver.resolve("flap").value == "flap"


def test_list_valued_action_map_does_not_raise_type_error():
    resolver = ActionResolver(
        {
            "name": "deadly_corridor",
            "fps_env": 35,
            "noop_action": "noop",
            "action_map": {
                "noop": [0, 0, 0, 0],
                "attack": [0, 0, 0, 1],
            },
            "oneshot_actions": ["attack"],
        }
    )

    noop = resolver.resolve("noop")
    attack = resolver.resolve("attack")

    assert noop.value == [0, 0, 0, 0]
    assert noop.is_noop
    assert attack.value == [0, 0, 0, 1]
    assert attack.is_oneshot
