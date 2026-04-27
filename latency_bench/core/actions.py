from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from latency_bench.core.types import Action


class ActionResolver:
    """Resolve config action references into runtime Action objects."""

    def __init__(self, env_config: Mapping[str, Any]):
        self.env_config = env_config
        self._action_map = self._normalize_action_map(env_config.get("action_map", {}))
        self._noop_ref = self._require_noop_ref(env_config)
        self._oneshot_refs = set(env_config.get("oneshot_actions", []))
        self._oneshot_ref_names = {str(item) for item in self._oneshot_refs}
        noop_value, noop_name = self._resolve_value_and_name(self._noop_ref)
        self.noop_action = Action(value=noop_value, name=noop_name, is_noop=True, is_oneshot=False)

    def resolve(self, action_ref: Any) -> Action:
        value, name = self._resolve_value_and_name(action_ref)
        is_noop = value == self.noop_action.value or name == self.noop_action.name
        is_oneshot = (not is_noop) and self._is_oneshot(action_ref=action_ref, value=value, name=name)
        return Action(value=value, name=name, is_noop=is_noop, is_oneshot=is_oneshot)

    def resolve_many(self, action_refs: Iterable[Any]) -> list[Action]:
        return [self.resolve(action_ref) for action_ref in action_refs]

    def default_action_refs(self) -> list[Any]:
        if self._action_map:
            return list(self._action_map.keys())
        return [self._noop_ref]

    def _resolve_value_and_name(self, action_ref: Any) -> tuple[Any, str]:
        action_name = str(action_ref)
        if action_name in self._action_map:
            return self._action_map[action_name], action_name

        for mapped_name, mapped_value in self._action_map.items():
            if action_ref == mapped_value:
                return mapped_value, mapped_name

        return action_ref, action_name

    def _is_oneshot(self, action_ref: Any, value: Any, name: str) -> bool:
        return (
            action_ref in self._oneshot_refs
            or value in self._oneshot_refs
            or str(action_ref) in self._oneshot_ref_names
            or str(value) in self._oneshot_ref_names
            or name in self._oneshot_ref_names
        )

    @staticmethod
    def _normalize_action_map(action_map: Any) -> dict[str, Any]:
        if action_map is None:
            return {}
        if not isinstance(action_map, Mapping):
            raise ValueError("env.action_map must be a mapping when provided")
        return {str(key): value for key, value in action_map.items()}

    @staticmethod
    def _require_noop_ref(env_config: Mapping[str, Any]) -> Any:
        if "noop_action" not in env_config:
            raise ValueError("Missing required field: noop_action")
        return env_config["noop_action"]
