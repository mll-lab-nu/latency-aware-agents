from __future__ import annotations

import time
from typing import Any

from training.common.sf_compat import (
    is_enabled_for_sample_factory_env,
    validate_sample_factory_version,
    validate_signature,
)


_PATCHED_ATTR = "_latency_bench_component_healthcheck_patched"
_ENABLED_ENV_NAMES_ATTR = "_latency_bench_component_healthcheck_env_names"
_ORIGINAL_CHECK_HEARTBEAT_ATTR = "_latency_bench_original_component_healthcheck_check_heartbeat"

_EXPECTED_CHECK_HEARTBEAT_PARAMS = ("self",)


def ensure_sample_factory_component_healthcheck_installed(*, env_name: str) -> bool:
    env_name = str(env_name)
    try:
        import sample_factory
        from sample_factory.algo.runners.runner import Runner
    except ImportError:
        return False

    validate_sample_factory_version(sample_factory, compat_label="component healthcheck compat")
    _patch_runner_check_heartbeat(Runner, env_name)
    return True


def _patch_runner_check_heartbeat(Runner: Any, env_name: str) -> None:
    current = Runner._check_heartbeat
    validate_signature(
        current,
        expected_params=_EXPECTED_CHECK_HEARTBEAT_PARAMS,
        label="Runner._check_heartbeat",
    )

    if getattr(current, _PATCHED_ATTR, False):
        getattr(current, _ENABLED_ENV_NAMES_ATTR).add(env_name)
        return

    original_check_heartbeat = current
    enabled_env_names = {env_name}

    def check_heartbeat_fail_fast(self):
        original_check_heartbeat(self)
        if self.stopped or not is_enabled_for_sample_factory_env(self, enabled_env_names):
            return

        failure_reason = _critical_component_failure_reason(self, current_time=time.time())
        if failure_reason is None:
            return

        from sample_factory.utils.utils import log

        log.error(f"Stopping training due to failed component healthcheck: {failure_reason}")
        self._stop_training(failed=True)

    setattr(check_heartbeat_fail_fast, _PATCHED_ATTR, True)
    setattr(check_heartbeat_fail_fast, _ENABLED_ENV_NAMES_ATTR, enabled_env_names)
    setattr(check_heartbeat_fail_fast, _ORIGINAL_CHECK_HEARTBEAT_ATTR, original_check_heartbeat)
    Runner._check_heartbeat = check_heartbeat_fail_fast


def _critical_component_failure_reason(runner: Any, *, current_time: float) -> str | None:
    startup_deadline = float(runner.start_time) + 3.0 * float(runner.heartbeat_report_sec)
    for component in _critical_components(runner):
        process = getattr(getattr(component, "event_loop", None), "process", None)
        if process is not None and not process.is_alive():
            return f"{component.object_id} process died"

        heartbeat_time = runner.heartbeat_dict[type(component)][component.object_id]
        if heartbeat_time is None:
            if current_time > startup_deadline:
                return f"{component.object_id} never sent heartbeat"
            continue

        if current_time - float(heartbeat_time) > float(runner.heartbeat_report_sec):
            return f"{component.object_id} heartbeat stale for {int(current_time - float(heartbeat_time))} seconds"
    return None


def _critical_components(runner: Any) -> list[Any]:
    components: list[Any] = []
    components.extend(runner.learners.values())
    components.extend(runner.batchers.values())
    sampler = getattr(runner, "sampler", None)
    if sampler is not None:
        components.extend(sampler.heartbeat_components())

    unique_components: list[Any] = []
    seen: set[str] = set()
    for component in components:
        object_id = str(component.object_id)
        if object_id in seen:
            continue
        seen.add(object_id)
        unique_components.append(component)
    return unique_components
