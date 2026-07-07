from __future__ import annotations

from types import SimpleNamespace


class _FakeProcess:
    def __init__(self, alive: bool) -> None:
        self._alive = bool(alive)

    def is_alive(self) -> bool:
        return self._alive


class _FakeSampler:
    def __init__(self, *components) -> None:
        self._components = list(components)

    def heartbeat_components(self):
        return list(self._components)


def _component(object_id: str, *, alive: bool = True):
    return SimpleNamespace(
        object_id=object_id,
        event_loop=SimpleNamespace(process=_FakeProcess(alive=alive)),
    )


def test_component_healthcheck_patch_enables_only_selected_env():
    from training.common.sf_component_healthcheck_compat import _is_enabled_for_sample_factory_env

    enabled_env_names = {"doom_deadly_corridor"}
    deadly_runner = SimpleNamespace(cfg=SimpleNamespace(env="doom_deadly_corridor"))
    other_runner = SimpleNamespace(cfg=SimpleNamespace(env="latency_flappy"))

    assert _is_enabled_for_sample_factory_env(deadly_runner, enabled_env_names) is True
    assert _is_enabled_for_sample_factory_env(other_runner, enabled_env_names) is False


def test_component_healthcheck_fails_when_process_dies():
    from training.common.sf_component_healthcheck_compat import _critical_component_failure_reason

    learner = _component("learner", alive=False)
    runner = SimpleNamespace(
        learners={0: learner},
        batchers={},
        sampler=_FakeSampler(),
        heartbeat_dict={type(learner): {"learner": 100.0}},
        start_time=0.0,
        heartbeat_report_sec=30.0,
    )

    assert _critical_component_failure_reason(runner, current_time=120.0) == "learner process died"


def test_component_healthcheck_fails_when_single_component_heartbeat_goes_stale():
    from training.common.sf_component_healthcheck_compat import _critical_component_failure_reason

    rollout = _component("rollout_worker_0")
    runner = SimpleNamespace(
        learners={},
        batchers={},
        sampler=_FakeSampler(rollout),
        heartbeat_dict={type(rollout): {"rollout_worker_0": 50.0}},
        start_time=0.0,
        heartbeat_report_sec=30.0,
    )

    assert (
        _critical_component_failure_reason(runner, current_time=90.0)
        == "rollout_worker_0 heartbeat stale for 40 seconds"
    )


def test_component_healthcheck_ignores_missing_startup_heartbeat_before_deadline():
    from training.common.sf_component_healthcheck_compat import _critical_component_failure_reason

    batcher = _component("batcher")
    runner = SimpleNamespace(
        learners={},
        batchers={0: batcher},
        sampler=_FakeSampler(),
        heartbeat_dict={type(batcher): {"batcher": None}},
        start_time=10.0,
        heartbeat_report_sec=30.0,
    )

    assert _critical_component_failure_reason(runner, current_time=80.0) is None


def test_component_healthcheck_fails_missing_startup_heartbeat_after_deadline():
    from training.common.sf_component_healthcheck_compat import _critical_component_failure_reason

    batcher = _component("batcher")
    runner = SimpleNamespace(
        learners={},
        batchers={0: batcher},
        sampler=_FakeSampler(),
        heartbeat_dict={type(batcher): {"batcher": None}},
        start_time=10.0,
        heartbeat_report_sec=30.0,
    )

    assert _critical_component_failure_reason(runner, current_time=101.0) == "batcher never sent heartbeat"
