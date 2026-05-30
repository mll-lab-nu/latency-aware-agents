from __future__ import annotations

from types import SimpleNamespace


def test_gpu_sampling_patch_only_enables_env_with_cpu_tensor_channel():
    from training.common.sf_gpu_sampling_compat import _is_enabled_for_sample_factory_env

    enabled_env_names = {"latency_flappy"}
    cpu_runner = SimpleNamespace(
        cfg=SimpleNamespace(env="latency_flappy"),
        vec_env=SimpleNamespace(),
    )
    gpu_runner = SimpleNamespace(
        cfg=SimpleNamespace(env="latency_flappy"),
        vec_env=SimpleNamespace(sample_factory_step_cpu_tensors={}),
    )

    assert _is_enabled_for_sample_factory_env(cpu_runner, enabled_env_names) is False
    assert _is_enabled_for_sample_factory_env(gpu_runner, enabled_env_names) is True


def test_gpu_sampling_patch_supports_multiple_gpu_env_names():
    from training.common.sf_gpu_sampling_compat import _is_enabled_for_sample_factory_env

    enabled_env_names = {"latency_flappy", "latency_demon_attack"}
    demon_runner = SimpleNamespace(
        cfg=SimpleNamespace(env="latency_demon_attack"),
        vec_env=SimpleNamespace(sample_factory_step_cpu_tensors={}),
    )
    unknown_runner = SimpleNamespace(
        cfg=SimpleNamespace(env="latency_unknown"),
        vec_env=SimpleNamespace(sample_factory_step_cpu_tensors={}),
    )

    assert _is_enabled_for_sample_factory_env(demon_runner, enabled_env_names) is True
    assert _is_enabled_for_sample_factory_env(unknown_runner, enabled_env_names) is False
