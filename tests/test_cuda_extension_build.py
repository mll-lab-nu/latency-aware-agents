from __future__ import annotations

import os
from pathlib import Path

import pytest


def _write_nvcc(path: Path, gpu_codes: str) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"--list-gpu-code\" ]]; then\n"
        f"  printf '%s\\n' '{gpu_codes}'\n"
        "fi\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_selects_env_cuda_when_system_nvcc_lacks_gpu_arch(monkeypatch, tmp_path):
    from training.common import cuda_extension_build

    system_cuda = tmp_path / "usr" / "local" / "cuda"
    env_cuda = tmp_path / "env" / "lib" / "python3.10" / "site-packages" / "nvidia" / "cu13"
    (env_cuda / "lib").mkdir(parents=True)
    (env_cuda / "lib64").mkdir(parents=True)
    _write_nvcc(system_cuda / "bin" / "nvcc", "sm_100")
    _write_nvcc(env_cuda / "bin" / "nvcc", "sm_100\nsm_103")

    monkeypatch.setattr(cuda_extension_build.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda_extension_build.torch.cuda, "get_device_capability", lambda: (10, 3))
    monkeypatch.setattr(cuda_extension_build.cpp_extension, "CUDA_HOME", str(system_cuda))
    monkeypatch.setattr(cuda_extension_build, "_system_nvcc_paths", lambda: iter([system_cuda / "bin" / "nvcc"]))
    monkeypatch.setattr(cuda_extension_build.site, "getsitepackages", lambda: [str(env_cuda.parents[2])])
    monkeypatch.setattr(cuda_extension_build.sys, "prefix", str(tmp_path / "env"))
    monkeypatch.setenv("CUDA_HOME", str(system_cuda))
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("LIBRARY_PATH", raising=False)

    selected = cuda_extension_build.ensure_compatible_cuda_extension_toolchain()

    assert selected == env_cuda
    assert os.environ["CUDA_HOME"] == str(env_cuda)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(env_cuda / "bin")
    assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep)[:2] == [
        str(env_cuda / "lib64"),
        str(env_cuda / "lib"),
    ]
    assert os.environ["LIBRARY_PATH"].split(os.pathsep)[:2] == [
        str(env_cuda / "lib64"),
        str(env_cuda / "lib"),
    ]
    assert cuda_extension_build.cpp_extension.CUDA_HOME == str(env_cuda)


def test_leaves_environment_unchanged_when_no_nvcc_supports_gpu_arch(monkeypatch, tmp_path):
    from training.common import cuda_extension_build

    system_cuda = tmp_path / "usr" / "local" / "cuda"
    env_cuda = tmp_path / "env" / "lib" / "python3.10" / "site-packages" / "nvidia" / "cu13"
    _write_nvcc(system_cuda / "bin" / "nvcc", "sm_100")
    _write_nvcc(env_cuda / "bin" / "nvcc", "sm_100")

    monkeypatch.setattr(cuda_extension_build.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda_extension_build.torch.cuda, "get_device_capability", lambda: (10, 3))
    monkeypatch.setattr(cuda_extension_build.cpp_extension, "CUDA_HOME", str(system_cuda))
    monkeypatch.setattr(cuda_extension_build, "_system_nvcc_paths", lambda: iter([system_cuda / "bin" / "nvcc"]))
    monkeypatch.setattr(cuda_extension_build.site, "getsitepackages", lambda: [str(env_cuda.parents[2])])
    monkeypatch.setattr(cuda_extension_build.sys, "prefix", str(tmp_path / "env"))
    monkeypatch.setenv("CUDA_HOME", str(system_cuda))
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    monkeypatch.setenv("LIBRARY_PATH", "/existing/link")
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)

    selected = cuda_extension_build.ensure_compatible_cuda_extension_toolchain()

    assert selected is None
    assert os.environ["CUDA_HOME"] == str(system_cuda)
    assert os.environ["PATH"] == "/usr/bin"
    assert os.environ["LD_LIBRARY_PATH"] == "/existing/lib"
    assert os.environ["LIBRARY_PATH"] == "/existing/link"
    assert "TORCH_CUDA_ARCH_LIST" not in os.environ
    assert cuda_extension_build.cpp_extension.CUDA_HOME == str(system_cuda)


@pytest.mark.parametrize(
    ("module_path", "function_name"),
    [
        ("training.flappy_sf.flappy_cuda_extension", "load_flappy_cuda_extension"),
        ("training.demon_attack_sf.demon_attack_cuda_extension", "load_demon_attack_cuda_extension"),
        ("training.demon_attack_sf.cule_atari_extension", "load_cule_atari_extension"),
        ("training.deadly_corridor_sf.deadly_corridor_cuda_extension", "load_deadly_corridor_cuda_extension"),
    ],
)
def test_cuda_extension_loaders_select_toolchain_before_jit_load(monkeypatch, module_path, function_name):
    module = pytest.importorskip(module_path)
    calls: list[str] = []

    if hasattr(module, "torch"):
        monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)

    def fake_ensure_toolchain():
        calls.append("toolchain")

    def fake_load(**kwargs):
        calls.append("load")
        return kwargs["name"]

    monkeypatch.setattr(module, "ensure_compatible_cuda_extension_toolchain", fake_ensure_toolchain)
    monkeypatch.setattr(module, "load", fake_load)

    load_extension = getattr(module, function_name)
    load_extension.cache_clear()
    try:
        result = load_extension()
    finally:
        load_extension.cache_clear()

    assert result
    assert calls[:2] == ["toolchain", "load"]
