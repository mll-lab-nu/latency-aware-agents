from __future__ import annotations

import os

import pytest


def test_uses_fixed_cuda_home_and_dual_arch_list(monkeypatch, tmp_path):
    from training.common import cuda_extension_build

    cuda_home = tmp_path / "cuda-12.8"
    (cuda_home / "bin").mkdir(parents=True)
    (cuda_home / "lib").mkdir()
    (cuda_home / "lib64").mkdir()
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("LIBRARY_PATH", raising=False)

    selected = cuda_extension_build.ensure_compatible_cuda_extension_toolchain()

    assert selected == cuda_home
    assert os.environ["CUDA_HOME"] == str(cuda_home)
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "9.0;12.0"
    assert os.environ["PATH"].split(os.pathsep)[0] == str(cuda_home / "bin")
    assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep)[:2] == [
        str(cuda_home / "lib"),
        str(cuda_home / "lib64"),
    ]
    assert cuda_extension_build.cpp_extension.CUDA_HOME == str(cuda_home)


def test_cuda_home_is_required(monkeypatch):
    from training.common import cuda_extension_build

    monkeypatch.delenv("CUDA_HOME", raising=False)

    with pytest.raises(KeyError, match="CUDA_HOME"):
        cuda_extension_build.ensure_compatible_cuda_extension_toolchain()


@pytest.mark.parametrize(
    ("module_path", "function_name"),
    [
        ("training.flappy_sf.flappy_cuda_extension", "load_flappy_cuda_extension"),
        ("training.demon_attack_sf.cule_atari_extension", "load_cule_atari_extension"),
    ],
)
def test_cuda_extension_loaders_activate_toolchain_before_jit_load(
    monkeypatch,
    module_path,
    function_name,
):
    module = pytest.importorskip(module_path)
    calls: list[str] = []

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
