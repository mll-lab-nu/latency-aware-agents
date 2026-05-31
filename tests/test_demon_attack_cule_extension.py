from pathlib import Path
import importlib.util

from training.demon_attack_sf import cule_atari_extension


def test_load_cule_atari_extension_uses_torch_pybind11_and_cccl(monkeypatch, tmp_path):
    cuda_home = tmp_path / "cuda"
    cccl_include = cuda_home / "include" / "cccl"
    cccl_include.mkdir(parents=True)
    env_include = tmp_path / "env" / "include"
    env_include.mkdir(parents=True)

    captured: dict[str, object] = {}

    def fake_load(**kwargs):
        captured.update(kwargs)
        return "demon_attack_cule_atari"

    monkeypatch.setattr(cule_atari_extension, "ensure_compatible_cuda_extension_toolchain", lambda: None)
    monkeypatch.setattr(cule_atari_extension.cpp_extension, "CUDA_HOME", str(cuda_home))
    monkeypatch.setattr(cule_atari_extension.sys, "prefix", str(tmp_path / "env"))
    monkeypatch.setattr(cule_atari_extension, "load", fake_load)

    cule_atari_extension.load_cule_atari_extension.cache_clear()
    try:
        result = cule_atari_extension.load_cule_atari_extension()
    finally:
        cule_atari_extension.load_cule_atari_extension.cache_clear()

    repo_root = Path(__file__).resolve().parents[1]
    cule_root = repo_root / "third_party" / "cule"
    include_paths = captured["extra_include_paths"]

    assert result == "demon_attack_cule_atari"
    assert str(cule_root) in include_paths
    assert str(cule_root / "third_party" / "agency") in include_paths
    assert str(env_include) in include_paths
    assert str(cccl_include) in include_paths
    assert str(cule_root / "third_party" / "pybind11" / "include") not in include_paths
    assert captured["extra_cflags"] == ["-O3", "-std=c++17"]
    assert captured["extra_cuda_cflags"] == ["-O3", "-std=c++17"]
