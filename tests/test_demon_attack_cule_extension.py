from pathlib import Path

from training.demon_attack_sf import cule_atari_extension


def test_load_cule_atari_extension_uses_torch_pybind11_and_cccl(monkeypatch, tmp_path):
    cuda_home = tmp_path / "cuda"
    cccl_include = cuda_home / "include" / "cccl"
    cccl_include.mkdir(parents=True)

    captured: dict[str, object] = {}

    def fake_load(**kwargs):
        captured.update(kwargs)
        return "demon_attack_cule_atari"

    monkeypatch.setattr(cule_atari_extension, "CUDA_HOME", str(cuda_home))
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
    assert include_paths == [
        str(cule_root),
        str(cule_root / "third_party" / "agency"),
        str(cccl_include),
    ]
    assert str(cule_root / "third_party" / "pybind11" / "include") not in include_paths
    assert captured["extra_cuda_cflags"] == ["-O3", "-DCULE_ONLY_4K_COMPILE", "-std=c++14"]
