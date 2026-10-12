from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

import torch.utils.cpp_extension as cpp_extension
import torch


CUDA_ARCH_LIST = "8.6;9.0;12.0"


def ensure_compatible_cuda_extension_toolchain() -> Path:
    cuda_home = Path(os.environ["CUDA_HOME"])
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCH_LIST
    _activate_cuda_home(cuda_home)
    return cuda_home


def _activate_cuda_home(cuda_home: Path) -> None:
    os.environ["CUDA_HOME"] = str(cuda_home)
    _prepend_paths("PATH", [cuda_home / "bin"])
    target_root = cuda_home / "targets" / "x86_64-linux"
    _prepend_paths("CPATH", [cuda_home / "include", target_root / "include"])
    library_paths = [
        cuda_home / "lib",
        cuda_home / "lib64",
        target_root / "lib",
        target_root / "lib" / "stubs",
    ]
    _prepend_paths("LD_LIBRARY_PATH", library_paths)
    _prepend_paths("LIBRARY_PATH", library_paths)
    cpp_extension.CUDA_HOME = str(cuda_home)


def _prepend_paths(name: str, paths: Iterable[Path]) -> None:
    prefixes = [str(path) for path in paths if path.is_dir()]
    existing = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    os.environ[name] = os.pathsep.join(prefixes + [part for part in existing if part not in prefixes])


def main() -> None:
    """Build only the CUDA extensions required by the selected simulator."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--env", choices=("flappy", "atari", "all"), default="all")
    args = parser.parse_args()
    loaders = []
    if args.env in ("flappy", "all"):
        from training.flappy_sf.flappy_cuda_extension import load_flappy_cuda_extension

        loaders.append(load_flappy_cuda_extension)
    if args.env in ("atari", "all"):
        from training.demon_attack_sf.cule_atari_extension import load_cule_atari_extension

        loaders.append(load_cule_atari_extension)
    for loader in loaders:
        extension = loader()
        torch.cuda.synchronize()
        code_objects = subprocess.check_output(["cuobjdump", "--list-elf", extension.__file__], text=True)
        for architecture in ("sm_86", "sm_90", "sm_120"):
            assert architecture in code_objects, code_objects
        print(f"extension={extension.__name__}\n{code_objects}", flush=True)


if __name__ == "__main__":
    main()
