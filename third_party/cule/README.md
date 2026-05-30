Vendored CuLE subset for Demon Attack CUDA Atari emulation.

Source: https://github.com/NVlabs/cule
License: BSD-3-Clause, see `LICENSE.TXT`.

Only the runtime source and headers needed by `training.demon_attack_sf` are
vendored here. Examples, tests, docs, and Python package wrappers are omitted.

Local patches:
- `torchcule/frontend.cpp` uses `TORCH_EXTENSION_NAME` so PyTorch can build the
  extension under a repo-specific module name.
- `cule/atari/games/detail/types.hpp` maps the `ale_py` Demon Attack ROM MD5
  `f0e0addc07971561ab80d9abe1b8d333` to `GAME_DEMON_ATTACK`.
- `cule/atari/wrapper.cpp` supports `CULE_ONLY_4K_COMPILE`, which compiles the
  4K mapper used by Demon Attack without the full mapper matrix.
- `third_party/agency/agency/coordinate/point.hpp` avoids instantiating an old
  agency scalar-multiply trait path that fails under CUDA 12.
- `training/demon_attack_sf/cule_atari_extension.py` builds `torchcule`
  against PyTorch's bundled `pybind11` headers instead of the vendored
  `third_party/pybind11` snapshot.
- `training/demon_attack_sf/cule_atari_extension.py` appends
  `CUDA_HOME/include/cccl` so the CUDA 13 pip toolkit exposes `thrust` under
  the include layout used by `nvidia-cuda-cccl`.
