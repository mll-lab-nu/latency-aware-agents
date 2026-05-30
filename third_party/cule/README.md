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
- `training/demon_attack_sf/cule_atari_extension.py` builds `torchcule`
  with `-std=c++17` and appends the active environment's `include` directory
  so CUDA 13 CCCL headers and `zlib.h` resolve in the conda-installed toolkit.
- `third_party/agency/agency/detail/shape.hpp` uses `std::numeric_limits`
  instead of the removed private header `thrust/detail/integer_traits.h`, so
  the vendored agency subset still builds against the CUDA 13 pip toolkit.
- `cule/config.hpp` explicitly includes `cstdint`, and
  `cule/atari/games/detail/types.hpp` explicitly includes `array`, so the
  vendored Atari headers no longer rely on transitive standard-library
  includes from older toolchains.
- `third_party/agency/agency/memory/allocator/detail/any_allocator.hpp`
  gives `concrete_allocator` an explicit `__AGENCY_ANNOTATION` destructor to
  match the abstract base allocator under CUDA 13's stricter host/device
  override checks.
