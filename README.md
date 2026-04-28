# Latency Sensitive Bench

This README currently only covers environment setup.

## Environment

The working reference environment is the local conda environment named `latency`:

- Python 3.10.20
- PyTorch 2.10.0 + CUDA 12.6
- Gymnasium 1.2.3
- Stable-Baselines3 2.8.0
- Local submodule build of `flappy-bird-gymnasium`

`requirements.txt` is not a full `pip freeze`. It pins the main direct dependencies used by this project, except PyTorch. Install PyTorch separately first so the CUDA/CPU wheel matches the machine you are using.

## Submodules

This repository depends on the `flappy-bird-gymnasium` submodule:

```bash
git submodule update --init --recursive
```

## Install

Create a Python 3.10 conda environment:

```bash
conda create -n latency python=3.10
conda activate latency
python -m pip install --upgrade pip
```

Install PyTorch 2.10.0 for your machine before installing the rest of the requirements.

For the reference CUDA 12.6 environment:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
```

For a CPU-only environment:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

For a different CUDA version, keep the PyTorch version fixed at `2.10.0` and replace the wheel index with the matching PyTorch CUDA index for that machine, for example `cu121`, `cu126`, or another supported build. Check the machine's driver/runtime with `nvidia-smi` and use the PyTorch installation selector if unsure.

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Do not skip the PyTorch step. `stable-baselines3` depends on PyTorch, so running only `python -m pip install -r requirements.txt` may cause pip to install a different PyTorch build indirectly.

To reinstall Flappy Bird environment package, run:

```bash
python -m pip install -e ./flappy-bird-gymnasium
```

## Common Entrypoints

Main benchmark example:

```bash
python -m latency_bench.run --config configs/experiments/flappy_ppo_zero_latency.yaml
```

CleanRL-style Flappy PPO training entrypoint:

```bash
python scripts/flappy_cleanrl_gpu_render_ppo.py --help
```

Historical SB3 scripts are still under `scripts/` and `bash/`. Activate the `latency` environment and initialize the submodule before running them.
