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

Install the Atari ROMs once after the Python packages are installed:

```bash
AutoROM --accept-license
```

If you want the default Sample Factory runs to report to Weights & Biases, log in once on the machine before training:

```bash
wandb login
```

## Common Entrypoints

Main benchmark example:

```bash
python -m latency_bench.run --config configs/experiments/flappy_ppo_zero_latency.yaml
```

Unified Sample Factory training entrypoint:

```bash
python -m training.train --config configs/experiments/deadly_corridor/sf_zero_latency.yaml
```

Non-zero latency training uses the same entrypoint with a different config, for example:

```bash
python -m training.train --config configs/experiments/demon_attack/sf_fixed_latency_4.yaml
```

Sample Factory checkpoints can be evaluated through the common latency bench entrypoint using the SF evaluation configs:

```bash
python -m latency_bench.run --config configs/experiments/flappy/sf_eval_zero_latency.yaml
```

Parallel eval is controlled by `evaluation.eval_parallel_envs`. It is supported for Flappy and DemonAttack Sample Factory queue eval, and for `latency_bench` simulated eval. See [Evaluation Parallelism](docs/eval_parallelism.md) for supported paths, recommended values, subprocess env stepping notes, and benchmark commands.

CleanRL-style Flappy PPO training entrypoint:

```bash
python scripts/flappy_cleanrl_gpu_render_ppo.py --help
```

Historical SB3 scripts are still under `scripts/` and `bash/`. Activate the `latency` environment and initialize the submodule before running them.

## Sample Factory Backend

The repository now includes a config-driven Sample Factory backend under `training/`. This backend is intended to unify training across Flappy, Demon Attack, and Deadly Corridor while keeping YAML configs as the source of truth.

The Python dependency is pinned in `requirements.txt` as `sample-factory==2.1.1`. The official Sample Factory docs cover base installation and environment-specific integrations:

- [Installation](https://www.samplefactory.dev/01-get-started/installation/)
- [Custom environments](https://www.samplefactory.dev/03-customization/custom-environments/)
- [Atari integration](https://www.samplefactory.dev/09-environment-integrations/atari/)
- [ViZDoom integration](https://www.samplefactory.dev/09-environment-integrations/vizdoom/)
