# Latency Sensitive Bench

## Installation

Clone the repository together with its third-party submodules:

```bash
git clone --recurse-submodules https://github.com/ZihanWang314/latency-sensitive-bench.git
cd latency-sensitive-bench
```

### For use

Use this setup to run the benchmark or evaluate a pretrained OpenVLA policy:

```bash
bash install.sh --tier use --model openvla --env all
```

This creates the benchmark environment (`latency`) and an OpenVLA runtime
environment (`starvla_rl_games_openvla`). The installer selects a compatible
Torch/CUDA build and installs all three benchmark task environments.

Install another VLA backend in its own environment when needed:

```bash
bash install.sh \
  --tier use \
  --model pi0 \
  --env all
```

Supported model targets are `openvla`, `pi0`, `pi05`, `gr00t`, and `wan_oft`.
Each target uses a separate `starvla_rl_games_<model>` Conda environment.

WanOFT checkpoints from the Flappy Bird, Demon Attack, and Deadly Corridor
StarVLA pipelines can be evaluated with temporal multi-image input. See
[docs/starvla_wan_oft_eval.md](docs/starvla_wan_oft_eval.md) for the three
configs, checkpoint command, and raw-frame history contract.

### For development and training

Use the development tier when training models, exporting datasets, running
analysis, or modifying the codebase:

```bash
bash install.sh \
  --tier dev \
  --model openvla \
  --env all
```

The `dev` tier adds training, testing, data-processing, FlashAttention, and
model-specific development dependencies. Change `--model` to develop or train a
different VLA backend.

Use repeated `--env` options to install only selected tasks. On a non-interactive
machine, pass `--accept-rom-license` after reviewing the Atari ROM license. Use
`--torch-profile cpu` for a CPU-only installation; otherwise `auto` selects the
profile from the available GPU and driver.

## Environment variables

Create a local `.env` and set the services used by your experiments:

```bash
HF_TOKEN=...
HF_PROFILE_REPO=latency-sensitive-bench/profiles
HF_PROFILE_REPO_TYPE=dataset
WANDB_API_KEY=...
```
