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
bash install.sh
```

This creates the benchmark environment (`latency`) and an OpenVLA runtime
environment (`starvla_rl_games_openvla`). The installer selects a compatible
Torch/CUDA build and installs all three benchmark task environments.

Install another VLA backend in its own environment when needed:

```bash
bash third_party/starVLA/examples/rl_games/install/bootstrap.sh \
  --tier use \
  --model pi0 \
  --env all
```

Supported model targets are `openvla`, `pi0`, `pi05`, `gr00t`, and `wan_oft`.
Each target uses a separate `starvla_rl_games_<model>` Conda environment.

### For development and training

Install the base benchmark first, then upgrade the model environment to the
development tier:

```bash
bash install.sh
bash third_party/starVLA/examples/rl_games/install/bootstrap.sh \
  --tier dev \
  --model openvla \
  --env all
```

The `dev` tier adds training, testing, data-processing, FlashAttention, and
model-specific development dependencies. Change `--model` to develop or train a
different VLA backend.

## Environment variables

Create a local `.env` and set the services used by your experiments:

```bash
HF_TOKEN=...
HF_PROFILE_REPO=latency-sensitive-bench/profiles
HF_PROFILE_REPO_TYPE=dataset
WANDB_API_KEY=...
```
