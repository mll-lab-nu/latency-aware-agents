# Latency Sensitive Bench

## Installation

Clone the repository together with its third-party submodules:

```bash
git clone --recurse-submodules https://github.com/ZihanWang314/latency-sensitive-bench.git
cd latency-sensitive-bench
```

The shared `latency` environment and every RL-games model environment use
Python 3.10, CUDA Toolkit 12.8.1, and PyTorch 2.11.0+cu128. The CUDA build
targets H200 (`sm_90`) and RTX PRO 6000 (`sm_120`); there is no automatic
Torch/CUDA profile selection.

Install one model explicitly:

```bash
bash install.sh --model openvla
```

Install all five model environments:

```bash
bash install.sh --model all
```

The accepted models are `openvla`, `pi0`, `pi05`, `gr00t`, `wan_oft`, and
`all`. Every invocation creates or updates the shared `latency` Conda
environment for benchmark training, teacher rollout, dataset export, and
development tools. Each selected model also creates a separate
`starvla_rl_games_<model>` environment. Both environment types install Flappy,
Demon Attack, and Deadly Corridor. Model-specific packages such as
Transformers, Diffusers, JAX, and Ray are resolved independently inside the
model environment; there is no dependency lock file.

WanOFT checkpoints from the Flappy Bird, Demon Attack, and Deadly Corridor
StarVLA pipelines can be evaluated with temporal multi-image input. See
[docs/starvla_wan_oft_eval.md](docs/starvla_wan_oft_eval.md) for the three
configs, checkpoint command, and raw-frame history contract.

To adapt a new finite-action Gymnasium task through latency-aware Sample
Factory training, teacher rollout, basic single-image OpenVLA SFT, and final
`latency_bench` evaluation, see
[docs/gymnasium_adapter.md](docs/gymnasium_adapter.md). The guide defines the
single task contract, user-owned design decisions, generated train/validation
handoff, commands, and current first-release support boundary.

`--tier use` omits development/test packages from the selected model
environment. The shared `latency` environment always includes the data and
teacher-training tools used by the experiment scripts.

## Environment variables

Create a local `.env` and set the services used by your experiments:

```bash
HF_TOKEN=...
HF_PROFILE_REPO=latency-sensitive-bench/profiles
HF_PROFILE_REPO_TYPE=dataset
WANDB_API_KEY=...
```
