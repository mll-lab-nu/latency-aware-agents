# Latency Sensitive Bench

## Environment

Create a local `.env` first, set these values:

```bash
HF_TOKEN=...
HF_PROFILE_REPO=latency-sensitive-bench/profiles
HF_PROFILE_REPO_TYPE=dataset
WANDB_API_KEY=...
```

## Install

Install the repository by running:

```bash
bash install.sh
```

If you want to train a VLA model, also install the `starVLA` RL-Games stack:

```bash
bash starVLA/examples/rl_games/install/bootstrap.sh --split-envs
```
