# New Machine Profile And Training Handoff

This is the shortest path for moving to a new machine:

1. Install the benchmark/model environment.

```bash
cd /path/to/latency-sensitive-bench
bash install.sh --model wan_oft
```

Use `--model openvla`, `pi0`, `pi05`, or `gr00t` for a different StarVLA model env.

2. Resolve and download the matching published profile automatically.

```bash
cd /path/to/latency-sensitive-bench
latency-bench profile resolve --hardware auto --revision latest
```

The resolver does all of this:

- fingerprints the local GPU with `nvidia-smi` and torch
- maps the GPU name to a configured device id, for example `rtx4090`
- hashes the GPU UUID into the local `instance_id`
- lists `profiles/*/*/*/*/profile.json` in `latency-sensitive-bench/profiles`
- downloads small profile manifests for compatible device/workload refs
- prefers exact same GPU UUID instance, then same GPU model/runtime metadata, then newest revision

It prints:

```text
PROFILE_REF=<device>/<instance>/<workload>/<revision>
PROFILE_PATH=<local-cache>/profiles/<profile-ref>/profile.json
PROFILE_SCORE=<score>
PROFILE_MATCH_REASONS=<comma-separated reasons>
```

To force a known ref instead of auto selection:

```bash
PROFILE_REF='rtx4090/<instance-id>/openvla-flappy/fit-0-4' \
  bash scripts/bash_scripts/cross_device/download_profile.sh
```

3. Train in StarVLA from HF-backed assets.

The StarVLA launcher already downloads/converts the HF rollout dataset, downloads base/init checkpoints, and can resume from a synced HF checkpoint repo. Example:

```bash
cd /path/to/starVLA
conda activate starvla_rl_games_wan_oft

bash scripts/run_flappy_wan_oft_pipeline.sh \
  --max-train-steps 2000 \
  --upload-repo latency-sensitive-bench/wanoft_flappy_200ep \
  --run-id wan_oft_flappy_l3_new_machine
```

For direct launcher usage, keep downloads online during setup and let the launcher materialize the dataset:

```bash
cd /path/to/starVLA
conda activate starvla_rl_games_openvla

python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=bridge \
  run_id=openvla_flappy_new_machine \
  dataset.source_hf=latency-sensitive-bench/memory-rollouts \
  dataset.config_name=flappy_fixed_latency_3_200ep_7k2steps \
  paths.dataset_local_dir=data/flappy_fixed_latency_3_200ep_7k2steps \
  checkpoint.sync.enabled=true \
  checkpoint.sync.repo_id=latency-sensitive-bench/openvla_flappy_new_machine
```

4. Use the downloaded profile for profile-shaped benchmark evaluation.

```bash
cd /path/to/latency-sensitive-bench
bash cross_device.sh sim \
  --config configs/experiments/cross_device.yaml \
  --profile-ref 'rtx4090/<instance-id>/openvla-flappy/fit-0-4' \
  --method temporal \
  --latency-seeds 0 1 2 \
  --local-profile \
  --resume
```

The important boundary is: StarVLA supervised training consumes rollout datasets and checkpoints; latency profiles are benchmark artifacts used by `latency-sensitive-bench` eval/simulation. A training recipe can call `latency-bench profile resolve` first and inject the printed `PROFILE_PATH` into eval/profile-aware steps.
