# Teacher dataset recipes

Run these entries from the repository root with the selected simulator dependencies installed:

```bash
bash scripts/experiments/memory/rollout.sh flappy
bash scripts/bash_scripts/rollout_data/flappy_fixed_l2.sh
bash scripts/bash_scripts/rollout_data/flappy_fixed_l2_sharded.sh
bash scripts/bash_scripts/rollout_data/rollout_latency024_atari.sh all
bash scripts/experiments/memory/rollout_inverted_pendulum_latency024.sh all
```

The Bash entries select configurations and order collection.
`configs/experiments/memory/collect/` records teacher sources, clocks, budgets, filters, and probe shards.
Use an overlay to select your checkpoint and output directory.
`PYTHON` selects the interpreter.
The launch environment owns GPU visibility and CUDA setup.

Memory recipes resolve the selected latency and seed from their teacher-training manifest.
Cross-task recipes use the generated memory small-model manifests.
Fixed-latency recipes download their declared teacher source and select the configured checkpoint.

## Run one collection

```bash
python -m scripts.experiments.memory.collect \
  --config configs/experiments/memory/collect/flappy_fixed_l2_sharded.yaml
```

Sharded collection starts independent probe processes.
It merges accepted seeds in shard order and replays the merged plan once.
The configured output contains canonical train/validation data and metadata.
A sibling directory with the `_collection` suffix contains generated configurations and probe logs.

## Other collection paths

```bash
bash scripts/bash_scripts/rollout_data/air_raid.sh
bash scripts/bash_scripts/generate_ghost_trail_data.sh
```

Set the local teacher root in the matching example YAML.
`REPLAY_RAW_FROM_HF=1` enables paired raw-RGB replay from the pinned ghost dataset.
The [collection guide](../../../docs/guides/rollout_collection.md) describes alignment, conversion, and local outputs.
