# Teacher dataset recipes

Run these Bash entries from the repository root with the intended Python environment:

```bash
bash scripts/experiments/memory/rollout.sh flappy
bash scripts/bash_scripts/rollout_data/flappy_fixed_l2.sh
bash scripts/bash_scripts/rollout_data/flappy_fixed_l2_sharded.sh
bash scripts/bash_scripts/rollout_data/rollout_latency024_three_games.sh all
bash scripts/experiments/memory/rollout_inverted_pendulum_latency024.sh all
```

The Bash files select recipes and order the collection and publication stages.
`configs/experiments/memory/collect/` records checkpoints, clocks, episode budgets,
filters, probe shards, and publication destinations. Edit the YAML for the intended
run. `PYTHON` selects the interpreter; GPU visibility and CUDA/compiler setup belong
to the launch environment.

The memory teacher recipes resolve one latency/seed row from the manifest produced
by `train_small_model.sh`. The cross-task recipes use the generated memory
small-model manifests. Fixed-latency game recipes name their HF teacher roots and
select the best checkpoint. Replace their source bindings when using another release.

For a manual stage:

```bash
python -m scripts.experiments.memory.collect \
  --config configs/experiments/memory/collect/flappy_fixed_l2_sharded.yaml --stage collect
python -m scripts.experiments.memory.collect \
  --config configs/experiments/memory/collect/flappy_fixed_l2_sharded.yaml --stage publish
```

Sharded collection starts independent probe processes, waits for completion,
merges their accepted seeds in shard order, and replays the merged plan once.
Generated configs, probe logs, and publication receipts are stored beside the
configured output in a directory with the `_collection` suffix.

Defend-the-Line L0 selects whole episodes from its pinned source dataset. It does
not require a teacher checkpoint. Its recipe records the source revision and
train/validation episode counts.

Standalone Air Raid and ghost-trail collection remain available through:

```bash
bash scripts/bash_scripts/rollout_data/air_raid.sh
bash scripts/bash_scripts/generate_ghost_trail_data.sh
```

Set the local teacher root in the corresponding example YAML first.
`REPLAY_RAW_FROM_HF=1` enables the paired raw-RGB replay in the ghost-trail entry.
Its YAML records the published ghost dataset revision and output path.
