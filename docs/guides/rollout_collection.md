# Collect and convert teacher demonstrations

Collection pairs student observations with the teacher's issued commands.
Generic collection writes canonical Parquet data that the LeRobot converter can consume.
Use your existing environment and teacher through the [component factories](custom_components.md).
Install data dependencies through the [installation guide](../installation.md).

## Run the complete CPU example

From the repository root, run:

```bash
export PYTHONPATH="$PWD/configs/examples/custom_env${PYTHONPATH:+:$PYTHONPATH}"
python -m latency_bench collect --config configs/examples/custom_env/collect.yaml
python -m latency_bench convert --config configs/examples/custom_env/convert.yaml
```

The example collects three MountainCar episodes with a 50-step limit and seed 7.
It uses 50 environment FPS, 10 observation FPS, and fixed 100 ms delay.
Student inputs contain RGB without state. The constant policy supplies action labels.
The [external component guide](custom_components.md) explains each output and runs the same example outside the checkout.

## Select teacher and student inputs

Collection uses the ordinary `env`, `policy`, `executor`, `latency`, `scheduler`, `evaluation`, and `logging` sections.
Generic collection supports paused and simulated execution, including batched simulation and inference workers.
It requires single-action teacher outputs. Realtime collection is not supported.
Task recipes define their own chunk labels.

Add student input and split settings:

```yaml
collection:
  student_observation:
    image_key: null
    state_key: null
    state_source: metadata
  val_fraction: 0.1
  split_seed: 42
```

`image_key: null` selects raw RGB.
The collector uses existing RGB metadata or renders in the environment's process before the environment advances.
If `Observation.data` already contains the student image, select that mapping key with `image_key`.
Use an RGB layout accepted by the image writer.

`state_key: null` omits student state.
To include state, select a key from `Observation.metadata` or `Observation.data`.
Set `state_source` to `metadata` or `data` respectively.
For a standard Gymnasium adapter, use `state_key: gymnasium_state` and declare `env.state_space.labels`.
Student selection does not change teacher input.

Declare `env.task_name`, `env.base_prompt`, and the task's action contract when required by your adapter.
Discrete tasks supply `action_order` and `action_map`.
Continuous tasks supply `action_space` with type, shape, dtype, labels, and native bounds.
The collector preserves native command units.
The generic prompt uses `env.base_prompt`. It does not append future sampled latency.

## Understand sample alignment

The collector snapshots student inputs when it submits the teacher request.
It matches the response by episode and request identity.
The label is the decoded teacher command for that source observation.
An older action executing during the delay does not replace the label.

Rewards cover raw steps until the next observation boundary or episode end.
A request without a teacher response produces no label.
Command admission and application remain separate execution facts.
Buffer reuse, asynchronous responses, and resets preserve the source pairing.
Privileged teacher state is not automatically included in student inputs.

## Read canonical outputs

The output directory contains:

```text
train.parquet
val.parquet
metadata.json
resolved_config.yaml
```

Parquet rows embed image bytes.
Metadata describes actions, clocks, teacher source, student inputs, splits, and label provenance.
The Python result contains `output_dir` and `metadata`.

Split assignment uses whole episodes with `val_fraction` and `split_seed`.
Their defaults are `0.0` and `0`.
An empty validation split remains an empty valid Parquet file.
The collector does not copy training rows into validation.

`evaluation.episodes` sets the episode target.
`evaluation.max_env_steps` sets the raw-step limit.
`executor.parallel_envs` sets concurrent environment slots.

## Filter, replay, and resume

Generic collection defaults to one pass with `filter_preset: none`.
Optional `episode_return_gt` and `episode_acceptance` select episode filters.
`max_attempts` bounds attempts when you enable filtering.
Rejected episodes do not count toward the accepted episode target.

`two_pass: true` probes without student images, then replays accepted seeds.
Use this only when reset seeds reproduce environment and teacher trajectories.
`replay_num_envs` defaults to the configured environment count.
`probe_only: true` requires two-pass collection and returns `probe_state` instead of a completed dataset.

The writer saves completed episodes as shards.
`image_writer_workers` defaults to `0`. `row_buffer_size` defaults to `256` pending image writes.
`export_image_size: null` preserves source resolution.
A shard becomes complete after the writer saves its images.
Write failures propagate without completed dataset metadata.

Resume uses saved probe state and completed shards.
Keep the checkpoint and collection settings unchanged when resuming.
Use a new output directory for a new dataset.
Diagnostic traces follow the ordinary logging settings.

## Convert canonical data

Save a converter configuration:

```yaml
conversion:
  format: lerobot
  input_dir: outputs/my_task/raw
  output_dir: outputs/my_task/lerobot
  action_carrier: native
```

Run conversion:

```bash
python -m latency_bench convert --config convert.yaml
```

Conversion writes the training dataset, its `lerobot__val` companion, and `manifest.json` in each directory.
It also writes `latency_prompt_map.json` and a generated mixture under `_generated_mixtures/`.
The manifest identifies paths, task contracts, active action dimensions, and state statistics.
An empty validation split stays empty.

`action_carrier: native` preserves action width.
`bridge` pads actions to seven dimensions and rejects wider actions.
`active_action_dim` identifies unpadded dimensions.
The converter preserves source image resolution. The reference loader resizes model images to 224 × 224.

An existing output fails unless you explicitly pass `--force`.
Input, normalization source, and output paths must not overlap.
Task conversion uses `conversion.task` and the task's existing contract name.
Use `normalization_source` when the task requires an existing training contract.

## Pass data to StarVLA or your trainer

For the generic StarVLA loader, map converter metadata to these training fields:

| Training field | Value or source |
| --- | --- |
| `datasets.vla_data.dataset_py` | `lerobot_datasets` |
| `data_root_dir` | Parent of the converted training directory |
| `data_mix` | Converted training directory name |
| `custom_mixtures_path` | Generated mixture path from the manifest |
| `gymnasium_task_contract` | Manifest `gymnasium_task` |
| `active_action_dim` | Manifest `active_action_dim` |
| `include_state` | Manifest `uses_state` |

The converter records raw student state and source statistics.
Generic Gymnasium training saves the actual mixture statistics in the run's `dataset_statistics.json`.
Validation and inference use the same training DataConfig and statistics.
Each path applies its state transform once.
The existing min/max transform does not clip values outside the training range.

A new run fits its own statistics.
Resume restores statistics from the selected checkpoint run before constructing loaders.
Missing statistics, incompatible dimensions, or another training DataConfig stop resume.
Exported inference bundles retain the identity needed to select that DataConfig.

Balance and MIKASA use task-specific preparation and the `latency_bench_tasks` loader.
Their contracts define views, chunks, masks, and frozen training-derived normalization.
Generic conversion does not generate those task labels.
Canonical data also remains usable by your own trainer.

## Use reference task collectors

Sample Factory uses `policy.type: sample_factory` with the ordinary model source and collection settings.
Its native collector preserves recurrent slots and existing history labels.
For fixed-batch GPU history collection, keep probe and replay batch sizes equal.

| Task | `collection.label_mode` | Budget |
| --- | --- | --- |
| Balance | `zero_full`, `profile_conditioned` | `row_budget`, horizon 8 or 16 |
| MIKASA | `successful_h1` | Successful `evaluation.episodes` |
| MIKASA | `closed_loop_h8`, `prefix_h8` | `train_targets` and `validation_targets` valid action heads |

Student-controlled collection selects the student under `policy` and the teacher under `collection.teacher`.
MIKASA saves native NPZ episodes and task metadata.
Balance and generic collection use canonical Parquet.
Follow the [MIKASA](../cookbooks/mikasa.md) or [Balance](../cookbooks/balance.md) cookbook for complete task commands.

New canonical data records `label_provenance.semantics: teacher_issued_command`.
Older continuous-action SF datasets can contain applied-action labels.
Collection does not rewrite those published datasets.

## Use the Python API

```python
from latency_bench.core.config import load_config
from latency_bench.data.rollout import collect_rollouts
from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot

result = collect_rollouts(load_config("collect.yaml"))
manifest = convert_canonical_rollout_to_lerobot(
    result["output_dir"], "outputs/student_data", action_carrier="native",
)
```

The collection CLI prints `OUTPUT_DIR` and `COLLECTION_STATUS`.
The conversion CLI prints `OUTPUT_DIR` and `CONVERSION_STATUS`.
A failed stage returns a nonzero exit status.
