# Collect teacher demonstrations

`collect_rollouts(config)` collects student observations paired with the commands
issued by a teacher. It writes canonical Parquet that the existing LeRobot
converter can consume directly. Environment and policy factories use the same
contracts as [external components](custom_components.md).

```python
from latency_bench.core.config import load_config
from latency_bench.data.rollout import collect_rollouts
from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot

result = collect_rollouts(load_config("collect.yaml"))
manifest = convert_canonical_rollout_to_lerobot(
    result["output_dir"], "outputs/student_data", action_carrier="native",
)
```

Use the existing environment, policy, executor, latency, scheduler, evaluation
and logging configuration. Collection supports paused and simulated execution,
including batched simulation and simulated inference workers. Realtime data
collection is not supported. Generic collection currently requires single-action
teacher outputs. Existing task-specific chunk recipes retain their own labels.

Add the following collection settings:

```yaml
collection:
  student_observation:
    image_key: null
    state_key: null
    state_source: metadata
  val_fraction: 0.1
  split_seed: 42
```

`image_key: null` selects raw RGB. The collector uses existing raw RGB metadata,
or calls `render_game_frame()` in the process that owns the environment, before
that environment advances. If the teacher observation is a mapping that already
contains the student image, set `image_key` to that key instead. Images must use
an RGB layout accepted by the existing image writer.

`state_key: null` omits student state. To include state, select a key from
`Observation.metadata` or `Observation.data` with `state_source: metadata` or
`data`. For a standard Gymnasium adapter, use `state_key: gymnasium_state` and
declare `env.state_space.labels`. State selection does not change teacher input.

For Gymnasium and custom environments, declare `env.task_name`, `base_prompt`
and the existing action contract. Discrete tasks provide `action_order` and
`action_map`. Continuous tasks provide `action_space` with `type: box`, `shape`,
`dtype`, `labels`, `low` and `high`. The collector preserves native command units.
Task metadata records the selected student state, not all privileged teacher state.
The generic prompt is `env.base_prompt`; sampled future latency is not appended
as student input.

## Sample alignment

The collector snapshots student inputs when the teacher request is submitted.
It matches the teacher response by episode and request identity. The action
label is the command issued for that source observation, after model decoding.
An older action executing during latency does not replace this label.

Rewards cover the raw environment steps from the source decision until the next
observation boundary or episode end. Requests without a teacher response do not
produce fabricated labels. Command admission and application are execution facts,
not implicit dataset filters. Reused buffers, asynchronous responses and episode
resets retain their source identities.

## Output and defaults

The output directory contains `train.parquet`, `val.parquet`, `metadata.json`
and `resolved_config.yaml`. Parquet embeds image bytes. Metadata describes action
layout, clocks, teacher source, student inputs, split counts and label provenance.
The return value contains `output_dir` and `metadata`.

Episodes are assigned to splits using `val_fraction` and `split_seed`. The defaults
are `0.0` and `0`, respectively. An empty validation split remains an empty valid
Parquet file. No training rows are copied into validation.

Collection uses `evaluation.episodes`, `evaluation.max_env_steps` and
`executor.parallel_envs`. The step budget is in raw frames. Generic collection defaults
to one pass and `filter_preset: none`. Optional `episode_return_gt` and
`episode_acceptance` use the existing episode filters. `max_attempts` bounds
attempts when filtering is enabled. Rejected episodes do not count toward the
requested accepted episode count. `attempted_episodes` counts attempts processed
for acceptance before that target; `next_attempt_idx` also includes allocated slots.

`two_pass: true` first probes episodes without student images, then replays the
accepted seeds. Use this only when the environment and teacher reproduce their
trajectories from reset seeds. `replay_num_envs` defaults to the configured parallel
environment count. `probe_only: true` requires two-pass collection and returns
`probe_state` instead of a completed dataset.

Images and rows are retained for active episodes, then written as episode shards.
`image_writer_workers` defaults to `0`; `row_buffer_size` defaults to `256` pending
image writes. `export_image_size` defaults to `null`, preserving source resolution.
A shard becomes complete only after its images are written. Write failures
propagate; they do not publish completed dataset metadata. Resume uses the saved
probe state and completed shards. Use the same checkpoint and collection settings
when resuming. A completed export should use a new output directory for a new run.
Complete diagnostic traces are optional and follow the existing logging settings.

## Sample Factory reference adapter

The existing SF export CLI parses the checkpoint and constructs the SF execution
adapter. It calls the same `collect_dataset` and `finalize_dataset` implementation
as the generic entrypoint. The common collector owns split assignment, acceptance,
probe/replay state, sample construction and writing. SF owns inference, recurrent
slots and environment stepping. Its environment already applies latency, so the
collector does not add another latency wrapper.

SF probe/replay retains fixed GPU batch shapes, inactive slots, raw RGB disabled
during probe, existing image/history recipes and its GPU-history resume restriction.
Use the same probe and replay batch size for fixed-batch GPU history collection.
The existing CLI arguments and publication controls remain available.

New datasets record `label_provenance.semantics: teacher_issued_command`.
The previous ordinary continuous-action SF path could replace the teacher command
with `applied_action`. This path now records the issued command. Published older
data remains readable, and is not relabeled or rewritten by collection.


## StarVLA training handoff

Convert the completed canonical directory directly:

```python
from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot

manifest = convert_canonical_rollout_to_lerobot("dataset", "lerobot/train")
```

The converter writes the training dataset, its `train__val` companion, and
`manifest.json` in each directory. It also writes `latency_prompt_map.json` and
`_generated_mixtures/train.json`. The returned manifest identifies these files,
the task contract, active action dimensions, carrier padding and state statistics.
An empty validation split stays empty.

Use `datasets.vla_data.dataset_py: lerobot_datasets` with the StarVLA reference
loader. Set `data_root_dir` to `lerobot`, `data_mix` to `train`, and
`custom_mixtures_path` to the path returned in the manifest. Copy
`gymnasium_task` to `gymnasium_task_contract`, `active_action_dim` unchanged, and
`uses_state` to `include_state`. These fields describe the collected student
inputs. Privileged teacher state is not an additional student input.

The converter writes raw state values and records source training min/max in
the Gymnasium handoff manifest. The Gymnasium training loader applies
its existing min/max transform using dataset statistics. Its language field is
`annotation.human.action.task_description`, mapped to `task_index` in the metadata.

For generic Gymnasium training, StarVLA's `prepare_data` saves the actual training
mixture statistics in the run's `dataset_statistics.json`. Validation binds those
same parameters before reading batches. Checkpoint and resident inference use
that file and the training DataConfig. Each path applies the state transform once.
The manifest retains task, dimension and source information; its min/max values
do not control model inference.

A new training run fits its own statistics. With `trainer.is_resume: true`, the
trainer selects the checkpoint before constructing loaders and restores that
run's statistics. Missing statistics, incompatible dimensions or a different
training DataConfig stop resume. A run in another directory copies the selected
statistics; resume in the original directory preserves the existing file.

The existing Gymnasium min/max transform does not clip values outside the
training range. History tensor-only transforms and native q99 transforms retain
their own behavior. Inference resolves the DataConfig from the saved robot type
or training mixture. Minimal inference bundles can supply that identity through
their explicit task manifest. Exported Gymnasium bundles retain `robot_type` so
they do not need the original training dataset directory.

The converter preserves source image resolution and records image geometry in
`meta/info.json`. The reference loader resizes model images to 224 × 224. `action_carrier="native"`
retains the action width. `"bridge"` pads to seven dimensions and rejects wider
actions; `active_action_dim` identifies the unpadded dimensions.

Balance and MIKASA reference recipes use
`latency_bench.integrations.starvla.task_contracts.prepare_lerobot` to prepare
existing LeRobot data. They write `task_contract.json` and training-derived
statistics for the `latency_bench_tasks` loader. Validation and DAgger datasets
reuse the specified training contract. Their existing multi-view, chunk and mask
recipes still define the labels. Generic conversion does not generate those labels.

Canonical data remains usable by external training code without StarVLA.
