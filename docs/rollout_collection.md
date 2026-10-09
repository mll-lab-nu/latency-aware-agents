# Collect teacher demonstrations

`collect_rollouts(config)` collects student observations paired with the commands
issued by a teacher. It writes canonical Parquet that the existing LeRobot
converter can consume directly. Environment and policy factories use the same
contracts as [external components](custom_components.md).

```python
from latency_bench.core.config import load_config
from latency_bench.data.rollout import collect_rollouts
from latency_bench.data.lerobot_export import convert_canonical_rollout_to_lerobot

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

Collection uses `evaluation.eval_episodes`, `eval_max_steps` and
`eval_parallel_envs`. The step budget is in raw frames. Generic collection defaults
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
