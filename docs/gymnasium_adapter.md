# Adapt a Gymnasium task to the latency-sensitive pipeline

This guide is the release path for adapting a new finite-action Gymnasium task
without changing the existing Flappy Bird, Demon Attack, or Deadly Corridor
interfaces:

```text
fixed-latency Sample Factory teacher
    -> teacher rollout (train.parquet + val.parquet)
    -> native-action LeRobot train/validation datasets
    -> single-image OpenVLA SFT
    -> latency_bench evaluation
```

The generic adapter preserves the configured Gymnasium constructor arguments,
native observation, native action values, reward, `terminated`, and `truncated`.
It adds a raw-tick latency clock and an explicitly ordered finite action
catalog. It does not guess task semantics or silently preprocess an
environment.

## 1. Release scope

The first release deliberately supports the smallest complete path:

| Stage | Supported generic Gymnasium path |
| --- | --- |
| Small-model training | Existing Sample Factory/APPO path with zero, fixed, normal, uniform, and their existing per-episode mixed latency modes. |
| Small-model evaluation | Simulated `latency_bench` evaluation using `policy.type: gymnasium_sf`. |
| Teacher data | Deterministic checkpoint rollout to canonical train/validation parquet. |
| StarVLA data | Native-width finite-action LeRobot datasets with a generated train/validation mixture. |
| StarVLA training | OpenVLA, `init=scratch`, `mode=single`, one current RGB image, native action width, no state, no online environment evaluation during SFT. |
| Final evaluation | Disk-backed OpenVLA checkpoint evaluation through `latency_bench` and the original Gymnasium task contract. |

The following are **not first-release claims**:

- Sample Factory observation-memory or policy frame-stack ablations;
- multi-image, KV-memory, WanOFT, Pi, GR00T, or cross-task StarVLA training;
- seven-dimensional bridge carriers;
- continuous, `MultiDiscrete`, or `MultiBinary` StarVLA control;
- generic GPU-batched simulation/rendering;
- automatic action discretization, observation preprocessing, reward shaping,
  frame skipping, or old-Gym API conversion.

`env.frame_stack` remains `1` in this release. It is a fixed raw-RGB buffer
setting required by the shared VLA input interface, not a supported
small-model memory experiment.

### Fast-path requirements

A new task fits the release path when:

- it implements the Gymnasium five-value API;
- its controller can be represented by a finite, explicitly ordered action
  catalog;
- each catalog value is JSON-stable and can be passed directly to
  `env.step(action)`;
- its unchanged observation space is accepted by Sample Factory;
- `render_mode: rgb_array` returns an HWC RGB image;
- one configured `step()` has a documented meaning as one benchmark raw tick.

If an action requires a NumPy array, tuple, class instance, or other value that
does not round-trip through JSON, register a task-owned Gymnasium wrapper whose
public action catalog is JSON-stable and reconstruct the native value inside
that wrapper. A truly continuous controller needs a new action head, loss, and
decoder and is outside this path.

## 2. Install the project and task package

Clone the formal submodule and install the OpenVLA environment:

```bash
git clone --recurse-submodules https://github.com/ZihanWang314/latency-sensitive-bench.git
cd latency-sensitive-bench
bash install.sh --tier dev --model openvla
```

The small model, rollout exporter, and converter run in `latency`. OpenVLA SFT
runs in `starvla_rl_games_openvla`. Install the same task package/version in
both environments:

```bash
conda activate latency
python -m pip install -e /absolute/path/to/acme-gym

conda activate starvla_rl_games_openvla
python -m pip install -e /absolute/path/to/acme-gym
```

Use the same environment ID, registration package, wrapper, and constructor
arguments at every stage.

## 3. Define one task contract

Users configure one public schema under `env`. The MountainCar reference is:

```yaml
env:
  name: gymnasium
  task_name: mountain_car
  env_id: MountainCar-v0
  registration_imports: []
  make_kwargs:
    render_mode: rgb_array
  env_fps: 30
  obs_fps: 10
  frame_stack: 1
  noop_action: coast
  action_map:
    push_left: 0
    coast: 1
    push_right: 2
  action_order:
    - push_left
    - coast
    - push_right
  base_prompt: >-
    Drive the car up the right hill. Choose exactly one action from:
    push_left, coast, push_right.
```

Reference configs:

- [`gymnasium_mountain_car_latency_bench_eval.yaml`](../configs/examples/gymnasium_mountain_car_latency_bench_eval.yaml)
- [`gymnasium_mountain_car_small_model_train.yaml`](../configs/examples/gymnasium_mountain_car_small_model_train.yaml)
- [`gymnasium_mountain_car_small_model_eval.yaml`](../configs/examples/gymnasium_mountain_car_small_model_eval.yaml)
- [`gymnasium_mountain_car_starvla_eval.yaml`](../configs/examples/gymnasium_mountain_car_starvla_eval.yaml)

### Field contract

| Field | Meaning |
| --- | --- |
| `name` | Integration dispatch key. It must remain `gymnasium`. |
| `task_name` | Stable semantic identity used by checkpoints, datasets, prompts, and output paths. |
| `env_id` | Exact ID passed to `gymnasium.make`. |
| `registration_imports` | Modules imported in listed order before `gymnasium.make`. Use this for packages that register environments as an import side effect. |
| `make_kwargs` | Passed unchanged to `gymnasium.make(env_id, **make_kwargs)`. |
| `env_fps` | Number of configured environment `step()` calls per simulated second. This defines the raw-tick unit. |
| `obs_fps` | Policy decision frequency. `env_fps / obs_fps` must be integral. |
| `frame_stack` | Fixed to `1` for the single-image first-release path. It does not alter the small-model observation. |
| `noop_action` | Semantic label of the real task no-op/default action. |
| `action_map` | Semantic label to exact native Gymnasium action value. |
| `action_order` | Canonical policy ID and StarVLA logit order. ID `i` always means `action_order[i]`. |
| `base_prompt` | Stable task instruction. It should name the legal semantic actions. |

For a separately packaged task:

```yaml
env:
  name: gymnasium
  task_name: acme_balance
  env_id: AcmeBalance-v0
  registration_imports:
    - acme_gym.envs
```

Every generic stage imports the listed modules in order and then calls the
same `gymnasium.make`. Do not create a second handwritten action or prompt
registry for StarVLA.

## 4. Decisions the task author must make

The framework preserves a contract; the task author must decide whether it is
scientifically correct.

| Decision | Required task-specific work |
| --- | --- |
| Raw tick | Document what one configured `step()` advances. Disable native action repeat/frame skip or define the clock at the aggregate rate. |
| Observation rate | Choose `obs_fps` and verify that `env_fps / obs_fps` is an integer. |
| No-op | Demonstrate that it is safe at reset and while a delayed action is pending. Do not assume native action `0` is a no-op. |
| Action catalog | Choose a sufficient finite set, assign semantic labels, and freeze `action_order` before RL training. |
| Observation preprocessing | If needed, implement one registered Gymnasium wrapper and reuse it in all stages. |
| Reward/episode boundary | Decide whether native reward, shaping, termination, and time-limit truncation match the experiment. |
| Prompt | Describe the goal and legal actions without renaming or reordering them. |
| Rollout filter | Start with `none`; add a task-specific filter only after inspecting returns, failures, class balance, and continuity. |
| Training budget | Choose RL steps, rollout episodes, SFT steps, batch size, learning rate, and checkpoint cadence. |

Changing any row after data collection creates a different task contract.
Train a new teacher and regenerate the data instead of reusing an incompatible
checkpoint.

## 5. Raw-tick latency semantics

The clock is:

```text
raw tick duration       = 1000 / env_fps ms
raw ticks per decision  = env_fps / obs_fps
fixed latency in ticks  = ceil(fixed_latency_ms * env_fps / 1000)
```

MountainCar uses:

```text
env_fps                 = 30
obs_fps                 = 10
raw ticks per decision  = 3
fixed_latency_ms        = 100
fixed latency           = 3 raw ticks
```

One Sample Factory policy decision therefore advances three native
MountainCar steps. The scheduler queues the newly issued catalog action in
raw-tick time, applies the configured hold/ordering policy, and accumulates
reward over those three native steps.

## 6. Gate 0: verify zero-latency parity

Before training, compare two seeded trajectories:

1. Construct the exact native/wrapped Gymnasium task and apply a fixed semantic
   action trace through `action_map`.
2. Construct the generic latency path with the same seed and
   `latency.method: zero`, and issue the same trace.

When `env_fps / obs_fps = K`, hold each reference action for the same `K`
native steps before comparing the next policy boundary. Compare:

- native action at every raw tick;
- observation value, dtype, and shape;
- summed reward per decision and total return;
- `terminated` and `truncated` separately;
- terminal raw tick;
- RGB render output;
- native `info` after excluding benchmark-owned metadata.

The repository's MountainCar reference is
[`tests/envs/test_gymnasium_env_adapter.py`](../tests/envs/test_gymnasium_env_adapter.py).
A stochastic task also needs repeated same-seed trajectory checks.

A construction smoke test is useful but does not replace parity:

```bash
python -m latency_bench.run \
  --config configs/examples/gymnasium_mountain_car_latency_bench_eval.yaml \
  --episode-start 0 \
  --episode-count 2
```

## 7. Train and evaluate the small model

Copy the complete training example and replace only the task contract, output
names, latency, and ordinary Sample Factory hyperparameters:

```bash
cp configs/examples/gymnasium_mountain_car_small_model_train.yaml \
  configs/examples/my_task_small_model_train.yaml
```

The reference fixed-latency block is:

```yaml
latency:
  method: fixed
  fixed_latency_ms: 100.0
  add_latency_info: false
```

Train:

```bash
conda activate latency
python -m training.train \
  --config configs/examples/my_task_small_model_train.yaml
```

The experiment directory is:

```text
<backend.train_dir>/<experiment.name>/
  config.json
  checkpoint_p*/checkpoint_*.pth
```

`config.json` contains the exact Gymnasium ID, constructor kwargs,
registration imports, action labels/values/order, no-op ID, prompt, and clock.
The teacher exporter reconstructs the task from this checkpoint contract.

Evaluate one checkpoint at the training latency:

```bash
python -m latency_bench.run \
  --config configs/examples/gymnasium_mountain_car_small_model_eval.yaml \
  --checkpoint-path \
    outputs/examples/gymnasium_mountain_car/checkpoints/gymnasium_mountain_car_fixed_l3/checkpoint_p0/checkpoint_XXXXXXXXX_YYYYY.pth \
  --episode-start 0 \
  --episode-count 5
```

The policy type is `gymnasium_sf`. Its raw policy ID is decoded through the
checkpoint's `action_order` and the unchanged `action_map`.

This release does not promise generic observation-memory, RNN, frame-stack,
GPU-render, or environment-specific ablation configs. Such experiments require
task-owned implementation and validation.

## 8. Export the teacher rollout

Start with no task-specific filter and one current decision image:

```bash
conda activate latency
python scripts/rollout_data/export_sf_teacher_dataset.py \
  --checkpoint-root \
    outputs/examples/gymnasium_mountain_car/checkpoints/gymnasium_mountain_car_fixed_l3 \
  --output-dir \
    outputs/examples/gymnasium_mountain_car/teacher_fixed_l3 \
  --max-episodes 200 \
  --max-steps-per-episode 200 \
  --val-fraction 0.1 \
  --num-envs 1 \
  --replay-num-envs 1 \
  --device cpu \
  --deterministic \
  --filter-preset none \
  --context-window 1 \
  --export-image-size 224,224
```

Do not pass `--env-name` for a Gymnasium checkpoint. The semantic task name is
part of the checkpoint contract; the exporter rejects attempts to relabel it.

The result is:

```text
teacher_fixed_l3/
  train.parquet
  val.parquet
  metadata.json
  filter_report.json
```

Important details:

- rows are policy decision steps, not raw ticks;
- train/validation assignment is by episode;
- `context-window` stays `1` for this release;
- `clean_v1` and `contiguous_v1` contain assumptions from existing tasks and
  are not generic quality filters;
- `metadata.json` is the canonical manifest for the teacher export. It records
  `integration_name: gymnasium`, `task_name`/`env_name`, action layout and
  labels, prompt, clock, checkpoint task contract, split counts, and filter
  report.

Inspect the following before conversion:

- accepted/rejected episodes and return distribution;
- no episode overlap between train and validation;
- one-hot width and class balance in exact `action_order`;
- prompt and raw-frame latency consistency;
- RGB image content and size;
- reset boundaries, terminal rows, and action continuity.

## 9. Convert train and validation data for StarVLA

Use the native action carrier only:

```bash
python scripts/rollout_data/prepare_starvla_lerobot_dataset.py \
  outputs/examples/gymnasium_mountain_car/teacher_fixed_l3 \
  third_party/starVLA/playground/Datasets/rl_games \
  mountain_car_fixed_l3 \
  --action-carrier native
```

This produces paired datasets and a generated mixture:

```text
third_party/starVLA/playground/Datasets/rl_games/
  mountain_car_fixed_l3/
    data/chunk-000/episode_000000.parquet
    meta/info.json
    meta/modality.json
    meta/episodes.jsonl
    meta/tasks.jsonl
    latency_prompt_map.json
    manifest.json
  mountain_car_fixed_l3__val/
    data/...
    meta/...
    manifest.json
  _generated_mixtures/
    mountain_car_fixed_l3.json
```

The train manifest records:

- `integration_name: gymnasium`, semantic `task_name`, and the complete
  `gymnasium_task` contract copied from the teacher checkpoint;
- exact `env_id`, registration imports, make kwargs, ordered action labels,
  native action values, no-op ID, base prompt, timing rates, and frame stack;
- `action_layout`, ordered `action_labels`, and
  `robot_type: rl_games_gymnasium_discrete`;
- `action_carrier: native`;
- `action_dim == active_action_dim == len(action_order)`;
- `obs_fps` and `obs_stride_raw_frames`;
- `validation_dataset_name` and validation counts;
- `latency_prompt_map_path` and `custom_mixtures_path`.

The validation manifest repeats the same task/action contract and points back
with `training_dataset_name`. The generated mixture JSON contains two named
entries:

```json
{
  "mountain_car_fixed_l3": [
    ["mountain_car_fixed_l3", 1.0, "rl_games_gymnasium_discrete"]
  ],
  "mountain_car_fixed_l3__val": [
    ["mountain_car_fixed_l3__val", 1.0, "rl_games_gymnasium_discrete"]
  ]
}
```

Do not edit these three handoff files independently. If the task/action contract
changes, regenerate the teacher export and both LeRobot splits.

## 10. Run basic single-image OpenVLA SFT

The public first-release combination is fixed:

```text
model=openvla
env=gymnasium
init=scratch
mode=single
action carrier=native
num_obs_frames=1
image_mode=single
KV memory=disabled
StarVLA environment evaluation=disabled
```

The recommended entry point reads the complete task contract and active action
dimension from the train manifest, so these values are not typed a second time:

```bash
cd third_party/starVLA
conda activate starvla_rl_games_openvla

bash commands/openvla/train_gymnasium_openvla.sh \
  playground/Datasets/rl_games \
  mountain_car_fixed_l3 \
  playground/Datasets/rl_games/_generated_mixtures/mountain_car_fixed_l3.json
```

The three positional values are:

| Argument | Meaning |
| --- | --- |
| dataset root | Parent directory containing the train and validation LeRobot directories. |
| dataset name | Train dataset directory and generated mixture key. |
| mixture path | Generated mixture JSON from the converter. |

The launcher fixes `model=openvla`, `env=gymnasium`, `init=scratch`, and
`mode=single`. It reads `gymnasium_task` and `active_action_dim` from
`<dataset-root>/<dataset-name>/manifest.json`, forwards optional extra Hydra
overrides after the three positional values, and derives
`datasets.vla_data.eval_data_mix` as `<dataset.converted_name>__val`, so the
generated validation mixture is selected without another user override.

For example, a short logic run is:

```bash
bash commands/openvla/train_gymnasium_openvla.sh \
  playground/Datasets/rl_games \
  mountain_car_fixed_l3 \
  playground/Datasets/rl_games/_generated_mixtures/mountain_car_fixed_l3.json \
  trainer.max_train_steps=50
```

The resolved configuration must keep `framework.action_model.action_dim` equal
to the native carrier width and `action_env_dim` equal to the active task
width. For this first-release native path, both are `N`.

Before a long run, verify one loader batch:

- train mix resolves `mountain_car_fixed_l3`;
- validation mix resolves `mountain_car_fixed_l3__val`;
- image input is one `224 x 224` current RGB frame;
- each training row resolves its prompt through LeRobot `task_index` and
  `meta/tasks.jsonl`, and that prompt matches the corresponding entry in
  `latency_prompt_map.json` used by final online evaluation;
- action tensor width and active dimension are `N`;
- ordered labels equal `env.action_order`;
- resolved config contains `rl_games.task: gymnasium`, the complete
  `rl_games.gymnasium.task_contract`, and the derived semantic
  `rl_games.gymnasium.task_name`.

The run directory is:

```text
third_party/starVLA/results/Checkpoints/<run_id>/
  config.full.yaml
  config.yaml
  checkpoints/...
  final_model/...
```

Keep `config.full.yaml` with the checkpoint. Final benchmark input setup reads
the saved model configuration; a checkpoint file alone is not the complete
experiment artifact.

## 11. Evaluate the OpenVLA checkpoint in `latency_bench`

Copy
[`gymnasium_mountain_car_starvla_eval.yaml`](../configs/examples/gymnasium_mountain_car_starvla_eval.yaml)
and replace its four artifact paths: checkpoint, saved model config, train task
manifest, and latency prompt map. Do not repeat or reorder actions under
`policy`: the generic StarVLA policy reads the canonical `env.action_order`.

Run from the repository root:

```bash
conda activate starvla_rl_games_openvla
cd /absolute/path/to/latency-sensitive-bench

python -m latency_bench.run \
  --config configs/examples/gymnasium_mountain_car_starvla_eval.yaml \
  --checkpoint-path \
    third_party/starVLA/results/Checkpoints/mountain_car_fixed_l3_openvla_sft/final_model \
  --model-config-path \
    third_party/starVLA/results/Checkpoints/mountain_car_fixed_l3_openvla_sft/config.full.yaml \
  --task-manifest-path \
    third_party/starVLA/playground/Datasets/rl_games/mountain_car_fixed_l3/manifest.json \
  --latency-prompt-map-path \
    third_party/starVLA/playground/Datasets/rl_games/mountain_car_fixed_l3/latency_prompt_map.json \
  --latency-raw-frames 3 \
  --episode-start 0 \
  --episode-count 20
```

The online path:

1. creates the same Gymnasium environment from `env_id`,
   `registration_imports`, and `make_kwargs`;
2. renders one current RGB frame;
3. selects the prompt for the scheduled raw-frame latency;
4. asks OpenVLA for native-width action scores;
5. takes the argmax over the first `len(action_order)` scores;
6. resolves that ID through `action_order` and then `action_map`;
7. schedules the exact native action in the raw-tick latency queue.

Before loading the model, disk and live policy paths compare the evaluation
task contract against both the train manifest and the saved StarVLA config.
Changing action order, native values, no-op, prompt, environment construction,
or timing fields therefore fails explicitly instead of silently changing the
experiment.

Final acceptance must compare the small model and OpenVLA under the same task
contract, seeds, raw-tick latency, hold policy, ordering policy, episode limit,
and reward/termination semantics.

## 12. Validation status and release acceptance

The generic code path and configuration composition can be checked on a local
Mac. A Mac logic check does **not** establish that a full CUDA OpenVLA SFT run
or complete online benchmark succeeds.

Before publishing results, run on the target GPU server:

1. one real StarVLA train and validation loader batch;
2. a short OpenVLA SFT smoke run that saves a checkpoint;
3. one zero-latency online episode through `latency_bench`;
4. one fixed-latency online episode through `latency_bench`;
5. the intended full-seed/full-episode experiment;
6. regression tests for the three existing environments.

Do not describe successful config composition, unit tests, or data conversion
as a completed GPU SFT or online evaluation.

## 13. Per-task release checklist

- [ ] Task package version and `env_id` are documented.
- [ ] One wrapper, if needed, is registered and reused everywhere.
- [ ] One configured `step()` has a documented raw-tick meaning.
- [ ] `env_fps / obs_fps` is integral.
- [ ] The chosen no-op is safe while an action is pending.
- [ ] `action_map`, `action_order`, and `base_prompt` are frozen.
- [ ] Native versus generic zero-latency action-trace parity passes.
- [ ] RGB rendering is HWC and visually inspected.
- [ ] Fixed-latency Sample Factory training produces a real checkpoint.
- [ ] That checkpoint evaluates with `policy.type: gymnasium_sf`.
- [ ] Teacher export begins with `filter_preset=none` and has disjoint episode
  splits.
- [ ] Teacher metadata, actions, prompts, latency, reward, images, and terminal
  rows are inspected.
- [ ] Train and validation LeRobot manifests agree on task/action metadata.
- [ ] The StarVLA loader accepts the manifest/config contract equality check.
- [ ] Generated mixture resolves both train and validation keys.
- [ ] OpenVLA config uses `env=gymnasium`, `init=scratch`, `mode=single`, one
  image, native action width, and no environment eval.
- [ ] One real GPU loader batch and short SFT checkpoint pass.
- [ ] The checkpoint decodes through the generic `latency_bench` policy.
- [ ] Final evaluation uses the matching train `task_manifest_path`; a
  deliberately reordered action catalog is rejected.
- [ ] Existing task behavior and interfaces remain unchanged.

## 14. Troubleshooting

### Gymnasium cannot find `env_id`

Install the same task package in both Conda environments. Put every required
registration module in `registration_imports` and verify that importing them in
order registers the configured ID.

### The small model selects the wrong native action

Inspect the checkpoint fields `gym_action_labels_json` and
`gym_action_values_json`. Policy ID `i` must mean `action_order[i]`, and its
native value must be `action_map[action_order[i]]`. Do not infer meaning from
mapping insertion order.

### Latency is off by the observation stride

Recompute raw ticks and decision stride separately:

```text
ceil(latency_ms * env_fps / 1000)
env_fps / obs_fps
```

`latency_raw_frames` is never a policy-decision count.

### Sample Factory rejects the observation

The generic path intentionally preserves the native observation. Implement one
task-owned registered observation wrapper, update the single `env_id` contract,
and repeat zero-latency parity. Do not add a training-only fallback.

### Teacher images are blank or malformed

Construct the environment with `render_mode: rgb_array` and inspect
`env.render()` directly. The state/vector observation is not substituted for
the VLA image.

### The data is empty or biased

Use `--filter-preset none`, inspect `filter_report.json`, returns, episode
lengths, and class balance, then implement an explicit task-specific filter if
the experiment requires one.

### StarVLA reports an unknown mixture or wrong action dimension

Check the generated mixture path, train and validation mixture keys,
`robot_type`, `task_name`, `action_dim`, and `active_action_dim` in both
manifests. They must agree with the Hydra overrides and `env.action_order`.
