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
runs in `starvla_rl_games_openvla`. The project installation includes
`ale_py` and the Atari ROM package in both environments. For a separately
packaged task, install the same task package/version in both environments:

```bash
conda activate latency
python -m pip install -e /absolute/path/to/acme-gym

conda activate starvla_rl_games_openvla
python -m pip install -e /absolute/path/to/acme-gym
```

Use the same environment ID, registration package, wrapper, and constructor
arguments at every stage.

## 3. Define one task contract

Users configure one public schema under `env`. AirRaid is the public Atari
reference:

```yaml
env:
  name: gymnasium
  task_name: air_raid
  env_id: LatencyBench/Atari-v0
  registration_imports:
    - latency_bench.envs.gymnasium_atari
  make_kwargs:
    ale_env_id: ALE/AirRaid-v5
    render_mode: rgb_array
    screen_size: 84
    noop_max: 30
    frameskip: 1
    repeat_action_probability: 0.0
    full_action_space: false
    terminal_on_life_loss: false
    max_num_frames_per_episode: 108000
    grayscale_obs: true
    mode: 1
    difficulty: 0
  env_fps: 50
  obs_fps: 12.5
  frame_stack: 1
  noop_action: noop
  action_map:
    noop: 0
    fire: 1
    right: 2
    left: 3
    rightfire: 4
    leftfire: 5
  action_order:
    - noop
    - fire
    - right
    - left
    - rightfire
    - leftfire
  base_prompt: >-
    Protect the two buildings by moving sideways and shooting the flying saucers.
    Choose exactly one action from: noop, fire, right, left, rightfire, leftfire.
```

Reference configs:

- [`latency_bench_eval.yaml`](../configs/examples/gymnasium/air_raid/latency_bench_eval.yaml)
- [`small_model_train.yaml`](../configs/examples/gymnasium/air_raid/small_model_train.yaml)
- [`small_model_eval.yaml`](../configs/examples/gymnasium/air_raid/small_model_eval.yaml)
- [`starvla_eval.yaml`](../configs/examples/gymnasium/air_raid/starvla_eval.yaml)

### Atari scaffold

Generate the complete `env` block for another Atari game, then paste it into
copies of the four AirRaid configs:

```bash
conda activate latency
python scripts/scaffold_gymnasium_atari.py \
  --ale-env-id ALE/AirRaid-v5 \
  --base-prompt \
  "Protect the two buildings by moving sideways and shooting the flying saucers. Choose exactly one action from: noop, fire, right, left, rightfire, leftfire."
```

The scaffold imports the Atari package, constructs the exact wrapper, reads the
minimal ALE action meanings in native order, detects 210-line NTSC versus
250-line PAL RGB output, and emits the frozen action catalog and clock. Its
latency-oriented defaults are:

```yaml
env_id: LatencyBench/Atari-v0
registration_imports: [latency_bench.envs.gymnasium_atari]
make_kwargs:
  ale_env_id: <ALE/Game-v5>
  render_mode: rgb_array
  screen_size: 84
  noop_max: 30
  frameskip: 1
  repeat_action_probability: 0.0
  full_action_space: false
  terminal_on_life_loss: false
  max_num_frames_per_episode: 108000
  grayscale_obs: true
  mode: <documented default mode>
  difficulty: <documented default difficulty>
frame_stack: 1
```

Freeze the ROM's minimal action meanings in their exact native order, with
semantic lowercase labels in `action_map` and `action_order`. Set
`obs_fps = env_fps / 4` to retain the usual four raw Atari frames per policy
decision while allowing the latency scheduler to change an action on every raw
frame. Do not use a blanket `env_fps: 60`: confirm the ROM display format first.
AirRaid is PAL and therefore uses 50 raw frames/s and 12.5 decisions/s; common
NTSC games such as Demon Attack use 60 and 15.

These defaults intentionally disable ALE v5's native four-frame repeat and
sticky-action probability. Leaving the v5 defaults (`frameskip=4`,
`repeat_action_probability=0.25`) creates a different task: one benchmark tick
would cover four emulator frames and sticky actions would confound controller
randomness with measured inference latency.

### Explicit wrapper contract

The Atari scaffold uses the project-owned `LatencyBench/Atari-v0` wrapper. Its
complete constructor contract is the `make_kwargs` block above: the wrapper
constructs the named ALE task with one emulator frame per `step()`, then applies
Gymnasium `AtariPreprocessing` with `frame_skip=1`, the configured random no-op
reset, 84-pixel grayscale policy observations, no life-loss terminal, and no
observation scaling. It finally exposes CHW `uint8` policy observations while
leaving `env.render()` as the original HWC RGB game frame used for teacher/VLA
data. AirRaid's native reward, game-over termination, and ALE frame-limit
truncation remain unchanged; no reward clipping, policy frame stack, or
`EpisodicLife` is added. ALE itself applies AirRaid's required starting FIRE
action before the preprocessing wrapper performs its no-op reset steps.

The AirRaid-specific facts are grounded in the
[official environment specification](https://ale.farama.org/environments/air_raid/)
and ALE's
[`AirRaidSettings`](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/games/supported/AirRaid.cpp).
The latter defines the six-action minimal set, score-difference reward,
game-over condition, and starting FIRE action. ALE detects the distributed
AirRaid ROM as PAL; its console clock maps PAL to 50 FPS.

If another Atari task needs different preprocessing, register a different
wrapper instead of changing `LatencyBench/Atari-v0` silently. Make that
wrapper's ID, registration import, base ALE constructor arguments,
preprocessing arguments, observation space, reward, and episode boundaries
explicit. Use the same registered ID and constructor arguments for Sample
Factory training, rollout, conversion metadata, and final evaluation. A
training-only wrapper is an incompatible experiment, not a convenience default.

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

### Agent adaptation protocol: Atari Discrete fast path

This subsection is an execution contract for a coding agent. Give the agent
this document and one installed ALE environment ID such as `ALE/AirRaid-v5`.
That is sufficient for a coding agent to prepare the Atari Discrete fast path
and pass the CPU logic gate. A bounded GPU pipeline smoke is conditional on
the resources, licences, budgets, and scientific approvals listed below; this
guide is not a push-button authorization to start it. The environment ID alone
is **not** sufficient to choose the scientific task definition, run a long
experiment, or claim policy quality.

In short: the standard ALE finite-Discrete logic adaptation is autonomous;
the bounded end-to-end smoke is autonomous only after its inputs and resource
budget are approved; a reusable scientific experiment always requires task
author approval.

#### Minimum user input

Required:

- the exact ALE environment ID, including its version;
- access to the ROM through the normal ALE/Gymnasium installation.

Optional inputs are a stable task name, an approved task prompt, a target
fixed latency, output locations, and explicit smoke budgets. When they are not
provided, the agent may derive the snake-case task name, use the wrapper and
clock defaults emitted by the Atari scaffold, use a fixed latency only for a
short integration check, and draft a prompt of the form emitted by the
scaffold. A drafted prompt is provisional: it is suitable for checking data
flow, but it is not an approved scientific instruction.

#### Steps the agent may execute autonomously

1. Inspect the current checkout and preserve all existing environment
   interfaces. Do not rewrite Flappy Bird, Demon Attack, Deadly Corridor, or
   the generic adapter to specialize them for the new game.
2. Install or import only the task dependencies already required by the Atari
   fast path. Construct the requested ALE task through
   `LatencyBench/Atari-v0`; do not download unrelated models or run broad
   experiment suites.
3. Run `scripts/scaffold_gymnasium_atari.py --ale-env-id <ID>`. Record the
   detected PAL/NTSC clock, minimal native action meanings, native action
   values, no-op label, wrapper ID, registration import, and all constructor
   arguments. Preserve the emitted action order exactly.
4. Create task-owned copies of the four current Atari reference configurations
   for small-model training, small-model evaluation, latency-bench
   construction/evaluation, and StarVLA evaluation. Replace each root
   configuration's complete `env` block as one unit, keep inherited evaluation
   configs pointed at that task-owned root, choose a new output namespace, and
   require the same resolved task contract in all four. Do not add a second
   action or prompt registry.
5. Add or update a task-level contract test modelled on the AirRaid scaffold
   and parity tests. Verify construction, seeded reset/step, action-ID to
   native-action mapping, observation dtype/shape, HWC RGB rendering, reward,
   separate `terminated`/`truncated` values, raw-tick clock, and exact contract
   equality across all four configurations.
6. Run the focused generic adapter, Atari wrapper, scaffold, configuration,
   teacher-contract, converter-contract, and StarVLA contract tests. Run a
   zero-latency trajectory parity check before any latency smoke.
7. If a small CUDA-capable smoke is authorized and dependencies are already
   available, bound every stage: a minimal Sample Factory run that saves and
   reloads one real checkpoint; enough accepted teacher episodes to leave both
   train and validation non-empty with `filter_preset=none`; conversion of
   those rows; one real train and
   validation loader batch; two OpenVLA optimizer steps with optimizer state
   and best-model saving disabled; and a few scheduled raw ticks of online
   evaluation. Stop after proving artifact compatibility. Do not continue to
   convergence or evaluate policy quality.
8. Report the exact commands, resolved contract, test results, artifact paths,
   disk use, and every skipped GPU or long-running gate. Separate static
   composition, CPU logic checks, short GPU smoke, and full experiment status.

The agent may use the scaffold's generic prompt to complete steps 1--6. Before
step 7 produces data intended for reuse, the task author must approve the
prompt and every scientific decision in the table above.

#### Mandatory stop-and-ask boundaries

The agent must stop and request a user decision, without inventing a fallback,
when any of the following occurs:

- the action space is continuous, `MultiDiscrete`, `MultiBinary`, dynamically
  changing, or cannot be represented by JSON-stable native values;
- the minimal action meanings are missing, duplicated, unstable, or do not
  contain a demonstrated safe pending-action no-op;
- the task does not produce HWC RGB through `render_mode: rgb_array`, or its
  unchanged/wrapped observation is not accepted by Sample Factory;
- one `step()` cannot be tied to a documented emulator tick, PAL/NTSC timing
  cannot be established, or the requested observation rate is not an integral
  raw-tick stride;
- correct play requires changing preprocessing, native frame skip, sticky
  actions, reward, reset behavior, life-loss handling, termination, truncation,
  mode, difficulty, or the finite action catalog;
- the goal, legal-action wording, reward interpretation, episode boundary,
  rollout filter, latency condition, seed set, or training/evaluation budget is
  scientifically ambiguous;
- a contract field differs between live environment, checkpoint, teacher
  metadata, LeRobot manifest, StarVLA config, and final evaluation;
- required ROM access, credentials, model weights, licences, GPU memory, or
  disk budget are unavailable.

A user may explicitly approve a task-owned wrapper or a different experiment
definition. The agent must then assign it a new contract/output identity and
repeat parity; it must not silently normalize that choice back into the Atari
defaults.

#### Required artifacts

A completed adaptation handoff contains:

- four task configurations with exactly equal resolved 11-field task
  contracts;
- any task-owned registration wrapper and its focused tests, if one was
  explicitly approved;
- scaffold output or an equivalent resolved-contract record;
- zero-latency parity evidence and focused-test output;
- for a bounded end-to-end smoke: a loadable Sample Factory checkpoint,
  canonical train/validation parquet plus metadata and filter report, paired
  LeRobot datasets, generated mixture, latency prompt map, train manifest,
  `config.full.yaml`, a model-only OpenVLA checkpoint, and an online action
  trace;
- a validation note stating which gates are logic-only, short-smoke, or not
  run, together with artifact sizes and remaining scientific approvals.

Temporary smoke artifacts are not publication artifacts. The agent may remove
only disposable outputs that it created and identified; it must preserve
checkpoints, datasets, caches, or submodule changes owned by the user.

#### Acceptance gates and failure conditions

The adaptation passes the **logic gate** only when the exact task contract is
identical across the four configurations, zero-latency trajectory parity
passes, the action catalog round-trips to the native environment without
reordering, RGB and policy observations have the documented values and shapes,
and focused legacy plus generic tests pass.

It passes the **bounded pipeline smoke gate** only when a real small-model
checkpoint reloads, teacher train/validation output is non-empty and split by
episode, LeRobot manifests preserve the checkpoint contract and native action
width, one train and validation batch load, two OpenVLA steps save a loadable
model-only checkpoint with its `config.full.yaml`, and online evaluation
applies at least one model action at the expected raw tick. Passing this gate
does not establish learning, return, robustness, or reproducibility.

The adaptation fails closed if any required contract field is missing or
different, an action is guessed/remapped, parity differs, an exception is
swallowed, an observation or prompt is silently substituted, a legacy task's
behavior changes, or a stage is reported as tested when it was only composed
or skipped. In those cases the agent must preserve the failing evidence and
report the first semantic mismatch rather than continue with a hidden
fallback.

## 5. Raw-tick latency semantics

The clock is:

```text
raw tick duration       = 1000 / env_fps ms
raw ticks per decision  = env_fps / obs_fps
fixed latency in ticks  = ceil(fixed_latency_ms * env_fps / 1000)
```

AirRaid uses:

```text
env_fps                 = 50 (PAL)
obs_fps                 = 12.5
raw ticks per decision  = 4
fixed_latency_ms        = 100
fixed latency           = 5 raw ticks
```

One Sample Factory policy decision therefore advances four native AirRaid
frames. The scheduler queues the newly issued catalog action in
raw-tick time, applies the configured hold/ordering policy, and accumulates
reward over those four native steps. Dataset and evaluation names use `l5`
because 100 ms maps to five AirRaid raw frames.

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

The repository's generic parity tests are in
[`tests/envs/test_gymnasium_env_adapter.py`](../tests/envs/test_gymnasium_env_adapter.py).
For AirRaid, hold each catalog action for four native wrapper steps at every
reference policy boundary. A stochastic wrapper also needs repeated same-seed
trajectory checks.

A construction smoke test is useful but does not replace parity:

```bash
python -m latency_bench.run \
  --config configs/examples/gymnasium/air_raid/latency_bench_eval.yaml \
  --episode-start 0 \
  --episode-count 2
```

## 7. Train and evaluate the small model

Copy the complete training example and replace only the task contract, output
names, latency, and ordinary Sample Factory hyperparameters:

```bash
cp configs/examples/gymnasium/air_raid/small_model_train.yaml \
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
  --config configs/examples/gymnasium/air_raid/small_model_eval.yaml \
  --checkpoint-path \
    outputs/examples/gymnasium_air_raid/checkpoints/gymnasium_air_raid_fixed_l5/checkpoint_p0/checkpoint_XXXXXXXXX_YYYYY.pth \
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
    outputs/examples/gymnasium_air_raid/checkpoints/gymnasium_air_raid_fixed_l5 \
  --output-dir \
    outputs/examples/gymnasium_air_raid/teacher_fixed_l5 \
  --max-episodes 200 \
  --max-steps-per-episode 27000 \
  --val-fraction 0.1 \
  --num-envs 1 \
  --replay-num-envs 1 \
  --device cuda \
  --deterministic \
  --filter-preset none \
  --context-window 1 \
  --export-image-size 224,224
```

Do not pass `--env-name` for a Gymnasium checkpoint. The semantic task name is
part of the checkpoint contract; the exporter rejects attempts to relabel it.

The result is:

```text
teacher_fixed_l5/
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
  outputs/examples/gymnasium_air_raid/teacher_fixed_l5 \
  third_party/starVLA/playground/Datasets/rl_games \
  air_raid_fixed_l5 \
  --action-carrier native
```

This produces paired datasets and a generated mixture:

```text
third_party/starVLA/playground/Datasets/rl_games/
  air_raid_fixed_l5/
    data/chunk-000/episode_000000.parquet
    meta/info.json
    meta/modality.json
    meta/episodes.jsonl
    meta/tasks.jsonl
    latency_prompt_map.json
    manifest.json
  air_raid_fixed_l5__val/
    data/...
    meta/...
    manifest.json
  _generated_mixtures/
    air_raid_fixed_l5.json
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
  "air_raid_fixed_l5": [
    ["air_raid_fixed_l5", 1.0, "rl_games_gymnasium_discrete"]
  ],
  "air_raid_fixed_l5__val": [
    ["air_raid_fixed_l5__val", 1.0, "rl_games_gymnasium_discrete"]
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

WANDB_MODE=disabled \
WANDB_ENTITY=disabled \
WANDB_PROJECT=starvla_air_raid \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash commands/openvla/train_gymnasium_openvla.sh \
  playground/Datasets/rl_games \
  air_raid_fixed_l5 \
  playground/Datasets/rl_games/_generated_mixtures/air_raid_fixed_l5.json \
  checkpoint.save_best_model=false \
  checkpoint.save_training_state=false \
  checkpoint.save_safetensors_file=true
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
WANDB_MODE=disabled \
WANDB_ENTITY=disabled \
WANDB_PROJECT=starvla_air_raid \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash commands/openvla/train_gymnasium_openvla.sh \
  playground/Datasets/rl_games \
  air_raid_fixed_l5 \
  playground/Datasets/rl_games/_generated_mixtures/air_raid_fixed_l5.json \
  trainer.max_train_steps=2 \
  trainer.save_interval=2 \
  checkpoint.save_best_model=false \
  checkpoint.save_training_state=false \
  checkpoint.save_safetensors_file=true
```

The checkpoint overrides keep only a loadable model-only safetensors file and
avoid optimizer/training-state duplication. The full default 2000-step command
above writes
`checkpoints/steps_2000_model.safetensors`; the two-step smoke writes
`checkpoints/steps_2_model.safetensors`. If you change
`trainer.max_train_steps`, use the matching step in the final evaluation path.
Set real W&B values instead of the disabled values when experiment logging is
required. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces allocator
fragmentation during the short smoke and does not change the model or data
contract.

The resolved configuration must keep `framework.action_model.action_dim` equal
to the native carrier width and `action_env_dim` equal to the active task
width. For this first-release native path, both are `N`.

Before a long run, verify one loader batch:

- train mix resolves `air_raid_fixed_l5`;
- validation mix resolves `air_raid_fixed_l5__val`;
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
  checkpoints/steps_<N>_model.safetensors
```

Keep `config.full.yaml` with the checkpoint. Final benchmark input setup reads
the saved model configuration; a checkpoint file alone is not the complete
experiment artifact.

## 11. Evaluate the OpenVLA checkpoint in `latency_bench`

Copy
[`starvla_eval.yaml`](../configs/examples/gymnasium/air_raid/starvla_eval.yaml)
and replace its four artifact paths: checkpoint, saved model config, train task
manifest, and latency prompt map. Do not repeat or reorder actions under
`policy`: the generic StarVLA policy reads the canonical `env.action_order`.

Run from the repository root:

```bash
conda activate starvla_rl_games_openvla
cd /absolute/path/to/latency-sensitive-bench

python -m latency_bench.run \
  --config configs/examples/gymnasium/air_raid/starvla_eval.yaml \
  --checkpoint-path \
    third_party/starVLA/results/Checkpoints/air_raid_fixed_l5_openvla_sft/checkpoints/steps_2000_model.safetensors \
  --model-config-path \
    third_party/starVLA/results/Checkpoints/air_raid_fixed_l5_openvla_sft/config.full.yaml \
  --task-manifest-path \
    third_party/starVLA/playground/Datasets/rl_games/air_raid_fixed_l5/manifest.json \
  --latency-prompt-map-path \
    third_party/starVLA/playground/Datasets/rl_games/air_raid_fixed_l5/latency_prompt_map.json \
  --latency-raw-frames 5 \
  --episode-start 0 \
  --episode-count 20
```

The reference dataset contains only the L5 prompt because its teacher rollout
uses one fixed five-raw-frame latency. Keep `policy.latency_prompt_key: 5` and
`--latency-raw-frames 5` together. To evaluate additional latencies with a
prompt map, collect those latency conditions so the generated map contains the
matching keys.

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

The AirRaid reference path has also been smoke-tested on CUDA: a fixed-L5
Sample Factory run produced and reloaded a checkpoint; two teacher episodes
were exported and converted to train/validation LeRobot datasets; basic
single-image OpenVLA completed two optimizer steps and saved a model-only
checkpoint; and a 12-raw-frame online run applied its first two model actions at
raw frames 5 and 9. This verifies integration and short closed-loop execution,
not policy quality, convergence, or the full experiment matrix.

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

The generic path preserves the observation exposed by the configured `env_id`.
The Atari scaffold deliberately exposes CHW `uint8`; other tasks keep their own
native observation unless their explicit registered wrapper changes it.
Implement one task-owned registered observation wrapper, update the single
`env_id` contract, and repeat zero-latency parity. Do not add a training-only
fallback.

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

### OpenVLA stops before training because W&B settings are missing

For an offline smoke, export `WANDB_MODE=disabled`, `WANDB_ENTITY=disabled`,
and a non-empty `WANDB_PROJECT` as shown above. For a tracked experiment,
replace them with the intended project settings.

### A short OpenVLA run runs out of CUDA memory despite enough free memory

Retry with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Keep the smoke at
batch size one, save only the final model file, and disable optimizer/training
state checkpoints. These settings limit allocator fragmentation and disk use;
they do not make a model that is intrinsically too large fit the GPU.
