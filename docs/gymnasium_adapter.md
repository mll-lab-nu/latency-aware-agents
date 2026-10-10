# Adapting a Gymnasium environment

This document is both the manual runbook for a task author and the primary
execution specification for an adaptation agent. It describes the separate
Gymnasium integration, the decisions that must be frozen, the files that must
be generated, and the minimum evidence required before a task is called
adapted.

For a complete worked example with real server commands, generated files,
figures, video, verified InvertedPendulum results, and unexecuted next-stage
commands, open the [Gymnasium adaptation examples](gym-adapt/gymnasium/README.md).
The runnable
example configs are in
[`configs/examples/gymnasium/inverted_pendulum/`](../configs/examples/gymnasium/inverted_pendulum/).

## Audience and document roles

| Reader | How to use this document |
| --- | --- |
| User adapting manually | Follow the manual workflow in order, copy the InvertedPendulum config structure, replace the complete task contract, and run every applicable verification gate. |
| User working with an agent | Provide the information in the agent prompt, approve the scientific decisions the API cannot answer, and review the generated report and evidence before running later stages. |
| Adaptation agent | Treat every required field, stop condition, verification gate, and reporting requirement below as part of the task. Do not replace a missing decision with a hidden default. |

The Markdown guide is normative: it defines what must be done. The HTML
walkthrough is explanatory and evidentiary: it shows how one real adaptation
was performed and defines the expected shape of a future agent's final report.

## Supported release path

The basic path is:

```text
Gymnasium task contract
  -> latency-aware Sample Factory training
  -> small-model teacher rollout
  -> LeRobot conversion
  -> basic single-image OpenVLA SFT
  -> latency-bench evaluation
```

The compatibility path supports native `Discrete` and bounded `Box` action
spaces. A native `gymnasium.spaces.Discrete` task declares its finite ordered
catalog in the contract. A native `gymnasium.spaces.Box` task preserves its
ordered vector, bounds, dtype, and no-op action through the generic adapter and
continuous rollout/export path. A task-owned wrapper is needed only when the
native action or observation structure is not supported by that path. The
framework does not silently discretize actions.

The existing Flappy Bird, Demon Attack, and Deadly Corridor integrations remain
separate. New tasks use `env.name: gymnasium` and are dispatched through
`GymnasiumEnvAdapter` and `training.gymnasium_sf`.

## Manual adaptation workflow

Use this sequence whether the work is performed by a user or an agent.

### 1. Pin and install the task package

Record the exact package version and Gymnasium environment ID. Install the
task-specific package in every runtime that constructs the environment: the
small-model/rollout environment and, when online VLA evaluation is required,
the StarVLA environment as well.

Check the pinned versions in `pyproject.toml` before installing additional
simulator packages. The base dependencies already include MuJoCo.
Keep optional task dependencies separate from the shared runtime.

### 2. Probe the installed API before writing configs

Run a bounded probe against the exact installed version. At minimum, record:

```python
import importlib
import gymnasium as gym

for module_name in REGISTRATION_IMPORTS:
    importlib.import_module(module_name)

env = gym.make(ENV_ID, render_mode="rgb_array", **MAKE_KWARGS)
observation, reset_info = env.reset(seed=0)
next_observation, reward, terminated, truncated, step_info = env.step(PROBE_ACTION)
rgb = env.render()

print(env.spec)
print(env.action_space, env.observation_space)
print(observation.shape, observation.dtype)
print(next_observation.shape, next_observation.dtype)
print(reward, terminated, truncated, reset_info, step_info)
print(rgb.shape, rgb.dtype)
env.close()
```

Adapt the probe action to the native action space. Also inspect task-specific
timing fields, frame skip, time limit, reset options, available action meanings,
and renderer requirements. Save the probe output rather than relying on memory
or a newer documentation page.

### 3. Separate API facts from task-author decisions

The probe establishes runtime facts. It does not establish the experiment. The
task author must explicitly approve the raw-tick meaning, clock, action
representation, no-op, preprocessing, task semantics, prompt, latency, and
budgets listed in the next section.

### 4. Decide whether a task-owned wrapper is required

- A native `Discrete(n)` action space with JSON-stable action values can usually
  be described directly in the `env` config.
- A native bounded `Box` action space can be described directly with its ordered
  dimension labels, per-dimension bounds, dtype, and pending no-op vector.
- A structured or otherwise incompatible action space needs one registered
  task-owned wrapper that exposes the exact policy-facing representation.
- An observation unsupported by the basic Sample Factory policy needs one
  explicit task-owned observation wrapper used identically in training and
  evaluation.
- If `render_mode: rgb_array` does not provide stable HWC RGB frames, rollout
  and VLA stages are not ready until the task owns an explicit rendering path.

Do not add environment-family behavior to `GymnasiumEnvAdapter`; new task
semantics belong in the registered task wrapper and its tests.

### 5. Freeze one contract and create four configs

Copy the directory structure, not the InvertedPendulum scientific choices:

```bash
cp -R configs/examples/gymnasium/inverted_pendulum \
  configs/examples/gymnasium/<task_name>
```

Replace the complete `env` block and task/output names. Keep
`small_model_eval.yaml` based on `small_model_train.yaml` and
`starvla_eval.yaml` based on `latency_bench_eval.yaml`. Resolve all four files
through `latency_bench.core.config.load_config()` and confirm that their final
`env` mappings are identical.

### 6. Add semantic tests and verify timing

When a wrapper is introduced, test its registered ID, policy-ID-to-native-action
mapping, observation/render behavior, and frozen contract. Then run the
zero-latency parity and requested fixed-latency trace described below.

### 7. Run only the authorized experiment stages

Config composition and logic tests do not require a long training budget. A
full small-model run, teacher rollout, data conversion, OpenVLA training, and
online evaluation are distinct stages. Run only stages authorized by the user
and label the others `NOT RUN`.

### 8. Deliver an adaptation report

An agent-assisted adaptation is not complete when code alone is generated. The
agent must create a task-specific report such as
`docs/gym-adapt/gymnasium/<task_name>/index.html`, following the structure and
visual language of the worked examples, and link it from
`docs/gym-adapt/gymnasium/README.md`. The required contents are specified in “Adaptation
report contract” below.

## What can be inspected from the environment API

Given an installed Gymnasium environment ID, a user or agent can probe:

- the registered environment version and constructor arguments;
- action and observation spaces, shapes, bounds, and dtypes;
- the `reset()` and `step()` return structure;
- `terminated` and `truncated` behavior observed during a bounded trace;
- rendered RGB shape and dtype when `render_mode: rgb_array` is supported;
- wrapper metadata such as `render_fps` when the environment publishes it;
- Atari minimal action meanings and PAL/NTSC screen height through ALE;
- MuJoCo `dt`, `frame_skip`, time limit, and the installed environment version.

API inspection cannot decide the experiment semantics. The task author must
approve:

- what one raw tick means;
- `env_fps`, `obs_fps`, and the latency condition;
- the safe pending no-op action;
- action labels/order/native values for Discrete, or dimension order/bounds/dtype
  for Box;
- any action discretization, scaling, or observation preprocessing;
- reset semantics, including whether Atari-style seeded random NOOP steps are
  enabled and their explicit maximum;
- reward, termination, truncation, and success semantics;
- the base language prompt;
- training, rollout, filtering, and evaluation budgets.

If one of these choices is scientifically ambiguous, the agent should stop and
ask rather than invent a hidden default.

## Agent interaction protocol

An adaptation agent should work in explicit phases:

1. **Reconnaissance:** read this guide and the InvertedPendulum walkthrough,
   inspect the repository and pinned task runtime, and produce an API-fact
   report before changing task code.
2. **Decision request:** show the user a compact table of unresolved choices,
   why each choice changes the experiment, and a proposed value only when a
   reasonable proposal is useful. Clearly label proposals as unapproved.
3. **Contract confirmation:** wait for the user to approve the action
   representation, no-op, clock, preprocessing, task semantics, prompt, latency,
   and authorized execution budget.
4. **Generation:** create the wrapper only if required, the four grouped
   configs, semantic tests, and reproducibility material. Do not edit a legacy
   environment path to make the new task work.
5. **Verification:** run config equality, zero-latency parity, latency-trace,
   observation/render, focused generic-path, and task-specific tests.
6. **Bounded execution:** run training or GPU stages only within the approved
   time, compute, storage, and download budget.
7. **Delivery:** create the task report, distinguish `VERIFIED`, `NOT RUN`, and
   `BLOCKED`, and give the user exact next-stage commands with placeholders
   replaced wherever artifacts already exist.

The agent must stop and ask again if probing contradicts the proposed contract,
the installed environment version differs, no safe pending no-op is known, the
chosen policy representation would materially redefine the scientific task, or
a later-stage manifest/checkpoint contract does not match the config.

## The environment contract

Every new task defines one `env` mapping and copies it unchanged into all four
task configs:

```yaml
env:
  name: gymnasium
  task_name: example_task
  env_id: Package/RegisteredEnv-v0
  registration_imports:
    - package_that_registers_the_env
  make_kwargs:
    render_mode: rgb_array
  env_fps: 30
  obs_fps: 10
  frame_stack: 1
  noop_action: noop
  action_map:
    noop: 0
    action_a: 1
    action_b: 2
  action_order:
    - noop
    - action_a
    - action_b
  base_prompt: >-
    Complete the task. Choose exactly one action from:
    noop, action_a, action_b.
```

For a native continuous `Box`, replace the discrete catalog fields with an
`action_space` block and a vector no-op:

```yaml
action_space:
  type: box
  labels: [joint_0, joint_1]
  low: [-1.0, -0.5]
  high: [1.0, 0.5]
  dtype: float32
noop_action: [0.0, 0.0]
```

`gymnasium_task_contract()` freezes the shared task fields plus the
action-specific fields:

| Field | Meaning |
| --- | --- |
| `task_name` | Stable semantic task identity |
| `env_id` | Exact registered Gymnasium environment |
| `make_kwargs` | Exact constructor arguments |
| `registration_imports` | Modules imported before `gym.make()` |
| `action_labels`, `action_values`, `noop_action_id` | Discrete action catalog and no-op position |
| `action_space`, `noop_action` | Native Box/vector representation and no-op vector |
| `base_prompt` | Task instruction before latency wording |
| `env_fps` | Raw environment ticks per second |
| `obs_fps` | Policy decisions per second |
| `frame_stack` | Raw RGB history used by the VLA handoff |

Action order and native values for Discrete tasks, or dimension order, bounds,
dtype, and no-op for Box tasks, are experiment contracts, not display metadata.
Changing any of them creates a different task and requires new checkpoints and
datasets.

Latency is expressed in raw ticks:

```text
raw_latency_frames = ceil(latency_ms / 1000 * env_fps)
observation_stride = env_fps / obs_fps
```

The selected rates must produce the intended integer observation stride.

## Files generated for one task

Place the four runnable configs together:

```text
configs/examples/gymnasium/<task>/
  latency_bench_eval.yaml
  small_model_train.yaml
  small_model_eval.yaml
  starvla_eval.yaml
```

`small_model_eval.yaml` inherits `small_model_train.yaml`.
`starvla_eval.yaml` inherits `latency_bench_eval.yaml`. The resolved `env`
mapping must be identical in all four files.

When the native interface is not directly compatible, also add:

```text
latency_bench/envs/tasks/<task_wrapper>.py
scripts/scaffold_<task_family>.py       # only when a reusable probe is useful
tests/envs/tasks/test_<task_wrapper>.py
```

The InvertedPendulum example shows the finite-catalog bridge for a native
continuous action API:

- [`latency_bench/envs/tasks/inverted_pendulum.py`](../latency_bench/envs/tasks/inverted_pendulum.py)
  maps five catalog IDs to five user-approved scalar forces;
- [`scripts/scaffold_gymnasium_mujoco.py`](../scripts/scaffold_gymnasium_mujoco.py)
  records the pinned API facts and prints the accepted `env` block;
- [`configs/examples/gymnasium/inverted_pendulum/`](../configs/examples/gymnasium/inverted_pendulum/)
  contains the four grouped configs;
- [`tests/envs/tasks/test_gymnasium_mujoco.py`](../tests/envs/tasks/test_gymnasium_mujoco.py)
  verifies registration, native-action mapping, and the frozen contract.

The Walker2d example shows the native Box case: its six-dimensional torque
vector passes through the generic Gymnasium adapter without a finite action
catalog wrapper.

The MuJoCo files are a worked task-owned implementation, not code that every
Gymnasium task imports. A new task should reuse the generic interfaces and the
file organization while generating only the wrapper and dependencies its own
API requires.

## Verification before training

The adaptation should pass these gates:

1. The exact registration module can be imported in a clean process.
2. The resolved `env` mapping is identical across all four configs.
3. The configured no-op is the intended native action or vector.
4. At zero latency, a fixed seed and native action trace match the reference
   Gymnasium environment in observation, reward, termination, and truncation.
5. At fixed latency, the issued and applied action trace matches the expected
   raw-frame delay.
6. Policy observations and rendered RGB retain their documented shapes and
   dtypes.
7. Existing Flappy, Demon Attack, and Deadly Corridor tests remain unchanged.

Focused release tests include:

```bash
pytest -q \
  tests/envs/adapters/test_gymnasium_env_adapter.py \
  tests/training/test_gymnasium_sf.py \
  tests/integration/sample_factory/test_gymnasium_teacher_export.py \
  tests/integrations/starvla/test_starvla_lerobot_exporter.py \
  tests/policies/test_generic_gymnasium_starvla_policy.py
```

Add the task-specific wrapper test when the adaptation introduces one.

## Basic experiment commands

Train the small model:

```bash
python -m training.train \
  --config configs/examples/gymnasium/<task>/small_model_train.yaml
```

Evaluate a saved small-model checkpoint:

```bash
python -m latency_bench.run \
  --config configs/examples/gymnasium/<task>/small_model_eval.yaml \
  --checkpoint-path <checkpoint.pth>
```

Export teacher data from the checkpoint:

```bash
python scripts/rollout_data/export_sf_teacher_dataset.py \
  --checkpoint-root <sample_factory_run> \
  --output-dir <teacher_output> \
  --max-episodes 200 \
  --val-fraction 0.1 \
  --num-envs 1 \
  --replay-num-envs 1 \
  --device cuda \
  --deterministic \
  --filter-preset none \
  --context-window 1 \
  --export-image-size 224,224
```

Do not pass `--env-name` for a generic Gymnasium checkpoint. The exporter reads
the task identity and complete contract from the checkpoint.

Convert canonical rows to the LeRobot format:

```bash
python scripts/rollout_data/prepare_starvla_lerobot_dataset.py \
  <teacher_output> \
  third_party/starVLA/playground/Datasets/rl_games \
  <dataset_name> \
  --action-carrier native
```

Discrete Gymnasium tasks keep the existing StarVLA handoff and categorical
carrier. Continuous Gymnasium tasks preserve the native Box vector in LeRobot
and emit the same StarVLA modality, robot type, prompt map, and generated
mixture without adding a carrier dimension.

Run basic single-image OpenVLA SFT for a registered discrete task:

```bash
cd third_party/starVLA
bash commands/openvla/train_gymnasium_openvla.sh \
  playground/Datasets/rl_games \
  <dataset_name> \
  playground/Datasets/rl_games/_generated_mixtures/<dataset_name>.json
```

Run final evaluation after filling the four StarVLA artifact paths in
`starvla_eval.yaml`:

```bash
python -m latency_bench.run \
  --config configs/examples/gymnasium/<task>/starvla_eval.yaml
```

Final evaluation strictly compares the eval YAML contract, dataset manifest,
and saved model config before decoding a model action ID through
`action_order` and `action_map`.

## Adaptation report contract

For manual work, keeping this record is strongly recommended. For an
agent-assisted adaptation, it is required. Use
the [worked examples index](gym-adapt/gymnasium/README.md) as the reference and
create a task-specific HTML report at
`docs/gym-adapt/gymnasium/<task_name>/index.html`, plus curated assets inside
that task directory. Add the report to the examples index; do not overwrite an
existing walkthrough.

The report must contain:

1. **Scope and status:** environment/package versions, selected task ID,
   scientific objective, target latency, support boundary, and a summary of
   which stages were actually run.
2. **API reconnaissance:** exact probe command and output for construction,
   spaces, reset, step, render, timing, and version-specific behavior.
3. **User decisions:** the questions asked, the user's explicit answers, and a
   table separating API facts from task-author choices.
4. **Frozen contract:** the complete resolved `env` mapping and all shared plus
   action-specific `gymnasium_task_contract()` fields.
5. **Generated implementation:** wrapper, registration, configs, tests, and an
   explanation of how policy action IDs reach native environment actions.
6. **Pipeline propagation:** where each contract field is consumed by the env
   factory, Sample Factory integration/checkpoint, teacher export, LeRobot
   conversion, StarVLA config/loader, and final evaluation validation.
7. **Command ledger:** exact user-facing and agent-executed commands, runtime
   environment, output paths, duration/steps when known, and exit status.
8. **Evidence ledger:** parity traces, latency traces, resolved configs, logs,
   metrics, images, videos, manifests, and checkpoints that were actually
   produced. Link curated release assets; keep bulky raw experiment output in
   ignored `artifacts/` or external storage.
9. **Continuation runbook:** commands for the first stage not yet executed and
   all subsequent basic stages—teacher rollout, conversion, OpenVLA SFT, and
   final latency-bench evaluation—using the task's real paths and contract.
10. **Isolation and limitations:** legacy-path regression status, unsupported
    experiment modes, unresolved decisions, blockers, and claims the smoke run
    does not establish.

Use visible status labels consistently:

- `VERIFIED`: the command ran and the linked evidence supports the statement;
- `NOT RUN`: the path or command is documented but was not executed;
- `BLOCKED`: execution was attempted or required but could not proceed, with
  the exact blocker recorded;
- `USER DECISION`: a scientific choice supplied or approved by the task author;
- `API FACT`: a value observed from the pinned runtime.

Never show a planned artifact as if it exists. If a stage is not run, list its
expected outputs separately from the evidence ledger.

## Current boundary

| Environment property | Release support |
| --- | --- |
| Native `Discrete(n)` actions | Direct finite catalog |
| Native bounded `Box` vectors | Direct ordered vector contract and continuous carrier |
| Other structured actions | Explicit task-owned representation/wrapper required |
| Stable Box/vector or CHW image policy observations | Supported by the basic small-model path |
| Other observation structures | Explicit wrapper/preprocessing required |
| `rgb_array` rendering | Required for rollout and VLA stages |
| Basic APPO training | Supported |
| Basic single-image OpenVLA SFT | Supported handoff |
| Memory/frame-stack experiments | Outside this release path |
| Automatic action discretization or prompt design | Not supported |

Logic tests and short smoke runs establish that the pipeline executes; they do
not establish policy quality, convergence, or scientific validity.

## Prompt for an adaptation agent

Provide the agent with:

1. the exact Gymnasium environment ID and package/version;
2. the scientific objective;
3. the desired latency condition or permission to propose one;
4. the task prompt, or permission to draft one for approval;
5. the available CPU/GPU, time, and disk budget.

Suggested request:

```text
Read docs/gymnasium_adapter.md and docs/gym-adapt/gymnasium/README.md. Adapt [ENV_ID] from
[PACKAGE + PINNED VERSION] into latency-sensitive-bench through the separate
env.name=gymnasium path. My task objective is: [OBJECTIVE]. Target latency is
[LATENCY]. Please first probe the exact Gymnasium API and report: constructor
arguments, action/observation spaces, reset/step/render behavior, native timing,
and anything you cannot infer. Ask me to decide any action representation,
dimension/order/no-op, raw-tick meaning, preprocessing, reward/termination policy,
prompt, and budgets.
After I answer, create a task-owned wrapper only if needed; generate grouped
small-model train/eval, latency-bench eval, and StarVLA eval configs; preserve
the resulting task contract across checkpoint, rollout, conversion, SFT, and
final eval; run parity/compatibility tests; and keep every legacy environment
path untouched. Record commands, generated files, artifact paths, and which
stages were actually run. Create a self-contained adaptation report such as
docs/gym-adapt/gymnasium/[TASK_NAME]/index.html that follows the worked examples.
Separate API facts from my decisions, show the frozen contract and generated
files, document the pipeline consumers, preserve exact commands and evidence,
explain how I can continue rollout/SFT/eval, and mark every stage VERIFIED,
NOT RUN, or BLOCKED.
Do not invent results.
```
