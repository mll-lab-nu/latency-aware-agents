# Native coding agent evaluation

`policy.type: coding_agent` connects either Codex (`provider: codex`) or Claude
Code (`provider: claude_code`) to the paper environments. One native process and
conversation belongs to one episode. Evaluation is serial, with the existing
simulated executor and no inference pool. The one-slot batched executor is also
supported for decision-action history. Simulator hardware remains an environment
setting; using a CPU agent does not make ManiSkill or a GPU renderer CPU-only.

## Reference inputs and native actions

Select an existing VLA evaluation and provide its **input sidecars**:
`model_config_path`, plus `task_manifest_path` or `task_contract_path` when the
reference uses them. Put that policy under `policy.reference_policy`. Its
checkpoint path is retained as provenance; weights are not loaded. Existing
StarVLA configuration helpers and CPU dependencies must be installed.

The adapter uses the same input builders as StarVLA: PIL image resizing,
temporal selection, stitched/packed views, ghost trails, normalized state,
decision-action history and action-prefix conditioning. State is omitted if the
checkpoint omits it. Legacy zero state is preserved only when the reference VLA
uses it. Task camera order follows `video_keys`; every resulting image is sent
to the native client in that order. Executor metadata, privileged environment
fields and future sampled latency are not added to the agent input.

The following files locate the existing environment contracts. Their historical
artifact paths, GPU settings and evaluation protocols are **not ready-made final
agent experiments**; select the intended exported reference before composing a run.

| Paper environment | Reference configuration | Agent action |
|---|---|---|
| Flappy Bird | `../flappy/starvla_eval_zero_latency.yaml` | Integer 0–1; preserves one-shot FLAP |
| Deadly Corridor | `../deadly_corridor/starvla_eval_zero_latency.yaml` | Seven semantic button bits; factorized Tuple/MultiDiscrete references also supported |
| Asterix | `../asterix/starvla_eval_zero_latency.yaml` | Nine native directions, including diagonals; distinct from six factorized model logits |
| Demon Attack | `../demon_attack/starvla_eval_zero_latency.yaml` | Six native actions |
| Atlantis | `../atlantis/starvla_eval_zero_latency.yaml` | Four native actions |
| AirRaid | `../../examples/gymnasium/air_raid/starvla_eval.yaml` | Six native actions |
| Ant | `../../examples/gymnasium/ant/starvla_eval.yaml` | Eight continuous torques |
| Walker2D | `../../examples/gymnasium/walker2d/starvla_eval.yaml` | Six continuous torques |
| InvertedPendulum | `../../examples/gymnasium/inverted_pendulum/starvla_eval.yaml` | Five force IDs in the existing discrete reference; native Box references retain their Box contract |
| Hopper | `../../examples/gymnasium/hopper/starvla_eval.yaml` | Three continuous torques |
| Balance (simple) | `../../examples/gymnasium/g1_sonic_balance_simple/starvla_profile.yaml` and exported task bundle | 43 absolute joint positions, policy order, full task horizon |
| InterceptGrab (Fast) | `../../examples/mikasa/official_starvla_profile.yaml` and exported task bundle | Seven native controller components, top/wrist views, full task horizon |

Bounds come from the constructed environment, including Balance's physical joint
ranges. Actions are in **environment units**, even though VLA inputs are
normalized. The initial prompt records those bounds, labels, no-op, action horizon,
scheduler rules, FPS and latency summary. A one-head policy returns exactly
`{"action": VALUE}`; a multi-head task returns exactly
`{"actions": [HEAD_0, ..., HEAD_H_MINUS_1]}`. Each head spans one raw frame.
Frozen prefix heads retain the controller plan, as in the VLA task runner.
Malformed, wrong-length, nonfinite or out-of-range responses fail the attempt;
there is no clipping, substitute action or repair loop.

For Balance, use the exported task's matching scene, state labels, control mode
and no-op. Profile H8/H16 contracts retain the returned-action state through the
existing `balance_profile` adapter. MIKASA requires its explicit simulator-device
configuration and ManiSkill assets. Neither native task's physical behavior is
established by contract tests alone.

## Compose a configuration without starting an experiment

`python -m latency_bench.policy.coding_agent_config` combines three explicit inputs:
the reference VLA evaluation, an existing native-client configuration, and a run
protocol containing `experiment`, `executor`, `latency`, `evaluation`, and `logging`.
The complete protocol sections replace inherited ones so old evaluation suites
cannot silently override a new run. Environment and scheduler settings stay with
the selected reference. Use resolved artifact paths valid on the execution host.

```bash
python -m latency_bench.policy.coding_agent_config \
  --reference /path/to/reference_vla_eval.yaml \
  --agent configs/experiments/coding_agent/flappy_codex_sol61_low.yaml \
  --provider codex \
  --protocol /path/to/confirmed_protocol.yaml \
  --output /path/to/agent_eval.yaml
```

This command only writes a configuration. Set `executor.mode: simulated`,
`executor.simulated_inference_pool: false`, and `evaluation.eval_parallel_envs: 1`
in the selected protocol. Choose the latency profile, capacity, FPS, seeds and
episode count deliberately before running `python -m latency_bench.run`.
The original `type: custom` Flappy pilot files retain their original prompt/input
contract for reproduction.

## Original Flappy pilots

The Luna pilot uses Codex CLI 0.155.1 and `gpt-6-luna` at low effort through the
native app-server. Use `flappy_codex_luna_low.yaml` with the same evaluation command
below. The CPU environment, observation contract, five seeds, and OpenVLA temporal
profile are identical to the Claude setup.

Codex login is stored natively in the dedicated `policy.codex_home`; authenticate
there using `codex login --device-auth` and file credential storage. The adapter
does not copy or delete `auth.json`. Each episode starts a new process and an
ephemeral native thread. Memory use/generation, apps and plugins are disabled;
no custom instructions are supplied. Require an empty `instructionSources` list.
Non-auth native state is archived outside the agent sandbox between episodes;
per-episode SQLite and log directories are temporary. Image paths are unique per
decision so prior image references never point at a newer observation.

`thread.json` records the resolved model/effort and instruction sources. Native
JSON-RPC events include tool activity, token usage, compaction and routing. A
model reroute or unsuccessful turn is a failed attempt, not a substituted action.
Authentication persists and refreshes in place independently of conversation
state. The sandbox mounts the installed Codex package read-only, not the checkout.

## Claude Code configuration

Run five serial episodes with Claude Code 2.1.296, `claude-haiku-5-5`, low effort,
and a fresh persistent CLI process/conversation for each episode. Within an
episode, native conversation history and tools remain available. No custom
system prompt, skill, MCP server, memory, controller, or action-repair loop is
installed. Safe mode may list product-bundled skills/plugins; preserve the init
event as the record of the actual native tool surface.

`flappy_claude_opus55_low.yaml` selects `claude-opus-5-5` at low effort with the
same five-episode protocol. Run the two presets sequentially because they share
the native login store.

The agent receives one 224x224 RGB image resized with the same PIL preprocessing
as StarVLA, plus a seven-dimensional zero state. This matches the selected
Flappy input contract; the agent can additionally remember its episode history.
No numeric game state or future sampled latency is sent. Final actions must be
exact JSON with one integer `action` equal to 0 (NOOP) or 1 (FLAP).

## Latency and provenance

Use the three profile files from `MLL-Lab/LAGEN-profiles` revision
`8531005becf942264a5c8e3965d1a94b657a90df`, under
`profiles/openvla/1x-rtx3090/flappy/instance_a5037b165aa0cedc/`:
`profile.json`, `latency_distribution.json`, `latency_burst_model.json`.
The temporal sampler retains its session state across episodes. Seed 42 controls
latency; environment seeds are 0 through 4. Environment/observation rates are
30/30 FPS, one virtual inference worker, latest-ready ordering, one-frame-then-noop
actions, and a 3600-raw-frame limit per episode.

The registry's OpenVLA name refers here to the QwenOFT checkpoint. Its migrated
input config is recorded in the YAML provenance fields. The old profile has no
immutable checkpoint digest; pinning its published revision fixes the latency
artifact, but does not establish bitwise identity of migrated model weights.
No model weights are loaded or re-profiled for this experiment.

## CPU runtime

For a Flappy experiment using the agent's own latency, use `executor.mode:
realtime`, `latency.method: measured`, and `executor.inference_devices: [cpu]`.
The legacy Flappy action/input contract is unchanged; its prompt describes real
wall-clock delay. The environment continues stepping while a request is in
flight. Setting both `env_fps` and `obs_fps` to 1 advances one original game frame
per second, without changing Flappy's per-frame physics.

Realtime episode preparation opens a fresh native conversation before the game
clock starts. `agent/episode_*/preparation.json` records this startup separately;
image preparation, API/network time and action decoding remain in measured
latency. Claude uses the native stream-JSON initialization handshake, without a
model prompt or action request. Invalid-action errors retain their classification
across the inference process boundary.

The standard evaluation driver still runs its normal realtime profiling warmup.
A short pilot can call the executor's `run_episode` directly after construction
and record `native_initialization_only` as its warmup protocol. Give concurrent
pilots separate code, output, workspace, runtime and native credential-state
directories so they cannot archive or reset the main campaign's native state.

Use an independent Python 3.10 environment with CPU PyTorch, the root project's
Flappy evaluation dependencies, and the vendored
`third_party/flappy-bird-gymnasium` package. Install a CPU ffmpeg executable for
post-run videos. Keep CUDA invisible, set SDL video/audio drivers to `dummy`, and
limit BLAS/OpenMP threads. Run the process at low priority on two CPU cores.

The policy requires Linux bubblewrap with unprivileged user/PID namespaces.
It exposes read-only system runtime files, an episode workspace, a private copy
of the existing Claude OAuth credentials, a fresh `/proc`, and a fresh `/dev`.
Training checkouts, host processes, GPU devices, and evaluation outputs are not
mounted. Tool permission bypass is used only inside this external sandbox.
Refreshed OAuth credentials are atomically saved back to the original login store
before removing the private copy. This preserves token rotation; no user settings
or conversation history are copied back. The copied credentials are removed when
the episode process closes.

Configure the executable, credentials, bubblewrap, profile, runtime, and workspace
paths in `flappy_claude_haiku55_low.yaml`. Use a new output directory for every run:

```bash
python -m latency_bench.run \
  --config configs/experiments/coding_agent/flappy_claude_haiku55_low.yaml \
  --output-dir /home/ubuntu/lzj/coding-agent/outputs/RUN_ID
```

## Evidence and failures

Standard benchmark files record environment steps, sampled latency, scheduled
actions, metrics, and replay videos. `agent/episode_*/` records the session ID,
CLI command/PID, stderr, and each decision's exact input, image, native JSONL
events, wall time, parsed action, and result usage. `modelUsage`, total cost, and
API duration can be cumulative: use differences or per-message usage rather than
summing cumulative results. Keep cache reads, cache writes, and uncached tokens
separate. CLI cost is a list-price estimate, not a subscription invoice.

An invalid action, changed model/session, CLI error, or 180-second decision timeout
fails the run explicitly. Preserve the failed attempt and do not silently replace
actions or retry for a better outcome. Agent wall time does not enter virtual
latency. Record compaction and native tool events; a long-lived process does not
eliminate cache-read tokens or context limits.

For the 20-episode Opus/Sol campaign, the selected failure rule permits one fresh
episode retry after an invalid model action. Restore that episode's environment
seed and initial latency-sampler state, and retain both attempts separately.
A second invalid action assigns zero to the episode's reported score; preserve
its partial physical return and the reason in metadata. Authentication,
transport, model-selection and simulator failures remain operational errors.
Completed episodes retain their original scores when a campaign resumes.
