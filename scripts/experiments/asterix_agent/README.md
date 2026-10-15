# Asterix agent harness

This experiment uses the benchmark's simulated executor, process inference pool,
IID sampler, `latest_ready` arbitration, action scheduler and trajectory logger.
The agent chooses every online action. Each worker owns one Codex app-server.
Each decision starts a fresh thread with an explicit public input.

Run from the repository root with the server's existing Python environment:

```bash
export PYTHON=/home/ubuntu/miniconda3/envs/latency/bin/python
export PATH=/home/ubuntu/lxy/runtime/bin:$PATH
bash scripts/experiments/asterix_agent/run.sh --phase preflight
bash scripts/experiments/asterix_agent/run.sh --phase evaluate
```

`--config` selects a different explicit configuration. Preflight freezes that
configuration and verifies subscription authentication, instruction selection,
model, effort, image input, action parsing and quota headroom. Evaluation requires
the same configuration and a successful preflight. Use a new output directory
for another experiment. There is no automatic recovery or additional seed search.
Preflight saves the strategy and worker instructions in `inputs/`. Evaluation
checks their contents against the source files and workers read the saved copies.
Editing a source file after evaluation starts does not change a later episode's text.

The reference game strategy in `guard_intercept_mid_robust.txt` comes from
`/home/ubuntu/wzh/expts-sol/results/asterix/session-approved-compact-20261015/guard_intercept_mid_robust-seed2/guard_intercept_mid_robust/prompt.txt`.
`prompt.py` preserves its image timestamps, held action and eight most recently
accepted commands. Initialization uses the existing Asterix adapter's 1–30 NOOP
frames. The harness captures raw RGB from ALE, then resizes with PIL bilinear.
The adapter's grayscale training observation is not sent to the agent.

The default runs one episode with seed 2 and a 3600 raw-frame limit. The virtual
clock uses 7.5 FPS and one observation per four raw frames. Normal game time uses 60 FPS.
Stable image history is t−16, t−8 and t. Warm-up retains the oldest available
image and removes duplicate image indices. Actions persist until replacement.

Copy the original `profile.json`, `latency_distribution.json` and
`latency_burst_model.json` into `/home/ubuntu/lxy/profiles/asterix/` without
refitting. IID uses pooled profile slot 0. Each episode starts the original
`random.Random(42 + seed)` stream. The experiment bypasses the executor factory's
additional episode seed derivation. Virtual capacity is eight. Three real CPU
workers make subscription calls. Logs distinguish their identities.

The configured 37% quota stop line comes from the original iterative
experiment's `status.json`. Every request checks ordinary subscription usage.
Unavailable usage, a reached stop line or an authentication failure stops the
run. Paid credits do not authorize a fallback. The shared login directory is
never archived. Worker overrides disable apps, plugins, memory, project documents,
host skill discovery and unrelated native tools. The saved startup arguments,
effective configuration, base instructions and token usage document this condition.
These results are reproduction controls under the same task and latency protocol.
The minimal instructions differ from the historical Codex coding instructions.

The current harness executor path supports one simulated environment. The
existing paused, realtime, batched and VLA defaults retain their behavior.
The frozen reproduction disables plans, online tools and extra memory.

Video recording requires `ffmpeg` on `PATH`. A server-local binary can live in
`/home/ubuntu/lxy/runtime/bin/` without changing the Python environment.

## Public input and optional response fields

`latency_bench/harness/` owns the public history and response protocol. The
coordinator saves every legal image, request snapshot, virtual return, acceptance,
replacement, discard and executed frame interval. Snapshots contain only explicit
public fields. They do not copy environment metadata or reveal sampled request
latencies. In-flight entries expose identifiers, source frames and elapsed time.
Real subscription wall time stays in the worker's call records.

The baseline accepts an integer or `{"action": 2}`. An experiment can explicitly
enable plans, memory and named tools in its harness configuration. Its prompt
formatter must describe the enabled protocol and supply the selected history.
For example:

```json
{
  "plan": {
    "actions": [{"action": 2, "frames": 4}, {"action": 0, "frames": 2}],
    "alignment": "acceptance",
    "end": "hold"
  },
  "memory": {"target_row": 3},
  "tools": [{"name": "inspect", "input": {"region": "center"}}]
}
```

Durations are positive integer raw frames. `alignment` is `observation` or
`acceptance`. Observation alignment skips the elapsed prefix on acceptance.
`end` is `hold` or `noop`. A new accepted response replaces the old plan.
The scheduler only plays the returned plan. It does not select actions from images.

Tool mappings use `module:function` references. A tool receives a copied public
request snapshot and its JSON arguments. It returns information, with no
environment or action submission handle. A returned model response can schedule
tools without another serial model turn. Simulated tool results appear after
their measured duration on the virtual clock. `ToolTasks` also supports background
realtime execution. The next request receives only completed visible results.
Memory entries retain their source observation and return frame.
Plan snapshots report the action scheduled for the current frame, including the
terminal hold or NOOP behavior. Interrupted episodes settle pending requests with
`episode_interrupted` and save an `episode_end` event before closing workers.

## Evidence and validation

Outputs include the frozen configuration, effective worker instructions, startup
arguments, CLI version, images, JSON-RPC requests and events, model responses,
public history, actual actions, video and report. Incomplete episodes receive no
score. A round receives a mean only after all configured episodes complete.

```bash
CUDA_VISIBLE_DEVICES= "$PYTHON" -m pytest tests/harness tests/executors tests/scheduler tests/policies/test_codex_policy.py tests/latency -q
git diff --check
```

The harness tests are retained behavioral contracts. They cover plan alignment
and termination, replacement, out-of-order results, future-information isolation,
tool visibility, memory provenance, shared authentication ownership and fresh
decision threads. The Asterix tests compare raw ALE and the existing adapter on
fixed trajectories, including pixels, rewards, life counts and termination.
The historical mean of 1466.7 is a comparison value, not a test assertion.

## Auto research

`run_research.sh` runs one research cycle after a completed episode. It keeps
the reproduction entrypoint and its exact reference checks separate.

```bash
bash scripts/experiments/asterix_agent/run_research.sh \
  --config configs/experiments/asterix_agent/research.yaml
```

The configuration explicitly selects the source revision/diff evidence, review
episode, active-version file, output directory, seeds, frame limit and budgets.
Use a fresh output directory for each cycle. The runner copies the framework,
tests and instructions into the output's `source/` directory, then executes that
copy. Preserve this directory after promotion: the default-version file points
to the tested candidate there. An absent default-version file selects the
shipped reference strategy for the first parent. A later cycle copies the active
candidate as its parent.

The first configuration permits one candidate, seeds 2, 11 and 17, and at most
1200 raw frames for each version on each seed. This gives six paired episodes
and at most 1800 game decisions. Two separate subscription preflight decisions
and one research turn are recorded outside that game-decision budget. Research
has a 20-minute limit. The complete cycle, including regression tests, has a
three-hour limit. Authentication or platform quota failure stops the cycle.
Research has no artificial usage-percentage stop and never falls back to paid
API credentials. The reproduction configuration retains its original stop line.

The researcher uses the existing subscription transport with native tools
disabled. It receives strategy source, a compact episode timeline and images
around life losses. It returns a diagnosis and complete replacement file texts.
`research_instructions.txt` is the fixed modification contract. The evaluator
applies only lowercase Python files, `prompt.txt` and `settings.json` to a fresh
candidate directory. Protected settings are not part of that document's schema.
The researcher can return `no_candidate` or `framework_issue` instead. Both
outcomes preserve the current default and stop evaluation.

Candidate Python handles prompt formatting, public-history organization,
information tools and optional request skipping. `should_request(snapshot)`
returns a boolean at an existing legal observation. A skipped request consumes
no latency sample and creates no in-flight entry. The latest-ready scheduler
continues to play only model-returned actions or plans.

Each candidate Python invocation runs in a fresh, networkless bubblewrap
process with a ten-second timeout. It sees read-only candidate code, staged
legal images, the supplied snapshot and a temporary writable workspace. It
cannot see the coordinator's environment state, sampled delays, future answers,
credentials or evaluation files. Candidate functions cannot submit actions.
Information-tool execution keeps the existing delayed visibility protocol.

Before subscription evaluation, the runner checks its protected regression
suite, replays the original actions to compare archived images/rewards/lives
and termination, then runs the candidate's targeted diagnostic on a public
snapshot at or before the reported problem. Replay establishes the original
scene; new paired games measure candidate performance. This release does not
restore controller state or branch games from a mid-episode checkpoint.

Promotion requires all checks and all six episodes to complete, with a strictly
higher candidate mean on the common seed set. Per-seed regressions, calls and
tokens are reported without separate rejection thresholds. A tie, failed check,
quota error or interruption leaves the default unchanged. Incomplete episodes
have no score. Candidate content is checked against its frozen copy before
each episode and promotion. A lock and atomic replacement protect default
updates from concurrent campaigns.

Outputs retain the source revision/snapshot, protocol, research instructions and
response, diagnosis, candidate diff, regression/replay/diagnostic results,
subscription preflights, complete game evidence and `REPORT.md`. Research and
preflight usage are separate from game calls. Scores use virtual latency;
subscription wall time is reported separately. No git commit or push occurs.
Use `run_research.sh --report /path/to/cycle` to rebuild the report from saved
evidence without making subscription calls or starting another evaluation.
