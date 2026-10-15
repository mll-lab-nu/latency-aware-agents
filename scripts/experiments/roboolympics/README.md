# RoboOlympics table tennis

The task adapter runs the upstream single-player match against its scripted opponent.
The example policy is the upstream camera-only reference player.

## Install the isolated CPU runtime

The frozen upstream source is included as the Git subtree `third_party/roboolympics`,
at upstream revision `da1f787f775ea6524d698c0b063157ded4c0fef9`.
The adapter locates it relative to the benchmark checkout. Use Python 3.11:

```bash
conda create --override-channels -c conda-forge -p /path/to/runtime python=3.11 pip
/path/to/runtime/bin/python -m pip install -r scripts/experiments/roboolympics/requirements.txt
/path/to/runtime/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

Run from this benchmark checkout. The launcher uses the bundled upstream source.
This runtime uses upstream MuJoCo 3.3.7 and NumPy 2.2.6, which differ
from this repository's general installation pins. Keep it separate from existing environments.
OSMesa must be available to the renderer. On the Debian test server, extract the
distribution's library inside the dedicated runtime, without changing system packages:

```bash
mkdir -p /path/to/runtime/osmesa
cd /path/to/runtime/osmesa
apt-get download libosmesa6
dpkg-deb -x libosmesa6_*.deb .
```

Return to the benchmark checkout and set the runtime paths:

```bash
export PYTHON=/path/to/runtime/bin/python
export LD_LIBRARY_PATH="/path/to/runtime/osmesa/usr/lib/x86_64-linux-gnu:/path/to/runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
bash scripts/experiments/roboolympics/eval.sh --config configs/examples/roboolympics/tabletennis.yaml
bash scripts/experiments/roboolympics/eval.sh --config configs/examples/roboolympics/tabletennis_100ms.yaml
```

Use `--output-dir` to select an artifact directory. Use `--max-env-steps` for a short
smoke run. A truncated run does not establish match completion.
The default opponent is L1 and the default seed is 1, from `sport.toml [official] seeds`.

## Latency units and the upstream slowdown

Configured latency is in simulated game milliseconds. This in-process executor does not
pace the world by the upstream wall clock. For comparison with an upstream match slowed
by S, convert wall-clock latency with `game_latency_ms = wall_latency_ms / S`.
At the frozen table-tennis slowdown of 10, a 100 ms model call represents 10 ms of game
time (one control step). The 100 ms example instead represents a 1,000 ms model call
at that slowdown. Native HTTP action acceptance also observes its next-step boundary.

The example uses LAGEN's action queues and configured inference capacity.
It does not reproduce the upstream HTTP service's per-seat request limits or Harbor's
direct-play audit. Those belong to a separate protocol-alignment experiment.

## Environment contract

- Task: `roboolympics_tabletennis`, integration: `roboolympics`.
- Execution: one simulated environment, inline policy inference, high-level native commands.
- Clock: 100 raw control steps/s, with 10 physics substeps per control step. Default observations: 20/s.
- Actions: upstream JSON commands, including `act`/`hit`, `move`, `toss`, and `none`.
  `Action(value=None, is_noop=True)` advances time without submitting a command.
  The native `none` command cancels the current stroke. Keep `hold_policy: one_frame_then_noop`
  so the scheduler submits each command once. The robot continues its installed trajectory between commands.
- Observation data: the upstream public `{state, observation}` view, including its PNG camera sheet.
  The sheet contains two cameras and four historical frames per camera, shape `(720, 1920, 3)`.
  Metadata supplies the public action specification and episode seed to the reference player.
  Hidden simulator state is not supplied to the policy.
- Reward: the change in own points minus the change in opponent points.
- Episode: one whole match, including all points. A point boundary resumes play and signals
  `soft_reset` to discard outstanding actions and inference results. Reviews take zero simulated time.
  The reference player retains its match state across those boundaries and restarts at step zero of a new match.
- End: native match completion sets `done`. The configured raw-step limit sets `truncated`.
  `task_metrics` records both states, final points, and match success.
  Success follows the native referee's winner, including disqualification decisions.
- Invalid commands: upstream refusal or execution rejection sets `invalid_action` and records the reason.
  The world still advances one control step.

The adapter uses the in-process simulator. It does not start the upstream HTTP service or its wall clock.
Our executor owns latency and simulated time. The experiment is a latency-bench integration check,
not the upstream Harbor protocol's official evaluation.

## Test on the server

Keep source under `/home/ubuntu/lxy/code/roboolympics-integration/`, runtime under
`/home/ubuntu/lxy/runtime/roboolympics-integration/`, and artifacts under
`/home/ubuntu/lxy/outputs/roboolympics-integration/`.
Use the SSH proxy specified in the main checkout's `servers.md`.

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
"$PYTHON" -m pytest -q tests/envs/adapters/test_roboolympics_env_adapter.py
```

The adapter tests use the bundled source and real simulator. They skip when the optional
MuJoCo 3.3.7 runtime is absent. Keep logs, resolved configs, dependency versions, and full run reports
in `outputs/`. Report the upstream revision, benchmark revision and local diff, opponent level,
episode count, seed, clock frequencies, and latency setting with any results.
