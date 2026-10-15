# Frozen RoboOlympics sports

The adapter runs each upstream single-seat match against its scripted opponent.
One policy controls the player or the whole team. The example policies are upstream
camera-only reference players.

| Task suffix | Native format / action level | Environment Hz | Observation Hz | Opponent / public seed |
| --- | --- | ---: | ---: | --- |
| `tabletennis` | `match` / high | 100 | 20 | L1 / 1 |
| `badminton` | `match-wallclock` / high | 100 | 20 | L1 / 101 |
| `soccer` | `match` / low | 50 | 25 | L2 / 202 |
| `soccer_ssl` | `match` / high | 100 | 20 | L2 / 1 |

Task names start with `roboolympics_`. Each has a zero-latency config and a
`_100ms.yaml` config under `configs/examples/roboolympics/`.

## Install the isolated CPU runtime

The frozen upstream source is included as the Git subtree `third_party/roboolympics`,
at upstream revision `da1f787f775ea6524d698c0b063157ded4c0fef9`.
The adapter locates it relative to the benchmark checkout. Use Python 3.11:

```bash
conda create --override-channels -c conda-forge -p /path/to/runtime python=3.11 pip
/path/to/runtime/bin/python -m pip install -r scripts/experiments/roboolympics/requirements.txt
/path/to/runtime/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
/path/to/runtime/bin/python -m pip install --no-deps sai-mujoco==0.1.30
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
bash scripts/experiments/roboolympics/eval.sh --config configs/examples/roboolympics/badminton.yaml
bash scripts/experiments/roboolympics/eval.sh --config configs/examples/roboolympics/soccer.yaml
bash scripts/experiments/roboolympics/eval.sh --config configs/examples/roboolympics/soccer_ssl.yaml
```

Use `--output-dir` to select an artifact directory. Use `--max-env-steps 20` for a quick
smoke run.
The table lists the default opponents and seeds, taken from each sport's public
protocol. Each new sport's raw-step limit is 200,000. Full matches retain the
upstream duration, score limits, halves and tie-break rules.

Humanoid soccer uses only the T1 model and mesh files from `sai-mujoco==0.1.30`,
the same version as the upstream soccer image. Install it with `--no-deps` because
its environment dependencies differ from the frozen simulator's pins. The adapter
locates the assets in the installed package. It does not import its environments.
The bundled walking controller is `sports/soccer/models/robot/lower_t1_control.pt`,
from `SaiResearch/booster_soccer_models` revision
`c742ca8668fa88011035ee2505e043bd36488913` (upstream `models/NOTICE.md`).

Run each sport in a separate Python process. Upstream modules use overlapping
names such as `skill_api` and `builtin_players`. Loading a second sport in the
same process raises an error. Repeated episodes of the same sport are supported.

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

Action release time belongs to latency bench. The wheel-football service's
`reaction_delay_steps = 2` is not applied here. Zero latency adds no artificial
waiting period. Fixed 100 ms releases a command after 10 control steps in table
tennis, badminton and wheel football, or 5 steps in humanoid soccer.

Badminton's HTTP planner normally reserves a fixed lead before command execution.
The adapter sets that lead to zero and reports zero in the public action spec.
It still runs the native snapshot, planning and lowering sequence at the scheduler's
release step. Physical movement and stroke durations remain native. Native camera
delays also remain, including wheel football's 40 ms vision delay. The executor
does not convert measured CPU planning or rendering time into extra simulated delay.

## Environment contract

- Integration: `roboolympics` for all four tasks.
- Execution: one simulated environment, inline policy inference, native JSON commands.
- Clock: each sport's native control step, as listed above. Environment Hz and
  observation Hz are separate settings.
- Actions: racket sports use `act`/`hit`, `move`, `toss`, and `none` as declared by
  their native schemas. Humanoid soccer uses `walk`, `stand` and `kick`, with the
  robot number in `entity`. Wheel football uses `move`, `kick`, `dribble`,
  `substitute`, and `batch` for commands to several robots.
  `Action(value=None, is_noop=True)` advances time without submitting a command.
  For racket sports, native `none` cancels the current stroke. Keep `hold_policy: one_frame_then_noop`
  so the scheduler submits each command once. The robot continues its installed trajectory between commands.
- Observation data: the upstream public `{state, observation}` view, including its PNG camera sheet.
  Racket sports show two cameras with four historical frames each, shape `(720, 1920, 3)`.
  Humanoid soccer shows two head cameras. Wheel football shows four overhead cameras,
  retaining their native capture timestamps and vision delay.
  Metadata supplies the public action specification and episode seed to the reference player.
  Hidden simulator state is not supplied to the policy.
- Reward: the change in own points minus the change in opponent points.
- Episode: one whole match, including all points or goals. A unit boundary resumes play and signals
  `soft_reset` to discard outstanding actions and inference results. Reviews take zero simulated time.
  The reference player runs its native review hook at unit boundaries and restarts
  at step zero of a new match.
- End: native match completion sets `done`. The configured raw-step limit sets `truncated`.
  `task_metrics` records both states, final points, and match success.
  Success follows the native referee's winner, including disqualification decisions.
- Invalid commands: upstream refusal or execution rejection sets `invalid_action` and records the reason.
  The world still advances one control step.

The adapter uses the in-process simulator. It does not start the upstream HTTP service or its wall clock.
Our executor owns latency and simulated time. The experiment is a latency-bench integration check,
not the upstream Harbor protocol's official evaluation.

## Test on the server

Keep source under `/home/ubuntu/lxy/code/roboolympics-frozen/`, reuse the isolated
runtime under `/home/ubuntu/lxy/runtime/roboolympics-integration/`, and save artifacts
under `/home/ubuntu/lxy/outputs/roboolympics-frozen/`.
Use the SSH proxy specified in the main checkout's `servers.md`.

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
"$PYTHON" -m pytest -q tests/envs/adapters/test_roboolympics_env_adapter.py
"$PYTHON" -m pytest -q tests/envs/adapters/test_roboolympics_frozen_sports.py
```

The adapter tests use the bundled source and real simulator. They skip when the optional
MuJoCo 3.3.7 runtime is absent. Keep logs, resolved configs, dependency versions, and full run reports
in `outputs/`. Report the upstream revision, benchmark revision and local diff, opponent level,
episode count, seed, clock frequencies, and latency setting with any results.
