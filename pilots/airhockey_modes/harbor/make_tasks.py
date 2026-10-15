#!/usr/bin/env python3
"""Build the two Harbor tasks of the air-hockey two-mode pilot (not published; for the pilot's model runs):

  airhockey-pilot-match-high   Agent as Athlete: play a game (x10, true state) against the scripted opponent with the
                               platform's skills move / strike; deliverable /app/output/skill/player.py (class Player),
                               evaluated after the match: three games (L1-L3) with the platform's skills
  airhockey-pilot-match-low    Agent as Robotics Engineer: play the game with your own strategy and your own
                               implementation of the skill API from the low-level tools (robot model, IK library,
                               joint plans); deliverable /app/output/skill/skill.py (class Impl), evaluated after the
                               match by the API test (hidden requests) + the competition test (our reference strategy
                               on your implementation, L1-L3)

    python3 pilots/airhockey_modes/harbor/make_tasks.py OUT_DIR

The opponent level and seed come from the run's environment (RCB_LEVEL, RCB_SEED). The sim sidecar runs the metering
proxy (players/metering/think_proxy.py; credentials stay there) and the match service (rt_server.py); the player's
container has Python + numpy / SciPy / mujoco and the toolkit in /opt/robot; the verifier is the simulator image with
the evaluation (hidden cases root-only; the deliverable runs as nobody in sandbox.py)."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PG = HERE.parent
ROOT = PG.parents[1]
IMAGE = "ghcr.io/mll-lab-nu/eai-olympiad:0.1.0"
PY = "docker.io/library/python:3.12.14-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f"
PIPS = "mujoco==3.3.7 absl-py==2.5.0 etils==1.14.0 fsspec==2026.9.0 glfw==2.10.2 PyOpenGL==3.1.10 typing_extensions==4.16.0 zipp==4.1.0"
CLIENT_REQ = """certifi==2026.7.22
charset-normalizer==3.5.2
idna==3.20
numpy==2.5.3
pillow==12.3.0
requests==2.34.2
scipy==1.18.1
urllib3==2.8.0
"""
SLOWDOWN = 10
AGENT_TIMEOUT = 5400
SIM_FILES = ("world.py", "planner.py", "contract.py", "rt_server.py")
EVAL_FILES = ("world.py", "planner.py", "contract.py", "harness.py", "apitest.py", "sandbox.py", "sandbox_child.py",
              "apitest_cases.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sport_bundle(dst: Path):
    """sports/airhockey (the simulator: hockey.py + scene/) and the KUKA limits (bodies/limits.py + body.toml)"""
    (dst / "sports" / "airhockey").mkdir(parents=True)
    shutil.copy(ROOT / "sports" / "airhockey" / "hockey.py", dst / "sports" / "airhockey" / "hockey.py")
    shutil.copytree(ROOT / "sports" / "airhockey" / "scene", dst / "sports" / "airhockey" / "scene")
    (dst / "bodies" / "kuka_iiwa14").mkdir(parents=True)
    shutil.copy(ROOT / "bodies" / "limits.py", dst / "bodies" / "limits.py")
    shutil.copy(ROOT / "bodies" / "kuka_iiwa14" / "body.toml", dst / "bodies" / "kuka_iiwa14" / "body.toml")


def pilot_bundle(dst: Path, files, level=None):
    d = dst / "pilots" / "airhockey_modes"
    (d / "toolkit" / "lib").mkdir(parents=True)
    for f in files:
        shutil.copy(PG / f, d / f)
    shutil.copy(PG / "toolkit" / "lib" / "ahkin.py", d / "toolkit" / "lib" / "ahkin.py")
    shutil.copy(PG / "toolkit" / "iiwa_mallet.xml", d / "toolkit" / "iiwa_mallet.xml")
    import spec as S
    for lv in ([level] if level else ("high", "low")):
        (d / f"spec_{lv}.json").write_text(json.dumps(S.spec(lv), indent=1))
    return d


def sim_bundle(dst: Path, level: str):
    sport_bundle(dst)
    pilot_bundle(dst, SIM_FILES, level)
    (dst / "metering").mkdir()
    shutil.copy(ROOT / "players" / "metering" / "think_proxy.py", dst / "metering" / "think_proxy.py")
    (dst / "core").mkdir()
    shutil.copy(HERE / "launch.py", dst / "core" / "launch.py")


def toolkit_bundle(dst: Path, level: str):
    """/opt/robot: the manifest, the client; low: the robot model, the kinematics library, the public API cases"""
    dst.mkdir(parents=True)
    shutil.copy(HERE / "client.py", dst / "client.py")
    man = (PG / "toolkit" / f"toolkit_{level}.toml").read_text()
    if level == "low":
        shutil.copy(PG / "toolkit" / "iiwa_mallet.xml", dst / "iiwa_mallet.xml")
        (dst / "lib").mkdir()
        shutil.copy(PG / "toolkit" / "lib" / "ahkin.py", dst / "lib" / "ahkin.py")
        man = man.replace('file = "toolkit/iiwa_mallet.xml"', 'file = "/opt/robot/iiwa_mallet.xml"')
        man = man.replace('module = "toolkit/lib/ahkin.py"', 'module = "/opt/robot/lib/ahkin.py"')
        cases = json.loads((PG / "apitest_cases.json").read_text())["cases"]
        pub = [c for c in cases if c["public"]]
        (dst / "apitest_cases_public.json").write_text(json.dumps(
            {"about": "public examples of the API test's cases: the recorded state (puck, puck_v, the arm's q, qd, "
                      "q_target, q_target_v, servo = the setpoint generator's [q, qd, qdd] of the 7 iiwa axes), the "
                      "request (times relative to the case start, t = 0 at the request), the ideal outcome; the hidden "
                      "cases come from other games of the same reference player",
             "cases": pub}))
    (dst / "toolkit.toml").write_text(man)


INSTR_COMMON = """
## The game
One game of air hockey (USAA rules, first to 7 goals, else the 120 s cap of play and the tiebreak chain) against the
scripted opponent (its level is in `game spec`). You are seat A: your half is x < 0, you attack the goal at x = +0.974.
The arms are KUKA LBR iiwa 14s held to the real robot's joint limits, so a mallet moves at about 1 m/s and a puck
leaves a hit at about 1.4 m/s. The world runs in **real time at 1/{S} of real speed** and never waits for you; after
every rally (goal, seven-second foul, dead puck) it freezes until you send your review: `game ready "WHY" "PLAN"`. If
your session ends before the game is over, you lose. **Pilot simplification: the observation is the TRUE state** (the
puck's position and velocity, both mallets, your arm); no camera.

Use the `game` command (`game spec`, `game status`, `game start`, `game observe`, `game act JSON`, `game wait`,
`game ready WHY PLAN`) or the Python client `/opt/robot/client.py` (`from client import Game`). Everything (rules,
frames, commands, limits, the deliverable) is in `game spec`; read it first. Write a program that observes and acts
every 20 ms of game time (200 ms of real time at 1/{S}); watch its log, stop, fix and restart it between rallies.

Your container: Python 3.12 with numpy, SciPy, Pillow, mujoco 3.3.7; `/opt/robot` holds the toolkit
(`toolkit.toml` lists it). No internet during the game. Keep your files in `/app/output/`.
"""

INSTR_HIGH = """# Air hockey, HIGH level (Agent as Athlete) - two-mode pilot

You play air hockey with a KUKA iiwa 14 arm holding a mallet. The platform gives you its skills: `move` (the mallet to
a point and held there) and `strike` (you say when and where the puck will be and in which direction and how fast the
mallet should hit it; the platform winds up, settles, times the drive and follows through within the arm's limits).
You perceive (true state), predict the puck, and decide where to defend, when and where to strike, at which part of
the goal, straight or off a rail.
{common}
## Your deliverable (graded after the match)
`/app/output/skill/player.py` with `class Player` (`__init__(self, spec)`, `act(self, view) -> commands`), called
every 20 ms with the observation; commands are the same `move` / `strike` / `none` you send in the game (see
`game spec` `deliverable`). After the match the platform runs your FINAL player.py, alone, in an isolated container,
with the platform's skills in three games against the scripted opponent at levels 1, 2 and 3. Keep it
self-contained (numpy + your files), deterministic, no network. Improve it after every rally; it is snapshot at every
review.
"""

INSTR_LOW = """# Air hockey, LOW level (Agent as Robotics Engineer) - two-mode pilot

You play air hockey with a KUKA iiwa 14 arm holding a mallet **from the low-level tools only**: you command the arm
with **plans of joint targets** (`plan`: rows for the next 10 ms control steps of the 9 joints, executed by the robot's
servo within the real joint ranges, velocity, acceleration and jerk limits; `hold`). The toolkit in `/opt/robot` gives
you the robot model (`iiwa_mallet.xml`, the arm alone in the table frame) and a kinematics library (`lib/ahkin.py`:
forward kinematics, Jacobians, damped-least-squares IK keeping the mallet flat, the limits). **There are no skills**:
the high level's `move` and `strike` do not exist here. You build them.
{common}
## Your deliverable (graded after the match)
`/app/output/skill/skill.py` with `class Impl` implementing **our high-level skill API** for this arm:
`__init__(self, spec)`, `move(self, view, x, y) -> commands`, `strike(self, view, req) -> commands`,
`step(self, view) -> commands` (the contract: `game spec` `skill_api` and `deliverable`). `req = {{t, x, y, dx, dy,
speed}}` asks: at game time t the puck centre is at (x, y); hit it with the mallet moving along the direction (dx, dy)
at `speed` m/s (as fast as the arm can if that is faster). `move` brings the mallet to (x, y) at rest. Your commands
are `plan` / `hold`. Use it in your own game: your strategy decides the moves and strikes, your Impl executes them.
After the match the platform takes your FINAL `/app/output/skill/skill.py` (with any modules of yours in that
directory) and evaluates it alone, in an isolated container:
1. **API test**: hidden strike requests from our reference player's games (public examples in
   `/opt/robot/apitest_cases_public.json`), each started from the recorded state of the arm and the puck, measured
   against the contract (contact time, mallet position, the puck's outgoing speed and direction, goals);
2. **competition test**: our fixed reference strategy calls YOUR `move` / `strike` in games against the scripted
   opponent at levels 1, 2 and 3.
Only the toolkit, numpy / mujoco and your own files; deterministic; no network; no state from the match process.
Improve it after every rally; it is snapshot at every review.
"""


def task_toml(level):
    name = f"airhockey-pilot-match-{level}"
    return f'''schema_version = "1.3"
artifacts = [
  "/app/output/",
  {{ source = "/data/final.json", service = "sim" }},
  {{ source = "/data/skill_snapshots/", service = "sim" }},
  {{ source = "/data/think_usage_A.json", service = "sim" }},
  {{ source = "/data/think_calls_A.jsonl", service = "sim" }},
]

[task]
name = "robot-coding-bench/{name}"
description = "RoboOlympics two-mode pilot (not published): air hockey with the real iiwa 14 limits (v5), {level} level, program play, a game vs the scripted opponent at x{SLOWDOWN}, true state; post-match evaluation of the deliverable in the isolated verifier."
authors = [{{ name = "RoboOlympics" }}]
keywords = ["program-play", "airhockey", "pilot", "two-mode", "{level}-level"]

[metadata]
revision = "airhockey-modes-pilot-1"
category = "robotics-sports-arena"
simulator = "mujoco"
robot = "kuka_iiwa14"
play = "program"
action_level = "{level}"

[[seats]]
name = "A"
service = "player"
agent_kwargs = {{ think_proxy_url = "http://sim:18090/v1" }}
setup_networks = ["egress"]

[verifier]
timeout_sec = 3600.0
environment_mode = "separate"
network_mode = "no-network"

[[verifier.collect]]
service = "sim"
command = "for i in $(seq 1 90); do [ -f /data/server.pid ] && break; sleep 2; done; kill -USR1 $(cat /data/server.pid); for i in $(seq 1 60); do grep -q '\\"collected\\": true' /data/final.json 2>/dev/null && break; sleep 1; done; head -c 1500 /data/final.json"
timeout_sec = 300

[agent]
timeout_sec = {AGENT_TIMEOUT}.0

[environment]
build_timeout_sec = 1200.0
cpus = 1
memory_mb = 2048
'''


COMPOSE = """# The pilot's sim sidecar: the metering proxy (credentials stay here) + the match service (rt_server.py).
services:
  main:
    depends_on: [sim]
    volumes: [agent_output:/app/output]
  sim:
    image: {IMAGE}
    volumes:
      - ./sim:/srv:ro
      - agent_output:/agent_output:ro
    working_dir: /srv
    command: python3 /srv/core/launch.py
    cpus: 2
    mem_limit: 4g
    pids_limit: 256
    networks: [default, play]
    environment:
      RCB_DATA: "/data"
      OPENBLAS_NUM_THREADS: "1"
      OMP_NUM_THREADS: "1"
      RO_SEATS: "A"
      RO_SEED: "${{RCB_SEED:-101}}"
      RO_LEVEL: "${{RCB_LEVEL:-1}}"
      RO_SLOWDOWN: "${{RCB_SLOWDOWN:-{S}}}"
      RCB_ACTION_LEVEL: "{level}"
      START_TIMEOUT_S: "1800"
      READY_TIMEOUT_S: "600"
      AGENT_OUTPUT: "/agent_output"
      RO_SEAT_SELECT_A: "${{RO_SEAT_SELECT_A:-}}"
      RO_SEAT_ENV_THINK_MODEL_A: "${{THINK_MODEL_A:-}}"
      RO_SEAT_ENV_THINK_OPENAI_ROUTE_A: "${{THINK_OPENAI_ROUTE_A:-}}"
      RO_SEAT_ENV_AZURE_AI_API_KEY_A: "${{AZURE_AI_API_KEY_A:-}}"
      RO_SEAT_ENV_AZURE_AI_ENDPOINT_A: "${{AZURE_AI_ENDPOINT_A:-}}"
      RO_SEAT_ENV_AZURE_DEPLOYMENTS_A: "${{AZURE_DEPLOYMENTS_A:-}}"
      RO_SEAT_ENV_AZURE_SOL_API_KEY_A: "${{AZURE_SOL_API_KEY_A:-}}"
      RO_SEAT_ENV_AZURE_SOL_DEPLOYMENTS_A: "${{AZURE_SOL_DEPLOYMENTS_A:-}}"
      RO_SEAT_ENV_AZURE_SOL_ENDPOINT_A: "${{AZURE_SOL_ENDPOINT_A:-}}"
      RO_SEAT_ENV_CLAUDE_CODE_OAUTH_TOKEN_A: "${{CLAUDE_CODE_OAUTH_TOKEN_A:-}}"
      RO_SEAT_ENV_OPENROUTER_API_KEY_A: "${{OPENROUTER_API_KEY_A:-}}"
      AZURE_AI_API_KEY: "${{AZURE_AI_API_KEY:-}}"
      AZURE_AI_ENDPOINT: "${{AZURE_AI_ENDPOINT:-}}"
      AZURE_DEPLOYMENTS: "${{AZURE_DEPLOYMENTS:-}}"
      CLAUDE_CODE_OAUTH_TOKEN: "${{CLAUDE_CODE_OAUTH_TOKEN:-}}"
      OPENROUTER_API_KEY: "${{OPENROUTER_API_KEY:-}}"
      THINK_OPENAI_ROUTE: "${{THINK_OPENAI_ROUTE:-openrouter}}"
  player:
    build:
      context: .
    depends_on: [sim]
    command: ["sh", "-c", "sleep infinity"]
    cpus: 2
    mem_limit: 4g
    pids_limit: 128
    volumes: [agent_output:/app/output]
    networks: [play, egress]
    environment:
      SEAT: A
      GAME_URL: "http://sim:8000"
networks:
  play:
    internal: true
  egress: {{}}
volumes:
  agent_output: {{}}
"""

PLAYER_DOCKERFILE = f"""# The player's container: Python + numpy / SciPy / mujoco and the toolkit (/opt/robot); no simulator.
FROM {PY}
COPY client-requirements.txt /opt/client-requirements.txt
RUN pip install --no-cache-dir -r /opt/client-requirements.txt
RUN pip install --no-cache-dir -c /opt/client-requirements.txt {PIPS} && pip check
COPY seat_files/opt/robot /opt/robot
COPY game /usr/local/bin/game
RUN chmod +x /usr/local/bin/game && chmod -R a+rX /opt/robot && mkdir -p /app/output/skill
ENV SEAT=A
WORKDIR /app
"""

VERIFIER_DOCKERFILE = f"""# The isolated evaluation: the simulator image, the evaluation code and hidden cases (root-only), the toolkit in
# /opt/robot (for the deliverable), the sandbox child readable by nobody.
FROM {IMAGE}
COPY eval /tests/eval
COPY robot /opt/robot
COPY test.sh grade.py /tests/
RUN chmod 700 /tests/eval && chmod -R a+rX /opt/robot && mkdir -p /opt/ahm && \\
    cp /tests/eval/pilots/airhockey_modes/sandbox_child.py /opt/ahm/sandbox_child.py && chmod -R a+rX /opt/ahm && \\
    chmod +x /tests/test.sh
"""

TEST_SH = """#!/bin/bash
# Verifier (isolated): the match result (/data/final.json from the sim) + the post-match evaluation of the deliverable.
mkdir -p /logs/verifier
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 AHM_TOOLKIT_DIR=/opt/robot AHM_SANDBOX_CHILD=/opt/ahm/sandbox_child.py
python3 /tests/grade.py 2> /logs/verifier/grade.log
"""

GRADE = '''"""The pilot's verifier: the match (from the sim's /data/final.json), then the FINAL deliverable in
/app/output/skill evaluated here (isolated, no network): high -> player.py in three games (L1-L3) with the platform's
skills; low -> the API test (hidden) + the competition test of skill.py. Writes /logs/verifier/reward.json (reward =
the post-match headline: high = games won / 3, low = the API-test success rate) and /logs/verifier/evaluation.json."""
import json
import os
import shutil
import sys
from pathlib import Path

LEVEL = "{level}"
sys.path.insert(0, "/tests/eval/pilots/airhockey_modes")
LOG = Path("/logs/verifier")
LOG.mkdir(parents=True, exist_ok=True)


def main():
    try:
        final = json.loads(Path("/data/final.json").read_text())
    except Exception as e:  # noqa: BLE001
        final = {{"error": f"no final.json: {{e}}"}}
    m = final.get("match") or {{}}
    sc = final.get("score") or [0, 0]
    out = {{"level_low": int(LEVEL == "low"), "opp_level": final.get("opponent_level"), "match_won": int(bool(final.get("won"))),
           "match_goals_for": sc[0], "match_goals_against": sc[1], "match_complete": int(bool(final.get("complete"))),
           "match_abandoned": int(bool(final.get("abandoned"))), "match_no_show": int(bool(final.get("no_show"))),
           "match_refusals": m.get("refusals"), "match_strikes": m.get("strikes"), "match_plans": m.get("plans"),
           "match_own_goals_against": m.get("own_goals_against"), "match_rallies": m.get("rallies")}}
    skill = Path("/app/output/skill")
    work = Path("/tmp/deliverable")
    if skill.is_dir():
        shutil.copytree(skill, work, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        os.system(f"chmod -R a+rX {{work}}")
    ev = {{}}
    import apitest
    p = work / ("player.py" if LEVEL == "high" else "skill.py")
    out["deliverable_present"] = int(p.exists())
    if p.exists():
        ev = apitest.evaluate(LEVEL, str(work))
        if LEVEL == "high":
            G = ev["high_test"]
            out.update(high_won=G["won"], high_games=G["games"], high_goals_for=G["goals_for"],
                       high_goals_against=G["goals_against"], high_errors=G["errors"], high_refusals=G["refusals"])
            out["reward"] = round(G["won"] / max(1, G["games"]), 4)
        else:
            A, C = ev["api_test"], ev["competition"]
            for k in ("n", "success", "contact", "ok_time", "ok_pos", "ok_puck_dir", "ok_puck_speed", "goals",
                      "own_goals", "ideal_goals", "refusals", "errors"):
                out[f"api_{{k}}"] = A[k]
            out["api_success_rate"] = A["success_rate"]
            for k in ("med_time_err_s", "med_pos_err_m", "med_puck_speed_err", "med_puck_dir_err_deg"):
                out[f"api_{{k}}"] = A[k]
            out.update(comp_games=C["games"], comp_won=C["won"], comp_goals_for=C["goals_for"],
                       comp_goals_against=C["goals_against"], comp_refusals=C["refusals"], comp_errors=C["errors"])
            out["reward"] = A["success_rate"]
    else:
        out["reward"] = 0.0
    out = {{k: (0 if v is None else v) for k, v in out.items()}}
    (LOG / "evaluation.json").write_text(json.dumps({{"final": final, "evaluation": ev}}, default=str))
    (LOG / "reward.json").write_text(json.dumps(out))
    print(json.dumps(out))


main()
'''


def build(out: Path, level: str):
    t = out / f"airhockey-pilot-match-{level}"
    if t.exists():
        shutil.rmtree(t)
    env = t / "environment"
    env.mkdir(parents=True)
    (t / "task.toml").write_text(task_toml(level))
    common = INSTR_COMMON.format(S=SLOWDOWN)
    (t / "instruction.md").write_text((INSTR_HIGH if level == "high" else INSTR_LOW).format(common=common))
    (env / "docker-compose.yaml").write_text(COMPOSE.format(IMAGE=IMAGE, S=SLOWDOWN, level=level))
    (env / "Dockerfile").write_text(PLAYER_DOCKERFILE)
    (env / "client-requirements.txt").write_text(CLIENT_REQ)
    shutil.copy(HERE / "game", env / "game")
    sim_bundle(env / "sim", level)
    toolkit_bundle(env / "seat_files" / "opt" / "robot", level)
    tests = t / "tests"
    tests.mkdir()
    ev = tests / "eval"
    sport_bundle(ev)
    d = pilot_bundle(ev, EVAL_FILES)
    (d / "programs").mkdir()
    for f in ("ref_strategy.py", "ahref.py"):
        shutil.copy(PG / "programs" / f, d / "programs" / f)
    shutil.copytree(env / "seat_files" / "opt" / "robot", tests / "robot")
    if level == "high":                  # the platform's skills in the verifier need the toolkit's model too
        shutil.copy(PG / "toolkit" / "iiwa_mallet.xml", tests / "robot" / "iiwa_mallet.xml")
    (tests / "Dockerfile").write_text(VERIFIER_DOCKERFILE)
    (tests / "test.sh").write_text(TEST_SH)
    (tests / "grade.py").write_text(GRADE.format(level=level))
    sol = t / "solution"
    (sol / "skill").mkdir(parents=True)
    shutil.copy(HERE / "play_ref.py", sol / "play_ref.py")
    for f in ("ref_strategy.py", "ahref.py"):
        shutil.copy(PG / "programs" / f, sol / "skill" / f)
    if level == "high":
        shutil.copy(PG / "programs" / "ref_high_player.py", sol / "skill" / "player.py")
    else:
        shutil.copy(PG / "programs" / "ref_low_skill.py", sol / "skill" / "skill.py")
        shutil.copy(PG / "planner.py", sol / "skill" / "planner.py")
    (sol / "solve.sh").write_text("#!/bin/bash\n# Oracle: our references (high: the reference player; low: the reference "
                                  "strategy + our low-level reference implementation of the skill API).\nset -e\n"
                                  "mkdir -p /app/output\ncd /solution && python3 play_ref.py\n")
    (sol / "solve.sh").chmod(0o755)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    a = ap.parse_args()
    sys.path.insert(0, str(PG))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for level in ("high", "low"):
        print(build(out, level))


if __name__ == "__main__":
    main()
