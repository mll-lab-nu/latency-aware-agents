#!/usr/bin/env python3
"""The v0.2 generator (docs/framework_zh.md §6, §7): a sport's declarations (sports/<sport>/sport.toml [[tasks]])
x the core -> self-contained Harbor task directories. Only this writes generated task directories.

    python3 tools/generate.py [SPORT ...]          write the tasks of every sport with status "sport"
    python3 tools/generate.py check [SPORT ...]    exit 1 if a written task differs from what would be written
    python3 tools/generate.py --out DIR SPORT      write into DIR (the checks' build directory)

Where: tasks/<id>/ for a sport with publish = true, build/tasks/<id>/ (gitignored) for publish = false (the
example sport). Task id: <sport>-<format>-<play>[-lowlevel]-standard ("-lowlevel" only when the sport also
has a high level). No sport name appears in this file; python >= 3.11, stdlib only.

A task directory:
  task.toml, instruction.md, README.md           rendered from sport.toml (rules, deviations, tiebreaks, timing)
  environment/docker-compose.yaml                the sim service (core/launch.py) + one container per seat
  environment/sim/                               tools/sportlib.assemble_sim: core/, renderers/, metering/, sport/
  environment/Dockerfile, [seat/]Dockerfile      the player image: standard client pins + the generic `game`
  tests/                                         core/grading.py, the sport's grade.py, contract.json, nocode audit
  solution/                                      the sport's reference player over HTTP (core/client.py)

Optional declarations (absent = nothing changes in the written task):
  [seat] / [seat.<action level>]  the player image's extras, for every task / that action level's tasks only:
                                  pip = ["pkg==version", ...] (installed with the client pins as constraints, then
                                  `pip check`), files = {"/image/path" = "path in sports/<sport>/"}, run = ["shell
                                  command run at build time", ...], note = "one sentence for the prompt"
  [body] name, hand, controllers  a robot of the robot library (bodies/README.md) with its attachment and the control
                                  libraries the sport imports: the simulator directory gets them with configuration.json
                                  (tools/sportlib.copy_body); task.toml names the robot and the configuration id
  [arm] body                      the real robot arm the sport's arms are held to (bodies/<body>/body.toml, enforced by
                                  bodies/limits.py): the simulator directory gets them (tools/sportlib.copy_arm_limits)
                                  and task.toml / README name the body as the robot; [arm] files = shared bodies/
                                  modules the simulator imports too (e.g. ["trajopt.py"])
  [legs] body                     a limits-only legged robot (bodies/__init__.py LEGS; the sport's scene carries the
                                  model): its body.toml and the limit / servo path (tools/sportlib.copy_leg_limits)
  [[tasks]] slowdown              realtime: the task's default slowdown (else [timing] slowdown)
  [[tasks]] agent_timeout_s       the agent's time budget (default 7200)
  [timing] replay_timeout_s,      the sim's limits after the match: the fresh-process replay (default 600 s), then the
           video_timeout_s        spectator video (default 300 s); the verifier's collect waits for both
  [official.hidden_seeds]         sha256 + counts per cell (format x level) of the sport's hidden official seed set (core 0.2.14,
                                  core/seeds.py, tools/seeds.py): contract.json gets them with the public commitments
                                  (sports/<sport>/seed_commitments.json, checked against the sha256); the sim service
                                  takes the seed's salt from RCB_SEED_SALT
  [score]                         the continuous score (core 0.2.14, core/score.py): contract.json and the README
  [modes.high] / [modes.low]      the two modes (skilleval/README.md): a high task's prompt names the skill API; a low
                                  task's prompt gets the engineering task (implement the skill API: deliverable,
                                  toolkit, practice requests, the post-match evaluation) and its task.toml the
                                  deliverable and the evaluation command
  [modes.low.inloop]              the skill in the loop (skilleval/inloop.py): a low task gets one skill container per
                                  seat (environment/skill/: the sim image + [modes.low] eval_pip, the toolkit at the
                                  seat's paths, skilleval's host; no network, read-only, no simulator directory) that
                                  runs the player's /app/output/skill (read-only) for the sim over a Unix socket
                                  (volume skillipc_<seat>_trusted); the sim gets RCB_SKILL_INLOOP_<seat>; the prompt
                                  the in-loop rules; solution/ the reference skill and the in-loop oracle
Play mode: program play only (play = "program"). Direct (no-code) play was removed from main on 2026-10-10 and kept on
branch archive/direct-play; the verifier's no-code audit (players/nocode/nocode_audit.py) is still in tests/ for every
task (core/grading.py applies it only to play = "direct").
"""
from __future__ import annotations

import filecmp
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sportlib  # noqa: E402

REPO = sportlib.REPO
PINS = REPO / "scripts" / "eai_standard" / "client-requirements.txt"
PY_IMAGE = ("docker.io/library/python:3.12.14-slim@sha256:"
            "f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f")
PROTOCOLS = {"realtime": "eai-standard-realtime/1.0", "turn": "eai-standard-turn/1.0"}
PER_SEAT = ("THINK_OPENAI_ROUTE", "THINK_MODEL",       # THINK_MODEL_<seat>: the seat's model lock (think_proxy)
            "THINK_CHATGPT_MODEL")                    # the ChatGPT-backend model a subscription seat is pinned to


def credentials() -> list[str]:
    """The credential variable NAMES the sim container passes to its metering proxies: every player's account_env
    (players/players.toml); the values stay in the runner's environment."""
    import tomllib
    plist = tomllib.loads((REPO / "players" / "players.toml").read_text()).get("player", [])
    return sorted({v for p in plist for v in p.get("account_env", [])})


def seat_only_credentials() -> set[str]:
    """The credential NAMES of players with account_scope = "seat" (e.g. a ChatGPT subscription): the sim container
    passes them per seat only (RO_SEAT_ENV_<VAR>_<seat> from <VAR>_<seat>), never unsuffixed, so in a multi-seat task
    no other seat's metering proxy has them."""
    import tomllib
    plist = tomllib.loads((REPO / "players" / "players.toml").read_text()).get("player", [])
    return {v for p in plist if p.get("account_scope") == "seat" for v in p.get("account_env", [])}


def agent_timeout(t) -> int:
    return int(t.get("agent_timeout_s", 7200))


def seat_extras(d, t) -> dict:
    """[seat] (every task) + [seat.<action level>] (that level's tasks): pip pins, files, build commands, a note."""
    top = d.get("seat") or {}
    lv = top.get(t.get("action_level", "high")) or {}
    return {"pip": list(top.get("pip", [])) + list(lv.get("pip", [])),
            "files": {**(top.get("files") or {}), **(lv.get("files") or {})},
            "run": list(top.get("run", [])) + list(lv.get("run", [])),
            "note": " ".join(x for x in (top.get("note"), lv.get("note")) if x)}


def robot_configuration(d) -> str:
    """task.toml's robot_configuration line for a sport declaring [body] (bodies/assembly.py), else nothing"""
    cid = sportlib.configuration_id(d)
    return f'\nrobot_configuration = "{cid}"' if cid else ""


def skill_lines(d, t) -> str:
    """task.toml lines of a low task of a two-mode sport: the deliverable and how it is evaluated after the match."""
    lo = (d.get("modes") or {}).get("low")
    if t.get("action_level", "high") != "low" or not isinstance(lo, dict):
        return ""
    if isinstance(lo.get("inloop"), dict):
        return (f'\nskill_deliverable = "{lo["deliverable"]["path"]}"\nskill_in_loop = true\nskill_evaluation = '
                f'"python3 tools/skill_eval.py {d["sport"]["name"]} <the trial\'s artifact app/output/skill> '
                f'--inloop-log <the trial\'s artifact inloop_<seat>.jsonl>"')
    return (f'\nskill_deliverable = "{lo["deliverable"]["path"]}"\nskill_evaluation = "python3 tools/skill_eval.py '
            f'{d["sport"]["name"]} <the trial\'s artifact app/output/skill>"')


def inloop(d, t) -> dict | None:
    """[modes.low.inloop] of a low task of a two-mode sport (the skill in the loop), else None."""
    lo = (d.get("modes") or {}).get("low")
    if t.get("action_level", "high") != "low" or not isinstance(lo, dict) or not isinstance(lo.get("inloop"), dict):
        return None
    return lo["inloop"]


def skill_subdir(d) -> str:
    """The deliverable's directory relative to /app/output (e.g. "skill")."""
    return str(Path(d["modes"]["low"]["deliverable"]["path"]).parent.relative_to("/app/output"))


def robot_name(d) -> str:
    """the robot named in task.toml / README: the shared body ([body] name), the arm ([arm] body) or the legs ([legs]
    body), else none"""
    return (d.get("body") or {}).get("name") or (d.get("arm") or {}).get("body") or (d.get("legs") or {}).get("body") \
        or "none"


def task_id(d, t) -> str:
    """The task directory name: [[tasks]] id when given (a migrated sport keeps its legacy task name), else derived."""
    if t.get("id"):
        return t["id"]
    low = t.get("action_level", "high") == "low" and "high" in d.get("modes", {}).get("action_levels", ["high"])
    return f"{d['sport']['name']}-{t['format']}-{t.get('play', 'program')}{'-lowlevel' if low else ''}-standard"


def tasks_of(sport: str) -> list[tuple[str, dict]]:
    d = sportlib.decl(sport)
    return [(task_id(d, t), t) for t in d.get("tasks", [])]


def out_root(sport: str) -> Path:
    return REPO / ("tasks" if sportlib.decl(sport)["sport"].get("publish") else "build/tasks")


# ------------------------------------------------------------------------------------------------ text
def shortcuts_of(d, t) -> dict:
    """The `game` shortcuts of the task's action level: sport.toml [actions.cli.<level>] {action: [fields]}."""
    return d.get("actions", {}).get("cli", {}).get(t.get("action_level", "high"), {})


def solo_seat(d, fmt) -> bool:
    """One seat in the format (alone or against a scripted opponent): decided by the seats, not by the format's name,
    so a single-player format may keep a legacy task name such as `match`."""
    return len(d["formats"][fmt]["seats"]) == 1


def one_side(d, fmt) -> bool:
    """Every seat on one side (alone, against a scripted opponent, or several seats of one team against it)."""
    f = d["formats"][fmt]
    return len(set((f.get("sides") or {s: s for s in f["seats"]}).values())) == 1


def rules_text(d, fmt) -> str:
    sp, f = d.get("spec", {}), d["formats"][fmt]
    lines = [f"- {d['sport']['summary']}."]
    for k, v in sp.items():
        lines.append(f"- `{k}`: {json.dumps(v) if not isinstance(v, str) else v}")
    if solo_seat(d, fmt) and not f.get("opponent", False):
        line = f.get("success", {})
        bound = ("at least", line["score_at_least"]) if "score_at_least" in line else ("at most", line.get("score_at_most"))
        lines.append(f"- You play alone: success = a valid, complete match with a score of {bound[0]} {bound[1]}.")
    elif d.get("tiebreaks", {}).get("text"):
        lines.append(f"- Winner: {d['tiebreaks']['text']}. There are no draws.")
    for dv in d.get("deviations", []):
        if dv.get("affects_result", True):
            lines.append(f"- Deviation {dv['id']} (rulebook {dv['rule']}): {dv['text']}")
    return "\n".join(lines)


def reaction_delay_text(d) -> str:
    """The declared reaction delay (sport.toml [timing] reaction_delay_steps, core 0.2.18) for the prompt; nothing
    when the sport declares none."""
    D = (d.get("timing") or {}).get("reaction_delay_steps", 0)
    if not D:
        return ""
    return (f" Reaction delay: an action takes effect no earlier than {D} control step{'s' if D != 1 else ''} after "
            f"the step of the latest world state you were sent (`now.step` of any reply; "
            f"`timing.reaction_delay_steps` in `game spec`), the same for every player; if your program needs longer, "
            f"its own time counts instead. The act's reply says when it took effect (`took_effect.step`).")


def instruction(d, t, tid) -> str:
    fmt, proto = t["format"], d["sport"]["protocol"]
    clock = d.get("timing", {}).get("clock_s", 3600)
    cli = shortcuts_of(d, t)
    shortcuts = "\n".join(f"| `game {k} {' '.join(x.upper() for x in v)}` | `POST /act {{\"type\":\"{k}\", ...}}` | "
                          f"one action (`game spec` lists the fields and ranges) |" for k, v in cli.items())
    f = d["formats"][fmt]
    seats = ", ".join(f"`{x}`" for x in f["seats"])
    vs = ("against a scripted opponent (you are seat `A`)" if solo_seat(d, fmt) and f.get("opponent", False) else
          "alone" if solo_seat(d, fmt) else
          f"with your teammates against a scripted opponent (one seat per player: {seats}; your seat is `$SEAT`)"
          if one_side(d, fmt) and f.get("opponent", False) else
          "against an opponent model in the same world (seat `A` or `B`)")
    timing = (f"Nothing moves while you think: the world waits for the player whose turn it is. Each player has a "
              f"chess clock of {clock // 60} minutes of real time, running only on its own turn (thinking, tool "
              f"calls, your programs and your reviews all count). **If your clock runs out, you lose.**"
              if proto == "turn" else
              "The world runs in real time at 1/S of real speed (S = `timing.slowdown` in `game spec`); your thinking "
              "costs game time. At the end of every unit the world freezes for the reviews." + reaction_delay_text(d))
    review = ('After each of your actions (except one that ends the match), send `game ready "WHY" "PLAN"` (a short review) '
              'before the game goes on.' if proto == "turn" else
              'Whenever the world freezes for a review (after every unit: a point, a shot, a heat; see `game spec`), send '
              '`game ready "WHY" "PLAN"` (a short review) before the next unit starts.')
    session_end = ("The match is graded when your session ends: stay until it is over. **If your session ends before "
                   "the match is over, you lose the match** (a valid result, success 0), also if you never sent "
                   "`game start`."
                   if one_side(d, fmt) else "The match is graded when your session ends: stay until it is over. **If "
                   "you never send `game start`, your side loses the match** (unless no side started), also when the "
                   "sessions end before the start window closes.")
    ex = seat_extras(d, t)
    container = "Python 3.12 with numpy, SciPy and Pillow" + (f"; also {', '.join(ex['pip'])}" if ex["pip"] else "") \
        + (f". {ex['note']}" if ex["note"] else "")
    play_section = PROGRAM_SECTION.format(container=container) + modes_section(d, t)
    head = ("Use the `game` command, or the same calls as HTTP with JSON at `http://sim:8000/<SEAT>/` (your seat is "
            "`$SEAT`):")
    act_rows = "| `game act JSON` | `POST /act JSON` | one action |" + ("\n" + shortcuts if shortcuts else "")
    ri = d.get("observation", {}).get("reply_images")
    reply_images = (f"The reply to an action also carries images of its result, in the format of `game observe` "
                    f"(`observation.image_png_b64` and `observation.tiles`; `game` saves them in `/app/output/views/`): "
                    f"{ri}.\n\n" if ri else "")
    tc = d.get("team_channel")
    side_of = f.get("sides") or {}
    team_fmt = bool(tc) and any(list(side_of.values()).count(x) > 1 for x in set(side_of.values()))
    team_rows = ("\n| `game say \"TEXT\"` | `POST /say {\"text\":\"...\"}` | a message to your teammates (at most "
                 f"{tc.get('max_bytes', 120)} bytes, one every {tc.get('min_interval_s', 1.0):g} game seconds, "
                 f"{tc.get('max_messages', 200)} per match; the other side never sees it) |\n"
                 "| `game team [SINCE]` | `POST /team {\"since\": SINCE}` | your teammates' messages after seq SINCE "
                 "(default 0) |" if team_fmt else "")
    team_note = ("Team messages: `observe` and `status` also show `team_messages` (from, seq, k, text) and `team_seq` (the "
                 "newest seq). Reading never consumes them: every client of your seat (you, your program, a monitor) "
                 "sees the same messages; send `team_since` (the largest seq you have: in the `observe` / `wait` body, "
                 "or `GET /status?team_since=N`) to get only newer ones, else you get those of the last "
                 f"{float(tc.get('recent_s', 2.0)):g} game seconds.\n\n" if team_fmt else "")
    wait_row = ('| `game wait DT` | `POST /wait {"dt": DT}` | let DT game seconds pass (0 < DT <= 1) |'
                if proto == "realtime" else
                '| `game wait` | `POST /wait {"timeout_s": 60}` | block until it is your turn (or the match is over) |')
    return f"""# {d['sport']['title']}: program play ({fmt})

You play {d['sport']['title'].lower()} {vs}. {d['sport']['summary'][0].upper() + d['sport']['summary'][1:]}.
This task is generated from `sports/{d['sport']['name']}/sport.toml` (task `{tid}`).

## Time
{timing}

{review}
{session_end}

{play_section}

## Interface
{head}

| `game` command | HTTP | |
| --- | --- | --- |
| `game spec` | `GET /spec` | the actions (types, fields, ranges), the cameras with their intrinsics, the rule constants |
| `game status` | `GET /status` | the current state, no image |
| `game start` | `POST /start {{}}` | you are ready; the match begins when every seat has started |
{wait_row}
| `game observe` | `POST /observe {{}}` | the state and the camera images (one PNG of tiles: `observation.tiles`) |
{act_rows}
| `game ready "WHY" "PLAN"` | `POST /ready {{"why":"...","plan":"..."}}` | your review after each of your actions |{team_rows}

{reply_images}{team_note}Errors: 400 = an invalid action (nothing happens), 409 = not now (not your turn, or a review is due), 429 = one
action in flight, or more than 20 requests per second or 4 open requests from your seat (empty body: retry more slowly).

## Rules
{rules_text(d, fmt)}

## Geometry and physics
Everything you need is in `game spec`: the world frame, the release point, the cameras (pose, field of view, size,
`focal_px`, `principal_point_px`; pinhole: u = cx + f * x_right / depth, v = cy - f * y_up / depth) and the
execution scatter. The physics engine is {d['sport']['physics']}.
"""


def modes_section(d, t) -> str:
    """The two modes (sport.toml [modes.high] / [modes.low]); "" for a sport that does not declare them."""
    md = d.get("modes") or {}
    hi, lo, level = md.get("high"), md.get("low"), t.get("action_level", "high")
    if level == "high" and isinstance(hi, dict):
        names = ", ".join(f"`{s}`" for s in hi["skills"])
        return (f"\n\nYour actions are the sport's skill API ({names}): the platform executes each one; `game spec` "
                "(`skill_contract`) says exactly what it delivers.")
    if level != "low" or not isinstance(lo, dict):
        return ""
    dl, cs = lo["deliverable"], lo["cases"]
    fn = Path(dl["path"]).stem
    tl = ", ".join(f"{k} {v} s" for k, v in dl.get("time_limits_s", {}).items())
    opt = "; optional: " + "; ".join(f"`{x}`" for x in dl["optional"]) if dl.get("optional") else ""
    il = inloop(d, t)
    if il is not None:
        return inloop_section(d, lo, il, fn, tl, opt)
    return f"""

## Your engineering task (low level)
At this level you are the robotics engineer: the platform does not give you the skill {', '.join(f'`{s}`' for s in lo['implements'])} (the high
level's action); you implement it yourself from your toolkit, and you play the match as usual with your own strategy
and your own commands.

- Skill API: the contract (request ranges, what is measured, the tolerances, success) is `skill_contract` in `game spec`.
- Deliverable: `{dl['path']}` (plus any files it needs in `{str(Path(dl['path']).parent)}/`) with
  `{dl['signature']}`{opt}. It may use {dl['allowed']}. It must be deterministic; time limits per call: {tl}.
- Toolkit: `{lo['toolkit_path']}` lists everything you are given ({', '.join(lo['tools'])}); not given: {', '.join(lo['forbidden'])}.
- Practice requests: `{cs['practice_path']}` (public; the evaluation uses hidden requests of the same kind).
- After the match the platform takes your FINAL `{str(Path(dl['path']).parent)}/` and evaluates it in a separate isolated
  container (no network, its own process): the API test (hidden requests: the success rate under the contract) and
  the competition test (our fixed reference strategy calls your `{fn}`). The match itself is graded as before; improve
  your skill and your strategy after every review."""


def inloop_section(d, lo, il, fn, tl, opt) -> str:
    """The low task's engineering section when the skill is in the loop ([modes.low.inloop])."""
    dl, cs = lo["deliverable"], lo["cases"]
    sk = list(il.get("skills") or lo["implements"])
    names = ", ".join(f"`{s}`" for s in sk)
    sdir = str(Path(dl["path"]).parent)
    free = il.get("free") or []
    return f"""

## Your engineering task (low level): your skill plays the match
At this level you are the robotics engineer: the platform does not give you the skill {names} (the high level's
action); you implement it yourself from your toolkit, and **the match is played with your implementation in the loop**:
your program decides when and where, your skill turns each decision into the robot's commands.

- Skill API: the contract (request ranges, what is measured, the tolerances, success) is `skill_contract` in `game spec`.
- Deliverable: `{dl['path']}` (plus any files it needs in `{sdir}/`) with
  `{dl['signature']}`{opt}. It may use {dl['allowed']}. It must be deterministic; time limits per call: {tl}.
- Toolkit: `{lo['toolkit_path']}` lists everything you are given ({', '.join(lo['tools'])}); not given: {', '.join(lo['forbidden'])}.
- Practice requests: `{cs['practice_path']}` (public; the evaluation uses hidden requests of the same kind).

### The skill in the loop
- To play {names}, send it as an action: `game act '{{"type": "{sk[0]}", ...}}'` with the fields of `skill_contract`. The
  simulator calls your `{fn}.py` in a sandbox of its own (a separate container: no network, no simulator files, a
  private copy of `{sdir}/`, the toolkit at the same paths as in your container) with `state` = your observation
  without images (`state["step"]` = the control step your call arrived) and `spec` (`skill_contract`, the toolkit, the
  robot's kinematics and limits, and `timing`), and applies the low-level command your function returns.
- **Your skill's computation counts.** The world runs on while it computes (1 wall second = 1/S game seconds, S =
  `timing.slowdown`); its command takes effect at the first control step after it returns (the reply's
  `took_effect.step`; `t` counts from `state["step"]`). If it raises, times out, returns nothing or an invalid command,
  nothing happens: the reply's `skill.error` says why, `skill.status` shows the loaded version and any load error.
- If your module defines `step(*, state, spec)`, it is called every {int(il.get('step_every', 2))} control steps during
  the match (in the background, its compute counts the same way); a command it returns is applied when it returns.
- Low-level commands that bypass the skill are refused (HTTP 409) {il.get('raw_rule', 'while the sport says the core skill is in play (see `game spec`)')}{(';' + ' ' + ', '.join(f'`{x}`' for x in free) + ' at any time') if free else ''}.
- The simulator loads your skill when the match starts and, at the start of every unit (a point), reloads it if its
  files changed: improve it between units (write each file completely, e.g. write and rename). Every call is logged
  with the sha256 of the files it ran (artifact `inloop_<seat>.jsonl`). Have `{dl['path']}` in place before you send
  `game start`.
- After the match the platform takes your FINAL `{sdir}/` and evaluates it in a separate isolated container (no
  network, its own process): the API test (hidden requests: the success rate under the contract) and the competition
  test (our fixed reference strategy calls your `{fn}`). The match itself is graded as before."""


PROGRAM_SECTION = """## Write a program
You may write a program that observes, measures in the images, computes your next action and sends it, without a
new model response for each action. You stay in the loop: watch its log, and stop, fix and restart it whenever
needed. Only you send `game start` and `game ready`.

- Use only the camera images, the service's replies and this document; no simulator internals or hidden state.
- One action at a time: an action sent before the previous one has returned is refused (HTTP 429).
- Your container: {container}; no internet during the match. Keep your scripts, logs
  and notes in `/app/output/`."""


def hidden_note(d) -> str:
    hs = (d.get("official") or {}).get("hidden_seeds")
    if not hs:
        return ""
    counts = ", ".join(f"{c}: {n}" for c, n in hs["counts"].items())
    return (f"Official matches use the sport's hidden seed set (seeds per format x level: {counts}; sha256 "
            f"`{hs['sha256']}`; `sports/" + d["sport"]["name"] + "/seed_commitments.json`): the runner gives the sim "
            "service one seed of the match's cell and its salt (`RCB_SEED_SALT`), the players never see them, no "
            "record shows them (seed -1), and the verifier counts a match as official only when its commitment is in "
            "the registered cell (`hidden_seed`, `hidden_seed_index`).\n")


def score_note(d) -> str:
    sc = d.get("score")
    if not sc:
        return ""
    return (f"Continuous score `{sc['name']}` (larger is better; `<seat>_continuous_score`, and `continuous_score` "
            f"for a single-side format): {sc['text']}; a side that forfeits gets {sc['forfeit']}.\n")


def inloop_note(d, t) -> str:
    if inloop(d, t) is None:
        return ""
    return ("\n\nSkill in the loop (`[modes.low.inloop]`, `skilleval/inloop.py`): the player's own skill "
            f"(`{d['modes']['low']['deliverable']['path']}`) plays the match from a skill container per seat "
            "(`skill_<seat>`, built from `environment/skill/`: no network, read-only, no simulator files); every call "
            "is logged in `inloop_<seat>.jsonl`.")


def readme(d, t, tid) -> str:
    rb = d.get("rulebook", {})
    devs = "\n".join(f"| {x['id']} | {x['rule']} | {x['text']} |" for x in d.get("deviations", [])) or "| - | - | - |"
    fmt = d["formats"][t["format"]]
    return f"""# {tid}

| | |
|---|---|
| Benchmark | RoboOlympics (v0.2 framework, `sports/{d['sport']['name']}/`) |
| Mode | standard (`{PROTOCOLS[d['sport']['protocol']]}`) |
| Twin | — (standard only) |
| Simulator / robot | `{d['sport']['physics']}` / `{robot_name(d)}` |
| Family / instance | `{d['sport']['name']}_{t['format']}` |
| Oracle | `full` — solution/: the sport's reference player (camera images and /spec only) |
| Agent time budget | {agent_timeout(t)} s; tokens: protocol default (PROTOCOL §5) |
| Base image | `{d['sport']['image']}` |
| Rules | [{rb.get('name', '-')}]({rb.get('url', '')}) version {rb.get('version', '-')} |

## Task
{d['sport']['summary'][0].upper() + d['sport']['summary'][1:]}; format `{t['format']}` (seats {', '.join(fmt['seats'])}),
action level `{t.get('action_level', 'high')}`, {t.get('play', 'program')} play. Generated by `tools/generate.py` from
`sports/{d['sport']['name']}/sport.toml`; do not edit by hand.{inloop_note(d, t)}

Deviations from the rulebook:

| id | rule | deviation |
|---|---|---|
{devs}

## Difficulty
Set at run time: the seed (`RO_SEED`; it draws the match's random conditions){' and the first player (`FIRST_PLAYER`; play both orders)' if not solo_seat(d, t['format']) else ''}.
{hidden_note(d)}{score_note(d)}
## Reference solution
`solution/solve.sh` runs the sport's reference player (`solution/{d.get('players', {}).get('module', 'builtin_players')}.py`) through the core's client
(`solution/ro_client.py`): camera images and `/spec` only.

## Verification
The `sim` service (the v0.2 core, `environment/sim/core/`) records the match (`episode.jsonl`) and a fresh process
replays it (state hash); the verifier (`tests/grade.py` = `core/grading.py`, no network) writes the common reward
fields (`success`, `valid_trial`, `deterministic`, `<seat>_won`, ...) and the sport's fields (`tests/sport_grade.py`).
"""


# ------------------------------------------------------------------------------------------------ files
def official(d) -> dict:
    """The task's official configurations (contract.json): sport.toml [official]; a registered hidden seed set
    ([official.hidden_seeds] sha256 + count) gets its public commitments (sports/<sport>/seed_commitments.json), which
    must hash to the registered sha256 (core/seeds.py): sha256, counts and commitments per cell (format x level)."""
    off = json.loads(json.dumps(d.get("official", {"seeds": [1, 2, 3], "first_any": True})))
    hs = off.get("hidden_seeds")
    if hs:
        sys.path.insert(0, str(REPO))
        from core import seeds
        name = d["sport"]["name"]
        pub = json.loads((sportlib.SPORTS / name / seeds.COMMITMENTS).read_text())
        hs = {"sha256": hs["sha256"], "counts": dict(hs["counts"]), "cells": pub.get("cells")}
        seeds.registered(name, hs)              # ValueError: the public commitments are not the registered set
        off["hidden_seeds"] = hs
    return off


def sim_env(d, t) -> dict:
    """The sim service's environment of a task (compose values; ${VAR:-default} = chosen at run time)."""
    seats, sides = d["formats"][t["format"]]["seats"], d["formats"][t["format"]].get("sides")
    single = len(seats) == 1
    hosts = "A=player" if single else ",".join(f"{s}=seat_{s.lower()}" for s in seats)
    seed = "${RCB_SEED:-1}" if single else "${DUEL_SEED:-1}"
    replay, video = sportlib.post_match_timeouts(d)
    env = {"RCB_DATA": "/data", "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "OPENBLAS_NUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "RO_PLUGIN": "/srv/sport/plugin.py", "RO_SPORT": d["sport"]["name"],
           "RO_FORMAT": t["format"], "RO_SEATS": ",".join(seats), "RO_SEED": seed,
           "RO_SEED_SALT": "${RCB_SEED_SALT:-}",          # a hidden official seed's salt (core/seeds.py; sim only)
           "RO_SEED_INDEX": "${RCB_SEED_INDEX:-}",        # and its index in the match's cell (information)
           **({"RO_SIDES": ",".join(f"{s}={x}" for s, x in sides.items())} if sides else {}),
           **({"RO_TEAM_CHANNEL": "${RO_TEAM_CHANNEL:-on}"} if d.get("team_channel") else {}),
           "RO_FIRST": "${FIRST_PLAYER:-A}", "RO_LEVEL": "${RCB_LEVEL:-}",
           **({"RO_SLOWDOWN": "${RCB_SLOWDOWN:-" + str(t.get("slowdown", d.get("timing", {}).get("slowdown", 20))) + "}"}
              if d["sport"]["protocol"] == "realtime" else {}),
           "RO_RENDERER": "${RO_RENDERER:-" + d.get("observation", {}).get("renderer", "mujoco") + "}",
           "RCB_ACTION_LEVEL": t.get("action_level", "high"), "RCB_PLAY": t.get("play", "program"),
           "RCB_ACT_RULE": "program" if t.get("play", "program") == "program" else "per_response",
           "CLOCK_S": str(d.get("timing", {}).get("clock_s", 3600)),
           "START_TIMEOUT_S": str(d.get("timing", {}).get("start_timeout_s", 1800)),
           "READY_TIMEOUT_S": str(d.get("timing", {}).get("ready_timeout_s", 300)), "THINK_RETRY_WAITS_S": "",
           "SEAT_HOSTS": hosts, "RCB_RENDER_VIDEO": "${RCB_RENDER_VIDEO:-1}", "RCB_BROADCAST": "${RCB_BROADCAST:-1}",
           "RO_REPLAY_TIMEOUT_S": str(replay), "RCB_VIDEO_TIMEOUT_S": str(video),
           "RO_VIDEO_RENDERER": "${RO_VIDEO_RENDERER:-}",
           **({"LP_NUM_THREADS": str(int(d["observation"]["render_threads"]))}     # llvmpipe threads per render
              if d.get("observation", {}).get("render_threads") else {})}         # process (osmesa)
    if inloop(d, t) is not None:                   # the skill in the loop: each seat's skill container's socket
        for s in seats:
            env[f"RCB_SKILL_INLOOP_{s}"] = f"unix:/skillipc/{s}/skill.sock"
        env["RCB_SKILL_INLOOP_LOG"] = "/data"
    creds = credentials()
    for s in seats:
        env[f"PLAYER_MODE_{s}"] = f"${{PLAYER_MODE_{s}:-${{RCB_PLAYER_MODE:-model}}}}"
        env[f"RO_SEAT_SELECT_{s}"] = f"${{RO_SEAT_SELECT_{s}:-}}"      # "VAR=SOURCE,..." (e.g. a second account)
        for k in PER_SEAT + tuple(creds):          # any of them may be set per seat as <VAR>_<SEAT>
            env[f"RO_SEAT_ENV_{k}_{s}"] = f"${{{k}_{s}:-}}"
    seat_only = seat_only_credentials()
    for k in creds:
        if k not in seat_only:                     # seat-scoped credentials: only the per-seat copies above
            env[k] = f"${{{k}:-}}"
    env["THINK_OPENAI_ROUTE"] = "${THINK_OPENAI_ROUTE:-openrouter}"
    return env


def sim_cpus(d) -> int:
    """The sim container's CPUs: sport.toml [observation] sim_cpus (a sport whose players' images need more, e.g. fine
    visual meshes drawn on the CPU), else 4."""
    return int(d.get("observation", {}).get("sim_cpus", 4))


def compose(d, t) -> str:
    fmt = d["formats"][t["format"]]
    seats = fmt["seats"]
    single = len(seats) == 1
    env = {k: json.dumps(v) for k, v in sim_env(d, t).items()}
    nets = ["default"] + (["play"] if single else [f"net_{s.lower()}" for s in seats])
    lines = ["# Generated by tools/generate.py from the sport's sport.toml (do not edit). The sim sidecar runs the",
             "# v0.2 core (one metering proxy per seat on 18090 + i; the model credentials live only here); each seat",
             "# shares a network only with sim (internal: no internet), plus egress while Harbor installs its agent.",
             "# Anti-cheat (audit 2026-10-08): the seat networks run with an isolated gateway (no route to the host's",
             "# services), seats and sim drop NET_RAW (no raw sockets, no SO_MARK) and sim does not forward packets.",
             "services:", "  main:", "    depends_on: [sim]"]
    if single:
        lines.append("    volumes: [agent_output:/app/output]")
    command = "python3 /srv/core/launch.py"
    il = inloop(d, t)
    lines += ["  sim:", f"    image: {d['sport']['image']}"]
    if il is not None:
        lines.append(f"    depends_on: [{', '.join(f'skill_{s.lower()}' for s in seats)}]")
    lines += ["    volumes:", "      - ./sim:/srv:ro"]
    if il is not None:
        lines += [f"      - skillipc_{s.lower()}_trusted:/skillipc/{s}" for s in seats]
    lines += ["    working_dir: /srv", f"    command: {command}", f"    cpus: {sim_cpus(d)}", "    mem_limit: 8g",
              "    pids_limit: 256", "    cap_drop: [NET_RAW]", "    sysctls:", '      net.ipv4.ip_forward: "0"',
              f"    networks: [{', '.join(nets)}]", "    environment:"]
    lines += [f"      {k}: {v}" for k, v in env.items()]
    for s in seats:
        name = "player" if single else f"seat_{s.lower()}"
        lines += [f"  {name}:", "    build:", f"      context: {'.' if single else './seat'}", "    depends_on: [sim]",
                  '    command: ["sh", "-c", "sleep infinity"]', "    cpus: 2", "    mem_limit: 4g", "    pids_limit: 128",
                  "    cap_drop: [NET_RAW]"]
        if single:
            lines += ["    volumes: [agent_output:/app/output]", "    networks: [play, egress]"]
        else:
            if il is not None:                      # the seat's output, read by its skill container
                lines.append(f"    volumes: [output_{s.lower()}:/app/output]")
            lines += [f"    networks: [net_{s.lower()}, egress_{s.lower()}]"]
        lines += ["    environment:", f"      SEAT: {s}"]
    if il is not None:
        lines += skill_services(d, il, seats, single)
    lines.append("networks:")
    isolated = ["    driver_opts:", "      com.docker.network.bridge.gateway_mode_ipv4: isolated"]
    ipc = [f"  skillipc_{s.lower()}_trusted: {{}}" for s in seats] if il is not None else []
    if single:
        lines += ["  play:", "    internal: true", *isolated, "  egress: {}", "volumes:", "  agent_output: {}", *ipc]
    else:
        lines += [line for s in seats for line in (f"  net_{s.lower()}:", "    internal: true", *isolated)]
        lines += [f"  egress_{s.lower()}: {{}}" for s in seats]
        if il is not None:
            lines += ["volumes:", *[f"  output_{s.lower()}: {{}}" for s in seats], *ipc]
    return "\n".join(lines) + "\n"


def skill_services(d, il, seats, single) -> list[str]:
    """The seats' skill containers (the skill in the loop): the player's skill directory read-only, the socket on a
    volume shared with the sim only; no network, read-only root, no capability but the host's (start the skill as
    nobody, read the player's files), small tmpfs for the private copy (at the seat's own path) and /tmp."""
    sub = skill_subdir(d)
    out = []
    for s in seats:
        vol = "agent_output" if single else f"output_{s.lower()}"
        out += [f"  skill_{s.lower()}:", "    build:", "      context: ./skill",
                f'    command: ["python3", "/opt/skilleval/skillhost.py", "serve", "--socket", "/ipc/skill.sock", '
                f'"--skill", "/player_output/{sub}", "--copy-to", "/app/output/{sub}"]',
                "    network_mode: none", "    read_only: true", "    tmpfs:", "      - /tmp:size=256m",
                "      - /app/output:size=320m,mode=755", "    cap_drop: [ALL]",
                "    cap_add: [SETUID, SETGID, DAC_READ_SEARCH]", '    security_opt: ["no-new-privileges:true"]',
                f"    cpus: {il.get('cpus', 1)}", f"    mem_limit: {il.get('memory', '2g')}", "    pids_limit: 64",
                "    volumes:", f"      - {vol}:/player_output:ro", f"      - skillipc_{s.lower()}_trusted:/ipc",
                "    environment:", f"      SEAT: {s}", "      HOME: /tmp", "      MUJOCO_GL: osmesa",
                "      PYOPENGL_PLATFORM: osmesa"]
    return out


def task_toml(d, t, tid) -> str:
    seats = d["formats"][t["format"]]["seats"]
    single = len(seats) == 1
    arts = ['"/app/output/"']
    if not single:                                 # every seat's own /app/output (its program, logs, notes): one
        arts += [f'{{ source = "/app/output/", service = "seat_{s.lower()}", destination = "seats/{s}/app/output" }}'
                 for s in seats]                   # entry per seat service, to its own place (Harbor destination)
    arts += [f'{{ source = "/data/{f}", service = "sim" }}' for f in
                                 ("final.json", "episode.jsonl", "replay_check.json", "replay.mp4", "video.json",
                                  "broadcast.jsonl")]
    for s in seats:
        arts += [f'{{ source = "/data/think_usage_{s}.json", service = "sim" }}',
                 f'{{ source = "/data/think_calls_{s}.jsonl", service = "sim" }}']
    if inloop(d, t) is not None:                   # every call of the skill in the loop (skilleval/inloop.py)
        arts += [f'{{ source = "/data/inloop_{s}.jsonl", service = "sim" }}' for s in seats]
    port = 18090
    seat_rows = "".join(
        f'\n[[seats]]\nname = "{s}"\nservice = "{"player" if single else "seat_" + s.lower()}"\n'
        f'agent_kwargs = {{ think_proxy_url = "http://sim:{port + i}/v1" }}\n'
        f'setup_networks = ["{"egress" if single else "egress_" + s.lower()}"]\n' for i, s in enumerate(seats))
    replay, video = sportlib.post_match_timeouts(d)  # the collect waits for the sim (90 x 2 s), the replay, the video
    collect = ("for i in $(seq 1 90); do [ -f /data/server.pid ] && break; sleep 2; done; kill -USR1 $(cat "
               f"/data/server.pid); for i in $(seq 1 {replay}); do grep -q '\\\"graded_ready\\\": true' "
               "/data/final.json 2>/dev/null && break; sleep 1; done; [ x$RCB_RENDER_VIDEO = x1 ] && for i in "
               f"$(seq 1 {video}); do [ -f /data/video.json ] && break; sleep 1; done; cat /data/final.json "
               "| head -c 2000")
    desc = (f"{d['sport']['title']}, PROGRAM PLAY, {t['format']}, action level {t.get('action_level', 'high')} "
            f"({PROTOCOLS[d['sport']['protocol']]}): {d['sport']['summary']}. Generated from sports/"
            f"{d['sport']['name']}/sport.toml by tools/generate.py on the v0.2 core.")
    return f"""schema_version = "1.3"
artifacts = [
  {(','.join(chr(10) + '  ' + a for a in arts)).strip()},
]

[task]
name = "robot-coding-bench/{tid}"
description = {json.dumps(desc)}
authors = [{{ name = "RoboOlympics" }}]
keywords = ["program-play", "{d['sport']['name']}", "sports", "{d['sport']['protocol']}", "vision", "roboolympics-v0.2"]

[metadata]
revision = "{d['sport']['revision']}"
author_name = "RoboOlympics"
category = "{d['sport'].get('category', 'robotics-sports-arena')}"
task_type = "trajectory"
simulator = "{d['sport']['physics']}"
robot = "{robot_name(d)}"{robot_configuration(d)}
family = "{d['sport']['name']}_{t['format']}"
mode = "standard"
protocol = "{PROTOCOLS[d['sport']['protocol']]}"
play = "{t.get('play', 'program')}"
action_level = "{t.get('action_level', 'high')}"
oracle = "full"
oracle_evidence = "solution/: the sport's reference player (camera images and /spec only); tools/check runs it live and it wins the default configuration"
parameters = "{'RCB_SEED' if single else 'DUEL_SEED, FIRST_PLAYER'}, RO_RENDERER (the players' image renderer)"
expert_time_estimate_hours = 0.5{skill_lines(d, t)}
{seat_rows}
[verifier]
timeout_sec = 900.0
environment_mode = "separate"
network_mode = "no-network"

[[verifier.collect]]
service = "sim"
command = "{collect}"
timeout_sec = {180 + replay + video}

[agent]
timeout_sec = {agent_timeout(t)}.0

[environment]
build_timeout_sec = 1200.0
cpus = 1
memory_mb = 1024
"""


def seat_layers(src: Path, d, t, player: Path) -> str:
    """The player image's extra layers ([seat], [seat.<level>]); "" if there are none."""
    ex, out = seat_extras(d, t), []
    for dst, rel in sorted(ex["files"].items()):
        ctx = Path("seat_files") / dst.lstrip("/")
        (player / ctx).parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if (src / rel).is_dir() else shutil.copy2)(src / rel, player / ctx)
        out.append(f"COPY {ctx.as_posix()} {dst}\n")
    if ex["pip"]:
        out.append(f"RUN pip install --no-cache-dir -c /opt/client-requirements.txt {' '.join(ex['pip'])} && pip check\n")
    out += [f"RUN {cmd}\n" for cmd in ex["run"]]
    if out:
        out.insert(0, f"# sport.toml [seat] / [seat.{t.get('action_level', 'high')}]: the "
                   "player image's extras for this task\n")
    return "".join(out)


def skill_context(d, t, src: Path, out: Path):
    """environment/skill/ (the skill in the loop): the seat's skill container's image = the simulator's image without
    the simulator, the packages the seat promises a skill ([modes.low] eval_pip, as the post-match evaluation's image),
    skilleval's host (skillhost.py, sandbox.py, host.py) and the low level's toolkit at the seat's paths."""
    lo = d["modes"]["low"]
    (out / "skilleval").mkdir(parents=True)
    for f in ("host.py", "sandbox.py", "skillhost.py"):
        shutil.copy2(REPO / "skilleval" / f, out / "skilleval" / f)
    layers = []
    for dst, rel in sorted(seat_extras(d, t)["files"].items()):
        ctx = Path("files") / dst.lstrip("/")
        (out / ctx).parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if (src / rel).is_dir() else shutil.copy2)(src / rel, out / ctx)
        layers.append(f"COPY {ctx.as_posix()} {dst}\n")
    pip = list(lo.get("eval_pip") or [])
    (out / "Dockerfile").write_text(
        "# The seat's skill container (the skill in the loop: sport.toml [modes.low.inloop], skilleval/inloop.py): the\n"
        "# simulator's image without the simulator, the packages the seat promises a skill ([modes.low] eval_pip), the\n"
        "# low level's toolkit at the seat's paths and skilleval's host, which runs the player's skill as nobody.\n"
        f"FROM {d['sport']['image']}\n" + (f"RUN pip install --no-cache-dir {' '.join(pip)}\n" if pip else "")
        + "COPY skilleval /opt/skilleval\n" + "".join(layers) + "WORKDIR /tmp\n")


def _conventions():
    """scripts/task_conventions.py (README, task.toml and tests/test.sh in the protocol v1.0 layout)."""
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "scripts" / "task_conventions.py"
    spec = importlib.util.spec_from_file_location("task_conventions", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_task(sport: str, tid: str, t: dict, root: Path) -> Path:
    d = sportlib.decl(sport)
    src = sportlib.SPORTS / sport
    out = root / tid
    if out.exists():
        shutil.rmtree(out)
    env = out / "environment"
    sportlib.assemble_sim(sport, env / "sim")
    seats = d["formats"][t["format"]]["seats"]
    single = len(seats) == 1
    player = env if single else env / "seat"
    player.mkdir(parents=True, exist_ok=True)
    cli = shortcuts_of(d, t)
    game = (REPO / "core" / "game.py").read_text()
    game = game.replace("@@COMMANDS@@", json.dumps({k: [k, *v] for k, v in cli.items()}))
    game = game.replace("@@SHORTCUTS@@", "\n".join(f"  game {k} {' '.join(x.upper() for x in v)}" for k, v in
                                                   cli.items()) or "  (none)")
    (player / "game").write_text(game)
    (player / "game").chmod(0o755)
    shutil.copy2(PINS, player / "client-requirements.txt")
    (player / "Dockerfile").write_text(
        "# The player's container: no simulator, only Python with the standard client pins and the generic `game`\n"
        "# command of the core (core/game.py, the sport's shortcuts filled in by tools/generate.py).\n"
        f"FROM {PY_IMAGE}\nCOPY client-requirements.txt /opt/client-requirements.txt\n"
        "RUN pip install --no-cache-dir -r /opt/client-requirements.txt\nCOPY game /usr/local/bin/game\n"
        "RUN chmod +x /usr/local/bin/game && mkdir -p /app/output\n" + seat_layers(src, d, t, player) +
        f"ENV SEAT={seats[0]}\nWORKDIR /app\n")
    if not single:
        (env / "Dockerfile").write_text("# `main` idles in this multi-seat task: the players run in the seat "
                                        f"containers (docker-compose.yaml).\nFROM {PY_IMAGE}\nWORKDIR /app\n")
    (env / "docker-compose.yaml").write_text(compose(d, t))
    il = inloop(d, t)
    if il is not None:
        skill_context(d, t, src, env / "skill")
    tests = out / "tests"
    tests.mkdir(parents=True)
    shutil.copy2(REPO / "core" / "grading.py", tests / "grade.py")
    shutil.copy2(src / d["sport"].get("grade", "grade.py"), tests / "sport_grade.py")
    shutil.copy2(REPO / "players" / "nocode" / "nocode_audit.py", tests / "nocode_audit.py")
    shutil.copy2(REPO / "core" / "seeds.py", tests / "seeds.py")
    contract = {"sport": d["sport"]["name"], "protocol": d["sport"]["protocol"], "task": tid, "format": t["format"],
                "formats": d["formats"], "official": official(d)}
    if d.get("score"):
        contract["score"] = d["score"]
    (tests / "contract.json").write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n")
    (tests / "test.sh").write_text("#!/bin/bash\n# Verifier (separate sandbox): the sim sidecar's artifacts were "
                                   "uploaded to /data by Harbor.\nmkdir -p /logs/verifier\nrm -rf /logs/verifier/*\n"
                                   "python /tests/grade.py 2> /logs/verifier/grade.log\n")
    (tests / "test.sh").chmod(0o755)
    (tests / "Dockerfile").write_text(
        "# Separate verifier sandbox: grades the sim service's artifacts; needs nothing but Python.\n"
        f"FROM {PY_IMAGE}\nCOPY test.sh /tests/test.sh\nCOPY grade.py /tests/grade.py\n"
        "COPY sport_grade.py /tests/sport_grade.py\nCOPY contract.json /tests/contract.json\n"
        "COPY nocode_audit.py /tests/nocode_audit.py\nCOPY seeds.py /tests/seeds.py\nRUN chmod +x /tests/test.sh\n")
    sol = out / "solution"
    sol.mkdir()
    ref = il["reference_class"] if il is not None else d.get("players", {}).get("reference_class",
                                                                                  "builtin_players:Reference")
    mod = ref.split(":")[0]
    shutil.copy2(src / f"{mod}.py", sol / f"{mod}.py")
    install = ""
    if il is not None:                             # the oracle's skill: our reference implementation (toolkit only)
        sub = skill_subdir(d)
        shutil.copytree(src / d["modes"]["low"]["reference"], sol / "skill", ignore=sportlib.IGNORE)
        install = (f"mkdir -p /app/output/{sub}\ncp -r /solution/skill/. /app/output/{sub}/   # the reference skill: it "
                   "plays the match in the loop\n")
    shutil.copy2(REPO / "core" / "client.py", sol / "ro_client.py")
    (sol / "oracle_play.py").write_text(
        f'"""Harbor oracle on any seat: the sport\'s reference player ({ref}, camera images and /spec only) over\n'
        'HTTP through the core\'s client loop (ro_client.py = core/client.py)."""\nimport ro_client\n\n'
        f'ro_client.main(["{ref}"])\n')
    (sol / "solve.sh").write_text("#!/bin/bash\n# Oracle on any seat: the sport's reference player.\nset -e\n"
                                  "mkdir -p /app/output\n" + install + "python3 -c \"import numpy, PIL\" 2>/dev/null || "
                                  "pip install -q "
                                  "--no-cache-dir -r /opt/client-requirements.txt\ncd /solution && python3 "
                                  "oracle_play.py\n")
    (sol / "solve.sh").chmod(0o755)
    (out / "task.toml").write_text(task_toml(d, t, tid))
    (out / "instruction.md").write_text(instruction(d, t, tid))
    (out / "README.md").write_text(readme(d, t, tid))
    _conventions().normalize(out)          # protocol v1.0 conventions, as every generator does (CI checks them)
    return out


def generated_sports(names):
    return [s for s in (names or sportlib.names()) if sportlib.decl(s)["sport"].get("status", "sport") == "sport"]


def same_tree(a: Path, b: Path) -> list[str]:
    cmp = filecmp.dircmp(a, b, ignore=["__pycache__"])
    diffs = [str(a / x) for x in cmp.left_only + cmp.right_only + cmp.funny_files]
    diffs += [str(a / x) for x in cmp.common_files if not filecmp.cmp(a / x, b / x, shallow=False)]
    for sub in cmp.common_dirs:
        diffs += same_tree(a / sub, b / sub)
    return diffs


def main(argv):
    out = None
    if argv[:1] == ["--out"]:
        out, argv = Path(argv[1]), argv[2:]
    if argv[:1] == ["check"]:
        bad = 0
        with tempfile.TemporaryDirectory() as tmp:
            for s in generated_sports(argv[1:]):
                root = out_root(s)
                if root.name == "tasks" and root.parent.name == "build":
                    continue                       # unpublished (the example): nothing committed to compare
                for tid, t in tasks_of(s):
                    new = write_task(s, tid, t, Path(tmp))
                    diff = same_tree(root / tid, new) if (root / tid).is_dir() else ["missing"]
                    if diff:
                        bad = 1
                        print(f"DIFF {tid}: {diff[:5]}")
        print("v0.2 generator: tasks match" if not bad else "v0.2 generator: run tools/generate.py")
        return bad
    for s in generated_sports(argv):
        for tid, t in tasks_of(s):
            print("wrote", write_task(s, tid, t, out or out_root(s)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
