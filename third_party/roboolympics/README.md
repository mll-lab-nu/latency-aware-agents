# RoboOlympics

RoboOlympics is a benchmark in which **coding agents play physically simulated sports**. A coding agent (Claude Code,
Codex, ...) controls a simulated robot or release mechanism in MuJoCo and sees the world **only through camera
images**, plus its own body state and the referee's calls. It plays in **program play**: it may write its own player
program (vision, physics fitting, strategy), run it against the live game, watch it, fix it and restart it during the
match, and it must review after every point, shot or end. Each sport is played under its official rulebook, with every
deviation listed.

Every sport has a **single-player task** (against a scripted opponent with levels, or alone against a success line)
and a **duel** (two agents in one world, one Harbor seat each). Real-time sports run at 1/S of real speed
(`eai-standard-realtime/1.0`, [docs/EAI_STANDARD_REALTIME_1_0.md](docs/EAI_STANDARD_REALTIME_1_0.md)); turn-based
sports give each player a 60-minute chess clock (`eai-standard-turn/1.0`). Matches are graded in a separate sandbox
from declared artifacts, and are valid only if a fresh-process replay reproduces them bit-exactly.

Code ids keep the original name "olympiad" (`scripts/olympiad/`, `eai-olympiad` images); task names keep the
`robot-coding-bench/` prefix. This repository was split out of
[robot_coding_bench](https://github.com/JamesKrW/robot_coding_bench) (branch `olympiad`); see [NOTICE](NOTICE).

## Layout (v0.2 framework, [docs/framework_zh.md](docs/framework_zh.md))

| Part | What it holds |
|---|---|
| `core/` | the match core: player protocol (HTTP envelope, `game` CLI, client), realtime and turn protocols (clock, slowdown, freeze and review, seats and sides), episode log + fresh-process replay, outcome and validity, reward fields, metering hookup, the generic test suite (`testkit.py`); names no sport and no harness |
| `sports/<sport>/` | a sport against contract ②: `sport.toml` (declarations), `plugin.py` (world, actions, referee, observation), built-in players, `grade.py`, tests, golden matches. `sports/hello/` is the example and template; every sport of Season 1 is migrated |
| `players/` | harness adapters (`players/harnesses/`, formerly `harbor_agents/`, whose old import paths still work), the verifier's no-code audit (`players/nocode/`; direct, no-code play itself is on branch `archive/direct-play`), the metering proxy, the players list `players/players.toml` |
| `renderers/` | renderer plugins (scene + camera definitions + state -> images): MuJoCo (default), MuJoCo showcase, Cycles, Isaac |
| `tools/` | `generate.py` (sport.toml -> task dirs), `check` (all checks of a sport), `golden.py`, `live.py` (offline reference/nop runs) |
| `seasons/` | season definitions (frozen versions) |
| `scripts/olympiad/`, `tasks/` | the run, dispatch and broadcast tools (below), and all generated tasks |

Check a sport: `PYTHON=python3.11 tools/check hello`. Everything: `bash ci_checks/run_all.sh`. How to add a sport, a
player or a renderer: [CONTRIBUTING.md](CONTRIBUTING.md).

## Sports

| sport | timing | single player | duel |
|---|---|---|---|
| table tennis | real time | `tabletennis-match-wallclock-program-standard` | `tabletennis-duel-wallclock-program-standard` |
| tennis | real time | `tennis-match-wallclock-program-standard` | `tennis-duel-wallclock-program-standard` |
| badminton | real time | `badminton-match-wallclock-program-standard` | `badminton-duel-wallclock-program-standard` |
| baseball | real time | `baseball-batting-wallclock-program-standard` | `baseball-duel-wallclock-program-standard` |
| darts (501) | turn based | `darts-501-program-standard` | `darts-duel-program-standard` |
| archery | turn based | `archery-program-standard` | — (the duel was retired on 2026-10-07: no interaction; compare single scores on the same seeds) |
| basketball | turn based | `basketball-shootout-program-standard` | — (the duel was retired on 2026-10-07: no interaction; compare single scores on the same seeds) |
| bowling | turn based | `bowling-program-standard` | — (the duel was retired on 2026-10-08: no interaction; compare single scores on the same seeds) |
| golf putting | turn based | `golf-putting-program-standard` | — (the duel was retired on 2026-10-08: no interaction; compare single scores on the same seeds) |
| billiards (8-ball) | turn based | `billiards-eightball-program-standard` | `billiards-duel-program-standard` |
| curling | turn based | `curling-match-program-standard` | `curling-duel-program-standard` |
| soccer (2 v 2 humanoids) | real time | `soccer-match-wallclock-program-standard` | `soccer-duel-wallclock-program-standard`; team format (4 seats, one agent per robot): `soccer-team-wallclock-program-standard` |
| air hockey | real time | `airhockey-match-wallclock-program-standard` | `airhockey-duel-wallclock-program-standard` |
| racing (3D cars) | real time | `racing-match-wallclock-program-standard` | `racing-duel-wallclock-program-standard`; multi-car race: `racing-race-wallclock-program-standard` |
| drone racing (FAI F9U) | real time | `drone-racing-wallclock-program-standard` | `drone-duel-wallclock-program-standard` |
| foosball | real time | `foosball-match-wallclock-program-standard` | `foosball-duel-wallclock-program-standard` |
| disc golf | turn based | `discgolf-course-program-standard` | — (the duel was retired on 2026-10-08: no interaction; compare single scores on the same seeds) |
| fencing (épée, FIE) | real time | `fencing-bout-wallclock-program-standard` | `fencing-duel-wallclock-program-standard` |

The authoritative list is the `[[tasks]]` of `sports/*/sport.toml` (`python3 tools/generate.py check` renders all of
them). Per-sport details: [docs/benchmarks/olympiad.md](docs/benchmarks/olympiad.md) and each task's
`README.md`.

## Requirements

- Linux with Docker (CPU only; no GPU needed). Python >= 3.11 for the scripts (they use `tomllib`).
- **Harbor with multi-seat support** (seats, per-seat `setup_networks`), from the fork
  [CHIGUI0/harbor](https://github.com/CHIGUI0/harbor), branch `rcb-multiseat` (the pinned commit is
  `3fc74678123a2eb1845938b099e9d7856f2c6f6d`, Harbor v0.23.0 plus multi-seat trials; proposed to upstream Harbor):

  ```bash
  uv tool install git+https://github.com/CHIGUI0/harbor@3fc74678123a2eb1845938b099e9d7856f2c6f6d
  # or: export HARBOR=/path/to/a/checkout/.venv/bin/harbor
  ```

  The run scripts check this before every match (`harbor_guard` in `scripts/olympiad/v02_run.sh`) and stop with
  exit 2 when `$HARBOR` (default: `harbor` on `PATH`) is anything else, e.g. upstream PyPI `harbor` 0.23.0: it has no
  `--seat`, `[[seats]]` or `setup_networks` and would run the agent in the `main` service on the default network, with
  internet. A checkout passes only as an editable install (`-e`) at the pinned commit with no changes to tracked
  files; a non-editable install from a directory is refused.

- The simulator images `ghcr.io/mll-lab-nu/eai-olympiad:0.1.0`,
  `eai-olympiad-billiards:0.1.0` and `eai-olympiad-soccer:0.1.0`. They are visible to members of the mll-lab-nu
  GitHub organisation: log in once with a GitHub token that has `read:packages`, then pull
  (or build them locally with `scripts/build_images.sh`):

  ```bash
  echo "$GITHUB_TOKEN" | docker login ghcr.io -u <your-github-user> --password-stdin
  docker pull ghcr.io/mll-lab-nu/eai-olympiad:0.1.0
  ```
- For model players: copy `.env.example` to `.env` and fill in the credentials you need (never commit `.env`).

## Run one match

```bash
scripts/olympiad/run_single.sh --task archery-program-standard --agent oracle --level 1 --seed 20260929
```

`--agent` is `oracle` (the reference player), `nop`, or a model player (`opus`, `fable`, `astra`, `luna`; see the
header of `run_single.sh`). Real-time sports also take `--slowdown S`. A duel:

```bash
scripts/olympiad/run_duel.sh --task tabletennis-duel-wallclock-program-standard --a oracle --b oracle --slowdown 10 --seed 1 --first A
```

The job lands in `jobs/<tag>-...` (gitignored) and the script prints its `reward.json`. A whole experiment round over
model-account lanes: `scripts/olympiad/run_round.py` (accounts in `scripts/olympiad/accounts.json`, which names
environment variables only).

## Add a sport

1. Read the construction and screening process: [docs/construction_zh.md](docs/construction_zh.md). A sport goes
   through six steps (define the sport, choose the robot and venue, design the interface, model-free validation,
   model trial runs, freeze), each ending with a check.
2. Build it as described in [docs/framework_zh.md](docs/framework_zh.md) §8 and [CONTRIBUTING.md](CONTRIBUTING.md)
   ("Add a sport"; copy `sports/hello`), and pass `tools/check <sport>`.
3. Open one pull request per sport, with its screening report in `docs/screening/<sport>.md`
   ([docs/screening/README.md](docs/screening/README.md)).

## Repository map

| path | what |
|---|---|
| `sports/<sport>/`, `tools/generate.py` | a sport (`sport.toml`, plugin, simulator, grader, tests, golden matches) and the generator that renders every `tasks/<task>` from it; `check` compares |
| `scripts/olympiad/api_errors.py` | the model-API validity rule for `regrade_transient.py` (byte-identical to `core/metering/api_errors.py`) |
| `scripts/olympiad/run_single.sh`, `run_duel.sh`, `run_team.sh`, `run_race.sh`, `run_round.py`, `dispatch.py`, `accounts.json` | the standard run scripts and the account dispatcher |
| `scripts/olympiad/broadcast*`, `showcase/`, `program_harness.py`, `live_board.py`, `monitor.py` | spectator broadcast (program feed with an automatic director), live board, run monitor |
| `scripts/olympiad/test_*.py`, `sports/<sport>/tests/` | unit tests: dispatcher, model lock, program-play act rule and metering failover, SOL route, broadcast (offline; CI runs them, see `.github/workflows/static-checks.yml`); each sport's own tests (`tools/check <sport>`, in its simulator image) |
| `scripts/task_conventions.py`, `scripts/eai_standard/` | task layout conventions; the standard client pins |
| `tasks/` | generated Harbor tasks (do not edit by hand) and `dataset.toml` |
| `players/harnesses/` (old import path `harbor_agents/`, shims) | Harbor agent adapters for metered model players (Claude Code, Codex) |
| `images/olympiad*/` | simulator images (MuJoCo 3.3.7, OSMesa, CPU; soccer adds CPU torch) |
| `ci_checks/` | static checks; `run_all.sh` runs them all (the sport rules: `tools/check <sport>`) |
| `docs/` | the rules contract, protocols (PROTOCOL.md, real-time standard) and benchmark notes |
| `prototypes/` (not on main) | research prototypes moved off main on 2026-10-10, kept on branch `archive/prototypes`: the table-tennis serve study (`tabletennis_serve/`: serve lab, serve library builder, real-limits and receive studies, fairness runs, results) and the tennis player-serve study (`tennis/`: serve duels, grids, fairness runs, pro statistics, results) |

## License

MIT, see [LICENSE](LICENSE) and [NOTICE](NOTICE).
