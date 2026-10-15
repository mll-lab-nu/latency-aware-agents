# RoboOlympics: rules for adding a sport

Status: **v1.0 draft** (2026-10-05). This page is the contract every RoboOlympics sport follows. It sits on top of the
repository protocol ([PROTOCOL.md](../PROTOCOL.md)), the real-time protocol
([EAI_STANDARD_REALTIME_1_0.md](../EAI_STANDARD_REALTIME_1_0.md)), the contribution process
([CONTRIBUTING.md](../../CONTRIBUTING.md)) and the review guide ([REVIEWING.md](../REVIEWING.md)). How to bring a
sport into the benchmark step by step is the construction and screening process
([../construction_zh.md](../construction_zh.md)) with [CONTRIBUTING.md](../../CONTRIBUTING.md); the reference
implementations are table tennis and tennis (real time) and billiards (turn based). When this page and a reference
implementation disagree, report it: this page is fixed, not the code.

## 0. Conventions

| Level | Meaning |
|---|---|
| **MUST** / **MUST NOT** | Required / forbidden. A sport that violates it is not merged. |
| **SHOULD** | Recommended. A deviation is explained in the task README. |
| **MAY** | Optional. If adopted, it is implemented as specified here. |

Basis tags: **[CI]** checked by `generate.py check`, `ci_checks/run_all.sh` or `test_program_play.py`; **[test]** the
sport's own unit tests must check it; **[run]** shown by Harbor runs listed in the pull request; **[review]** checked by
the reviewer reading the code.

**Principle.** A RoboOlympics sport is a real sport, played by coding agents through cameras and a simulated body, under
the sport's official rules, where the result depends on the agent's decisions and not on loopholes, leaks or luck. Two
players get the same conditions and every match has a winner.

## 1. Choosing a sport

| # | Rule | Level | Basis |
|---|---|---|---|
| S1 | The sport has an **official rulebook** published by a governing body or an official robot competition (e.g. ITTF, ITF, BWF, MLB, WDF, World Archery, FIBA, USBC, WMF, WPA, World Curling, RoboCup HSL). Self-made "benchmark rules" are not accepted. | MUST | review |
| S2 | It adds something the existing sports do not have: a sport category (net/wall, striking/fielding, target, invasion, racing, combat, athletics) or a capability (locomotion, multi-agent collaboration, partial observability, dexterity). The pull request states which. | MUST | review |
| S3 | The physics that decides the result is simulated and credible (rigid bodies, contacts, flight). Sports whose outcome rests on physics the simulator cannot model well (fluids, snow, cloth) are not added. | MUST | review |
| S4 | The agent's decisions matter: the result is not dominated by a fixed low-level controller or by chance. | MUST | run |
| S5 | It has a single-player task and a duel (§5). | MUST | CI |
| S6 | It can be watched: it writes the spectator broadcast log (§9). | MUST | CI |

## 2. Rules of the game

| # | Rule | Level | Basis |
|---|---|---|---|
| R1 | `spec.py` defines `RULEBOOK` (name, edition or commit, URL) and `DEVIATIONS`; the README cites rule numbers. | MUST | CI |
| R2 | Every deviation from the rulebook is listed with the rule it changes and the reason (shorter match, robot limits, simulator limits). A rule that is not listed is followed as written. | MUST | review |
| R3 | The single-player task and the duel play by **identical rules**; only the opponent differs. | MUST | review |
| R4 | **No draws.** A tie goes to the rulebook's tie-break; if that can still end level, a stated last resort decides: real time, the winner of the last point (define the case of no point); turn based, more chess-clock time left. The grader reports a winner for every valid duel (`A_won` xor `B_won`). | MUST | test |
| R5 | Points come from playing the sport (goals, points, bases, pins, times), not mainly from fouls, timeouts or penalties. A violation is penalised the way the rulebook does it (e.g. loss of possession, faceoff, time penalty), not by handing out points the rulebook does not give. | MUST | run |
| R6 | **Exploit review.** The pull request lists degenerate strategies that were tried and how the rules stop them: stalling or never playing (cf. baseball "never pitch, concede walks"), blocking a goal or lane permanently, ramming or wrecking the opponent, cutting corners or flying over obstacles, parking the ball where the opponent cannot reach it, leaving the play area, forcing timeouts. A strategy that wins without playing the sport is a bug. | MUST | test |
| R7 | The match fits the agent time budget at every official slowdown (setup + match + reviews). | MUST | run |
| R8 | Rules are frozen for a season. A rule change makes a new rules version; it applies from the next season (§12). | MUST | review |

## 3. Observation

| # | Rule | Level | Basis |
|---|---|---|---|
| O1 | The agent observes only: camera images (every camera's pose and intrinsics documented in `/spec` and the prompt), its own robot or vehicle state, referee information (score, clock, calls, whose turn) and the task document. | MUST | review |
| O2 | No world state of anything else: not the ball, puck, shuttle, opponent, other vehicles, gates, track progress or race position — in no endpoint, reply field, log, file, environment variable or mounted path. Results are given after a point, shot or heat, not during it. | MUST | test |
| O3 | The seed is not exposed to the agent (it determines hidden state); `/spec` gives the level and the slowdown. The seed is reported in `final.json` and the reward. | MUST | test |
| O4 | Endpoints for the reference solution, if any, answer HTTP 403 to the agent; the seat image contains only the client and `game` (no simulator, no solution). | MUST | test |
| O5 | Every reply contains what the shared prompt promises (current state; the result of the last point, shot or heat). | MUST | CI |

## 4. Action

| # | Rule | Level | Basis |
|---|---|---|---|
| A1 | Actions are high-level targets that the simulator executes under physical limits (joint and actuator limits, speed and acceleration limits, IK, motor torques): e.g. "racket state at contact T s later", "walk velocity", "throttle and steering", "shot direction, speed, spin". No teleporting, no setting of state. | MUST | review |
| A2 | Units and ranges are documented; out-of-range commands are refused (400), not clipped silently. The reply predicts feasibility where the sport has it (e.g. `ik_residual`, `speed_scale`). | MUST | test |
| A3 | **One action in flight per seat**: a second one gets HTTP 429 `busy` and is counted (`refused_busy`); actions during a freeze get 409. | MUST | CI |
| A4 | When a command takes effect is documented (next control step, plus any planning lead). | MUST | review |
| A5 | Contacts between bodies (robots, vehicles, ball, walls) are simulated physically. | MUST | review |
| A6 | Execution scatter, if the sport has it, is seeded, and within one match each player's scatter is its **own independent stream** (also against a scripted opponent): a player must not be able to read its coming deviation off the opponent's actions. The duel is played as a pair of matches on the same seed with the first player swapped; the stream follows the role in the pair (e.g. stream 1 = the first player's, 2 = the other's), so the second match swaps the streams and over the pair both seats meet exactly the same draws. Shared environment conditions (wind, a shared course) stay shared on purpose. The prompt says so: "each player's execution scatter is independent of the opponent's within a match; the two matches of a duel pair swap the scatter sequences". (Until 2026-10-08 this rule asked for identical scatter for both seats, i.e. the n-th action of either player scattered alike; the player acting later could pre-compensate, which the models did in darts.) | MUST | test |

## 5. Formats and fairness

| # | Rule | Level | Basis |
|---|---|---|---|
| F1 | **Single-player task**: against a scripted opponent with levels (at least 2; level 1 is beatable by the reference player on every official seed), or alone with a success line for scoring sports. | MUST | run |
| F2 | **Duel**: two seats in one world, same rules (R3), seats `A` and `B`, who starts first set at run time; both seat orders are played in a competition. | MUST | run |
| F3 | **Symmetry**: both seats have identical bodies, parameters and conditions. A mirror test compares the two sides' geometry and physical parameters and mirrored actions' outcomes. | MUST | test |
| F4 | No seat can affect the other's observation or timing except through play: one render process per seat; per-seat rate limits where needed. | MUST | review |
| F5 | A **review** (`game ready "WHY" "PLAN"`) after every point, shot, heat or stoppage before play continues. Duel reviews have a time limit (`REVIEW_S`) after which the player forfeits. | MUST | CI |
| F6 | A single-player session that ends mid-match is a **valid loss** (unplayed points or shots score nothing), not a technical failure. On the v0.2 core (from 2026-10-07, core 0.2.6, real-time and turn-based alike): a single side whose match has started and whose session ends before the end loses, `reason = "abandoned"`, `success = 0`; a session that ends before the start is a no-show (F8). | MUST | test |
| F7 | **Team formats** (one agent per robot) MAY be added (≥ 4 seats). A team channel is part of the game service: bounded message size and rate, delivered with the sim time sent, logged, invisible to the opponent; no other path between teammates. A team is one model. | MAY | test |
| F8 | **No-show** (from 2026-10-06): the match begins when every seat has started; after the first start the others have the start window (`START_TIMEOUT_S`). In a match of two or more sides, a side that has not started when the window closes (any of its seats) **loses**, `reason = "no_show"`: the trial is valid and the side that showed up wins (with more than two sides the absent side retires and the others play on, if the sport allows it). If no side shows up, the match is a technical failure and nobody wins. **Single-player formats**: the window opens when the match service starts; a player who has not started when it closes loses, a **valid loss** (`no_show`, `success = 0`, as F6). **A session that ends before its side started** (from 2026-10-07, core 0.2.7) is a no-show at once, without waiting for the window: at the collect a single player loses (`no_show`, `success = 0`), and in a match of two or more sides every side that never started loses (`no_show`, valid; with two sides the present side wins); if no side started, the match does not count. | MUST | test |

## 6. Time

| # | Rule | Level | Basis |
|---|---|---|---|
| T1 | Real-time sports follow `eai-standard-realtime/1.0`: the world runs at 1/S of real speed, S chosen at run time; it freezes only before `start` and during reviews; all agent latency counts. Turn-based sports follow `eai-standard-turn/1.0` (60-minute chess clock per player). | MUST | CI |
| T2 | Control step 10 ms; a different step is a documented deviation of the protocol. | MUST | review |
| T3 | Official slowdowns come from the sport's `sport.toml` `[official] slowdowns` (until 2026-10-10 the legacy `common.py`); a sport that cannot fit one documents why (R7). | MUST | CI |
| T4 | Simulator lag is measured and reported (`sim_lag` maximum and p99); timing policy `report_only` like the reference sports, or a documented validity threshold. | MUST | run |

## 7. Isolation and model access

| # | Rule | Level | Basis |
|---|---|---|---|
| I1 | Each seat has its own internal network to the simulator only; internet only while the agent is installed (`setup_networks`); nothing of another seat is reachable. | MUST | CI |
| I2 | Model access as in the reference sports (currently the per-seat sidecar metering; PROTOCOL D11 alignment in progress), and the shared files `think_proxy.py`, `think_clock.py`, `api_errors.py`, `broadcast_log.py` are **byte-identical** to the reference sport's. | MUST | CI |
| I3 | The model-API validity rule (`api_errors.py`): recovered transient errors are only counted; an unrecovered error, or more than 5 transient ones in a seat, is a technical failure. | MUST | CI |
| I4 | Runs use the standard scripts (`run_single.sh`, `run_duel.sh`, `run_team.sh`, `run_round.py`): Harbor-pinned harness versions, web search disabled (Claude Code `disallowed_tools=WebSearch,WebFetch`, Codex `web_search=disabled`). Matches run any other way are not results. | MUST | review |

## 8. Grading and determinism

| # | Rule | Level | Basis |
|---|---|---|---|
| G1 | The verifier runs in a separate sandbox (`[verifier] environment_mode = "separate"`, `tests/Dockerfile`, no network) and grades only declared artifacts. | MUST | CI |
| G2 | The match is replayed in a fresh process (the verifier replays it again) and must reproduce the score and the final state hash; the source fingerprint covers all simulator code. | MUST | test |
| G3 | `reward.json` has the fields of the reference sports: `success`, `valid_trial`, `technical_failure`, `deterministic`, `official_config`, `level`/`seed`/`slowdown`; duels `A_`/`B_` points, `A_won`/`B_won`, forfeits, per-seat `transient_api_errors`/`unrecovered_api_errors`; timing fields. | MUST | CI |
| G4 | `spec.py` asserts that the grader's official seeds and slowdowns equal the spec's; files the grader copies from the simulator are copied by the generator, never by hand. | MUST | CI |
| G5 | The grader never reads the broadcast log. | MUST | review |

## 9. Generation, conventions and broadcast

| # | Rule | Level | Basis |
|---|---|---|---|
| C1 | Tasks are generated by `tools/generate.py` from `sports/<sport>/sport.toml` (`[[tasks]]`; until 2026-10-10 the legacy `scripts/olympiad/generate.py` from `scripts/olympiad/<sport>/spec.py`); `tools/generate.py check` is clean; no hand edits in `tasks/`. | MUST | CI |
| C2 | Names `tasks/<sport>-<family>[-wallclock]-program-standard`, `<sport>-duel-...`; README in the standard layout; `task.toml` complete (protocol, mode standard, keywords, category, timing policy, parameters per task, timeouts ≤ 14400 s, pinned images, `/app/output/` in the artifacts); `task_conventions.py` knows the sport's prefix. | MUST | CI |
| C3 | The prompt uses the shared structure (written by `tools/generate.py`; formerly `program_prompt.py`, `turn_prompt.py`) unchanged; the sport writes only its own sections; no tactics advice, no configuration numbers, no history. | MUST | review |
| C4 | Images are pinned, built from `images/<name>/`, and public before release. | MUST | CI |
| C5 | Real-time sports set `broadcast=True` and log the events a viewer needs (points, shots, goals, heats, contacts, reviews); the pull request notes what the broadcast adapter needs. | MUST | CI |
| C6 | `tasks/dataset.toml` gets the sport's entries when the branch is merged. | MUST | CI |
| C7 | A presentation scene (`sports/<sport>/presentation/`, spectator renders only) keeps the match scene's kinematics (same bodies, joints, frames, inertias, cameras; match geoms unchanged except their appearance; added geoms visual-only), stays out of the simulator directory, labels every frame it is drawn in, and never feeds the players' images. Third-party models are fetched at a pinned commit with sha256 checks, read as data only, with their licences kept and credited. | MUST | test |

## 10. Evidence in the pull request

1. All checks: `generate.py check`, `ci_checks/run_all.sh tasks/<sport>-*`, `task_conventions.py --check`,
   `test_program_play.py`, the sport's unit tests (including R4, R6, O2–O4, F3, F6, G2).
2. Harbor runs, every one valid with a deterministic replay and the configuration it was given:
   - single: the default configuration; the first official seed of every level; x1; nop scores 0;
   - duel: reference against reference at every duel slowdown with both first players, each with a winner;
   - team (if any): the same with four reference players.
3. Timing: `sim_lag` at x1 and x10 on a loaded host.
4. The exploit review (R6): strategies tried, how each is stopped, the tests that keep it stopped.
5. Optional model pilots, only via the standard scripts (I4), with their token use. Pilots are evidence that the task
   can be played; they are never season results.

## 11. Process

1. **Proposal**: an issue with the sport, what it adds (S2), the rulebook (S1), the simulator and licence, the action
   and observation design, the formats, the exploits considered, the compute per match.
2. **Build** on the sport's own branch (`olympiad-<sport>`), following the build guides; commit only the sport's files.
   A change to a shared file (prompt structure, common settings, run scripts, generator) goes in a separate pull request.
3. **Review** by someone other than the author: [REVIEWING.md](../REVIEWING.md) plus every rule on this page, with
   file and line evidence for each finding.
4. **Merge** when the checks pass and every MUST is met.

## 12. Seasons

- A new sport enters as an **exhibition** sport: it can be played and broadcast, but it is not in the medal table.
- It joins the official programme only at the start of a season. During a season, tasks, rules, harness and model
  access are frozen; fixes that change play wait for the next season.
- A season states its sports, formats, players, seeds, slowdowns and the versions (commits) of everything it runs.

## Changes

| Date | Rule | Change |
|---|---|---|
| 2026-10-07 | F8 | **A session that ends before its side started is a no-show.** Cores 0.2.2-0.2.6 left a match whose player sessions ended (the verifier's collect) before the start window closed, with a side that never started, unfinished and invalid (real time `collected`, turn based `incomplete`); only the window closing made it a no-show. Now the collect decides at once as the window would (core 0.2.7, `Service.no_show_at_collect`): a single side that never started has a valid loss (`no_show`, `success = 0`); with two or more sides every side that never started loses (`no_show`, valid; with two sides the present side wins; with more, the absent sides retire if the sport allows it and the match, never begun, stays unfinished); if no side started, the match still does not count. API and infrastructure failures are Harbor agent errors and are rerun. The legacy single-player tasks already scored such a session a valid loss; the golden records `nop` of the single formats (the session collected inside the window) are tagged `rule-change:core-0.2.7-prestart-noshow`. |
| 2026-10-07 | F6 | **A single session that ends mid-match loses on the core too.** The v0.2 core had ended a started single-player match collected before its end as `collected` (unfinished, a technical failure) in the real-time sports (air hockey, badminton, baseball batting, drone racing, fencing, foosball, racing, soccer, table tennis and tennis singles); the turn-based core already scored a seat that stopped on its turn as `abandoned`. Now a single side (alone or against a scripted opponent) whose match has started loses at the collect, `reason = "abandoned"`, a valid trial with `success = 0` (`reward.json` gains `abandoned`), in both protocols and whoever is on turn (core 0.2.6, `Service.abandon_single`; docs/framework_zh.md §9 core rule 6). Ending the session early is the player's own problem; API and infrastructure failures are Harbor agent errors and are rerun. Before the start F8 applies unchanged; matches of two or more sides are unchanged. Baseball batting drops its legacy success rule (any end with at least 4 bases): success needs a complete game, as every solo format. |
| 2026-10-06 | C7 | **Presentation scenes.** Spectator renders may dress the match scene with fine visual meshes on the same bodies (renderer rule 6 of docs/framework_zh.md §9, core 0.2.4: `renderers/presentation`, `scene_check.same_kinematics`, `core/spectator.spectator_scene`). The players' images still use the match scene; switching them would be a task version bump with the reference players' perception re-validated. |
| 2026-10-06 | F8 | **No-show loses.** Before, the legacy duels disagreed (golden recordings of 2026-10-06, seat B never starting): archery, badminton, baseball, basketball, darts, fencing, soccer (duel and team), table tennis and tennis marked the match an invalid trial (`valid_trial = 0`); air hockey, billiards, bowling, curling, disc golf, drone, foosball and golf putting a valid one. Now every sport does what the drone duel did: the absent side loses (`reason = "no_show"`), the trial is valid, the present side wins; if nobody starts, the match is still a technical failure. In a single-player format a player who has not started when the window closes (it opens when the match service starts) has a valid loss (`success = 0`), as the legacy single-player tasks already had; before, the core left such a match unfinished and invalid. Implemented in the v0.2 core (`core/service.py` `close_start_window`, core 0.2.2; `reward.json` gains `no_show` and `<seat>_no_show`) and in force for every sport on the core. **The legacy servers (`scripts/olympiad/<sport>/`, the generated `tasks/` of sports not yet migrated) keep their old start-window behaviour until the sport migrates to the core**; their results are read with the rule they were played under. |

## Automatic checks

For the sports on the v0.2 framework (all of them since 2026-10-10) the automatic part is `tools/check <sport>`
(stage 1 "contract" reads sport.toml, the generated tasks and the sport's tests) and `ci_checks/check-v02-shared.sh`
(run by `ci_checks/run_all.sh`: `tools/generate.py check`, C1). The table below is the rule map of the legacy checker
`ci_checks/check-olympiad.py`, removed on 2026-10-10 together with the legacy generator (it checked 0 sports by then;
both are in the git history before that date).

| Check | Rules |
|---|---|
| EXPERIMENT has single and duel; their task directories exist | S5 |
| README "Rules"/"Rulebook" row cites a rulebook with a link, no "benchmark-specific" wording; deviations listed | R1, R2 |
| a duel grader has no "draw" result | R4 |
| real-time tasks ship `broadcast_log.py` byte-identical to `scripts/olympiad/broadcast/` | C5 |
| `think_proxy.py`, `think_clock.py` byte-identical to table tennis's (single-seat copy for single-player tasks, per-seat copy for duels); `api_errors.py` to `scripts/olympiad/api_errors.py` | I2, I3 |
| the grader writes the common reward fields; duels the per-seat ones | G3 |
| `task.toml`: protocol, mode, category, keywords, parameters, timing policy, oracle evidence, timeout | C2 |
| `instruction.md` has the shared prompt sections | C3 |
| duel service: start window; real time a review limit with forfeit, turn based a clock | F5 |
| duel seats: setup-only egress, one internal network per seat | I1 |
| unit tests named `*no_draw*`, `*mirror*`, `*session_end*`, `*world_state*`, `*seed_hidden*`, `*exploit*` exist | R4, F3, F6, O2, O3, R6 |
| `oracle = "full"` single tasks say the reference wins the default configuration (`validate-task.yml` runs the default and wants reward 1) | V (warning) |

The 11 sports built before these rules get warnings instead of failures for the unit tests, `category` and
`timing_policy`; new sports must pass. Everything else on this page is checked by review and by the runs in §10.

## Appendix: lessons from reviews

| Finding | Rule |
|---|---|
| Baseball: the leader never pitches and concedes walks (no runners, so a walk is worth only 1 base). | R6, R5 |
| Air hockey: self-made rules; draws; points almost only from possession timeouts; puck areas no mallet can reach; the two mallets with different contact parameters; leaving a lost single-player match turned it into a technical failure. | S1, R4, R5, R6, F3, F6 |
| 3D racing: both cars' live track progress in every reply (and used by a winner); ramming made the rammer win; draws; old copies of the shared metering files; one render process for both seats. | O2, R6, R4, I2, F4 |
| Drone racing: no duel; reviews only after the race; flying over a moving obstacle; the seed in `/spec`; not committed. | S5, F5, R6, O3, §11 |
| Pilots run with a custom script and web tools enabled. | I4 |
