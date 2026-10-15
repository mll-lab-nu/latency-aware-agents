# 3D racing after the review: validation (2026-10-05)

Branch `olympiad-racing-fix`. Task code as committed (`SIM_SOURCE_SHA` single `462b05bf…`, duel `315906af…`).
Harbor `/data/guichi/harbor-src/.venv/bin/harbor` (multi-seat), runs from `scripts/olympiad/run_single.sh` /
`run_duel.sh`, tag `racefix`, on a shared host (other sports' Harbor runs in parallel).

## Checks

| Check | Result |
|---|---|
| `generate.py check` | tasks match their generator (only racing tasks change) |
| `ci_checks/run_all.sh` on both tasks | all checks passed; protocol v1.0: 2 tasks conform |
| `task_conventions.py --check` | rc 0 (`racing-` prefix added) |
| `test_program_play.py` | ALL OK for both; metering files identical to table tennis match (single) / duel (duel) |
| `ci_checks/check-olympiad.py racing` (olympiad 996a9aeba) | 52 ok, 0 warnings, 0 failures |
| `scripts/olympiad/racing/test_racing.py` | 15 tests OK (ramming experiment, wedged cars, blocking/stalling exploits at L1/L4, reverse, no draws, heat-3 grid, mirror, replay/tampering, input bounds, world state, seed hidden, F6, both-forfeit) |

## Harbor runs (all `valid_trial` 1, `deterministic` 1, configuration as passed, `official_config` 1)

| Run | Result | Heats (A:B) | sim_lag max / p99 (s) |
|---|---|---|---|
| oracle, default L1 x20 seed 101 | win | 2:0 | — |
| oracle, L1 x1 seed 101 | win | 2:0 | 0.0028 / 0.00017 |
| oracle, L1 x5 seed 111 | win | 2:0 | |
| oracle, L1 x5 seed 121 | win | 2:0 | |
| oracle, L2 x5 seed 202 | win | 2:0 | 0.0025 / 0.00010 |
| oracle, L3 x5 seed 303 | win | 2:1 | |
| oracle, L4 x5 seed 404 | loss (valid) | 1:2 | 0.0025 / 0.00030 |
| nop, default L1 x20 seed 101 | loss (start forfeit, valid) | 0:0 | |
| duel oracle-oracle x10 seed 1, first A | B wins | 1:2 | 0.111 / 0.00010 |
| duel oracle-oracle x10 seed 1, first B | A wins | 2:1 | 0.0010 / 0.00010 |
| duel oracle-oracle x20 seed 1, first A | B wins | 1:2 | 0.0029 / 0.00010 |
| duel oracle-oracle x20 seed 1, first B | A wins | 2:1 | 0.0021 / 0.00010 |

No crash was called in any reference run. Each seat order wins once per slowdown. The one 0.111 s lag in the first
x10 duel is a single late step (p99 0.1 ms) on the loaded host; the timing policy is report-only.

Offline (in-process renderer, `test`-style harness, not Harbor): the camera reference beats the scripted car on every
official seed of L1 and L2 (10 ms and 40 ms observe delay), wins all 3 L3 seeds with a 10 ms delay but only 1 of 3
with 40 ms, and loses every L4 seed; the scripted L1 car laps in 17.6-20.0 s (limit 25 s).

## Model smoke (luna only, standard scripts, web search disabled)

| Run | Result | Tokens (requests / input / output) |
|---|---|---|
| single, luna, L1 x10 seed 101 | win 2:0 (14.65 s, 9.87 s), 0 crashes, 0 API errors | 36 / 1,032,888 / 18,617 |
| duel, luna vs luna, x10 seed 1, first A | B wins 2:0 (13.21 s, 10.72 s); A lost the stripe in the first hairpin in heat 1 | A 78 / 3,011,219 / 30,742; B 48 / 1,609,971 / 21,762 (B 1 transient API error, recovered) |

These pilots show the tasks can be played; they are not results.
