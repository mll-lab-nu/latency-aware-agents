# 3D racing v3: validation (2026-10-05)

Branch `olympiad-racing-v3` (revision `racing-program-v3`): seeded circuits, fixed obstacles, multi-car race. Harbor
`/data/guichi/harbor-src/.venv/bin/harbor` (multi-seat), runs from `run_single.sh` / `run_duel.sh` / `run_race.sh`,
tag `racev3`, on a shared, loaded host (load 20-28 on 32 cores; other sports' runs in parallel).

## Checks

| Check | Result |
|---|---|
| `generate.py check` | tasks match their generator (only racing tasks change) |
| `ci_checks/check-olympiad.py --root <wt> racing` (olympiad 996a9aeba) | 82 ok, 0 warnings, 0 failures |
| `ci_checks/run_all.sh tasks/racing-*` | rc 0; protocol v1.0: 3 tasks conform |
| `task_conventions.py --check` | rc 0 |
| `test_program_play.py` (match, duel, race) | ALL OK; metering files identical to table tennis (single-seat / per-seat) |
| `scripts/olympiad/racing/test_racing.py` | 30 tests OK (circuit shape/radius/clearance/determinism, v2 interfaces, obstacle path/contact/cutting, crash rule and ramming, wedged cars, N-car grid rotation, count-back, idle race, heat order, per-pair rammer retired, retirement replay, exploits vs scripted car, no draws, mirror x2, replay/tampering, input bounds, seed hidden, world state, F6, forfeits, start forfeit, broadcast events) |

## Harbor runs on the final code (all `valid_trial` 1, `deterministic` 1, `official_config` 1)

| Run | Result | Heats | sim_lag max / p99 (s) |
|---|---|---|---|
| oracle, default L1 x20 seed 101 | win | 2:0 | 0.005 / 0.0023 |
| oracle, L1 x1 seed 101 | win | 2:0 | 0.007 / 0.0026 |
| oracle, L2 x5 seed 202 | win | 2:0 | 0.007 / 0.0024 |
| oracle, L3 x5 seed 303 | win | 2:0 | 0.006 / 0.0028 |
| oracle, L4 x5 seed 404 | loss (valid) | 0:2 | 3.06 / 0.82 (host load) |
| nop, default L1 x20 seed 101 | loss (start forfeit, valid) | 0:0 | - |
| duel oracle-oracle x10 seed 1, first A | B wins | 1:2 | 3.07 / 0.0007 |
| duel oracle-oracle x10 seed 1, first B | B wins | 1:2 | 0.024 / 0.0013 |
| duel oracle-oracle x20 seed 1, first A | B wins | 1:2 | 0.003 / 0.0001 |
| duel oracle-oracle x20 seed 1, first B | A wins | 2:1 | 0.004 / 0.0001 |
| race 4 x oracle, x10 seed 1 | D wins: D 73, C 70, B 70, A 67 (B/C equal on count-back, the last heat decides) | 4 heats, every car finished at least 3 | 0.117 / 0.0026 |

No obstacle crash in any reference run; the 4-car race had 6 car-car crash calls (one warning each, none
disqualified). The 3 s lag peaks are single late steps on the loaded host (p99 below 1 ms except the L4 run);
the timing policy is report-only. Offline (in-process renderer): the reference wins every official seed of L1-L3
and loses L4; 3-, 4- and 6-car oracle races finish all heats.

## Model smoke (luna only, standard scripts, web search disabled)

| Run | Result | Tokens (requests / input / output) |
|---|---|---|
| single, luna, L1 x10 seed 101 | win 2:1 (35.7 s, 28.8 s), 1 obstacle crash, 0 API errors | 83 / 4,739,950 / 53,597 |
| race, 4 x luna, x10 seed 1 | C wins (C 86, A 76, B 40, D 0); D disqualified in heat 1 (three crashes at the start), B in heat 3; 10 crash calls, 3 obstacle crashes; 0 API errors | A 137 / 13,721,486 / 64,968; B 113 / 6,887,858 / 47,267; C 102 / 5,843,259 / 27,688; D 101 / 5,467,468 / 27,364 |

These pilots show the tasks can be played; they are not results.

## Bugs found by the runs (fixed)

- Broadcast: a crash call carries its own `heat`; the spectator log got it twice (TypeError in the physics loop).
  Latent since v2 (no v2 reference run had a crash). Test added.
- Race sim `pids_limit` 384 was too low (llvmpipe threads per render context); the in-sim replay's ffmpeg could not
  start. Now 1024.
- Reference: waits for the service to answer; ignores the start/finish line as a "car".
- Do not regenerate tasks while trials run: Harbor bind-mounts `tasks/<task>/environment/sim` into the sim.
