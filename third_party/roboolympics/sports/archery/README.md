# archery: target archery in the wind

Migrated from the legacy task `archery-program-standard` (docs/framework_zh.md §10, stage 1). The task name is kept;
`tools/generate.py` writes it from `sport.toml`. One format: the single round. The legacy duel
(`archery-duel-program-standard`) was retired on 2026-10-07 (review item B10): both archers shot identical lanes
without any interaction, so a head-to-head comparison is the two players' single scores on the same seeds.

**Rules** (World Archery Rulebook, Book 3, with the deviations D1-D2 of `sport.toml`). A round of 4 ends x 3 arrows at
a 122 cm 10-ring face, at 50, 70, 70 and 90 m, in a seeded gusty wind that the archer reads from windsocks in the
camera images. Each arrow is `shoot ELEV AZIM DRAW` (a release primitive with a stated scatter). Alone, success = a
complete round of at least 110 of 120.

| File | Part of contract ② | What it is |
|---|---|---|
| `sport.toml` | declarations | rule book and deviations, formats, modes, timing, `game shoot`, observation whitelist and renderer, rule constants, tiebreaks, exploit list, players, fairness, tasks, capabilities |
| `archery.py` | simulator | the legacy simulator (ballistics, wind, scoring, `Round`, the scene parts); its unused `Duel`, shoot-off wind and `replay_duel` were removed on 2026-10-08 (no change to play) |
| `plugin.py` | world, actions, referee, observation | `Archery` (MatchBase) wrapping `Round` (the module seed installed per call), `reply_views` (the target close-up in each arrow's reply); `ShootLayer`; `scene()`: one MuJoCo scene for every distance (boss, poles, windsocks, distance lines are mocap bodies placed per end); the four cameras |
| `builtin_players.py` | built-in players | `Reference` (the legacy reference archer: windsocks -> wind fit -> flight solve, aiming at the gold centre), `Random` |
| `grade.py` | reward fields | `evaluate(final)`: the legacy single task's reward keys |
| `tests/test_archery.py` | exploit list | one test per `[[exploits]]` entry |
| `golden/` | golden replays | recorded single rounds; `convert.py` replays the legacy single records of the golden-data store bit-exactly (the duel cases and the duel records were retired with the duel) |

Run everything: `PYTHON=python3.11 tools/check archery` (legacy records: `RO_GOLDEN_DATA=<store>`).

## What changed from the legacy task

- **No duel** (review item B10, 2026-10-07): `archery-duel-program-standard` and everything only it needed are gone
  (deviations D3-D5, the shoot-off and its last resort, the tiebreaks, the fairness check, exploits E4, E5 and E7,
  the reference's seeded personal aim error, the duel golden cases). The exploit ids E1-E3, E6, E8 are kept.
- **Reply images** (core 0.2.8, revision `archery-4`): the reply to each arrow carries the target close-up of the face
  with this end's arrows (`observation.image_png_b64` + `tiles`, drawn by the task renderer off the service lock),
  as the legacy reply did; the third arrow of an end is seen there before the arrows are pulled (between the v0.2
  migration and core 0.2.8 the reply had the spotter's call only, and the third arrow could not be seen at all).
  Exploit E9 checks that it shows nothing about the next arrow.
