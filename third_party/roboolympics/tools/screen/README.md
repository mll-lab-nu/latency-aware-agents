# tools/screen — the model-free screening checks

The checks of step ④ of the construction and screening process ([docs/construction_zh.md](../../docs/construction_zh.md)
"第四步：无模型验证") that need many matches: reference solution, skill necessity, simple baselines and capability
sensitivity. They run only built-in programs (no model) in the sport's own image, on the same simulator directory a
generated task gets. Table tennis is the first sport through them (`sports/tabletennis/screen/hooks.py`,
`docs/screening/tabletennis.md`). B9 (the upper-bound feasibility test of step ②) stays the sport's own test
(`sport.toml [modes] b9_test`, `tools/check` stage 6); engineering and isolation checks are `tools/check` stages 1–6.

```
python3 tools/screen/batch.py SPORT --variants reference,fixed,... --levels 1-4 --seeds 1-10 --data DIR [--jobs 4]
python3 tools/screen/gates.py SPORT --data DIR --out DIR/gates.md --json DIR/gates.json
```

- `batch.py` assembles the simulator once (`DIR/sim`, `tools/sportlib.assemble_sim`, its commit in `DIR/sim/STAMP`),
  copies `tools/screen/play.py` and the sport's hooks to `DIR/code`, and runs one container per match
  (`screen-<sport>-<variant>-L<level>-s<seed>`, no network, `--cpus 2`), at most `--jobs` at once (2 when the host's
  load is above `--max-load`; a number in `DIR/JOBS` changes the limit while it runs). Results:
  `DIR/runs/<variant>/L<level>_s<seed>/result.json`. Re-running skips finished matches, so a batch can be extended
  with more seeds or variants.
- `play.py` is `core/local.py`'s real-time loop plus a latency model: an action takes effect `D` control steps after
  the observation it answers and is lowered by the action layer at that step (as the live service does), one action
  per seat in flight. `D` is the sport's declared reaction delay (`sport.toml [timing] reaction_delay_steps`, core
  0.2.18), the rule of the live service, so screening and the official matches agree; a hooks `DELAY` that differs is
  an error. A sport that declares none keeps the hooks' `DELAY` (default 0) until it adopts the declaration. The
  `[reference]` hook's `delay=` reaches play.py as `--sport-delay` and follows the same rule as the hooks' `DELAY`.
  `--delay` overrides it for an experiment; `result.json` records `delay` and `delay_source`.
- `gates.py` reads the results and the hooks' lists (without importing the hooks) and prints the four gates:

| gate | rule (defaults) |
|---|---|
| reference | wins and mean point share per level (P4: the reference completes meaningful matches) |
| simple baselines | every baseline loses the majority of its matches at every level (wins < n / 2) |
| skill necessity | a program with only the non-core actions: no win, point share <= 0.05 |
| capability sensitivity | per capability, its strong impairment scores below the reference (paired over seeds, 95 % bootstrap interval below 0) and the weak one not above it by more than the replicas' noise floor, at all levels but one |

## What a sport provides: `sports/<sport>/screen/hooks.py`

The directory is not part of a task's simulator (`tools/sportlib.py` NOT_IN_SIM). The hooks run in the sport's image
with the simulator directory at `/srv` (the sport's own modules importable, e.g. its built-in players).

| name | kind | what |
|---|---|---|
| `FORMAT`, `ACTION_LEVEL`, `DELAY`, `LEVELS` | constants | the format the gates play (`match`: one seat against the scripted opponent), the action level, the effect delay in control steps (only for a sport that declares no `[timing] reaction_delay_steps`; deprecated), the official opponent levels |
| `REFERENCE` | constant | the reference player's variant name |
| `REPLICAS` | list | variants that change nothing that matters (the noise floor of a chaotic simulation) |
| `BASELINES` | list | the simple baselines (fixed, random, reflex-only, middle-only, ... as the sport allows) |
| `NECESSITY` | list | programs that use only the non-core actions |
| `SENSITIVITY` | dict | capability -> [weak, strong] impairment variants (impair the capability, keep the skill) |
| `make_player(variant, spec, seat, seed, delay, opts)` | function | a player object with `act(view)` (a built-in player's interface) |
| `prepare(variant, opts)` | optional | process-wide changes before the world is built (e.g. a player's own physics constants) |
| `patch_match(m, variant, opts)` | optional | a change of the world for a variant (e.g. an opponent with a planted weakness) |
| `wants_images(player, m, seat)` | optional | False when the player would not look at this observation's images (speed only: skips drawing) |
| `summary(result, players)` | optional | `won`, `point_share` and anything the sport wants per match (else the scores and winner) |

Statistics: wins with Wilson 95 % intervals; paired differences of the point share over common seeds with a 95 %
bootstrap interval. Matches of a realistic simulator are chaotic (a 0.1 mm change makes a match diverge within a few
strokes): the replicas measure that noise, and ten seeds per level detect only large effects (about 0.1 of point
share); use more seeds for a closer call.

## The shared `[reference] match` hook: `reference_hook.py`

`tools/check` stage 5's reference check (docs/screening/README.md, "The shared hook") plays its matches with
`play.py` through `reference_hook.hook(...)`, so the calibration uses this loop. A sport's `screen/reference.py` is two
lines (`from reference_hook import hook`; `match = hook("hooks.py", camera=..., truth=...)`, or `hook()` for the
sport's `[players] reference_class` alone); `summary()` may also return `scores` ([ours, theirs]) and `report`
({name: [count, of]}), which the hook passes to stage 5. Without a truth variant the camera variant plays the truth
matches and the hook says so: stage 5 prints a NOTE, and FAILS once the sport is frozen (`[official.hidden_seeds]`).
Tests: `python3 tools/screen/test_reference_hook.py` (also run by `tools/test_check.py`, which CI runs).
