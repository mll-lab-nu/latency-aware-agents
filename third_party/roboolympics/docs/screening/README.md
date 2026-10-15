# Screening reports

Every sport goes through the construction and screening process ([../construction_zh.md](../construction_zh.md)) before
it enters a release: define the sport → choose the robot and venue → design the interface → model-free validation →
model trial runs → freeze and evaluate. The owner of a sport makes the choices (which modes, which robot, how the
skills are split), checks that every step's test really passes, writes it down here, and reports it at group meeting.

Most sports already have their simulation, rules, reference players and match flow; going through the process means
re-checking them: add the evidence where a check is already met, and change the sport (or say exactly where it is
stuck) where it is not.

## Pull requests

- **One pull request per sport**, touching only `sports/<sport>/`, its generated tasks and its report
  `docs/screening/<sport>.md`. A large sport may split its work into several pull requests.
- Changes to shared parts (`core/`, `bodies/`, `skilleval/`, `tools/`, shared check scripts) go in their own pull
  requests first; the sport's pull request may be stacked on them.
- `tools/check <sport>` passes before merging; the pull request description lists each step's result.
- Merging into `main` is not a release: a sport enters a release when its screening report passes.

## Report template (`docs/screening/<sport>.md`)

1. **Conclusion**: which checks pass; the chosen configuration (robot and assembly, venue, interface and modes,
   opponent levels, seeds, scoring).
2. **Candidates and pruning**: the candidates considered, and the evidence for each one dropped.
3. **Step 1, define the sport**: the core challenge kept; rule book, edition and numbered deviations; the
   capabilities claimed (and which ones remain after the capability-sensitivity check); adjustable parameters;
   opponents, seeds and scoring.
4. **Step 2, robot and venue**: sources of the real limits; real counterpart; feasibility (upper-bound player);
   physical consistency (self-penetration, contacts on, release speeds).
5. **Step 3, interface**: modes; skill contracts (inputs, timing, feedback, control authority, failure handling);
   low-level toolkit manifest; observation and time model; skill in the loop.
6. **Step 4, model-free validation**: reference solution (the reference check below), skill necessity, simple
   baselines, capability sensitivity, engineering checks, isolation probe; numbers, match counts, intervals.
7. **Step 5, model trial runs**: models, matches, scores; shortcuts and loopholes found and fixed.
8. **Freeze**: versions, hidden seed set, opponent levels, scoring. Frozen = registered hidden seeds
   (`[official.hidden_seeds]` + verifying `seed_commitments.json`, CONTRIBUTING.md "Freeze a sport");
   `python3 tools/frozen.py` lists the frozen sports (`--json` for release tooling).
9. **Process issues**: checks that did not work well, missing tools, time and cost of each step.

## Reference check (step 4, rule of 2026-10-11)

The reference player is checked in two parts (docs/construction_zh.md §2 problem 6, step 4 "参考解"):

- **Opponent calibration** (one-side formats against a scripted opponent, and one-side formats with graded levels of
  their own such as baseball batting and drone racing): the reference with **truth perception** (the
  true state, with the noise the sport declares) plays at least 20 seeds per official level: it wins the majority at
  the lowest level and its wins do not rise with the level. Plus evidence that **camera-only can win**: the camera-only
  reference wins the majority of the lowest level, **or** a model won a step-5 match with cameras only.
- **End to end** (every format): the camera-only reference's matches run normally (seen, through the platform's
  interface, in time); report its playability, e.g. the legal return rate (with the truth reference's for
  comparison). **No numeric gate**; a clearly low rate is explained and fixed along the fallback path.
- A duel (two agents) only has to be playable: the reference on every seat plays a valid match.

The report gives, per level, the truth reference's wins / matches and point share, the camera evidence (the camera
reference's wins, or which model won which level with what score, and where it is recorded) and the camera
reference's playability numbers.

`tools/check` stage 5 runs it from two `sport.toml` declarations:

```toml
[reference]                                 # tools/check stage 5, the reference check
match = "builtin_players:reference_match"   # a function in sports/<sport>/ (run in the sport's image)
seeds = 20                                  # per level: a count n (seeds 1..n) or a list; at least 20
camera_seeds = 10                           # the camera reference's matches at the lowest level (default: the first 5 seeds)

[[reference.camera_wins]]                   # optional, one per step-5 match a model won with cameras only
format = "match-wallclock"
level = 1                                   # one of the format's levels
who = "Opus 5.5 (claude-code)"              # the model and harness
score = "14:12"
evidence = "docs/screening/badminton.md"    # a file in the repository (an #anchor may follow)
```

`function(format=..., level=..., seed=..., perception="truth" | "camera")` plays one match of the reference (platform
skills only) against the scripted opponent at that level, with `MatchConfig` seed `seed`, and returns
`{"won": bool, "scores": [ours, theirs], "report": {name: [count, of], ...}}`; `report` holds the playability
counters (e.g. `"legal returns": [102, 201]`), summed over the matches. With `"camera"` it plays the sport's
`[players] reference` unchanged (camera images only); with `"truth"` the same player with its perception replaced by
the true state. The check: calibration passes with >= 20 matches per level, a majority at the lowest level and wins
not rising; "camera-only can win" passes with a camera-reference majority at the lowest level or a valid
`camera_wins` entry; the camera matches and the live task run must play to their end (a match that cannot be
played fails), their wins are reported only. `tools/check --jobs N` runs N of these containers at once. The simple
baselines' perception rule (truth first, a winning baseline re-run with cameras, the highest level lost even with
truth) is `tools/screen`'s, not `tools/check`'s.

### The shared hook (`tools/screen/reference_hook.py`)

Most realtime sports need no loop of their own: the shared hook plays each match with `tools/screen/play.py`, the
step-4 screening loop (effect delay D, lowering at the effect step, one action in flight), so stage 5 and the step-4
record agree. The hook lives in the sport's screening directory (`sports/<sport>/screen/`, not part of a task's
simulator); `tools/check` stage 5 copies that directory and the shared files into the container. What the sport
writes:

```python
# sports/<sport>/screen/reference.py
from reference_hook import hook                  # tools/screen/reference_hook.py (stage 5 provides it)

match = hook()                                   # [players] reference_class, camera only (no truth variant yet)
# or, with step-4 hooks (tools/screen/README.md) that have a truth variant:
match = hook("hooks.py", camera="reference", truth="reference_truth", delay=5)
```

```toml
[reference]
match = "screen.reference:match"
seeds = 20
```

What the sport chooses: the hooks file with its camera and truth variants (the reference program), `delay` / `every` /
`action_level` (default: the hooks' `DELAY` and `ACTION_LEVEL`, else delay 0 observing every 5 steps, the high level
if the sport has one), and how to read the score: by default `won` and `report` come from the hooks' `summary()` (if
any) and the scores from the sport's `scores()` (seat A's side against the others, when they are numbers);
`read=lambda result: {...}` (play.py's result.json) replaces that, e.g. for racing, whose scores are per-side dicts
(the wins are right without it, the point share needs it). Levels and seeds stay in `sport.toml`.

Without a truth variant (`hook()` alone) the camera reference plays the truth matches as a **stand-in** and stage 5
prints a NOTE with their count: that shows the opponent ladder, not a truth calibration. **Once the sport is frozen**
(`[official.hidden_seeds]` registered) a stand-in **fails** stage 5 (user decision 2026-10-13): add a truth variant
before the freeze. The foosball hook (`sports/foosball/screen/reference.py`, the PR stacked on the shared hook's) is the
example: the true ball and figures through `augment_view`, about 30 lines. The `[reference] match` line in `sport.toml` changes the tasks' copy of it (regenerate,
`harbor sync tasks`); the hook file does not. A format that is not named `match` needs play.py to read its seats from
`sport.toml` (open PRs #103, #106); a turn-based sport needs #103's turn loop.

Transition (2026-10-11): a sport that does not declare `[reference] match` gets a **warning** at stage 5 with the
hint; once it is frozen (hidden official seeds registered, `[official.hidden_seeds]`) the missing declaration fails.

## Owners (2026-10-10 round)

| Owner | Sports |
|---|---|
| Chi Gui | table tennis (the first one through the process; its report is the example), badminton, soccer; the process itself |
| Changhe Chen | basketball 3x3, tennis, fencing |
| Kangrui Wang | sprint and hurdles, racing, drone, bowling, archery |
| Pingyue Zhang | volleyball, field hockey, throwing, baseball |
| Qineng Wang | air hockey, curling, billiards, darts, foosball |

Out of this round: basketball shooting, golf putting, disc golf. New sports in progress (basketball 3x3, volleyball,
field hockey, throwing, sprint and hurdles) join the round once they pass the process.
