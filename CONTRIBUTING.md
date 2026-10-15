# Contributing to RoboOlympics

The design is [docs/framework_zh.md](docs/framework_zh.md) (v0.2): four parts (`players/`, `sports/`, `core/`,
`tools/`), the engines underneath (one physics engine per sport, pluggable `renderers/`), and three contracts between
them: ① the player protocol, ② the sport interface, ③ the match record. **Each kind of change touches one place**, and
every boundary has a machine check:

| You want to | Change only | Do not touch | Check |
|---|---|---|---|
| add a sport | `sports/<sport>/` | core, players, tools | `tools/check <sport>` |
| add a model or a harness | `players/` (adapter + one row of `players/players.toml`) | sports, core | smoke match + no-code red-team test |
| add a renderer | `renderers/<name>/` + one registry line | sports, core | `renderers/test_renderers.py` |
| open a season | `seasons/<name>.toml` | all code | version freeze |
| add a viewer or a report | `tools/` | anything that affects a match | runs on existing records |
| change match rules (protocol, formats, validity) | `core/` in its own PR, with a version bump | the sports | every sport's golden replay |
| add or change a robot (model, limits, attachment, control library) | `bodies/` (the robot library) | core; the sports that use it change in their own PRs | `python -m bodies.conformance <robot>`, `ci_checks/check-robot-limits.py` |
| give a sport the two modes (high / low) | `sports/<sport>/` (uses the shared `skilleval/` library) | core, `skilleval/` | `tools/check <sport>` (stages 1 and 6), `ci_checks/check-skilleval.py` |

A pull request touches **one** part (`ci_checks/check-pr-scope.sh`, run with `RO_PR_BASE=<target>`); documentation
may come with it. Python >= 3.11 on the host (the tools use `tomllib`) and Docker (the simulators run in their images).

## Add a sport

> **Before a sport enters a release** it goes through the construction and screening process
> ([docs/construction_zh.md](docs/construction_zh.md)): six steps, each ending with a check. Open **one pull request per
> sport**, with its screening report in `docs/screening/<sport>.md` ([docs/screening/README.md](docs/screening/README.md)).

**Before you start: the five entry gates** (docs/framework_zh.md §8; state each one in the PR description):
1. **Physics is credible**: the engine simulates the sport for real (swimming and figure skating fail here).
2. **The model has real decisions**: the result is not decided by a low-level controller alone.
3. **Playable from the cameras only**.
4. **An official rulebook**, and a referee can judge by it.
5. **A reference player wins, and the duel is fair**: against a scripted opponent, the reference with truth
   perception wins the lowest level and camera-only play can win (the camera reference, or a model in step 5); the
   camera reference plays every format end to end (docs/screening/README.md, "Reference check").

The PR also states **what the sport adds**: a new physics type, a capability that is covered too little, or which
exhibition sport it replaces (docs/capabilities_zh.md §7).

**Capability declaration** (docs/capabilities_zh.md, v3): `sport.toml [capabilities]` gives seven values, each 2
(primary), 1 (secondary) or 0 (none), in this order: `a1` 3D 空间理解, `a2` 物理动力学理解, `a3` 具身与执行, `a4` 时间与延迟
(real-time sports only: a turn-based sport declares 0, and tools/check requires it), `b1` 规则, `b2` 策略与读对手,
`b3` 团队协作; and `physics`, the physics types tested, from the fixed vocabulary `ballistics` (弹道), `wind` (风),
`spin_bounce` (旋转与弹跳), `drag` (空气阻力), `friction_rolling` (摩擦与滚动), `collision` (碰撞), `inertia_latency` (惯性与延迟),
`locomotion` (行走与移动), `contact` (接触); the Chinese names of docs/capabilities_zh.md map onto it as in
`tools/capabilities.py` (`PHYSICS_ZH`). C1 实时编程 and C2 在比赛中学习 apply to every sport and are not declared. The
modes are not declared either: they come from `[modes]`. **Evaluation settings (modes, slowdown) never introduce new
abilities**: they change who does an ability and how hard it is, so a sport's capabilities are the same in every
setting. `tools/check` stage 1 requires the section; `python3 tools/capabilities.py` prints the sport x capability
matrix (JSON) the site uses.

1. **Copy the example**: `cp -r sports/hello sports/<sport>`; rename in `sport.toml` and `plugin.py`; delete
   `golden/`. hello is target throwing, turn-based, single + duel, a high and a low action level, about 200 lines, and the two modes (the skill `aim`; the section "如何给一项运动加两档" below).
2. **Write the five required items**:
   - `sport.toml`, the declarations (data, not code): `[sport]` (protocol `turn` or `realtime`, revision, physics
     engine, simulator image), `[rulebook]` + every `[[deviations]]` entry (id, rule-book article, text,
     `affects_result`), `[formats.*]` (seats, first orders; a solo single format has a `success` line),
     `[modes]` (`low` is required), `[timing]` (a real-time sport: `reaction_delay_steps`, below), `[observation]` (the **whitelist** of `observe()` fields, the players'
     renderer, the frame budget), `[spec]` (the rule constants, shown in `/spec` and the prompt), `[tiebreaks]`
     (order and text; the last rule always decides), `[[exploits]]`, `[players]`, `[fairness]`, `[[tasks]]`, and
     `[score]`, the continuous score (below, "Freeze a sport").
   - **world + referee** (`plugin.py`, a `MatchBase` subclass): `step()` (realtime), `state_digest()`, `phase()`,
     `scores()`, `tiebreaks()`, `to_move()` (turn). Random numbers only from `core.rng.stream(cfg, name)`; the
     referee judges physical events and never edits the physics to produce a result.
   - **low-level actions**: `schema(seat)` + `apply(seat, cmd)`; the core validates bodies against the schema
     (`SchemaLayer`). A high level is optional: an `ActionLayer` whose `lower()` turns it into a low-level command (the
     episode logs low-level commands only).
   - **observation**: `observe(seat, query)` (own body + referee information, nothing else) and `cameras(seat, query)`
     (camera definitions; the task's renderer draws them from `Sport.scene(cfg)` and `render_state()`).
   - **reference player** (`builtin_players.py`): plays from the camera images and `/spec` only, imports no
     simulator, and plays every format end to end. It is also the task's oracle (`solution/`). A sport with a
     scripted opponent (or a one-side format with graded levels) also declares `[reference] match`, a function that plays the reference with truth or camera
     perception for the opponent calibration (docs/screening/README.md, "Reference check"); for a realtime sport the
     shared hook makes it two lines in `sports/<sport>/screen/reference.py` (`from reference_hook import hook`,
     `match = hook(...)`; docs/screening/README.md, "The shared hook"). Without a truth variant the camera reference
     stands in for the truth matches: a NOTE before the freeze, a FAIL once the sport is frozen.
   Optional: a high-level plugin, scripted opponents (single format), the robot's own model (low level), a broadcast
   adapter. Stage-0 hooks a sport may use (all optional, defaults in `core/sport.py`): several seats per side and
   `entities()` (one act in flight per entity), more than two sides with `ranking()` and `retire()`, `delay_steps`
   on an action layer, observation queries, a `[team_channel]`, reply images (`reply_views(seat, cmd, info)`: the
   cameras, or a short strip of frames, that show an action's result in its reply; declare them in
   `[observation] reply_images`; `sports/hello` shows the overhead view of the ball at rest).
3. **Tests**: `tests/test_*.py` with one test per `[[exploits]]` entry (e.g. "stalling gains nothing", "an
   out-of-range command is refused"). The generic suite (determinism, no draw, mirror symmetry, whitelist, purity,
   malformed actions, frame budget) comes from `core/testkit.py`; you do not write it.
4. **Golden matches**: `python3 tools/golden.py record <sport> <case> --format duel --seed 1 --first A` (built-in
   players); for a sport migrated from the legacy tree add `golden/convert.py` so that the legacy records of the
   golden-data store (`RO_GOLDEN_DATA`) replay identically.
5. **Check**: `PYTHON=python3.11 tools/check <sport>` runs (1) the `sport.toml` contract, (2) the generic tests in the
   sport's image, (3) the golden replays in a fresh process, (4) the fairness check (built-in players, both first
   orders: the first-mover and the seat advantage with their 95% confidence intervals; it fails only if an interval
   lies entirely beyond ±5 percentage points; an interval wider than ±5 pp (too few matches) also fails for a turn-based
   sport and only warns for a realtime one; N is
   reported; set `[fairness] matches_per_order` so that N gives an interval within ±5 pp, about 400 per order), (5) a live run of the generated tasks (nop on
   every seat = success 0; reference on every seat = success 1 in a duel and a solo format, a valid match to its end
   against a scripted opponent or in a one-side format with levels; `--harbor` runs it through Harbor) and, for those
   formats, the reference check (truth reference >= 20 seeds per level, camera-only evidence, camera playability
   reported; `[reference]`, docs/screening/README.md; without the declaration a warning until the sport is frozen). All must pass. A slow sport may raise the time
   limits in sport.toml `[check]`: `stage_timeout_s` (each in-image stage, default 3600) and `live_player_timeout_s` (the
   offline live run's players, default 3000), and a generated task's post-match limits in `[timing]`: `replay_timeout_s`
   (the fresh-process replay, default 600; stage 2's fresh-process replays take it too when it is above their 1800 s,
   so a long match also needs a `stage_timeout_s` that holds two plays and the replay) and `video_timeout_s` (the
   spectator video, default 300). A sport with the
   two modes also runs (6): the post-match evaluator on our implementation, the low level's reference skill and
   boundary baseline, the B9 test and the isolation probe (next section).
6. **Generate and open the PR**: `python3 tools/generate.py <sport>` writes `tasks/<sport>-*` (tasks are never
   edited by hand); refresh `tasks/dataset.toml` (`cd tasks && harbor add --scan .`); the PR changes only
   `sports/<sport>/` and its generated tasks.

### Declare the reaction delay (real-time sports, core 0.2.18)

`sport.toml [timing] reaction_delay_steps = D` (an integer >= 0, in control steps, the same for every seat) is the
sport's declared reaction delay. In a match an act takes effect, and is lowered by the action layer, at step
**max(the next control-step boundary, s + D)**, where s is the step of the latest world state the service sent to the
seat before the act arrived (the `now` of any reply: observe, act, wait, status, ready). A program that answers within
D steps therefore always gets exactly D, whatever the machine's load; a slower one gets its own latency, as before.
`/spec` shows `timing.reaction_delay_steps`, the act's reply `took_effect.seen_step` (s), and the generated prompt
states the rule. A layer's `delay_steps` and the sport's own leads (e.g. a planned stroke's start) count from the effect
step. The screening harness (`tools/screen/play.py`) plays with the same D (a hooks `DELAY` that differs is an error),
and `core/local.py` too, so step ④ and the official matches agree by construction. Not declared: D = 0, an act takes
effect at the next control-step boundary (the rule before 0.2.18; `tools/check` stage 1 warns).

How to choose D: the median observe-to-effect latency a normally written program reaches at the sport's official
slowdown (measure the reference program through the service), not the tail; D changes the game, so set it together
with the calibration (step ④) and re-run step ④ when it changes (wheeled 5v5 at x10: D = 2, review LATENCY 2026-10-13;
at D = 3 passing stopped paying). Remove `DELAY` from the screening hooks in the same PR.

### Freeze a sport: hidden official seeds and the continuous score (core 0.2.14)

Step ⑥ of the construction process (docs/construction_zh.md, "冻结"): before the official matches of a season, the sport
owner does three things, in the sport's own PR (docs/framework_zh.md §6.4 explains the design).

1. **Declare the continuous score** in `sport.toml [score]`. **One principle for every sport**: one number per side,
   **higher is better, in the same direction as winning**, and a **fixed `forfeit` value** (the worst value the
   definition allows) for a side that forfeits (no_show, abandoned, clock, review time) or retires, so that nobody keeps
   a good number by walking away. Within it the owner picks one of four kinds:
   ```toml
   [score]
   name = "point_share"                  # an id: [a-z][a-z0-9_]*
   text = "points won / points played"   # the definition in one sentence (README, site)
   from = "share"                        # share | margin | own (from Match.scores()) | hook (Match.continuous_score())
   forfeit = 0.0                         # the fixed value of a side that forfeits: the worst one
   range = [0, 1]                        # optional; the generic tests check the values against it
   ```
   `share` = own score / every side's score (0.5 if nothing was scored), `margin` = own score minus the best other
   score, `own` = own score; for anything else implement `Match.continuous_score() -> {side: number}` and declare
   `from = "hook"` (`sports/hello` is the example: the mean points per ball). Suggested choices:

   | sport | what `scores()` counts | suggested `[score]` |
   |---|---|---|
   | table tennis | points of the game (single: with `opponent`) | `share` (point share), `forfeit = 0`, `range = [0, 1]` |
   | badminton | points | `share` |
   | tennis | points | `share` (a `hook` on points if `scores()` counts games or sets) |
   | soccer | goals | `margin` (goal difference), a fixed `forfeit` floor (e.g. the rulebook's 0:3 forfeit score, -3) |
   | darts | **legs** | `hook`: the mean score per dart, `forfeit = 0` |
   | air hockey, foosball, fencing, curling | points / goals | `share` or `margin` |
   | billiards | **balls left (lower is better)** | `hook` (e.g. the share of own balls pocketed); never `share` |
   | archery, basketball | total score / makes (single) | `own` (or a `hook` per arrow / per shot) |
   | golf putting, disc golf, bowling, baseball single, drone, racing | strokes and the like, not always higher-is-better | a `hook` that makes higher better (e.g. minus the strokes) |

   `tools/check` stage 1 checks the declaration (a sport with hidden seeds must have one); the generic test
   `continuous_score` checks the values (finite, within `range`, never below `forfeit`; a no-show gets `forfeit`).
2. **Generate the hidden seed set.** The set is partitioned by **format x level**: one cell per level of a single format
   with levels (`match/L1`, `match/L2`, ...) and one per other format (`duel`, `team`, ...), each with its own seeds.
   - **Where the private file lives**: only on the official run host, **mll-4090-1**, in a restricted directory, e.g.
     `/data/guichi/secrets/seeds/` (directory mode 0700, files 0600; set `RO_SEEDS_DIR` to it). **The user holds it and
     keeps the backup**; it never goes into the repository, a job directory, a chat or an e-mail. Without it no official
     match can be run, and the registration cannot be changed without a new sport revision.
   - **Who generates it**: on mll-4090-1, `RO_SEEDS_DIR=/data/guichi/secrets/seeds python3 tools/seeds.py new <sport>
     --count N` (N seeds per cell, e.g. 20; `--counts match/L3=40` for single cells; `--from FILE` keeps seeds
     generated earlier, `{"<format>": [seeds] | {"<level>": [seeds]}}`, with fresh salts). A collaborator who owns a
     sport asks the user to run it (or runs it there) and commits only what it writes into the repository:
     `sports/<sport>/seed_commitments.json` and `sport.toml [official.hidden_seeds]` (`sha256`, `counts`).
     `python3 tools/seeds.py check <sport>` checks the three agree.
   - **Collaborators develop with the public seeds** (`[official] seeds`): they never need the private set. Official
     matches are run on mll-4090-1 with `--seed hidden:I` (the run scripts resolve it in the match's cell:
     `tools/seeds.py resolve`); `scripts/olympiad/plan.py show <sport>` lists `hidden:0..N-1` per cell.
   - **Nothing shows a hidden seed**: the sim gets it through its environment only; the episode, `final.json`,
     `replay_check.json`, the broadcast log, `reward.json` (`seed = -1`) and the players' replies carry the commitment
     and the index (`hidden_seed_index`), never the seed or its salt. An offline replay of such an episode works only
     on the host with the private set (`RO_SEED`, `RO_SEED_SALT` from `tools/seeds.py resolve`).
3. **Regenerate and check**: `python3 tools/generate.py <sport>` (the verifier's `contract.json` gets the commitments),
   `PYTHON=python3.11 tools/check <sport> --stages 1`, `bash ci_checks/check-hidden-seeds.sh`. From now on only the
   hidden seeds of the right cell are official (`official_config = 1` needs `hidden_seed = 1`).

**Frozen = registered hidden seeds.** A sport is frozen (official) exactly when its `sport.toml` has
`[official.hidden_seeds]` and `sports/<sport>/seed_commitments.json` verifies against it (core/seeds.py: the
commitments hash to the registered `sha256`, one commitment per seed in each cell's count). A release's official part
takes only frozen sports; every other published sport is in screening. `python3 tools/frozen.py` lists every sport
with its status, `publish`, frozen yes/no, revision and registered cells (`--json` for release tooling, `--names` for
the frozen ones); a registration that does not verify is reported as not frozen and the tool exits 1.

## 如何给一项运动加两档（Add the two modes to a sport）

*English summary: the two modes are defined in docs/framework_zh.md §6.3 and the robot and mode rules of §9. High
(Agent as Athlete): the platform provides the sport's skill API. Low (Agent as Robotics Engineer): the agent implements
the same API from the robot model, the joint interface with the real limits and a disclosed toolkit, delivers it at a
fixed path under `/app/output/skill/`, plays the match as usual, and after the match the final implementation is
evaluated in an isolated container (API test on hidden requests + competition test with our fixed reference
strategy). A sport declares both in `sport.toml [modes.high]` / `[modes.low]` and builds on the shared `skilleval/`
library ([skilleval/README.md](skilleval/README.md)); `sports/hello` is the minimal complete example (skill `aim(x, y)`).
Checklist: robot from `bodies/` (or a declared machine), real limits through the shared servo, a B9 test, the skill
contract, the toolkit manifest, reference solutions for both modes, a boundary baseline that scores about 0, the
evaluator hooks; then `tools/check <sport>` (stages 1-6) and `python3 tools/skill_eval.py <sport> reference|boundary|platform
--record`. The skill in the loop (item 12): `[modes.low.inloop]` + `layer_from_env(...)` in the plugin's low level make
the generated low tasks play the match through the player's own skill, run in a skill container per seat.*

两档的定义在 [docs/framework_zh.md §6.3](docs/framework_zh.md) 和第 9 节"机器人与模式规则"；共同的部分在共享库
[`skilleval/`](skilleval/README.md)，运动只写自己特有的东西。最小的完整例子是 `sports/hello`（技能 `aim(x, y)`：高层由平台把落点降成
一次出手，底层由选手用发射器的飞行模型自己实现）。按下面的清单做，每一项都有自动检查：

1. **选机器人**：从 `bodies/` 选一个（`sport.toml [body]`，固定基座的机械臂用 `[arm] body`），装配配置（机器人 + 手或工具 +
   控制器）由 `bodies/assembly.py` 给出版本 hash；没有机器人的项目在 `[modes.low] machine` 里声明自己的机器（出手装置、赛车、
   桌上足球杆）。`declare.py` 要求二者有且只有一个。
2. **真实限位（B0）**：机器人的每条命令都经过共享的限位和伺服模块（`bodies/servo.py`、`bodies/limits.py`），平台的高层技能
   和选手的底层命令过同一道检查；`ci_checks/check-robot-limits.py` 检查不绕开、不放宽。工具包清单的 `[limits]` 写明来源和
   执行者（用资源库机器人时必须是 `bodies/`）；声明的机器以它的命令 schema 为限位。
3. **B9 可行性测试**：`[modes] b9_test` 指向 `tests/` 里的一个测试：在真实限位下，上界选手（知道真实状态）完成这项运动的
   基本动作（球拍类：合法发球上界可回 ≥ 90%）；至少要证明 API 测试的每个请求都可行。`tools/check` 第 6 步在运动镜像里运行它。
4. **技能 API 合同**：一个合同模块（hello：`skill_api.py`）导出 `CONTRACTS = {"<技能>": SkillContract(...)}`
   （`skilleval/contract.py`）：请求字段和范围、返回什么、怎么测、容差、成功条件、为什么每个请求都可行。高层动作的 schema
   **由合同生成**（单一来源），`[modes.high]` 写 `skills`、`contract`、`implementation`（我们的实现，`文件.py:名字`）。
   运动的 `Sport.spec()` 把 `skill_contract` 放进 `/spec`。
5. **工具包清单**：`seat/toolkit.toml`（格式见 `skilleval/toolkit.py`）列出底层给的一切：每个工具（模型、库、策略、控制器、
   接口）的座位路径、源文件、sha256、版本、API、许可证；`[limits]`；`[forbidden]`（核心技能，以及不进座位的模块，如平台的
   实现）；`[boundary]`。`[seat.low] files` 把清单里的文件、清单本身（`toolkit_path`）和练习集原样装进座位；
   `python3 -m skilleval.toolkit check sports/<运动>/seat/toolkit.toml --fix` 刷新 sha256。清单是任务版本的一部分：改了就升
   `[toolkit] version` 和运动的 `revision`，重跑参考解。
6. **交付约定**（规则 10）：`[modes.low.deliverable]` 写 `path`（`/app/output/skill/<技能>.py`）、`signature`
   （`<技能>(..., *, state, spec)`）、可选函数、`allowed`（能用什么）、`time_limits_s`（`load` 和每个函数）、`memory`
   （`per_run`：一次评测一个进程；`per_case`：每个请求一个新进程）。生成器据此给底层任务的提示词加上"工程任务"一节，并在
   `task.toml` 里写明交付路径和评测命令。
7. **两档的参考解**（规则 4）：高层 = 平台的实现 + `[players] reference`（只看相机就能赢）；底层 = `lowlevel/reference/`，
   只用清单里的工具实现技能，API 测试 ≥ `thresholds.reference_min`，并且确定性。
8. **边界基线 ≈ 0**（规则 5）：`lowlevel/boundary/` 是只用工具、不开发技能的最简实现（乒乓：IK 把拍子挡在球路上；hello：
   对准方向、固定出手速度），API 测试 ≤ `thresholds.boundary_max`（= 清单 `[boundary] threshold`）。做不到就收紧合同的
   成功条件，不拿掉工具。
9. **隐藏请求**：从我们参考选手的真实比赛里采样，只留真实限位下可行的（平台的实现能完成）；`skilleval.cases.split` 按 id 的哈希
   分成 `hidden`（只在模拟器一侧：生成的任务里没有，座位里更没有）和 `public`（练习集，只含请求，经 `[seat.low] files` 放到
   `practice_path`）。hello：`python3 sports/hello/lowlevel/evaluator.py make-cases`（在运动镜像里）。
10. **评测钩子**：`[modes.low] evaluator` 模块提供 `skill_spec()`（技能每次调用收到的 `spec`，即底层座位的 `/spec`）、
    `platform()`（我们的实现，同样的函数）、`api_case(case, skill)`（执行一个请求，返回测量值），可选
    `competition_match(skill, seed)`、`COMPETITION_SEEDS`、`fallback`。回合制项目的比赛测试直接用 `skilleval.layer.SkillLayer`
    （固定参考策略的高层动作由选手的实现降成底层命令）；实时项目在 `competition_match` 里写自己的循环，用同一个
    `skill.call(...)` 接口。
11. **检查**：`tools/check <运动>`：第 1 步查声明（`skilleval/declare.py`），第 6 步在隔离容器里跑平台实现（每个隐藏请求可行、
    合同 = 高层 schema）、参考解、边界基线、B9 测试和隔离探针。`python3 tools/skill_eval.py <运动> platform|reference|boundary
    --record` 把参考行写进 `lowlevel/reference_results.json`（每份报告都附上它）。赛后评某次试次：
    `python3 tools/skill_eval.py <运动> <试次>/artifacts/app/output/skill --out DIR`。
12. **技能在环**（底层比赛必须经过选手自己的技能，2026-10-10 的决定；`skilleval/inloop.py`）：`sport.toml` 加
    `[modes.low.inloop]`（`skills`、`step_every`、`free`：球在场上也随时放行的底层命令类型、`noop`：失败时生效的空命令、
    技能容器的 `cpus` / `memory`、`reference_class`：把参考选手的决策发成技能调用的 oracle、`raw_rule`：提示词里"什么时候
    拒绝关节命令"的说法；格式见 `skilleval/declare.py`）。`plugin.py` 的 `action_layer("low")` 返回
    `skilleval.inloop.layer_from_env(<原来的底层>, CONTRACTS, decl, spec=..., control_dt=..., raw_allowed=...) or <原来的底层>`
    （`raw_allowed(seat, match, cmd)` 看得到命令，可以按类型放行，例：走路随时可以、手臂动作必须经过技能；多机器人的座位
    可给 `state_of(seat, match, cmd)`），`apply` 认 `noop`。`tools/generate.py` 据此给底层任务每个座位生成一个技能容器
    （没有网络、只读、看不到模拟器、技能以 nobody 运行，计算时间计入比赛时间）。本地实跑：
    `python3 tools/live.py tasks/<底层任务> OUT oracle`（一个容器）。乒乓是例子（`sports/tabletennis`）。

模式开关不引入新的能力（`[capabilities]` 两档相同）。改 `skilleval/` 本身是另一个 PR（它被所有两档运动共用，改后跑每个两档运动
的第 6 步）。

## Add or change a robot (the robot library, `bodies/`)

Robots are a shared resource library, not a fifth part: `bodies/README.md` (机器人资源库) has the three concepts (the
robot's base model with its real limits; optional control libraries; modes declared by the sport), how a sport uses a
robot and an attachment (`sport.toml [body]`, or `[arm] body` for the limits-only arms), and the admission rules:

1. Source and licence: the model and meshes are redistributable (a licence file in the robot's directory, a line in
   `NOTICE`); `tools/make_model.py` records where they come from and every change; fine meshes stay out of git. A robot
   whose licence is not cleared is `status = "test-use"` and may not be used by a published sport.
2. Real limits with sources (binding, 2026-10-08): every limit in `body.toml` has a source in `[limits.<group>.source]`
   (the SDK / controller value over the URDF's; an unpublished value is `ESTIMATE:` with its method, or `NOT
   PUBLISHED:` with the reason it is not limited). Change a number: bump the robot's `revision`.
3. One limit / servo path: the robot's servo is built in `bodies/servo.py`; nothing else writes its actuators, and a
   sport drives it through its rig. `ci_checks/check-robot-limits.py` fails a sport that bypasses or loosens it.
4. Conformance: `python -m bodies.conformance <robot>` passes in a sport image (limits declared, servo reaches
   targets, limits enforced, appearance meshes, deterministic replay, the assembled configuration's hash).

A sport that adopts a new robot revision re-records its goldens in its own PR.

## Add a player

A player = a model + its harness + its play configuration (program play; direct, no-code play was removed from main on
2026-10-10 and is kept on branch `archive/direct-play`). Put the harness adapter in `players/harnesses/` (a Harbor
agent class) and add one `[[player]]` row to `players/players.toml` (model, harness, pinned version, play modes,
account **variable names** only, `config_version`); `python3 players/players.py check`. Evidence: one smoke match.
Players never know a sport and never reach match state. See [players/README.md](players/README.md).

## Add a renderer

Implement `renderers.base.Renderer` (`render(state, cameras) -> images`) in `renderers/<name>/` and register it in
`renderers/__init__.py`. Rules: it never changes physics or match state; it draws the same scene (geometry and camera
poses equal to the default MuJoCo renderer's, checked before drawing); for the players' images it meets the task's
frame budget (`renderers/budget.py`) and declares a GPU; re-rendered frames are labelled with its name. See
[renderers/README.md](renderers/README.md).

## Change the core

`core/` names no sport and no harness (`ci_checks/check-core-purity.sh`), computes no physics, and changes in its own
PR with a version bump (`core/__init__.py`), the core tests (`core/tests/test_core.py`) and every sport's golden replay
(`tools/golden.py replay <sport>`) passing.

## Legacy sports

Every sport of Season 1 is migrated to `sports/<sport>/` and `tools/generate.py`. The legacy generator
(`scripts/olympiad/generate.py`), its rule check (`ci_checks/check-olympiad.py`), the stage-0 `wrapper` status and the
legacy metering reference copies were removed on 2026-10-10; they are in the git history before that date.

## Rules for every contribution

- Generated tasks are never edited by hand; change the sources and regenerate.
- Pin everything (`FROM` digests, pip versions); no network during verification; the verifier runs in a separate
  sandbox and grades only declared artifacts.
- Shared metering files stay byte-identical (`ci_checks/check-v02-shared.sh`, rule I2).
- No credentials (only variable names), personal absolute paths, large binaries (> 5 MB), job directories, videos or
  agent transcripts in the repository (`ci_checks/check-no-host-paths.sh` for the v0.2 parts).
- New Python code is formatted with `ruff format` (line length 120).
- Before any PR: `bash ci_checks/run_all.sh` (all checks passed).
