# RoboOlympics: play modes and action levels

> **2026-10-10: direct (no-code) play was removed from main.** Only program play remains; the action-level switch
> (high / low) is unchanged. The direct-play code (the `game` MCP tool, direct-gate, the direct-native and
> direct-player harnesses), the two tasks `tabletennis-{match,duel}-wallclock-direct-standard` and their tests are on
> branch [`archive/direct-play`](https://github.com/CHIGUI0/roboolympics/tree/archive/direct-play). The sections on
> direct play below (§2.5, §13, §15-§18 and the direct columns elsewhere) are kept as the design record. Still on
> main: the `play` field (always `program`), the verifier's no-code audit `players/nocode/nocode_audit.py` that every
> task ships in `tests/` (`core/grading.py` applies it only when `play` = `direct`) and the reward keys `direct_play`
> (always 0) and table tennis' `nocode_*` / `gate_blocks` (always 0), so the program tasks' reward.json is unchanged.

Status: **draft v0.3** (2026-10-06), from the meeting of 2026-10-06; v0.3: the low level's robot model and information
boundary (§2.7), same slowdown for both levels, score ladder (§5), high vs low results (§14.1); v0.2: the modes are two switches of one
simulator core per sport (§2.6). Implemented for table tennis single player:
`tasks/tabletennis-match-wallclock-program-standard` (high, program: unchanged behaviour),
`tasks/tabletennis-match-wallclock-program-lowlevel-standard` (low, program) and
`tasks/tabletennis-match-wallclock-direct-standard` (high, direct). It extends the sport rules
([roboolympics-rules.md](roboolympics-rules.md)) and the real-time protocol
([EAI_STANDARD_REALTIME_1_0.md](../EAI_STANDARD_REALTIME_1_0.md)); where it changes a rule of those pages it says so.

## 中文摘要

- **两个维度**:(a) 是否允许写代码:`program`(现行,程序参赛)/ `direct`(无代码,模型本身当控制器,像 VLA,每个动作都要一次模型调用);(b) 动作层级:`high`(现行,高层插件打开:模拟器替你做 IK / 规划 / 步行控制)/ `low`(高层插件关闭,关节位置目标 100 Hz 等执行器级命令,模拟器只保留物理、关节/力矩/速度限制、关节 PD 控制器和裁判,不做 IK、轨迹规划、动作预演)。
- **不是每个项目都做 2x2**,按项目挑选:低层控制适合手臂类项目(乒乓、网球、羽毛球、击剑、棒球击球、飞镖/射箭/保龄的出手)和田径类的人形运动;无代码适合回合制和慢节奏的项目(台球、冰壶、高尔夫、飞镖、射箭、保龄、篮球投篮、飞盘高尔夫),实时快节奏项目在无代码下需要 x50–x100 的减速。
- **结构(v0.2)**:每个项目只有一个模拟器核心。最底层永远是低层(关节/执行器)接口:`arm_commands.py`（原 `core.py`）的手臂命令层是关节目标的唯一来源。高层接口(IK、挥拍规划、预演)是叠在上面的插件 `action_high.py`:打开就是现行的高层模式(把 act 规划成一条 `traj` 交给命令层),关闭就是低层模式(`action_low.py`,命令直接进命令层)。物理、裁判、计分、回放、转播日志全部共用。play 开关只改三处:说明书中间一节(共用片段拼出)、服务器的“每回复一次 act”规则、验证器的工具审计。`spec.py` 用 `MODES` 声明支持的组合,`generate.py` 按开关把同一份任务源渲染成各个任务目录(填 `@@KEY@@` 占位符、复制共用审计);原来的 `base=` 叠加机制已删除。
- **现有任务不变**:原高层服务在 Harbor 上录下的一场比赛(L1、seed 101、97 个动作、2260 步),用新核心加高层插件回放,关节目标与最终状态逐位一致(单元测试);同一批动作分别走 Match 原有的规划路径和新插件,每一步结果相同。现有任务目录的 instruction.md、README.md 字节一致;变化的只有 server/replay(走开关)、game(多了低层命令)、task.toml(+play/action_level/category/timing_policy)、compose(+3 行开关)、grader(+审计,只对 direct 生效)和新增的共用文件。
- **违规 = 判负**(已定)。哪些模式进入赛季尚未决定(赛季和任务清单都还没定)。
- **任务命名**:`<sport>-<kind>[-wallclock]-<play>[-lowlevel]-standard`,高层动作不写(保持现有 id 不变);例:`tabletennis-match-wallclock-program-lowlevel-standard`、`tabletennis-match-wallclock-direct-standard`。`task.toml` 新增 `play` 与 `action_level`(`high`/`low`)字段;线协议仍为 `eai-standard-realtime/1.0`(低层动作只是 `act` 的另一种请求体);无代码的“每个动作一次模型回复 + 工具审计”规则写进规则 v1.1,第二个项目采用时再升协议版本。
- **无代码的强制手段**:三层——(1) harness 限制工具(Claude Code `allowed_tools=Bash(game:*),Read`;Codex 只能关掉网页搜索,无法白名单命令);(2) 服务器端每个 `act` 必须有一次新的、包含 `game act` 工具调用的已完成模型回复(计量代理 `RCB_ACT_RULE=per_response`,否则 429);(3) 验证器审计整场会话的每一个工具调用(`think_calls.jsonl`):只允许字面量 `game ...` 命令(无管道、重定向、变量、命令替换)和打开保存的相机图像,其他任何调用判负(有效的输局,不是技术故障)。容器里去掉代码运行时不可行(harness 本身就在容器里运行 node/python)。
- **原型**(乒乓单打):低层任务 oracle 用自己写的 numpy FK/IK/挥拍规划,流式发送关节目标,在默认配置(L1、x20、seed 101)获胜;无代码任务用高层 oracle 在 x50 获胜;nop 均为 0;单元测试通过(见第 8 节);`check-olympiad` 0 失败。luna 冒烟一次:0:5 负,但它自己写了 FK、IK 和三角测量并碰到一次球,说明模型能玩低层,但远未能回球。
- **低层的信息边界(2026-10-06 定)**:①给:相机图像、服务回复(自己手臂状态、比分、听到的弹跳、击球结果)、`game spec`、说明书,以及机器人自己的模型 `/opt/robot/robot_only.xml`(只有手臂+球拍:运动学树、关节限位、质量惯量、电机、球拍安装;没有球、球台、网、对手、场景)和预装的 MuJoCo 3.3.7(与模拟器同版本)、NumPy、SciPy(比赛中不能 pip);②智能体自己算:找球、三角测量、预测飞行、击球点与时机、IK、挥拍、跟踪滞后补偿;③不给/禁止:模拟器状态(球的位置速度旋转、对手计划)、比赛模拟器本身及其场景文件、任务专用求解器。`robot_only.xml` 由 `robot_model.py` 从模拟器场景自动截取(不会漂移),单元测试:500 个随机关节构型的 FK、质量矩阵、偏置力与模拟器差 < 1e-9,双人赛两个座位在各自坐标系中也一致;只发给低层座位,高层任务不变。见 §2.7。
- **同一减速比**:高层与低层比较时用同一个减速比(本实验都用 x20)。
- **得分阶梯(分析用,不改胜负)**:每分、每球:碰到球 → 主动碰到(拍速 >= 0.2 m/s;静止的拍子在准备姿势也会碰到 2-3 个发球)→ 过网 → 落在对方台面 → 合法回球 → 得分;得分再分成“靠加速规则得分”(接发方合法回球满 13 次,ITTF 2.15.4)和“打穿对手得分”;写进 `final.json` `ladder` 和 `reward.json` `ladder_*`(单打)。双人赛报告每方靠加速规则得的分 `A/B_points_expedite`。
- **加速规则(2026-10-06 定)**:不改规则,两个层级、单打和双人赛同一规则;报告里把加速规则得分和打穿对手得分分开列。单打里模型永远是接发方,这条规则偏向它;双人赛每分轮换接发方,规则对称。
- **热身球**:暂不加;判定规则:若两个模型在整场比赛中都没有一次主动触球,才建议加热身球。本次结果见 §14.1。
- **田径**:人形 100 m / 跨栏只做低层,可行性见第 9 节:物理和规则都简单,但需要给智能体一个练习用模拟器(否则几乎必然 0 分)以及一个训练好的策略作 oracle。

## 1. The two axes

| Axis | Values | Meaning |
|---|---|---|
| Play mode (`play`) | `program` (current) | The agent may write a player program that observes and acts on its own, between model calls. |
| | `direct` (no code) | The model is the controller: it looks at the images and chooses every action itself; no program runs between its calls. |
| Action level (`action_level`) | `high` (current) | Sport-level commands; the simulator turns them into actuator commands (IK, stroke planners, walking controllers, flight controllers, aiming). |
| | `low` | Actuator-level commands at the robot's control rate (joint position targets, joint velocities or torques, motor thrusts); the simulator keeps the physics, the limits, the robot's own joint controller and the referee, and plans nothing. |

Not every sport gets every combination (meeting 2026-10-06: "tasks matter more than modes"). A combination is offered
when it tests something the others do not and can be played inside the session budget (§3).
`direct` + `low` is defined but only useful for slow sports: a model call per 10 ms command cannot hold a fast
body up.

## 2. Definitions

### 2.1 Common to all modes

Unchanged from the sport rules: the world (physics, opponent, referee, cameras), the observation (camera images, own
robot state, referee information; no world state), the rules and their deviations, the formats (single player and
duel), reviews after every point, the evidence (`episode.jsonl` with every applied command and its step, deterministic
replay in a fresh process, separate verifier with no network), the model-API validity rule and the isolation of the
seats. A mode changes only (a) what an `act` contains and what the service does with it, (b) who may send it, (c) the
slowdowns and the session limit, and (d) the prompt's "Write a program" section.

### 2.2 Action level `high` (current)

The sport's high-level action (rules §4 A1): e.g. table tennis "blade centre, normal and velocity at contact T s
later" (the service plans the stroke: IK, a joint-speed-limited swing, rehearsals on a simulator copy that correct the
controller's lag, seeded execution scatter); baseball a swing plan; soccer walk/kick commands for a walking
controller; racing throttle and steering; drone attitude/thrust set-points for a flight controller.

### 2.3 Action level `low`

| | Definition |
|---|---|
| Action space | The commands of the robot's own actuator interface, the same interface the high-level planner drives: an arm's 7 joint position targets (rad) per control step; a humanoid's joint position targets (PD) or torques; a car's wheel torque and steering angle; a quadrotor's four rotor thrusts; foosball rod slide and spin velocities. Sport-specific, documented in `/spec` and the prompt with units and ranges. |
| Control rate | The robot's control step (10 ms, 100 Hz) unless the sport's robot runs slower (a humanoid PD at 50 Hz). One command may carry a chunk of future targets (at most 3 s), stamped with absolute control steps, so that a program need not hit every step over HTTP. |
| What the simulator still does | Physics and contacts; joint ranges, joint speed limits, torque limits (refused with 400 when a command breaks them, never clipped silently; rules A2); the robot's joint controller (PD + gravity/Coriolis compensation, torque clipping at 1 kHz), which is part of the robot, as on the real arm; the referee and the opponent. |
| What it no longer does | IK, stroke / trajectory planning, rehearsals on a simulator copy, return-to-ready motions, walking or flight controllers, aiming. No execution scatter: the controller's tracking error is the physics. |
| Observation | As high-level, plus the robot's full proprioception (joint angles, velocities, the target in force) and its kinematic model in `/spec` (body chain, joint axes, tool frame, limits, controller gains). The model is the agent's own body, not world state (rules O1). |
| Robot model and tools (decided 2026-10-06) | The seat image holds the robot's own model as a file (table tennis: `/opt/robot/robot_only.xml`, MJCF of the arm + paddle only: kinematic tree, joint limits, armature/damping, masses/inertias, motors, paddle mount; no ball, table, net, floor, opponent or scene) and generic tools preinstalled (MuJoCo at the simulator's version, NumPy, SciPy; no pip at match time). **No task-specific solver** (no "paddle to pose" or IK helper of ours). The match simulator stays in the `sim` service; the agent reaches it only through `observe` / `act` / `wait` (§2.7). |
| Timing | Same clock and slowdowns as high-level program play. A command takes effect at the start of the next control step after it arrives. |
| Continuity | A command whose first used target jumps from the target in force by more than one step's speed limit is rejected when it would take effect (`result.rejected`); the previous command goes on. This is the real arm's behaviour (libfranka stops on a discontinuity). |
| Between points | The command is cleared at the end of a point: the arm holds its targets. Nothing returns it to a ready pose. |

Table tennis (`shared/tabletennis/sim/arm_commands.py`, formerly `core.py`, + `action_low.py`): `act` takes `{"type":"traj","k0":K0,"q":[[7 targets],...]}`
(row i = target of control step K0 + i, <= 300 rows; passed rows skipped; before K0 the target is held; after the last
row the last row is held), `{"type":"slew","q":[7],"speed":F}` (each joint independently towards q at F x its speed
limit; a servo, not a planner) or `{"type":"hold"}`. The replay re-applies the logged commands at their steps.

### 2.7 Information boundary of the low action level (table tennis; decided 2026-10-06)

| | What | Where |
|---|---|---|
| (1) Given | Camera images; the service's replies (own joint angles/velocities/target in force, blade pose, score, heard bounces, shot and point results); `game spec` (cameras, kinematic chain, limits, controller law and gains, ready pose); the prompt (table, net, ball and physics numbers as in the high level); the robot's own model `robot_only.xml` + meshes; MuJoCo 3.3.7, NumPy, SciPy, OpenCV, Pillow | seat image (`/opt/robot/`), HTTP replies, `instruction.md` |
| (2) Computed by the agent | Ball detection, triangulation and flight prediction; where and when to meet the ball; paddle pose and velocity at contact; IK; the swing (joint-speed-limited trajectory); tracking-lag compensation (the agent may simulate its own arm with MuJoCo and the documented controller) | agent's programs in `/app/output/` |
| (3) Forbidden / not available | Simulator state (ball position, velocity, spin; opponent plan; serve ids; seed); the match simulator, its scene (ball, table, net, contact parameters) and its files; a task-specific solver; installing packages during the match | stays in the `sim` container (no shared volume with the seat; the seat has only `/app/output`) |

How it is kept honest:

- `robot_only.xml` is **generated** from the simulator's scene (`scripts/olympiad/tabletennis/robot_model.py`: copy of
  the `link0` subtree, the arm's defaults, meshes, contact exclusions and motors; everything else dropped), so it
  cannot drift: `generate.py check` compares it, and `test_modes.py` checks that forward kinematics of 500 random joint
  configurations (every chain body and the blade centre, position and orientation), the arm's mass matrix and bias
  forces, ranges, armature, damping, masses, inertias and motors equal the simulator's within 1e-9, that both duel
  seats equal it in their own frames, and that the file holds nothing but the robot (bodies = the chain, 7 hinges, no
  free joint, contact pair, light, camera, texture, material; no ball/table/net/floor/target/pedestal names). Built
  seat image vs sim image: FK of 1000 random configurations bit-identical (max difference 0.0).
- Only the low-level seats get it (single and duel). The high-level seat images are unchanged: their planner does the
  kinematics, the model would add nothing for them, and the Season 1 tasks stay byte-identical (instruction, README,
  Dockerfile).
- Isolation (checked on the Harbor runs of §14): the seat container mounts only the `agent_output` volume at
  `/app/output`; `/srv` is empty and `/data` absent in the seat; none of the simulator's files (`tt_scene.xml`,
  `rcb_tt.py`, `match.py`, `serves_pool.json`, `instance.npz`) exists in the seat image; the sim mounts its code
  read-only at `/srv` in its own container; the seat is on the internal `play` network only during play.
- Exploit check: the model holds no ball, so it cannot reveal ball state; everything it encodes (the arm, its mount
  pose) was already in `/spec`. MuJoCo lets an agent build its own ball/table model from the documented numbers - that
  is computation (2), the same as with NumPy, and its contact parameters are not the simulator's (they are not
  published). The seat has no GL library, so MuJoCo cannot render; irrelevant to play. Privileged state still only
  comes through the cameras.

### 2.4 Play mode `program` (current)

The agent may write, run, stop and fix a player program; `RCB_ACT_RULE=program` (acts need a live, metered model
session, not a new response each); one act in flight per seat. Session limit 2 hours; slowdowns `OFFICIAL_SLOWDOWNS`
(single) / `DUEL_SLOWDOWNS`.

**Same slowdown for both action levels** (decided 2026-10-06): a comparison of `high` and `low` uses the same slowdown
for both (x20 in §14); a level never gets extra time to compensate for its harder interface.

### 2.5 Play mode `direct` (no code)

| | Definition |
|---|---|
| Interface | The same `game` commands (`observe`, `act`, `wait`, `start`, `ready`, `spec`, `status`). The model calls them as tools, one command per tool call with literal arguments, and opens the saved camera images with its image tool. Nothing else. |
| Who acts | Every `act` needs a new completed model response that contains a `game act` tool call (`RCB_ACT_RULE=per_response`; the metering proxy counts such responses; an act without one is refused, HTTP 429, `refused_not_model`). |
| Timing | Real time as in program play, slowed more: `DIRECT_SLOWDOWNS = [20, 50, 100]` (default x50) for real-time sports; turn-based sports keep their chess clock (60 min per player). Session limit 3 hours for real-time single-player tasks (agent timeout 10800 s). |
| Reviews | As in program play (`game ready "WHY" "PLAN"` after every point; duel limit `REVIEW_S`, proposed 600 s in direct play because every review is a model call with images in context). |
| Observation | As in program play. A sport may add a cheaper image (one camera, fewer frames) for direct play; table tennis keeps the 2 x 4 image (about 1.5k image tokens per look). |
| Oracle | The sport's scripted reference player in the oracle seat (`RCB_PLAYER_MODE=oracle`, exempt from the per-response rule and the audit). It shows that the task can be won at that clock; it is not a model. |

**Enforcement of "no code"** (three layers; the third decides):

1. *Harness*. Claude Code: `allowed_tools=Bash(game:*),Read` and `disallowed_tools=Write,Edit,NotebookEdit,Task,WebSearch,WebFetch`.
   Codex has no command allow-list (its exec policy can forbid prefixes, not allow only one): `web_search=disabled`
   only; it relies on layers 2 and 3. The harness layer prevents accidents; it is not trusted.
2. *Service*. The per-response act rule: a background program cannot act faster than the model's own responses, and
   only responses with a `game act` tool call count.
3. *Verifier audit*. The metering proxy records every tool call of every model response (`think_calls.jsonl`). The
   grader allows only (a) a shell call whose command is a literal `game <subcommand> <args>` (quoted texts may hold
   anything but `$`, backticks and backslashes; outside quotes only plain words: no `;`, `|`, `&`, `<`, `>`, `$`,
   backticks, parentheses, braces), (b) opening `/app/output/views/*.png`, (c) the harness's planning note tools
   (`TodoWrite`, `update_plan`). Any other tool call in the session (writing a file, running Python, `cat`, `curl`,
   `cd /app && game ...`, `game act $(...)`) **forfeits the match** (decided 2026-10-06): a valid trial that is lost (`nocode_forfeit`,
   `nocode_violations`, the calls in `nocode_audit.json`), not a technical failure. In a duel the violating seat
   loses.

Why not a container without a code runtime: Claude Code (node) and Codex run inside the seat container, and a shell has
arithmetic and `awk`; a container cannot be made code-free while it hosts the harness. A harness that runs outside the
container and exposes only a `game` tool (an MCP server in the sidecar) would make layer 1 a guarantee; it needs a
Harbor change and is an open decision (§10).

### 2.6 One core, two switches (structure)

Each sport has **one simulator**. Its bottom layer is always the low-level (actuator) interface; the high-level
interface is a plugin on top of it. The play switch never touches the simulator.

| Layer | Table tennis files (`scripts/olympiad/shared/tabletennis/sim/`, copied into every task) | Shared by |
|---|---|---|
| Physics, referee, opponent, scoring | `rcb_tt.py`, `rally.py`, `match.py` (unchanged; `match.py` also holds the stroke planner the plugin calls) | all switches |
| Arm command layer (the only source of joint targets) | `arm_commands.py` (formerly `core.py`): `Core(match.Match)`; commands `traj` (absolute-step joint targets, optional `then`), `slew`, `hold`; continuity rejection; the level's rest command at the end of a point | all switches |
| Action plugin, switch `action_level` | `action_high.py` (on: paddle state at contact -> `plan_paddle` -> one `traj` + `then` slew to the ready pose; rest = slew to ready at 60 %) or `action_low.py` (off: commands straight to the core; rest = hold; kinematic chain and limits for `/spec`); `levels.py` picks one | - |
| Service, replay, evidence, broadcast | `tabletennis/match/environment/sim/server.py`, `replay.py` (switch from `RCB_ACTION_LEVEL`; the replay reads the level from the episode's start record) | all switches |
| Play switch | `RCB_PLAY` + `RCB_ACT_RULE` in the compose file (per-response acts), the prompt's middle section (`program_prompt.single(play=...)`), the verifier audit (`players/nocode/nocode_audit.py` -> `tests/`; `grade.py` applies it when `final.json` says `play` = `direct`) | all sports |
| Agent side | one `game` CLI (high- and low-level `act` forms), one `solution/` (`oracle_play.py` starts `oracle_low.py` + `arm.py` when `/spec` says `action_level` = `low`) | - |
| Generation | `tabletennis/spec.py`: `MODES = {"single": [(high, program), (low, program), (high, direct)]}`, `single_task(level, play)` builds the task spec (prompt, README texts, placeholder values); `scripts/olympiad/modes.py`: names, switch values, defaults; `generate.py`: renders one task source per combination (fills `@@KEY@@` in `task.toml` and `docker-compose.yaml`, copies the audit) | all sports |

The high-level plugin reproduces the original high-level service exactly: a match recorded on Harbor by the original
service (scripted oracle, L1, seed 101, x50, 97 acts, 2260 steps) replays through `Core` + `action_high` to the same
joint targets (actions hash), final state hash and score (`test_highlevel_plugin_reproduces_a_match_recorded_by_the_original_service`),
and the same acts through `match.Match.rt_act` and through the plugin give identical targets
(`test_highlevel_plugin_equals_the_original_planner_path`).

## 3. Sports and modes

Proposed modes per sport. `P-high` = program play, high-level (all sports today); `P-low` = program play,
low-level; `D-high` = direct play, high-level; `D-low` = direct play, low-level. ✓ offer, (✓) later / exhibition,
– not offered.

| Sport | Body / high-level action today | Timing | P-high | P-low | D-high | D-low | Reason |
|---|---|---|---|---|---|---|---|
| Table tennis | Franka + paddle; blade state at contact (service IK + stroke planner) | real time, 0.6-0.9 s per ball | ✓ | ✓ prototype | ✓ prototype (x50) | – | The arm sport where low level means exactly "write your own IK and swing"; direct play is the hardest real-time test (one or two looks per ball at x50). D-low: a 100 Hz arm by model calls cannot meet a ball. |
| Tennis | Franka on a pedestal + x/y base rail; racket state at contact + base target | real time, ~1-1.5 s per ball | ✓ | ✓ | (✓) x50-x100 | – | Same arm interface as table tennis (9 targets: base + 7 joints) - the low-level layer ports directly; longer flights make direct play more plausible than table tennis. |
| Badminton | Franka on a pedestal + sliding base; racket state at contact | real time, ~1 s per shuttle | ✓ | ✓ | (✓) | – | As tennis; the shuttle's drag makes the agent's own swing timing the crux. |
| Fencing | Franka (faster) + epee on a rail cart; point/blade at T, cart moves | real time, bout 3 min | ✓ | ✓ | – | – | Arm sport with contact against an opponent: low-level joint control is natural (and parries need it). Too fast for direct play. |
| Baseball | Kinematic swing rig (a Franka cannot reach 30 m/s); swing plan / pitch | real time, 0.4-0.6 s flight | ✓ | – | (✓) batting at x100 | – | No low-level body: the rig is a deliberate abstraction (S3). Batting is one decision per pitch: direct play possible at x100. |
| Air hockey | KUKA iiwa + mallet; mallet x/y target (service path + IK) | real time | ✓ | (✓) | – | – | Low level (7 joint targets) is meaningful but adds little over table tennis; exhibition after the TT/tennis low-level tasks. Too fast for direct play. |
| Foosball | 4 rods; slide/angle targets (service speed/acc-limited path) | real time | ✓ | (✓) rod velocities | – | – | The rods already are near actuator level; "low level" = rod motor velocities/torques, small difference. Too fast for direct play. |
| Soccer (2v2, 5v5) | Booster T1 humanoid; walk velocity / kick (released walking policy) | real time | ✓ | (✓) late | (✓) team tactics at x50 | – | Low level = joint targets for a humanoid that must walk and kick: the hardest locomotion problem in the programme, expect 0 for a long time (see athletics first). Direct play could command walk/kick at a slow clock (tactics, not control). |
| Racing | 1/10 car; speed + steering set-points (servos) | real time, heats | ✓ | (✓) wheel torque + steering | (✓) x50 | – | Low level = drive torque and steering angle without the velocity servo: traction control becomes the agent's job. Direct driving at x50 is a classic VLA-style test (steer from images). |
| Drone | Quadrotor; velocity + yaw-rate set-points (onboard controller) | real time | ✓ | ✓ rotor thrusts | – | – | Low level = the 4 rotor thrusts (or body rates + collective): the agent writes the attitude controller; a well-known but genuine control/coding problem. Unstable at model-call rates, so no direct play. |
| Darts | release yaw/pitch/speed | turn | ✓ | – | ✓ | – | One decision per throw: direct play is natural and cheap. No arm in the loop, so no low level (a Franka throw would be a new sport variant). |
| Archery | elevation/azimuth/draw | turn | ✓ | – | ✓ | – | As darts (wind reading from images). |
| Bowling | board/angle/speed/rpm/axis | turn | ✓ | – | ✓ | – | As darts. |
| Golf (putting) | aim/speed | turn | ✓ | – | ✓ | – | As darts; green reading from images. |
| Disc golf | disc/speed/launch/nose/bank/spin/aim | turn | ✓ | – | ✓ | – | As darts. |
| Billiards | aim/speed/spin/cue position | turn | ✓ | – | ✓ | – | Program play computes geometry; direct play tests visual geometry by the model itself - the clearest no-code contrast. |
| Curling | speed/angle/spin | turn | ✓ | – | ✓ | – | As billiards (team strategy, house reading). |
| Basketball (shootout) | a declared shooting machine (D7, like Toyota's CUE; revision basketball-5); angle/speed/spin/yaw | turn | ✓ | (✓) | ✓ | (✓) | Low level = a real body's throw (joint targets, release timing) instead of the machine; the Panda arm was removed (2026-10-09): its hand reaches about 2 m/s, no spot is reachable with that (B9 HB7), so a low level needs a body that can throw 7-10 m/s. |
| Athletics (new) | humanoid; joint targets / torques | real time | – | ✓ only | – | – | §9: locomotion is low level by definition; no high-level version (a walking controller would make the result the controller's). |

Summary: low level first for the arm sports (table tennis, tennis, badminton, fencing) and the drone, then racing,
air hockey, foosball, basketball; direct play first for the turn-based sports (all seven) and table tennis (the hard
real-time case), then baseball batting, racing, tennis at x50-x100.

## 4. Naming and versioning

- **Task ids**: `<sport>-<kind>[-wallclock]-<play>[-<level>]-standard`, with `play` in {`program`, `direct`} and
  `level` in {`lowlevel`} (action level `low`); the high level is not written, so every existing id stays valid:
  `tabletennis-match-wallclock-program-standard` (P-high), `tabletennis-match-wallclock-program-lowlevel-standard`
  (P-low), `tabletennis-match-wallclock-direct-standard` (D-high), `billiards-direct-standard`,
  `athletics-sprint-wallclock-program-lowlevel-standard`. `-standard` stays last (PROTOCOL mode suffix,
  `task_conventions.py`).
- **task.toml**: `[metadata] play = "program" | "direct"` and `action_level = "high" | "low"` (absent = program, high;
  the tasks with switches always write both). `check-olympiad.py` checks the `play = "direct"` prompt with the "Play directly" section instead of "Write a
  program".
- **spec.py**: `EXPERIMENT` keeps the default (high, program) task of each format; `MODES = {kind: [(action_level,
  play), ...]}` declares the supported combinations (the first is the default). `check-olympiad.py` checks every
  declared combination (§12).
- **Protocol**: the wire protocol stays `eai-standard-realtime/1.0` (resp. `eai-standard-turn/1.0`) for every mode:
  a low-level `act` is a sport-defined request body, like every other sport's. The direct play mode adds a validity
  rule (per-response acts + tool audit); it is written into the sport rules as v1.1 (§11 below) and becomes
  `eai-standard-realtime/1.1` (an optional `play` profile in `/spec` and `task.toml`) when a second sport adopts it.
  `/spec` reports `action_level` and `decision_rule`.
- **Seasons**: which modes enter a season is **undecided** (the season and its task list are not decided yet).
  Results of different modes are never pooled.

## 5. Scoring, validity, oracle and nop per mode

| | P-high | P-low | D-high / D-low |
|---|---|---|---|
| Success | Win a valid match (single) / a valid match with a winner (duel), as today | Same | Same, and no audit violation |
| Technical failure | Simulator, replay, model-API (api_errors.py) or metering failure | Same; a command refused (400) or rejected never is one | Same; an audit violation is **not** one (it is a loss) |
| Replay | From the logged acts (high-level decisions re-planned) | From the logged low-level commands, incl. rejected ones (the rejection is recomputed) | As the action level |
| Oracle | Scripted camera-only player | The same perception and shot choice + its **own** arm model (FK from `/spec`, IK, swing), streamed as joint targets: the oracle demonstrates that the documented interface is sufficient | The P-high oracle in the oracle seat (exempt) |
| nop | Never moves: loses every point (0) | Same (tested: 0:5) | Same |
| Extra reward fields | – | `lowlevel_actions`, `rejected_acts` | `direct_play`, `nocode_tool_calls`, `nocode_violations`, `nocode_forfeit` |
| Score ladder (analysis only, all modes; single player) | `ladder_*` | same | same |

**Score ladder** (decided 2026-10-06; never changes the win rule). Per point and per incoming ball, how far the agent
got: **touched** (the blade touched the ball) -> **active** (touched with the blade moving at >= 0.2 m/s: a swing or a
push; a blade held still at the ready pose meets 2 of 5 serves at L1 seed 101 and 3 of 5 at seed 111, so "touched"
alone is not evidence of play) -> **over the net** (the first contact after the touch, net excepted, is beyond the
net: the opponent's half or the floor behind it) -> **landed in** (that contact is the opponent's half) ->
**returned** (a good return: also one own bounce, one paddle contact, no arm contact) -> **won** (the point). A point
reaches a rung if any of its balls did; lets are not points. `ladder.py` (shared simulator directory) computes it from
the match's own per-ball results; `replay.py` computes it again from the replayed match together with the blade speed
at every paddle contact (read only; the replay's hashes are unchanged) and the replay's ladder goes into `final.json`
`ladder` (with `per_point`); `grade.py` copies the counts into `reward.json` as `ladder_points`, `ladder_balls`,
`ladder_points_{touched,active,over_net,landed_in,returned,won}`, `ladder_points_won_{expedite,rally}`,
`ladder_balls_{touched,active,over_net,landed_in,returned}` (-1 when missing). **Won points by how** (decided
2026-10-06; the rules are unchanged): `points_won_expedite` counts the points the agent won by the expedite rule (13
good returns as the receiver, ITTF 2.15.4; the point's `how` carries "(expedite rule)"), `points_won_rally` every
other won point (the opponent missed or hit out); `per_point` has `expedite`. In single play the agent always
receives, so the rule favours it; reports list both. The duel has no per-ball results yet (its referee is separate):
no ladder; its `reward.json` reports `A_points_expedite` / `B_points_expedite` (points each seat won by the rule;
the receiver changes every point, so the rule is symmetric there).

## 6. Exploits and how each mode stops them

| Risk | Mode | Stop |
|---|---|---|
| Program loops over `game act` in "no-code" play | direct | Per-response rule (429) + audit (forfeit) |
| `game act $(python3 plan.py)`, `cd /app && game act ...`, pipes, here-docs | direct | Audit: literal arguments only |
| Starting a background program before `game start`, reading its output via `game`-like wrappers | direct | Audit covers the whole session (preparation included); only `game` and image opening are allowed at any time |
| Replacing `/usr/local/bin/game` | direct | Needs a non-`game` tool call: forfeit |
| Out-of-range or too-fast joint targets, NaN | low | 400 on arrival; the simulator's own check (which would end the match as a simulator error) is never reached (unit test) |
| Discontinuous jump between commands | low | Rejected at effect; the previous command goes on (unit test) |
| Parking the paddle as a wall | low | Legal block; still needs one own bounce and a landing on the opponent's half; the opponent's pressure rule decides |
| Hitting the table or net with the arm to stop the ball | low | Loses the point (robot-table contact, ITTF 2.10.1.9-11) |
| Faster swings than the high-level planner allows (100 % vs 90 % of the speed limits, no scatter) | low | Accepted: it is the agent's own control; the opponent's max-speed rule still caps the pressure |
| Stalling between points (the arm is not reset) | all | The serve comes 0.8 s after the review regardless |

## 7. Cost and time

Measured on the prototype (L1, seed 101):

| | Wall time | Calls | Model tokens |
|---|---|---|---|
| P-low oracle, x10 (local) | 253 s for the match (5:2, 7 points, 23.9 s of game time) | 1080 observe, 108 act, 462 wait | – |
| P-low oracle, x20 (Harbor) | 643 s of running world (5:1, 6 points) | 142 act (1 rejected) | – |
| D-high oracle, x50 (Harbor; the scripted player, not a model) | 1130 s of running world (5:2) | 97 act | – |
| P-low luna smoke (gpt-6-luna, Codex), x20 | 971 s session, 188 s of running world (lost 0:5) | 81 model calls; its program: 3141 observe, 59 act | 4.82 M input, 48 k output |

Estimates for direct play (table tennis single, a match of 7-13 points of about 3-4 s game time each, 2-3 incoming
balls per point; a model call with one 2 x 4 image takes 8-25 s, about 1.5k image tokens + context):

| | x20 | x50 | x100 |
|---|---|---|---|
| Wall time per incoming ball (0.7 s of flight) | 14 s | 35 s | 70 s |
| Model calls per ball (observe + act) | 1 (often too late) | 2-3 | 4-6 |
| Calls per match (incl. reviews and waiting) | 60-120 | 120-250 | 250-500 |
| Match wall time (game time 25-50 s + reviews) | 15-25 min | 30-55 min | 55-100 min |
| Input tokens per match (context 40-80k, cached) | 3-8 M | 6-15 M | 12-30 M |
| Output tokens per match | 30-80 k | 60-150 k | 120-300 k |

Turn-based sports in direct play cost what they cost today (one call per shot, 10-60 shots). Low-level program play
costs about what high-level program play costs: the model writes and fixes a program; the extra work is in the
program (IK, swing), not in model calls.

## 8. Implementation status (table tennis single player)

| Item | Status |
|---|---|
| Structure | One task source `scripts/olympiad/tabletennis/match/` and one simulator (`shared/tabletennis/sim/`: `arm_commands.py` (formerly `core.py`), `action_high.py`, `action_low.py`, `levels.py` on top of the unchanged `rcb_tt.py`, `rally.py`, `match.py`); `spec.py` declares `MODES["single"] = [(high, program), (low, program), (high, direct)]`; `generate.py` renders the three task directories. The duel has the same three combinations (below). |
| Duel | One task source `tabletennis/duel/` and its own world `duel.py` (two robots): its per-side command layer (`plan`/`plan_j` rows, `slew`, rest = return to ready for the high level, hold for the low level; `install()` with the same validation and continuity rejection as `core.py`) is the duel's form of the core; the high-level plugin stays in the plan workers + `trajectory()` (unchanged); `server_duel.py` switches on `RCB_ACTION_LEVEL` (low: `action_low.validate`, `install`, kinematics and limits in each seat's own frame) and `RCB_PLAY`; the grader audits each model seat (`think_calls_<seat>.jsonl`); a violating seat loses (both: nobody wins). The duel replays from joint targets, so the replay itself does not depend on the switch. |
| Duel unchanged | A duel recorded by main's code (Harbor, oracle vs oracle, x10, seed 1: 3:5, 2303 steps) replays with the new `duel.py` to the same state hash, targets hash, score and winner; driving main's `duel.py` and the new one with the same seeded high-level commands through a whole match (6:7, 2436 steps, arm resets at every point) gives identical state hashes every 100 steps (`test_highlevel_duel_command_layer_equals_main`, main's file kept in `tabletennis/testdata/duel_main.py`). |
| Existing tasks | `tabletennis-match-wallclock-program-standard`: `instruction.md` and `README.md` byte-identical; changed: `server.py`, `replay.py` (switch-aware), `game` (+ low-level `act` forms), `solution/oracle_play.py` (+4 lines: starts `oracle_low.py` at the low level), `task.toml` (+`play`, `action_level`, `category`, `timing_policy`), `docker-compose.yaml` (+`RCB_ACTION_LEVEL`, `RCB_PLAY`), `tests/grade.py` (+audit, only applied for direct play; `official_config` per play mode) and `tests/Dockerfile` (+`nocode_audit.py`); new files `core.py`, `action_*.py`, `levels.py`, `solution/arm.py`, `solution/oracle_low.py`, `tests/nocode_audit.py`. `tabletennis-duel-wallclock-program-standard`: only the four new shared simulator files (unused by the duel). |
| Behaviour unchanged | A match recorded by the original service (Harbor, oracle, L1 x50 seed 101) replays through the new core + high-level plugin to the same actions hash, state hash and score (unit test, fixture `tabletennis/testdata/`); a match recorded by the new service (Harbor, oracle, L1 x20: 6:4) replays through the original `Match.rt_act` path to the same hashes. |
| Low-level oracle | `solution/oracle_low.py` + `arm.py` (numpy FK from `/spec`, matching MuJoCo to 4e-7; DLS IK; Hermite swing). |
| Robot model in the seat | `robot_only.xml` + meshes in `/opt/robot/` and MuJoCo 3.3.7 (+ its pinned dependencies) in the low-level seat images (single and duel), generated by `tabletennis/robot_model.py` through `generate.py` (`seat_files` / `seat_docker` of the task spec); information boundary in the prompt and README (§2.7); unit tests `test_robot_model_*` (3). Harbor oracle with the new image: 5:3 (L1 x20 seed 101). |
| Score ladder | `shared/tabletennis/sim/ladder.py`; `final.json` `ladder` (from the replay, with `per_point` and the blade speed at each contact), `reward.json` `ladder_*`; unit tests `test_ladder_*` (2) and the low-level replay test. Single player only. |
| Harbor (L1, seed 101) | high/program oracle x20: 6:4, success 1; low/program oracle x20: 5:1, success 1 (1 rejected command); high/direct oracle x50: 5:1, success 1 (a first attempt was invalid because I regenerated the task directory while it ran; rerun clean); nop: 0 and valid for all three. |
| Unit tests | `scripts/olympiad/tabletennis/test_modes.py` 21/21 in the sim image (no_draw 2, mirror 2, session_end, world_state, seed_hidden, exploit 5, low-level replay, high-level equivalence 3, robot model 3, ladder 2). |
| Checks | `generate.py check` clean; `check-olympiad.py` 0 failures over 17 sports (rule M for the declared switches); `test_program_play.py` OK on all 38 tasks. |
| Luna smoke (v0.1, one run) | gpt-6-luna (Codex, OpenRouter), low level L1 x20: lost 0:5, valid; it wrote its own FK, a scipy IK and stereo ball tracking, touched one ball in five. A model can play low level; it is far from returning the ball. |
| Runner | Direct tasks: `run_single.sh` / `run_duel.sh` give Claude Code `permission_mode=dontAsk` and `allowed_tools=Bash(game:*),Read,TodoWrite` (everything else is denied by the harness; Codex has no such switch, the audit covers it). |
| Not done | Direct + low; other sports. |

## 9. Athletics (feasibility note; no build)

Candidate sport: **athletics, 100 m sprint and 110 m / 100 m hurdles**, low level only (P-low).

| Question | Assessment |
|---|---|
| Rulebook (S1) | World Athletics Competition and Technical Rules, Book C2.1 (TR 16 start, TR 17 lanes, TR 19 timing, TR 22 hurdles: 10 hurdles, a hurdle knocked over is not a fault unless trailed outside / deliberately). Deviations: no crouch start (standing start on a signal), no false-start rule needed (the world releases the robots), scaled distance for the robot (e.g. 20-40 m, hurdles scaled to leg length). |
| What it adds (S2) | The athletics category and the capability "locomotion from actuators", which no sport has today (soccer uses a released walking policy). |
| Body and simulator | Unitree H1 (MuJoCo Menagerie, already in `images/humanoidbench`: the repo's `humanoidbench-loco-walk/-hurdle` tasks) with PD position targets at 50 Hz, or the Gymnasium Humanoid (17 torque actuators). H1 is closer to a real robot and has the hurdle scene. CPU physics, about 10-20 kHz per core: no simulator lag at x1-x10. |
| Action / observation | Joint position targets (19) at 50 Hz with PD gains in `/spec`; proprioception (joint angles/velocities, IMU orientation and rates, foot contact forces). Cameras only for the lane and hurdles (positions are fixed and documented, so images are optional; a head camera keeps O1). |
| Formats | Single: a time trial with a success line (finish within T s without falling, falling = DNF) and a scripted "pacer" ghost per level; duel: two robots in adjacent lanes, no contact possible (lanes), winner = first across the line; dead heat to 1 ms -> last resort: the better split at half distance (R4). Reviews after each heat (3 heats per match). |
| Will models score? | Not without a practice simulator. A program that runs a humanoid from joint targets needs either a model-based controller (ZMP / capture-point / SLIP-style, from the documented model only, never tested before the race) or a learned policy. Neither is plausible blind; expect 0 (the meeting accepted that). With a **practice simulator** in the seat (MuJoCo + the robot and track MJCF, not the match world: there is no hidden state in a sprint, so this is no leak) an agent can tune or train before `game start`: CPU PPO/SAC on 4 cores for 2-3 hours gets a walking H1 (HumanoidBench reports h1-walk solved by RL in ~1-2 M steps for TD-MPC2/FastTD3, more for PPO), not a sprint; the time trial line should be set by a walking pace (e.g. 20 m in 30 s). |
| Oracle | No scripted oracle can run. Use a trained policy as the reference (e.g. FastTD3 on HumanoidBench h1-run / h1-hurdle, trained offline on a GPU, weights in `solution/`, deterministic inference on the CPU); the oracle then also bounds what a 2-3 hour agent could do. |
| Exploits (R6) | Falling forward over the line (finish = torso crosses while not fallen); diving / crawling (require both feet off the ground at least once per 1 s, i.e. running gait, or simply count crawling as DNF); knocking hurdles deliberately (TR 22.6: disqualification); leaving the lane (TR 17.3). |
| Cost | A heat of 20-40 m at 1-3 m/s is 10-40 s of game time; at x10 2-7 min per heat. The agent's program streams targets at 50 Hz over HTTP: 20 ms per step at x1 is too tight, x5-x10 fine (official slowdowns [5, 10]). |
| "Done to death" | Yes as RL; no as a coding-agent task: the agent must build the training or control loop itself from documentation, inside a time budget, and then race closed loop through an HTTP interface. Include it as an exhibition with the expectation of zeros at first. |

Recommendation: build after the arm low-level tasks; start with the H1 hurdle scene from the repo's HumanoidBench
image, a practice simulator in the seat, a walking-pace success line and an RL oracle. Gymnasium Walker2d / Swimmer
are too toy-like for "athletics" (planar, no rulebook) and are not proposed.

## 10. Open decisions

1. **Practice simulators for low level.** The table-tennis low-level oracle shows the documented interface suffices
   without a simulator copy; for locomotion (athletics, soccer low level) a practice simulator in the seat is
   probably necessary. Allow it per sport (only when the sport has no hidden state the copy would reveal)?
2. ~~Direct-play harness~~ **Decided 2026-10-06**: the main configuration is **direct-native** - the same native
   coding-agent harness as program play with code execution disabled (§17), so that the mode comparison changes one
   variable; higher token use is accepted. **direct-player** (option A, §15) is the second, documented configuration
   (one simple loop for all models, cheaper). E3 is void (harness fault), E4 counts; direct-player keeps the Claude Code
   print-mode backend.
3. ~~Audit scope~~ **Decided 2026-10-06**: the audit window starts at the first `game start`; read-only
   self-inspection before it (listing tools, reading the instruction or spec) is allowed; running code after it, or
   starting a program at any time, is a violation, and a violation is a loss.
4. **Direct-play clock.** x50 default and [20, 50, 100] official; or a "decision-point freeze" variant (the world
   pauses at fixed decision points, as the old interactive tasks did) for sports where even x100 is unplayable?
5. **Low-level continuity rule.** Reject a discontinuous command (prototype, like libfranka) or blend it in with a
   rate limiter (friendlier, but a small planner)? Clear the command at the end of a point and hold (prototype) or
   return the arm to the ready pose automatically?
6. **Seasons.** Which modes enter a season is undecided; the season and its task list are not decided yet.
7. **Naming.** `-direct` / `-program-lowlevel` with the high level implicit (proposed), or always spell both axes
   (`-program-high-standard`, renaming the Season 1 tasks)?
8. **Image budget in direct play.** Keep the 2 x 4 tile image (about 1.5k tokens) or offer a single-camera, two-frame
   view to cut tokens and latency?
9. **Warm-up practice balls for the low level (rule decided 2026-10-06, not added).** Add a warm-up (practice serves
   before `game start`, not scored) only if, at the same slowdown as the high level, no tested model makes a single
   active touch (ladder rung "active", blade >= 0.2 m/s at contact) in a whole low-level match. Table tennis (§14.1):
   both models touch actively in every match, Opus wins: no warm-up. (A plain touch is not the criterion: a held blade
   meets serves.)
10. **Expedite rule at the low level.** Opus' low-level wins are all expedite-rule points (13 good central returns per
    rally). Keep the rule as it is (it bounds rallies and is ITTF), or raise the count / require placement for the
    low level? **Decided 2026-10-06: keep the rule** (ITTF; same for both levels, single and duel) **and report won
    points by how**: expedite-rule points and rally points separately (§5; single `ladder_points_won_{expedite,rally}`,
    duel `A/B_points_expedite`).

## 11. Changes to the sport rules (proposed v1.1)

- §4 A1: "Actions are high-level targets ..." becomes "at the sport's action level: high-level targets executed by a
  planner, or (action level `low`) actuator commands at the control rate, executed by the robot's own joint
  controller under its physical limits; never state setting".
- §4 A2 applies to both levels; a low-level task documents its robot model (kinematic chain, limits, controller) in
  `/spec`.
- §6 T3: direct-play slowdowns come from `common.DIRECT_SLOWDOWNS`.
- §7 new I5: a direct-play task runs with `RCB_ACT_RULE=per_response` and its grader audits every tool call
  (`think_calls.jsonl`) with the allow-list of §2.5; a violation is a loss, not a technical failure.
- §9 C2: task names may carry `-direct` and `-lowlevel` (§4 above); `task.toml` carries `play` and `action_level`.
- §10: a mode's evidence: its oracle wins the default configuration, nop scores 0, the unit tests cover the mode's
  own rules (low level: refusals, rejections, replay of low-level commands; direct: the audit).

## 12. How to add a mode to a sport

1. **Split the simulator into a core and an action plugin.** The core is the sport's simulator with one command layer
   that is the only source of actuator commands (table tennis: `core.py`, joint targets per 10 ms step). Move the
   high-level interface (IK, planners, controllers, aiming) into `action_high.py`; it must turn every high-level act
   into core commands. Write `action_low.py` for the actuator interface (validation with HTTP 400, `/spec` model of the
   body: kinematics, limits, controller). Both export `LEVEL`, `REST`, `ACT_CALL`, `OBSERVE_CALL`, `validate`,
   `apply`, `reply`, `state`, `spec`; `levels.plugin(level)` selects one.
2. **Prove the high level is unchanged.** Record a match with the old service (Harbor oracle run), keep its acts and
   replay result in `scripts/olympiad/<sport>/testdata/`, and add a unit test that the core + `action_high` reproduces
   its actions hash, final state hash and score; generated `instruction.md` and `README.md` of the existing tasks must
   stay byte-identical.
3. **Make the service and the replay switch-aware**: read `RCB_ACTION_LEVEL` and `RCB_PLAY`, write `action_level`,
   `play` and `rejected_acts` into `final.json` and the episode's start record (the replay builds the core with the
   recorded level).
4. **Template the task source**: `task.toml` (`name`, `description`, `keywords`, `oracle_evidence`, `revision`,
   `parameters`, `play`, `action_level`, agent `timeout_sec`) and `environment/docker-compose.yaml` (`SLOWDOWN` default,
   `WALLCLOCK_RULE_WAIT_S`, `RCB_ACTION_LEVEL`, `RCB_PLAY`, `RCB_ACT_RULE`) get `@@KEY@@` placeholders
   (`modes.fill()` gives the switch values, the spec the sport's texts). The grader imports `nocode_audit` and applies
   it when `play` = `direct`; `tests/Dockerfile` copies `nocode_audit.py`.
5. **Prompt**: the sport's prompt function takes `(level, play)`: the level chooses the command table, the Action
   section and the body section; `program_prompt.single(play=...)` supplies the shared middle section.
6. **Declare** `MODES = {kind: [(level, play), ...]}` in `spec.py` (default first) and build `TASKS` from it
   (`modes.task_name`, `modes.fill`); give the low level an oracle that uses only the documented interface (its own
   IK/controller) and keep the high-level oracle for direct play (oracle seats are exempt from the act rule and the
   audit).
7. **Check**: `generate.py check`; `check-olympiad.py` (rule M: every declared combination generated with matching
   `task.toml` switches, compose switches, prompt section and shared audit; non-default combinations get all task
   checks); the sport's unit tests (`*no_draw*`, `*mirror*`, `*session_end*`, `*world_state*`, `*seed_hidden*`,
   `*exploit*`, plus the high-level equivalence test); Harbor oracle and nop for every combination.

## 14. Model test matches (2026-10-06, L1 / duel seed 1)

Opus 5.5 = Claude Code 2.1.284 on the slot-2 subscription; luna = gpt-6-luna on Codex 0.158.0 via OpenRouter. All
matches valid with deterministic replays.

| # | Task (switches) | Players | Slowdown | Score | Contacts / good returns | Model calls | Input / output tokens | Session | Audit |
|---|---|---|---|---|---|---|---|---|---|
| M1 | single, low / program | luna | x20 | 0:5 | 2 / 0 | 57 | 3.39 M / 26 k | 8.6 min | - |
| M2 | single, low / program | Opus | x20 | 0:5 | 6 / 3 | 53 | 4.50 M / 50 k | 18.5 min | - |
| M3 | single, high / direct | luna | x50 | 0:5 | 2 / 0 | 67 | 2.96 M / 17 k | 10.8 min | 66 tool calls, 0 violations (after the fix below) |
| M4 | single, high / direct | Opus | x50 | 2:5 | 4 / 2 (two winners: too wide) | 72 | 5.94 M / 23 k | 16.5 min | 83 tool calls, 0 violations |
| M5 | duel, low / program | luna (A) vs Opus (B) | x20 | 1:5, Opus wins | 6 hits in the match | A 90, B 38 | A 5.39 M / 41 k, B 2.46 M / 35 k | 17 min | - |
| M6 | duel, high / direct | luna (A) vs Opus (B) | x50 | 6:7, Opus wins | 5 hits in the match | A 110, B 81 | A 7.29 M / 39 k, B 6.15 M / 27 k | 26 min | A: 1 violation (strict: forfeit), B: 0 |

What the models did: at the low level both wrote their own player programs from the documented kinematic chain (FK,
IK, stereo ball tracking; luna 221 lines, Opus 363 lines + helpers) and streamed `traj`/`slew` commands; Opus returned
the ball three times (12 of its commands were rejected as discontinuous). In direct play both only typed `game`
commands and opened images; with 1-2 looks per ball at x50 they mostly missed; Opus won two points with wide returns.
In the duels most points came from errors (serve receptions missed, shots into one's own half, arm contacts).

Audit findings:
- **Auditor misfire (fixed)**: Codex 0.158 with GPT-6 has no direct shell tool; every tool use is a JavaScript `exec`
  cell calling `tools.exec_command` / `tools.view_image`. The first audit flagged all 66 calls of M3 (and 108 of luna's
  109 in M6) as violations. The audit now accepts exactly one literal tool call per cell (a literal `game` command,
  opening a camera image, or polling the running command with empty `write_stdin`) followed only by printing its result;
  everything else in a cell is code and still a violation (unit-tested: loops, two calls, other commands). Escaped
  characters inside double quotes are literal (luna's `\u0027` in a review text). M3 and M6 were re-graded from their
  artifacts: M3 0 violations; M6 luna 1.
- **Remaining M6 violation (policy question)**: luna's first call, before `game start`, was a code cell listing the
  harness's tools (`ALL_TOOLS.filter(...)`): harness introspection, no game action. Under the strict whole-session rule it
  forfeits luna's match (Opus had won 7:6 anyway). Open decision 3 (audit scope) decides whether pre-start introspection
  should be allowed.
- No model tried to run a program in direct play; Claude Code's `dontAsk` + allow-list denied nothing (Opus used only
  `Bash(game ...)` and `Read` of the views), and the per-response rule refused one act per Opus match.

### 14.1 High vs low with the robot model (2026-10-06, single, scripted L1, x20 for both levels, program play)

Opus 5.5 = Claude Code 2.1.284 on the slot-2 subscription (`OPUS_ACCOUNT=2`); luna = gpt-6-luna on Codex 0.158.0 via
OpenRouter. Low = `tabletennis-match-wallclock-program-lowlevel-standard` with `robot_only.xml` + MuJoCo in the seat
(commit 59d229c2); high = `tabletennis-match-wallclock-program-standard` (unchanged seat). Ladder columns, per point:
touched / active / over the net / landed in / good return / won, of the points played; per ball the same without
"won". Active touches come from the replay (the matches were played with the first ladder version and re-replayed
offline with the final `replay.py`; replay state and target hashes equal the recorded ones for every match).

| Level | Player | Seed | Score | Points: T / A / N / L / R / W | Balls: T / A / N / L / R | Used the model? | Rejected | Acts | Model calls | Input / output tokens | Session |
|---|---|---|---|---|---|---|---|---|---|---|---|
| low | Opus | 101 | **5:2 won** (5 expedite + 0 rally) | 7: 6 / 6 / 5 / 5 / 5 / 5 | 67: 66 / 66 / 65 / 65 / 65 | yes: FK + Jacobian IK on `robot_only.xml`; simulates its own arm with the documented PD law (`mj_step`) to plan the swing | 0 | 727 | 67 | 5.71 M / 41 k | 48.5 min |
| low | Opus | 111 | **6:4 won** (5 + 1) | 10: 10 / 10 / 7 / 7 / 6 / 6 | 70: 70 / 70 / 67 / 67 / 66 | yes: FK/IK + an arm simulator class on `robot_only.xml` | 15 | 1367 | 69 | 6.34 M / 51 k | 50.4 min |
| low | luna | 101 | 0:5 | 5: 3 / 1 / 0 / 0 / 0 / 0 | 5: 3 / 1 / 0 / 0 / 0 | yes: FK + Jacobian on `robot_only.xml`, SciPy least-squares IK | 0 | 31 | 107 | 10.18 M / 103 k | 25.9 min |
| low | luna | 111 | 0:5 | 5: 4 / 3 / 0 / 0 / 0 / 0 | 5: 4 / 3 / 0 / 0 / 0 | yes: FK on `robot_only.xml` | 0 | 47 | 69 | 4.50 M / 39 k | 12.0 min |
| high | Opus | 101 | **5:3 won** (1 + 4) | 8: 8 / 8 / 5 / 5 / 5 / 5 | 20: 20 / 20 / 17 / 17 / 17 | (no model in the seat) | 0 | 510 | 42 | 2.90 M / 39 k | 22.8 min |
| high | Opus | 111 | **7:5 won** (0 + 7) | 12: 12 / 12 / 10 / 10 / 10 / 7 | 28: 27 / 27 / 23 / 23 / 23 | - | 0 | 346 | 59 | 4.60 M / 36 k | 25.4 min |
| high | luna | 101 | 0:5 | 5: 4 / 2 / 0 / 0 / 0 / 0 | 5: 4 / 2 / 0 / 0 / 0 | - | 0 | 14 | 80 | 3.72 M / 44 k | 13.2 min |
| high | luna | 111 | 1:5 (0 + 1) | 6: 3 / 2 / 1 / 1 / 1 / 1 | 6: 3 / 2 / 1 / 1 / 1 | - | 0 | 9 | 59 | 2.57 M / 37 k | 14.1 min |
| low | oracle | 101 | 5:3 won (0 + 5) | 8: 8 / 8 / 5 / 5 / 5 / 5 | 10: 10 / 10 / 7 / 7 / 7 | own numpy arm model | 2 | 131 | - | - | 8.5 min |
| low | oracle | 111 | 6:7 (0 + 6) | 13: 13 / 13 / 9 / 9 / 9 / 6 | 20: 20 / 20 / 13 / 13 / 13 | own numpy arm model (final commit; ladder straight from the Harbor run) | - | - | - | - | - |
| low/high | nop (held blade) | 101 / 111 | 0:5 | touched 2 / 3, active 0 | | | | | | | |

Not in the table (invalid trials, rerun): high / Opus / 101, first attempt, played 5:7 but invalid by the model-API
rule (6 transient errors, HTTP 529 overloaded at the provider: points 12: 11 / 11 / 8 / 8 / 8 / 5); low / Opus / 111,
first attempt, invalid because my offline analysis script created the trial's artifact directory as root while the
match ran, so Harbor could not download the sim's evidence (player log: 0:5; not graded). Both reruns are in the table.

Findings:

- **The model file is used and helps.** All four low-level sessions loaded `robot_only.xml` with MuJoCo (no session
  wrote its own FK from the chain this time; no `pip`, no look for simulator files). Opus also simulated its own arm
  with the documented controller to predict the tracking lag. Opus went from 0:5 with 3 good returns (M2, chain only,
  no model file) to two wins with 65 and 66 good returns.
- **Same slowdown, same verdict per model.** Opus won 2 of 2 at both levels; luna lost 2 of 2 at both levels (one
  point at the high level). luna's problem is upstream of the action level: it gets 0-1 balls over the net at either
  level.
- **How the low-level wins come.** Opus' low-level swing returns almost every ball to the centre (|x| <= 0.16 m), so
  the opponent never misses; it wins points by the expedite rule (13 good returns as the receiver, ITTF 2.15.4: 5 of 5
  points in seed 101, 5 of 6 in seed 111). Its high-level wins come from wide placements (the opponent's reach rule).
  Won points by how (expedite + rally, counted from the points' `how`): Opus low 5 + 0 and 5 + 1, Opus high 1 + 4
  and 0 + 7; the oracle wins all its points by the rally. The low-level matches are therefore long (67-70 balls,
  about 50 min) and the low level rewards consistency rather than placement. Not an exploit (it is the rule that
  bounds rallies): the rule stays and reports list both kinds of points (§10, item 10).
- **The ladder separates the players** where the score does not: luna's losses are "touched, active, never over the
  net", the held blade (nop) is "touched, never active". Plain "touched" is not evidence of play (2-3 of 5 serves
  meet a blade held at the ready pose), hence the active rung.
- **Warm-up practice balls: not needed** (§10, item 9): every low-level match had active touches (both models), and
  Opus won both. Revisit only if a sport's low level shows a whole match without an active touch for every model.

## 13. Options for a dedicated no-code harness (the discussion; A was chosen, §15)

Direct play currently runs the coding agents (Claude Code, Codex) with their tools restricted. Kangrui suggested a
different harness for no-code play. Three concrete options:

| | A. Minimal loop as a Harbor agent | B. `game` tools only, via MCP, inside the existing harnesses | C. Minimal loop outside Harbor (in the sidecar) |
|---|---|---|---|
| What | A small Python agent (`players/harnesses/direct_loop.py`) in the seat: each turn it calls `observe`, sends the prompt + the last images + the running notes to the model API through the metering proxy, parses exactly one action (a JSON tool call `act`/`wait`/`ready`) and posts it; no shell, no files. One loop for all providers (Anthropic Messages, OpenAI Responses / OpenRouter). | A `game` MCP server (observe/act/wait/ready/spec as tools, images returned as image content) in the sim sidecar; Claude Code `--tools` reduced to the MCP tools only (no Bash/Read), Codex with the shell and patch tools disabled and the MCP server configured. | The sidecar itself runs the loop (model call -> action -> world), so no seat container at all; the model only ever sees what the loop sends. |
| No-code guarantee | Complete by construction (the model has no tools but the actions) | Strong for Claude Code; Codex can disable shell only partly (version-dependent), so the audit is still needed | Complete |
| Comparability to program play | Different harness (no coding-agent scaffolding, no compaction / planning tools) - a model-vs-model comparison of perception + decision, close to a VLA evaluation | Same harnesses as program play (same system prompts and context handling), only the tools differ | As A |
| Latency per action | Lowest: one API call per act, image inline, no tool round trips | One model turn per tool call (observe, then act): ~2x A | As A, minus HTTP to the seat |
| Context / tokens | Fully controlled (we choose how many past images and notes to keep): predictable cost | Harness-managed context, grows with every image (compaction rules of each harness) | Controlled |
| Work | ~300 lines + per-provider adapters; must reimplement retries, the API-error rule, thinking settings; Harbor seat works as today | MCP server ~200 lines; harness flags; Harbor/harness version pinning | Loop as A + sidecar changes; leaves Harbor's agent model (no seat), so not a Harbor agent run (I4 needs a rule change) |
| Risks | We design the agent: results depend on our prompt/loop design (must be frozen per season and published) | Codex tool disabling and MCP image support vary by version | Same as A, and less isolation evidence (the sidecar both judges and plays) |

Recommendation for the discussion: **A** for the no-code track (clean guarantee, controlled cost, simple to reason
about; publish the loop as part of the benchmark), and keep **B**'s MCP server as the way to run the commercial coding
agents in no-code mode when we want the same harness as program play. C only if Harbor seats become a bottleneck.

## 15. direct-player: the second no-code configuration (option A)

**direct-player** (`players/harnesses/direct_player.py`, the Harbor agent, and `players/harnesses/direct_loop.py`, the loop it
runs in the seat container; standard library only) runs every model in the same simple loop. Use it with
`DIRECT_HARNESS=player` in `run_single.sh` / `run_duel.sh` (the default is direct-native, §17).

| | `direct-player/1.2` (all of it is in `direct_loop.py`, versioned by `VERSION`, logged on the first line) |
|---|---|
| Turn | observe (HTTP, no model call) -> one model call with the instruction, the recent turns and the newest image -> exactly one action -> send it -> next turn. The world keeps running throughout. |
| Action | one call of the single tool `game` whose argument is one `game` command line (`game act ...`, `game wait DT`, `game ready "WHY" "PLAN"`, `game start`, `game spec`, `game status`; the task's interface); parsed like a shell line (quotes) but never run by a shell. A reply without exactly one valid call is no action that turn (`malformed`, logged, the next turn says why). |
| Context policy | system = the task instruction + the loop's fixed "How you play" section (after it: v1.0 put a preamble first, and that preamble with the duel instruction tripped Opus's safety classifier on every call); the last 6 turns as text (observation <= 800 chars, the command, the service's reply <= 1500 chars); the model's newest note (its reply text, <= 1500 chars); the current observation (text <= 6000 chars) and **one image: the newest observation, the current turn only** (the service's PNG as is). |
| Model settings | output <= 4096 tokens; Claude thinking 2048 tokens; OpenAI-style reasoning effort medium. |
| Retries | a failed model call is retried after 2, 5, 10, 20 s, then the turn passes without an action (the proxy's API-error rule judges validity as for every agent). A Claude Code call is killed after 120 s (v1.2: a 300 s stall forfeited a duel review in v1.1) and Claude Code's own retries are off. |
| Backends | OpenAI-style (luna via OpenRouter, Astra via the Azure route): the Responses API through the seat's metering proxy. Claude on the subscription: the subscription token serves Claude Code only, so the loop calls **Claude Code 2.1.284 in print mode as a bare model endpoint** - no built-in tools (`--tools ""`), only a stub `game` MCP tool (`direct_game_mcp.py`, it does nothing), `--max-turns 1`: one model call per turn, through the same proxy. |
| Metering | every model call goes through the seat's `think_proxy.py`: tokens, think time and the API-error validity rule exactly as for the coding agents; the `game` tool call carries `game act`, so the per-response act rule counts it. |
| No-code guarantee | structural: the model has no shell, no files and no tool but `game`; the verifier's audit stays as a backstop (it allows `game` / `mcp__game__game` calls whose argument is a `game` line). |

## 16. direct-player vs the coding agents in direct play (2026-10-06, L1 / duel seed 1, x50)

All matches valid with deterministic replays; 0 audit violations with direct-player (by construction).
"Calls" = model calls through the metering proxy; latency = median duration of one model call.

| Match | Harness | Score | Contacts / good returns | Acts (per ball) | Calls | Input / output tokens | Latency | Session |
|---|---|---|---|---|---|---|---|---|
| single luna (M3) | Codex 0.158 | 0:5 | 2 / 0 | 11 (2.2) | 67 | 2.96 M / 17 k | 4.0 s | 650 s |
| single luna (E1) | direct-player 1.1 | 0:5 | 1 / 0 | 11 (2.2) | 41 | 0.31 M / 14 k | 3.5 s | 489 s |
| single Opus (M4) | Claude Code 2.1.284 | 2:5 | 4 / 2 | 16 (2.3) | 72 | 5.94 M / 23 k | 1.9 s | 991 s |
| single Opus (E2) | direct-player 1.1 | 0:5 | 0 / 0 | 14 (2.8) | 34 | 0.38 M / 14 k | 5.8 s | 583 s |
| duel luna vs Opus (M6) | Codex / Claude Code | 6:7, Opus | 5 hits | A 11, B 15 | A 110, B 81 | A 7.29 M, B 6.15 M | A 5.0 s, B 3.1 s | 1545 s |
| duel luna vs Opus (E3) | direct-player 1.1 | 6:6, luna by review forfeit | 2 hits | A 7, B 16 | A 79, B 68 | A 0.64 M, B 0.77 M | A 2.6 s, B 4.5 s | 1564 s |
| duel luna vs Opus (E4) | direct-player 1.2 | 2:5, Opus | 2 hits | A 6, B 11 | A 49, B 46 | A 0.39 M, B 0.51 M | A 2.9 s, B 3.6 s | 853 s |

Findings:
- **Tokens drop about 10x** (0.3-0.8 M instead of 3-7 M input tokens per player and match): the coding agents resend a
  growing history with every image; direct-player keeps 6 text turns and one image.
- **Play is no better yet**: actions per ball stay at 2-3 (the clock, not the harness, limits looks per ball); Opus
  scored 2 points in Claude Code (M4) and none in direct-player (E2), where it sees one image per turn and no past
  images - a context-policy question for the next version (e.g. keep the previous image), not decided here.
- **Harness bugs found and fixed during the runs** (each a new version): v1.0's preamble before the duel instruction
  tripped Opus's safety classifier on every call (the duel never started for Opus; that run was stopped); Claude
  sometimes names the MCP tool `game__game` (accepted as the game tool; the audit backstop had flagged it); in E3 a
  Claude Code process stalled for ~300 s during a review and Opus forfeited at 6:6 (v1.2: 120 s kill and retry); E4 is
  the clean rerun.
- The single direct oracle is unaffected (5:2, success 1) and nop scores 0.

## 17. direct-native: the main no-code configuration (decided 2026-10-06)

`direct-native/1.2` (`players/harnesses/direct_native.py`; `run_single.sh` / `run_duel.sh` default, `DIRECT_HARNESS=native`):
the same harnesses and versions as program play (Claude Code 2.1.284, Codex 0.158.0), with code execution disabled and
the game reachable only through the **`game` MCP tool** (`players/nocode/game_mcp.py`, in every seat image as
`/usr/local/bin/game-mcp`; one argument, a `game` command line, parsed without a shell; `game observe` returns the
camera image as MCP image content). The adapters switch it on with `direct_native=true`.

| | Claude Code | Codex (GPT-6 models) |
|---|---|---|
| Harness flags | all built-in tools removed (`--disallowedTools`, list in `direct_native.py`) and anything not allowed denied (`--permission-mode dontAsk --allowedTools mcp__game__game`) | `shell_tool` and `unified_exec` off (no process can be started); the game MCP tool auto-approved. Codex 0.158 still runs every tool call of GPT-6 as a JavaScript cell (code mode cannot be switched off without losing all tools) and keeps `apply_patch` (no switch). |
| Gateway (1.1) | passes through (`/v1/messages` is not gated) | **direct-gate** (`players/nocode/direct_gate.py`) checks every response before Codex sees it (below) |
| Guarantee | **structural**: the model can only call the game tool | **structural since 1.1**: only the canonical cell (or Codex's `wait` on a running cell) is ever delivered to Codex; every other tool call is replaced by a stub cell that prints a refusal, so nothing else executes |
| Audit | backstop (allows `mcp__game__game` calls with a `game` line) | backstop (the code-cell rule of 1.0 stays) |

**direct-gate** (direct tasks only; program tasks run the same compose command and proxy as before):
- Topology: seat -> gate (the port the seat is given, 18090 / 18091) -> `think_proxy.py` (unchanged and byte-identical in
  every task; in direct tasks it listens on 18190 / 18191 with `THINK_CLIENT_HOST=localhost`, so only the gate can
  reach it - a direct call from the seat gets 403) -> provider. Metering, usage and the API-error rule are unchanged;
  think_proxy records the response as the model sent it (the audit sees attempts), the gate rewrites what Codex gets.
  Single: the sim service's command in the compose file (per play mode); duel: `launch.py --direct-gate` starts one
  gate per seat with the seat-host check.
- For OpenAI Responses calls the gate buffers the whole reply (stream or JSON; a cell split across chunks is judged
  only when complete) and checks every tool call. The canonical cell, shown in the direct instruction:

  ```
  const r = await tools.mcp__game__game({command: "game observe"});
  for (const c of r.content) { if (c.type === "image") image(c); else text(c.text); }
  ```

  with one literal `game` command line as the string (whitespace, semicolons and the two identifiers free). Also
  allowed: Codex's `wait` (only `cell_id`, `yield_time_ms`, `max_tokens`) and a direct call of the game tool with a
  `game` line. Blocked: everything else - a second call, loops, arithmetic, `Math`, `atob` or any other expression,
  `apply_patch`, other tools, extra statements.
- A blocked call is replaced (same call id) by a stub that prints one fixed text (direct-gate/1.1, direct-native/1.2;
  no detail of what was detected, no echo of code; the canonical cell is only in the instruction): `Blocked: writing or
  running code is not allowed in direct play. Use only the game tool exactly as shown in your instructions. Do not retry
  code.` Codex runs the stub, the model reads it and continues. Pre-start tool listing stays blocked (decided). Blocks are logged in `/data/direct_gate[_<seat>].jsonl` and reported as
  `gate_blocks` (`A_`/`B_gate_blocks`).
- **Decided 2026-10-06**: a block is **not** a violation and not a loss - nothing ran; repeated
  blocks only cost the model time. The post-hoc audit stays the backstop and judges what reached the agent: it skips
  the item ids in the gate's log (`gate_blocked_ids`).
- Tested: `scripts/olympiad/test_direct_gate.py` (canonical passes; loop, two calls, `Math`, `atob`, `apply_patch`,
  other tools, padding blocked; SSE rewritten consistently; split mid-cell; the instruction's cell passes) and a live
  red-team run of Codex 0.158 + luna behind the real gate and proxy: asked to loop over the game tool, its loops and
  tool-listing cells were all replaced (12 blocks), the canonical cell ran, the session went on, and a direct request
  to the meter's port was refused (403).

Harbor runs with direct-native/1.1 (L1, seed 101 / duel seed 1, x50):

| Run | Result | Gate | Audit | Notes |
|---|---|---|---|---|
| luna single (G1) | 0:5, valid, deterministic; 1 contact; 11 acts; 54 calls, 2.41 M / 16 k tokens | 0 blocks | 0 violations | luna wrote the instruction's cell every time |
| luna single, red team (R2): the instruction without the cell | 0:0 (never started), valid loss | 12 blocks (all tool-listing cells) | 0 violations after the skip (the first grading flagged 2 cells that only printed the refusal text: the inspection rule now ignores string literals) | every cell Codex executed was a refusal stub; the model gave up after 13 calls |
| duel oracle vs oracle (G0) | 3:5, valid, deterministic | 0 / 0 | - | `launch.py --direct-gate` topology works for both seats |
| luna single (G2), direct-native/1.2 | 0:5, valid, deterministic; 8 acts; 49 calls, 2.21 M tokens | 0 blocks | 0 violations | the short refusal changes nothing when the instruction is followed |
| red team again (R3), 1.2: instruction without the cell | 0:0 (never started), valid loss, 110 s | 2 blocks (a tool listing, a guessed `tools.mcp__game__spec`) | 0 violations | with the short fixed text luna stopped after 3 calls instead of 13; both executed cells were the stub; without the cell in the instruction it never found the game |

When to use which:
- **direct-native** for results: it isolates the variable "may write and run code" (program vs direct play run the
  same harness), at the coding agents' token cost.
- **direct-player** for cheap pilots and for comparing models in an identical loop (about a tenth of the tokens), or
  when a harness cannot be restricted.

## 18. Three configurations of direct play compared (2026-10-06, L1 / duel seed 1, x50)

allowlist = the first runs (Claude Code / Codex with `game` via the shell + image reading, M-runs); native =
direct-native/1.0 (§17, N-runs); player = direct-player/1.2 (§15; E1/E2 ran 1.1, which differs only in the Claude call
timeout). All valid with deterministic replays. Violations as re-scored with the current audit (in-run verdicts of
M3, M6, N1 and N3 were auditor misfires on Codex code cells, fixed and unit-tested).

| Match | Config | Score | Contacts / good returns | Acts (per ball) | Calls | Input / output tokens | Latency | Violations |
|---|---|---|---|---|---|---|---|---|
| single luna | allowlist (M3) | 0:5 | 2 / 0 | 11 (2.2) | 67 | 2.96 M / 17 k | 4.0 s | 0 |
| | native (N1) | 0:5 | 0 / 0 | 13 (2.6) | 52 | 3.60 M / 23 k | 4.3 s | 0 |
| | player (E1) | 0:5 | 1 / 0 | 11 (2.2) | 41 | 0.31 M / 14 k | 3.5 s | 0 |
| single Opus | allowlist (M4) | 2:5 | 4 / 2 | 16 (2.3) | 72 | 5.94 M / 23 k | 1.9 s | 0 |
| | native (N2) | 0:5 | 0 / 0 | 16 (3.2) | 41 | 1.72 M / 17 k | 3.4 s | 0 |
| | player (E2) | 0:5 | 0 / 0 | 14 (2.8) | 34 | 0.38 M / 14 k | 5.8 s | 0 |
| duel luna (A) vs Opus (B) | allowlist (M6) | 6:7, Opus | 5 hits | A 11, B 15 | A 110, B 81 | A 7.29 M, B 6.15 M | A 5.0 s, B 3.1 s | 0 / 0 |
| | native (N3) | 6:7, Opus | 3 hits | A 7, B 18 | A 104, B 74 | A 7.67 M, B 4.23 M | A 4.6 s, B 2.4 s | 0 / 0 |
| | player (E4) | 2:5, Opus | 2 hits | A 6, B 11 | A 49, B 46 | A 0.39 M, B 0.51 M | A 2.9 s, B 3.6 s | 0 / 0 |

Reading: the configuration changes cost far more than play. Native and allowlist cost about the same (the harness
resends its history with every image; Opus's native run is cheaper because each observation is one tool call with the
image inside it instead of two); player costs about a tenth. At x50 every configuration gets 2-3 actions per ball and
no model returns the ball reliably; the duels are decided by errors (Opus won all three). One match per cell is not
enough to rank the configurations by play.

