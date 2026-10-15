# skilleval/ — 两档的技能评测共享库

两档的定义见 [docs/framework_zh.md §6.3](../docs/framework_zh.md) 和第 9 节的机器人与模式规则：

- **高层 Agent as Athlete**：平台提供这项运动的技能 API（例：乒乓 `hit`、击剑 `strike`、hello 的 `aim`），选手做竞技决策。
- **底层 Agent as Robotics Engineer**：选手拿到机器人模型、带真实限位的关节接口和一份公开的工具包，**自己实现同一个技能 API**，
  放在 `/app/output/skill/` 下的固定路径；比赛流程不变；赛后平台把最终实现放进隔离容器，跑 API 测试（隐藏请求）和比赛测试
  （我们固定的参考策略装在选手的实现上）。

名字的意思：这里是**技能评测的机制**（合同基类、工具包清单、隐藏请求、沙箱、评测器）；技能 API 本身（合同、平台实现、钩子）留在各项运动里。`sport.toml` 里的节名仍是 `[modes.high]` / `[modes.low]`，它描述的是两档。

`skilleval/` 是框架定义、各项运动共用的库，和 `bodies/` 一样**不是框架的第五部分**：核心不知道它，运动和工具用它。它不出现任何
运动名（`ci_checks/check-skilleval.py`）。怎样给一项运动加两档，见 [CONTRIBUTING.md「如何给一项运动加两档」](../CONTRIBUTING.md)；
最小的完整例子是 [`sports/hello`](../sports/hello/README.md)。

*English summary: `skilleval/` is the shared library of the two modes (high = the platform provides the sport's skill API;
low = the agent implements the same API from the robot model, the joint interface with real limits and a disclosed
toolkit, evaluated after the match in an isolated container). A sport declares both in `sport.toml [modes.high]` /
`[modes.low]`; the files below give every sport the same contract base, toolkit manifest, hidden-request handling,
sandbox and post-match evaluator.*

## 里面有什么

| 文件 | 做什么 | 谁用 |
|---|---|---|
| `declare.py` | `sport.toml` 里 `[modes.high]` / `[modes.low]` 的格式和交叉检查（只用标准库，不 import 运动） | `tools/check` 第 1 步、`ci_checks/check-skilleval.py` |
| `toolkit.py` | 工具包清单 `seat/toolkit.toml` 的格式和 sha256 检查：`python3 -m skilleval.toolkit check <清单> [--fix]` | 同上；运动作者 |
| `contract.py` | `SkillContract`：请求字段和范围（就是高层动作的 schema，核心的格式：number、vector、enum）、怎么测、容差、单个请求的成功、汇总指标 | 运动的合同模块、模拟器（`/spec` 的 `skill_contract`）、评测器 |
| `cases.py` | API 测试的请求：按 id 的哈希分成隐藏集和公开练习集；练习集只给请求本身 | 运动生成案例；`declare.py` 检查 |
| `host.py` / `sandbox.py` | 在单独进程里运行选手的技能：`python -I`、清空的环境变量、自己的目录、资源和时间限制；容器里以 root 启动时降成 `nobody` | 评测器 |
| `layer.py` | `SkillLayer`：高层动作由某个技能实现来降成底层命令（比赛测试：固定参考策略 + 选手的实现；回合制项目直接可用） | 运动的评测钩子 |
| `inloop.py` | 技能在环（底层，docs/construction_zh.md「技能在环」）：比赛中选手程序发技能调用（如 `hit`），平台运行选手自己的技能文件，把调用降成底层命令；`step()` 每 `step_every` 个控制步经核心的 `tick` 钩子运行（命令照常记录，比赛可不带技能重放）；`free` 里的底层命令随时放行，其他的由运动的 `raw_allowed(seat, match, cmd)` 决定（看得到命令类型），否则 409；**计算时间计入比赛时间**（比赛服务里技能调用在世界锁外进行，世界照走，算完后的下一个控制步生效；`step()` 和加载在后台线程；`core.local.play` 用 `timing="free"`）；每分开始时文件变了才重载；每次调用记下文件 sha256、调用和返回时的控制步、墙钟时间和结果。`layer_from_env(...)` 按环境变量 `RCB_SKILL_INLOOP_<座位>` 打开 | 运动的动作层（底层的在环开关） |
| `skillhost.py` | 技能的宿主：只取普通文件的私有副本（不跟随符号链接、跳过 FIFO、有大小上限）、文件哈希、沙箱进程；进程内（`SkillHost`）或经 Unix 套接字（`serve` / `RemoteHost`：生成的任务里每个座位一个技能容器，没有网络、只读、看不到模拟器）。只用标准库，在技能容器里作为脚本运行 | `inloop.py`；`tools/generate.py` 的技能容器；`tools/live.py` |
| `evaluate.py` | 赛后评测器（在评测容器里运行）：API 测试、比赛测试、确定性检查、合同检查、隔离探针 | `tools/skill_eval.py` |
| `runtest.py` | 在运动镜像里运行指定的测试函数（`[modes] b9_test`） | `tools/check` 第 6 步 |
| `probe/probe.py` | 隔离探针：试着读模拟器和隐藏请求、写文件、连网，什么都不应该成功 | `tools/check` 第 6 步 |
| `tests/` | 单元测试，用一个合成的运动（`tests/fixture`），只要标准库：`python3 skilleval/tests/test_skilleval.py` | 改 `skilleval/` 的 PR |

主机一侧的启动器是 `tools/skill_eval.py`（工具部分：起容器、收报告），`tools/check` 的第 6 步用它。

## 运动要提供什么

```
sports/<运动>/
  sport.toml              [modes] b9_test；[modes.high] skills / contract / implementation；
                          [modes.low] implements / toolkit / tools / forbidden / evaluator / reference / boundary /
                          thresholds / deliverable / cases（字段说明：skilleval/declare.py 的文档字符串）；[seat.low] files
  <合同模块>.py            CONTRACTS = {"<技能>": SkillContract(...)}；高层动作的 schema 由它生成（单一来源）
  seat/toolkit.toml       工具包清单（属于运动，不属于机器人）
  seat/lib/...            清单里的库（例：IK 库、飞行模型）
  lowlevel/evaluator.py   评测钩子：skill_spec()、platform()、api_case(case, skill)，可选 competition_match(skill, seed)
  lowlevel/reference/     我们只用清单工具做出的底层参考解（规则 4）
  lowlevel/boundary/      边界基线：只用工具、不开发技能（规则 5，API 测试 ≈ 0）
  lowlevel/cases/         hidden.jsonl（只在模拟器一侧）、public.jsonl（练习集，进座位）
  lowlevel/reference_results.json   平台实现 / 参考解 / 边界基线的结果（tools/skill_eval.py --record）
  [modes.low.inloop]      技能在环（可选，推荐）：skills / step_every / free / noop / cpus / memory / reference_class / raw_rule
                          （skilleval/declare.py）；plugin.py 的底层用 inloop.layer_from_env(...)；CONTRIBUTING.md 第 12 项
```

## 技能在环（比赛中）

```
座位容器 /app/output/skill/<技能>.py ──只读卷──► 技能容器 skill_<座位>（skillhost.py serve；没有网络、只读根、无 /srv、
                                                  子进程 nobody + rlimit）──Unix 套接字（卷 skillipc_<座位>_trusted）──►
模拟器 sim：InLoopLayer（RCB_SKILL_INLOOP_<座位>=unix:/skillipc/<座位>/skill.sock）
```

设计和失败模式：olympiad_review `screening/skill_in_loop_design_zh.md`（阅读器 id `skillloop`）。

## 赛后评测

```
python3 tools/skill_eval.py <运动> <试次的 artifacts>/app/output/skill --out DIR      # 一个选手的最终实现
python3 tools/skill_eval.py <运动> platform|reference|boundary [--record]          # 参考行
python3 tools/skill_eval.py <运动> probe                                           # 隔离探针
```

评测容器：运动的镜像（或 `[modes.low] eval_image`，`eval_pip` 补上座位承诺给技能的包），没有网络，根文件系统只读（只有一个小的
`/tmp`），限 CPU、内存和进程数，去掉所有 capability，只留评测器需要的三个（读只对属主开放的模拟器目录、把技能进程降成
`nobody`），不许提权。只读挂载：模拟器目录（含隐藏请求）在 `/srv/sim`，技能在 `/app/output/skill`（先复制，只保留普通文件，
不跟随符号链接），工具包在座位里的同一路径。没有可写的挂载：报告从标准输出回来，写到 `DIR/report.json`。

报告里有三个数中的两个（第三个是比赛本身的结果，照常由验证器给出）：

- **API 测试**：每个隐藏请求经运动的 `api_case` 执行、按合同判定；报成功率、各项误差（均值、中位数、p90、最大）、被拒的命令、
  技能的错误和超时。
- **比赛测试**：运动的 `competition_match` 在每个种子上让我们固定的参考策略调用选手的实现；报成功率和平均分。
- 另有确定性检查（前几个请求在新进程里各跑两遍，回答必须一致）和合同检查（`--checks`：合同的请求 schema 等于高层动作的 schema，
  只是不比核心的 `entity` 字段——一个座位指挥几台机器人时，动作用它指明哪一台；每个案例的请求都合法）。

## 三个试点怎样对上这套接口

| | 乒乓（PR #31） | 击剑（PR #36） | 空气曲棍球（PR #37） | hello |
|---|---|---|---|---|
| 技能 API | `hit(t, pos, normal, vel)` | `strike(t, point, dir, vel)`，可选 `guard`、`step` | `strike(t, x, y, dx, dy, speed)`、`move(x, y)` | `aim(x, y)` |
| 合同 | `lowlevel/contract.py` → `SkillContract`：`pos` 是带 `box` 的 vector，`normal` `nonzero`，`vel` `max_norm` 8；容差 2 cm / 3° / 0.15 m/s + 10 % / 10 ms | 同形；容差 2 cm / 3° / 0.15 + 10 % / 10 ms | 容差在出球结果上（时间、位置、出球方向和速度） | 落点 3 cm |
| 交付 | `/app/output/skill/hit.py`，`memory = "per_case"` | `/app/output/skill/strike.py` | 试点用类接口，迁移时改成函数 | `/app/output/skill/aim.py` |
| 评测钩子 | `api_case`：从快照恢复、每 2 步调 `step`；`competition_match`：自己的实时循环（L1–L4） | 同乒乓 | 同 | `api_case` 一次飞行；`competition_match` 用 `SkillLayer` |
| 边界基线 | `ik_block` | `ik_rest` | `baseline_ik` | 朝向对准、固定出手速度 |
