# 全仓库重构候选清单

这是一份供后续逐项讨论、逐步实施的审计名单。**共 40 项，首批已完成 RF-002、RF-013、RF-029、RF-030、RF-036，余下 35 项继续按依赖分批推进。默认保留全仓库现有能力；删除实验分支、兼容路径、公共接口或归档证据，须与用户讨论后决定。** 勾选框表示该条已经完成约定的处理与验收，不表示“看过源码”或“提出建议”。

## 审计范围与基准

- **审计日期**：2026-09-30。
- **源码基准**：HEAD `95feef681de3517512d12f54fafa00ce6d2818fe` 加本次核查时的工作树。规划时的 HEAD 为 `2133ec9e`，写文档前已按当前 HEAD 重新定位；本清单不是某个干净 commit 的完整审计报告。
- **工作树事实**：包含已有 executor/policy/cross-device 修改，以及未跟踪的 `budget*`、`grouped_replay`、`study_parallel`、`study_publication`、实验脚本和任务文档。有真实入口/调用方的未跟踪模块属于当前能力，不能依据 Git 状态判为 dead code。
- **第三方基准**：Sample Factory pin 为 `ad256a6312db1ff46b19f539081abdbdf20fc424`；MIKASA pin 为 `017490bbb787c6efa8c025f9558e1a691965c0e0`；StarVLA pin 为 `ccca13c5177fe3d3c884b6e2de4965d916016649`，并有 QwenGR00T/QwenPI_v3/flow head 本地修改与未跟踪的 `action_noise.py`。因此父仓 commit 和子模块 pin 本身不足以标识当前 seeded-noise 行为；这些补丁也要纳入后续基准。HumanoidBench 子模块本地未检出，其内部实现未作当前源码核查。
- **覆盖方式**：对仓库自有的 core、scheduler、executors、latency、logging、envs、policy、data、integrations、training、cross_device、eval，以及相关 scripts/configs/tests/program/docs 做结构、调用关系和合同审计；第三方核查集中在实际集成接口、本地补丁和版本归属。未逐行穷举所有第三方源文件，也未遍历所有大型 outputs/checkpoint 内容。
- **初次审计验证**：源码阅读、静态调用搜索、模块定位、既有来源清单的摘要/大小核对、本文链接和编号校验。RF-033 的缺失模块和 RF-034 的 7 任务/17 文件不匹配均在初次审计工作树确认。
- **初次审计限制**：初次文档交付没有执行 pytest、真实模型推理、GPU/模拟器基准、训练、正式评测、SSH 或发布。后续已经执行的测试与 CPU 对照见下方批次记录；未完成条目中的“验收”仍是未来工作。

历史审计 [infra_release_cleanup](../docs/audits/infra_release_cleanup.md) 和 [archive](../docs/audits/archive/README.md) 只提供其日期下的背景。当前约束以 [AGENTS.md](../AGENTS.md)、[infra](../docs/infra.md)、[时序合同](../docs/latency_execution_semantics_en.md)、[测试约定](../tests/README.md) 及具体任务合同为依据。各条原始证据保留审计快照下的事实与行号，已完成条目追加当前实现与验收；后续应按函数/调用者重新定位，不能把过期行号当作缺陷仍存在的证明。

## 阅读与维护规则

**状态**：`候选` 表示有代码证据和明确维护价值，仍不是排期；`待核实` 表示成本、影响或支持合同尚待验证；`待讨论` 表示需要用户选择语义或删减方案。实现后才可改为 `已完成` 并勾选；讨论后决定不改可改为 `保留`，记录理由。一个条目既有确定事实又有条件风险时，分别写清，不能以高置信度替代实验验证。

**优先级**：P1 先处理实验含义、协议一致性和来源可复现性；P2 处理重复实现、职责与所有权；P3 处理接口、artifact 名称和文档组织。P1 不等于已经发生结果错误。例如 RF-008 的 fallback 存在已确认，但真实支持 wrapper 是否触发它尚待核实。

**动作**：Prune 为讨论后收缩/删除；Consolidate 为合并确实相同的实现；Redesign 为局部职责或状态模型调整；Document 为明确已有合同/单位/来源。动作可以组合，不按文件长度强制拆分，也不以“单一入口”强制合并语义不同的模式。

- 原因标为“已记录”“源码事实”或“推测”；没有 Git/文档证据时，不编造某个 workaround 的起因或已失效的实验结论。
- `mapping.get()`、局部 import、宽异常捕获、cast 或重复代码都只是线索。先核对必需字段合同、产品明确默认、optional dependency、清理/重新抛异常和性能边界；只把有具体语义问题的用法列为候选。
- 保持仓库的工程偏好：不添加内部防御检查、隐式字段替代、默认规范化或新的全局框架。若未来确实需要检查，先说明既有安全/API/测试合同和检查边界；不以这份清单授权新增 guard。
- 共用简单实现优先于新 registry、插件框架、通用 runner、resolved marker 或兼容 alias。共享短算式若反而增加间接层，可以讨论后保留，如 RF-027。
- 优先更新拥有该行为的既有测试。测试应锁住输入/输出、失败语义或真实不变量；性能探测是临时 probe，只有持续产品合同才成为长期测试，不写“代码不调用 .get()”之类风格断言。
- 编号 RF-001–RF-040 保持稳定；后续新增从 RF-041 起。合并或关闭条目时保留编号并指向接续项，每条只维护最新决定和验收证据，不追加操作时间线。
- 正式实验结果和最新任务状态仍归 [PROGRESS](PROGRESS.md)/任务报告，操作历史归 `logs/`，原始产物归 `outputs/`。本文不改写历史分数，不推断进程正在运行。

<a id="batch-01"></a>

## 首批实施与验收（2026-09-30）

用户确定每批 5 条、按依赖推进，并指定 [servers.md](../servers.md) 的四卡 RTX PRO 6000 主机建立临时测试环境。首批完成 **RF-002、RF-013、RF-029、RF-030、RF-036**；RF-014 因数值运算顺序实差后移，未修改归一化实现。

- **工作树边界：** HEAD 仍为上述审计基准，保留实施前所有未提交修改和 StarVLA 本地补丁。新增两个私有 helper，各有两个现有生产调用方；没有公共 API、配置字段或实验能力删减，没有新增内部防御检查或 fallback。没有改动 PROGRESS.md、来源 assets、子模块代码；验收基于完整工作树，原有修改与本批重构分别保留。
- **测试环境：** `ubuntu@136.115.188.185`，临时目录 `/home/ubuntu/sim2real/tmp/refactor-batch-01-20260930-01a0f1df`；独立 before/after 快照与 venv，Python 3.10.21，依赖复用现有 qwenoft 环境而新增安装仅进入 venv。测试进程 `CUDA_VISIBLE_DEVICES` 为空。版本与命令记录在 [environment.json](../outputs/refactor/batch-01/environment.json) 和 pytest command JSON。
- **行为结果：** 改动前 **156 通过、8 失败**；改动后 **159 通过、8 失败**。原有 164 例的结果及失败信息全部一致，新增 3 例通过。8 个原有失败为 RF-034 的七个 overlay 来源摘要，以及现有 Atlantis translation fixture 缺 `data_root_dir`；没有刷新期望值或补默认值。四个导入相关 suite 正反序各 71 例，路径均恢复、模块均来自当前快照，真实 Atari 对齐用例通过。
- **CPU 对照：** scheduler 组件全部输出一致，单 FIFO update 微操作仍有约 +0.164μs（+5.7%）开销；进一步对真实 SimulatedLatencyExecutor 的 CPU 单集循环做四组对照，动作、ready/apply 帧与指标一致，最大中位数增幅 1.4%。这是固定 DummyEnv/constant policy 的 CPU 证据，不代表真实模型、GPU 吞吐或正式实验结果。
- **主证据：** [首批验收报告](../outputs/refactor/batch-01/report.md)，附 [前后测试结果比较](../outputs/refactor/batch-01/outcome-comparison.json)、[实现差异](../outputs/refactor/batch-01/implementation.patch)、原始日志、JUnit XML 和临时 probe 结果。

## 优先级总表

共 **40 项**：P1 9 项、P2 27 项、P3 4 项；其中 **5 项已完成，35 项未完成**。可从编号直达详细证据；未完成条目的动作仍是建议。

| 编号 | 优先级 | 候选 | 状态 | 动作 |
|---|---|---|---|---|
| [RF-001](#rf-001) | P1 | 批量 streaming 与记录式 episode metrics 有语义分叉 | 待讨论 | Document → Consolidate（语义讨论后） |
| [RF-002](#rf-002) | P2 | live scheduler 与 chunk peek 重复实现 ready-event 选择 | 已完成 | Consolidate |
| [RF-003](#rf-003) | P2 | action-prefix projection 每次都深拷贝完整 scheduler | 待核实 | Redesign（性能收益验证后） |
| [RF-004](#rf-004) | P2 | raw-frame 边界容差分别散落在 clock、scheduler、日志回填 | 待核实 | Document / Consolidate（边界验证后） |
| [RF-005](#rf-005) | P2 | 时间记录字段混用秒、毫秒与进程内 monotonic 时间 | 候选 | Document |
| [RF-006](#rf-006) | P2 | executor clock 与 EnvAdapter 各自计算 simulation time | 待核实 | Document / Redesign（支持合同确认后） |
| [RF-007](#rf-007) | P3 | ProcessInferencePool 接收但丢弃 `now_ms` | 待讨论 | Prune（接口讨论后） |
| [RF-008](#rf-008) | P1 | Deadly Corridor runtime button order 存在静默假设和静默映射 | 待核实 | Document / Prune（fallback 合同讨论后） |
| [RF-009](#rf-009) | P3 | `latencies.jsonl` 与 `infer_latencies.jsonl` 写入相同 latency records | 待讨论 | Document / Prune（artifact 讨论后） |
| [RF-010](#rf-010) | P2 | single 与 batched executor factory 只在部分 assembly 上重复 | 候选 | Consolidate |
| [RF-011](#rf-011) | P2 | 按边界拆分通用 StarVLA 适配器 | 候选 | Redesign（现有职责整理） |
| [RF-012](#rf-012) | P1 | 合并 StarVLA 模型输入栈的重复解析 | 候选 | Consolidate |
| [RF-013](#rf-013) | P2 | 让 policy action label 引用 canonical label 表 | 已完成 | Consolidate |
| [RF-014](#rf-014) | P2 | 共用状态归一化数值核心 | 待核实（数值合同） | Consolidate |
| [RF-015](#rf-015) | P1 | 明确 mapless latency prompt 的时钟来源 | 待核实 | Document / Consolidate（生产路径核实后） |
| [RF-016](#rf-016) | P2 | 切开StarVLA evaluation对training CLI的依赖 | 候选 | Consolidate |
| [RF-017](#rf-017) | P2 | 将teacher rollout的大型双阶段collector恢复成清晰编排边界 | 候选 | Redesign（现有编排边界整理） |
| [RF-018](#rf-018) | P2 | 整理checkpoint文件内部边界，不改变保留生命周期 | 候选 | Redesign（文件职责整理） |
| [RF-019](#rf-019) | P1 | 先决定两种SF checkpoint loader的坏文件合同 | 待讨论 | Document → Consolidate（失败策略讨论后） |
| [RF-020](#rf-020) | P2 | 明确filter kwargs是否是长期公开接口 | 待讨论 | Document / Consolidate（现有 filter 合同讨论后） |
| [RF-021](#rf-021) | P1 | 统一 Flappy 预算实验的 55-run 总体定义 | 候选 | Consolidate |
| [RF-022](#rf-022) | P1 | 让两 episode session 宽度只有一个内部定义 | 候选 | Consolidate |
| [RF-023](#rf-023) | P2 | 收敛 generic profile 与 sim2real 的重复 runtime overlay | 候选 | Consolidate |
| [RF-024](#rf-024) | P2 | 合并 resident replay 的共同准备与资源生命周期 | 候选 | Consolidate |
| [RF-025](#rf-025) | P2 | 合并 grouped replay 的 episode 持久化公共路径 | 候选 | Consolidate |
| [RF-026](#rf-026) | P2 | 成本汇总从 timing JSON 读语义字段，不解析 artifact 路径 | 待讨论 | Consolidate（producer/report 坐标归属） |
| [RF-027](#rf-027) | P2 | 共用每 case 归一化误差公式，保持统计汇总分开 | 待讨论 | Consolidate 或 Document（净收益讨论后） |
| [RF-028](#rf-028) | P2 | 明确 evaluation config resolution 的嵌套边界 | 待讨论 | Document / Consolidate（入口合同确认后） |
| [RF-029](#rf-029) | P2 | 将 Sample Factory policy type 分类集中到单一出处 | 已完成 | Consolidate |
| [RF-030](#rf-030) | P2 | 抽取 fixed-latency sweep 的配置准备，不动运行汇总层 | 已完成 | Consolidate |
| [RF-031](#rf-031) | P2 | 明确 generic profile 命令的工作目录契约 | 待核实 | Document / Consolidate（CWD 支持范围讨论后） |
| [RF-032](#rf-032) | P3 | 评估 18 份 sim2real archived eval snapshot 的重复，不覆盖来源档案 | 待讨论 | Document（runtime overlay 提取另议） |
| [RF-033](#rf-033) | P1 | 已不存在的 formal_chain_watchdog 仍有测试调用 | 待讨论 | Prune 或 Consolidate（原测试合同讨论后） |
| [RF-034](#rf-034) | P1 | pi0.5 来源快照与 source_map 的摘要、大小不一致 | 待讨论 | Document（来源调查后决定修复） |
| [RF-035](#rf-035) | P2 | 一部分测试锁定源码表达或生产对象私有状态 | 候选（删减具体测试需讨论） | Consolidate（删减测试另议） |
| [RF-036](#rf-036) | P2 | 第三方导入的测试作用域会修改全局 sys.path | 已完成 | Consolidate |
| [RF-037](#rf-037) | P2 | MIKASA 多代 launcher 继续维护相近的完整实验编排 | 待讨论 | Consolidate / Prune（逐入口讨论后） |
| [RF-038](#rf-038) | P2 | 跨主机实验入口绑定固定主机、解释器和 GPU 布局 | 待讨论 | Document / Consolidate / Prune（用途讨论后） |
| [RF-039](#rf-039) | P2 | MIKASA publisher 重复上传和 revision 回读机制 | 候选（发布/兼容行为改动待讨论） | Consolidate（发布语义变更另议） |
| [RF-040](#rf-040) | P3 | 协议版本、历史状态与正式结果的文档入口需要更明确 | 候选（文档职责调整） | Document（归档/删文档另议） |

## 16 类问题覆盖映射

下表列出本轮已经找到具体证据的候选。类别覆盖不代表已经穷尽全仓库所有问题；没有独立确认的事项在“待核实”中说明，避免把检索线索写成删除名单。

| 分类 | 名称 | 对应条目 / 当前核查范围 |
|---|---|---|
| C01 | 失效历史路线 | [RF-037](#rf-037) 是退役候选，当前仍有调用方，路线是否失效待核实；RF-033 仅确认孤儿测试 |
| C02 | 妥协债务 / workaround | [RF-008](#rf-008) |
| C03 | 概念与语义漂移 | [RF-001](#rf-001)、[RF-005](#rf-005)、[RF-008](#rf-008)、[RF-013](#rf-013)、[RF-015](#rf-015)、[RF-023](#rf-023)、[RF-029](#rf-029)、[RF-040](#rf-040) |
| C04 | 重复实现 | [RF-001](#rf-001)、[RF-002](#rf-002)、[RF-004](#rf-004)、[RF-007](#rf-007)、[RF-009](#rf-009)、[RF-010](#rf-010)、[RF-012](#rf-012)、[RF-013](#rf-013)、[RF-014](#rf-014)、[RF-022](#rf-022)、[RF-023](#rf-023)、[RF-024](#rf-024)、[RF-025](#rf-025)、[RF-027](#rf-027)、[RF-028](#rf-028)、[RF-029](#rf-029)、[RF-030](#rf-030)、[RF-037](#rf-037)、[RF-039](#rf-039) |
| C05 | 抽象不匹配 | [RF-003](#rf-003)、[RF-010](#rf-010)、[RF-011](#rf-011)、[RF-017](#rf-017)、[RF-018](#rf-018)、[RF-035](#rf-035) |
| C06 | 隐式状态 / 隐藏耦合 | [RF-006](#rf-006)、[RF-016](#rf-016)、[RF-017](#rf-017)、[RF-026](#rf-026)、[RF-028](#rf-028)、[RF-031](#rf-031)、[RF-036](#rf-036)、[RF-038](#rf-038) |
| C07 | 配置债务 | [RF-015](#rf-015)、[RF-016](#rf-016)、[RF-020](#rf-020)、[RF-028](#rf-028)、[RF-030](#rf-030)、[RF-031](#rf-031)、[RF-032](#rf-032)、[RF-037](#rf-037)、[RF-038](#rf-038) |
| C08 | 数据 / checkpoint / 实验协议债务 | [RF-001](#rf-001)、[RF-017](#rf-017)、[RF-019](#rf-019)、[RF-021](#rf-021)、[RF-022](#rf-022)、[RF-026](#rf-026)、[RF-027](#rf-027)、[RF-034](#rf-034)、[RF-037](#rf-037)、[RF-039](#rf-039)、[RF-040](#rf-040) |
| C09 | 训练 / 评测 / 部署漂移 | [RF-012](#rf-012)、[RF-014](#rf-014)、[RF-015](#rf-015)、[RF-023](#rf-023)、[RF-032](#rf-032) |
| C10 | 时间逻辑与单位 | [RF-004](#rf-004)、[RF-005](#rf-005)、[RF-006](#rf-006)、[RF-007](#rf-007)、[RF-015](#rf-015)、[RF-023](#rf-023) |
| C11 | 状态 / 资源所有权 | [RF-003](#rf-003)、[RF-006](#rf-006)、[RF-010](#rf-010)、[RF-018](#rf-018)、[RF-022](#rf-022)、[RF-024](#rf-024)、[RF-025](#rf-025)、[RF-036](#rf-036)、[RF-039](#rf-039) |
| C12 | API 泄漏实现细节 | [RF-002](#rf-002)、[RF-007](#rf-007)、[RF-009](#rf-009)、[RF-011](#rf-011)、[RF-016](#rf-016)、[RF-020](#rf-020)、[RF-025](#rf-025)、[RF-035](#rf-035) |
| C13 | 错误处理 / 静默失败 | [RF-008](#rf-008)、[RF-019](#rf-019) |
| C14 | 可复现性 | [RF-021](#rf-021)、[RF-026](#rf-026)、[RF-031](#rf-031)、[RF-032](#rf-032)、[RF-034](#rf-034)、[RF-038](#rf-038)、[RF-039](#rf-039)、[RF-040](#rf-040) |
| C15 | 依赖 / 环境债务 | [RF-024](#rf-024)、[RF-036](#rf-036)、[RF-038](#rf-038) |
| C16 | 仓库卫生 | [RF-009](#rf-009)、[RF-033](#rf-033)、[RF-034](#rf-034)、[RF-040](#rf-040) |

## 时序、调度与运行时

<a id="rf-001"></a>

### RF-001 — 批量 streaming 与记录式 episode metrics 有语义分叉

- [ ] **状态：待讨论**
- **动作层级**：Document → Consolidate（语义讨论后）。
- **优先级 / 置信度 / 变更类型：** P1 / 高 / 合并提取逻辑，语义决策后再改行为。
- **分类：** C03、C04、C08。
- **证据：** 批量执行器在 [`_EpisodeBuffers.record_step()`](../latency_bench/executors/batched_simulated.py#L64-L91) 累加 score、raw return、invalid-action 和 task metrics；[`_compute_episode_metrics()`](../latency_bench/executors/batched_simulated.py#L693-L729) 在 step 与 action records 都启用时走记录式 `compute_episode_metrics()`，否则走聚合式 `compute_episode_metrics_from_aggregates()`。记录式提取位于 [`compute_episode_metrics()`](../latency_bench/logging/metrics.py#L12-L42)、[`_extract_game_score()`](../latency_bench/logging/metrics.py#L190-L198) 和 [`_extract_explicit_return()`](../latency_bench/logging/metrics.py#L201-L219)。
- **当前消费者 / 测试所有者：** 批量 evaluation 的低内存路径与保留 step/action 记录的路径；单集及 realtime 通过记录式路径。行为所有者是 [`tests/executors/test_batched_simulated_executor.py`](../tests/executors/test_batched_simulated_executor.py) 和 [`tests/logging/test_logging_records.py`](../tests/logging/test_logging_records.py)。
- **已证事实与历史：** [`docs/infra.md`](../docs/infra.md#L86-L91) 明确记载二者对 `score`/`game_score`、raw-return 别名和缺失最终值的解释不同，并记录同时出现 `game_score=1, score=2` 时 trace 为 1、streaming 为 2。代码也能直接证实：streaming 顺序赋值使 `score` 覆盖 `game_score`，而 trace 优先取 `game_score`。文档称基础设施清理有意维持现有解释；分叉最初为何形成仍未知。
- **价值与最小动作：** 先把每个不同输入的输出规则写成明确表格，并识别受影响的既有实验；再讨论能否共用一个增量 accumulator，同时保持按需保存 records 的性能路径。此项有真实语义变化风险，不能把“抽出共享 helper”当成纯机械清理。
- **需保留 / 风险：** 不得为复用逻辑而在 streaming 路径构造整集 `StepRecord` / `ActionEvent`；这会改变设计目的和内存开销。共享实现若统一优先级，会改变已记录的 episode metrics。
- **验收与依赖：** 在两种记录设置下比较 dual-score、最终步缺失 score、早期步有/最终步无 score、每个 raw-return alias、`episode_extra_stats`、task metrics、invalid action 与 action counts。先确定输出语义和受影响实验，再改 implementation；不要仅凭现有单一路径测试推定两路等价。
- **必须讨论：** `score` 与 `game_score` 同时存在时的优先级；中间步与最终步缺值时是否沿用最近值；raw-return 别名和 `episode_extra_stats` 的优先顺序。任何修正都要单独报告实验数据影响。

<a id="rf-002"></a>

### RF-002 — live scheduler 与 chunk peek 重复实现 ready-event 选择

- [x] **状态：已完成**
- **动作层级**：Consolidate。
- **优先级 / 置信度 / 变更类型：** P2 / 高 / 局部合并。
- **分类：** C04、C12。
- **证据：** [`ActionScheduler.update()`](../latency_bench/scheduler/action_queue.py#L77-L120) 选择 FIFO 队首或 latest-ready 事件、丢弃较旧 ready 事件并更新 live state；[`scheduled_chunk_event()`](../latency_bench/scheduler/action_queue.py#L122-L143) 另行实现 ready 过滤、同一 ordering 分支和 chunk head 选择。后者由 [`with_action_prefix()`](../latency_bench/policy/action_prefix.py#L82-L105) 用于编码 `returned_action_context`；固定 prefix 投影还会走 [`project_actions()`](../latency_bench/scheduler/action_queue.py#L63-L71)。
- **当前消费者 / 测试所有者：** executor 的 live action scheduling、policy action-prefix 和 Balance returned-action context。scheduler 行为归 [`tests/scheduler/test_action_queue.py`](../tests/scheduler/test_action_queue.py) 与 [`tests/scheduler/test_action_chunk.py`](../tests/scheduler/test_action_chunk.py)；其中 [`test_returned_action_state_and_prefix_share_the_observation_snapshot()`](../tests/scheduler/test_action_chunk.py#L235-L255) 已间接覆盖 chunk peek，但不是完整的 FIFO/latest-ready 对照矩阵。
- **已证事实与历史：** 重复的 readiness/ordering 分支可从源码直接确认；peek 需要不改变队列状态，live update 则会标记丢弃和应用事件。为何分开实现未在代码/文档中说明，不能据此推断为历史遗留。
- **价值与最小动作：** 提取一个只负责选择“本帧将被考虑的事件/ready 集合”的纯内部 helper；由 live update 独自处理 drop 标记、队列移除、chunk 计数和 persistence，peek 独自决定如何描述投影。只共享已确认相同的选择规则，不共享 mutating transition。
- **需保留 / 风险：** peek 必须无副作用；live update 需继续记录丢弃和 underflow。chunk 正好到尾部、latest-ready 丢旧事件、旧 chunk 在新事件 ready 前继续执行时可能出现边界差异。
- **验收与依赖：** 先复用 scheduler ordering contract 与 `test_returned_action_state_and_prefix_share_the_observation_snapshot()`；若局部合并会触及未覆盖的 FIFO/latest-ready 组合，再扩展现有 `tests/scheduler` owner suite：比较 peek 的下一 head 与真实下一次 `update()`，并确认 peek 不修改 applied/drop count 或 chunk metrics。
- **必须讨论：** 若共享 helper 会改变 latest-ready 丢弃标记的时机或 peek 在 underflow 时返回值，应先确定 policy 输入合同；不要借机改 ordering。
- **首批实施与验收（2026-09-30）：** 共享 [`_select_ready_events()`](../latency_bench/scheduler/action_queue.py#L77)，[`update()`](../latency_bench/scheduler/action_queue.py#L99) 独自处理 live transition，[`scheduled_chunk_event()`](../latency_bench/scheduler/action_queue.py#L133) 只读。FIFO 返回选中事件及 `None`，latest-ready 返回选中事件及 ready list；update 使用该结果而不重复判 ordering。新增既有 scheduler suite 的两策略 pending-chunk peek 合同用例通过，既有 ordering/chunk/persistence 用例保持结果。CPU 组件输出全等；单 FIFO update 微操作仍 +5.7%（+0.164μs），进一步核查实际单集 CPU 循环最大中位数增幅为 1.4%，详见首批报告，不宣称提速或真实模型性能已验收。 主证据：[首批报告](../outputs/refactor/batch-01/report.md)，整体结果与边界见[批次记录](#batch-01)。


<a id="rf-003"></a>

### RF-003 — action-prefix projection 每次都深拷贝完整 scheduler

- [ ] **状态：待核实**
- **动作层级**：Redesign（性能收益验证后）。
- **优先级 / 置信度 / 变更类型：** P2 / 代码路径确定、性能收益未测 / 局部重设计。
- **分类：** C05、C11。
- **证据：** [`ActionScheduler.project_actions()`](../latency_bench/scheduler/action_queue.py#L63-L71) 对整个 scheduler 做 `deepcopy`，只显式排除了 `_applied_events`；随后对副本重放 `update()`。[`action_prefix_arrays()`](../latency_bench/policy/action_prefix.py#L65-L73) 是已证实调用方，并由 [`with_action_prefix()`](../latency_bench/policy/action_prefix.py#L82-L84) 在观察进入 policy 前调用。待拷贝对象包括 pending event、metadata 和 action chunk；是否达到可见开销尚未测量。
- **当前消费者 / 测试所有者：** 配置了固定/profile action prefix 的 policy 输入路径。相关 contract 在 [`tests/scheduler/test_action_chunk.py`](../tests/scheduler/test_action_chunk.py#L19-L101)；目前未找到专门测量投影开销的 benchmark。
- **已证事实与历史：** 每次调用都会 deepcopy 是源码事实；“它显著拖慢 latency-sensitive 路径”目前只是待验证推断。注释说明它是为了使用 live scheduling rules 做无 mutation 的 projection，并排除了 completed history。
- **价值与最小动作：** 先用代表性 chunk 大小和 pending event 数量测 projection 的时间/分配；若成本显著，再建立只含预测所需状态的 projection snapshot，避免复制完整 event history 和无关 metadata。不要新增通用 clone framework。
- **需保留 / 风险：** action prefix 必须与 scheduler 的真实 FIFO/latest-ready、persistence、active chunk 和 hold-last 规则相同，并且不改变 live queue。过早手写快照可能遗漏新加入状态，导致 policy 获得和执行动作不一致的条件信息。
- **验收与依赖：** 优先复用 `tests/scheduler/test_action_chunk.py` 的 prefix/live 行为测试；若 snapshot 重设计触及尚未覆盖的 delayed action 或 ordering case，再扩展该 owner suite，对比 prefix 数组与逐帧 live 结果并检查原 scheduler 未变；随后才做同 workload 基准。未测出成本前保持现状。
- **必须讨论：** 只有测量证实有实际收益时才进入实现；性能比较要固定 pending event / chunk size 和执行模式。

<a id="rf-004"></a>

### RF-004 — raw-frame 边界容差分别散落在 clock、scheduler、日志回填

- [ ] **状态：待核实**
- **动作层级**：Document / Consolidate（边界验证后）。
- **优先级 / 置信度 / 变更类型：** P2 / 高（分散规则事实）、中（结果分叉）/ 文档或局部归并。
- **分类：** C10、C04。
- **证据：** [`frames_float_to_raw_frames()`](../latency_bench/core/clock.py#L16-L23) 在 ceil 前减 `1e-10`；[`ActionScheduler.__init__()`](../latency_bench/scheduler/action_queue.py#L21-L35) 和 [`_is_ready()`](../latency_bench/scheduler/action_queue.py#L201-L204) 用基于 frame_ms 的 ready-time epsilon；日志的 [`_latency_ms_to_raw_frames()` / `_ceil_step_float()`](../latency_bench/logging/metrics.py#L239-L250) 复用一套 frame 转换，但对浮点 step 又单独以 `1e-12` `isclose` 后 ceil。
- **当前消费者 / 测试所有者：** sampler 的 ms-to-frame 转换、无 raw-frame 事件的 scheduler ready 判断、旧/不完整记录的 raw-frame backfill。测试归 [`tests/core/test_clock.py`](../tests/core/test_clock.py)、[`tests/executors/test_latency_execution_semantics_contract.py`](../tests/executors/test_latency_execution_semantics_contract.py#L23-L30)、[`tests/scheduler/test_action_queue.py`](../tests/scheduler/test_action_queue.py#L27-L54)。
- **已证事实与历史：** 三种 tolerance 事实可由源码证实；没有当前证据表明它们已经让同一输入得到不同的最终动作或统计。现有契约明确采用 ceil raw-frame 语义。
- **价值与最小动作：** 给每处 tolerance 补充它保护的数值边界与调用对象；用专门边界样例验证是否实际需要统一。若确认语义相同，再把共同的帧舍入规则放回 core clock；scheduler ready epsilon 和历史数据 backfill 只有证明等价后才共享。
- **需保留 / 风险：** 整数 `ready_raw_frame` 是直接权威；不要改成 `latency_ms` 再算一次。`ready_time_ms` 的比较与 `ready_step_float` 的记录回填不是天然同一入口。
- **验收与依赖：** 优先复用现有 clock ceil 测试和 scheduler float-boundary 测试；若这些现有 case 无法区分三处 tolerance，再在对应 owner suite 扩展边界值，确认 apply frame、`ready_raw_frame` 和记录 latency frames 的关系。
- **必须讨论：** 若统一会改变任一近边界样例的 frame/apply frame，需要先决定保留哪个历史语义，并盘点依赖这些 trace 的对比实验。

<a id="rf-005"></a>

### RF-005 — 时间记录字段混用秒、毫秒与进程内 monotonic 时间

- [ ] **状态：候选**
- **动作层级**：Document。
- **优先级 / 置信度 / 变更类型：** P2 / 高 / 文档与字段责任说明，不改 wire schema。
- **分类：** C03、C10。
- **证据：** [`LatencyRecord`](../latency_bench/core/types.py#L57-L71) 把 `latency_ms`、`inference_ms`、`start_wall_time`、`end_wall_time` 都声明为 `float`，字段没有单位/时钟源说明；[`RuntimeLatencyMeasurer.run_and_measure()`](../latency_bench/latency/measurement.py#L19-L36) 写入以秒为单位的 `perf_counter()` 起止值，同时把差值乘 1000 写成毫秒。Stage timings 是毫秒 [`StageProfiler._timer()`](../latency_bench/core/timing.py#L35-L43)；realtime helper 是 `perf_counter()*1000` [`monotonic_ms()`](../latency_bench/executors/realtime/timing.py#L1-L7)；env backend 的 `worker_step_wall_sec` 则是秒 [`SerialEnvStepBackend.step_slots()`](../latency_bench/executors/env_step_backend.py#L175-L182)。
- **当前消费者 / 测试所有者：** latency measurement 将记录送进 executor/logger；测试 [`tests/latency/test_latency_measurement.py`](../tests/latency/test_latency_measurement.py#L8-L19) 只检查 latency 非负、inference_ms 相等和 end>=start，没有断言单位或 `perf_counter` provenance。realtime 输出合同归 [`docs/realtime_pipeline_logging.md`](../docs/realtime_pipeline_logging.md#L1-L24)。
- **已证事实与历史：** 每个计时点的单位和计时源由写入代码确认；没有证据证明当前消费者把秒当毫秒。`start_wall_time` 名称是否沿用旧语义不明。
- **价值与最小动作：** 给 `LatencyRecord` 字段说明单位及“进程内 monotonic、非 Unix epoch”的性质，并对 profiling 字段按模块声明 `_ms` / `_sec` 单位。保留现有 JSON 字段名和数值，仅消除读者误解。
- **需保留 / 风险：** `perf_counter` 起止时刻不能跨机器或跨独立 clock domain 直接比较；不同模块测得的 stage durations 是同类毫秒持续时间，但不能把绝对起止点混加到 episode simulation time。
- **验收与依赖：** 文档逐字段对照写入点；测试补充适当的量级/单位 contract 时，应避免基于运行时长设脆弱门槛。Realtime Pipeline 文档“durations are ms”适用于其表格，不应被误认为包括所有 backend metadata。
- **必须讨论：** 不改 `start_wall_time`/`end_wall_time` 名称或导出格式；任何重命名、换 epoch clock 的提议都需另做 schema/下游兼容评估。

<a id="rf-006"></a>

### RF-006 — executor clock 与 EnvAdapter 各自计算 simulation time

- [ ] **状态：待核实**
- **动作层级**：Document / Redesign（支持合同确认后）。
- **优先级 / 置信度 / 变更类型：** P2 / 中 / 明确 FPS 所有权，条件成立时才改构造方式。
- **分类：** C10、C11、C06。
- **证据：** [`EnvClock`](../latency_bench/core/clock.py#L25-L50) 从 `env_fps` 提供 `frame_ms` / step-time；[`EnvAdapter.frame_ms`](../latency_bench/envs/base.py#L17-L20)、[`_metadata()` / `_make_observation()`](../latency_bench/envs/base.py#L29-L49) 和 [`get_state_info()`](../latency_bench/envs/base.py#L82-L91) 则从 `env.env_fps` 计算。factory 同时保留 config-built clock 和可注入 env：[`_build_executor_runtime_components()`](../latency_bench/executors/factory.py#L115-L132)；clock 的来源是 [`_build_clock()`](../latency_bench/executors/factory.py#L382-L387)。
- **当前消费者 / 测试所有者：** executor 用 EnvClock 给调度器、事件与 metrics 标时；adapter 在 observation/step info 上写自己的 sim_time_ms。factory 注入 env 的现有路径由 [`test_realtime_mode_uses_injected_env_and_inference_devices()`](../tests/executors/test_executor_factory.py#L744-L755) 覆盖，但该用例只确认 env/device 注入，没有 fps 对照。
- **已证事实与历史：** 两个对象独立保存/计算 fps 是事实。正常 factory adapter 从同一 config 取得 env_fps，未发现默认路径有漂移。不同值只可能在调用方注入 fps 与 config 不一致的 env 时发生；这种注入是否支持不一致尚无文档合同，故属于条件风险。
- **价值与最小动作：** 先声明 injected EnvAdapter 的 `env_fps` 必须与 executor config `env_fps` 相同，或相反地声明 adapter 是该路径的唯一 FPS source。若当前行为要求一致，后续可在创建 adapter 时使用一个已解析值；无需增加内部层层检查或多余 Clock 抽象。
- **需保留 / 风险：** adapter 自身必须继续给原始环境 observation 提供其帧率语义；executor clock 还负责 `obs_fps` 与 raw-frame cadence，职责不等同。
- **验收与依赖：** 先检查所有 `build_executor(..., env=...)` 调用方；若 FPS 一致性被定为调用合同，扩展现有 injected-env owner test，确认 observation sim time、scheduler ready frame 和 action metadata 使用同一速率。仅文档合同则审阅已有注入测试和配置生成，不要求另建测试。
- **必须讨论：** 明确 fps 不匹配目前是非法输入、允许 adapter 覆盖 config，还是仅测试便利条件。不要未经约定添加 `ValueError` 防御检查。

<a id="rf-007"></a>

### RF-007 — ProcessInferencePool 接收但丢弃 `now_ms`

- [ ] **状态：待讨论**
- **动作层级**：Prune（接口讨论后）。
- **优先级 / 置信度 / 变更类型：** P3 / 高 / 删除无用内部参数或定义其语义。
- **分类：** C04、C10、C12。
- **证据：** [`ProcessInferencePool.drain_completed()`](../latency_bench/executors/realtime/pool.py#L168-L171) 的第一步是 `del now_ms`，实际只轮询 pipe 并读当前已完成消息。realtime 调用方传 `monotonic_ms()` 或 elapsed ms [`RealtimeLatencyExecutor`](../latency_bench/executors/realtime/executor.py#L745-L781)；simulated pooled path 传 `0.0` [`SimulatedLatencyExecutor`](../latency_bench/executors/simulated.py#L177-L193)。
- **当前消费者 / 测试所有者：** 生产调用方只有 realtime 与 simulated inference executor；测试文件中的 fake pool 也镜像了当前签名 [`tests/executors/test_realtime_executor.py`](../tests/executors/test_realtime_executor.py#L117-L117)、[`tests/executors/test_realtime_executor.py`](../tests/executors/test_realtime_executor.py#L495-L495)、[`tests/executors/test_simulated_inference_pool.py`](../tests/executors/test_simulated_inference_pool.py#L95-L95)，但测试替身本身不构成保留参数的生产需求。
- **已证事实与历史：** 真正完成条件来自 OS pipe poll，参数值对当前实现没有影响。测试替身需要跟随生产方法签名，但这只是修改时的同步维护工作；接口最初是否为时间截止设计无从源码确认。
- **价值与最小动作：** 在确认它是内部接口后，移除参数及所有 call-site/fake 形参，让调用表达“drain all IPC completions currently available”。如预期未来有截止时间，要改成有可观察效果的契约，而非保留无效参数。
- **需保留 / 风险：** 不要改成只返回 `now_ms` 之前完成的结果；执行器的 completion time 和 profile timestamps 是实际 worker/parent 事件时间。签名更改不应延迟读取已到 pipe 的消息。
- **验收与依赖：** 若讨论后移除参数，同步更新两个测试模块中跟随接口的 fake signature；先用现有 worker completion 事件、latest-ready、episode drain 和 cleanup 行为测试确认结果不变，再核对仓库生产调用点。测试替身只随接口更新，不作为接口需求证据。
- **必须讨论：** 要移除 `now_ms` 前确认这个方法不被视为外部 API。若将来 executor 需要可注入时钟，应单独定义使用场景而非把无效值继续传入。

<a id="rf-008"></a>

### RF-008 — Deadly Corridor runtime button order 存在静默假设和静默映射

- [ ] **状态：待核实**
- **动作层级**：Document / Prune（fallback 合同讨论后）。
- **优先级 / 置信度 / 变更类型：** P1 / 中 / fallback 合同审查，可能需要边界行为变更。
- **分类：** C02、C03、C13。
- **证据：** [`_deadly_runtime_button_names()`](../latency_bench/envs/deadly_corridor.py#L289-L316) 尝试从 env/game 获取按钮顺序，找不到时返回语义常量顺序；[`_semantic_to_runtime_multibinary()`](../latency_bench/envs/deadly_corridor.py#L319-L325) 遇到 runtime 名称不在语义表中会填 0。VLA adapter 在 [`__init__()`](../latency_bench/envs/deadly_corridor.py#L380-L390) 保存这个顺序，在 [`step()`](../latency_bench/envs/deadly_corridor.py#L409-L425) 将 action 重新映射。factory 对 multibinary StarVLA 条件路由到此实现 [`_build_deadly_corridor_env()`](../latency_bench/envs/factory.py#L197-L220)。
- **当前消费者 / 测试所有者：** `action_layout=multibinary_7` 的 Deadly Corridor VLA evaluation；单测 [`test_deadly_corridor_multibinary_buttons_follow_runtime_order()`](../tests/envs/test_deadly_corridor_env_adapter.py#L59-L71) 覆盖一个显式非语义顺序，integration [`test_hist8_deadly_vla_uses_the_teacher_resolution_and_hud()`](../tests/integration/test_deadly_render_contract.py#L16-L38) 验证真实 env 的图像合同但未验证动作布局。
- **已证事实与历史：** 两个 fallback 分支确实存在。docstring 表明 runtime order 不保证等于 semantic order，和默认 semantic order 存在需要确认的张力；是否当前受支持的 ViZDoom wrapper 总能 introspect button list，没有通过现有测试确证。代码注释称逻辑移植自 `eval_core`；这是来源说明，不是 fallback 安全性的证据。
- **价值与最小动作：** 先在现有真实 wrapper 环境验证 `get_available_buttons()` 可见性、输出名字完整性及不同按钮版本。若默认语义顺序只对明确版本可靠，应把它写成支持合同；若未知布局不能安全运行，再讨论在 adapter 边界报错或提供显式配置映射。不要未经验证直接删 fallback。
- **需保留 / 风险：** arbitrary-button subset 是该 VLA adapter 与 teacher factorised-action adapter 的主要语义区别。映射错误会悄悄忽略按钮或触发错误按钮，直接改变轨迹；任何动作变更都属高风险。
- **验收与依赖：** 先复用现有 runtime-order mapping 单测及真实 ViZDoom 图像 integration contract。若后续选择改变 fallback，再扩展 `tests/envs/test_deadly_corridor_env_adapter.py` 覆盖无法 introspect/未知名及 semantic one-hot 映射；运行条件依赖 ViZDoom/Gymnasium integration 环境，本次未运行。
- **必须讨论：** 哪些 runtime button 名可用、顺序是否稳定、缺失 introspection 是否为受支持状态、未知按钮是错误还是合法扩展。任何删除 `get(...,0)`/默认顺序都需先与用户确认。

<a id="rf-009"></a>

### RF-009 — `latencies.jsonl` 与 `infer_latencies.jsonl` 写入相同 latency records

- [ ] **状态：待讨论**
- **动作层级**：Document / Prune（artifact 讨论后）。
- **优先级 / 置信度 / 变更类型：** P3 / 高 / artifact alias/公开输出合同整理。
- **分类：** C04、C12、C16。
- **证据：** [`TrajectoryLogger._flush_episode_records()`](../latency_bench/logging/trajectory_logger.py#L92-L111) 将同一 `self.latency_records` 连续写到两个文件；文件清理名单也同时包括两名 [`_OUTPUT_FILENAMES`](../latency_bench/logging/trajectory_logger.py#L10-L20)。`infer_latencies.jsonl` 在 [`docs/realtime_pipeline_logging.md`](../docs/realtime_pipeline_logging.md#L6-L13) 被描述为 inference request latency；`latencies.jsonl` 有真实内部分析消费者 [`LATENCY_RECORD_CANDIDATES`](../latency_bench/analysis/eval_stats.py#L16-L16)。
- **当前消费者 / 测试所有者：** 多个 executor/eval tests 用 `latencies.jsonl`；logging tests 读取输出 [`test_trajectory_logger_writes_inference_and_e2e_latency_traces()`](../tests/logging/test_logging_records.py#L229-L271) 并逐字比较 incremental/reference output [`test_trajectory_logger_incremental_flush_matches_one_shot_reference()`](../tests/logging/test_logging_records.py#L403-L441)；runtime logging 文档使用 `infer_latencies.jsonl`。现存结果/外部 notebook 是否依赖任一名称未在仓库搜索中完全可见。
- **已证事实与历史：** 同一个 list 被 append 到两个文件，可证实两份序列化内容相同。文件名职责是否历史 alias 未有说明。`latencies.jsonl` 与 `infer_latencies.jsonl` 都有现行读者和文档/测试，不能按 dead output 处理。
- **价值与最小动作：** 文档明确两文件的当前关系、保留理由和 canonical 选择；如果未来决定只保留一份，先做仓库全量读者检索以及 experiment artifact / 下游消费者确认，再迁移读取方和相关文档/测试。
- **需保留 / 风险：** 删除任一文件会影响已存在的 scripts、analysis、tests 或外部结果分析；当前实现并非“无消费者的重复写文件”。不要在重构中顺手省略。
- **验收与依赖：** 若只补文档，确认它说明当前输出相同且分别记录 reader expectations；若未来删 alias，需所有已知消费者迁移并验证历史文件仍可被分析工具读取。该工作不应改变单行字段 schema。
- **必须讨论：** 将来是否允许 breaking artifact rename；由谁负责处理已归档结果、外部分析脚本和对外文档。

<a id="rf-010"></a>

### RF-010 — single 与 batched executor factory 只在部分 assembly 上重复

- [ ] **状态：候选**
- **动作层级**：Consolidate。
- **优先级 / 置信度 / 变更类型：** P2 / 中 / 只合并可证明相同的 preparation。
- **分类：** C04、C05、C11。
- **证据：** single path 使用 [`_build_executor_runtime_components()`](../latency_bench/executors/factory.py#L115-L132)，然后在 [`_build_single_simulated_executor()`](../latency_bench/executors/factory.py#L184-L227) 组装 inference pool/policy、episode sampler、logger；batched path 在 [`_build_batched_simulated_executor()`](../latency_bench/executors/factory.py#L230-L310) 再次创建 `ActionResolver`、clock、inference pool 或 policy、logger、sampler factory 与 action schedulers。明显相同片段包括 inference-pool policy incompatibility 分支和 device resolution；batched sampler 初始化必须保留按 slot 派生 seed 的逻辑。
- **当前消费者 / 测试所有者：** 全部由 [`build_executor()`](../latency_bench/executors/factory.py#L47-L85) 按模式/并行数路由；factory matrix 由 [`tests/executors/test_executor_factory.py`](../tests/executors/test_executor_factory.py) 所有，包含 injected env/device 测试 [`:744-755`](../tests/executors/test_executor_factory.py#L744-L755)。
- **已证事实与历史：** 相同 pool/policy 分支、clock 与 logger construction 被复制可证实；当前 helper 已把 realtime/single 共用的小组件抽出。为何 batched 未纳入 helper 未有证据。`docs/infra.md` 说明 single/batched/realtime 的 loop 和 resource ownership 是刻意不同的，因此“把两个 executor factory 合成一条路径”不是结论。
- **价值与最小动作：** 在计划实现前并排标出逐字段相同和 slot-specific 部分；只考虑抽出无 episode/slot 语义的重复工厂片段（例如 pool device selection），让两种组装仍各自负责单一对象与 per-slot 对象。不新建总控 framework，不把专属 seed/sampler 行为藏进布尔参数。
- **需保留 / 风险：** batched 路径必须继续创建一套 scheduler/latency source per slot，派生 vector seed，并保留 env-backend resource cleanup；single 允许注入单 env。错误复用会导致不同 slot 随机流相同或 backend 未关闭。
- **验收与依赖：** 对 executor factory 测试矩阵核对 paused/simulated/realtime、single/batched、injected env/backend、policy injected、inference pool on/off、seeded IID/Temporal sampler 和 logger config。执行器对象类型、slot count、worker/device 列表、sampler seed 结果与 cleanup 都保持原行为。只在确有重复行可共同维护时做小步合并。
- **必须讨论：** 要抽取哪些 preparation 字段、是否保留私有 helper；不能以统一抽象为由合并单集/批量 scheduling loop。性能影响只在确实挪动执行热路径时测量。

## Policy、数据与训练

<a id="rf-011"></a>

### RF-011 — 按边界拆分通用 StarVLA 适配器

- [ ] **状态：候选**
- **动作层级**：Redesign（现有职责整理）。
- **优先级 / 置信度 / 改动类型 / 分类：** P2 / 高 / 按职责拆分模块、保留现有入口 / C05 抽象不匹配；C12 API 泄漏实现细节。
- **证据与当前消费者：** [starvla.py](../latency_bench/policy/starvla.py) 将 policy runner 和输入构造放在 StarVlaPolicyRunner（L77–395），将 RL-Games 多种输出布局解码放在同一模块（L431–604），随后又实现模型配置兼容、raw-frame 输入栈推导（L616–808）、checkpoint/live wrapper（L811 起）、Gymnasium manifest 合同验证（L913–1030）和两个 builder（L1031–1217）。正式构建入口是 [factory.py](../latency_bench/policy/factory.py) 的 build_policy（L16–30）；resident 推理消费者是 [starvla_rl_games_eval_runner.py](../latency_bench/integrations/starvla_rl_games_eval_runner.py)（L593–602）。
- **测试归属：** [test_starvla_policy.py](../tests/policies/test_starvla_policy.py) 负责一般checkpoint/live-wrapper、图像输入与动作 decode；[test_generic_gymnasium_starvla_policy.py](../tests/policies/test_generic_gymnasium_starvla_policy.py) 负责 Gymnasium manifest、state/action contract 和时钟校准路径。两套测试各自拥有不同外部协议。
- **历史依据：** starvla.py 有明确注释要求图像缩放与第三方训练 _pack_sample 一致（L268–270），docs/infra.md 也说明第三方模型导入保持 lazy（L30、L46–50）；没有文档说明这些独立职责必须聚合在一个文件中。文件职责分布及变动风险是本次审计推断。
- **价值与最小建议：** 让新改动能落在输入合同、模型包装、任务动作 decode 或 manifest 校验的单一边界；先把纯动作 decode 与模型配置/输入计划移到有真实调用者的内部模块。保留 factory 和 resident evaluator 实际调用的 build_starvla_policy、build_live_starvla_policy；重排后迁移本仓调用点，不为旧文件路径或仅供测试的helper额外造re-export facade。
- **风险与保持项：** 拆分时保持 import 延迟、resident 模型不重载、动作输出布局、profile 计时名、manifest 比较和训练图像预处理一致。不得把连续 action chunk 的 task adapter 合并进离散/连续 RL-Games head decoder。
- **验收与依赖：** 逐项运行上述两个 policy suites 及 resident eval caller 检查；同一输入在重构前后产生相同 example、Action 值、Action 元数据和输入帧数。只整理文件边界，不改变第三方接口。

<a id="rf-012"></a>

### RF-012 — 合并 StarVLA 模型输入栈的重复解析

- [ ] **状态：候选**
- **动作层级**：Consolidate。
- **优先级 / 置信度 / 改动类型 / 分类：** P1 / 高 / 抽取小型输入计划 resolver / C04 重复实现；C09 训练、评测、部署漂移。
- **证据与当前消费者：** [starvla.py](../latency_bench/policy/starvla.py) 的 runner 构造器在 L134–156 解析 pack_image_sequence、image_sequence_length、num_obs_frames、image_mode、stitch_grid 和 kv_memory.enabled，推理侧随后在 L225–248 选帧；apply_starvla_model_input_config 又在 L700–739 重新解析 packed sequence、KV memory、image mode 和所需原始帧数，再写回 env.frame_stack。pre-env调用者包括 [driver.py](../latency_bench/eval/driver.py)（L48–56）和 StarVLA RL-Games runner（starvla_rl_games_eval_runner.py L287）。
- **测试归属：** [test_starvla_policy.py](../tests/policies/test_starvla_policy.py) 维护输入栈和模型输入测试；现有 packed 序列用例在 L949–982，另有 multiframe、stitch、KV memory 用例（约 L585–883、L1020 起）。
- **历史依据：** 源码注释明确要匹配 StarVLA 训练的 _pack_sample（starvla.py L246–269）；env frame stack 必须在 env 初始化前准备、帧选择发生在实际 policy 推理时，这两个阶段分开有真实生命周期原因。重复分支是后来扩展不同输入模式的推断，未发现要独立维护两套模型语义的文档决策。
- **价值与最小建议：** 建一个纯函数，基于已归一化 model config 返回必要输入帧数/帧选择模式；env 初始化层据此提高 frame_stack，runner据此选帧。图像变换、stitch与PIL变换仍由调用层执行。
- **风险与保持项：** 不能把 temporal frame、packed sequence、KV memory 与 stitch 统称为“多帧”：packed按 raw stride采样；KV memory只喂当前帧；generic StarVlaPolicyRunner先选择时间窗口，再由 image_mode=stitch 把这些时间帧拼成一张图（starvla.py L225–264）；MIKASA task adapter 的 stitch_views 才把同时采集的多个camera view合为一张图（policy/starvla_tasks.py L31–42）。两种stitch目的不同。非 raw transform 不一定需要增加原始 frame stack。保留 (image_sequence_length - 1) * raw_stride + 1 的时间范围。
- **验收与依赖：** 对照 env stack 和推理 example 在 packed/unpacked、single/multiframe/stitch、KV memory及不同 FPS stride 下逐项一致；保留当前政策测试owner，不改 config wire fields。

<a id="rf-013"></a>

### RF-013 — 让 policy action label 引用 canonical label 表

- [x] **状态：已完成**
- **动作层级**：Consolidate。
- **优先级 / 置信度 / 改动类型 / 分类：** P2 / 高 / 合并相同常量，不合并布局语义 / C04 重复实现；C03 概念或语义漂移。
- **证据与当前消费者：** [action_layouts.py](../latency_bench/data/action_layouts.py) L26–31 定义 Demon Attack 与 Deadly Corridor 分量标签，L95–124 用于 canonical rollout ActionSpec。相同字符串又在 [starvla.py](../latency_bench/policy/starvla.py) L48–74 定义，并由 L563–603 解码模型动作名称。前者写数据行、action id、label metadata；后者把模型动作头变成 evaluator Action。
- **测试归属：** [test_rollout_action_layouts.py](../tests/data/test_rollout_action_layouts.py) 锁 canonical联合id与文本；[test_deadly_corridor_starvla_converter.py](../tests/policies/test_deadly_corridor_starvla_converter.py) 锁模型11维因子头解码到环境tuple；两者不应合成一个只测labels的测试。
- **历史依据：** 两处常量值相同，且分别由数据管道和模型 adapter 消费。没有找到重复的历史原因说明；由职责分层和独立演进推断产生。
- **价值与最小建议：** StarVLA decoder import canonical分量标签，减少显示名称或 component 顺序各自漂移的入口。只复用 labels，不把数据布局编码整体移入policy。
- **风险与保持项：** Dataset 默认 Deadly joint_54、policy head的 semantic_7 / factorized_11 / joint_54 是不同wire布局；DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER 是第三方multibinary训练顺序，也与四元tuple不同，均须保留。
- **验收与依赖：** 对照两组现有测试逐项核 action values、ids、label strings、noop component处理；这属于常量所有权重排，不应改 action protocol。
- **首批实施与验收（2026-09-30）：** 五组标签由 [`data.action_layouts`](../latency_bench/data/action_layouts.py#L27) 单独定义，StarVLA 用私有导入别名在 [`Demon decoder`](../latency_bench/policy/starvla.py#L542) 和 [`Deadly decoder`](../latency_bench/policy/starvla.py#L552) 查表。移除重复 literal；数据布局和三个模型输出布局仍分开。rollout layout、converter 及 StarVLA policy 既有行为用例通过，动作值、ID、名称与 metadata 合同保留。 主证据：[首批报告](../outputs/refactor/batch-01/report.md)，整体结果与边界见[批次记录](#batch-01)。


<a id="rf-014"></a>

### RF-014 — 共用状态归一化数值核心

- [ ] **状态：待核实（数值合同）**
- **动作层级**：Consolidate。
- **优先级 / 置信度 / 改动类型 / 分类：** P2 / 高（重复公式）、待核实（数值合同） / 合并数值公式，保留各自数据合同 / C04 重复实现；C09 训练、评测、部署漂移。
- **证据与当前消费者：** [state_normalization.py](../latency_bench/data/state_normalization.py) L6–18 把状态按 min/max 缩放到 [-1,1]，相等范围变0并裁剪；[starvla_tasks.py](../latency_bench/data/starvla_tasks.py) L253–262 又按 stats["min"/"max"] 实现状态/action normalize；L265–269另有denormalize反变换，它不执行[-1,1]裁剪，不属于重复normalize公式。前者被 [starvla.py](../latency_bench/policy/starvla.py) L292–299 与 [starvla_lerobot.py](../latency_bench/data/exporters/starvla_lerobot.py) L835–841 消费；后者被 [policy/starvla_tasks.py](../latency_bench/policy/starvla_tasks.py) L27–30、L44–47、L58–59 消费。
- **测试归属：** [test_state_normalization.py](../tests/data/test_state_normalization.py) 管独立状态归一化；[test_starvla_tasks.py](../tests/data/test_starvla_tasks.py) L143–181 管合同stats、action往返和边界/常量维。
- **历史依据：** 两套数据协议分别是 generic state manifest的 {min,max} 和任务task_contract的 stats mapping；分开的外层接口有实际消费者，但数值核心重复。未找到说明需要不同公式的历史决策。
- **价值与最小建议：** 共用仅负责数组范围映射的最小数值函数，由两侧各自读取合同并传入min/max；task合同的normalize(stats)继续适配这个数值函数，denormalize(stats)保留为独立逆变换。不要让 task action stats schema泄漏到 generic manifest。
- **风险与保持项：** 保留float32计算、常量维为0、裁剪到[-1,1]与支持前导batch维；不要把task的prefix/action归一化流程换成state专用接口。
- **验收与依赖：** 在两种合同、常量维、边界值、越界值及batch数组下比较旧输出；现有两个数据测试是验证owner。改行为前需先证明差异不存在或经讨论接受。
- **首批决定（2026-09-30）：** 后移、不改公式。generic 先 divide 后乘 2，task 先乘 2 后 divide；float32 的 `values=[2e38], min=[0], max=[3e38]` 可分别得到约 0.33333325 和 1.0，不能视为严格等价的机械合并。常规随机 batch probe 未观察到差异，但没有据此证明生产动作不受影响；后续先明确所需数值合同，不添加 mode flag 或 guard 来掩盖差异。


<a id="rf-015"></a>

### RF-015 — 明确 mapless latency prompt 的时钟来源

- [ ] **状态：待核实**
- **动作层级**：Document / Consolidate（生产路径核实后）。
- **优先级 / 置信度 / 改动类型 / 分类：** P1 / 中 / 将prompt timing变成显式resolved输入，先核实命中路径 / C03 概念漂移；C07 配置债务；C09 训练、评测、部署漂移；C10 时间逻辑与单位。
- **证据与当前消费者：** 数据prompt renderer [prompts.py](../latency_bench/data/prompts.py) L26–60 把 latency raw frame/ms、env FPS、observation FPS都作为显式输入。policy resolver [starvla_prompts.py](../latency_bench/policy/starvla_prompts.py) L34–53 有map、metadata动态渲染和base prompt三条分支；动态分支在L45–48把缺失ms设0、缺失两种FPS设35。StarVlaPolicyRunner._build_example 在 [starvla.py](../latency_bench/policy/starvla.py) L250–256 调用它，builder有env_cfg但L1059–1082未把FPS传入runner。标准env producer例如 [gymnasium.py](../latency_bench/envs/gymnasium.py) L116–128通过_metadata带状态；evaluation raw FPS则写进episode metrics的 [driver.py](../latency_bench/eval/driver.py) L190–219，并非由这段代码并入每个policy observation。实际是否存在mapless且带latency_raw_frames的正式caller尚未确认。
- **测试归属：** prompt路径测试在 [test_starvla_policy.py](../tests/policies/test_starvla_policy.py)（resolver测试约L1288）；Gymnasium合同测试在 [test_generic_gymnasium_starvla_policy.py](../tests/policies/test_generic_gymnasium_starvla_policy.py)。当前搜索未发现测试锁定mapless prompt fallback必须显示35/35。
- **历史依据：** 35 FPS默认值的来源没有代码文档；常见Flappy/VizDoom任务可解释其来源只是推断。数据生成要求明确FPS，显示这是跨路线合同可能不一致，而非已证明的用户可见错误。
- **价值与最小建议：** 先查清此dynamic branch的正式调用者；若确有使用，让评测配置resolve阶段把物理时钟、延迟frame和ms作为一个明确prompt context传入，移除来源不明的35/35归一化。保留map选择、raw/latency_neutral和纯base提示。
- **风险与保持项：** 当前branch只有metadata中确有latency frame且未选map才运行，影响不能写成已复现缺陷。错误改动会让prompt不再匹配训练map、数据label或校准后的env clock。
- **验收与依赖：** 先确认一个生产caller和所用FPS/latency协议，再选API。至少比较训练prompt、map hit、mapless运行时latency、neutral/base模式；未确认caller前不改默认行为。

<a id="rf-016"></a>

### RF-016 — 切开StarVLA evaluation对training CLI的依赖

- [ ] **状态：候选**
- **动作层级**：Consolidate。
- **优先级 / 置信度 / 改动类型 / 分类：** P2 / 高 / 将evaluation可复用部分移到中立模块，不搬训练算法 / C06 隐式状态/隐藏耦合；C12 API泄漏实现细节；C07 配置债务。
- **证据与当前消费者：** [evaluate.py](../scripts/starvla/evaluate.py) L13–16 直接从 scripts.starvla.train_task 导入 TASK_SETTINGS、environment、evaluation_command、run_command，L95–105再从训练表拼evaluation policy。训练表本身在 [train_task.py](../scripts/starvla/train_task.py) L45–65 同时放teacher、conda env、task config、state key、batch size、SFT steps、episode cap和seed；数据合同的name/dim/horizon/fps则在 [starvla_tasks.py](../latency_bench/data/starvla_tasks.py) L35–104。表中的训练超参与数据合同并非完全重复，不应粗暴合成一个更大的全局registry。
- **测试归属：** [test_starvla_train_task.py](../tests/scripts/test_starvla_train_task.py) L383–390检查evaluate写出的policy state/image key；[test_mikasa_h8.py](../tests/scripts/test_mikasa_h8.py) L1021–1055覆盖train_task.eval_config的20Hz/profile contract。
- **历史依据：** 没有发现当前文档说明evaluation必须依赖training CLI模块；evaluate.py反向复用launch与env helper是源码可见事实，模块耦合为结构推断。
- **价值与最小建议：** 把共享的环境准备、命令构造和evaluation可用状态/image key映射放进轻量runtime helper；evaluate.py和train_task.py都调用它。训练teacher路径、batch和step设置留在训练owner；不要把这些值塞进StarVLATaskSpec。
- **风险与保持项：** task alias（H1/H8/conditioned H8）可能共享teacher和config但不共享state_dim/action_horizon；必须从task_spec与task contract读取数据维度，不能被runtime setting alias覆盖。
- **验收与依赖：** 针对balance、Mikasa H1、H8、conditioned H8对比生成的env/policy/eval config、seed和launcher命令；保持两个既有测试owner。

<a id="rf-017"></a>

### RF-017 — 将teacher rollout的大型双阶段collector恢复成清晰编排边界

- [ ] **状态：候选**
- **动作层级**：Redesign（现有编排边界整理）。
- **优先级 / 置信度 / 改动类型 / 分类：** P2 / 高 / 将probe/replay和metadata收尾分离成内部模块，保留collector入口 / C05 抽象不匹配；C06 隐式状态/隐藏耦合；C08 数据与实验协议债务。
- **证据与当前消费者：** [sf_teacher_rollout.py](../latency_bench/data/sf_teacher_rollout.py) 当前1532行，主接口 collect_teacher_dataset 在L1296–1532同时恢复probe/shard状态（L1350起）、probe only（L1393–1415）、容量/种子相关replay（L1432起）及最终filter/protocol metadata（L1484–1532）。本文件已有内部阶段_probe_accepted_specs（L598–850）、_replay_accepted_specs（L853起）、_build_rollout_driver_runtime（L1198–1293）；相邻职责已经分到 sf_teacher_config.py、sf_teacher_runtime.py、sf_teacher_drivers.py、sf_teacher_rows.py、sf_teacher_export.py。用户入口是 [sf_teacher_export.py](../latency_bench/data/sf_teacher_export.py) L85–205和CLI [export_sf_teacher_dataset.py](../scripts/rollout_data/export_sf_teacher_dataset.py)。
- **测试归属：** [test_sf_teacher_dataset_export.py](../tests/data/test_sf_teacher_dataset_export.py) 是采集owner（3173行），包含正常采集、probe拒绝重试、无RGB probe后同seed replay、shard resume、probe-only等场景；分别见测试名在L557、L659、L774、L1909、L3028附近。单独rollout driver contract已有独立模块测试。
- **历史依据：** 源码docstring和probe/replay命名明确记载无RGB筛选与accepted seed重放；probe_state与shard resume是已发布导出协议。当前模块仍大，说明已有拆分未覆盖collector transaction收尾，这是结构观察，不表示双阶段流程是遗留路线。
- **价值与最小建议：** 把probe状态存取/合并与最终metadata报告分别放在已有data职责周边，留下一个小型collect coordinator调用probe→replay并管理资源；不要把所有task/GPU差异抽成泛型插件。
- **风险与保持项：** 保持episode accepted specs及seed顺序、无RGB probe、重放时相同行动、已有shard跳过、attempt预算、episode filter、probe-only输出、原始frame/FPS/checkpoint metadata。改变probe/replay规则会改变最终训练数据集。
- **验收与依赖：** 不做算法变化时，以前后相同seed/config比较episode决策/action/return、rows/metadata/shard索引和resume输出；沿用test_sf_teacher_dataset_export.py为workflow所有者。需先完成模块所有权图再拆，避免迁移期间双重职责。

<a id="rf-018"></a>

### RF-018 — 整理checkpoint文件内部边界，不改变保留生命周期

- [ ] **状态：候选**
- **动作层级**：Redesign（文件职责整理）。
- **优先级 / 置信度 / 改动类型 / 分类：** P2 / 高 / 按checkpoint用途拆内部模块或分区，保留现有owner合同 / C05 抽象不匹配；C11 状态/资源所有权。
- **证据与当前消费者：** [checkpoints.py](../training/common/checkpoints.py) L22–114管理原子本地写、RNG和目录/命名；L123–217管理SF读、Learner patch与候选读取；L220–298管理evaluation lease/retention marker和checkpoint pruning。[sample_factory/runtime.py](../latency_bench/integrations/sample_factory/runtime.py) L32–52调用load；[latency_bench_periodic_eval.py](../training/common/latency_bench_periodic_eval.py) L20–30、L100–118使用patch/lease；训练checkpoint与eval并发清理共享相同目录。
- **测试归属：** [test_training_checkpoint_layout.py](../tests/training/test_training_checkpoint_layout.py)负责本地路径、atomic写和RNG；[test_sample_factory_checkpoint_io.py](../tests/training/test_sample_factory_checkpoint_io.py)负责SF patch、lease相关加载保护和并发prune边界。
- **历史依据：** [infra.md](../docs/infra.md) L54–68明确SF版本固定，checkpoints.py拥有atomic write及eval输入保留期；这项生命周期已有文档决策。单文件包含几类实现的历史原因没有明确记载，拆分价值是结构判断。
- **价值与最小建议：** 如果未来触及checkpoint功能，可把本地通用文件命名/RNG与SF learner patch、eval lease/prune分入各自owner模块，并同步迁移现有training/runtime调用点；不为保留旧模块路径而加只有转发功能的facade，也不复制marker解析或拆散prune/lease的竞态边界。
- **风险与保持项：** marker目录/文件名、lease生成与释放、best保留、learner保存原子性、checkpoint读取时机均是资源生命周期合同；docs/infra.md明确说仅代码清理不足以删除patch或改安装时序。
- **验收与依赖：** 按既有两个tests suites覆盖native checkpoint/rng、SF resume、evaluation并发读时prune、成功/异常cleanup；文件组织变动不得改变最终选中checkpoint路径或保护窗口。

<a id="rf-019"></a>

### RF-019 — 先决定两种SF checkpoint loader的坏文件合同

- [ ] **状态：待讨论**
- **动作层级**：Document → Consolidate（失败策略讨论后）。
- **优先级 / 置信度 / 改动类型 / 分类：** P1 / 高（行为差异已确认，哪种语义该统一未决） / 先决策，再统一或明确命名 / C08 checkpoint/实验协议债务；C13 错误处理与静默失败。
- **证据与当前消费者：** [checkpoints.py](../training/common/checkpoints.py) 的SF monkeypatch在L147–160只取checkpoints[-1]，同一路径重试3次后返回None；同文件L202–217的eval helper则从最近到最旧候选逐一尝试，找到可读checkpoint即返回实际路径，全部失败时raise。[runtime.py](../latency_bench/integrations/sample_factory/runtime.py) L37–52走后者；patch接管Sample Factory learner恢复流程。该后续调用已按本仓固定pin ad256a6312db1ff46b19f539081abdbdf20fc424复核：[learner.py](../third_party/sample-factory/sample_factory/algo/learning/learner.py) L306–316在load_checkpoint返回None时会记录“Did not load from checkpoint, starting from scratch”，随后继续训练。
- **测试归属：** [test_sample_factory_checkpoint_io.py](../tests/training/test_sample_factory_checkpoint_io.py) 约L160–220覆盖patch幂等/取最后checkpoint和eval runtime选择kind的调用；当前用例未证明“最新坏而较旧可读”的两条路径必须相同。此处是协议待定，不是假定测试缺陷。
- **历史依据：** docs/infra.md记录训练checkpoint与periodic eval的保留生命周期，但没有说明训练resume允许向旧checkpoint回退或eval为何会回退。training resume与eval artifact identity不同是可能的理由，属于推断，需由用户/维护者确认。Sample Factory pin明确将None解释为从头开始；不能据此推断本仓的resume配置希望接受静默冷启动。
- **价值与最小建议：** 先明确坏文件时resume的允许结果：停在最新并报告失败、重试、或回退并记录恢复路径；eval选择则需记录真正加载的文件。之后可共用读取原语，但策略参数要显式表达，不能无声地把两个入口的fallback合并。
- **讨论点：** 这项会决定是否改变恢复状态、训练连续性和评测artifact身份；实施前要选“训练与eval同策略”或“训练最新严格、eval回退并记录”。
- **风险与不变量：** 保护resume所选checkpoint内的模型、optimizer、train_step/env_steps与RNG连续性；不可把损坏/缺失的resume静默变成冷启动。当前上游pin会在loader返回None后继续从头训练，故需要在改策略前审阅这个下游效果。也要保持eval最终记录实际加载路径，不能让“最近有效版本”与正式artifact id脱钩。
- **验收与依赖**：先确定训练 resume 与自动 eval 选取各自的失败合同，再在既有 checkpoint IO suite 验证最新可读、最新损坏但旧版可读、全部损坏和显式路径加载；对照实际加载路径、模型/optimizer/步数/RNG 与 learner 的后续行为。关联 RF-018 的 retention/lease，不能让所选旧版在加载期间被清理。没有本次运行通过的声明。

<a id="rf-020"></a>

### RF-020 — 明确filter kwargs是否是长期公开接口

- [ ] **状态：待讨论**
- **动作层级**：Document / Consolidate（现有 filter 合同讨论后）。
- **优先级 / 置信度 / 改动类型 / 分类：** P2 / 高（公开透传和CLI消费可见，后续API形状待决） / 把扩展映射收敛为明确的episode acceptance配置，或正式保留映射 / C07 配置债务；C12 API泄漏实现细节。
- **证据与当前消费者：** collector [sf_teacher_rollout.py](../latency_bench/data/sf_teacher_rollout.py) L1311–1318公开filter_kwargs，自身docstring在L1335–1337称其为“Temporary exporter experiment arguments retained at the public boundary”；参数向_probe_accepted_specs L613/L635和filter dispatcher L748–754传递。exporter [sf_teacher_export.py](../latency_bench/data/sf_teacher_export.py) L110–120也暴露同名mapping。CLI [export_sf_teacher_dataset.py](../scripts/rollout_data/export_sf_teacher_dataset.py) L90–100给用户--demon-episode-threshold，L143–177再把它包装成{"demon_episode_threshold": ...}；filter合同 [filters.py](../latency_bench/data/filters.py) L51–71其实有具名的demon_episode_threshold参数。
- **测试归属：** [test_rollout_filters.py](../tests/data/test_rollout_filters.py) L30–43锁默认Demon raw score门槛；[test_sf_teacher_dataset_export.py](../tests/data/test_sf_teacher_dataset_export.py) L732–771锁episode_return_gt筛选和元数据（这是另一项显式门槛，不等同kwargs）。当前定向搜索未找到CLI自定义Demon threshold的行为测试。
- **历史依据：** 代码docstring明确写temporary；但同一项已经到达用户CLI并由probe接受过滤调用，故不能仅按注释认作dead path。具体扩展映射曾用于哪些历史实验未从当前材料确认。
- **价值与最小建议：** 默认建议把当前唯一的Demon threshold覆盖值作为具名参数沿collector/exporter链传递，保留CLI开关及同一接受episode集合；不要为一个现存字段新增EpisodeAcceptance schema。若用户确认其他过滤preset也需要任意可扩展项，再讨论显式的通用合同；不要以清理temporary注释为由删除threshold功能。
- **讨论点：** 确认现有CLI展示的Demon阈值是否就是长期支持的配置面，或是否另有真实消费者需要mapping的通用透传能力。若无此消费者，先具名传当前参数；不要仅为了未来可能的preset扩展保留泛化接口。
- **验收与依赖：** 自定义门槛应影响probe acceptance、attempt/reject统计及filter metadata，并在同一阈值下复现相同accepted seeds。决策后由rollout filter与dataset export suites负责验证；本轮未运行它们。
- **风险与保留行为**：参数透传形式变化不能改变 Demon 自定义门槛、accepted seed 集合、attempt/reject 统计或写入的 filter metadata；也不能把 kwargs 的其他真实字段静默丢弃。先搜索所有公开调用方，再决定具名参数范围。与 RF-017 的 collector 整理保持独立，避免一次同时改变过滤和 probe/replay。

## Cross-device、评测与配置

<a id="rf-021"></a>

### RF-021 — 统一 Flappy 预算实验的 55-run 总体定义

- [ ] **状态：候选**
- **动作层级**：Consolidate。
- **分类：** C08 实验协议债务、C14 可复现性
- **优先级 / 置信度 / 变更类型：** P1 / 高 / 收敛重复协议常量，不改实验协议。
- **现状与证据：** [budget.py](../latency_bench/cross_device/budget.py#L51) 的 split 构造以 0–54 为全集，原始 split 固定验证 5–9、拟合顺序 0–4 后接 10–54；[measure_real](../latency_bench/cross_device/budget.py#L129) 再独立遍历 55 个 run，并用 run_id 小于 10 区分导入的原始数据与新采集的 10–54。[budget_report.py](../latency_bench/cross_device/budget_report.py#L44) 的每 run 延迟诊断也遍历 55 个目录并按小于 10 标为旧采集；[budget_repeats.py](../latency_bench/cross_device/budget_repeats.py#L25) 又用 range(55) 生成五次随机划分。budget.json 已保存 split seed、完整 split 列表、fit sizes、episode 数和方法（[budget.py](../latency_bench/cross_device/budget.py#L81)），所以协议输入有记录，但实际遍历/来源边界仍在几处各自编码。
- **当前消费者与测试归属：** 入口为 [run_budget.sh](../scripts/experiments/sim2real/run_budget.sh#L1) 调用 latency_bench.cross_device.budget；报告和 repeats 分别由 budget_report、budget_repeats 入口消费。现有归属测试是 [test_sim2real.py](../tests/cross_device/test_sim2real.py#L364) 的 test_budget_split_sizes_and_total_replay_budget、:390 的 test_budget_real_resume_and_fit_data_isolation、:470 的 test_budget_report_aggregates_equal_case_weight_and_exports、:592 的 test_random_repeats_preserve_first_split_and_average_absolute_errors。
- **历史事实与推断：** 这些预算模块在本次工作树中未跟踪，但上述脚本、正式预算协议和测试都引用它们；未跟踪不代表失效。该预算实验模块当前没有可用提交历史，本项不推断其形成原因。
- **为什么值得：** 未来若采集总数或旧/新数据边界调整，只改一处而漏掉诊断或 repeats，会造成 profile 训练集、验证集、run 级指标和 GPU 成本使用不同run 总体。
- **最小建议：** 将全集及导入边界命名为当前协议常量，供 split、采集、延迟诊断和 repeats 共用；或由既有冻结 manifest 推导。优先用代码常量收敛，不必给已冻结的 budget.json 增字段，也不增加任意 run 数配置。
- **风险与必须保留的契约：** 保持 55 个 run、原始数据 0–9、新采集 10–54、原始 validation 5–9、固定随机种子 20260929、原有 fit 顺序、两个模型共用 split、source revision 和 reference return。禁止把原始 validation 的 run 混入其自身 profile 拟合，也禁止用 episode 级重排替代 run 级划分。
- **依赖：** 依赖现有 budget protocol 常量与 freeze/resume 语义；不要为了收敛引入新的 manifest 版本或恢复兼容分支。
- **后续验收：** 复用上述四个 cross-device 测试；对相同 frozen split 比较 fit/valid ID、导入目录、新采集数、延迟诊断的旧/新标签和报告成本归属。若拟改 manifest wire shape 或历史 campaign resume 格式，先讨论兼容范围。

<a id="rf-022"></a>

### RF-022 — 让两 episode session 宽度只有一个内部定义

- [ ] **状态：候选**
- **动作层级**：Consolidate。
- **分类：** C04 重复实现、C08 实验协议债务、C11 状态/资源所有权
- **优先级 / 置信度 / 变更类型：** P1 / 高 / 合并固定协议宽度；不开放新的泛化参数。
- **现状与证据：** [study.py](../latency_bench/cross_device/study.py#L35) 定义 EPISODES=2 和 ENV_SEED_STRIDE=20；[grouped_replay.py](../latency_bench/cross_device/grouped_replay.py#L9) 在 grouped_inputs 直接对两个 episode 建组，RunTrajectoryLogger 又在收到两条 episode metrics 后写 DONE 并调用完成回调（[同文件](../latency_bench/cross_device/grouped_replay.py#L28)）。[study_parallel.py](../latency_bench/cross_device/study_parallel.py#L21) 以 2*parallel_runs 设 eval episode 数；[budget.py](../latency_bench/cross_device/budget.py#L83) 把 episodes_per_run 记录为 2，simulate 另以 2*parallel_runs 构造执行配置（[同文件](../latency_bench/cross_device/budget.py#L244)）。
- **当前消费者与测试归属：** study 的串行/并行 replay 和 budget 的 grouped replay 都走此两 episode 语义。已有 [test_sim2real.py](../tests/cross_device/test_sim2real.py#L364) 与 :547 覆盖 replay 预算/恢复；[test_batched_simulated_executor.py](../tests/executors/test_batched_simulated_executor.py#L1422) 的 test_grouped_runs_keep_session_streams_and_seeds_when_slots_change 覆盖 session sampler 和 seed；:1480 的 test_ten_run_slots_keep_repeated_run_ids_separate_across_settings 覆盖相同 run id 跨 setting 的身份映射。
- **历史事实与推断：** 两 episode 不是从名称推断的旧实现；当前协议常量、预算 manifest 和测试都明确固定为 2。grouped_replay 文件当前未跟踪但被两个生产路径和 executor 测试调用。
- **为什么值得：** 现在 width=2、stride=20、logger flush 阈值和 eval episode 总数由相关代码各自表达，后续调整协议时存在“组宽变了而 DONE 条件/seed block 没变”的可能。
- **最小建议：** 在现有协议所有者处引用同一个内部两 episode 常量；分组、完成写盘阈值与两条 replay 配置统一读取它。该常量仍固定为 2，不能为将来方便而暴露可变宽度 API。
- **风险与必须保留的契约：** 保持 logical episode ID = 20*run_id + episode_in_run、环境 seed = 2,000,000 + episode_id、Temporal 在同一 run 两 episode 中的连续状态、下一 Temporal run 的 sampler 隔离，以及 seeded IID/Normal 已有的 per-episode 随机流、仅两份 metrics 和 trace 完整写入后才落 DONE。Temporal replica-zero seed、episode action-noise seed 也必须保持。
- **依赖：** 依赖固定 EPISODES=2 与当前 ENV_SEED_STRIDE=20 的协议所有者；不新增任意 session width 配置。
- **后续验收：** 复用 grouped session 与 ten-slot 两个 executor 测试，以及 test_budget_split_sizes_and_total_replay_budget。检查 slots=1/2/4/8/10 时 run 顺序、logical IDs、seed、latency source 生命周期、DONE 写入时机和 run publication 位置相同。不要仅比较平均 return。

<a id="rf-023"></a>

### RF-023 — 收敛 generic profile 与 sim2real 的重复 runtime overlay

- [ ] **状态：候选**
- **动作层级**：Consolidate。
- **分类：** C04 重复实现、C03 概念/语义漂移、C09 训练/评测/部署漂移、C10 时间单位
- **优先级 / 置信度 / 变更类型：** P2 / 重复事实高、可共享范围中 / 只抽取确实相同的 overlay。
- **现状与证据：** [study.run_config](../latency_bench/cross_device/study.py#L44) 从 archived base 派生配置，重写 env_fps 与保留 stride 的 obs_fps，设 inference device、realtime/simulated mode、按 method 写 latency 参数、episode seed/episodes、输出路径和 raw trace 开关。[runtime_configs.simulation_config](../latency_bench/cross_device/runtime_configs.py#L36) 为 generic profile replay 也做 FPS/obs stride、simulated mode、profile latency、episode count 和 trace 开关，但还写 deployment profile coordinates、slot capacity/devices，并使用通用 profile 的并行环境配置。两边都活跃：sim2real 串行路径在 [study.py](../latency_bench/cross_device/study.py#L380) 调 run_config；generic profile CLI 经 protocol 到 runner.run_simulation 调 simulation_config。
- **当前消费者与测试归属：** sim2real 的 archived task 路径由 [test_sim2real.py](../tests/cross_device/test_sim2real.py#L86) 的 test_archived_task_contract_survives_clock_calibration 和 :184 的 test_real_upload_resume_and_pinned_cross_host_replay 覆盖；generic profile 由 [test_profile_protocol.py](../tests/cross_device/test_profile_protocol.py#L456) 的 test_simulation_uses_profile_slots_and_selected_artifact 覆盖。
- **历史事实与推断：** 提交 7dd2e15f（2026-09-28）加入 pinned sim2real study，而 generic profile runner 已先存在。两个 builder 在不同工作流中出现的先后是事实；它们刻意不同于工作流的意图属于推断，应通过调用契约确认。
- **为什么值得：** 当前两个执行流程的共同 clock/method/record 写法可以独立漂移，时间单位及 profile 模式变化可能只改一条路径。
- **最小建议：** 先列对照表，只抽取 clock/obs-stride 与 fixed/normal latency field 中语义和读写完全一致的部分；profile_ref/deployment coordinates、profile slot/capacity、measurement realtime、study run seed、archived config 语义留在各自所有者。只在共享的小型纯变换减少总维护面时提取，否则保留两个 builder。不要把两个 builder 改成一个带大量 flags 的万能函数或统一 runner。
- **风险与必须保留的契约：** Study 保留原任务 physics、frame skip、action mapping、history/prompt/normalization/horizon 和 seed；generic profile 保留 profile artifact identity、worker slot、capacity 和 GPU mapping。env FPS 改动仍按原 stride 缩放 obs FPS，不替换训练 profile FPS。
- **依赖：** 依赖逐字段确认 generic profile 与 archived study 的配置差异；未证明相同的字段继续由原 builder 所有。
- **后续验收：** 对 archived 18 task/model configs 逐项验证 run_config 仅改变既有运行坐标；对 generic profile 验证 simulation_config 写出的 profile provenance、slots 和 simulator/inference device 不变。复用上述两个测试，不比较配置文本顺序作为验收目标。

<a id="rf-024"></a>

### RF-024 — 合并 resident replay 的共同准备与资源生命周期

- [ ] **状态：候选**
- **动作层级**：Consolidate。
- **分类：** C04 重复实现、C11 状态/资源所有权、C15 依赖/环境债务
- **优先级 / 置信度 / 变更类型：** P2 / 高 / 小范围提取 executor 准备和关闭边界。
- **现状与证据：** [study_parallel.build_replay_executor](../latency_bench/cross_device/study_parallel.py#L18) 复制 config、应用 slot/batch、解析 evaluation config、准备 StarVLA checkpoint 输入、seed、建 environment backend 和 executor；[budget.simulate](../latency_bench/cross_device/budget.py#L244) 同样为 resident replay 写 config、resolve config、准备 StarVLA 输入、seed、build executor、warmup，并在 finally 关闭（[同文件](../latency_bench/cross_device/budget.py#L261)）。前者为 InterceptGrab 延迟导入 MikasaEnvStepBackend；预算路径当前只运行 Flappy 并用默认 backend。
- **当前消费者与测试归属：** study 在 parallel_runs>1 时通过 [study.py](../latency_bench/cross_device/study.py#L392) 延迟调用 study_parallel；budget 的 sim stage 直接调用 budget.simulate。现有 owners 是 [test_sim2real.py](../tests/cross_device/test_sim2real.py#L184) 的 pinned cross-host replay 参数化案例和 :547 的 test_budget_simulates_fixed_validation_runs_once_and_resumes；真实 GPU parallel gate 在 budget.verify_parallel。
- **历史事实与推断：** study_parallel 和 budget 是当前工作树中的未跟踪模块，但它们被入口、测试和预算 runbook 使用，不能判作废弃。形成两个初始化路径的历史原因未由 Git 可核验。
- **为什么值得：** 同一 resident model 评测生命周期若改变 warmup、seed 初始化或 executor close 顺序，两个长期运行入口会出现不同的可复现性和资源行为。
- **最小建议：** 提取现有共同的配置准备/seed/executor 生命周期；由调用者注入或选择 env backend，不把 Flappy 与 MIKASA 差异塞入更多隐藏 fallback。明确 warmup/close 的所有者，调用方仍拥有自己的 stage timing 和 publication；若需要新增模式 flags 才能复用，则保持两个简单入口。
- **风险与必须保留的契约：** 保留 Mikasa 可选依赖的局部 import、simulator interpreter 与 worker_python 分离、device map、action_noise_seeded 的模型范围、每个 executor 恰好关闭一次。不得因初始化失败被吞掉异常或自动降低 parallelism。
- **依赖：** 依赖明确执行端选择 env backend、模型 worker interpreter 和可选 MIKASA import 的责任边界。
- **后续验收：** 用 pinned replay 的 parallel_runs 参数化测试、预算模拟恢复测试和现有 GPU serial/batch gate。对照 executor warmup 数、输入配置、seeds、action traces、latency traces、close 后 worker/backend 退出。

<a id="rf-025"></a>

### RF-025 — 合并 grouped replay 的 episode 持久化公共路径

- [ ] **状态：候选**
- **动作层级**：Consolidate。
- **分类：** C04 重复实现、C11 状态/资源所有权、C12 API 泄漏实现细节
- **优先级 / 置信度 / 变更类型：** P2 / 高 / 提取窄小公共执行过程，保留两种调度器。
- **现状与证据：** [budget.replay_group](../latency_bench/cross_device/budget.py#L183) 和 [study_parallel.replay_runs](../latency_bench/cross_device/study_parallel.py#L41) 都清理仅传入的不完整目录、绑定 RunTrajectoryLogger、调用 grouped_inputs、把 group/seed/latency sources 传给 executor.run_episodes 并 flush。budget 的 [replay_settings](../latency_bench/cross_device/budget.py#L206) 另将多 setting 的相同 run IDs 映射到内部 slot ID，并恢复公开 run identity。
- **当前消费者与测试归属：** replay_group 是 budget 单 setting 的调用点；replay_runs 是多 task study 并行路径；replay_settings 是 repeated budget 合并 setting 的路径。测试由 [test_sim2real.py](../tests/cross_device/test_sim2real.py#L547) 与 [test_batched_simulated_executor.py](../tests/executors/test_batched_simulated_executor.py#L1422) / :1480 分别拥有。
- **历史事实与推断：** grouped_replay.py 当前未跟踪，但两种 orchestration 均调用其 logger/sampler，且现有 executor suite 直接导入 logger。无提交历史可证明复制代码的先后或动因。
- **为什么值得：** 运行同一 episode-group contract 的代码维护在两个 wrapper 中，记录开关、flush 和 group 参数传递容易分叉。
- **最小建议：** 提取创建 logger、组织 episode inputs、调用 run_episodes、flush 的内部 helper；让两个调用者提供 run identity/path/publish callback。replay_settings 的 slot-to-public-ID 映射和 queue 填槽调度保留为单独算法，只在提取减少总实现面时共享，不抽象成有许多可选模式的 orchestrator。
- **风险与必须保留的契约：** 仅不完整 run 可清理重跑；DONE 只在 run 内两个 episode metrics 和 raw traces 写完之后建立；publication callback 在 DONE 后按既有目录生效。保留 across-settings 重复 run ID 不冲突，以及Temporal sampler 只跨同一 run 的两 episode，seeded IID/Normal 按既有 episode factory 分配。
- **依赖：** 依赖 RunTrajectoryLogger 与 grouped_inputs 已有接口；调度策略、出版路径和内部 ID 映射仍由各流程持有。
- **后续验收：** 复用 test_grouped_runs_keep_session_streams_and_seeds_when_slots_change、test_ten_run_slots_keep_repeated_run_ids_separate_across_settings、预算模拟恢复测试。核对 raw trace episode ids、metadata run_id/episode_in_run、发布路径和 run completion 次序。

<a id="rf-026"></a>

### RF-026 — 成本汇总从 timing JSON 读语义字段，不解析 artifact 路径

- [ ] **状态：待讨论**
- **动作层级**：Consolidate（producer/report 坐标归属）。
- **分类：** C06 隐式耦合、C08 实验协议债务、C14 可复现性
- **优先级 / 置信度 / 变更类型：** P2 / 高 / 将已知坐标存入 timing record。
- **现状与证据：** [budget_report.make_report](../latency_bench/cross_device/budget_report.py#L106) 从多种 timing 路径收集 stage、seconds、gpu_count，并把 source path 作为记录。随后在 [成本归因](../latency_bench/cross_device/budget_report.py#L117) 通过 stem.split('_')[1] 推断 real run_id，通过 Path.parts[2] 推断 split、parts[3] 推断 fit count。producer 已有坐标却未统一写入：real timing 在 [budget.py](../latency_bench/cross_device/budget.py#L140) 写 run_id；sim timing 创建点位于 split/count/method 循环中，但只写 run_ids、duration、gpu_count、parallel_runs（[同文件](../latency_bench/cross_device/budget.py#L275)）；fit timing 创建点也已知 split/count（[同文件](../latency_bench/cross_device/budget.py#L156)）。
- **当前消费者与测试归属：** 这些 timing 文件由 report 的 costs.csv 和 cumulative_costs.csv 消费。测试 owner 是 [test_sim2real.py](../tests/cross_device/test_sim2real.py#L470) 的 test_budget_report_aggregates_equal_case_weight_and_exports；计划阶段只引用该测试，不在本次运行。
- **历史事实与推断：** 该报告模块当前未跟踪但由 budget report stage 与 SIM2REAL_BUDGET.md 使用；这里仅核对当前 report glob 中消费的 timing，未把 combined-setting 的 `runtime/groups` 记录冒充现有输入；路径依赖是否曾导致错算目前未有证据，因此不声称存在已发生的数据错误。
- **为什么值得：** 当前目录树同时承担存放路径、汇总接口和协议坐标三种角色；迁移/调整路径会改变 GPU hour 的累计归因而无显式 schema 错误。
- **最小建议：** 按已批准的候选方向，由 producer 写入它已知的 split、fit_runs、method、run_ids 等坐标，report 按这些字段聚合；这些新增字段的消费者是当前 costs/cumulative-costs 报告，不新造通用 artifact schema 框架。已有 source path 可继续保留作可读证据与成本 CSV provenance，不必改变本地/HF artifact 目录命名。
- **风险与必须保留的契约：** real 原始 runs 0–9 被重用，其 acquired GPU cost 不应被误计为新工作；55-run 全量收集成本、original prefix equivalent cost 和每 split 累积 replay cost维持现有定义。历史 timing JSON 若需读取，先明确哪些 field 真正缺失并决定旧 report 的处理方式，不添加默认推断掩盖缺项。
- **依赖：** 依赖 timing JSON 可增加语义字段且旧目录/CSV 消费者维持可读；先确认历史 campaign 的旧 timing 是否还需汇总。
- **后续验收：** 复用 budget report 导出测试；对同一 campaign 对比 costs.csv 行数、real cost、original prefix cost 和各 split/count replay cost，逐项确认字段归属与源路径一致。

- **待讨论事项**：历史 timing 格式是否要继续作为可重算成本的输入？先决定保留旧解析、明确迁移旧记录，还是只对新的 campaign 使用新增坐标；旧路径解析若仍是合同，记录并集中维护它，不增加自动猜测或缺字段默认回退。

<a id="rf-027"></a>

### RF-027 — 共用每 case 归一化误差公式，保持统计汇总分开

- [ ] **状态：待讨论**
- **动作层级**：Consolidate 或 Document（净收益讨论后）。
- **分类：** C04 重复实现、C08 实验协议债务
- **优先级 / 置信度 / 变更类型：** P2 / 重复存在高、抽 helper 的净收益中 / 仅合并 case-level arithmetic。
- **现状与证据：** [budget_report.comparison](../latency_bench/cross_device/budget_report.py#L24) 将 run mean 差乘 100/reference 并产出 signed/absolute error 与 bootstrap intervals；[budget_repeats.repeated_report](../latency_bench/cross_device/budget_repeats.py#L76) 另写 simulated mean、real mean 与 100*abs(sim-real)/reference，再对模型等权求每 seed NMAE。现在只有 run_mean 被两份报告共用。
- **当前消费者与测试归属：** budget_report 输出旧/随机 split 的成对或非成对 bootstrap；budget_repeats 将五个 validation partitions 的 per-model absolute error 等权平均，再对 split 报 mean±SD。既有测试分别是 [test_sim2real.py](../tests/cross_device/test_sim2real.py#L457) 的 test_budget_error_uses_absolute_mean_difference_and_paired_runs、:470 的 test_budget_report_aggregates_equal_case_weight_and_exports、:592 的 test_random_repeats_preserve_first_split_and_average_absolute_errors。
- **历史事实与推断：** 两种报告是不同 follow-up campaign，首个 random split被明确复用；它们应保持各自分析状态。公式重复是一行 case arithmetic 的事实，是否抽 helper 的净收益需审阅后讨论。
- **为什么值得：** 同一 normalizer/reference return 的每模型 case error 在两个报告路径分别编码；后续有不同归一化值时可能只更新一边。
- **最小建议：** 若能让定义更清楚，只提取输入真实/模拟 run means 与 frozen reference 后的 case-level signed/absolute error。bootstrap 的 pair/index resampling、跨模型权重及 across-split SD 留在各自报告层；若 helper 新增后只有一处有效调用或阅读更绕，则保留重复短表达式并只把共识写在协议文档。
- **风险与必须保留的契约：** validation real/sim 按匹配 run 配对；fit reference 的 real/sim 独立重采样；repeats 是每 split 先取绝对误差，再对两模型等权，最后以五 split 的样本 SD 描述变化。不能把 signed 值跨 model/split 平均后取绝对值，也不能把 episodes 当独立 sessions。
- **依赖：** 依赖两份报告确认 case normalizer 完全相同；paired/unpaired bootstrap 和五 split SD 不是共享 helper 的输入参数。
- **后续验收：** 保留上述三个测试的既有输出和不确定区间语义。若讨论后决定不引入 helper，此项可关闭为 keep，不要求为了抽象新建测试或模块。

<a id="rf-028"></a>

### RF-028 — 明确 evaluation config resolution 的嵌套边界

- [ ] **状态：待讨论**
- **动作层级**：Document / Consolidate（入口合同确认后）。
- **分类：** C04 重复实现、C06 隐式耦合、C07 配置债务
- **优先级 / 置信度 / 变更类型：** P2 / 多层调用事实高、消除工作的必要性中 / 梳理直接入口后减少同一路径的重复解析。
- **现状与证据：** [run_eval_sweep_from_config](../latency_bench/eval/sweeps.py#L97) 在 :108 调 resolve_evaluation_config；suite entry 在 :195 调它；两者继续经 _run_one_sweep 到 [run_from_config](../latency_bench/eval/driver.py#L34)，其 :47 再 resolve；[build_executor](../latency_bench/executors/factory.py#L48) 也 resolve。一次常规 CLI sweep 的调用链由 latency_bench.run 进入 sweeps、driver、executor factory；suite 中重复该尾部。resolver 目前会写入既定 defaults、规范 episode 字段与 absolute path（[eval/config.py](../latency_bench/eval/config.py#L21)）。
- **当前消费者与测试归属：** run CLI、direct run_from_config、直接 build_executor、cross-device resident replay 均可能作为独立入口。现有测试归属包括 [tests/eval](../tests/eval/test_eval_suites.py#L68) 的 reward/default 和 suite 配置契约、[tests/integration/test_run.py](../tests/integration/test_run.py) 的 CLI 链路，以及 cross-device model artifact 测试。
- **历史事实与推断：** [infra_release_cleanup.md](../docs/audits/infra_release_cleanup.md#L8) 是 2026-09-20 的 dated audit，并记载 defaults/path 有“one resolution owner”；这不是当前调用图的证明。当前代码调用链已按 HEAD 静态核对，原因可能是为了保留不同公开 entry point 的独立可调用性，属于推断。
- **为什么值得：** 同一 eval 执行会重复规范化 config；调用者不容易看清默认值、路径 base 和 mutability 的真正边界。
- **最小建议：** 先逐一标记对外直接调用的入口，保留这些入口的默认物化能力；只避免标准 CLI 调用链在同一层级反复解析。若做到这点必须加 resolved marker、私有 mode flag 或大量传参，则优先记录边界并保持现状。
- **风险与必须保留的契约：** eval defaults 是既有产品行为；保留缺 seed 与 seed=0 的区别、默认 raw reward / serial slots / worker capacity optionality，以及明确输入路径转 absolute 时的 HF symlink/checkpoint sibling 行为。本项不是把所有 .get 判为错误，也不要求新增验证或更改缺字段失败时机。
- **依赖：** 依赖保留 direct-dict、direct-driver、direct-build_executor 等独立入口；不添加“已解析”标记或额外 config 状态。
- **后续验收：** 复用 eval suite default/path tests、直接 run_from_config 与 build_executor 的现有测试、cross-device model artifact test。验收重点是同一 config 经 CLI 与直接 API 仍得到相同 defaults、seeds、suite 输出路径和 symlink-aware resolved config。

<a id="rf-029"></a>

### RF-029 — 将 Sample Factory policy type 分类集中到单一出处

- [x] **状态：已完成**
- **动作层级**：Consolidate。
- **分类：** C04 重复实现、C03 概念/语义漂移
- **优先级 / 置信度 / 变更类型：** P2 / 重复事实高、接口设计中 / 集中字符串分类，无需重做 policy registry。
- **现状与证据：** [eval/config.py](../latency_bench/eval/config.py#L13) 与 [policy/factory.py](../latency_bench/policy/factory.py#L8) 分别定义完全相同的四个 policy type：flappy_sf、demon_attack_sf、deadly_corridor_sf、gymnasium_sf。eval 使用该 set 判断 seed offset 和 checkpoint override（[eval/config.py](../latency_bench/eval/config.py#L134)），policy factory 用它路由 Sample Factory policy construction（[policy/factory.py](../latency_bench/policy/factory.py#L16)）。
- **当前消费者与测试归属：** run config、eval driver/sweeps 和 policy factory 都有真实调用者；tests/eval 对 episode seeds/checkpoint suites 有覆盖，tests/policies/test_sample_factory_policy.py 覆盖实际构造。
- **历史事实与推断：** 当前两组字符串逐项相同；是否曾为分层独立而刻意复制没有历史证据。
- **为什么值得：** 新增或改名一个 Sample Factory type 时，构造分派和 eval seed/checkpoint 分支可能只更新其中一处。
- **最小建议：** 把 membership set 放在轻量且两边可依赖的 policy type 模块或现有合适 owner，两个消费者引用同一分类。避免把 training task→policy map 也并入同一 set；那张映射语义不同。
- **风险与必须保留的契约：** policy.type 当前四个名字及它们的 seed offset、checkpoint kind override、Sample Factory 构造路径完全不变。检查模块导入方向，避免 eval.config 与 policy factory 循环导入；无需引入注册插件框架。
- **依赖：** 依赖 eval 与 policy 模块可共享轻量类型归属且无 import cycle；若需新注册框架，本项不值得实施。
- **后续验收：** 复用 tests/eval、tests/policies/test_sample_factory_policy.py 与 checkpoint eval suite tests。核对每个现有 policy type 的识别、seed 和构造/override 都保持原行为。
- **首批实施与验收（2026-09-30）：** 集合继续由 [`policy.factory`](../latency_bench/policy/factory.py#L8) 定义，[`eval.config`](../latency_bench/eval/config.py#L13) 直接引用。保持 evaluation 的 strip/lower 与 factory 精确匹配、四种类型及 SF 惰性导入。现有 factory 参数化用例补齐 gymnasium_sf，相关 seed 和 checkpoint-suite 用例通过。 主证据：[首批报告](../outputs/refactor/batch-01/report.md)，整体结果与边界见[批次记录](#batch-01)。


<a id="rf-030"></a>

### RF-030 — 抽取 fixed-latency sweep 的配置准备，不动运行汇总层

- [x] **状态：已完成**
- **动作层级**：Consolidate。
- **分类：** C04 重复实现、C07 配置债务
- **优先级 / 置信度 / 变更类型：** P2 / 高 / 合并固定延迟配置准备的重复步骤。
- **现状与证据：** [eval/sweeps.py](../latency_bench/eval/sweeps.py#L152) 的固定延迟 sweep 和 suite fixed branch（[同文件](../latency_bench/eval/sweeps.py#L211)）都 deepcopy config、把 raw frames 转成 ms、调用 _apply_eval_latency、改输出目录并构造 sweep metadata。两者已共用 _run_one_sweep（[同文件](../latency_bench/eval/sweeps.py#L39)），但准备配置仍有两份。
- **当前消费者与测试归属：** latency_bench.run 的 eval latency sweep 和 eval_suites 是两个当前 CLI/API 入口。现有验收归属包括 [test_eval_suites.py](../tests/eval/test_eval_suites.py#L375) 的 per-suite output、:430 的 CLI print，以及 tests/integration/test_run.py 的固定 latency sweep 行为。
- **历史事实与推断：** 当前实现已刻意抽出 shared run+summary tail；“caller prepares config”由 _run_one_sweep docstring 明确，说明配置准备被留在 caller 是事实，最初原因未知。
- **为什么值得：** frame-to-ms、fixed/zero method 和目录命名若只调整一个分支，同样的固定 sweep 可产生不同 config 或 summary。
- **最小建议：** 抽取只接受 base config、frame value 和 output/metadata context 的固定 sweep preparer；suite 名称嵌套目录与 standalone sweep 扁平目录继续由调用者明确给出，不把 normal/uniform/profile suite 塞到该 helper。
- **风险与必须保留的契约：** 保留 suite_name/latency_type、train/eval latency metadata、profile provenance、queue summary 名称、每 suite output path 和 episode selection。配置复制后不能写回原 base config。
- **依赖：** 依赖 fixed sweep 与 suite fixed case 共有的 frame-to-ms 语义；normal/uniform/profile suite 不属于该提取范围。
- **后续验收：** 复用 fixed suite output、CLI suite summary、固定 latency sweep integration cases；比较生成 config 的 latency units/method、目录及 summary metadata，不依赖 YAML 文本排序。
- **首批实施与验收（2026-09-30）：** 两种固定 sweep 共用 [`_prepare_fixed_sweep_config()`](../latency_bench/eval/sweeps.py#L39)，仅复制、换算、应用 latency、设置 output dir，返回 copy 与毫秒值。目录策略、metadata、policy reuse、run/summary 层和其他 suite 分支留在调用方。现有 fixed suite 用例扩展零/非零延迟与 source latency config 保持断言，普通 sweep CLI 及 suite 行为用例通过。 主证据：[首批报告](../outputs/refactor/batch-01/report.md)，整体结果与边界见[批次记录](#batch-01)。


<a id="rf-031"></a>

### RF-031 — 明确 generic profile 命令的工作目录契约

- [ ] **状态：待核实**
- **动作层级**：Document / Consolidate（CWD 支持范围讨论后）。
- **分类：** C06 隐式耦合、C07 配置债务、C14 可复现性
- **优先级 / 置信度 / 变更类型：** P2 / 相对路径现状高、外部调用受影响程度待核实 / 选择文档化或固定解析根。
- **现状与证据：** [protocol.py](../latency_bench/cross_device/protocol.py#L22) 的 REGISTRY_PATH 是相对路径 configs/profiles.yaml；[runtime_configs._base_config](../latency_bench/cross_device/runtime_configs.py#L113) 也把 registry 的 base_config 直接交给 load_config。相较之下，study.py 和 study_assets.py 按 __file__ 推 repo root。正式 shell 入口 [profile.sh](../scripts/profile.sh#L5) 主动 cd 到 repo root，所以标准入口有明确行为；protocol Python module/API 是否承诺可在任意 CWD 运行，目前没有证据。
- **当前消费者与测试归属：** supported shell 入口是 scripts/profile.sh；[test_profile_protocol.py](../tests/cross_device/test_profile_protocol.py#L53) 在 repo root 以相对路径读 profiles.yaml。cookbook 也使用 profile shell wrapper。没有找到 outside-CWD 的 generic profile CLI contract。
- **历史事实与推断：** 当前 shell wrapper 固定 repo-root CWD 是事实；直接从任意目录运行 protocol 是否受支持仍待核实，不能把未验证的失败说成 bug。
- **为什么值得：** Python API 的 registry/base_config 路径语义隐含依赖调用者 CWD；同一命令经 wrapper 和 module 直接调用可能表现不同。
- **最小建议：** 先确认支持面。若 wrapper 是唯一支持的入口，明确记录“从 wrapper 调用/依赖 checkout root”；若 module/API 也要支持其他 CWD，再用 repo anchor 解析 registry 与仓库内 base_config。artifact_root、model_path、output_root 等用户选择的路径继续按现有语义处理，不能一并改基准。
- **风险与必须保留的契约：** 不改变 configs/profiles.yaml preset 的意义、HF/local artifact 路径、相对 output root 以及 wrapper 环境变量设置。不应用 path normalization 的改动到 archived config 中嵌入的模型路径。
- **依赖：** 依赖确认支持的 CLI/API 入口范围；在此确认前保留现有 scripts/profile.sh 行为和 CWD 语义。
- **后续验收：** 讨论后若选 repo-root contract，复用 profile shell CLI；若选择外部 CWD 支持，须检查相对 registry、base_config、artifact_root 和 explicit CLI output 的路径基准各自是否正确。不要只用 repo-root 测试宣称可移植。

<a id="rf-032"></a>

### RF-032 — 评估 18 份 sim2real archived eval snapshot 的重复，不覆盖来源档案

- [ ] **状态：待讨论**
- **动作层级**：Document（runtime overlay 提取另议）。
- **分类：** C07 配置债务、C09 训练/评测/部署漂移、C14 可复现性
- **优先级 / 置信度 / 变更类型：** P3 / 重复矩阵高、可安全收敛范围中 / 先做差异盘点，再考虑仅限 runtime overlay 的提取。
- **现状与证据：** [sim2real registry](../configs/experiments/sim2real.yaml#L6) 为 6 个任务各列 3 个 model variant，共 18 个 eval_config 路径；HEAD 7dd2e15f（2026-09-28）同时新增该 registry、18 个 task/model YAML archive 和 pinned study。当前静态解析确认 18 个引用文件均存在；study_assets.prepare_model_assets 按每条 entry 的 eval_config 读取，study.main 从该 registry 选择 task/model。脚本 README 把这些文件定义为 verbatim archived snapshots；每条 entry 的 checkpoint、model profile、training profile revision、policy_files 可能不同。
- **当前消费者与测试归属：** [study_assets.py](../latency_bench/cross_device/study_assets.py#L34) 负责读取/移动这些 inputs；[test_sim2real.py](../tests/cross_device/test_sim2real.py#L30) 的 test_asset_resolution_pins_profile_model_and_relocates_archive、:86 的 test_archived_task_contract_survives_clock_calibration 和 :184 的 test_real_upload_resume_and_pinned_cross_host_replay 是直接 consumers/contracts。入口通过 study main 和 scripts/experiments/sim2real/run.sh。
- **历史事实与推断：** 2026-09-28 commit 具体加入 18 个 YAML 与 registry 是历史事实；README 说明它们是原样 archive。即使文件存在大量相似字段，也不能据此推断为死配置或生成残留；相同代码是否应从运行配置层收敛属于待讨论的架构选择。
- **为什么值得：** 同一任务的 model variants 有重复 env/scheduler/eval 结构，逐份审读很难发现不一致；后续维护也需要明确哪些差别来自模型数据、哪些差别是意外漂移。
- **最小建议：** 先只产出 field-level diff/invariant 清单。若决定收敛，只对新生成的 runtime overlay 中确证相同的字段建立共享来源；18 份来源 YAML 保留原样、保留原路径引用或在单独的 provenance 字段记录源。不要用 template 重写/删除 archived YAML，也不要让当前统一值追溯覆盖其训练期配置。
- **风险与必须保留的契约：** exact model/checkpoint/training-profile revision、FPS/obs stride、max steps、frame skip、action order/scaling、history/normalization、policy task contract 和 task metrics 逐 case 保留。尤其 ant 两种 FPS/horizon、Mikasa action-prefix/task-contract 行为不能因“相似配置”被抹平。
- **依赖：** 依赖 archived eval YAML 继续作为来源档案；仅在决定运行层有共享字段时才讨论 runtime overlay，来源文件不做覆盖生成。
- **后续验收：** 复用 asset pinning、archived task contract、pinned cross-host replay 三个现有 tests；逐 pair 比较 resolved runtime config 与原 archive，明确每个覆盖字段及原因。若要改变任何 archive 内容、条目数量或 task/model 能力，必须先单独讨论；默认不删除任何 variant。

## 测试、实验脚本与文档

<a id="rf-033"></a>

### RF-033 — 已不存在的 formal_chain_watchdog 仍有测试调用

- [ ] **状态：待讨论**
- **动作层级**：Prune 或 Consolidate（原测试合同讨论后）。
- **优先级 / 置信度 / 动作**：P1；高（模块缺失已定位，未运行 pytest）；Prune 或迁移现有行为测试。
- **分类**：C16 仓库卫生；旧工作流是否已退役仍待确认。
- **位置与证据**：[test_starvla_collect.py](../tests/scripts/test_starvla_collect.py) L57–84 的 `test_formal_chain_publishes_source_and_target_with_matching_profile` 在 L58 导入 `formal_chain_watchdog`，并调用其 `_command`。当前任务目录没有对应模块；只读 `importlib.util.find_spec('scripts.tasks.humanoidbench_balance_simple.formal_chain_watchdog')` 返回 `None`。这是函数执行时的导入问题，不应描述成整个测试文件无法收集。
- **现有用途 / 测试归属**：该测试想约束 source/target 两阶段发布、resume 和 matching-profile 参数；当前实际编排在 [vla_profile_dagger.py](../scripts/tasks/humanoidbench_balance_simple/vla_profile_dagger.py) 与 [queue_profile.py](../scripts/tasks/humanoidbench_balance_simple/queue_profile.py)。这些入口是否继续承诺原来的两模型串联行为，需按现有任务合同确认。归属仍为 `tests/scripts/test_starvla_collect.py`。
- **历史原因**：源码显示测试与模块发生脱节；本次未确定模块删除的历史提交，不推断原功能已经废弃。
- **为什么值得处理**：失效测试会阻塞后续回归，也会使重构者误以为应该重新创建一个旧控制器。
- **最小建议与边界**：若用户仍需要该行为，把测试迁移到当前实际入口的命令或产物合同；若旧串联流程已经终止，经讨论后删除这一个孤儿测试。单纯增加兼容模块、重新导出旧 `_command` 或给测试加永久 skip 都不能证明需求。
- **风险与保留行为**：不能误删 matching-profile、阶段发布或连续恢复的当前合同；不能为了修测试重新引入已没有消费者的架构。
- **验收与依赖**：先追溯当前任务手册和调用者，再在具备现有依赖的环境运行此用例及实际流程所在的既有测试。若迁移，断言最终命令/产物的 profile、resume 和发布身份；若删除，记录用户确认的废弃行为。与 RF-037 的入口盘点有关，不必等待整套脚本重构。
- **待讨论**：还需要旧 source→target 串联能力吗？选择删除孤儿测试，还是将仍有效的行为合同迁移到当前入口？

<a id="rf-034"></a>

### RF-034 — pi0.5 来源快照与 source_map 的摘要、大小不一致

- [ ] **状态：待讨论**
- **动作层级**：Document（来源调查后决定修复）。
- **优先级 / 置信度 / 动作**：P1；高（当前文件已按既有清单核对）；Document，然后按来源调查结果修复资产组织。
- **分类**：C08 数据/checkpoint/实验协议债务、C14 可复现性、C16 仓库卫生。
- **位置与证据**：[source_map.json](../configs/examples/gymnasium/pi05_sources/source_map.json) L2–5 记录来源仓库与 revision，各项记录 `path/sha256/bytes`；[test_pi05_environment_overlays.py](../tests/core/test_pi05_environment_overlays.py) L23–40 消费此清单，并将 overlay 与来源 profile/manifest 对照。2026-09-30 的只读核对发现 **7 个任务、17 个文件的 SHA-256 和字节大小均不匹配**；没有把历史审计中“三个任务”的结论直接沿用。

| 任务 | 不匹配文件，相对于 `configs/examples/gymnasium/pi05_sources/<task>/` | 文件数 |
|---|---|---:|
| air_raid | `profile_latency/profiling/sessions/0/generated.yaml` | 1 |
| inverted_pendulum | `zero_latency/source_contract/inverted_pendulum_l0_small_model_train.yaml` | 1 |
| hopper | `profile_latency/profiling/sessions/0/generated.yaml`；`zero_latency/source_contract/config.json`、`metadata.json` | 3 |
| half_cheetah | `profile_latency/profiling/sessions/0/generated.yaml`；`zero_latency/source_contract/small_model_train.yaml` | 2 |
| walker2d | `profile_latency/profiling/sessions/0/generated.yaml`；`zero_latency/source_contract/small_model_train.yaml` | 2 |
| ant | `profile_latency/profiling/sessions/0/generated.yaml`；`zero_latency/source_contract/config.json`、`metadata.json`、`small_model_train.yaml` | 4 |
| humanoid | `profile_latency/profiling/sessions/0/generated.yaml`；`zero_latency/source_contract/config.json`、`metadata.json`、`small_model_train.yaml` | 4 |

- **现有用途 / 测试归属**：七份 `configs/examples/gymnasium/<task>/pi05_train.yaml` 的 `source_manifest`、`source_clock` 和 `hydra_overrides` 是已有训练配方；来源文件是协议对照和 provenance 输入。`tests/core/test_pi05_environment_overlays.py` 是现有所有者；本次只复核其来源完整性子条件，没有执行配置合成或整套测试。
- **历史原因**：[历史清理审计](../docs/audits/infra_release_cleanup.md) L43–51 已记录当时的 digest 问题。当前 17 个文件为什么发生变更，需要逐文件追溯 Git 和原来源；“后续迁移改写了快照”只是可能原因。
- **为什么值得处理**：同一文件既声称是固定来源快照、又可能被当作当前可修改配方，会混淆实验来源并使现有回归失效。
- **最小建议与边界**：先查清每项代表原始快照还是经过批准的迁移产物；原始快照应按对应 revision 恢复，迁移配方应放在独立 overlay 并记录关系。如果来源身份确实变了，经过确认后更新来源描述及现有清单。不能仅重新计算期望 SHA/大小来让测试通过，也不能移除有实际消费者的来源校验。
- **风险与保留行为**：保留原来源 revision、task/action/state/FPS 语义和用户批准的当前配方；修复不能把新的配置冒充旧来源。现有摘要是可复现性合同的消费者，本项不建议增加额外 checksum 文件。
- **验收与依赖**：逐个确认上表文件的来源；运行既有七任务 overlay 用例，同时检查配置合成、manifest 维度、来源时钟和 checkpoint 初始化设置。与 RF-032 的来源/运行 overlay 分离原则一致。
- **待讨论**：这 17 个文件应保持原始快照身份，还是属于要保留的迁移版本？分别选择恢复来源、独立 overlay，或显式变更来源身份。

<a id="rf-035"></a>

### RF-035 — 一部分测试锁定源码表达或生产对象私有状态

- [ ] **状态：候选（删减具体测试需讨论）**
- **动作层级**：Consolidate（删减测试另议）。
- **优先级 / 置信度 / 动作**：P2；高（断言形式可见，覆盖是否冗余需逐项核对）；Consolidate。
- **分类**：C05 抽象不匹配、C12 API 泄漏实现细节。
- **位置与证据**：[test_eval_bash_scripts.py](../tests/scripts/test_eval_bash_scripts.py) L315–322 直接查源码里的函数名、环境变量表达；L752–757 以源码不存在某些字符串作测试；L927–930 再检查 helper 源码字符串。[test_latency_samplers_core.py](../tests/latency/test_latency_samplers_core.py) L185–198 有直接修改 `clone._slot_to_rank` 的私有状态断言；[test_generic_gymnasium_starvla_policy.py](../tests/policies/test_generic_gymnasium_starvla_policy.py) L195、L238 等读取真实 policy 的 `_env_name`。
- **现有用途 / 测试归属**：这些测试分别保护固定安装栈、sampler 克隆隔离、policy 动作路由。安装脚本测试本来已有 fake executable/subprocess 日志边界；policy 测试已有输出动作、wrapper 注入 seam；sampler 用例已有采样序列对照。保持各自既有 suite 归属，避免创建一个新的全局“重构测试”文件。
- **历史原因**：[tests/README.md](../tests/README.md) L43–78 已明确要求可观察行为，并允许真实性能合同、参考实现和注入 collaborator seam；当前部分断言与该原则不一致。不能据此推断整套测试没有价值。
- **为什么值得处理**：移动函数、调整 shell 表达或换内部表示会让保持行为的重构失败；生产逻辑也可能因为测试名称而错误保留过时结构。
- **最小建议与边界**：逐个把真正的合同改到公开输入/输出或边界上：检查实际调用固定版本安装命令和 CUDA 环境、检查动作输出、检查 clone 后一方 reset/sample 不改变另一方序列。删除冗余断言前确认同一合同已有覆盖。保留注入 fake 的调用断言、有效的 artifact 摘要校验、GPU 参考实现一致性与有文档的内存共享合同。
- **风险与保留行为**：删除字符串断言可能丢失固定安装栈等真实需求；因此按行为保留，而不是一律删除所有 `read_text`、私有函数测试或 call-count。
- **验收与依赖**：在各拥有者 suite 中证明保持行为的内部改写不会破坏测试，而错误安装版本、错误动作或共享可变 RNG 状态仍被捕获。优先更新已有用例；与 RF-002、RF-011 的后续重构协调，但可以先处理明显过度约束的断言。
- **待讨论**：没有行为合同且已被高层用例覆盖的断言，是否允许直接删除？保留的每条测试应能解释它保护的长期合同。

<a id="rf-036"></a>

### RF-036 — 第三方导入的测试作用域会修改全局 sys.path

- [x] **状态：已完成**
- **动作层级**：Consolidate。
- **优先级 / 置信度 / 动作**：P2；高（直接路径修改已确认，测试顺序故障未复现）；Consolidate。
- **分类**：C06 隐式状态/隐藏耦合、C11 状态所有权、C15 依赖/环境债务。
- **位置与证据**：[test_pi05_environment_overlays.py](../tests/core/test_pi05_environment_overlays.py) L12–16 在模块导入时直接插入 StarVLA 路径；[test_starvla_policy.py](../tests/policies/test_starvla_policy.py) L326–336 在测试中直接插入路径；[test_eval_core_frameskip_alignment.py](../tests/envs/eval_align/test_eval_core_frameskip_alignment.py) L142–145 与 [test_latency_bench_reset_modes.py](../tests/envs/eval_align/test_latency_bench_reset_modes.py) L176–178 的 helper 也直接修改路径。这些点没有各自的作用域恢复。
- **现有用途 / 测试归属**：导入 pinned StarVLA 的配置合成、saved-checkpoint 兼容函数和 simulator 对齐入口。`tests/conftest.py` 为仓库/共享 helper 设定的 suite 启动路径是不同责任，不在本项机械删除范围。
- **历史原因**：推测是在第三方源码不按基础环境安装时便于测试导入；代码未记录这些特定全局改动应永久存活的需求。
- **为什么值得处理**：测试结束后路径优先级仍改变，并且 `sys.modules` 还可能缓存第三方模块，使后续测试的解释器状态依赖执行顺序。
- **最小建议与边界**：普通用例使用现有 pytest `monkeypatch.syspath_prepend` 或有恢复的局部 fixture；模块级导入和真正需要隔离 import cache 的跨环境用例，考虑在它本来就拥有的 subprocess 边界执行。先解决具体作用域，不新造全局第三方插件加载框架。
- **风险与保留行为**：仅恢复 `sys.path` 不能自动撤销 `sys.modules`；隔离方案要按真实导入行为选择。保留 optional integration 的跳过条件和 pinned 第三方身份。
- **验收与依赖**：相应用例单独运行与调整执行顺序运行结果一致，测试结束后路径状态恢复；导入到的仍是预期 pinned 模块。归属上述四个既有 suite。与 RF-035 独立，但整理测试时可在同一次小改动中处理同一文件。
- **待讨论**：当前没有删除功能建议；若要移除第三方集成测试，另行确认其长期合同。
- **首批实施与验收（2026-09-30）：** 四处改用 pytest monkeypatch 的 syspath 作用域：[`pi0.5 overlay`](../tests/core/test_pi05_environment_overlays.py#L19)、[`saved checkpoint`](../tests/policies/test_starvla_policy.py#L326)、[`frameskip alignment`](../tests/envs/eval_align/test_eval_core_frameskip_alignment.py#L58)、[`reset alignment`](../tests/envs/eval_align/test_latency_bench_reset_modes.py#L43)。只有 overlay 原先有模块级第三方 import，现移入用例；删除两个被替代的 path helper。四个 suite 正反序各 71 例，逐例 teardown 后 sys.path 恢复，真实模块均来自同一 StarVLA 快照，七个实际 Atari 对齐用例通过。overlay 的七个原摘要失败保留，未删测试或改变来源期望。 主证据：[首批报告](../outputs/refactor/batch-01/report.md)，整体结果与边界见[批次记录](#batch-01)。


<a id="rf-037"></a>

### RF-037 — MIKASA 多代 launcher 继续维护相近的完整实验编排

- [ ] **状态：待讨论**
- **动作层级**：Consolidate / Prune（逐入口讨论后）。
- **优先级 / 置信度 / 动作**：P2；高（流程与调用链确认，哪些历史入口可退役待确认）；Consolidate，历史入口的 Prune/归档另议。
- **分类**：C01 失效历史路线候选、C04 重复实现、C07 配置债务、C08 实验协议债务。
- **位置与证据**：[run_fixed_v2_h8.sh](../scripts/tasks/mikasa_intercept_grab_fast/run_fixed_v2_h8.sh) L22–64 包含 collect→package→audit→train→eval；[run_fixed_prefix63.sh](../scripts/tasks/mikasa_intercept_grab_fast/run_fixed_prefix63.sh) L47–122 维护相近的流程并增加恢复、DAgger、阶段发布。`run_h8_v2_six_way.sh`、`run_h8_v2_cross_host.sh`、`run_h8_v2_queue.sh` 和 `run_h8_seed_reset_queue.sh` 仍引用 v2 launcher；[MIKASA_FIXED_LATENCY_GRID.md](MIKASA_FIXED_LATENCY_GRID.md) L48、L97 仍保存旧入口，新的 [重跑协议](MIKASA_PREFIX63_RESTART.md) L36–50 引用 prefix63 入口。
- **现有用途 / 测试归属**：历史路线的运行复现、seed-reset 队列和当前 fixed-N 从头采集/SFT/DAgger 都有文档或脚本消费者。共同 collector/packager 不意味着这些 launcher 都执行原来的 71D 输入；当前被导入的模块已迁移，必须实际比较解析后合同。回归主要归属 [test_mikasa_h8.py](../tests/scripts/test_mikasa_h8.py)、[test_mikasa_latency_pipeline.py](../tests/scripts/test_mikasa_latency_pipeline.py) 和各命令边界用例。
- **历史原因**：文档已记录 71D v2 输入缺陷、63D 修复、独立 run ID 的重采集和阶段发布方案；多代入口是这些实验决策的积累，而非凭文件名判断的 dead code。
- **为什么值得处理**：相近阶段被独立修改，数据目录、训练预算、eval seeds、发布阶段和恢复判据可能逐步分叉；历史名称又容易被误当作当前协议。
- **最小建议与边界**：先做入口→当前调用者→解析后协议→产物目录矩阵；只抽取实际相同的阶段命令或轻量任务参数描述。保留 fixed/profile/naive 的不同选优、prefix、heldout 与发布合同。对确认终止的完整入口，让用户选择直接删、归档为历史复现材料，或保留简化入口；不预设向所有旧入口添加兼容 wrapper。
- **风险与保留行为**：尤其保留 train/validation seed 隔离、frozen normalization、latest 驱动连续 DAgger、optimizer/RNG 恢复、matched periodic 选优，以及训练/终评/发布分别完成的状态。不能让旧入口在新组件下声称能逐步复现旧训练。
- **验收与依赖**：先完成 RF-040 的协议版本盘点，并参考 RF-016 的共享运行描述；比较保留入口生成的命令、配置与目标目录，既有测试验证阶段恢复和合同；只有改动实际执行语义时才另做相应 smoke/正式验收。
- **待讨论**：哪些 v2、六路、跨主机、seed-reset 入口还需要作为可执行能力保留？对每个历史入口选择删除、历史归档或保留能力精简，不能按一个全局答案批量处理。

<a id="rf-038"></a>

### RF-038 — 跨主机实验入口绑定固定主机、解释器和 GPU 布局

- [ ] **状态：待讨论**
- **动作层级**：Document / Consolidate / Prune（用途讨论后）。
- **优先级 / 置信度 / 动作**：P2；高（硬编码可见，主机当前用途未查询）；Document / Consolidate。
- **分类**：C06 隐藏耦合、C07 配置债务、C14 可复现性、C15 环境债务。
- **位置与证据**：[run_h8_v2_cross_host.sh](../scripts/tasks/mikasa_intercept_grab_fast/run_h8_v2_cross_host.sh) L5–16 固定第二主机、源码目录和三套解释器；L64–88 固定 GPU 卡对/端口；L55–62 在固定远端目录删除、重传数据。[run_h8_seed_reset_queue.sh](../scripts/tasks/mikasa_intercept_grab_fast/run_h8_seed_reset_queue.sh) L5–26 固定带日期的路径、解释器和 W&B entity。相比之下，prefix63 入口 L5–12 已要求调用者显式传入 condition、run ID、devices、port、VLA_PY 和 UV。
- **现有用途 / 测试归属**：这些入口曾执行具体跨主机和 seed-reset 实验，手册仍引用其相关流程；未查询 SSH、GPU、进程或 W&B 当前状态。`tests/scripts/test_mikasa_h8.py` 的队列/命令用例是相关所有者；实际远端 bootstrap 的端到端覆盖未核验。
- **历史原因**：运行脚本和日期目录说明它们是具体实验部署快照；主机是否退役不能从历史记录确定。
- **为什么值得处理**：更换机器时会修改算法脚本；直接复制旧命令可能操作错误的部署目录；解释器/模型环境和协议参数混在一起，难以说明复现条件。
- **最小建议与边界**：若继续作为公共入口，接受明确的主机、源码根目录、解释器与设备分配；固定实验 seeds、模型/数据 revision、预算和 FPS 继续留在实验协议中。若仅为历史操作记录，经讨论后迁到合适的归档并保留说明。沿用现有参数传递，不添加自动探测候选主机或隐式环境路径回退。
- **风险与保留行为**：保留物理 GPU 与逻辑设备映射、端口隔离、同步范围和阶段顺序。不能把 shell 中的既有远端删除命令当成这次审计的执行指令；未来保留或取消这些行为应随入口用途一起确认。
- **验收与依赖**：依赖 RF-037 的保留入口矩阵；用命令生成或 fake SSH/rsync 边界验证传入配置出现在正确位置，不在文档整理阶段连接旧主机。若改为公共入口，后续实际部署验证须单独记录。
- **待讨论**：跨主机脚本应作为可移植工具继续维护，还是只保留原实验部署快照？若保留，哪些操作是当前确实需要的阶段？

<a id="rf-039"></a>

### RF-039 — MIKASA publisher 重复上传和 revision 回读机制

- [ ] **状态：候选（发布/兼容行为改动待讨论）**
- **动作层级**：Consolidate（发布语义变更另议）。
- **优先级 / 置信度 / 动作**：P2；高（重复传输机制与不同阶段合同均确认）；Consolidate。
- **分类**：C04 重复实现、C08 checkpoint/协议债务、C11 artifact 所有权、C14 可复现性。
- **位置与证据**：[publish_h8_checkpoints.py](../scripts/tasks/mikasa_intercept_grab_fast/publish_h8_checkpoints.py) L164–197 的 `_verify/_publish` 与 [publish_prefix63.py](../scripts/tasks/mikasa_intercept_grab_fast/publish_prefix63.py) L22–39 的 `publish_folder` 都上传目录并在返回 revision 检查路径/大小。前者 L181–189 对已有 receipt 再回读，并处理旧发布路径；后者 `publish_training_endpoint` L50–55 只确认 receipt 的端点身份便返回。后者还按 100 路径分批查询，过滤 cache/receipt；这些差异不是可机械删掉的重复。
- **现有用途 / 测试归属**：旧 H8 模型的打包发布入口消费前者；[run_fixed_prefix63.sh](../scripts/tasks/mikasa_intercept_grab_fast/run_fixed_prefix63.sh) L33–43、L75–81、L112–121 消费后者的失败诊断、阶段 checkpoint 和 final/best 上传。既有回归在 `tests/scripts/test_mikasa_h8.py` 的 publication path、receipt 和回读用例。
- **历史原因**：prefix63 重跑协议明确要求阶段结束上传完整恢复状态，并在返回 revision 按文件核对；旧 H8 publisher 同时包含历史模型目录迁移。这两套合同都是实际需求。
- **为什么值得处理**：相同 HF 传输/核对原语分散，修改 query batching 或文件过滤时容易遗漏；但阶段完成和“已发布”不能靠一个通用布尔值合并。
- **最小建议与边界**：只共享“给定文件集合上传并在指定 revision 逐路径核对”的小函数。阶段文件选择、实验完整性、选中 checkpoint、历史路径迁移、receipt 内容和重跑策略仍归各 owner。是否在已有 receipt 分支重新回读，应单独明确；不把其中一个 publisher 的处理默默当作全局默认。
- **风险与保留行为**：保持完整 trainer recovery state、选优权重、统计/合同、失败日志、远端 prefix 与 revision 身份；校验失败不能写成功 receipt。现有发布合同检查应保留，不另造额外摘要或校验框架。
- **验收与依赖**：在既有测试中分别验证首次上传、相同 receipt 恢复、旧路径迁移、文件缺失/大小不符和分批查询；stub HF 边界，不能触发实际上传。依赖 RF-037 的阶段归属，影响发布状态记录时参考 RF-040。
- **待讨论**：两个 publisher 的恢复语义是否应都要求重新回读？旧路径迁移还需要多久？任何删除迁移分支或改变成功判据的选择都单列确认。

<a id="rf-040"></a>

### RF-040 — 协议版本、历史状态与正式结果的文档入口需要更明确

- [ ] **状态：候选（文档职责调整）**
- **动作层级**：Document（归档/删文档另议）。
- **优先级 / 置信度 / 动作**：P3；高（多文档角色和版本说明已定位，未查询实际运行状态）；Document。
- **分类**：C03 语义漂移、C08 实验协议债务、C14 可复现性、C16 仓库卫生。
- **位置与证据**：[MIKASA_ACTION_CHUNK_H8.md](MIKASA_ACTION_CHUNK_H8.md) L115–124 已明确说明后文为 71D v2 历史合同；[MIKASA_FIXED_LATENCY_GRID.md](MIKASA_FIXED_LATENCY_GRID.md) L3–5 已指向 63D 修复；[MIKASA_FIXED2_PREFIX_REPAIR.md](MIKASA_FIXED2_PREFIX_REPAIR.md) L3–4 又将当前启动与发布方案指向 [MIKASA_PREFIX63_RESTART.md](MIKASA_PREFIX63_RESTART.md)。[README](README.md) 的任务表仍同时列出这些入口；[HANDOFF](HANDOFF.md) L3–5、L17–36 自称“当前”资产状态，另有 [RESULTS](RESULTS.md) 的正式汇总。不同表格可能对应不同模型、horizon 与 seeds，不能因为数字不同就判断结果冲突。
- **现有用途 / 测试归属**：任务手册提供可执行协议，HANDOFF/ARTIFACT_HANDOFF 负责迁移资产，RESULTS 提供对外表格，PROGRESS 负责 Balance/MIKASA 最新已核验结论，logs/archive 负责时间线和历史计划。文档链接校验与人工合同对照是本项验收，不新造数值快照测试。
- **历史原因**：已有显式迁移标注证明作者在保留历史事实；后续实验扩展使正文、索引和状态表分散。本项不是宣称“71D 与 63D 当前仍同时使用”或“历史 running 代表仍运行”。
- **为什么值得处理**：读者可能跨文档拼出混合合同，或拿旧交接状态判断当前完成/发布；协议范围没有集中呈现时尤其容易混用 H1/H8、历史/修复后输入。
- **最小建议与边界**：为任务手册建立清楚的协议版本/适用资产/当前接续位置；索引将历史合同与当前入口标明。资产交接状态注明核查日期和协议范围，正式结果链接到 exact checkpoint/revision 和主证据。重复数值可用主证据链接取代，但按用户决定保留仍有比较用途的历史基线。
- **风险与保留行为**：保留 71D 结果的失效原因、63D 输入尺度与重跑决策，以及训练/评测/acceptance/publication 四种独立状态。不能为清理文档移动或改写原始证据，也不把游戏状态全部塞入只负责 Balance/MIKASA 的 PROGRESS。
- **验收与依赖**：新读者从 README 能唯一定位指定协议的入口与合同，历史页指向准确接续页；对照相同模型/输入/seed/FPS 的结果才核对数值。涉及最新状态只在有新验证证据时更新；本次候选清单不替各实验刷新状态。为 RF-032、RF-037 的入口保留决策提供依据。
- **待讨论**：哪些旧手册需要继续可执行复现，哪些只需历史证据？若要归档或删除文档，先确认保留基线和外部引用。

## 删除、归档与语义变更讨论区

**下面没有任何已批准删除项。** 讨论时先确认“当前仍需要的能力/资产”，再在直接删除、保留证据归档、保留能力精简之间选择。Git 可保存源码历史，但不能自动代替未纳入 Git 的 dataset、checkpoint、profile 或远端 revision 证据。

| 条目 | 需要用户决定的事项 | 讨论前必须补齐的依据 |
|---|---|---|
| [RF-001](#rf-001) | score/raw-return/final-value 的唯一语义，是否修正历史结果 | 受影响输入及实验目录、双路径输出对照 |
| [RF-004](#rf-004) | 只有证明存在差异时，选择帧边界的语义 | 对相同输入的舍入、ready、apply、回填对照 |
| [RF-006](#rf-006) | injected env 与 config 的 FPS 谁是权威 | 当前公开调用方及支持合同，不添加防御检查来替代决定 |
| [RF-007](#rf-007) | 是否删除无效时间参数，是否涉及外部调用者 | 全部实际调用方；fake 签名本身不建立兼容需求 |
| [RF-008](#rf-008) | 缺失/未知按钮布局如何处理，是否去掉静默替代 | 支持 wrapper 的真实按钮名、顺序和 introspection 合同 |
| [RF-009](#rf-009) | 双文件保留、选 canonical 后迁移，或允许破坏 artifact 名称 | 分析工具、公开文档、历史产物和外部读者 |
| [RF-019](#rf-019) | 恢复最新 checkpoint 与评估选择候选是否应有不同失败行为 | SF resume 合同、显式路径/自动选择的消费者 |
| [RF-020](#rf-020) | filter 参数是长期 CLI 合同还是仅历史实验开关 | 当前 Demon 等调用、既有接受集合与文档 |
| [RF-026](#rf-026) | 增加 timing 字段后如何处理旧 campaign 的 timing | 历史报告需求及真实缺失字段，不自动猜值 |
| [RF-027](#rf-027)、[RF-028](#rf-028) | 简单共享/减少重复是否真正降低维护成本 | 保持所有入口与统计合同的最小替代方案 |
| [RF-031](#rf-031) | generic profile Python API 是否支持任意 CWD | 支持入口、用户路径和仓库资源路径的不同基准 |
| [RF-032](#rf-032)、[RF-034](#rf-034) | 来源快照、迁移版本和运行 overlay 分别如何保留 | 原 revision 与现有来源差异；不得只刷新摘要 |
| [RF-033](#rf-033)、[RF-035](#rf-035) | 删除失效/冗余测试，还是迁移仍需保护的合同 | 当前真实能力与已有高层覆盖 |
| [RF-037](#rf-037)、[RF-038](#rf-038) | 每个旧 launcher 和跨主机入口删除、归档或精简 | 入口→调用者→当前解析协议→资产的对应表 |
| [RF-039](#rf-039) | receipt 回读规则和旧发布路径迁移是否继续保留 | 两条发布合同及实际历史 receipt 消费者 |
| [RF-040](#rf-040) | 哪些旧手册继续可执行，哪些只保留历史事实 | 外部引用、正式基线及当前接续位置 |

## 明确应保留的设计

这些实体容易被表面重复或本仓局部调用次数误判。只有实际合同改变后才重新讨论去留。

| 设计 / 当前消费者 | 保留理由与主要证据 | 后续重构必须保持 |
|---|---|---|
| single、batched、realtime；local-env 与 env-process loops | [infra](../docs/infra.md) L26–30 明示不同资源/性能合同；[realtime executor](../latency_bench/executors/realtime/executor.py) 与 [env_worker](../latency_bench/executors/realtime/env_worker.py) 分别持有 parent/IPC 和 env deadline/ack 生命周期 | observation cadence、ready/apply、late completion、timeout、cleanup；只共享证明相同的纯规则 |
| GPU batching、shared observation、streaming metrics、CPU/generic 参考路径 | 由 [shared_observation](../latency_bench/executors/realtime/shared_observation.py)、GPU env/backend、批量 executor 与 owning suites 消费；性能机制已有文档 | tensor device、render/动作等价、共享内存释放、记录关闭后的低内存路径；不因去重强制保存整集 records |
| action-ready 与 worker-service 两个 latency 维度 | [LatencySample](../latency_bench/latency/sample.py) L9–61 明确两类量；[simulated executor](../latency_bench/executors/simulated.py) 分别用于 action readiness 和 worker occupancy | service 不等于 ready 的样例、worker capacity 与 action apply frame；不能合成一个 latency_frames |
| profile fitting 与 sampler/replay 的边界 | [burst_profile](../latency_bench/latency/burst_profile.py) 拟合 session/worker regime，[samplers](../latency_bench/latency/samplers.py) 持有 persistent replay state；分别有拟合/采样 tests | session reset、worker相关性、Temporal RNG/state 和拟合输入隔离；文件长不等于应该合并 |
| grouped replay 当前新增接口 | [grouped_replay](../latency_bench/cross_device/grouped_replay.py)、study_parallel 和 budget 消费 `episode_groups` / `latency_sources_by_episode`；[批量 executor suite](../tests/executors/test_batched_simulated_executor.py) 有 session/slot 对照 | 同 run 两集共享 Temporal stream，seeded IID/Normal 仍使用 per-episode 随机流；slot 改变不改逻辑 seeds、跨 setting run ID 不冲突、trace 完整后 DONE |
| Sample Factory GPU sampling / episode-extra-stats / raw-reward patches | [infra](../docs/infra.md) L54–68 指明 pin 和职责；Flappy/Demon/Gymnasium integrations 实际安装；[stats compat tests](../tests/training/test_sf_episode_extra_stats_compat.py) 覆盖嵌套安装 | patch 安装顺序、idempotence、raw reward 与学习 reward 的区别、episode metadata 和 GPU 快路径 |
| checkpoint 原子写入、RNG 恢复和 evaluation lease | [checkpoints](../training/common/checkpoints.py) 被 SF runtime、policy loader、periodic eval 使用；[checkpoint IO tests](../tests/training/test_sample_factory_checkpoint_io.py) 和 [layout tests](../tests/training/test_training_checkpoint_layout.py) 为 owner | 不读半写文件、不删正在评估的 checkpoint、完整恢复 optimizer/scheduler/RNG；RF-018 仅整理职责 |
| MIKASA ActionLatencyBatch / TorchActionLatencyWrapper | 子模块 [get_mikasa_robo_datasets.py](../third_party/MIKASA-Robo/mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py) L493–496 与 [get_dataset_collectors_ckpt.py](../third_party/MIKASA-Robo/mikasa_robo_suite/vla/dataset_collectors/get_dataset_collectors_ckpt.py) L1688–1694 有真实调用 | 原生 tensor vec env 的设备张量、autoreset/transport reset 与 trace parity；不能按父仓局部引用判为只有测试使用 |
| task-specific StarVLA runner 与 generic RL-Games runner | [starvla_tasks](../latency_bench/policy/starvla_tasks.py) 拥有 normalized chunk/frozen prefix；[starvla](../latency_bench/policy/starvla.py) 拥有单步 task/head decode；模型任务合同不同 | H1/H8/H16、continuous state、prefix 和离散/连续 head 解码；共享标签不意味着共享 action ID 表示 |
| 当前 seeded action noise 与 calibrated-clock 路径 | 父仓 [starvla](../latency_bench/policy/starvla.py) L280–282 消费 episode/frame noise seed；子模块 [action_noise](../third_party/starVLA/starVLA/model/modules/action_model/flow_matching_head/action_noise.py) 有本地实现；[generic policy suite](../tests/policies/test_generic_gymnasium_starvla_policy.py) L630–657 区分 scaled clock、ratio/prompt 改变 | batching/slot 不改变 per-episode/frame 噪声、全局 RNG 不被扰动；比例缩放通过，changed ratio/prompt 仍拒绝；同时记录父仓和本地补丁基准 |
| 固定版本环境和有理由的 lazy imports | [pyproject](../pyproject.toml)、[QUICK_START](../docs/QUICK_START.md)、MIKASA/model worker 安装合同使用隔离环境；SF cli integration 已共用 common config adapter | 不把 optional simulator/LeRobot/model 依赖提前导入基础环境；不把安装 target 的历史名称误当成模型算法名 |
| archived config、来源摘要、模型/profile revision 和已发布资产 | RF-032/RF-034 的原始来源有 provenance/test 消费者；预算 SOURCES 的 ref/revision/reference_return 是固定实验身份 | 既有正式基线与证据链；不按重复目录名、未提交状态、硬编码外观删除资产，不为本次文档新增 checksum 机制 |

## 渐进处理顺序与验收方式

1. **先收敛需要决定的语义**：RF-001、RF-008、RF-012、RF-015、RF-019、RF-021、RF-022、RF-034。源码已证明的差异与可能触发的风险分开；改变指标/prompt/checkpoint 行为要说明受影响实验，不能归类为行为不变的整理。
2. **完成最小独立修整**：讨论 RF-033 的旧合同后处理孤儿测试；RF-013/RF-014/RF-029/RF-030 只合并稳定定义和相同算式；RF-035/RF-036 在各 suite 内减少实现约束和导入污染。RF-007/RF-009 涉及接口/产物名，等待各自讨论结论。
3. **按调用责任收敛准备与编排**：RF-016、RF-023/RF-024/RF-025、RF-026/RF-027/RF-028。先保留统计粒度、路径、seed 和资源合同，再决定最小提取；如果提取需要新 framework、marker 或大量 mode flag，重新评估净收益。
4. **最后整理大型模块与历史入口**：RF-011、RF-017、RF-018 的拆分沿现有职责开展，不以行数为目标；RF-037/RF-038/RF-039/RF-040 根据入口/资产盘点和用户选择精简或归档。RF-003 与 RF-004 在性能/边界验证前保持现状，RF-006/RF-031 先明确支持合同。

上述顺序是建议，独立条目可以单独做；“依赖”指完成该条前必须弄清的合同，不要求先把整组重构做完。用户已确定**每批 5 条，按依赖排期**；每条仍保持能独立验收的行为边界，记录编号、决定、实际修改和一份主验收证据。遇到数值或实验语义差异时先保留当前行为、后移该条，不把整批合成一次大型重写。

**行为保持的改动**：对照同 checkpoint、resolved config、episode IDs、环境/latency seeds、profile输入和 FPS，核对 action、ready/apply frames、termination、returns 与输出字段。只运行拥有该合同的既有 suite；新增失败或边界改动才扩大检查，不做无关的全仓回归循环。

**语义修正**：先确定新规则，再用旧/新实现对照受影响输入，保留原结果及失效原因。需要正式重新评测时另行完成，不能仅凭 unit test 更新 accepted/published 状态。

**性能改动**：只有触及测量、热路径、batching、projection 或并发时，按 [infra](../docs/infra.md) 的固定硬件/输入、warmup 和交替对照验证；median 回退超过 5% 要调查，该值不是允许消耗的预算。测试保持语义不能替代真实性能验收。

## 尚待核实的线索与当前空白

- **C01/C02/C15 的全量退役判断**：本轮没有确认某个仍有入口的完整实验能力可以删除，也没有确认哪种 simulator/model dependency 已完全无用。历史/兼容字样、路径日期或 monkeypatch 不是退役证据。先盘点当前需求、第三方消费者和支持版本，再添加具体编号。
- **RF-003/RF-004**：projection 的实际成本、不同 epsilon 的实际分歧均未测量；不能承诺提速或宣布出现 temporal bug。
- **RF-006/RF-008/RF-031**：注入环境 FPS、不完整 button layout、任意工作目录 API 的支持面需要合同或真实边界验证。
- **RF-015/RF-019**：错误 prompt fallback 是否进入现有正式任务、checkpoint 返回 None 在当前 SF pin 中会进入从头训练分支，这是已确认的静态调用语义；是否曾影响某条实际训练仍未执行端到端复现。文档列明具体分支，不宣称已经污染所有实验。
- **RF-034/RF-037/RF-040**：来源文件发生变化的理由、历史 launcher 当前解析的协议与真正需要长期维护的手册范围，需要逐项追溯；本次没有远端读回资产或查询运行状态。
- **第三方与 outputs**：HumanoidBench 内部代码未本地核验，大型 checkpoint/dataset/历史报告没有做逐文件复现审计。若后续要收缩 fork、删除 artifact 或更换环境，补充专门的 provenance/consumer 核查；本轮不作 blanket 判断。

## 初次文档交付验收（2026-09-30，历史记录）

- [x] RF-001–RF-040 每条包含定位证据、消费者/测试归属、原因证据等级、建议边界、风险、验收和讨论事项；初次交付时候选本身全部保持未完成。
- [x] 16 类问题建立映射；收益/影响尚不确定的条目与未确认删除的类别明确说明。
- [x] 代码/文档链接目标、行号范围、条目编号和内部锚点已核对；README 已增加长期入口。
- [x] 本次仓库写入仅为新增 REFACTOR.md 和 README 索引；既有工作树改动保持原样。
- [x] 验证声明限定为静态审计和来源清单核对；没有以现有测试名代替测试执行结果。
