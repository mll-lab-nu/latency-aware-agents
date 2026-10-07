# 全仓库重构计划与验收

共 40 项，**39 项实现或文档工作已完成，RF-003 测量后保留现有实现**。此状态限于已列条目，不代表当前全仓测试或真实环境发布验收全部通过。以下记录各项当前决定和验收证据；重叠批次按 RF 编号计数。

## 已确定原则

- 修正错误规则并删除被替代的实现；文档描述当前行为。
- 使用明确调用前提，直接读取必需字段；不添加内部防御检查、隐藏 fallback 或通用框架。
- 一次性操作脚本放仓库外，用完清理；主仓维护长期工具与实验代码。
- 数据、checkpoint、来源快照、正式结果报告按实验产物管理。
- 优先更新行为所属的既有测试；热路径改动做固定输入、同硬件对照。

## 历史合并验收分组

| 批次 | 条目 |
|---|---|
| 1 | [RF-002](#rf-002)、[RF-013](#rf-013)、[RF-029](#rf-029)、[RF-030](#rf-030)、[RF-036](#rf-036) |
| 2 | [RF-016](#rf-016)、[RF-033](#rf-033)、[RF-037](#rf-037)、[RF-038](#rf-038)、[RF-040](#rf-040) |
| 3 | [RF-001](#rf-001)、[RF-008](#rf-008)、[RF-014](#rf-014)、[RF-019](#rf-019)、[RF-034](#rf-034) |
| 4 | [RF-006](#rf-006)、[RF-007](#rf-007)、[RF-009](#rf-009)、[RF-015](#rf-015)、[RF-031](#rf-031) |
| 5 | [RF-021](#rf-021)、[RF-022](#rf-022)、[RF-026](#rf-026)、[RF-027](#rf-027)、[RF-028](#rf-028) |
| 6 | [RF-010](#rf-010)、[RF-023](#rf-023)、[RF-024](#rf-024)、[RF-025](#rf-025)、[RF-032](#rf-032) |
| 7 | [RF-012](#rf-012)、[RF-011](#rf-011)、[RF-020](#rf-020)、[RF-017](#rf-017)、[RF-018](#rf-018) |
| 8 | [RF-003](#rf-003)、[RF-004](#rf-004)、[RF-005](#rf-005)、[RF-035](#rf-035)、[RF-039](#rf-039) |

## 验收记录

- 首批 RF-002、013、029、030、036 已完成：[报告](../outputs/refactor/batch-01/report.md)。
- 历史合并批次以 HEAD `51ed017d` 加开始时的工作树为基准。该批相关用例按最终结果去重为 **934 passed**；CPU 前后对照动作/时序/指标一致，整集最大中位数增幅 **2.08%**。这些结果对应当时选定的用例与源码快照，不是当前全仓完整回归结果。
- 批次 2–8 的历史联合验收：[主报告](../outputs/refactor/consolidation/report.md)、[测试结果](../outputs/refactor/consolidation/verification-summary.json)、[删除清单](../outputs/refactor/consolidation/deleted-scripts.json)。
- RF-034 来源调查：17 个文件在 `aa728d0f` 中均匹配清单；差异来自 `558af6c2` 删除配置字段及 `bdafb10f` 添加 worker capacity。
- RF-004 边界调查：35 FPS、ready step 为 `1 + 5e-10` 时，时间路径在第 1 帧应用，整数路径等到第 2 帧。纳入统一容差回归。

## Profiling 拆分与收尾验收

- 拆分实现为 `a180efe3`：共享部署能力归入 `latency_bench/profiling`，实验协议归入 `scripts/experiments/sim2real`，固定参数由实验配置初始化并冻结到 manifest；旧包路径及一次性 timing 迁移入口已删除。
- 在该提交加本次测试/文档收尾改动上，profiling 相关 suites 复验 **227 passed**；训练入口、日志与 batched executor 用例 **75 passed**。两组选定用例有重叠，不累加，也不替代全仓回归。完整范围与命令见[收尾报告](../outputs/refactor/profiling-closeout/report.md)。
- RF-038 遗漏的补跑专属测试已清除，有效的评测局数与 seed 合同归入现有训练入口 suite。脚本目录收集为 **223 collected、2 errors**，剩余原因是验证环境未安装 `datasets`、`humanoid_bench`，详见[收集日志](../outputs/refactor/profiling-closeout/collection.log)。
- RF-040 同步当前指标语义，并明确旧批次证据的源码与范围。真实模型 GPU 性能、模拟器/训练链路、干净 checkout 的完整安装与复现矩阵仍待单独验收。

## Data 原地精简验收（2026-10-01）

- 实现完成：`latency_bench/data` 保持原地组织，从 23 个 Python 文件、6,839 行收敛至 19 个文件、5,584 行；ghost-trail 合一，SF 阶段归拢，LeRobot 共用写出，删除无消费者的协议和路径。
- 四卡 RTX PRO 6000 主机上，同环境 CPU 回归由 **455 passed / 3 failed** 变为 **459 passed / 1 failed**。两处 fixture 补齐既有字段；剩余 StarVLA language-key 不一致是基线已有问题。
- 11 组转换场景、198 个产物逐项一致；真实 CUDA Flappy 配固定小网络的 batch 4/16 对照中，accepted state、metadata、数据行、PNG 字节与峰值显存一致。性能范围和限制见[验收报告](../outputs/refactor/data-cleanup/report.md)。

<a id="batch-02"></a>

## 历史独立批次记录：10条验收（2026-09-30）

本批完成 **RF-005、RF-010、RF-016、RF-021、RF-022、RF-023、RF-024、RF-025、RF-032、RF-040**：七项代码收敛、三项Document；加首批共15项。用户在本对话将后续批次改为每批10条。

- 基准为HEAD `51ed017dabd75d29ac26f52619c969e207347678` 加开始时完整工作树。四卡RTX PRO 6000主机建立独立before/after/venv，Python3.10.21，CUDA_VISIBLE_DEVICES为空，新增依赖仅入临时venv。
- 主套件before **293通过、3失败**，after **303通过、同样3失败**；原296例结果及失败信息一致，新增10例通过。缺失backfill模块的collection error另行对照，也前后一致。详情见[报告](../outputs/refactor/batch-02/report.md)。
- 180份study和四份generic config共184份全部字段一致，frozen splits、五次partitions与55个latency seeds一致，见[配置对照](../outputs/refactor/batch-02/config-comparison.json)。没有训练、GPU/model benchmark、正式评测、发布或自动commit。
- RF-012/014/018/039未在本批实施。同期另有源码、脚本和测试清理及清单结构重写；这些改动保留，不计入本批通过声明。验收与[本批增量diff](../outputs/refactor/batch-02/implementation.patch)基于冻结快照；边界见[并行改动记录](../outputs/refactor/batch-02/concurrent-changes.json)。

## 条目

<a id="rf-001"></a>

### RF-001 — 统一 episode 增量指标

- [x] **状态：已完成**
- **决定：** 完整记录和 streaming 共用 EpisodeMetricAccumulator。最终步 game_score 优先于 score，最终步缺失则为 None；raw return 取最近有效步，顶层优先于 episode_extra_stats，同层按 return_raw、raw_return、episodic_raw_return、episode/raw_return 顺序。task metrics 取最终步，chunk metrics 增量聚合。
- **代码 / 验收归属：** [metrics.py](../latency_bench/logging/metrics.py)；[test_batched_simulated_executor.py](../tests/executors/test_batched_simulated_executor.py)。

<a id="rf-002"></a>

### RF-002 — live scheduler 与 chunk peek 重复实现 ready-event 选择

- [x] **状态：已完成**
- **决定：** live update 与只读 peek 共用 ready-event 选择；FIFO/latest-ready 的顺序与丢弃规则保持一致。
- **代码 / 验收归属：** [action_queue.py](../latency_bench/scheduler/action_queue.py)；[test_action_chunk.py](../tests/scheduler/test_action_chunk.py)。

<a id="rf-003"></a>

### RF-003 — action-prefix projection 每次都深拷贝完整 scheduler

- [x] **状态：保留（测量完成）**
- **决定：** 代表性 pending=0/4/16、H8×7/H16×43 的 CPU 投影中位数为 48–462 µs，projection 与 live 动作一致。本轮保留 deepcopy。
- **代码 / 验收归属：** [action_queue.py](../latency_bench/scheduler/action_queue.py)；[test_action_chunk.py](../tests/scheduler/test_action_chunk.py)。

<a id="rf-004"></a>

### RF-004 — 统一 raw-frame 舍入与容差

- [x] **状态：已完成**
- **决定：** 统一 core clock 的 ceil 舍入容差；已有整数 ready frame 为权威。时间比较与日志回填采用相同的帧误差尺度。
- **代码 / 验收归属：** [clock.py](../latency_bench/core/clock.py)；[test_action_queue.py](../tests/scheduler/test_action_queue.py)。

<a id="rf-005"></a>

### RF-005 — 明确时间字段单位和时钟来源

- [x] **状态：已完成（Document）**
- **决定：** 字段说明明确 perf_counter 起止值为秒、duration 为毫秒以及 clock domain；字段名和数值格式保持。
- **代码 / 验收归属：** [types.py](../latency_bench/core/types.py)；[realtime_pipeline_logging.md](../docs/realtime_pipeline_logging.md)。
- **历史第二批验收：** 计时字段单位与clock domain已按写入点说明；字段名、数值和测量边界保持，measurement/profiler用例通过。 [证据](../outputs/refactor/batch-02/report.md)，验收范围见[批次记录](#batch-02)。

<a id="rf-006"></a>

### RF-006 — 明确配置与注入环境的时钟合同

- [x] **状态：已完成**
- **决定：** 评测配置提供 FPS，注入环境由调用者按同一 FPS 创建；在构造入口说明前提，不增加内部检查。
- **代码 / 验收归属：** [factory.py](../latency_bench/executors/factory.py)；[test_executor_factory.py](../tests/executors/test_executor_factory.py)。

<a id="rf-007"></a>

### RF-007 — 删除无效的 IPC drain 时间参数

- [x] **状态：已完成**
- **决定：** 删除 drain_completed 的无效 now_ms 参数，同步实际调用方和测试替身。
- **代码 / 验收归属：** [pool.py](../latency_bench/executors/realtime/pool.py)；[test_realtime_executor.py](../tests/executors/test_realtime_executor.py)。

<a id="rf-008"></a>

### RF-008 — 按真实按钮布局映射动作

- [x] **状态：已完成**
- **决定：** 按环境真实按钮名映射；取不到布局时报错，未知名直接查表失败。
- **代码 / 验收归属：** [deadly_corridor.py](../latency_bench/envs/adapters/deadly_corridor.py)；[test_deadly_corridor_env_adapter.py](../tests/envs/adapters/test_deadly_corridor_env_adapter.py)。

<a id="rf-009"></a>

### RF-009 — 统一 latency 日志文件

- [x] **状态：已完成**
- **决定：** 只写 latencies.jsonl，更新现有分析、日志测试和文档。
- **代码 / 验收归属：** [trajectory_logger.py](../latency_bench/logging/trajectory_logger.py)；[test_logging_records.py](../tests/logging/test_logging_records.py)。

<a id="rf-010"></a>

### RF-010 — single 与 batched executor factory 只在部分 assembly 上重复

- [x] **状态：已完成**
- **决定：** 共用 single/batched 的 policy/pool 组装；各执行器保留自己的 slot、seed、sampler 与资源生命周期。
- **代码 / 验收归属：** [factory.py](../latency_bench/executors/factory.py)；[test_executor_factory.py](../tests/executors/test_executor_factory.py)。
- **历史第二批验收：** single/batched在原构造位置共用私有policy/pool准备；设备、batch专属检查与cleanup保持。pool四组合用例通过。 [证据](../outputs/refactor/batch-02/report.md)，验收范围见[批次记录](#batch-02)。

<a id="rf-011"></a>

### RF-011 — 按边界拆分通用 StarVLA 适配器

- [x] **状态：已完成**
- **决定：** 把模型输入配置和动作解码分别归入内部模块；factory 和 resident evaluator 保留实际构建入口，迁移调用方。
- **代码 / 验收归属：** [starvla.py](../latency_bench/policy/starvla.py)；[test_starvla_policy.py](../tests/policies/test_starvla_policy.py)。

<a id="rf-012"></a>

### RF-012 — 共用 StarVLA 模型输入计划

- [x] **状态：已完成**
- **决定：** 模型输入计划由一个 resolver 解析；环境 frame stack 与推理选帧共用 raw span/stride、packed、KV memory、stitch 规则。
- **代码 / 验收归属：** [_starvla_input.py](../latency_bench/policy/_starvla_input.py)；[test_starvla_policy.py](../tests/policies/test_starvla_policy.py)。

<a id="rf-013"></a>

### RF-013 — 让 policy action label 引用 canonical label 表

- [x] **状态：已完成**
- **决定：** policy action labels 引用 data.action_layouts 的定义。
- **代码 / 验收归属：** [action_layouts.py](../latency_bench/data/action_layouts.py)；[test_rollout_action_layouts.py](../tests/data/test_rollout_action_layouts.py)。

<a id="rf-014"></a>

### RF-014 — 共用先除后乘的归一化核心

- [x] **状态：已完成**
- **决定：** task 与 generic 共用先除后乘的 float32 归一化；常量维归零、裁剪和 batch 维保持。
- **代码 / 验收归属：** [state_normalization.py](../latency_bench/data/state_normalization.py)；[test_state_normalization.py](../tests/data/test_state_normalization.py)。

<a id="rf-015"></a>

### RF-015 — 动态提示词使用明确时钟

- [x] **状态：已完成**
- **决定：** 动态提示词使用评测配置中的 FPS 和 observation 的明确延迟；删除 35 FPS、0 ms 猜测。map、neutral/base 路径按其合同运行。
- **代码 / 验收归属：** [starvla_prompts.py](../latency_bench/policy/starvla_prompts.py)；[test_generic_gymnasium_starvla_policy.py](../tests/policies/test_generic_gymnasium_starvla_policy.py)。

<a id="rf-016"></a>

### RF-016 — 切开StarVLA evaluation对training CLI的依赖

- [x] **状态：已完成**
- **决定：** evaluation 与 training 共用 scripts.starvla.runtime；训练超参数继续由 train_task 持有。
- **代码 / 验收归属：** [runtime.py](../scripts/starvla/runtime.py)；[test_starvla_train_task.py](../tests/scripts/test_starvla_train_task.py)。
- **历史第二批验收：** runtime持有三个process/evaluation helpers和canonical task输入键；训练参数留train_task。真实调用方迁移，配置/命令用例及独立导入探测通过。 [证据](../outputs/refactor/batch-02/report.md)，验收范围见[批次记录](#batch-02)。

<a id="rf-017"></a>

### RF-017 — 将teacher rollout的大型双阶段collector恢复成清晰编排边界

- [x] **状态：已完成**
- **决定：** collector 保留 probe→replay 编排，probe 状态持久化和最终报告按职责整理；保持 accepted seeds、shards、resume、attempt 和数据输出。
- **代码 / 验收归属：** [sf_teacher_rollout.py](../latency_bench/data/sf_teacher_rollout.py)；[test_sf_teacher_dataset_export.py](../tests/data/test_sf_teacher_dataset_export.py)。

<a id="rf-018"></a>

### RF-018 — 整理checkpoint文件内部边界，不改变保留生命周期

- [x] **状态：已完成**
- **决定：** checkpoint 模块按 native IO/RNG、SF loading、lease/pruning 分区；lease 和并发清理保持集中。
- **代码 / 验收归属：** [checkpoints.py](../training/common/checkpoints.py)；[test_sample_factory_checkpoint_io.py](../tests/training/test_sample_factory_checkpoint_io.py)。

<a id="rf-019"></a>

### RF-019 — checkpoint 加载失败直接报错

- [x] **状态：已完成**
- **决定：** 选定 checkpoint 读取失败立即抛出；不尝试旧文件、不返回 None。新训练初始化明确跳过恢复，resume 必须加载成功。
- **代码 / 验收归属：** [checkpoints.py](../training/common/checkpoints.py)；[test_sample_factory_checkpoint_io.py](../tests/training/test_sample_factory_checkpoint_io.py)。

<a id="rf-020"></a>

### RF-020 — 具名传递 Demon 筛选阈值

- [x] **状态：已完成**
- **决定：** Demon threshold 通过具名参数传递，删除 filter_kwargs；保留 CLI 和相同的接受集合、统计及 metadata。
- **代码 / 验收归属：** [filters.py](../latency_bench/data/filters.py)；[test_sf_teacher_dataset_export.py](../tests/data/test_sf_teacher_dataset_export.py)。

<a id="rf-021"></a>

### RF-021 — 统一 Flappy 预算实验的 55-run 总体定义

- [x] **状态：已完成**
- **决定：** Flappy 预算协议共用 55-run 总体及复用 runs 定义。
- **代码 / 验收归属：** [budget.py](../scripts/experiments/sim2real/budget.py)；[test_sim2real.py](../tests/scripts/test_sim2real.py)。
- **历史第二批验收：** 当时集中定义55-run总体及复用run列表；original fit顺序、两frozen splits、五次partitions及55个latency seeds前后一致。 [证据](../outputs/refactor/batch-02/report.md)，验收范围见[批次记录](#batch-02)。

<a id="rf-022"></a>

### RF-022 — 让两 episode session 宽度只有一个内部定义

- [x] **状态：已完成**
- **决定：** 每 run 两 episode 的 session 宽度由一处定义，grouped replay、seed 和完成计数共同使用。
- **代码 / 验收归属：** [replay.py](../scripts/experiments/sim2real/replay.py)；[test_sim2real.py](../tests/scripts/test_sim2real.py)。
- **历史第二批验收：** 当时在回放模块集中定义每run两局，调用方共享该值。session sampler、seed stride、trace/DONE与ten-slot公开ID用例通过。 [证据](../outputs/refactor/batch-02/report.md)，验收范围见[批次记录](#batch-02)。

<a id="rf-023"></a>

### RF-023 — 收敛 generic profile 与 sim2real 的重复 runtime overlay

- [x] **状态：已完成**
- **决定：** 共用 clock/obs stride 和 latency summary 的相同变换；运行坐标和各任务专属配置由调用方提供。
- **代码 / 验收归属：** [runtime_configs.py](../latency_bench/profiling/runtime_configs.py)；[test_profile_protocol.py](../tests/profiling/test_profile_protocol.py)。
- **历史第二批验收：** 仅共享clock/obs stride与profile scalar变换，IID/Temporal不读取summary；两builder差异留caller。184份config全部字段一致。 [证据](../outputs/refactor/batch-02/report.md)，验收范围见[批次记录](#batch-02)。

<a id="rf-024"></a>

### RF-024 — 共用 resident replay 配置准备，保留资源生命周期归属

- [x] **状态：已完成**
- **决定：** resident replay 共用配置准备和 executor 生命周期；backend、warmup、close 的所有者明确。
- **代码 / 验收归属：** [replay.py](../scripts/experiments/sim2real/replay.py)；[test_sim2real.py](../tests/scripts/test_sim2real.py)。
- **历史第二批验收：** 只共享resolve→checkpoint inputs→config seed，真实Python/NumPy/Torch下一抽样验证run seed并恢复测试RNG。backend/warmup计时/close保持caller。 [证据](../outputs/refactor/batch-02/report.md)，验收范围见[批次记录](#batch-02)。

<a id="rf-025"></a>

### RF-025 — 合并 grouped replay 的 episode 持久化公共路径

- [x] **状态：已完成**
- **决定：** grouped replay 共用 episode 结果持久化；完整 trace 写完再产生 DONE 和 publication。
- **代码 / 验收归属：** [replay.py](../scripts/experiments/sim2real/replay.py)；[test_sim2real.py](../tests/scripts/test_sim2real.py)。
- **历史第二批验收：** 五参数helper只flatten/run/flush/return；logger安装、cleanup、identity映射、publication与计时保持caller。grouped/ten-slot/resume用例通过。 [证据](../outputs/refactor/batch-02/report.md)，验收范围见[批次记录](#batch-02)。

<a id="rf-026"></a>

### RF-026 — 显式 timing 坐标

- [x] **状态：已完成**
- **决定：** timing producer 写明 stage、model、split、fit_runs、method、run_ids；报告直接读语义字段。报告通过显式 timing-root 读取已有独立计时产物，原记录保留。
- **代码 / 验收归属：** [budget_report.py](../scripts/experiments/sim2real/budget_report.py)；[test_sim2real.py](../tests/scripts/test_sim2real.py)。

<a id="rf-027"></a>

### RF-027 — 共用 case 误差计算

- [x] **状态：已完成**
- **决定：** 共用 case 归一化误差算式；paired/unpaired bootstrap、模型等权和跨 split 样本 SD 仍由各报告计算。
- **代码 / 验收归属：** [budget_report.py](../scripts/experiments/sim2real/budget_report.py)；[test_sim2real.py](../tests/scripts/test_sim2real.py)。

<a id="rf-028"></a>

### RF-028 — 保留单一配置解析实现

- [x] **状态：已完成**
- **决定：** 一个配置 resolver 供独立入口调用；保持现有调用，不增加 resolved 标记、模式参数或包装层。
- **代码 / 验收归属：** [config.py](../latency_bench/eval/config.py)；[test_eval_suites.py](../tests/eval/test_eval_suites.py)。

<a id="rf-029"></a>

### RF-029 — 将 Sample Factory policy type 分类集中到单一出处

- [x] **状态：已完成**
- **决定：** Sample Factory policy 类型由 policy factory 单独定义。
- **代码 / 验收归属：** [factory.py](../latency_bench/policy/factory.py)；[test_policy_factory.py](../tests/policies/test_policy_factory.py)。

<a id="rf-030"></a>

### RF-030 — 抽取 fixed-latency sweep 的配置准备，不动运行汇总层

- [x] **状态：已完成**
- **决定：** fixed-latency sweep 共用配置准备，运行与汇总由各入口持有。
- **代码 / 验收归属：** [sweeps.py](../latency_bench/eval/sweeps.py)；[test_eval_suites.py](../tests/eval/test_eval_suites.py)。

<a id="rf-031"></a>

### RF-031 — 明确根目录调用约定

- [x] **状态：已完成**
- **决定：** shell 入口设置仓库根目录；直接 Python 调用以仓库根目录为前提。
- **代码 / 验收归属：** [profile.sh](../scripts/profile.sh)；[test_profile_protocol.py](../tests/profiling/test_profile_protocol.py)。

<a id="rf-032"></a>

### RF-032 — 保留 archived eval 来源配置

- [x] **状态：已完成（Document）**
- **决定：** 18 份来源 YAML 原样保留；明确 runtime overlay 的实际覆盖字段和模型/任务差异。
- **代码 / 验收归属：** [assets.py](../scripts/experiments/sim2real/assets.py)；[test_sim2real.py](../tests/scripts/test_sim2real.py)。
- **历史第二批验收：** 18份archived YAML的per-task差异与overlay职责已写入sim2real README；文件/registry保持原样，按Document完成。 [证据](../outputs/refactor/batch-02/report.md)，验收范围见[批次记录](#batch-02)。

<a id="rf-033"></a>

### RF-033 — 删除孤儿 watchdog 测试

- [x] **状态：已完成**
- **决定：** 删除 formal_chain_watchdog 孤儿测试；当前恢复、发布与 profile 配置行为由实际入口测试保护。
- **代码 / 验收归属：** [test_starvla_collect.py](../tests/scripts/test_starvla_collect.py)。

<a id="rf-034"></a>

### RF-034 — 恢复来源快照内容

- [x] **状态：已完成**
- **决定：** 恢复 17 个来源文件到 aa728d0f 中匹配既有清单的原始内容；运行修改留在运行配置层。
- **代码 / 验收归属：** [source_map.json](../configs/examples/gymnasium/pi05_sources/source_map.json)；[test_pi05_environment_overlays.py](../tests/core/test_pi05_environment_overlays.py)。

<a id="rf-035"></a>

### RF-035 — 以行为测试替换实现细节断言

- [x] **状态：已完成**
- **决定：** 删除无行为合同和重复覆盖的断言；固定安装栈、动作输出、RNG 隔离等要求在现有行为测试中验证。
- **代码 / 验收归属：** [README.md](../tests/README.md)；[test_eval_bash_scripts.py](../tests/scripts/test_eval_bash_scripts.py)。

<a id="rf-036"></a>

### RF-036 — 第三方导入的测试作用域会修改全局 sys.path

- [x] **状态：已完成**
- **决定：** 第三方导入测试使用作用域内的路径修改并验证导入来源。
- **代码 / 验收归属：** [test_pi05_environment_overlays.py](../tests/core/test_pi05_environment_overlays.py)。

<a id="rf-037"></a>

### RF-037 — 共用 MIKASA conditioned H8 阶段

- [x] **状态：已完成**
- **决定：** MIKASA conditioned H8 共用显式参数的 collect/train/eval 入口，Fixed-N 正式流程复用共同阶段。
- **代码 / 验收归属：** [run_conditioned_h8.sh](../scripts/tasks/mikasa_intercept_grab_fast/run_conditioned_h8.sh)；[test_mikasa_h8.py](../tests/scripts/test_mikasa_h8.py)。

<a id="rf-038"></a>

### RF-038 — 全仓一次性实验脚本清理

- [x] **状态：已完成**
- **决定：** 全仓直接删除 agent 为一次监控、调度、补跑、迁移、收尾生成的临时脚本及专用配置、测试、失效引用。通用能力使用已有入口和参数。临时脚本放仓库外，用完清理。
- **代码 / 验收归属：** [AGENTS.md](../AGENTS.md)；[run_manifest_jobs.py](../scripts/experiment_runner/run_manifest_jobs.py)。
- **收尾：** 删除仍导入已移除补跑模块的孤儿测试；显式评测局数与 seed 的有效合同移入 [test_starvla_train_task.py](../tests/scripts/test_starvla_train_task.py)，直接测试现行 `eval_config`。

<a id="rf-039"></a>

### RF-039 — 统一发布与完成记录复用

- [x] **状态：已完成**
- **决定：** 发布由现行 publisher 的 publish_folder 负责；成功核对后写 receipt，接续直接复用确切版本。删除一次性 H8 publisher 及其旧路径迁移分支。
- **代码 / 验收归属：** [publish_prefix63.py](../scripts/tasks/mikasa_intercept_grab_fast/publish_prefix63.py)；[test_mikasa_h8.py](../tests/scripts/test_mikasa_h8.py)。

<a id="rf-040"></a>

### RF-040 — 整理当前协议和运行入口

- [x] **状态：已完成（Document）**
- **决定：** 当前手册和 README 指向正式入口及当前协议；结果、模型和来源证据保持各自归属，状态只在验证后更新。
- **代码 / 验收归属：** [README.md](../program/README.md)。
- **收尾：** [infra.md](../docs/infra.md) 同步 RF-001 的统一指标语义；验收记录区分历史批次、profiling 拆分与当前检查。历史 HANDOFF 的逐行来源核实仍属于实验记录整理，不作为当前运行状态。
- **历史第二批验收：** 7D/71D/63D版本及9/24→9/25接续清楚，旧部署按历史日期读。HANDOFF绑定9/30收录原稿，原状态日期/逐行数值来源待核实，原表保留。 [证据](../outputs/refactor/batch-02/report.md)，验收范围见[批次记录](#batch-02)。
