# LAGEN Release Cleanup

## 工作约定

- 在 `release` 分支开发，每次只完成一个正式模块任务。
- 每项任务包含必要的调用方迁移、所属测试和文档更新。
- 修改前后验证，检查完整 diff 后只暂存本任务的改动。
- 提供建议 commit message，等待用户 review，不自动 commit 或 push。
- 保留已有工作区改动。本文档与第一个正式任务一起交付，不单独作为任务。
- 需要远端验证时使用 `ssh ubuntu@34.131.176.130`，先检查机器状态。
- 原始日志和报告存入 `outputs/`，本文档只记录结论与证据位置。
- 临时操作脚本放在仓库外，完成后删除。

用户明确要求维护此文件，因此它是仓库内开发计划文档的例外。

## 定位和接口边界

LAGEN 提供可独立使用的延迟环境、数据采集、评测与 profiling。
Sample Factory 和 StarVLA 是论文参考实现，用户自行选择训练框架。

- 复用 `EnvAdapter`、`PolicyRunner`、观测/动作类型及现有执行器。
- 使用 `env_fps` 推导步长，保留 `obs_fps`、state、RGB、等待动作、排序和 chunk 规则。
- 环境和策略允许在用户自己的包中接入，无须修改 LAGEN 源码。
- 训练 wrapper 与评测执行器共享时间、采样和调度语义。
- 模型预处理、动作解码和模型状态属于策略适配。
- 区分 teacher 输入、student 输入、teacher 标签与实际执行动作。
- 特权状态和延迟诊断信息不自动成为 student 输入。
- 原始 rollout 与训练框架转换器分开，chunk 标签和 DAgger 由参考配方负责。
- 训练与评测可以使用不同延迟，但动作和输入表示必须兼容。
- 已发布 checkpoint、dataset 和 profile 保持可读。

## 目标目录与职责

| 位置 | 职责与整理方式 |
| --- | --- |
| `latency_bench/core/` | 共用类型、时钟、配置读取与合并 |
| `latency_bench/envs/base.py`、`factory.py` | 现有环境协议及内置/外部构造 |
| `latency_bench/envs/latency_wrapper.py` | 公共 Gymnasium 延迟 wrapper |
| `latency_bench/envs/latency_info.py` | CPU/GPU 共用的延迟 info 构造 |
| `latency_bench/envs/adapters/`、`tasks/` | 模拟器适配、官方任务注册及任务契约 |
| `latency_bench/policy/` | 现有策略协议、factory 和框架的推理适配 |
| `latency_bench/latency/` | 延迟采样、测量、profile 读取与拟合；sampler 操作归入 `samplers.py` |
| `latency_bench/scheduler/` | 观测时机、动作排序与保持；共享模拟时间线归入 `simulated_timeline.py` |
| `latency_bench/executors/` | 模拟、批量及 realtime 执行；事件构造归入 `_simulated_events.py` |
| `latency_bench/data/` | 通用 `rollout.py`、现有 schema、记录、存储与共享数据处理 |
| `latency_bench/integrations/sample_factory/` | 框架适配；`data/sf_teacher_*.py` 的专用部分迁为 `teacher_*.py` |
| `latency_bench/integrations/starvla/` | 框架加载、`lerobot_export.py`、`task_contracts.py`、评测回调与 resident DAgger |
| `latency_bench/eval/` | 配置解析、episode 编排、sweep 和结果报告 |
| `latency_bench/profiling/` | 测量流程、拟合编排、profile 获取与发布 |
| `latency_bench/logging/`、`analysis/` | 记录和输出；离线分析，包括长期漂移 |
| `training/common/` | 训练配置、checkpoint、SF 兼容逻辑；训练统计归入 `metrics_wrappers.py` |
| `training/*_sf/` | 官方 teacher 训练集成及已有 GPU 实现 |
| `scripts/starvla/`、`tasks/`、`experiments/` | VLA 参考训练、任务配方和长期实验工具 |
| `configs/examples/custom_env/` | 通用接入示例；不批量搬迁已有 `configs/train/` 和 `configs/training/` |
| `docs/`、`tests/`、`outputs/` | 用户契约、按源码归属的测试、验证 artifact |

公共通用路径不依赖训练代码或实验脚本。官方可选集成可以调用现有参考实现，按需导入。
不为此轮整理重写 GPU 内核。`paper/`、`lagen/`、`third_party/` 保留各自职责。

文件按职责命名。框架目录内不重复框架前缀。`factory.py` 只负责构造；`_` 前缀表示内部实现。
不新增 trainer 插件体系或万能 manager/helper 层。迁移后删除旧实现和无消费者的转发别名。

## 正式任务

| 编号 | 任务 | 状态 |
| --- | --- | --- |
| 1 | 公共延迟 wrapper、共享模拟调度、训练统计分离，迁移调用和测试 | 已实现并验证可运行范围；等待 review，存在基线验证限制 |
| 2 | 外部环境和策略构造；通用 seed 处理归到 `utils/seeding.py` | 待开始 |
| 3 | 通用采集入口，分离 SF 专用采集和导出 | 待开始 |
| 4 | StarVLA 数据交接和 integration 目录收敛 | 待开始 |
| 5 | Realtime 测量可信度及有证据的漂移修复 | 待开始 |
| 6 | 接受通用运行配置的 profiling，保留官方 preset | 待开始 |
| 7 | 薄 CLI：`eval`、`collect`、`convert`、`profile` | 待开始 |
| 8 | 分离基础、数据、模拟器和参考训练框架的安装依赖 | 待开始 |
| 9 | 用户文档、仓库外接入示例和整体验收 | 待开始 |

统一入口使用 `python -m latency_bench <command> --config ...`。
根目录 `__main__.py` 和 `cli.py` 只分发命令，参数解析分别归到 `eval/cli.py`、`data/cli.py` 和已有 `profiling/cli.py`。
Teacher/VLA 训练仍使用参考配方或用户自己的命令。默认本地运行，上传显式选择。

### 任务 1 的边界

- 将 `ActionLatencyWrapper` 移到环境公共层，要求任务显式传入 `noop_action`。
- 将公共延迟 info 和 profile sampler 操作从训练配置中分离。
- 将 `SimulatedResultTimeline`、`SimulatedWorkerCapacity` 移到 scheduler。
- 将执行器事件构造保留在 executor 层。
- 将 reward 和 episode 统计 wrapper 留在训练层。
- 迁移 CPU/GPU 参考集成、必要调用方和所属测试，保持既有执行轨迹。
- 不修改 realtime 执行循环、通用采集或 CLI。

## 验证规则

每项任务固定源码状态、配置、seed 和 artifact，先执行基线再执行修改版。
分别记录基线失败、新增回归和未验证项。测试放在所属模块，迁移测试不复制重复用例。

任务 1 覆盖 zero/fixed/profile、观测间隔内的动作应用、reward 累计、提前终止、排序、容量、chunk、显式 no-op 和 seed。
受影响的参考集成同时验证。完整训练和性能测试没有执行时不能声称通过。

Realtime 专项保留现有 GC、绝对时刻调度、进程隔离与共享内存。
现有 `collect → reset_state 确认 → episode 开始` 已形成屏障，不增加重复同步。
重型日志写出放在计时区间外，正式 profiling 默认关闭视频。

持续运行检查单个长 episode 和多个 episode 的对象保留、内存、阶段耗时、尾延迟、环境滞后与 deadline miss。
同硬件交替比较基线与修改版三次，至少运行 30 分钟，并覆盖现有最长正式测量时长。
先测轻量可控策略，再测固定真实模型。中位性能退化超过现有 5% 调查阈值必须处理。
新增漂移和尾延迟异常不受该阈值豁免。不能通过降速、删除慢样本或扣除平均开销掩盖 harness 缺陷。

最终验收使用仓库外项目完成接入、采集、模拟评测和 profiling，不修改 LAGEN，也不依赖两个参考训练框架。

## 任务 1 验证记录

### 实现结果

已按上述模块归属迁移 wrapper、延迟 info、四个有调用方的 sampler 函数、模拟时间线、容量和事件构造。
删除旧 wrapper/timeline 文件，以及无调用方的 `build_iid_profile_sampler`。
显式 no-op 使仅用于重建调度器的 `GymnasiumActionLatencyWrapper` 子类不再需要，已删除并迁移调用方。
新公共 wrapper 在构造时复制 no-op，避免调用方修改数组改变执行语义。
训练统计保留在 `training/common/metrics_wrappers.py`。Realtime 循环未改动。

13 个已有 wrapper 测试迁到 `tests/envs/test_action_latency_wrapper.py`；一个 sampler 测试迁到现有 latency suite。
新增长期契约测试覆盖离散/连续非零 no-op、terminated/truncated 提前停止，以及不导入参考训练框架的公共运行路径。
这些测试归属环境模块。既有 chunk 测试改从 `humanoidbench/contract.py` 读取常量，避免只为常量导入完整模拟器。

### 证据与限制

- 基线 revision：`641082d7`；完整值见 [baseline_revision.txt](outputs/cleanup/task1/baseline_revision.txt)。
- 在 `ubuntu@34.131.176.130` 的隔离源码目录验证，使用 latency 环境的 Python 3.10.22。
- `CUDA_VISIBLE_DEVICES` 设为空，`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，不占用已有 GPU 训练。
- [基线测试](outputs/cleanup/task1/baseline-tests-complete.log)：345 passed、20 failed、12 skipped。
- [修改版测试](outputs/cleanup/task1/candidate-tests.log)：364 passed、6 failed、12 skipped；无新增失败。
- 14 个既有 chunk 用例在解除不必要的模拟器导入后通过；新增参数化契约用例共 5 项，全部通过。
- [基线轨迹](outputs/cleanup/task1/baseline-traces.json)与[修改版轨迹](outputs/cleanup/task1/candidate-traces.json)逐项相同：72 组场景，各运行 2 个 episode。
- 轨迹比较覆盖离散动作、连续向量和 chunk；观测 stride 为 1/3；延迟为 fixed 0/2 或 uniform 0–4 帧；覆盖两种排序和两种保持规则，worker capacity 为 1。
- 时间线、容量、事件构造和训练 episode 统计的主体 AST 与迁移前相同。

剩余 6 项失败中，4 项来自已有 Sample Factory 版本契约不匹配（预期 2.1.1，源码版本 2.1.3）；2 项来自 HumanoidBench 的 JAX 依赖缺失。
本任务不修改框架兼容约束，也不伪造依赖以使测试通过。
这些失败是 release 验收的待解决项，不能将本次结果解释为全部训练集成已通过。

远端四张 GPU 在验证时均有训练负载。CUDA 用例、完整训练和性能比较未执行。
Realtime 的 30 分钟漂移验收属于任务 5，本任务未执行。
