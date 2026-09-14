# Extreme Parkour：预算对齐与性能记录

日期：2026-09-14。按用户要求，当前只保留 CPU 优化，PhysX 使用 Isaac Sim 安装自带的默认库。自定义库加载入口、kernel 补丁、构建脚本和 simulate/fetch overlap 已从训练代码移除；原安装库从未被替换。本文后半部分为历史记录。

## 当前保留的 CPU 优化与收益

`ActionLatencyBatch` 用 NumPy 批量维护各环境的时钟、worker 占用和待执行命令；动作张量保留在原设备上。每个环境的 sampler、随机数调用顺序、latest/FIFO、worker 释放与动作就绪时序、action chunk 和 reset 语义保持一致。Parkour 采集器每控制步只构造一次全批调度元数据。此前的动作批量操作、缓冲区复用、Kit 8 线程和双卡预算改动继续保留。

OFT 实测：双卡、每卡 3,072 env，预热 10 轮、测量 30 轮，包含原 runner 的 rollout、PPO、DAgger 和 bookkeeping，关闭日志和 checkpoint 写入；每轮取较慢 rank 的 wall time，再取中位数。

| 版本 | 秒/轮，中位数 | 测量期间实际全局控制步 |
|---|---:|---:|
| 本次 CPU 批量调度改动前，原安装 PhysX | 13.591973 | 22,053,126 |
| CPU 批量调度，原安装 PhysX | 11.614363 | 22,052,538 |

整轮耗时减少 **14.5%**，吞吐提高约 **17%**。控制步已经跨 rank 汇总，不能再相加。[基线记录](../../outputs/latency_rl/parkour/kernel_scheduler_20260914/oft_baseline_a/)、[CPU 优化记录](../../outputs/latency_rl/parkour/kernel_scheduler_20260914/oft_scheduler_a/)。这些是已完成的 CPU-only 对照，本次删除实验代码没有重新跑整轮性能矩阵。PI 的 CPU-only 完整性能尚未单独测量。

CPU-only 双卡轨迹检查已通过：状态、动作、观测、奖励、done 和调度元数据逐位一致；仅 contacts/contact_history 沿用原库重复运行的末位舍入容差 rtol=5e-7、atol=0。[基线轨迹](../../outputs/latency_rl/parkour/kernel_scheduler_20260914/oft_baseline_trace/)、[CPU 优化轨迹](../../outputs/latency_rl/parkour/kernel_scheduler_20260914/oft_scheduler_trace/)。回退后的 Parkour wrapper 与当时 CPU-only 测量冻结的源码一致；调度器清理了未使用的预计算接口、缓存及初始化参数，调度路径保持一致。

## 其他任务能否直接受益

CPU 批量调度不依赖 Parkour 或 PhysX；是否直接受益取决于实际入口是否使用 `ActionLatencyBatch`。

| 小模型延迟训练 | 当前是否共用 | 代码入口 |
|---|---|---|
| Parkour native PPO | 是 | [latency_vecenv.py](../../third_party/Isaaclab_Parkour/scripts/rsl_rl/latency_vecenv.py) |
| HAIC native | 是，`SimpleEnv.command_latency` 直接使用 | [helpers.py](../../third_party/HAIC/scripts/helpers.py)、[base.py](../../third_party/HAIC/active_adaptation/envs/base.py) |
| Mikasa native | 是，经 `TorchActionLatencyWrapper` 使用 | [get_dataset_collectors_ckpt.py](../../third_party/MIKASA-Robo/mikasa_robo_suite/vla/dataset_collectors/get_dataset_collectors_ckpt.py)、[action_latency.py](../../training/common/action_latency.py) |
| Balance native FastTD3 / Sample Factory | 否，仍使用逐环境 `GymnasiumActionLatencyWrapper` / `ActionScheduler` | [integration.py](../../training/gymnasium_sf/integration.py)、[latency_wrapper.py](../../training/common/latency_wrapper.py) |

HAIC、Mikasa 的上述训练入口无需另行接入即可使用这次 CPU 改动，但尚未测量各自端到端收益，不能套用 Parkour 的 17%。Balance 需要在 wrapper/向量环境边界做适配，本次没有扩展修改范围。

采集与 VLA eval 需要分别看入口：HAIC bootstrap 和 Mikasa latency pipeline 采集也使用该批量调度器；通用 VLA latency eval 仍经 executor 使用逐 slot 的 `ActionScheduler`，不会自动获得这项收益。普通无延迟采集及 HAIC 官方 DAgger 采集也不因此改变。

## 本次撤回与验证

- 删除自定义 PhysX 加载参数、GPU kernel 补丁和构建脚本；本地 fork 的三个修改文件恢复干净，并移入被 git 忽略的实验留档。
- 删除物理 step hook、下一帧预计算 API/缓存和只覆盖这些实验入口的测试参数化。
- 保留 CPU 调度语义测试。回退后 scheduler、native PPO、启动/pipeline 测试共 **24 passed in 4.24s**；另以旧调度器为参照的 **96 种配置 × 70 帧**差分检查通过。

被撤回的合并候选曾测得 OFT 10.489835 秒/轮、PI 10.298025 秒/轮，但未通过完整轨迹逐位检查。这些结果不属于当前 CPU-only 版本，不能作为当前收益。[历史实验说明](../../outputs/latency_rl/parkour/kernel_scheduler_20260914/reverted_cpu_only/OPTIMIZE.md)、[合并候选统计](../../outputs/latency_rl/parkour/kernel_scheduler_20260914/combined_end_to_end.json)、[浮点差异诊断](../../outputs/latency_rl/parkour/kernel_scheduler_20260914/cuda_roundoff_diagnosis.json)。源码及补丁的留档不参与训练或测试。

## 此前 PhysX 实验回退：结果与原因

此前 profile 中最大的单个 GPU 热点是 `convexCoreTrimeshNphase_Kernel32`，负责机器人圆柱碰撞体与地形三角网格的窄相检测：一轮调用 480 次、累计约 4.46 秒，占该 profile 的 GPU 活跃时间约 59%。**这证明了单个 GPU 热点，尚未证明它是整轮训练最值得优化的瓶颈。** GPU 活跃时间占比不能直接当作训练 wall time 占比。

当前 Parkour 已经使用 GPU PhysX；基线记录为 `device: cuda:0`，加载原安装的 `libPhysXGpu_64.so`。[运行记录](../../outputs/latency_rl/parkour/strict_physics_20260914/ddp_baseline_clean/rank_0/manifest.json)

| 实验 | 已测结果 | 回退原因 |
|---|---|---|
| CUDA equal cache 偏好 | 目标内核约快 10%；双卡整轮中位数 13.856 → 13.946 秒，增加约 0.65% | 没有整轮收益，删除设置缓存偏好的实验代码 |
| BV32 shared 分支 | 目标内核 9.911744 → 10.111552 ms，约慢 2.0% | 没有采用依据，删除补丁 |
| 接触对象原位构造 | sm_120 编译元数据的 stack/thread 10,568 → 5,560 B；双卡输出比较通过 | 整轮训练收益未验证，删除候选补丁，不作为已完成优化 |

缓存候选的计时使用同一原安装库、双卡、每卡 3,072 env；每轮 24 次决策、最多 5 个控制步/决策、4 个物理子步/控制步，dt=0.005。每组 10 轮预热、30 轮测量，直接运行原 runner 的 rollout、PPO、DAgger 和 bookkeeping，关闭日志写入和 checkpoint；按每轮较慢 rank 的 wall time 取中位数。两组实际全局控制步分别为 22,052,540 和 22,052,824，没有通过减少预算提速。[双卡计时比较](../../outputs/latency_rl/parkour/strict_physics_20260914/benchmark_comparison.json)

第一组与 CPU 编译重叠的基线已整组排除。后续比较的编译只在预热阶段发生，正式测量没有编译重叠；另一次诊断确认两 rank 在第一次 act 前仍为 equal cache，未发生设置失效。[编译时间记录](../../outputs/latency_rl/parkour/strict_physics_20260914/inplace_build_timing.json)、[rank 0 缓存诊断](../../outputs/latency_rl/parkour/strict_physics_20260914/cache_context_diagnostic/rank_0/benchmark_cache.json)、[rank 1 缓存诊断](../../outputs/latency_rl/parkour/strict_physics_20260914/cache_context_diagnostic/rank_1/benchmark_cache.json)

内核耗时是 Nsight Compute 单次采样的诊断值，不能代替完整训练计时。[BV32 指标比较](../../outputs/latency_rl/parkour/strict_physics_20260914/kernel_comparison.json)、[缓存内核报告](../../outputs/latency_rl/parkour/strict_physics_20260914/kernel_equal.ncu-rep)、[原位构造资源比较](../../outputs/latency_rl/parkour/strict_physics_20260914/inplace_resources.json)

原库 local/global 访存中，local 约占 97.0%；**这是访存流量占比，不是训练耗时占比**。它说明该内核的局部存储值得排查，但不足以证明调整缓存或缩小 stack 能加速整轮训练。[访存统计](../../outputs/latency_rl/parkour/strict_physics_20260914/local_memory_traffic.json)

浮点验收没有成为采用门槛的阻碍：原库重复运行的接触力也存在最多 4 ULP 的末位差异，最大绝对误差 1.52587890625e-5、最大相对误差 4.6823657839922816e-7。仅 contacts/contact_history 允许 rtol=5e-7、atol=0，其余状态、观测、奖励、done 和调度元数据严格比较。三个候选均通过双卡 1,250 份快照比较；输出通过不能代替速度通过。[原库浮点统计](../../outputs/latency_rl/parkour/strict_physics_20260914/contact_roundoff.json)、[缓存验收](../../outputs/latency_rl/parkour/strict_physics_20260914/ddp_cache_compare_0.log)、[BV32 验收](../../outputs/latency_rl/parkour/strict_physics_20260914/ddp_shared_compare_0.log)、[原位构造验收](../../outputs/latency_rl/parkour/strict_physics_20260914/ddp_inplace_compare_0.log)

### 已回退的范围

- 删除本轮新增的 `probe_physics.py`、`compare_physics_traces.py`，以及既有测试文件中仅服务该比较器的两项测试和相关导入。
- 删除 `physx_bv32_shared.patch`、`physx_contact_inplace.patch`；恢复本轮临时 PhysX 源码目录中的候选修改。
- 正式启动脚本没有加入 cache 设置或候选库加载参数；安装目录中的原 PhysX GPU 库从未被替换，继续使用原库。
- 保留此前批量动作操作、缓冲区复用、Kit 8 线程、GC 参数，以及双卡预算、实际控制步统计和 W&B 日志改动。

本轮复杂度来自过早展开缓存、BVH、初始化和整库重编译，还增加了工具链、ABI 与输出验收工作。局部指标改善没有得到整轮收益支撑，因此回退这些实验代码。后续只有匹配预算的整轮计时证明收益，才值得引入底层库修改。

原始测量和构建产物保存在被 git 忽略的 [证据目录](../../outputs/latency_rl/parkour/strict_physics_20260914/)，已撤文件的副本保存在其中的 [撤回留档目录](../../outputs/latency_rl/parkour/strict_physics_20260914/reverted_experiments/)。留档不参与训练或测试。

回退后验证：`tests/training/test_action_latency.py`、`parkour_test/test_command_ppo.py` 和 `tests/scripts/test_parkour_latency_pipeline.py` 共 **20 passed in 3.60s**。训练与测试代码已无实验脚本、比较器、cache 设置或候选库加载参数的引用；两个临时 PhysX 源码目录均恢复干净，相关文件的 `git diff --check` 通过。

## 此前预算与性能结论

- 默认训练改为双卡 DDP，每个 rank 3,072 env，总计 6,144 env；每轮 24 次策略决策，每次最多推进 5 个控制步；训练 3,000 轮。
- 名义总预算为 **2,211,840,000 个控制步**，与原 teacher 的双卡 15,000 轮预算一致。
- W&B 的默认 `Step` 映射到原 teacher 的等价采样位置；真实 PPO iteration 和实际累计控制步分别记录为 `Train/iteration`、`Train/control_steps`。
- 历史 OFT 单卡短测记录为 **13.75 秒/轮**，相对旧实现 21.31 秒/轮，耗时减少 **35.5%**，吞吐提高 **55%（1.55×）**；本轮真实双卡结果见开头。
- 新预算的纯训练外推约 **11.5 小时/模型**；暂按 **12–16 小时/模型**预留。PI 优化后吞吐尚未验证。

## 预算为什么需要调整

原 teacher 在 50 Hz 下每次决策推进 1 个控制步。延迟 Command actor 在 10 Hz 下决策，冻结 Decoder 继续以 50 Hz 执行，因此每次决策最多推进 5 个控制步。

同样设置每轮 24 次决策时，当前每个 env 每轮的模拟时长是原 teacher 的 5 倍。直接照搬 15,000 轮，会让每个 env 经历五倍的名义控制步预算。

| 项目 | 原 teacher | 对齐后的延迟策略 |
|---|---:|---:|
| GPU/rank 数 | 2 | 2 |
| env/rank | 3,072 | 3,072 |
| 总 env 数 | 6,144 | 6,144 |
| 策略决策频率 | 50 Hz | 10 Hz |
| 决策次数/env/轮 | 24 | 24 |
| 控制步/决策，最多 | 1 | 5 |
| PPO 轮数 | 15,000 | 3,000 |
| 名义控制步/env | 360,000 | 360,000 |
| 名义总控制步 | 2,211,840,000 | 2,211,840,000 |

```text
teacher：6144 × 24 × 1 × 15000 = 2,211,840,000 控制步
当前：   6144 × 24 × 5 ×  3000 = 2,211,840,000 控制步
```

物理配置保持每个控制步 4 个物理子步，physics dt 为 0.005 秒。预算对齐的是控制步采样量；PPO 更新次数减少到原来的 1/5，正则系数调度和 DAgger 更新频率没有在本次修改中重标定。

**当前是名义预算对齐。** Episode 在一个决策周期内提前终止时，剩余控制步不会执行，因此实际累计控制步可能低于上述预算。训练仍按固定 3,000 轮结束，没有追补到严格的实际控制步目标。CLI 后置参数仍可覆盖脚本默认的 env 数、轮数等配置。

## 此前预算改动

| 文件 | 改动 |
|---|---|
| [训练启动脚本](../../scripts/tasks/extreme_parkour_go2/train_small_policy_with_latency.sh) | 使用 `torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2` 和 `--distributed`；默认每 rank 3,072 env、3,000 轮；run name 改为 `scratch_seed1_3k`。复用原 teacher 的系统 NCCL preload，保留已有 Kit 8 线程参数。 |
| [pipeline 配置](../../configs/training/parkour_latency_pipeline.yaml) | 默认 native command checkpoint 从 `model_14999.pt` 改为 `model_2999.pt`。Teacher Decoder 的 checkpoint 仍来自原 teacher。 |
| [训练入口](../../third_party/Isaaclab_Parkour/scripts/rsl_rl/train.py) | 分布式训练时将 latency seed 加上 `1009 × global_rank`，避免两个 rank 使用相同的延迟随机序列。 |
| [PPO runner](../../third_party/Isaaclab_Parkour/scripts/rsl_rl/modules/on_policy_runner_with_extractor.py) | 在 GPU 上累计本轮实际执行控制步，每轮通过 all-reduce 汇总所有 rank；记录全局累计控制步和真实 iteration。Episode 日志使用未折扣奖励及控制步长度；PPO 继续接收原来的折扣 decision reward。 |
| [W&B writer](../../third_party/Isaaclab_Parkour/scripts/rsl_rl/modules/teacher_step_wandb_writer.py) | 将日志 Step 映射到原 teacher 的采样尺度，使新 run 能与旧 teacher run 按默认 Step 横轴叠加。 |
| [使用文档](../../docs/extreme_parkour/cookbook.md) | 更新双卡预算、checkpoint、实际步数和 W&B 横轴说明。 |
| [启动/pipeline 测试](../../tests/scripts/test_parkour_latency_pipeline.py)、[native PPO 测试](../../third_party/Isaaclab_Parkour/parkour_test/test_command_ppo.py) | 在既有测试文件中增加预算、执行步汇总、episode 日志单位和 W&B Step 的语义检查。 |

Latency checkpoint 新增 `total_control_steps`，用于恢复累计采样进度。训练恢复时读取该必需字段；旧 latency checkpoint 没有进行字段迁移。Inference load 不恢复该训练计数。

已读取确认的初始化状态是：**Command actor、critic、estimator 和优化器从零开始，Decoder 加载训练好的 teacher 并冻结。** 该初始化改动由此前主线程完成，本侧线程没有再次实现。

## W&B 如何与原曲线比较

原 teacher 的默认 Step 是零起始 PPO iteration，每次更新采集：

```text
6144 env × 24 控制步 = 147456 控制步
```

新 writer 使用所有 rank 的实际累计控制步：

```text
Step = floor((Train/control_steps - 1) / 147456)
```

完整采样时，新 run 的第 0 轮对应旧 teacher 的 Step 4，第 1 轮对应 Step 9，第 2 轮对应 Step 14。提前终止会使位置按实际执行步数调整。默认 Step 表示 teacher 等价采样位置，真实更新编号保留在 `Train/iteration`。

`Train/mean_reward` 改为 episode 未折扣奖励总和，`Train/mean_episode_length` 改为 episode 控制步数，使这两个指标与原 teacher 使用相同单位。该比较反映同等控制步采样进度，不保证相同收敛行为。

映射存在整数分桶：极小 env 数的 smoke 可能连续多轮落在同一个 Step，W&B 会合并同 Step 的日志行。本次没有修改或回填已有线上 run；运行中的旧进程也不会自动获得新 writer。

## 性能证据：已读取的测量

历史来源目录：[perf_20260914](../../outputs/latency_rl/parkour/perf_20260914/)。以下数字沿用此前记录，本轮未找到该目录中的原始 JSON，不能作为本轮重新验收的证据。

OFT 使用相同延迟 profile、单卡 3,072 env，每轮完整执行 24 次决策 rollout 和 PPO 更新。每组预热 3 轮，测量 3 轮，表内为测量部分的 wall time 中位数。

| 版本 | 秒/轮 | 决策步/s | 相对旧实现耗时减少 |
|---|---:|---:|---:|
| [旧实现](../../outputs/latency_rl/parkour/perf_20260914/oft_baseline.json) | 21.3143 | 3,459 | — |
| [批量动作操作](../../outputs/latency_rl/parkour/perf_20260914/oft_bulk_p4.json) | 16.3083 | 4,521 | 23.5% |
| [批量操作＋Kit 8 线程](../../outputs/latency_rl/parkour/perf_20260914/oft_k8.json) | 15.4038 | 4,786 | 27.7% |
| [当前代码＋Kit 8 线程](../../outputs/latency_rl/parkour/perf_20260914/oft_code_k8.json) | 15.3151 | 4,814 | 28.2% |
| [当前代码＋Kit 8＋NUMA 绑定](../../outputs/latency_rl/parkour/perf_20260914/oft_numa_k8.json) | 14.3665 | 5,132 | 32.6% |
| [当前代码＋Kit 8＋GC 阈值 100000](../../outputs/latency_rl/parkour/perf_20260914/oft_gc_k8.json) | **13.7513** | **5,362** | **35.5%** |

这些是不同组合的结果，不能把各行收益相加。NUMA 与最佳 GC 参数的组合没有在本侧线程核对到测量，NUMA 绑定也没有加入本次启动脚本。13.75 秒是短测结果，不是新双卡配置的稳态实测。

此前已落地、在本侧线程读取确认的性能改动包括：

- [动作调度](../../training/common/action_latency.py)：整批保存 command、整批构建 admission mask 和动作张量，减少逐 env 的 CUDA 小操作。
- [scheduler](../../latency_bench/scheduler/action_queue.py)：训练只保留最近已执行事件，独立保留总计数，减少历史事件驻留。
- [latency sampler](../../latency_bench/latency/samplers.py)：直接返回 `LatencySample`，删除重复字典和 sample 转换。
- [latency wrapper](../../third_party/Isaaclab_Parkour/scripts/rsl_rl/latency_vecenv.py)：复用 observation 等缓冲区、env IDs，向量化奖励和执行步累计。
- [PPO 更新](../../third_party/Isaaclab_Parkour/scripts/rsl_rl/modules/ppo_with_extractor.py)：日志标量在 GPU 累计后统一回传 CPU，减少逐 minibatch readback。
- [训练入口](../../third_party/Isaaclab_Parkour/scripts/rsl_rl/train.py)：提高 GC 阈值，减少对大量长期驻留对象的频繁扫描。

在 `oft_gc_k8.json` 的测量轮中，探针的提交＋取动作计时中位数约 **3.20 秒/轮**，物理模拟调用计时约 **8.54 秒/轮**。这些子阶段计时包含主机调度和调用等待，不等于独立的 GPU kernel 占用时间。

## 原训练耗时与本次 ETA

本侧线程读取了完整 [teacher console log](../../third_party/Isaaclab_Parkour/logs/teacher_zero_delay_fixed_lr_ddp_seed1_15k.console.log)：

- 15,000 条 iteration time 的总和约 **9.1998 小时**。
- 每轮中位数 **2.19 秒**，最后 20 轮中位数 **2.18 秒**。
- 日志时间从 `2026-09-01 16:49:09 UTC` 到 `2026-09-02 02:06:51 UTC`，约 9 小时 18 分钟，包含启动等开销。
- 作者 [官方 README](https://github.com/chengxuxin/extreme-parkour/blob/main/README.md#usage) 给出的基础策略参考是 RTX 3090 上 8–10 小时。当前找到的本机完整 15k 记录支持约九小时，没有找到所提到的一小时完整训练记录。

用现有 OFT 单卡短测外推新预算：

```text
3000 × 13.7513 / 3600 ≈ 11.46 小时/模型
```

双卡同时把总 env 数从 3,072 增加到 6,144；每张卡仍负责 3,072 env，每轮时间不能再简单除以二。考虑尚未实测的双卡通信、CPU 竞争和保存开销，侧线程暂按 **12–16 小时/模型**预留。两种模型各使用双卡依次训练时，粗预算为 **24–32 小时合计**；PI 没有对应优化后实测，因此这不是两个模型的测量结论。

同等每 env 120 个控制步的粗略对照为：原 teacher 五轮约 `5 × 2.19 = 10.95 秒`，当前延迟策略一轮约 13.75 秒，净多约 2.8 秒。该对照跨越双卡正式训练和单卡短测，不能把差额全部归因于延迟调度；本轮以匹配配置的 kernel profile 和双卡计时判断瓶颈及收益。

预算从 15,000 轮改为 3,000 轮带来的五倍时间缩短属于采样预算纠正；代码性能优化本身的历史短测吞吐收益为 1.55×，原始短测 JSON 本轮未找到。

## 此前预算改动的验证与边界

运行检查：

```bash
PYTHONPATH=/pfs/users/lxy/latency-sensitive-bench:/pfs/users/lxy/latency-sensitive-bench/third_party/Isaaclab_Parkour/scripts/rsl_rl \
OMP_NUM_THREADS=1 \
/pfs/users/lxy/conda/envs/extreme-parkour/bin/python -B -m pytest -p no:cacheprovider \
  third_party/Isaaclab_Parkour/parkour_test/test_command_ppo.py \
  tests/scripts/test_parkour_latency_pipeline.py -q
```

结果：**10 passed in 3.26s**。检查覆盖启动默认参数与名义总预算、不同 rank 执行步汇总、episode 奖励/长度单位、PPO 收到的折扣奖励保持不变，以及 W&B Step 映射。

分布式汇总测试在 CPU 上模拟不同 rank 的步数；启动脚本测试捕获实际生成的命令参数，没有启动 Isaac Sim 或真实 NCCL 双卡训练。Shell 语法、Python AST 和相关文件的 `git diff --check` 通过。

以上 CPU 测试本身不验证双卡速度；本轮真实双卡重放、profile 和计时证据见本文开头。PI 最终吞吐、严格实际控制步预算停止条件和 3,000 轮训练收敛尚未验证。此前预算配置在后续按新脚本启动时生效。
