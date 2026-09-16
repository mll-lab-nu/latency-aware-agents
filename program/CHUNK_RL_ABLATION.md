# H8 FastTD3 第一版消融结果与 D/E 基线

第一版消融已定稿：五个配置、四个 training seeds、每 seed 10M raw frames，共 20 个 run，训练和最终评估全部完成。

2026-09-15 复核发现，这 20 个 run 的 learner 以 raw frames 累积 optimizer update credit，
但 chunk replay 实际只加入完成的 execution segment；因此 horizon/执行长度会改变每条 replay
transition 的重复更新次数。以下结果保留为修复前历史证据，不再作为等训练预算比较。新实验以
500,000 critic optimizer steps 为统一预算，credit 由新完成的 replay transitions 累积，并单独记录 raw frames。

**生产第一版的额外模块只保留 D（critic planned counts）和 E（planned head mask）。** 本次 Balance 的 actor 使用 198D 物理输入，输出始终为 8×43=344D；critic 使用首次执行时的 198D 物理状态和 8D planned counts。探索和 target smoothing 使用独立高斯噪声，网络按实际宽度普通初始化。

这是用户确定的 Occam 简化选择。**本轮没有直接训练“只保留 D/E”的组合，不能把它写成已验证的性能结论。** 后续将分别单独消融 D、E，具体实验协议另行确定。

## 第一版固定协议

- 固定既有 QwenOFT recorded temporal profile、capacity=1、128 env，不重新测延迟，不调超参数，不重跑 H1。
- 每个配置使用相同的 training seed1/2/3/4，各训练约 10M raw frames。Seed1/2 在本地，seed3 在 parkour，seed4 在 HAIC；机器与 seed 绑定，跨机器差异不能只归因于随机 seed。
- 每个最终 checkpoint 评估相同的 25 env reset seeds × 4 latency seeds，共 100 个场景。Actor 直接输出均值动作；保留环境 reset 扰动和按 seed 采样的 profile 延迟。
- 以四 seed 最终平均存活 / 同四 seed full ≥0.95 作均值筛选，同时报告各 seed、样本 SD、return 和存活 ≥100 的比例。“通过”仅表示达到筛选门槛，不证明等价或每项模块分别无效。
- 保持 lr=3e-4、tau=0.1、target noise=0.001、clip=0.5、AdamW weight_decay=0.1、policy_delay=2、batch=32768、replay=1M，更新预算按 raw frames 计算。

本地各 run 最终为 10,000,512 raw frames / 156,206 critic updates，远端各 run 为 10,000,256 / 156,202；同一 training seed 内各候选预算相同。

## 四 seed 最终结果

下面的字母表示**删除的模块**；± 为四个 training seeds 之间的样本标准差。

| 删除模块 | Seed1 | Seed2 | Seed3（parkour） | Seed4（HAIC） | 均值 ± SD | 平均 return | ≥100 | / full | 筛选 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full | 71.90 | 554.35 | 145.90 | 446.76 | 304.73 ± 232.32 | 269.956 | 0.742 | 1.0000 | 基线 |
| ABCDEHJ | 152.93 | 92.33 | 95.07 | 68.00 | 102.08 ± 36.02 | 85.134 | 0.410 | 0.3350 | 失败 |
| ABCHJ | 334.43 | 369.02 | 430.34 | 535.15 | 417.24 ± 88.05 | 378.106 | 1.000 | 1.3692 | 通过 |
| ABCDHJ | 288.89 | 347.85 | 223.13 | 488.07 | 336.99 ± 112.87 | 300.880 | 0.992 | 1.1059 | 通过 |
| ABCDFGHIJ | 466.70 | 152.54 | 502.95 | 181.21 | 325.85 ± 184.54 | 293.109 | 0.955 | 1.0693 | 通过 |

四 seed 汇总：[means](../outputs/tasks/humanoidbench_balance_simple/chunk_rl_ablation_20260913/two_seed_seeded_20260913T1025Z/additional_seeds_20260913T1754Z/means/)。

### 相邻配置的条件效应

| 比较 | Seed1/2/3/4 存活变化（帧） | 均值变化 |
|---|---|---:|
| ABCHJ → ABCDHJ：再删 D | −45.54 / −21.17 / −207.21 / −47.08 | −19.23% |
| ABCDHJ → ABCDEHJ：再删 E | −135.96 / −255.52 / −128.06 / −420.07 | −69.71% |
| ABCDHJ → ABCDFGHIJ：再删 F/G/I | +177.81 / −195.31 / +279.82 / −306.86 | −3.30% |

1. **D/E 有条件收益，E 的证据最强。** 两次删除都是四个 seed 全下降。这些效应对应表中的删除背景；尚未测 full 背景下单删 D 或 E。ABCDHJ 虽通过相对 full 的筛选，仍低于保留 D 的 ABCHJ。
2. **ABCHJ 是已测组合中均值最高的简化候选。** Actor 输入 550→198，未观察到平均性能损失，终点跨 seed 差异更小；不能据此认定其中每项分别无效，或训练过程已经稳定。
3. **F/G/I 整组没有显示一致收益。** 再删它们有两个 seed 升、两个降，终点差异增大。最后一组 ABCDFGHIJ 只保留 E，已经删除 D；其结果不能直接推广到只保留 D/E。

二分用于寻找简化组合、识别相邻背景下的条件效应，不穷尽交互。生产第一版选择删除 ABCFGHIJ、保留 D/E，但该完整组合没有本轮终点评估。

## 模块定义与生产第一版

下表保留历史 full 的因子定义，便于解释上述结果。

| ID | 历史模块 | 第一版行为 |
|---|---|---|
| A | actor 344D scheduled chunk | 删除旧 chunk 动作输入 |
| B | actor 8D scheduled index | 删除旧 chunk one-hot 输入 |
| C | actor certain-prefix 裁剪 | 随旧 chunk 输入删除其生产和传递链 |
| **D** | **critic 8D planned counts** | **保留计划窗口内各 head 的执行次数，critic state 为 206D** |
| **E** | **planned head mask** | **保留 counts>0 的 mask，统一作用于 current/target/actor Q 和动作 L2** |
| F | chunk 内 AR(1) exploration，ρ=0.9 | 同尺度独立高斯探索噪声 |
| G | chunk 内 AR(1) target smoothing | 同尺度、同 clipping 的独立高斯噪声 |
| H | actor 零列扩展与 RNG 保持 | 按实际 198D 输入普通初始化 |
| I | critic 零列扩展与 RNG 保持 | 按实际 206D 状态输入普通初始化 |
| J | actor 仅物理量归一化 | 对全部保留的 198D actor 输入做经验归一化 |

A/B 删除后 C/H/J 失去作用对象，历史实验将它们合并为 ABCHJ。I 的 RNG 保持仍是独立因素。

固定主干继续保留：按真实 source 串接 issue/execution，聚合实际 reward 和 γ^duration，正确处理 termination/timeout，保留经验方差修复、隐藏层、C51、优化器及 raw-frame 更新预算。

D/E 的计划窗口在首次执行时，用复制的 sampler、admission、result timeline 和 scheduler 状态推演下一次交接；不推进物理环境或真实 sampler RNG。计划假设 episode 继续，提前终止不截掉 actor 的计划后缀；hold-tail 将超出 H 的次数累加到末 head。计划时钟仅供 replay 使用，不进入 actor。每个原始帧的 replay clock 保留 7 个字段；本次 Balance 的 actor 和评估均真实使用 198D 输入。

## 通用 D/E 算法与时钟语义

算法入口仍为 `FAST_TD3`：`fasttd3_action_chunk_horizon=1` 沿用普通 FastTD3，`H>1` 使用 D/E。向量状态与有限边界连续 Box 动作的宽度由实际空间决定：actor 输入 `n_obs`、输出 `H*n_act`，critic 状态为 `n_obs+H`。198/344/206 是上述 Balance 实验的具体维度。

共享配置位于 [fasttd3_chunk.yaml](../configs/training/algorithms/fasttd3_chunk.yaml)，默认 H=8、capacity=1、latest_ready、hold-tail，并提供 FastTD3 所需的同步、非 batched sampling 布局和既有算法参数。组合配置按 base 列表从左到右合并，本地字段最后覆盖。例如在 `configs/training/` 下组合既有 Walker2d 任务与算法：

```yaml
__base__:
- ../examples/gymnasium/walker2d/small_model_train.yaml
- algorithms/fasttd3_chunk.yaml
```

换任务时替换第一个 base；预算、profile、设备和日志路径属于任务配置。现有 G1 H8 配方也组合该共享配置，保留 6M raw-frame 预算。原生环境示例：[Pendulum-v1](../configs/examples/gymnasium/pendulum/fasttd3_chunk_train.yaml)、[Walker2d-v4](../configs/examples/gymnasium/walker2d/fasttd3_chunk_train.yaml)。启动命令：

```bash
.venv/bin/python -m training.train --config configs/examples/gymnasium/pendulum/fasttd3_chunk_train.yaml
```

动作在训练与评估中统一从 [-1, 1] 映射到原生 Box 边界，一维动作仍是长度为 1 的向量。G1 的关节编码只负责环境动作转换；chunk 与 replay 不依赖 task name 或 G1 状态宽度。

对 `H>1`，令 `M=env_fps/obs_fps`，沿用现有 `EnvClock` 的整数 stride 契约。策略只在观测边界发请求；每个 head 对应一个原始帧，H 与 M 独立。环境每步返回物理边界状态 `obs`，以及以下训练记录，actor 只读取 `obs`：

| 字段 | 形状 | 语义 |
|---|---|---|
| `replay_obs` | `[M+1, n_obs]` | 每个原始转移前后的物理状态快照 |
| `replay_rewards` | `[M]` | 每个原始环境转移的 reward |
| `replay_clock` | `[M, 7]` | episode、转移前原始帧、来源请求帧、head、admitted、计划窗口起止偏移 |
| `replay_length` | 标量 | 本步实际有效原始帧数；reset 为 0 |

Replay 消费 SF 的 `next_obs` 记录，按实际来源串接 segment，允许同一步内来源交接及跨步延续。Actor 状态取请求时观测，critic 状态取首次执行时物理状态与计划 counts。回报为 `Σγʲrⱼ`，bootstrap 折扣为 `γᴸ`；reward scale/clip 对每个原始帧生效一次，更新预算与 chunk episode 的 raw_frames 累计实际有效帧数。

提前终止时只读取有效记录；真正终止不 bootstrap，timeout 丢弃最后未完整结束的 segment，两者同时出现时真正终止优先。计划 counts 保持假设 episode 继续的窗口，hold-tail 计入末 head；autoreset 保留终止 trace。改变观测频率也会改变请求时刻，因此轨迹和性能可能变化；逐帧 credit 的含义保持一致。

现有同形状 D/E checkpoint 可恢复模型和 optimizer，replay 冷启动。旧 full checkpoint 继续使用对应冻结源码复现。通用化验证仅证明数据流与恢复可运行，不替代 D/E 的训练和性能实验。

本次通用化通过 257 项相关检查（含 CUDA compile）。原生 Pendulum 训练 3,104 raw frames／8 次更新、Walker2d 训练 1,552／4 次更新，均完成保存、加载和单次／batch 评估。G1 的既有 QwenOFT temporal profile 在 stride=1/2/3、两个 latency seeds 下共检查 6,147 raw frames，计划交接与实际交接一致，预测未推进真实 RNG 或物理环境。三个既有 G1 H8 配方的配置值保持一致。

## 尚未解决的训练不稳定

周期评估每次使用相同的 8 env × 4 latency 场景、均值动作，不加探索噪声。历史 checkpoint 的同场景重测确认了真实退步：

| Run | 同场景平均存活 |
|---|---|
| 旧 full seed1，探索 seed 修复前 | 5M 122.12 → 6M 74.47；最终 checkpoint 重测 80.12 |
| 历史成功 H8 repair seed1 | 5,748,608 时 405.06 → 5,908,608 时 336.62 |
| 本轮 full seed1，探索 seed 已修复 | 8M 265.06 → 9M 219.31 → 10M 81.69 |
| 本轮 full seed2 | 8M 347.50 → 9M 319.66 → 10M 676.62 |

旧 full 5M→最终的配对平均差为 −42.00，env seed 聚类 bootstrap 95% 区间 [−61.91, −20.16]；历史 repair 的相邻 checkpoint 差为 −68.44，区间 [−108.84, −19.19]。这些区间衡量评估场景差异，不能替代多个 training seeds。

Inference 子进程原先没有设 torch seed，已按 seed、policy 和 worker 编号修复。同权重、同 observation、同请求顺序的新进程动作及 noise scales 修复后 exact equal。该缺口修复仍保留，但没有消除本轮 full seed1 的后期断崖，也不保证并行请求调度下全训练逐位复现。

根因及全部 run 的下跌发生率仍未确定。已有 Q、critic loss 和动作幅度变化只是相关证据；执行 head 分布及延迟窗口没有突变，尚不能归因于某个具体更新机制。D/E 第一版的代码整理不构成稳定性修复。

证据：[checkpoint 配对差](../outputs/tasks/humanoidbench_balance_simple/chunk_rl_ablation_20260913/stability/checkpoint_differences.json)、[相邻更新](../outputs/tasks/humanoidbench_balance_simple/chunk_rl_ablation_20260913/stability/adjacent_checkpoints.json)、[seed 复现](../outputs/tasks/humanoidbench_balance_simple/chunk_rl_ablation_20260913/stability/seed_reproduction.json)、[完整训练曲线](../outputs/tasks/humanoidbench_balance_simple/chunk_rl_ablation_20260913/stability/training_stability.png)。

当前 D/E 实现的隔离候选通过 161 项相关测试（含 CUDA compile、更新预算和 checkpoint 恢复），现有工作区通过 134 项相关测试。真实 QwenOFT temporal profile 的两 latency seeds、共 1,024 raw frames 调度检查确认计划交接与实际交接一致，预测不推进真实 sampler RNG 或物理环境；三个 H8 配置的解析检查通过。这些代码验证不替代训练和性能评估。[检查记录](../outputs/tasks/humanoidbench_balance_simple/chunk_rl_ablation_20260913/de_v1_review_20260914T123952Z/review_notes.md)。

## 评估与归档入口

最终场景：env 1000100–1000124 × latency 1000100–1000103；周期场景：env 1000000–1000007 × latency 1000000–1000003。两种评估均关闭探索噪声，但场景和 checkpoint 不完全相同，不能把排名差异归因于噪声。

W&B 的 `episode/length` 混有单 episode 长度和 SF 窗口均值；训练窗口应看 `episode_train/rolling_mean_length`。本轮 final.json 在训练 W&B 结束后生成，最终结果尚未上传 W&B，本文使用固定最终评估文件。原始 W&B 审计文件保留在实验目录。

- [本轮实验目录](../outputs/tasks/humanoidbench_balance_simple/chunk_rl_ablation_20260913/two_seed_seeded_20260913T1025Z/)：冻结的 sources/base、各变体源码、配置、checkpoint、final.json 和日志。旧 full 权重继续使用对应冻结源码复现。
- [新增 seed 目录](../outputs/tasks/humanoidbench_balance_simple/chunk_rl_ablation_20260913/two_seed_seeded_20260913T1025Z/additional_seeds_20260913T1754Z/)：四 seed means 与完成状态；两台远端队列和本地采集均已结束。
- [第一版整理前全文](../outputs/tasks/humanoidbench_balance_simple/chunk_rl_ablation_20260913/de_v1_review_20260914T123952Z/program/CHUNK_RL_ABLATION.md)：两 seed 阶段、清理验证、历史诊断和运行过程的完整记录。
- [更早全文归档](../outputs/tasks/humanoidbench_balance_simple/chunk_rl_ablation_20260913/two_seed_seeded_20260913T1025Z/document_archive_20260913T1650Z/CHUNK_RL_ABLATION.md)、[原始差异归档](../outputs/tasks/humanoidbench_balance_simple/chunk_rl_ablation_20260913/archive/)保留原始证据。

后续分别单独消融 D、E；只保留 D/E 的性能和 zero/fixed40 对照尚待验证，本次不启动新训练。不得用最好 seed、最好 checkpoint 或多训到回升点替换本轮固定终点结果。
