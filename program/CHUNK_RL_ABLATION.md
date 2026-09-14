# H8 FastTD3 第一版消融结果与 D/E 基线

第一版消融已定稿：五个配置、四个 training seeds、每 seed 10M raw frames，共 20 个 run，训练和最终评估全部完成。

**生产第一版的额外模块只保留 D（critic planned counts）和 E（planned head mask）。** Actor 使用 198D 物理输入，输出始终为 8×43=344D；critic 使用首次执行时的 198D 物理状态和 8D planned counts。探索和 target smoothing 使用独立高斯噪声，网络按实际宽度普通初始化。

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

D/E 的计划窗口在首次执行时，用复制的 sampler、admission、result timeline 和 scheduler 状态推演下一次交接；不推进物理环境或真实 sampler RNG。计划假设 episode 继续，提前终止不截掉 actor 的计划后缀；hold-tail 将超出 H 的次数累加到末 head。计划时钟仅供 replay 使用，不进入 actor。训练观测的 replay clock 保留 7 个字段，actor 和评估均真实使用 198D 输入。

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
