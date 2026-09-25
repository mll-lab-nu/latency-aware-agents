# Benchmark 核心进展

最后核查：**2026-09-20**。本文只记录四个主任务的已核验结果、关键决策、当前缺口和下一步；清理前的完整时间线见
[历史快照](logs/PROGRESS-pre-cleanup-2026-09-17.md)，运行手册和专题记录见 [program 索引](README.md)。

固定公共版本：

- zero-latency 模型：[benchmark-models@e05ca26](https://huggingface.co/placeholder/benchmark-models/tree/e05ca26afde4f509b0f838742c522f3a068596a8)
- RTX 3090 profiles：[profiles@e8be1e5](https://huggingface.co/datasets/placeholder/profiles/tree/e8be1e5dfd309f85ab7ff704dc02bd4898b7ceb4/profiles)
- Parkour latency small：[benchmark-models@6348f44](https://huggingface.co/placeholder/benchmark-models/tree/6348f44ce65c1414dbc4f2ec46e0d5eb525ab770/latency-aware/extreme-parkour-go2/small-policy)
- MIKASA WanOFT latency-aware VLA：[benchmark-models@ad40d57](https://huggingface.co/placeholder/benchmark-models/tree/ad40d57c5747761705770aa557db0f287888f727/latency-aware/mikasa-intercept-grab-fast/vla/placeholder-wanoft-h1/mikasa_wanoft_latency_3090_20260916T094307Z)

## 阅读口径

- “训练完成”“评测完成”“达标”和“已发布”是四个独立状态。
- VLA return 使用 `episode_return_env`，长度使用 `survival_steps`；原生 PPO 保留源指标名。
- `±` 是 episode return 的总体标准差；跨 training seeds 的标准差会单独注明。
- 不同频率、latency profile、seed 集和 episode 数的结果不可直接混用。
- smoke 只证明流水线可运行，不作为策略质量结果；旧 running 消息不代表当前进程状态。
- 缺失指标写“未记录”，不从其他指标或模型推断。

## 四任务总览

| 任务 | Zero VLA | RTX 3090 profile | Latency small | Latency VLA | 当前结论 |
|---|---|---|---|---|---|
| Balance Simple | 4/4 完成 | 4/4 完成 | H8/H16 专题实验完成多组，尚无统一正式产物 | 正式四模型终点未完成 | 延迟控制可由 oracle/BC 部分恢复，但现有 RL/PPO 未超过 BC；新 DAgger 矩阵待终点结果 |
| MIKASA InterceptGrabFast | 4/4 完成 | 4/4 完成 | 4/4 达标并发布 | 4/4 SFT 5k；四组矩阵完成，仅 WanOFT 已发布 | latency-aware VLA 能恢复部分 profile 性能，模型间恢复幅度差异明显 |
| Extreme Parkour Go2 | 3/4 完成 | 3/4 完成 | 3/4 训练、评测、发布完成 | 未找到正式产物 | 三个 small policy 在 profile 下均完成正式评测；未预设 acceptance threshold |
| HAIC PullCart | 1/4 完成 | 1/4 完成 | 正式终点未核验 | 未找到正式产物 | 当前只有 QwenOFT 正式 VLA 结果，任务覆盖仍不完整 |

## 公共 zero-latency small policy

| 任务 | checkpoint / 训练进度 | 已核验结果 | 状态与证据 |
|---|---|---|---|
| Balance | `checkpoint.pth`；25M 预算，选中 20,864,000 步 | 独立 L0 return/SR/n 未记录 | 已发布；checkpoint 名中的 702.976 不是独立 eval；[BALANCE](BALANCE.md) |
| MIKASA | `checkpoint.pt`；12,288,000 步 | strict success 100%；canonical 100/100 | 已发布；[MIKASA](MIKASA.md) |
| Parkour | `model_14999.pt`；15k iterations | reward 23.14±6.02；length 907.95±296.15；progress 1.00±0.01 | n/seed 未记录；[EXTREME_PARKOUR](EXTREME_PARKOUR.md) |
| HAIC | `checkpoint_final.pt`；499,908,608 frames | native success 93.8965%；源 `episode_cnt=2308` | 未达到 native 100% gate；[HAIC](HAIC.md) |

## 正式 zero-latency VLA 与 profile

### Balance Simple

50/50Hz；L0 各 100 episodes。profile 为 1×RTX 3090 temporal observation-to-action latency。

| 模型 | 训练 | L0 return ± SD | mean length | profile mean |
|---|---:|---:|---:|---:|
| QwenOFT | SFT 15k | 98.4357±15.4165 | 111.10 | 37.881ms |
| QwenGR00T | SFT 15k | 27.4931±2.4959 | 39.12 | 60.013ms |
| QwenPI v3 | SFT 15k | 41.1143±7.7280 | 54.44 | 99.176ms |
| WanOFT | SFT 15k | 90.6691±13.8048 | 100.97 | 300.024ms |

模型、评测和 profile 均在上方固定公共版本中；这些 L0 文件没有 task success 指标。

### MIKASA InterceptGrabFast

QwenOFT/QwenGR00T/QwenPI v3 为 20/20Hz，WanOFT zero VLA 主表也按 20/20Hz。
L0 使用 50 个共同 seeds `4242424242–4242424291`。

| 模型 | 训练 | L0 return ± SD | task success | profile mean |
|---|---:|---:|---:|---:|
| QwenOFT | SFT 5k | 21.7417±16.4361 | 56% | 99.050ms |
| QwenGR00T | SFT 5k | 21.2928±16.6083 | 56% | 99.352ms |
| QwenPI v3 | SFT 5k | 21.9671±17.2095 | 54% | 148.689ms |
| WanOFT | SFT 5k | 18.9461±16.4028 | 42% | 300.539ms |

本表是共同 50-seed 补评测；中央模型中的其他 100-episode 文件使用不同协议，不与本表合并。

### Extreme Parkour Go2

50Hz control、10Hz decision；L0 各 100 episodes。

| 模型 | 训练 | L0 return ± SD | waypoint progress | profile mean |
|---|---:|---:|---:|---:|
| QwenOFT | SFT 5k + DAgger 5k | 23.3283±7.7106 | 0.9075 | 103.426ms |
| QwenGR00T | SFT 5k + DAgger 5k | 22.3789±6.7888 | 0.9163 | 157.357ms |
| QwenPI v3 | SFT 5k + DAgger 5k | 21.8744±7.9515 | 0.8500 | 238.729ms |
| WanOFT | 未完成 | — | — | — |

### HAIC PullCart

50/50Hz。QwenOFT 完成 SFT 5k + DAgger 5k；100-episode L0 return
13.3497±8.2155、mean length 187.09、task success 56%，RTX 3090 profile mean 126.288ms。
其余三模型未找到正式 zero VLA、profile 或 latency-aware 产物。

## Latency-aware small policy

### MIKASA

单 worker、temporal profile、deterministic mean action；环境 seed `20260910`、latency seed `271828`，各 256 episodes。
既有 gate 为 `success_at_end >= 90%`。WanOFT 为 10Hz，其余为 20Hz，不能把四行当作同频率公平比较。

| Profile | checkpoint step | success_at_end | success_once | mean return | 状态 |
|---|---:|---:|---:|---:|---|
| QwenOFT 20Hz | 13,824,000 | 98.83% | 99.61% | 33.6446 | 达标、已发布 |
| QwenGR00T 20Hz | 20,951,040 | 98.05% | 98.05% | 31.7504 | 达标、已发布 |
| QwenPI v3 20Hz | 15,360,000 | 96.88% | 98.83% | 28.6463 | 达标、已发布 |
| WanOFT 10Hz | 23,040,000 | 93.75% | 98.44% | 30.8664 | 达标、已发布 |

完整训练/eval 配置和 profile 固定版本见各发布 bundle；旧 WanOFT 20Hz 结果不替代当前 10Hz 版本。

### Parkour

三个模型均训练 3,000 PPO iterations，并在各自 RTX 3090 profile 与 fixed 0ms 下各评测 100 episodes；
协议为 50Hz control、10Hz decision、seed 1、1500 raw-frame cap。

| 模型 | Profile return ± SD / progress | Zero return ± SD / progress | 状态 |
|---|---|---|---|
| QwenOFT | 16.2985±5.8936 / 0.8314 | 16.7771±6.1831 / 0.8329 | 已发布 |
| QwenPI v3 | 17.1554±4.3398 / 0.9214 | 11.8676±4.5410 / 0.6757 | 已发布 |
| QwenGR00T | 18.6098±3.5573 / 0.9814 | 12.7761±4.3001 / 0.7900 | 已发布 |

原始结果与 checkpoint 审计见 [completion audit](../outputs/latency_rl/parkour/h200_20260916_completion_audit.md)。
该项目没有预设 acceptance threshold，因此这里只声明训练、评测和发布完成。

## MIKASA latency-aware VLA

所有矩阵均使用 100 episodes、reset seeds `4242424242–4242424341`、deterministic inference；
temporal 项使用 latency seed `271828`。WanOFT 为 10Hz，其余为 20Hz。

| 模型 | Baseline zero | Baseline profile | Latency-aware profile | 训练 / 发布 |
|---|---|---|---|---|
| QwenOFT | success 57%；return 22.0986 | 0%；0.4274 | 57%；19.4161 | SFT 5k 完成；未发布 |
| QwenPI v3 | 60%；23.0983 | 0%；0.2862 | 48%；14.9605 | SFT 5k 完成；未发布 |
| QwenGR00T | 63%；24.1050 | 0%；0.4136 | 35%；11.8917 | SFT 5k 完成；未发布 |
| WanOFT | 41%；16.8500 | 0%；0.3368 | 21%；6.8902 | SFT 5k 完成；已发布 |

证据：[QwenOFT](../outputs/latency_rl/mikasa/qwenoft_rtx3090_h1_v1/direct_eval_revalidation_20260911/comparison.json)、
[QwenPI v3](../outputs/tasks/mikasa_intercept_grab_fast/qwenpi_latency_20260914T122034Z/eval_matrix/results.json)、
[QwenGR00T](../outputs/tasks/mikasa_intercept_grab_fast/qwengr00t_latency_20260915T035000Z/eval_matrix/results.json)、
[WanOFT](../outputs/tasks/mikasa_intercept_grab_fast/wanoft_latency_20260916T094307Z/eval_matrix/results.json)。

结论：四个 zero-latency VLA 在对应 profile 下均降到 0% success；latency-aware 训练恢复到 21%–57%，
但除 QwenOFT 外仍低于各自 zero/zero 成功率。WanOFT 的 21% 也低于其 41% zero/zero ceiling，尚未作 acceptance 决定。

## Balance 关键实验与决策

### Action chunk 与监督合同

- H8 profile 合同修复后，6M-step 配对结果从 mean length 72.88 提升到 172.14，`>=100` 从 3% 提升到 93%；旧结果不能代表修复后实现。[CHUNK](CHUNK.md)
- 第一版 4-seed 消融中，full mean length 304.73±232.32，删除 `ABCHJ` 后为 417.24±88.05 且 `>=100` 为 100%。这支持删除对应机制，但 D/E 组合与单独消融仍未完成。[CHUNK_RL_ABLATION](CHUNK_RL_ABLATION.md)
- 延迟标签必须对应实际 application state 和同一 request 真正执行的 heads；“整段 teacher chunk”与“executed heads”是需要正式对照的两个不同合同。

### Fixed100 诊断

- startup no-op 曾把绝对 SONIC 默认姿态当作 normalized MuJoCo action；修复前依赖该启动行为的 delayed 结果不再作为当前证据。
- 修复后 H16 QwenPI 50k optimizer-step 对照仍失败：temporal profile mean length 61.75，fixed100 mean length 69.625，各 n=8，均低于 100 rejection gate。[pair record](../outputs/tasks/humanoidbench_balance_simple/h16_profile_fixed100_corrected_noop_50k/20260915T174801Z/launch.json)
- native fixed100 oracle 在 32 seeds 上达到 mean length 673.1875、`>=100` 32/32、`>=500` 23/32，说明该延迟合同本身可解；问题在学习和状态覆盖，而不是单纯“延迟必然不可控”。[oracle](../outputs/tasks/humanoidbench_balance_simple/h16_fixed100_delayed_imitation/20260916T061955Z/oracle.json)

### BC、RL 与 PPO

所有下表 fixed100 结果均使用 H16、50/50Hz、capacity 1；不同表行的 seed 集按证据文件保留。

| 路线 | 正式结果 | 结论 |
|---|---|---|
| Execution-aligned BC | mean length 387.906，`>=100` 31/32，`>=500` 8/32 | 明显优于 mapped-teacher baseline 71.0，但只有 oracle 的 57.6% |
| Student-state correction | mean length 464.188，`>=100` 30/32，`>=500` 13/32 | 比等预算旧数据继续训练好；仍只有一个 training seed |
| Joint TD3 updates | 464.188 → 24.313，32/32 paired seeds 退化 | 现有在线 actor update 会摧毁监督策略 |
| Lower LR + frozen-BC anchor | 四个 endpoint mean length 29.69–65.47 | 只能减缓漂移，不能修复 |
| Return-guided bounded regression | 463.156 vs BC 464.188，95% interval 跨 0 | 保留点性能但没有证明增益 |
| SF MC PPO | candidate latest 514.563 vs BC 527.406，n=64 | 训练/评测完成；stability gate 与 policy acceptance 均失败 |

主要证据：[BC/监督修正](../outputs/tasks/humanoidbench_balance_simple/h16_fixed100_delayed_imitation/20260916T061955Z/followup_20260916T070429Z/REPORT.md)、
[anchored RL](../outputs/tasks/humanoidbench_balance_simple/h16_fixed100_delayed_imitation/20260916T061955Z/anchored_rl_20260916T074351Z/REPORT.md)、
[return-guided](../outputs/tasks/humanoidbench_balance_simple/h16_fixed100_delayed_imitation/20260916T061955Z/return_guided_rl_20260916T085703Z/REPORT.md)、
[MC PPO](BALANCE_MC_PPO.md)。

### Profile BC/DAgger 与当前状态

- Profile BC 在 64 fresh paired streams 上将 mean length 从 69.125 提升到 111.156，paired gain +42.031、95% interval `[32.296, 52.391]`；但仍有 24/64 在 100 frames 前结束，`>=500` 为 0。[report](../outputs/tasks/humanoidbench_balance_simple/profile_bc/formal_s1_20260916T1330Z/REPORT.md)
- 旧 QwenOFT profile DAgger 每轮错误地从长期不变的 `best` model 重启，不能作为连续 5k DAgger 结果；修正版要求 `latest` 驱动采集并连续恢复 optimizer/scheduler/RNG。
- **QwenOFT H8 训练与发布完成**：`zero_full` 与 `profile_conditioned` 均完成 5k bootstrap + 5k DAgger optimizer updates，best/latest 均为各自 `bundles/dagger_10_5000`。`zero_full` 在 zero-latency、共同 held-out seeds 820001--820064 下为 mean length `297.8125`、mean return `275.56409`、`>=100` `64/64`、`>=500` `4/64`。发布 revision 分别为 `24cc874d59bec339bcf40046764e52fa26334dc8` 和 `04bf16cfef9a469354256ed179db4fe548be03d3`。
- **QwenPI H16 训练与发布完成**：`zero_full` 与 `profile_conditioned` 均完成 5k bootstrap + 5k DAgger optimizer updates，best/latest 都是各自 `bundles/dagger_10_5000`。`zero_full` 在 zero-latency、共同 held-out seeds 820001--820064 下为 mean length `351.59375`、mean return `328.60200`、`>=100` `64/64`、`>=500` `7/64`。发布 revision 分别为 `72747b40b4289766a624b7cf004ab93cf48e5668` 和 `01bf53c4d69a845ef76bcd595b606fc66a4c6398`。
- **QwenGR00T H16 训练与发布完成**：`zero_full` 与 `profile_conditioned` 均完成 5k bootstrap + 5k DAgger optimizer updates，best/latest 均为各自 `bundles/dagger_10_5000`。`zero_full` 在 zero-latency、共同 held-out seeds 820001--820064 下为 mean length `287.609375`、mean return `267.52374`、`>=100` `64/64`、`>=500` `3/64`。发布 revision 分别为 `19f71fe13865a7d8a8bae95e3d4939822d6ad1c0` 和 `891220398157a663aaf3233e42145c077a327acc`。

### 固定均值 prefix 的因果评测（2026-09-20）

三模型两臂均复用上面的 `bundles/dagger_10_5000` 与既有发布 revision，未重新训练。
**六组正式评测及协议核验完成**，各 64 episodes；50/50Hz、capacity 1、1000-frame cap，
各自 RTX 3090 temporal profile（instance `3a0d42681a03715c`）、profile seed 271828、
inference seed 941731、共同 held-out seeds 820001–820064。Baseline 不传显式 prefix；
conditioned 使用 `K=ceil(profile mean_ms/20)`，OFT/PI/GR00T 为 2/5/4，
不读取本次未来延迟。QwenOFT 不消费显式 prefix，其原有 action-context 输入保持不变。

| 模型 | Baseline length ± SD | Conditioned length ± SD | Baseline / conditioned return | >=100（baseline / conditioned） | Paired length gain [95% CI] |
|---|---:|---:|---:|---:|---:|
| QwenOFT H8 | 63.29688 ± 12.23555 | 133.35938 ± 34.77039 | 50.74563 / 114.62513 | 1/64 / 51/64 | +70.06250 [61.20312, 79.01562] |
| QwenPI H16 | 57.03125 ± 7.63006 | 86.21875 ± 30.93515 | 46.84542 / 72.31230 | 0/64 / 24/64 | +29.18750 [22.03125, 36.54727] |
| QwenGR00T H16 | 56.81250 ± 13.75667 | 104.03125 ± 31.15935 | 45.34041 / 90.42080 | 0/64 / 37/64 | +47.21875 [39.32812, 54.98438] |

六组 `>=500` 均为 0/64，chunk underflow 均为 0。10,000 次 seed-paired bootstrap
（seed 42）下三组 length/return gain 区间均为正；这仅是这些 checkpoint 的相对优势，
不代表跨 training seeds 结论或新增性能达标判定。本轮不重新发布模型。
旧 oracle-delay / 强制 prefix baseline 比较已被本协议替代，原始结果保留用于审计。
GR00T 取消 baseline 强制 prefix 后 mean length 提升 12.21875，95% CI
[8.546875, 16.078125]；两个 flow conditioned 改用固定 K 后的 length 变化区间均跨 0。
QwenOFT 两臂的逐 seed return/length 与旧结果完全一致。
[主证据与完整指标](../outputs/tasks/humanoidbench_balance_simple/causal_profile_eval_20260920/REPORT.md)；
[评测合同与复现入口](BALANCE_CAUSAL_EVAL.md)。

## 下一决策

- **Balance**：固定均值 prefix / no-prefix 六组因果评测完成，以新协议比较为准；已决定新训练数据与所有学生推理统一使用 profile 均值固定 K；公共合同迁移及相关测试已验证（[证据](../outputs/tasks/humanoidbench_balance_simple/fixed_k_unification_20260920/README.md)），尚未重训；下一步决定是否按新标签合同重训。D/E 消融仍是独立待办。
- **MIKASA**：评审 WanOFT 21% profile recovery 是否可接受；为 QwenOFT/QwenPI/QwenGR00T 补齐 canonical bundle/发布状态，保持 10Hz 与 20Hz 分开报告。
- **Parkour**：补 WanOFT H1/profile/latency-small；在新的质量声明前先定义 acceptance threshold。
- **HAIC**：核验其余三模型、RTX 3090 profiles、latency-small 终点和正式 latency-aware VLA；不以 smoke 代替质量结果。
