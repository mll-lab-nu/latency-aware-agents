# FastTD3 延迟学习实验记录

统一口径：training seed=1；存活单位为环境帧。本文只记录实验动机、改动、指标和结论。

## 结论摘要

- zero latency下，H只给actor有效；H进入critic会明显退化。
- fixed1下，执行时序value更新F2成功，6M平均存活525.44帧。
- fixed2和QwenOFT profile上的后续修正均未恢复；主要退化来自service2造成的25Hz接纳/两帧命令保持。
- 最新admission-boundary SMDP失败：5M固定100场景平均存活62.48帧，0/100达到100帧。

## 1. H输入拆分（zero latency）

**为什么改：** 定位完整H输入的学习退化来自actor还是critic。

**改动：** 比较198D、完整H、H网络两侧遮蔽、仅actor使用H、仅critic使用H。

| 条件 | 步数 | 平均存活 | 平均回报 | 达到100帧 | 结论 |
|---|---:|---:|---:|---:|---|
| 198D baseline | 6M | 306.64 | 281.76 | 25/25 | 成功 |
| 完整H | 5M | 67.92 | 50.95 | 0/25 | 失败 |
| H网络，两侧只用198D | 6M | 223.68 | 201.50 | 25/25 | 部分改善 |
| H只给actor | 6M | 549.44 | 516.77 | 25/25 | 成功 |
| H只给critic | 6M | 92.84 | 74.82 | 3/25 | 失败 |

**结论：** H信息本身不是问题；退化来自H进入critic。

证据：[实验记录](../../outputs/latency_rl/balance_fasttd3_debug_s1_20260909/RUN.md)。

## 2. Critic字段拆分（zero latency）

**为什么改：** 区分critic退化来自53D context还是1595D history。

**改动：** actor固定使用H，critic分别使用198D、198D+context、198D+history。

| Critic输入 | 平均存活 | 平均回报 | 达到100帧 | 结论 |
|---|---:|---:|---:|---|
| 198D | 549.44 | 516.77420 | 25/25 | 成功 |
| 198D + context | 163.08 | 140.16326 | 25/25 | 退化 |
| 198D + history | 105.04 | 89.29747 | 15/25 | 失败 |

**结论：** context和history都会伤害critic，history更严重。

## 3. Fixed1信用分配：F1/Q与F2/value

**为什么改：** fixed1中，当前动作不会影响当前reward和下一帧198D状态；原单步Q target无法区分当前候选动作。真实反事实中三个动作的折扣回报为35.57、33.69、38.13，但原生单步target都为26.699。

**改动：**

- F1：actor输入`(198D physical, 43D pending)`，critic学习`Q(o,u,a)`。
- F2：相同actor输入，critic学习执行时序`V(o,u)`。

| 条件 | 1M存活 | 5M存活 | 6M存活 | 6M回报 | 达到100帧 | 达到500帧 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| F0：198D baseline | 49.84 | 76.20 | 87.52 | 74.13425 | 6/25 | 0/25 | 失败 |
| F1：pending + Q | 54.48 | 120.52 | 158.56 | 139.92020 | 25/25 | 0/25 | 部分改善 |
| F2：pending + value | 69.08 | 469.12 | 525.44 | 485.85413 | 25/25 | 16/25 | 成功 |

**结论：** fixed1需要显式执行时序信用；F2是唯一成功修正。

证据：[实验记录](../../outputs/latency_rl/balance_fasttd3_fixed1_s1_20260910/RUN.md)。

## 4. QwenOFT profile-value推广

**为什么改：** 把fixed1成功的F2推广到真实QwenOFT temporal profile。

**改动：** 保留241D actor；critic改为198D physical + 138D transport的queue value。

| 步数 | 平均存活 | 平均回报 | 达到100帧 | 达到500帧 |
|---:|---:|---:|---:|---:|
| 1M | 56.93 | 44.06464 | 0/100 | 0/100 |
| 5M | 85.17 | 71.65363 | 7/100 | 0/100 |
| 10M | 112.59 | 96.80489 | 71/100 | 0/100 |
| 15M | 114.27 | 99.80030 | 84/100 | 0/100 |

**结论：失败。** 最终停在约114帧。

证据：[15M评估](../../outputs/latency_rl/balance_fasttd3_qwenoft_profile_s1_20260910/fasttd3_qwenoft_profile_value_15m_s1_r1/final_backfill.json)。

## 5. Execution-boundary value

**为什么改：** 检查profile-value失败是否来自单物理帧target没有覆盖完整等待段。

**改动：** critic改为实际执行边界间的可变长度前缀；累计reward并使用`gamma^k`。

| 步数 | 平均存活 | 平均回报 | 达到100帧 |
|---:|---:|---:|---:|
| 1M | 47.43 | 33.69522 | 0/100 |
| 5M | 101.12 | 86.23539 | 50/100 |

**结论：失败。** 5M平均存活101.12帧。

证据：[5M评估](../../outputs/latency_rl/balance_fasttd3_qwenoft_execution_s1_20260910/fasttd3_qwenoft_execution_value_15m_s1_r1/matched_5000064.json)。

## 6. Queue critic对照

**为什么改：** 区分额外queue输入、fixed2和真实profile的影响。

**改动：** 相同241D actor和336D critic宽度；只改变延迟条件及critic的transport投影。

| 条件 | 平均存活 | 平均回报 | 结论 |
|---|---:|---:|---|
| fixed1，pending-only critic | 574.08 | 538.17931 | 成功 |
| fixed1，full queue critic | 657.32 | 613.29038 | 成功 |
| fixed2，full queue critic | 103.20 | 87.64558 | 失败 |
| QwenOFT profile，full queue critic | 106.00 | 89.79386 | 失败 |

**结论：** full queue输入不是失败原因；退化随fixed2/service2出现。

证据：[汇总](../../outputs/latency_rl/balance_fasttd3_queue_ablation_s1_20260910_r1/summary.json)。

## 7. Execution critic

**为什么改：** 让critic只从真实新命令执行起点学习，避免hold帧重复采样。

**改动：** 新增execution起点采样，critic按执行边界聚合reward；actor保持原接纳时刻更新。

| 条件 | 平均存活 | 平均回报 | 达到100帧 | 结论 |
|---|---:|---:|---:|---|
| fixed1 execution | 397.52 | 367.41461 | 25/25 | 较旧fixed1明显退化 |
| fixed2 execution | 97.80 | 82.14970 | 12/25 | 失败 |

**结论：失败。** Execution采样没有修复fixed2，并损害fixed1。

## 8. Fixed2接纳归一化与低噪声

**为什么改：** 分别检查drop导致的actor梯度缩放和过强探索。

**改动：** 一组按实际接纳数归一化actor目标；另一组只把`std_max`从0.4降到0.1。

| 条件 | 1M存活 | 5M存活 | 6M存活 | 6M回报 | 达到100帧 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 原fixed2 | 57.12 | 87.28 | 97.80 | 82.14970 | 12/25 | 失败 |
| 接纳归一化 | 40.76 | 97.56 | 107.00 | 90.54665 | 21/25 | 失败 |
| `std_max=0.1` | 41.08 | 111.20 | 117.44 | 102.33255 | 22/25 | 失败 |

**结论：** 两项都有小幅改善，但都停在约100帧。

证据：[接纳归一化](../../outputs/latency_rl/balance_fasttd3_fixed2_actor_credit_s1_20260911_r1/runs/balance_fasttd3_fixed2_actor_credit_s1_20260911_r1_F_fixed2_execution/matched_6000000.json)、[低噪声](../../outputs/latency_rl/balance_fasttd3_fixed2_noise01_s1_20260911_r1/runs/balance_fasttd3_fixed2_noise01_s1_20260911_r1_F_fixed2_execution/matched_6000000.json)。

## 9. Service/ready交叉对照

**为什么改：** 区分service2造成的25Hz接纳与ready2造成的两帧返回。

**改动：** 补齐`service2/ready1`和`service1/ready2`两个交叉条件。

| Worker service | ready1平均存活 | ready2平均存活 |
|---|---:|---:|
| service1 | 397.52 | 225.52 |
| service2 | 114.32 | 97.80 |

| 新条件 | 平均回报 | 达到100帧 | 达到500帧 | 结论 |
|---|---:|---:|---:|---|
| service2/ready1 | 99.39141 | 20/25 | 0/25 | 失败 |
| service1/ready2 | 203.50115 | 25/25 | 0/25 | 失败 |

**结论：** 主导退化来自service2的25Hz接纳/两帧命令保持；ready2是次要退化。

证据：[汇总](../../outputs/latency_rl/balance_fasttd3_service_ready_s1_20260911_r1/summary.json)。

## 10. Admission-boundary SMDP

**为什么改：** service2下约一半动作被drop；尝试只在真实接纳边界分配credit。

**改动：** actor/critic使用`198D physical + 43D pending + 138D causal transport`；replay只保存接纳动作，并聚合接纳间reward与`gamma^k`。actor从fixed1成功策略热启动，critic和optimizer重置，`std_max=0.1`。仍输出单个43D action，不使用action chunk监督。

| 运行 | 平均存活 | 平均回报 | 达到100帧 |
|---|---:|---:|---:|
| Smoke（2,055 admitted） | 18.92 | 12.52254 | 0/100 |
| Formal 1M | 38.07 | 28.69803 | 0/100 |
| Formal 5M | 62.48 | 49.39153 | 0/100 |

**结论：失败。** 5M结果低于profile-value的85.17帧和execution-boundary的101.12帧。

证据：[1M评估](../../outputs/latency_rl/balance_fasttd3_qwenoft_admission_s1_20260911/balance_fasttd3_qwenoft_admission_10m_s1_r1/matched_1000064.json)、[5M评估](../../outputs/latency_rl/balance_fasttd3_qwenoft_admission_s1_20260911/balance_fasttd3_qwenoft_admission_10m_s1_r1/matched_5000064.json)。

## 11. 25Hz接纳边界正对照

**为什么改：** 旧slow-world同时降低了物理频率，没有验证50Hz物理、25Hz决策、两帧hold本身是否可学。

**改动：** 新增zero与fixed40/F2两个25Hz配方；每个决策推进两个50Hz环境帧，累计两帧reward，decision discount为`0.99²`。评估的`length`和`mean_length`改为真实raw frames，另报`decision_count`，接纳率仍以decision为分母。

**等价性证据：** 确定性测试逐个接纳边界比较25Hz/fixed40与50Hz-attempt/fixed2；physical observation、两帧累计reward、pending command、实际执行command和终止位置一致。目标回归结果为153 passed；另有13项仅因当前环境缺少可选`flappy_bird_gymnasium`依赖而未收集。

**A smoke命令：**

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python -m scripts.experiments.ppo_latency.balance --config configs/training/latency/balance_fasttd3_25hz_zero.yaml --phase zero --train-for-env-steps 2048 --device gpu --root outputs/latency_rl/balance_fasttd3_25hz_s1_20260911/a_zero_smoke_r1
```

**A smoke结果：** 2,048 decisions完成；replay已进入update，`critic_loss=9.17795`、`actor_loss=0.05253`且有限；checkpoint中的120个tensor全部有限。25个评估场景平均存活14.52 raw frames。此短测只验证运行链路，不用于判断学习效果。

**A formal命令：**

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python -m scripts.experiments.ppo_latency.balance --config configs/training/latency/balance_fasttd3_25hz_zero.yaml --phase zero --device gpu --root outputs/latency_rl/balance_fasttd3_25hz_s1_20260911/a_zero_6m_r1
```

| 检查点 | 平均存活（raw frames） | 平均回报 | 达到100帧 | 达到500帧 |
|---:|---:|---:|---:|---:|
| 初始 | 14.60 | 8.96267 | 0/25 | 0/25 |
| 1,000,064 decisions | 68.16 | 54.27284 | 0/25 | 0/25 |
| 5,000,064 decisions | 127.92 | 110.39096 | 24/25 | 0/25 |
| 6,000,000 decisions | 156.08 | 137.46856 | 24/25 | 0/25 |

**A formal结果：失败。** 连续6M decisions完成，训练耗时1,604.5秒，所有场景接纳率为1。`>=100`比例通过，但平均存活未达到300 raw frames，因此按预定gate不启动B。

**失败层定位：** 1M和5M的原生update snapshot均有限，重放后的learner、replay、optimizer和target状态与下一snapshot逐项相等，最大绝对差为0；对应归一化observation RMS为0.950/0.943。6M反事实为24/48个严格Q排序正确，而已恢复的fixed1 F2也是23/48，故该指标不是A独有的错误。最后将原始强50Hz zero teacher（50Hz固定场景平均663.38帧）不改权重地放入真实25Hz/hold2环境：平均只剩57.96 raw frames，1/25达到100帧。反向把A actor改为50Hz调用也只有148.68帧，未揭示被wrapper隐藏的好策略。

**结论：** 第一个失败层是两帧动作保持造成的控制带宽变化，不是已检查的环境时序、flat replay或learner update实现错误。当前证据不支持修改learner；继续B会把固定延迟与已失败的hold2控制基线混在一起，因此停止。若要继续，需要另立动作chunk或低层插值控制假设，超出本轮已批准的单43D command方案。
