# Balance latency RL handoff（2026-09-09）

## 当前结论与停止状态

1. **PPO 无延迟 baseline 同样学得很差。** 同一套 H 输入、随机初始化、
   actor/critic LR `3e-4`、target-KL `0.03` 的 zero 对照跑满约 10M 后，
   平均存活只有 **79.44 帧**，fixed1 对照为 **70.00 帧**。这支持用户的判断：
   **PPO 当前的主要学习瓶颈应先从无延迟下就存在的问题解释，不能归咎于
   fixed latency。** 这不等于已经排除延迟的额外影响，也没有定位具体根因。
2. **FastTD3 修复后可以正常训练，但与 PPO 一样出现了学习性能瓶颈。**
   当前 H + fixed1 配方从随机初始化训练到约 **5.24M**，训练过程正常，
   但周期评估平均存活仍徘徊在约 **60–70 帧**；最高 **72.38 帧**，
   最后一次为 **70.50 帧**，尚无评估 episode 达到 100 帧。
   修复有效，但没有解决策略学不好的问题。此前“FastTD3 训练异常崩塌”
   不再是当前状态；现在需要解释的是正常训练下仍然存在的性能瓶颈。
   这与 PPO 的现象相似，但尚不能认定二者具有同一个具体根因。

后续围绕 PPO / FastTD3 的学习瓶颈排查。

## 已核对的结果

| 实验 | checkpoint / 评估口径 | 平均存活帧 | 平均回报 | ≥100 | ≥500 |
|---|---|---:|---:|---:|---:|
| PPO zero，LR 3e-4 / KL 0.03 | 10,010,624；最终 25 seeds | 79.44 | 63.9734 | 2/25 | 0/25 |
| PPO fixed1，LR 3e-4 / KL 0.03 | 10,010,624；最终 25 seeds | 70.00 | 55.8271 | 0/25 | 0/25 |
| PPO fixed1，LR 3e-4 / KL 0.1 | 10,010,624；最终 25 seeds | 64.68 | 45.0936 | 0/25 | 0/25 |
| PPO fixed1，actor LR 3e-5 / KL 0.008 | 10,010,624；最终 25 seeds | 79.72 | 60.0872 | 0/25 | 0/25 |
| 旧 FastTD3 zero，198D | 25M 运行的 best；旧评估 10 episodes | 735.00 | 702.2445 | 10/10 | 10/10 |
| 修复后 FastTD3 fixed1，H | 4,800,000；最高周期均值，8 seeds | 72.38 | 59.8061 | 0/8 | 0/8 |
| 修复后 FastTD3 fixed1，H | 5,200,000；最后周期评估，8 seeds | 70.50 | 56.7974 | 0/8 | 0/8 |

本轮周期 seeds 为 `1000000–1000007`，PPO 最终 seeds 为
`1000100–1000124`，每局最多 1000 帧。旧 FastTD3 使用不同评估口径和更长
训练预算；其结果作为“原 zero 配方能学好”的参照，不是严格同预算因果对照。
本轮所有训练均为 seed 1。PPO 的 `status=rejected` 表示性能门槛未通过，
三组新 zero / fixed1 KL 对照的训练预算均已完成。修复后 FastTD3 提前停止，
与 PPO 的训练预算和评估 seeds 不同；两者同样遇到性能瓶颈是现象判断，不是严格排名。

## 本轮配方与关键差异

- **PPO zero / fixed1：**相同 seed 1 随机 actor/critic，空 Adam，H 1846D；
  actor/critic LR 均 `3e-4`，target-KL `0.03`。冻结 teacher observation
  统计，未加载 teacher actor/critic 权重。两组实际初始模型 tensor 已核对一致。
- **FastTD3 fixed1：**`initial_model_path: null`，随机初始化，对完整 H 在线
  empirical normalization。保留原 actor `512/256/128`、critic `1024/512/256`、
  LR `3e-4`、128 workers、rollout 1、replay batch 32,768、每 64 transitions
  的 update credit 和 compile。
- **历史去重 replay：**保留全部 **6,553,600** 条 float32 transitions，按真实
  env_id 保存单帧事件并重建 H；事件环额外保留 29 帧。实测 replay 14.69 GiB，
  GPU 全容量 compiled 更新和原生 smoke 通过，相关 103 项测试通过。
  这些检查验证了所测实现行为，不能代替当前学习失败的根因排查。
- **旧 FastTD3 zero：**输入为 198D，旧调度为 `issue_order_fifo`、未设 capacity、
  noop 为全零；本轮使用下述当前 fixed1 合同。不能将新旧差距全部直接归因于
  一帧延迟。

配置、初始化核验、测试和显存记录见
[本轮 RUN.md](../../outputs/latency_rl/balance_ppo_fasttd3_comparison_s1_20260909/RUN.md)。

## 当前 fixed1 合同

- 任务 `g1_sonic_balance_simple`，50Hz，每步 `2ms × 10` 次 MuJoCo physics。
- 43D normalized action 转为 absolute-position command；固定延迟 20ms，
  capacity=1，`hold/latest_ready`，reset 后先执行原 absolute-position noop。
- 每帧接纳一个 command，admission fraction=1，等待恰好一帧。
- H=`198 + 53 + 29×55 = 1846`，actor/critic 都读取完整 H，全部为因果信息。
  198D 主体已包含 `112D qpos/qvel + 43D q_cmd + 43D v_cmd`；
  **先前“H 缺少 controller 状态”的判断已纠正，不应继续沿用。**
- PPO 使用冻结 observation 统计，actor loss/KL 仅用 admitted frames，
  GAE/critic 使用全部有效帧；FastTD3 对完整 H 在线归一化，两者不可混写。

H 字段见 [latency_observation.py](../../training/common/latency_observation.py)，
queue 见 [latency_wrapper.py](../../training/common/latency_wrapper.py)。

## 仍有价值的既有排查结论

- **action/reward 时序已验证。** fixed1 的第 t 次 action 入队，当前物理帧执行
  上一 observation 的 command，入队结果供下一帧执行。五步 trace 的
  `applied_source_obs_id=[-1,0,1,2,3]`，`-1` 为 reset noop；无需额外平移
  reward、done 或 advantage。见 [时序 trace](../../outputs/latency_rl/balance_fixed1_root_cause_s1_20260909/fixed1_timing_trace.json)。
- **fixed1 存在高质量解。** 精确模拟器一步预测后执行 teacher，25 seeds 平均
  存活 **723.36**，≥100 为 100%、≥500 为 96%，预测误差为 0。该 oracle
  保留原 queue/noop/controller/scheduler，但访问模拟器副本，不是可部署策略。
  见 [oracle report](../../outputs/latency_rl/balance_fixed1_root_cause_s1_20260909/oracle_v1/REPORT.md)。
- **现有 H actor 的 BC 没有学会 oracle。** episode-split 验证 MSE 仅下降
  51.6%，闭环平均 **42.12** 帧；不足以单独证明输入缺失、模型容量或数据覆盖
  是根因。见 [BC report](../../outputs/latency_rl/balance_fixed1_root_cause_s1_20260909/bc_v1/REPORT.md)。
- **GAE 单 batch 对照没有支持 λ=1 的优势。** 在实际 KL 匹配的一次更新中，
  λ=.95 / 1 的存活为 35.625 / 34.750，不能据此排除所有信用分配问题。
  见 [matched update report](../../outputs/latency_rl/balance_fixed1_root_cause_s1_20260909/matched_v2/REPORT.md)。
- **调 LR / 放宽 target-KL 未解决 PPO 平台。** 较大 LR 和 KL 带来前期加速，
  KL=0.1 也出现明显回撤；最终结果仍差。target-KL 只阻止后续 actor 更新，
  不撤销已执行的更新；原 `shuffle_minibatches=false`，不能把前几个 minibatch
  当作随机抽样。详见 [KL sweep RUN.md](../../outputs/latency_rl/balance_ppo_kl_sweep_s1_20260909/RUN.md)。

当前应解释的是：**为什么 PPO 连 zero baseline 都学不好，以及为什么 FastTD3
在修复后正常训练的情况下，仍与 PPO 一样停留在低存活时长的平台。** 两者的具体
学习瓶颈尚未定位。此前的 LR/KL、oracle/BC 候选方向不再作为当前执行计划；
所有训练均已停止。

## 产物索引

- [本轮比较报告](../../outputs/latency_rl/balance_ppo_fasttd3_comparison_s1_20260909/REPORT.md)、
  [PPO zero 最终逐局结果](../../outputs/latency_rl/balance_ppo_fasttd3_comparison_s1_20260909/ppo_zero_lr3e4_kl003_10m_s1/final.json)。
- [修复后 FastTD3 最后周期结果](../../outputs/latency_rl/balance_fasttd3_c51_fix_s1_20260909/fasttd3_history_fixed1_c51_10m_s1/periodic/5200000.json)、
  [本次停止状态与 checkpoint](../../outputs/latency_rl/balance_fasttd3_c51_fix_s1_20260909/completion.json)。
- [旧 FastTD3 zero 结果](../../results/humanoidbench_g1_balance_control_comparison_fasttd3_25m/g1_sonic_balance_simple/raw_ramp_0p1s/eval_best_l0/queue_eval_latency_0.json)。
- [停止记录](../../outputs/latency_rl/balance_ppo_fasttd3_comparison_s1_20260909/stop_requested.json)、
  [FastTD3 停止状态与 checkpoint](../../outputs/latency_rl/balance_ppo_fasttd3_comparison_s1_20260909/fasttd3_history_fixed1_25m_s1/completion.json)。
- 历史细节留在各 RUN/REPORT 和 [LATENCY_PLAN.md](LATENCY_PLAN.md)，
  PPO 字段合同见 [latency_ppo.md](../../docs/latency_ppo.md)。
