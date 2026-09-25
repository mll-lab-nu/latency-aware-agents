# Balance Simple 本地 H8 训练

审计：2026-09-15。当前路线只训练 QwenOFT 的 zero / temporal profile 两个分支。

正式 RL 源运行于 2026-09-14 14:38:29 UTC 启动，group `balance-qwenoft-h8-20260914T143829Z`。
原 replay=1M 的 zero/profile RL 已完成，各最终 checkpoint 为 raw-frame step 10000256；两路旧教师的最终 zero / temporal 25/100-episode 矩阵均已完成。
[旧教师矩阵](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/formal/previous_round)。
旧教师单次前向 H8 标签的数据与 smoke 保留为历史；本版各分支重新采集自己的 bootstrap。

当前代码入口：[train_task](../scripts/starvla/train_task.py)、[闭环采集](../scripts/starvla/balance_dagger.py)。
实验冻结实现与验证记录保存在 `outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_v2`。
双分支完整 smoke 已通过（各 SFT 2 + DAgger 2×2；50 Hz、最多 8 帧）；[数值与 seed 协议](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_v2/verification/smoke_final_matrices.json)。
2026-09-14 按用户要求停止并删除新增接续调度，`formal` 目录已删除；[取消记录](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_v2/verification/continuation_cancelled.json)。
以上为当时取消接续的历史记录；2026-09-15 最新的直接启动指令已覆盖该暂停状态。

2026-09-15 按要求停下第二轮 10M RL 并改为每路 25M；group `balance-qwenoft-h8-replay6553600-25m-20260914T162609Z` 已完成。
[最终状态](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/formal/status.json)、[启动记录](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/formal/launch.json)、[调度日志](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/formal/pipeline.log)。
GPU 0 zero / GPU 1 profile 已并行从头重训，replay=**6,553,600**。旧 10M 调度 PID 3200200 已退出，两路停止时 checkpoint 为 884608 / 1085824 steps；[停止审计与旧产物](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/formal/budget25m_restart/stopped.json)。
新一轮是 fresh seed=1、H8、各 nominal **25M raw frames**；checkpoint 不包含 replay，因此未从停止点恢复。两路均已实际更新权重，W&B online 的预算、replay 与本地实际加载配置一致。
[第二轮 zero 配置](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/formal/zero_rl.yaml)、[第二轮 profile 配置](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/formal/profile_rl.yaml)、[W&B 配置核对](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/formal/budget25m_restart/wandb25m_verification.json)、[训练推进核对](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/formal/budget25m_restart/training25m_verification.json)。
冻结实现位于 `outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/implementation`；[25M 修改补丁](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/formal/budget25m_restart/budget25m.patch)。启动 cwd/PYTHONPATH 随 launch 保存。
两轮模型与相同 seed 协议的 25/100-episode 矩阵分别保留。
03:04:30 UTC 和 03:54:28 UTC 启动的含 DAgger 流水线分别在 zero SFT step 710 和 260 停止；均未达到首个 2500-step checkpoint，也未进入 DAgger。最新纯 SFT 流水线于 2026-09-15 04:08:26 UTC 启动，PID **3522591**，按 zero 后 profile 的顺序各训练 5000 steps，不执行 DAgger。
[VLA 状态](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/vla/sft5k_only/status.json)、[启动记录](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/vla/sft5k_only/launch.json)、[zero 日志](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/vla/sft5k_only/zero/bootstrap.log)。
实际生成配置为每 rank `num_workers=8`、`prefetch_factor=3`，每卡 batch 64、global 128，本地周期 checkpoint 保存间隔 **2500 optimizer steps**。zero 已验证开始更新，W&B online。每个 5k checkpoint 随后运行最终 zero 25 局 / temporal 100 局矩阵，再发布模型；发布使用 `delete_patterns=["**"]` 覆盖同路径旧文件。训练完成、评测完成、acceptance 和 publication 将分别记录。

| 阶段 | 配方 |
|---|---|
| RL | replay 1M 的 nominal 10M 旧轮已完成；replay 6,553,600 的 zero/profile fresh RL 也已完成，各保存 raw-frame step 25000320；seed=1、198→8×43，共同 FastTD3 D/E 配方 |
| 数据 | 各自新教师，90 train / 10 val episodes；教师闭环运行完整 episode；每个当前 RGB/state anchor 的标签为未来 8 个 20ms 物理帧实际选中的绝对 request，控制器限幅前 |
| SFT | 先 zero 后 profile；各双卡；Qwen3-VL-4B + seed=0 新 action head；5k optimizer updates；每 rank workers 8、prefetch 3；本地 checkpoint interval 2500 |
| DAgger | 不执行；每个分支的 `dagger_rounds=0` |
| Batch | 初始每卡 64，global 128；OOM 缩 microbatch、增 accumulation 保持 global 128 |
| 归一化 | action 仅统计 mask 有效元素，43D；各自 bootstrap statistics 固定用于自己的 validation 和 SFT |

标签为 `issued_command_chunk [8,43]` 与必需的 `action_loss_mask [8]`（bool）。zero 每帧新规划、通常执行 head 0；temporal 遵循原 profile、capacity=1、latest_ready 和 hold-tail。
继承的 teacher 请求、startup noop、episode 尾部 padding 为无效；终止帧自身有效动作保留。全无效 anchor 丢弃。
每个样本对有效的 43D 元素求平均 L1，再对 batch 求均值。zero 只使用 zero small model/bootstrap/stats，profile 只使用自己的；数据合同 `latency_bench_starvla_h8_v2`。

VLA 使用已完成 25M 预算、保存/重载及评测的对应教师；当前 zero→profile 流水线正在运行。
bootstrap 使用各条件独立的 `teacher-rollouts-h8-state-v2`，VLA 使用新 run ID；旧标签版的本地数据与路径保留。

RL 每 1M 保留 checkpoint，既有 observer 在 RL 结束后处理周期评测。
周期：zero 8 reset seeds 1000000..1000007；profile 同 8×4 latency seeds 1000000..1000003。
最终两个教师和两个 VLA 均做矩阵：zero 25 reset seeds 1000100..1000124；profile 同 25×4 latency seeds 1000100..1000103。
每个 scenario 新 sampler，50 Hz，最多 1000 raw frames，确定性策略。
记录 return mean/std、mean length、fraction≥100/500、episodes。SF 的终止条件为严格超过预算，记录实际 saved raw-frame step。
固定使用已有 RTX 3090 QwenOFT H1 profile，**没有把它记作新的 H8 latency measurement**。

正式训练和评测使用 W&B online；smoke 只在本地验证，W&B disabled，不发布 HF。正式实验共用 fresh group、zero/profile 使用不同 run ID：

- RL / teacher eval：`placeholder/placeholder`。
- VLA / student eval：`placeholder/starvla_tasks`。

各阶段 log、checkpoint、seed protocol 和发布 revision 随运行目录保存。
验证后更新 [PROGRESS](PROGRESS.md)，训练完成、评测完成、acceptance、publication 分别记录。


2026-09-15 W&B 周期评测修复已验证：本轮 [zero RL](https://wandb.ai/placeholder/placeholder/runs/balance-qwenoft-h8-20260914T143829Z-zero-rl) / [profile RL](https://wandb.ai/placeholder/placeholder/runs/balance-qwenoft-h8-20260914T143829Z-profile-rl) 各回填 10 个 checkpoint 评测点（共 20），`eval/*` 横轴为实际 checkpoint 的 `global_step`，保留 `eval/global_steps`。远端已逐点核对。两份冻结实现均已改为由原训练进程上传周期评测，子评测不再新建独立 run。旧的 20 条独立 run 标记 `[migrated]` 并移到 `balance-qwenoft-h8-20260914T143829Z-periodic-eval-archive` 分组；这些源 run 可删除，父 run 已独立保存回填结果。[完整审计](PROGRESS.md)、[数值核对](../outputs/tasks/humanoidbench_balance_simple/local_h8/balance-qwenoft-h8-20260914T143829Z/wandb_repair/verification.json)、[迁移核对](../outputs/tasks/humanoidbench_balance_simple/local_h8/balance-qwenoft-h8-20260914T143829Z/wandb_repair/archive_verification.json)。本修复未改变训练计划。
