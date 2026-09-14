# Balance Simple 本地 H8 训练

审计：2026-09-14。当前路线只训练 QwenOFT 的 zero / temporal profile 两个分支。

正式 RL 源运行于 2026-09-14 14:38:29 UTC 启动，group `balance-qwenoft-h8-20260914T143829Z`。
GPU 0/1 的 zero/profile RL（PID 3097021/3097022）继续各自 nominal 10M 预算；原调度 PID 3096908 已 SIGSTOP，停在等待 RL 的阶段。
[原运行状态](../outputs/tasks/humanoidbench_balance_simple/local_h8/balance-qwenoft-h8-20260914T143829Z/status.json)。
旧教师单次前向 H8 标签的数据与 smoke 保留为历史；本版各分支重新采集自己的 bootstrap。

当前冻结实现：`outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_v2/implementation`。
[实现补丁](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_v2/implementation.patch)、
[主环境与分支的真实状态/RNG检查](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_v2/verification/clone_contract.json)。
根目录入口 `scripts/tasks/humanoidbench_balance_simple/run_local_h8.py` 转入本版。
双分支完整 smoke 已通过（各 SFT 2 + DAgger 2×2；50 Hz、最多 8 帧）；[数值与 seed 协议](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_v2/verification/smoke_final_matrices.json)。
当前接续调度 PID 3173489、group `balance-qwenoft-h8-closed-loop-v2-20260914T152830Z`，已在 `existing_rl_running` 等待现有两路 RL；[实时状态](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_v2/formal/status.json)、[启动命令](../outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_v2/formal/launch.json)。
正式 5k SFT、10×500 DAgger、最终 25/100-episode 矩阵、acceptance 和 publication 尚未完成。

| 阶段 | 配方 |
|---|---|
| RL | GPU 0 zero、GPU 1 profile 并行；fresh seed=1；各 nominal 10M raw frames；198→8×43；共同 FastTD3 D/E 配方 |
| 数据 | 各自新教师，90 train / 10 val episodes；教师闭环运行完整 episode；每个当前 RGB/state anchor 的标签为未来 8 个 20ms 物理帧实际选中的绝对 request，控制器限幅前 |
| SFT | 先 zero 后 profile；各双卡；Qwen3-VL-4B + seed=0 新 action head；5k optimizer updates |
| DAgger | 各 10×500 updates；64k anchors/round、8 envs；seed=10000000+round×1000000；独立专家分支从学生当前状态推进 8 帧；optimizer/scheduler 跨轮恢复 |
| Batch | 初始每卡 64，global 128；OOM 缩 microbatch、增 accumulation 保持 global 128 |
| 归一化 | action 仅统计 mask 有效元素，43D；各自 bootstrap statistics 固定用于自己的 val/DAgger |

标签为 `issued_command_chunk [8,43]` 与必需的 `action_valid_mask [8]`（bool）。zero 每帧新规划、通常执行 head 0；temporal 遵循原 profile、capacity=1、latest_ready 和 hold-tail。
DAgger 主轨迹持续由学生控制，专家分支复制 MuJoCo、controller、history、RNG、time-limit、sampler、scheduler、pending timeline 的当前状态。
继承的学生请求、startup noop、episode 尾部 padding 为无效；终止帧自身有效动作保留。全无效 anchor 丢弃。
每个样本对有效的 43D 元素求平均 L1，再对 batch 求均值；分开记录学生物理帧与专家分支物理帧。
zero 只使用 zero small model/bootstrap/DAgger/stats，profile 只使用自己的；数据合同 `latency_bench_starvla_h8_v2`。

Mixture 配置权重 bootstrap 0.5、累计 DAgger 0.5；沿用 sequential sampler，配置权重不等于已测实际采样比例。
两个教师均完成预算、保存/重载及评测后启动 VLA，性能指标不作为阻断条件。
bootstrap 使用各条件独立的 `teacher-rollouts-h8-state-v2`，DAgger/VLA 使用新 run ID；旧标签版的本地数据与路径保留。

RL 每 1M 保留 checkpoint，既有 observer 在 RL 结束后处理周期评测。
周期：zero 8 reset seeds 1000000..1000007；profile 同 8×4 latency seeds 1000000..1000003。
最终两个教师和两个 VLA 均做矩阵：zero 25 reset seeds 1000100..1000124；profile 同 25×4 latency seeds 1000100..1000103。
每个 scenario 新 sampler，50 Hz，最多 1000 raw frames，确定性策略。
记录 return mean/std、mean length、fraction≥100/500、episodes。SF 的终止条件为严格超过预算，记录实际 saved raw-frame step。
固定使用已有 RTX 3090 QwenOFT H1 profile，**没有把它记作新的 H8 latency measurement**。

正式训练和评测使用 W&B online；smoke 只在本地验证，W&B disabled，不发布 HF。正式实验共用 fresh group、不同 stage IDs，DAgger 一个连续 run：

- RL / teacher eval：`saberrr-zju/latency-sensitive-bench`。
- VLA / student eval：`zihanwang-ai-northwestern-university/starvla_tasks`。

当前接续命令（已经运行）：

```bash
.venv/bin/python -m scripts.tasks.humanoidbench_balance_simple.run_local_h8 \
  --group balance-qwenoft-h8-closed-loop-v2-20260914T152830Z \
  --output-dir outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_v2/formal \
  --continue-rl-from outputs/tasks/humanoidbench_balance_simple/local_h8/balance-qwenoft-h8-20260914T143829Z \
  --rl-pids 3097021 3097022 --previous-runner-pid 3096908
```

运行状态在该目录 `status.json`；各阶段 log、checkpoint、seed protocol、Hub revision 随目录保存。
验证后自动追加 [PROGRESS](PROGRESS.md)，训练完成、评测完成、acceptance、publication 分别记录。
