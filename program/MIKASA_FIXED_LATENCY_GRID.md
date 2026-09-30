# MIKASA QwenOFT H8 fixed-latency grid

> 本页保留 71D v2 合同的原矩阵方案与历史状态。当前工作已转为
> [63D Fixed-2 合同修复与重训](MIKASA_FIXED2_PREFIX_REPAIR.md)，旧结果只作对照。
> 新评测入口显式使用传入 DAgger run 的 best，不再按 train latency=2 自动加载旧 Hub 模型。

2026-09-24 当时阶段：用户已批准同步和正式启动，并明确要求跳过 GPU 短流程。两台 H200 的独立
运行目录已同步。2026-09-24 用户在旁支明确要求停止第二台 H200 GPU 2/3、4/5 上的
任务，改为启动 Asterix GR00T、PI；对应 Fixed-3/4 启动器及训练进程已停止，产物保留，
不应按意外故障自动重启并抢占这四张卡。用户随后明确要求也停止 Fixed-1（GPU 0/1），
其启动器及训练进程现已停止，0/1 保持空闲；Fixed-1/3/4 均为用户主动停止。
远端 Fixed-1 的 `phase` 为 `stopped-by-user`，Fixed-3/4 为
`stopped-by-user-gpu-reassignment`，各自的 `user_stop.json` 记录原因，产物均保留。
Fixed-2 的五档评测已全部完成；新模型训练与完整5×5尚未完成。仅暂存相关代码，不自动 commit。
[执行记录](logs/mikasa-fixed-grid-2026-09-24.md)。

后续 63D v3 四档重采集与阶段发布使用 [prefix63 重跑协议](MIKASA_PREFIX63_RESTART.md)；
本页的停止、运行目录与完成描述按上述历史日期读取。最新核验状态归
[PROGRESS](PROGRESS.md)，旧 71D 矩阵仍保留作对应 checkpoint 的历史证据。

## 冻结协议

- 模型：QwenOFT，H8×7；新训练 Fixed-1/3/4，复用已有 Zero 和 Fixed-2。
- 20Hz control / observation，60 个实际 transitions；reset 后立即开始计时、奖励与步数。
- 每帧尝试请求，capacity 1，忙时丢弃请求而不排队；等待时继续旧 chunk。
- 请求 t 的 head j 对应 t+j。到达时选 `current_frame - observation_frame`；耗尽保持末动作。
- Conditioned state 为 71D，K 固定在训练合同中。先接收已到达结果，再投影本次请求的
  K 个旧计划动作；推理返回后在任务适配层覆盖 prefix heads。L0 同样保留 K。
- Naive state 为 7D、无 prefix。不同列同时改变训练 latency 和固定动作承诺，不解释为
  完全相同输入/控制器下只改变训练 latency 的消融。
- 不向模型传入本次实际 L。L>K 时等待轨迹可偏离 teacher 生成标签时的 K 帧等待假设。

下面的 `C_t[j]` 表示最初产生该动作的请求及 head：

| 策略 / L | 第 0–5 帧执行来源 |
|---|---|
| Naive / 0 | C0[0], C1[0], C2[0], C3[0], C4[0], C5[0] |
| Naive / 2 | noop, noop, C0[2], C0[3], C2[2], C2[3] |
| K2 / 0 或 1 | noop, noop, C0[2], C1[2], C2[2], C3[2] |
| K2 / 2 | noop, noop, C0[2], C0[3], C2[2], C2[3] |
| K2 / 5 | 前五帧 noop；第 5–9 帧 C0[5], C0[6], C0[7], C0[7], C0[7] |

## 本地审查范围

本次仅合入远端 H8 所需的任务注册、profile/fixed prefix、SFT 配置与评测审计；
保留本地其他任务的实现。QwenOFT 不在模型内部使用显式 prefix，71D state 提供旧计划
上下文，硬覆盖由普通推理和 resident 推理共用的 `TaskStarVlaPolicyRunner` 执行。
DAgger 使用既有通用 bundle 导出接口，不引入无用 task 参数或 backbone 下载回退。

Fixed 条件统一使用 `fixed-1/2/3/4`。`eval_config(..., eval_latency_frames=L)` 独立控制
评测延迟，`None` 保留 profile 评测；单 checkpoint CLI 使用 `--setting zero|fixed-N|profile`。

必要回归放在已有 MIKASA、task policy、scheduler 和 resident 测试中；都是长期协议测试：
请求/到达/prefix 时间线，K 不随 L 改变，K 帧标签 mask 与终止尾部 mask，episode reset，
checkpoint/resident 共用硬覆盖，SFT→DAgger 的 batch、checkpointing 和恢复配置。
既有图表发布/诊断的未提交改动不属于本次暂存范围。

## 训练与恢复

每路重新采集对应 K 下的 teacher 数据：train/validation 至少 64,288 / 6,884 个有效
suffix head；不改名复用 Fixed-2 标签。prefix 和 episode 终止后的 tail 不参与 loss。
沿用现有 PPO teacher；Qwen backbone 固定到
`Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17`。

SFT 5,000 updates，seed 0，双 H200，per-device batch 64，global 128，accumulation 1，
ZeRO-2，**gradient checkpointing 关闭**，OOM 不自动缩 batch。base/action LR 为
3e-5/1e-4，cosine floor 1e-6，warmup 100，沿用既有 AdamW 设置。
每 500 steps 保存模型与完整 trainer state；activation checkpointing 开关不影响恢复保存。

DAgger 从 SFT 权重初始化新优化器，10×500 updates；每轮至少 16,000 有效 suffix heads。
首轮 bootstrap/current=50/50；以后 bootstrap/current/history=40/40/20，历史轮等权。
增量数据沿用该路 bootstrap normalization 和 validation。两个 rank 暂停训练后用常驻
模型评测/采集，完成后继续原优化器状态，不启动第二份常驻 VLA。

在 step 0、500…5000 做 matched-latency 开发评测，各 100 局，seeds
`4242424442–4242424541`。best 按成功数、mean return、较晚 step 排序；完整预算仍跑完。
DAgger 每轮采集 seed 起点为 `10,000,000 + round×1,000,000`。原 baseline/best/latest
独立终评保持 200 局 seeds `4242424542–4242424741`，不与下面主矩阵混用。

### 正式命令入口

在两台 H200 任意已确认空闲的卡上安排三组双卡；每台训练机额外安排空闲 simulator 卡，
沿用已有 GPU lock 串行化同卡采集/评测。以下卡号仅演示一组，启动前重新核实。
每次实验使用独立数据、SFT 与 DAgger 输出目录；入口不删除旧数据/训练目录。


按用户最新指令直接运行上述正式入口，不运行短流程。正式 SFT
恢复使用相同 run-id/output-dir 加 `--resume --train-only`；DAgger 重启相同命令，复用完整
双 rank 边界与已打包增量，不重复已完成轮次。partial eval 保留供检查，用新 label 重跑。

## 5×5 正式矩阵

横轴 train latency、纵轴 eval latency，均为 0–4 帧。每列只固定一次 best checkpoint，
不能按测试格子选优。主矩阵每格 200 episodes，seeds `4242424242–4242424441`，
deterministic inference，10 个并行环境，inference batch 8，单 worker 的模拟延迟。

- Zero：复用 `figure2_h1_h8_naive_dagger_fixed_latency_20260923/combined_results.json` 中
  `qwenoft_h8`，Hub `5b533937a72e9fa923ecf28b1c1be5a16bad488d`、DAgger step 5000。
  L0–L4 成功数依次为 **195、113、115、44、15 / 200**。
- Fixed-2：从 Hub `795468e53947d26314e891cbc375775c6e0bd491` 下载对应 DAgger bundle，
  step 5000；包括 L2 在内的五格都按矩阵 seeds 重评，不复用旧 heldout 结果。
- Fixed-1/3/4：使用完成的 DAgger run 中的 best；若开发选优保留 SFT step 0，使用 source
  bundle，训练完成记录仍必须是 SFT 5k + DAgger 5k。

新增 20 格共 4,000 局，开发与原 DAgger 独立终评另计。每格写独立配置、manifest、
逐 episode/step/action/latency 记录和 summary。源码 revision/dirty 状态按实际运行记录。

```bash
# Fixed-2 的 model-dir 是固定 revision 的下载缓存；其他列是完成的 DAgger run。
python -m scripts.tasks.mikasa_intercept_grab_fast.fixed_latency_grid eval \
  --train-latency 2 --eval-latency 0 \
  --model-dir "$PWD/outputs/starvla/artifacts/fixed2-grid-795468e5" \
  --backbone-path /path/to/Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --vla-python "$QWENOFT_PY" --task-python "$MIKASA_PY" --devices 0,1 \
  --output "$PWD/outputs/tasks/mikasa_intercept_grab_fast/fixed-grid-20260924"

# 在20格完成后汇总，复用已经核验的Zero列。
python -m scripts.tasks.mikasa_intercept_grab_fast.fixed_latency_grid aggregate \
  --zero-results outputs/tasks/mikasa_intercept_grab_fast/figure2_h1_h8_naive_dagger_fixed_latency_20260923/combined_results.json \
  --output outputs/tasks/mikasa_intercept_grab_fast/fixed-grid-20260924
```

汇总重新核算新格子的原始记录，核对共同 seeds、60 transitions、实际 latency、固定 K
和同列 checkpoint。输出 `grid.json` 与 `grid.csv`，包含成功率、成功数/200、mean return、
population SD、checkpoint 和证据路径。成功率低也保留，不追加训练或挑测试结果。

全部正式结果完成后，在 `paper/figures/5_1_latency_robustness` 增加独立 MIKASA 图。
只画 success rate，线性色标固定 0–100%，格内标百分比；读取归档数据，使用现有
`plot_style.mplstyle` 与 `paper_colors.py`，输出 PDF/300-DPI PNG。未完成格子不填造数值。

训练完成、评测完成、性能验收和发布分别记录；本阶段不发布或迁移 Hub 目录。
