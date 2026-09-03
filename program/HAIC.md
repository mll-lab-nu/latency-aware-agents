# HAIC / PullCart 原生 teacher 与 VLA student 手册

状态：HAIC 已作为 fork submodule 接入；正式 500M teacher 长训已完成。旧 256D latent-only
bootstrap/DAgger 的 step 0、round 0、1、2 均为 `0/100`，已在 round 3 停止且原输出完整保留。
action-only 数据、loss、固定 actor 导出和 oracle gate 已实现；下一次从 GR00T base fresh bootstrap，
只执行 round 0。

## 1. 固定口径

- 任务：`G1/haic/pull_cart`，PullCart motion `data/motion/g1/mirobotA/pull_cart/motion.npz`。
- 原生时间：physics 200 Hz / 5 ms，policy 50 Hz / 20 ms，decimation 4；motion 561 帧 / 11.22 s；动作是 23 个非 wrist 关节。
- 训练路线：`ppo_haic_train` 同一在线 rollout 同时训练 privileged teacher、privileged target world model/adapter，并将 teacher 均值蒸馏到 student；不是离线 BC。原生 `ppo_haic_finetune` 是可选的后续 student on-policy PPO + online adaptation 路径，本轮不执行。
- 延迟语义：上游 `JointPosition` 在 reset 为每个环境采样一次并在整个 episode 固定 FIFO delay，不是每个 action 重采样；默认代码 `[2,6]` physics ticks，即 10–30 ms。论文附录写 `U[40,120] ms`，与当前代码不一致。
- 本项目的 latency 定义覆盖 observation→action-ready/apply 的完整区间，已包含 actuation delay；因此内部配置固定为 `task.action.min_delay=0`、`task.action.max_delay=0`，不重复计时。动作平滑也固定关闭：`task.action.alpha=1.0`。后续 latency 由 benchmark scheduler 单独注入，本轮不接入 scheduler。
- 固定 seed：`0`。原生 teacher 复现阶段不做延迟扫描或其他任务；后续 VLA student 仍保持任务内部 zero-delay，视觉和状态契约见 Phase 2。

## 2. 官方 baseline、预算与性能结论

### 2.2 PPO 语义与多卡实现

官方一轮 rollout 是 `4096 × 32 = 131,072` transitions，切成 8 个 16,384-sample minibatch，并做 3 epochs。`num_minibatches` 是切分数，不是 minibatch size：从 8 减小会让 minibatch 更大且减少 optimizer steps；从 8 增至 128 会把官方 16,384 的 minibatch 改成 1,024，并把 optimizer step 数放大 16 倍，不属于“保持官方语义”的免费加速。

本次实现只覆盖正式需要的 `scripts/train.py` + `ppo_haic_train` / `ppo_haic_finetune`：

- 双卡每 rank 2048 env，global batch 仍为 4096 env；仍用 8 minibatches、3 epochs，固定 frames 下 iteration 和 optimizer-step 数不变。
- 同步初始参数、PPO/adapter/distillation 梯度、advantage moments、KL/LR、ValueNorm、训练 VecNorm 和 checkpoint 后的 EMA/buffer；不同 rank 使用 `seed + rank` 生成不同 rollout。
- W&B、checkpoint 写入只由 global rank 0 执行；两 rank 都参加最终 eval，避免 VecNorm collective 死锁。W&B run checkpoint 只由 rank 0 下载，再广播本地路径。
- 范围外：`train_sequential.py`、`train_est`、`adapt_est` 和 DR estimator 没有宣称支持 DDP；不要对这些入口使用 `torchrun`。

本机 Isaac Sim 5.1/G1 在 NCCL P2P/SHM transport 下可复现 illegal memory access；IsaacLab 也有相同版本、G1 和双卡组合的[公开 issue](https://github.com/isaac-sim/IsaacLab/issues/4011)。本机验证命令需要 `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`，因此 NCCL 走 socket；这是当前双卡收益有限的硬件/runtime 条件，不把它隐藏成算法问题。

### 2.3 实测 profile

本节全部数字都是 native PPO rollout/learner 性能，不包含 VLA camera、跨进程推理或 MP4 编码，不能作为 VLA collector 吞吐量。

统一 profile：PullCart、zero delay、seed 0、global 4096 env、32 horizon、8 minibatches、5,242,880 global frames；前 5 iterations 作为 warm-up，不含 1000-step final eval。

| 配置 | rollout FPS 中位数 | learner / iter | train e2e FPS | 500M / stage 估算 | 训练窗口 SM | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| 1 GPU × 4096 env | 33,850 | 0.431 s | 30,490 | 4.55 h | 52.4% | 22.16 GB |
| 2 GPU × 2048 env | 39,447 | 0.552 s | 33,807 | 4.11 h | 49.5% / 51.2% | 17.69 / 17.62 GB |

双卡只提升约 **1.11×** e2e training FPS，500M 每阶段节省约 26.8 分钟；teacher + student 估算从 9.11 小时降到 8.22 小时。原因很明确：单卡每 iteration 约 90% 时间在 PhysX/rollout，learner 只有约 10%；双卡使 rollout 快 14.2%，但 socket collective 令 learner 慢 28.1%。没有 2× free lunch。

双卡 40-iteration profile 后，90 个 parameters、2 个 buffers 和 9 个 VecNorm state tensors 在两 rank 间逐元素完全一致（max diff 0）。另一个双卡 finetune smoke 从 teacher checkpoint 加载，完成 2 次 PPO update、周期/最终 checkpoint 和两 rank 的 1000-step final eval，退出码 0。运行证据：

```text
/pfs/users/lxy/runs/haic/profile_single_4096
/pfs/users/lxy/runs/haic/profile_dual_2048
/pfs/users/lxy/runs/haic/ddp_smoke_v6
/pfs/users/lxy/runs/haic/ddp_finetune_load_smoke
```

正式双卡命令：

```bash
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  algo=ppo_haic_train task=G1/haic/pull_cart \
  task.num_envs=2048 algo.num_minibatches=8 \
  total_frames=500000000 seed=0 \
  task.action.min_delay=0 task.action.max_delay=0 task.action.alpha=1.0
```

`total_frames` 是 global frames；脚本内部按 world size 分配，不要再手工除 2。student 将 `algo` 改成 `ppo_haic_finetune` 并传 teacher `checkpoint_path`。

### 2.4 不改变 PPO 配置的 rollout 优化

正式配置仍是 4096 env、32 horizon、8 minibatches、3 epochs、5 ms physics dt 和 decimation 4。远端 RTX PRO 6000 上固定 seed 0，每个候选运行 30 iterations、丢弃前 5 个 warm-up；e2e FPS 按 rollout 中位数和 learner 中位时间计算：

| 远端 4096-env 配置 | rollout FPS | PhysX / step | reward / step | learner / iter | e2e FPS | 500M / stage |
|---|---:|---:|---:|---:|---:|---:|
| 上游原始，8 partitions | 27,391 | 98.52 ms | 12.08 ms | 0.634 s | 24,186 | 5.74 h |
| 跳过 disabled reward + 合并 metrics sync，8 partitions | 28,580 | 97.68 ms | 8.33 ms | 0.643 s | 25,064 | 5.54 h |
| 上述优化 + 4 partitions | **30,767** | **90.56 ms** | **7.90 ms** | **0.622 s** | **26,845** | **5.17 h** |

最终两项改动相对同机上游原始配置提升 rollout 12.3%、e2e 11.0%，teacher + student 的 500M cap 估算约 10.35 小时。保留的代码改动只有：

- `enabled=False` reward 不再执行 `compute()`；它们本来就不进入 PPO reward，状态 callback 仍照常推进，但对应 debug 指标不再记录；
- `stats_ema` 的多次 CUDA scalar `.item()` 合并为一次 device→host 传输；
- `gpu_max_num_partitions` 从 IsaacLab 默认 8 固定为实测更快的 4。它改变 GPU 调度与浮点执行顺序，不改变 solver、碰撞、dt、reward 或 PPO 更新口径。30-iteration 日志无 PhysX/CUDA/overflow/NaN，全部记录值 finite，24 个有效 reward 的中位分布与 8 partitions 对照接近。

没有保留的候选：

- 主 contact sensor 从 ankles+wrist 缩到 ankles 只有 +0.13% rollout，属于噪声；恢复原配置。
- G1 已使用 19 个 primitive collider，cart 使用 23 个 primitive collider；没有发现可无损删除的复杂 mesh。contact history、air-time、手-车 filtered sensor、self-collision 和 8/4 solver 均有正式消费者或会改变动力学，因此未改。
- 没有强开 `torch.compile`：rollout 热路径混合 Python 循环、TensorDict 写入和随机采样，当前没有稳定、低风险的 compile 边界。
- `replicate_physics=True` 必须同时删除 cart 的 `[0.9,1.1]` per-env 几何 scale DR；IsaacLab 2.3.2 没有等价的 runtime collider-scale tensor API。上限实验把首个训练 metric runtime 从 76.1 s 降到 29.8 s，稳态 e2e 只再提升 1.8%，但改变任务随机化语义，因此正式配置保持 `replicate_physics=False`。IsaacLab 对 replication 的契约见[官方配置说明](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/scene/interactive_scene_cfg.html)，碰撞优化约束见[官方性能指南](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/simulation_performance.html)。

远端运行证据：

```text
/home/xinyuan/haic_runs/profile_rollout_original_remote
/home/xinyuan/haic_runs/profile_rollout_opt_remote
/home/xinyuan/haic_runs/profile_contact_ankle_remote
/home/xinyuan/haic_runs/profile_partitions4_remote
/home/xinyuan/haic_runs/profile_replicate_upper_remote
```

## 3. 远端环境和固定命令

远端：`xinyuan@204.12.163.107`；计划使用一张 GPU。外部 checkout 和运行输出放在：

```text
/home/xinyuan/HAIC
/home/xinyuan/haic_runs
/home/xinyuan/latency-sensitive-bench-mikasa-h1
/home/xinyuan/latency-sensitive-bench-mikasa-h1/third_party/Isaac-GR00T/.venv
```

要求：Python 3.11、Isaac Sim 5.1.0、IsaacLab v2.3.2、Git LFS；Phase 1 原生 teacher 的 `HAIC` 外部 checkout 保持 fork commit `73976dbaf42494fa20d385599758eced7f7a16ce`，Phase 2 VLA 使用当前父仓库中的 HAIC submodule commit `f7a2bf5`，两者不可混用。

通用环境：

```bash
export HAIC_ROOT=$HOME/HAIC
export HAIC_PY=$HOME/miniconda3/envs/haic/bin/python
export PATH=$HOME/.local/bin:$HOME/miniconda3/bin:$PATH
export OMNI_KIT_ACCEPT_EULA=YES
export HEADLESS=1
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PYTHONUNBUFFERED=1
cd "$HAIC_ROOT"
git rev-parse HEAD
```

Vulkan 使用系统与内核同源的 NVIDIA 580.173.02 userspace，不再把早期不完整的
`$HOME/.local/nvidia-580.173.02` 放进 `LD_LIBRARY_PATH`。已安装 `libnvidia-gl-580=580.173.02-1ubuntu1`、
`libnvidia-common-580=580.173.02-1ubuntu1`、`libnvidia-egl-gbm1=1.1.3-1ubuntu1`、
`libnvidia-egl-wayland1=1:1.1.21-1ubuntu1`、`libnvidia-egl-xcb1=1.0.5-1ubuntu1`、
`libnvidia-egl-xlib1=1.0.5-1ubuntu1` 和 `libglu1-mesa=9.0.2-1.1build1`，并把账号加入
`render,video` 组。新 SSH 会话中 `/dev/dri/renderD128` 可读写；用户态
`$HOME/.local/vulkan-tools/usr/bin/vulkaninfo --summary` 退出 0，识别 RTX PRO 6000、Vulkan
1.4.312、driver 580.173.02。

所有长期运行使用远端 `tmux` 或 `nohup`；每次在本节和下表记录 PID/session、stdout/stderr、W&B run 和 checkpoint 路径。W&B 可用但不将 token 写入日志。

## 4. 阶段和验收

### Phase 0：预检与 runtime smoke

安装/导入 Isaac Sim、IsaacLab、HAIC，检查 PullCart 资源和用户态 headless GPU。运行 motion replay、reset 和单步命令：

```bash
timeout --signal=TERM --kill-after=20s 120s "$HAIC_PY" scripts/play.py \
  algo=ppo_haic_train task=G1/haic/pull_cart \
  task.command.replay_motion=true task.num_envs=1 \
  task.action.min_delay=0 task.action.max_delay=0 task.action.alpha=1.0 \
  seed=0 wandb.mode=disabled \
  hydra.run.dir=/home/xinyuan/haic_runs/phase0_motion_replay
```

通过条件：进程成功退出或按明确的运行时长安全停止；没有 CUDA architecture、USD/LFS、Isaac import、Vulkan 或 NaN 错误；reset/step 和 motion 资源可用。系统 GL/Vulkan 栈已通过 `vulkaninfo --summary` 和 runtime smoke；此前不完整 userspace 产生的 `ERROR_INCOMPATIBLE_DRIVER` 仅保留在历史失败日志，不作为通过结果。

### Phase 1：teacher + native student preparation

先做工程 smoke（不向 W&B 在线同步），使用与正式 run 相同的零延迟覆盖：

```bash
timeout --signal=TERM --kill-after=30s 7200s "$HAIC_PY" scripts/train.py \
  algo=ppo_haic_train task=G1/haic/pull_cart \
  task.num_envs=256 total_frames=5000000 seed=0 \
  task.action.min_delay=0 task.action.max_delay=0 task.action.alpha=1.0 \
  wandb.mode=offline \
  hydra.run.dir=/home/xinyuan/haic_runs/phase1_smoke_v2
```

在上述命令前设置：

```bash
export WANDB_NAME=haic-phase1-smoke-v2
export WANDB_DIR=/home/xinyuan/haic_runs/phase1_smoke_v2
```

上游 `scripts/train.py` 在 `wandb.mode=disabled` 下仍无条件把 `cfg` 保存到 W&B `run.dir/files/cfg.yaml`，而 disabled run 不创建该目录；首次严格命令因此在 Isaac 初始化后于 `OmegaConf.save` 失败。`wandb.mode=offline` 默认虽创建本地 run 目录，但上游随后依赖 `run.name`，默认 offline run 返回 `None`，在 `scripts/train.py:57` 失败。为保持不在线同步且完整走原生 trainer，smoke 额外显式设置 `WANDB_NAME=haic-phase1-smoke-v2` 和 `WANDB_DIR=/home/xinyuan/haic_runs/phase1_smoke_v2`；该组合已用 standalone probe 验证 `run.name` 非空、`run.dir/files` 存在。失败证据和本次修复后的结果见下表，不把失败尝试记作训练通过。smoke 只证明 rollout/update 持续运行、checkpoint 可写且 checkpoint 张量 finite，不作为最终 teacher。

正式 phase 1、finetune 与 eval 的精确命令如下；代码经用户 review/commit 后再执行。正式训练使用现有远端 `.env` 的 W&B 凭据，token 不写入日志：

```bash
set -a
source /home/xinyuan/latency-sensitive-bench-mikasa-h1/.env
set +a
export WANDB_NAME=haic-phase1-teacher
export WANDB_DIR=/home/xinyuan/haic_runs/phase1_teacher
timeout --signal=TERM --kill-after=60s 172800s "$HAIC_PY" scripts/train.py \
  algo=ppo_haic_train task=G1/haic/pull_cart \
  task.num_envs=4096 total_frames=500000000 seed=0 \
  task.action.min_delay=0 task.action.max_delay=0 task.action.alpha=1.0 \
  wandb.mode=online \
  hydra.run.dir=/home/xinyuan/haic_runs/phase1_teacher
```

训练完成后以原生 `scripts/eval.py` 验收 teacher；`latest-run` symlink 指向当前 W&B run，
`TEACHER_CKPT` 指向其 `files/checkpoint_final.pt`，stdout 与原生 YAML 都要保留：

```bash
export TEACHER_CKPT=/home/xinyuan/haic_runs/phase1_teacher/wandb/latest-run/files/checkpoint_final.pt
mkdir -p /home/xinyuan/haic_runs/phase1_eval_teacher
timeout --signal=TERM --kill-after=60s 3600s "$HAIC_PY" scripts/eval.py \
  algo=ppo_haic_train task=G1/haic/pull_cart task.num_envs=2048 seed=0 \
  checkpoint_path="$TEACHER_CKPT" \
  task.action.min_delay=0 task.action.max_delay=0 task.action.alpha=1.0 \
  hydra.run.dir=/home/xinyuan/haic_runs/phase1_eval_teacher \
  > /home/xinyuan/haic_runs/phase1_eval_teacher/stdout.log 2>&1
```

原生 `ppo_haic_finetune` 的历史启动条件是 teacher 的 `eval/success == 1.0`、finite metrics、记录的
`episode_cnt` 和完整 episode gate；本轮 native student 不在范围内，以下命令仅保留为历史参考，不执行：

```bash
export WANDB_NAME=haic-finetune
export WANDB_DIR=/home/xinyuan/haic_runs/finetune
export TEACHER_CKPT=/home/xinyuan/haic_runs/phase1_teacher/wandb/latest-run/files/checkpoint_final.pt
timeout --signal=TERM --kill-after=60s 172800s "$HAIC_PY" scripts/train.py \
  algo=ppo_haic_finetune task=G1/haic/pull_cart \
  task.num_envs=4096 total_frames=500000000 seed=0 \
  checkpoint_path="$TEACHER_CKPT" \
  task.action.min_delay=0 task.action.max_delay=0 task.action.alpha=1.0 \
  wandb.mode=online \
  hydra.run.dir=/home/xinyuan/haic_runs/finetune
```

student 使用相同的 `scripts/eval.py` 命令，改为 `algo=ppo_haic_finetune`、student
`checkpoint_final.pt` 和独立 `phase1_eval_student` 输出目录；命令为：

```bash
set -a
source /home/xinyuan/latency-sensitive-bench-mikasa-h1/.env
set +a
export STUDENT_CKPT=/home/xinyuan/haic_runs/finetune/wandb/latest-run/files/checkpoint_final.pt
mkdir -p /home/xinyuan/haic_runs/phase1_eval_student
timeout --signal=TERM --kill-after=60s 3600s "$HAIC_PY" scripts/eval.py \
  algo=ppo_haic_finetune task=G1/haic/pull_cart task.num_envs=2048 seed=0 \
  checkpoint_path="$STUDENT_CKPT" \
  task.action.min_delay=0 task.action.max_delay=0 task.action.alpha=1.0 \
  hydra.run.dir=/home/xinyuan/haic_runs/phase1_eval_student \
  > /home/xinyuan/haic_runs/phase1_eval_student/stdout.log 2>&1
```

teacher/student 的 YAML 由原生 `scripts/eval.py` 写入其 `scripts/eval/G1PullCart/`，stdout
分别保存在上述固定目录；native finetune 的历史验收要求 `eval/success == 1.0`、finite metrics、
`episode_cnt` 与完整 episode gate。论文表格指标仅作参考，不因零延迟设置差异强行逐项复刻。

若原生 finetune 路径继续执行而最终 checkpoint 未通过，逐个评估已有周期 checkpoint；按成功率优先、tracking error 次之选择。若没有 checkpoint 通过，停止并记录证据，不临时调参或修改 reward/任务定义。

### Phase 2：用 VLA 替代原生 student

控制链保持 HAIC 原有 teacher、归一化和 50 Hz actor 结构，只替换 privileged latent 的来源：

```text
256×256 RGB + 605D normalized command/policy state @ 10 Hz
  -> VLA 输出 256D privileged latent
  -> 固定 native actor_adapt: 605D state + 256D latent -> 23D action @ 50 Hz
```

每个 VLA 输出保持 5 个 HAIC control steps。runtime 从 `policy_config` 创建 LatencyBench
inference worker，因此可替换为任何遵守相同 observation/policy 接口并输出 256D latent 的 VLA；
当前提供并验收的训练 backend 是 official GR00T N1.7。换用其他训练 backend 时仍必须产生同一
256D 输出和可供 runtime 加载的 checkpoint，不能静默改变维度或字段含义。

原生 `eval/success == 1.0` gate 只控制是否启动原生 `ppo_haic_finetune`，不控制当前 VLA 路线；按当前
VLA 替换路线，phase-1 final teacher 是固定的 privileged label source。smoke 的验收目标是数据、权重和闭环执行
完整，不把 2-step 模型的成功率当作策略质量结论。

训练分两段：

- bootstrap 从 GR00T-N1.7-3B fresh base 开始。privileged teacher 的成功 episode 每个 10 Hz row 除 RGB、605D state 和兼容 VLA 输出的 256D latent 外，还保存 `actor_input[5,605]`、`teacher_action[5,23]` 和 `termination[5]`；正式配置为 250 个 train episode、25 个 val episode、global batch 64、gradient accumulation 2、peak LR `5e-5` 和 5,000 optimizer updates；
- bootstrap 与 DAgger 都只优化 23D motor-action MSE，不再优化 256D latent MSE。VLA latent 与每个 50 Hz actor input 送入同一个冻结 `actor_adapt`；terminal action 保留，terminal 后子步由累计 mask 排除。actor 不进入 optimizer、GR00T model 或 checkpoint；
- DAgger 流程是 `step-0 eval -> rollout -> training -> eval`。当前默认只执行 round 0：两个顺序 collector 各 32,000 rows，合计 64,000 rows、gradient accumulation 2 和 500 updates。采样权重固定为 50% action-rich bootstrap replay + 50% 累积 DAgger 数据，bootstrap prior 不随 round 数稀释；eval 始终使用 100 env / seed 0，每个 env 只统计从 motion frame 0 开始的首个 episode；
- DAgger collector 使用 inference batch 32、1,000-row shard；每个 round eval 后删除 raw shard，保留 LeRobot dataset、dataset manifest、training dataset manifest、checkpoint、eval 结果和 policy config。

预算和预计耗时（同一张 RTX PRO 6000 上按已有 Parkour profile 外推，正式 HAIC 运行前需做短 batch profile）如下：

| 阶段 | optimizer updates | microsteps | rows / exposure | 预计耗时 |
|---|---:|---:|---:|---:|
| SFT bootstrap | 5,000 | 10,000（GA2） | 640,000 | 3–4 h |
| DAgger（10 × 500） | 5,000 | 10,000（GA2） | 640,000 | 8–12 h |

修正前的 DAgger 配置虽然同样只记 5,000 optimizer updates，但 GA72 实际是
360,000 microsteps / 23.04M rows，加上 batch-8 rollout 需按约 10–14 天预留，不是小时级任务。

bootstrap 和 DAgger 都写原子 `NPZ + MP4` shard，再转换为 video-backed LeRobot Parquet。
正式采集前先用 100 个完整 episode 验证 `teacher latent -> frozen actor_adapt`，并将 native
checkpoint 的 `actor_adapt` 导出成纯 PyTorch state dict。训练期 action loss 加载该文件，collector
和 eval 则从同一固定 teacher checkpoint 构造相同 actor。

DAgger training、step 0 与所有 round eval 使用同一个固定 W&B run，记录
`train/global_step=0,500,...,5000` 以及 `dagger_eval/episodes`、`dagger_eval/successes`、`dagger_eval/success_rate`，不记录固定执行上限 `dagger_eval/steps`；bootstrap SFT 使用独立的 `<dagger-run-id>_sft` run，避免两段训练都从
global step 0 起步而污染同一条曲线。
launcher 将这两个名字显式传给 GR00T 的 `--experiment-name`，因此默认 W&B 名称分别是
`haic_pull_cart_vla_dagger_action_5k_sft` 和 `haic_pull_cart_vla_dagger_action_5k`，不再从输出目录继承成
`training`。对应 checkpoint 分别位于 `bootstrap/training/<dagger-run-id>_sft/` 和
`training/<dagger-run-id>/`。
不设置自动分数门禁，分数只作对比和记录。

远端 `.env` 必须定义有效的 `HF_TOKEN` 和 FFmpeg runtime prefix：

```bash
HF_TOKEN=...
HAIC_VLA_FFMPEG_PREFIX=/home/xinyuan/.cache/latency-sensitive-bench/ffmpeg7
```

launcher 会 source `.env`，并把该 prefix 的 `bin`/`lib` 加入 `PATH`/`LD_LIBRARY_PATH`。
正式入口如下；必须在 code review 和用户确认后执行：

```bash
bash scripts/bash_scripts/train_haic_vla_5k.sh \
  --run-bootstrap \
  --teacher-checkpoint /home/xinyuan/haic_runs/phase1_teacher/wandb/latest-run/files/checkpoint_final.pt \
  --base-model /home/xinyuan/latency-sensitive-bench-mikasa-h1/outputs/official_gr00t/models/GR00T-N1.7-3B \
  --output-root /home/xinyuan/haic_runs/haic_pull_cart_vla_action_dagger_round0 \
  --haic-python /home/xinyuan/miniconda3/envs/haic/bin/python
```

fresh run 显式传 `--run-bootstrap`，它会先执行 frozen-actor oracle eval，再重新采集并训练 action-only bootstrap；旧 latent checkpoint 不得作为初始化。续跑使用
`--bootstrap-checkpoint`、`--bootstrap-dataset`、`--actor-checkpoint` 和 `--start-round N`。`--end-round` 是右开上界，
默认 1，即只执行 round 0。训练子进程成功返回后才原子写入
round-complete marker；用同一 `--start-round N` 重试未完成轮次时，会重新采集该轮数据。
已完成并归档的 round 只删除 raw，保留 `lerobot/` 供后续 DAgger round 聚合训练；旧 current-only DAgger
输出缺少可复用的 round `lerobot/` 和聚合训练 manifest，不能作为该流程的 resume 根目录。

## 5. 实际运行记录

| 阶段 | 状态 | PID/session | 日志 / checkpoint / 结果 |
|---|---|---|---|
| Phase 1 4096 env / 500M teacher cap | 已完成，训练退出码 0 | tmux `haic_teacher`，outer PID `4003533`，已自然退出 | `/home/xinyuan/haic_runs/phase1_teacher/stdout.log`；W&B online run `m4s51yy5` / `https://wandb.ai/saberrr-zju/haic/runs/m4s51yy5`；4096 env、499,908,608 实际 global frames、3814 iterations、seed 0、zero delay、`alpha=1.0`；W&B runtime 18,116 s（5:01:56），12 个周期 checkpoint 和 `checkpoint_final.pt` 均已保存；trainer 内置 final eval `success=0.9296875`、`episode_cnt=4626`、metrics finite |
| native `ppo_haic_finetune` | 未启动（本轮范围外） | — | teacher final 和全部周期 checkpoint 均未达到 native gate；因此不启动原生 student，Phase 2 VLA 仍使用该 teacher 作为 label source |
| teacher/student eval gate | teacher 未达到 native gate；VLA label source 已接受；student 未执行 | 所有 eval 已自然退出，退出码均为 0 | 原生 2048-env `scripts/eval.py`：final `success=0.93896484375`、`episode_cnt=2308`；`checkpoint_300` 至 `checkpoint_3600` 的 success 依次为 `0.48193359375, 0.73291015625, 0.82373046875, 0.818359375, 0.8525390625, 0.8701171875, 0.9140625, 0.91162109375, 0.9306640625, 0.92529296875, 0.9345703125, 0.93310546875`；13 次 eval 各 96 个数值 metrics 全部 finite，但没有 `success == 1.0`；汇总 `/home/xinyuan/haic_runs/phase1_eval_teacher/periodic_summary.tsv`，逐 checkpoint stdout 位于同目录 `periodic/checkpoint_*/stdout.log` |
| 旧 VLA bootstrap + DAgger smoke | 已完成但作废 | 所有进程自然退出 | 275D/Actor 联合路径的旧 smoke；与当前 256D output + action-only schema 不兼容，正式训练必须 fresh bootstrap |
| 旧 Phase 2 latent-only bootstrap SFT | 已完成但作废 | 首次 launcher outer PID `5894`，已退出 | 250 train / 25 val 成功 episode，13,605 / 522 frames；`checkpoint-5000` 已保存；5,000 updates 用时 7,638 s，汇总 `train_loss=0.0113649884`。该 loss 只证明 latent fitting，不能作为 action-only 初始化；W&B run `haic_pull_cart_vla_dagger_latent_5k_sft` 仅保留作历史证据。 |
| Phase 2 DAgger 首次尝试 | round 0 prepare 失败，退出码 1 | 首次 launcher 已退出 | step-0 eval 已写入主 W&B run；两个 collector 各完成 32,000 rows，但 dataset prepare 子进程没有继承仓库 `PYTHONPATH`，于 2026-09-02 21:54:51 UTC 报 `ModuleNotFoundError: latency_bench`，因此尚未产生 DAgger `train/loss` |
| Phase 2 DAgger current-only 尝试 | 不作为正式 DAgger 结果；已停止并清理 DAgger 残留 | tmux `haic_vla_5k` 旧会话已停止；outer PID `241017` 已停止 | 2026-09-03 01:29:38 UTC 曾从现有 bootstrap `checkpoint-5000`、dataset 和 `--start-round 0` 恢复；旧 driver 只训练当前 round，缺少 bootstrap/历史 round 聚合，因此不能作为正式聚合 DAgger resume 根目录；2026-09-03 已删除旧 `round_*`、`training` 和 top-level DAgger stdout/exit_code，保留 `bootstrap/lerobot` 与 `bootstrap/training/checkpoint-5000` |
| Phase 2 DAgger 聚合修复、旧 eval 口径 | 已停止，结果作废 | tmux `haic_vla_5k` 已停止；driver PID `268530` 已退出 | 2026-09-03 03:34:56 UTC 从保留 bootstrap checkpoint/dataset 和 `--start-round 0 --end-round 10` 重启；W&B run `haic_pull_cart_vla_dagger_latent_5k_aggfix_20260903` 完成 round 0–3 并保存 `checkpoint-2000`，于 round 4 collector rank 1 期间按用户要求停止；该 run 的 eval 在固定 1,000 steps 内重复统计 auto-reset episode，导致 episode 分母不固定，不作为正式 DAgger 结果；旧输出保留在 `/home/xinyuan/haic_runs/haic_pull_cart_vla_dagger` |
| Phase 2 latent-only 100-full-episode run | 已停止，输出保留，结果作废 | tmux `haic_vla_5k_eval100` 已停止 | 输出 `/home/xinyuan/haic_runs/haic_pull_cart_vla_dagger_eval100`；step 0、round 0、round 1、round 2 均为 `episodes=100`、`successes=0`、`success_rate=0.0`；原 run 在 round 3 期间停止。它仍监督 latent 且使用会逐轮稀释 bootstrap 的聚合比例，不能 resume 到 action-only 流程。 |
