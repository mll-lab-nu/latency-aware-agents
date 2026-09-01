# HAIC / PullCart zero-delay 原生复现手册

状态：HAIC 已作为 fork submodule 接入；单卡/双卡 profile、双卡 teacher smoke、checkpoint 保存、finetune 加载与最终 eval 链路均已通过。正式 teacher/student 长训尚未启动。本文件是唯一执行记录和事实来源；未写入的结果不能视为通过。

## 1. 固定口径

- 上游仓库：`https://github.com/ldt29/HAIC`，fork submodule：`git@github.com:mindorigin150/HAIC.git`，当前基线 `e262500ee0e68e9d82f399aa18a4c10bf5b4ca1a`。分布式改动先在 submodule 内提交，父仓库随后再提交新的 gitlink；上游仓库级未发现 LICENSE。
- 任务：`G1/haic/pull_cart`，PullCart motion `data/motion/g1/mirobotA/pull_cart/motion.npz`。
- 原生时间：physics 200 Hz / 5 ms，policy 50 Hz / 20 ms，decimation 4；motion 561 帧 / 11.22 s；动作是 23 个非 wrist 关节。
- 训练路线：`ppo_haic_train` 同一在线 rollout 同时训练 privileged teacher、privileged target world model/adapter，并将 teacher 均值蒸馏到 student；不是离线 BC。随后用通过 phase-1 checkpoint 的 `ppo_haic_finetune` 做 student on-policy PPO + online adaptation。
- 延迟语义：上游 `JointPosition` 在 reset 为每个环境采样一次并在整个 episode 固定 FIFO delay，不是每个 action 重采样；默认代码 `[2,6]` physics ticks，即 10–30 ms。论文附录写 `U[40,120] ms`，与当前代码不一致。
- 本项目的 latency 定义覆盖 observation→action-ready/apply 的完整区间，已包含 actuation delay；因此内部配置固定为 `task.action.min_delay=0`、`task.action.max_delay=0`，不重复计时。动作平滑也固定关闭：`task.action.alpha=1.0`。后续 latency 由 benchmark scheduler 单独注入，本轮不接入 scheduler。
- 固定 seed：`0`。本轮不做 RGB/VLA、离线 demonstrations、dataset 转换、延迟扫描或其他任务。

## 2. 官方 baseline、预算与性能结论

### 2.1 官方到底是不是 800M

- 官方 release 的 [`cfg/train.yaml`](https://github.com/ldt29/HAIC/blob/e262500ee0e68e9d82f399aa18a4c10bf5b4ca1a/cfg/train.yaml#L13) 默认 `total_frames=800_000_000`，README 的 teacher/student 命令都没有覆盖该值，所以“直接跑官方命令”的代码口径确实是每次 800M frames。
- 论文没有声明“每阶段必须 800M”。论文只报告两阶段训练在单张 RTX 4090 上合计约 8 小时，并在 Table XII 给出 4096 environments、32-step horizon、8 minibatches、3 epochs；见[论文实现细节](https://arxiv.org/pdf/2602.11758#page=6)与[附录超参数](https://arxiv.org/pdf/2602.11758#page=17)。
- 论文 Fig. 8 的 3,800 `training steps` 是 **Push Cart w/ Box 的方法消融曲线**，不是本任务 PullCart 的公开停止条件，也不是 3,800 environment steps。
- 官方没有发布 PullCart checkpoint、逐 checkpoint 成功率、wall-clock 日志或“800M 才收敛”的证据。因此 800M 是 release config 的上限，不应当当作不可提前停止的科学结论。
- 官方 baseline 是同配置从头重训的 HDMI：`HDMI(sim)` 使用 privileged object state 作为 sim-to-sim 上界，`HDMI*` 是只用 proprioception history 的 real-world 下界；不是另一套更短的 HAIC 训练预算。

当前执行策略：teacher 和 finetune 各以 500M 为 cap，保留周期 checkpoint；优先按完整 episode 的 `eval/success == 1.0` 选择最早通过的 checkpoint。500M 不是“官方新 baseline”，而是用验收 gate 换 wall-clock 的工程上限。若较早 checkpoint 已通过就停止；若全部失败，不伪称复现成功。

### 2.2 PPO 语义与多卡实现

官方一轮 rollout 是 `4096 × 32 = 131,072` transitions，切成 8 个 16,384-sample minibatch，并做 3 epochs。`num_minibatches` 是切分数，不是 minibatch size：从 8 减小会让 minibatch 更大且减少 optimizer steps；从 8 增至 128 会把官方 16,384 的 minibatch 改成 1,024，并把 optimizer step 数放大 16 倍，不属于“保持官方语义”的免费加速。

本次实现只覆盖正式需要的 `scripts/train.py` + `ppo_haic_train` / `ppo_haic_finetune`：

- 双卡每 rank 2048 env，global batch 仍为 4096 env；仍用 8 minibatches、3 epochs，固定 frames 下 iteration 和 optimizer-step 数不变。
- 同步初始参数、PPO/adapter/distillation 梯度、advantage moments、KL/LR、ValueNorm、训练 VecNorm 和 checkpoint 后的 EMA/buffer；不同 rank 使用 `seed + rank` 生成不同 rollout。
- W&B、checkpoint 写入只由 global rank 0 执行；两 rank 都参加最终 eval，避免 VecNorm collective 死锁。W&B run checkpoint 只由 rank 0 下载，再广播本地路径。
- 范围外：`train_sequential.py`、`train_est`、`adapt_est` 和 DR estimator 没有宣称支持 DDP；不要对这些入口使用 `torchrun`。

本机 Isaac Sim 5.1/G1 在 NCCL P2P/SHM transport 下可复现 illegal memory access；IsaacLab 也有相同版本、G1 和双卡组合的[公开 issue](https://github.com/isaac-sim/IsaacLab/issues/4011)。本机验证命令需要 `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`，因此 NCCL 走 socket；这是当前双卡收益有限的硬件/runtime 条件，不把它隐藏成算法问题。

### 2.3 实测 profile

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
```

要求：Python 3.11、Isaac Sim 5.1.0、IsaacLab v2.3.2、Git LFS；`HAIC` 外部 checkout 必须保持上述 commit。

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

只有 teacher 的 `eval/success == 1.0`、finite metrics、记录的 `episode_cnt` 和完整 episode
gate 都通过，才启动同预算原生 student finetune：

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
分别保存在上述固定目录；验收均要求 `eval/success == 1.0`、finite metrics、`episode_cnt`
与完整 episode gate。论文表格指标仅作参考，不因零延迟设置差异强行逐项复刻。

若最终 checkpoint 未通过，逐个评估已有周期 checkpoint；按成功率优先、tracking error 次之选择。若没有 checkpoint 通过，停止并记录证据，不临时调参或修改 reward/任务定义。

## 5. 实际运行记录

| 阶段 | 状态 | PID/session | 日志 / checkpoint / 结果 |
|---|---|---|---|
| 本地 worktree | submodule 改动待用户 review/commit | — | 分支 `research/haic-pull-cart-zero`；HAIC 基线 `e262500ee0e68e9d82f399aa18a4c10bf5b4ca1a` |
| 远端硬件预检 | 已通过 | — | hostname `computeinstance-u00mv1dams6a8gc51r`；1× RTX PRO 6000 Blackwell 97887 MiB；driver `580.173.02`；约 325 GB 可用 |
| HAIC checkout | 已通过 | — | `/home/xinyuan/HAIC`，commit `e262500ee0e68e9d82f399aa18a4c10bf5b4ca1a`；`git-lfs/3.6.1`，8 个 LFS object，`git lfs fsck` OK；PullCart `motion.npz` 1,040,842 bytes；外部 checkout 有未跟踪 `asset_meta.json`，不纳入本仓库 |
| 安装 | 已通过 | — | `/home/xinyuan/haic_setup.log` 记录初次 IsaacLab flat-layout 安装失败；随后 IsaacLab v2.3.2 commit `37ddf626871758333d6ed89cf64ad702aef127d0` 用其安装脚本/`source/isaaclab` editable 安装成功；HAIC editable 成功；`isaacsim 5.1.0.0`、`isaaclab 0.54.2`、`active-adaptation 0.0.0`、`torch 2.7.0+cu128`、`torchrl 0.7.0` 导入通过 |
| Phase 0 runtime smoke | 已通过 | 根侧 outer PID `3962889/3962891`；关闭拖到 kill-after | `/home/xinyuan/haic_runs/phase0_runtime_smoke_fixed/stdout.log`；系统 GL/Vulkan 栈识别 GPU 并输出 `RESET_OK`、`STEP_OK` finite、`ACTION_OK [1, 23]`，无 `ERROR_INCOMPATIBLE_DRIVER`、GLU 或 CUDA/USD/LFS/NaN failure |
| Phase 1 256 env / 5M | 工程 smoke 已通过（未跑满 5M；用户确认稳定后主动停止） | v1 outer PID `3964294` 已精确停止；v2 outer PID `3965590` 已精确停止于约 `308/610` | v1 `/home/xinyuan/haic_logs/phase1_smoke_offline_named.log` 约 `20/610` 后停止，timeout 估算不足、无 checkpoint；v2 `/home/xinyuan/haic_logs/phase1_smoke_v2.log` 约 4.5 s/iteration，`308/610` 后用户确认稳定并停止；W&B offline run `df3dkvt8` / `v2-G1PullCart-ppo_haic-2026-09-01-08-41`；resolved `/home/xinyuan/haic_runs/phase1_smoke_v2/.hydra/config.yaml` 与 overrides；`checkpoint_300.pt` 18,231,330 bytes，TOP_KEYS=`cfg,env,policy,vecnorm,wandb`，110 floating tensors，`NONFINITE=0`；配置核对为 G1PullCart/256/5M/seed0/0/0/1.0；无 `checkpoint_final.pt`，未执行 trainer final eval 或正式 eval gate |
| 远端 rollout 优化 profile | 已通过 | 已停止，GPU 空闲 | 同机原始/优化/partitions/contact/replication A/B 路径见 2.4；最终保留 e2e +11.0%，无 nonfinite 或 PhysX/CUDA error |
| Phase 1 4096 env / 500M teacher cap | 已准备但未启动 | — | 命令见上；不存在 teacher checkpoint 或 eval 结果 |
| native `ppo_haic_finetune` | 已准备但未启动 | — | 必须先通过 teacher eval gate；命令见上 |
| teacher/student eval gate | 已准备但未启动 | — | 原生 `scripts/eval.py`；不存在 `eval/success`、`episode_cnt` 或 student 结果 |

## 6. 明确 deferred

- fixed/realtime latency sweep、benchmark scheduler 接入和 latency-aware teacher 对比；
- RGB/VLA、offline dataset、视觉 student 和真实机器人部署；
- PushCart、其他 HAIC task、论文与当前代码 delay discrepancy 的额外消融。
