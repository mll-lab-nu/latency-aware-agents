# HAIC / PullCart zero-delay 原生复现手册

状态：本轮 256-env/5M 工程 smoke 已运行至稳定点并通过有限值审计（未跑满 5M）；正式 teacher/student 训练与 eval 按用户决定未启动。本文件是唯一执行记录和事实来源。未写入的结果表示尚未完成，不能视为通过。

## 1. 固定口径

- 上游仓库：`https://github.com/ldt29/HAIC`，checkout `e262500ee0e68e9d82f399aa18a4c10bf5b4ca1a`；源码和资产只保留在远端外部 checkout，不进入本仓库、submodule 或发布物。仓库级未发现 LICENSE。
- 任务：`G1/haic/pull_cart`，PullCart motion `data/motion/g1/mirobotA/pull_cart/motion.npz`。
- 原生时间：physics 200 Hz / 5 ms，policy 50 Hz / 20 ms，decimation 4；motion 561 帧 / 11.22 s；动作是 23 个非 wrist 关节。
- 训练路线：`ppo_haic_train` 同一在线 rollout 同时训练 privileged teacher、privileged target world model/adapter，并将 teacher 均值蒸馏到 student；不是离线 BC。随后用通过 phase-1 checkpoint 的 `ppo_haic_finetune` 做 student on-policy PPO + online adaptation。
- 延迟语义：上游 `JointPosition` 在 reset 为每个环境采样一次并在整个 episode 固定 FIFO delay，不是每个 action 重采样；默认代码 `[2,6]` physics ticks，即 10–30 ms。论文附录写 `U[40,120] ms`，与当前代码不一致。
- 本项目的 latency 定义覆盖 observation→action-ready/apply 的完整区间，已包含 actuation delay；因此内部配置固定为 `task.action.min_delay=0`、`task.action.max_delay=0`，不重复计时。动作平滑也固定关闭：`task.action.alpha=1.0`。后续 latency 由 benchmark scheduler 单独注入，本轮不接入 scheduler。
- 固定 seed：`0`。本轮不做 RGB/VLA、离线 demonstrations、dataset 转换、延迟扫描或其他任务。

## 2. 远端环境和固定命令

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

## 3. 阶段和验收

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

下一轮正式 phase 1、finetune 与 eval 的精确命令如下；本轮按用户决定不执行。正式训练使用现有远端 `.env` 的 W&B 凭据，token 不写入日志：

```bash
set -a
source /home/xinyuan/latency-sensitive-bench-mikasa-h1/.env
set +a
export WANDB_NAME=haic-phase1-teacher
export WANDB_DIR=/home/xinyuan/haic_runs/phase1_teacher
timeout --signal=TERM --kill-after=60s 172800s "$HAIC_PY" scripts/train.py \
  algo=ppo_haic_train task=G1/haic/pull_cart \
  task.num_envs=4096 total_frames=800000000 seed=0 \
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
  task.num_envs=4096 total_frames=800000000 seed=0 \
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

## 4. 实际运行记录

| 阶段 | 状态 | PID/session | 日志 / checkpoint / 结果 |
|---|---|---|---|
| 本地 worktree | 已确认干净，基于 `8470a50b7680f64ee1527017d7177bc2a7246ad3` | — | 分支 `research/haic-pull-cart-zero`，本文件 |
| 远端硬件预检 | 已通过 | — | hostname `computeinstance-u00mv1dams6a8gc51r`；1× RTX PRO 6000 Blackwell 97887 MiB；driver `580.173.02`；约 325 GB 可用 |
| HAIC checkout | 已通过 | — | `/home/xinyuan/HAIC`，commit `e262500ee0e68e9d82f399aa18a4c10bf5b4ca1a`；`git-lfs/3.6.1`，8 个 LFS object，`git lfs fsck` OK；PullCart `motion.npz` 1,040,842 bytes；外部 checkout 有未跟踪 `asset_meta.json`，不纳入本仓库 |
| 安装 | 已通过 | — | `/home/xinyuan/haic_setup.log` 记录初次 IsaacLab flat-layout 安装失败；随后 IsaacLab v2.3.2 commit `37ddf626871758333d6ed89cf64ad702aef127d0` 用其安装脚本/`source/isaaclab` editable 安装成功；HAIC editable 成功；`isaacsim 5.1.0.0`、`isaaclab 0.54.2`、`active-adaptation 0.0.0`、`torch 2.7.0+cu128`、`torchrl 0.7.0` 导入通过 |
| Phase 0 runtime smoke | 已通过 | 根侧 outer PID `3962889/3962891`；关闭拖到 kill-after | `/home/xinyuan/haic_runs/phase0_runtime_smoke_fixed/stdout.log`；系统 GL/Vulkan 栈识别 GPU 并输出 `RESET_OK`、`STEP_OK` finite、`ACTION_OK [1, 23]`，无 `ERROR_INCOMPATIBLE_DRIVER`、GLU 或 CUDA/USD/LFS/NaN failure |
| Phase 1 256 env / 5M | 工程 smoke 已通过（未跑满 5M；用户确认稳定后主动停止） | v1 outer PID `3964294` 已精确停止；v2 outer PID `3965590` 已精确停止于约 `308/610` | v1 `/home/xinyuan/haic_logs/phase1_smoke_offline_named.log` 约 `20/610` 后停止，timeout 估算不足、无 checkpoint；v2 `/home/xinyuan/haic_logs/phase1_smoke_v2.log` 约 4.5 s/iteration，`308/610` 后用户确认稳定并停止；W&B offline run `df3dkvt8` / `v2-G1PullCart-ppo_haic-2026-09-01-08-41`；resolved `/home/xinyuan/haic_runs/phase1_smoke_v2/.hydra/config.yaml` 与 overrides；`checkpoint_300.pt` 18,231,330 bytes，TOP_KEYS=`cfg,env,policy,vecnorm,wandb`，110 floating tensors，`NONFINITE=0`；配置核对为 G1PullCart/256/5M/seed0/0/0/1.0；无 `checkpoint_final.pt`，未执行 trainer final eval 或正式 eval gate |
| Phase 1 4096 env / 800M teacher | 已准备但本轮按用户决定未启动 | — | 下一轮命令见上；不存在 teacher checkpoint 或 eval 结果 |
| native `ppo_haic_finetune` | 已准备但本轮按用户决定未启动 | — | 必须先通过 teacher eval gate；下一轮命令见上 |
| teacher/student eval gate | 已准备但本轮按用户决定未启动 | — | 原生 `scripts/eval.py`；不存在 `eval/success`、`episode_cnt` 或 student 结果 |

## 5. 明确 deferred

- fixed/realtime latency sweep、benchmark scheduler 接入和 latency-aware teacher 对比；
- RGB/VLA、offline dataset、视觉 student 和真实机器人部署；
- PushCart、其他 HAIC task、论文与当前代码 delay discrepancy 的额外消融。
