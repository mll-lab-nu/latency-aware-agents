# Extreme Parkour Isaac Lab / Go2 zero-delay baseline 执行手册

> 任务：`Isaac-Extreme-Parkour-{Teacher,Student}-Unitree-Go2-v0`
>
> Extreme Parkour fork：当前 commit `6caa34adf73a0a1acc366bfee81320fc1e202618`；
> 性能优化 base `4265e663627a3f5e50a076508b78cfdad05b2b39`，后续恢复 student height
> scanner 的 20 ms `update_period`、启用 cuDNN benchmark，并将 teacher PhysX partitions 设为 4；
> zero-delay commit `991c4b02aa08a802e57d01924ef2f3826d0d7334`；upstream base
> `d6766d8883cffe427f2d4c087717d4eca1f23d8c`
>
> Isaac Lab：v2.2.0，commit `46dff135f44683f031edf346e544fcfd8456b2bb`
>
> Isaac Sim：5.0.0；PyTorch：2.7.0 + cu128
>
> 状态：依赖、双卡 smoke、目标容量 A/B 和精确 resume 已完成。首个 adaptive-LR teacher
> 正式 run 已确认 collapse，并在 iteration 1418 停止；fixed-LR teacher 已通过 iteration
> 100/300/1000 门禁并正常完成 15k；Teacher evaluation 已通过
> 34D terrain-latent+yaw VLA 的 RGB collector、LeRobot/GR00T 数据转换、5k 训练入口和
> zero-delay evaluator 已实现；单卡 8-env oracle、1+1 episode 数据、1-step GR00T 训练及
> 闭环推理 smoke 已通过，尚未运行正式 250/25、5k 和 50-env VLA gate。

## 1. 目标与实验口径

原生 teacher/student baseline 为：

```text
zero-delay privileged teacher PPO
  -> zero-delay 10 Hz depth student distillation
  -> upstream fixed evaluation
```

这是 **Extreme Parkour 的 Isaac Lab / Unitree Go2 adaptation**，不是 Robot Parkour
Learning 的五技能复现，也不是官方 Isaac Gym / Unitree A1 数值复现。社区实现保留了统一
privileged teacher、depth student 和 `parkour` / `hurdle` / `step` / `gap` 等 terrain，
但机器人、simulator、terrain 网格、相机和 evaluator 都已改变。

以下设置固定，不在执行中静默替换：

- robot：Unitree Go2。
- simulator：Isaac Sim 5.0.0 + Isaac Lab v2.2.0。
- control：50 Hz；physics `dt=0.005`，decimation 4，每个 control step 20 ms。
- depth：10 Hz encoder cadence，每 5 个 control steps 编码一次当次最新帧。
- action delay：关闭；本 control step 产生的 action 在本 control step 施加。
- depth delay：关闭旧帧选择；使用 `depth_buffer[:, -1]`，不使用 `-2`。
- teacher：两卡同步 PPO，总计 6144 envs，每卡 3072。
- student：两卡同步 depth distillation，总计 192 camera envs，每卡 96。
- iteration：按作者建议区间锁定 teacher 15,000、student 5,000。
- training seed：base seed 1；DDP rank 0/1 的 environment seed 分别是 1/2。
- teacher PPO：fixed learning rate `2e-4`；不启用 KL-adaptive learning-rate schedule。
- evaluation：沿用社区 `evaluation.py`，不另造 evaluator 或 A1 成功率口径。
- latency integration：本轮不接 `latency_bench` wrapper；后续只从本仓库 scheduler/executor
  注入 latency。

### 1.1 作者官方 A1 参考配置

论文正文未列出完整训练超参；以下复现预算来自作者官方仓库的
[`README`](https://github.com/chengxuxin/extreme-parkour/blob/master/README.md#L27-L40) 和
[`legged_robot_config.py`](https://github.com/chengxuxin/extreme-parkour/blob/master/legged_gym/legged_gym/envs/base/legged_robot_config.py#L368-L407)。
它对应 Isaac Gym / Unitree A1，只作为当前 Isaac Lab / Go2 adaptation 的训练预算参照。

| phase | envs | control steps/env/iteration | batch/update | 作者训练区间 | 推荐预算 | 作者 RTX 3090 时间 |
|---|---:|---:|---|---:|---:|---:|
| teacher/base | 6144 | 24 | 147,456 transitions；PPO 5 epochs × 4 minibatches（36,864/minibatch） | 10k--15k iterations | 至少 15k；2.21184B transitions | 8--10 h |
| student/vision | 192 camera envs | 120；每 5 control steps 更新 depth | 23,040 transitions；一次 full-batch distillation update | 5k--10k iterations | 至少 5k；115.2M transitions | 5--10 h |

官方配置的 `max_iterations=50000` 是默认最大更新次数，不是 README 推荐的复现预算。
按优化后本机短 profile，`15k teacher + 5k student` 的纯训练 ETA 约 14.3 h
（teacher 9.74 h + student 4.60 h）；正式排期预留 16 h。

## 2. 上游 delay 到底做了什么

### 2.1 官方 Extreme Parkour action delay

官方 Isaac Gym 环境维护长度为 8 的 `action_history_buf`。每次 policy step 先把新 action
追加到 buffer；启用 delay 后使用：

```python
indices = -delay - 1
actions = action_history_buf[:, indices]
```

因此：

- `delay=0`：`buffer[-1]`，使用本拍 action；
- `delay=1`：`buffer[-2]`，使用上一拍 action，即 20 ms action delay。

不能只靠“不传 `--delay`”得到严格 zero-delay。官方 helper 对 headless、非 resume、非 camera
的从零 base training 会自动设置：

```text
action_delay = true
action_curr_step_scratch = [0, 1]
delay_update_global_steps = 24 * 8000
```

也就是前 8000 PPO iterations 使用本拍 action，之后切到固定上一拍 action。显式传
`--delay` 时，配置 `[1, 1]` 从开始就使用上一拍 action。

### 2.2 社区 Isaac Lab action delay

社区将同一逻辑迁移为 `DelayedJointPositionAction`：

```text
history_length = 8
action_delay_steps = [1, 1]
delay_update_global_steps = 24 * 8000
```

teacher 配置显式覆盖为 `use_delay=False`；student 配置原本显式设置
`use_delay=True`，所以 student 从开始到结束固定施加上一拍 action，即 20 ms delay。

该 FIFO 不是本仓库 latency model：

- delay 只能按离散 control step 表示；
- 没有 observation/action 的统一 timestamp；
- 不支持本仓库的 fixed/profile/distribution/temporal latency source；
- policy observation 中的 previous action 是最新发出的 command，而 simulator 可能实际施加
  buffer 中的旧 command。

因此 baseline 的 teacher 和 student 都固定 `use_delay=False`，以后也不能同时开启上游 FIFO
和本仓库 `ActionScheduler`。

### 2.3 depth frame staleness

官方和社区 depth 路径都包含另一处不叫 `delay`、但会产生旧 observation 的逻辑：

```text
control rate = 50 Hz
depth update/encoder interval = 5 control steps = 10 Hz
depth buffer length = 2
network input = depth_buffer[:, -2]
```

每 100 ms 把新图加入 buffer，却把上一张 10 Hz 图送进网络。因此在每次 depth inference
时显式使用约一帧旧图。两个更新之间继续复用同一个 depth latent。

本 baseline 保留 **10 Hz sampling/inference cadence**，因为它是模型计算频率；只把网络输入
改为 `depth_buffer[:, -1]`，即每次 inference 使用当次最新可用帧。这里的 zero-delay 不等于
把相机或网络改成 50 Hz。

### 2.4 名字里有 delay、但不属于 latency 的状态

以下内容保持不动：

- `reach_goal_delay=0.1`：到达 waypoint 后的 goal 切换 debounce，不是通信或执行 delay；
- proprio/action history：recurrent policy 的输入状态，不是 delay queue；
- physics decimation：一个 action 持续 4 个 physics substeps，是 50 Hz 控制定义；
- domain randomization、camera pose noise、push 和 friction：鲁棒性训练，不是 temporal latency；
- 10 Hz depth cadence：传感/推理频率，和额外 frame delivery latency 分开记录。

### 2.5 checkpoint 口径

社区公开 teacher/student checkpoints 只用于安装和模型加载 smoke。当前代码默认 student 带
20 ms action delay 且使用 `depth_buffer[:, -2]`，公开 artifact 没有独立 provenance 证明它是
zero-delay 训练，因此不能作为本轮正式 baseline。正式结果只使用本手册从零训练的 checkpoint。

新 checkpoint 支持 teacher/student 精确 resume；旧 checkpoint 仍可用于 inference 和
teacher-to-student 初始化，但不能精确 resume。

## 3. 本机硬件与双卡语义

当前节点实测：

```text
GPU 0: 97,887 MiB, compute capability 12.0
GPU 1: 97,887 MiB, compute capability 12.0
driver: 580.159.04
topology: PHB; peer read/write supported; peer atomics unsupported
RAM: 1.8 TiB
```

使用 `torch.distributed.run` 启动两个 Isaac Sim 进程。社区 `--num_envs` 是 **每 rank** 数量，
不能把单卡默认值直接用于双卡，否则 teacher/student 总环境数会翻倍。正式命令固定：

```text
teacher: 3072 env/rank * 2 = 6144 total
student:   96 env/rank * 2 =  192 total
```

社区原始 DDP 只同步主 PPO policy。fork 的 `fix/extreme-parkour-ddp` 分支在 commit
`e3fe6e3d3b62adbaf355f337732ea3b125e6d37d` 补齐 teacher/student 辅助网络同步和精确
checkpoint resume，并包含当前 runtime 必需的兼容修复。这些是 correctness infra，不包含
zero-delay 实验设置，也不声明相对原始错误 DDP 的吞吐加速。

`research/zero-delay-go2` 在其上增加 zero-delay commit
`991c4b02aa08a802e57d01924ef2f3826d0d7334` 和 `.gitignore` commit `44dc7c862c920153e77de5cf739b3e73b2f88a2b`，
包含关闭 student action delay、读取最新 depth frame、关闭 headless depth debug window，以及
两个既有 Python/config bug 修复。性能优化已提交为 `4265e663627a3f5e50a076508b78cfdad05b2b39`
并由父仓 `ff70f83fcaac4534bffb4bd415d97972d442f949` 固定。上游 PR compare
入口是 [master...mindorigin150:fix/extreme-parkour-ddp](https://github.com/CAI23sbP/Isaaclab_Parkour/compare/master...mindorigin150:Isaaclab_Parkour:fix/extreme-parkour-ddp?expand=1)。

## 4. Worktree 与 submodule

固定 worktree：

```bash
export PARKOUR_ROOT=/pfs/users/lxy/latency-sensitive-bench
cd "${PARKOUR_ROOT}"

git submodule sync --recursive
git submodule update --init --recursive
git -C third_party/IsaacLab rev-parse HEAD
git -C third_party/Isaaclab_Parkour rev-parse HEAD
```

期望输出分别为：

```text
46dff135f44683f031edf346e544fcfd8456b2bb
6caa34adf73a0a1acc366bfee81320fc1e202618
```

Parkour submodule URL 固定为 `git@github.com:mindorigin150/Isaaclab_Parkour.git`。fresh checkout
不再执行 `git apply`；commit 不匹配时直接检查 submodule URL/gitlink，不静默改用其他 revision。
当前实验 workspace 的父仓是 `master@37641dc8cc6faae6f8a65979b7d9e6676f1784ea`，Parkour
submodule 是 `master@0fc5411818697fbc8c8f40c363119a5cd0fbd07b`。本轮 fixed-LR 对照只允许
Teacher runner config 中的 `schedule` 一行实验修改，不混入其他训练语义变化。

## 5. 环境安装

环境独立位于 `/pfs/users/lxy/conda/envs/extreme-parkour`：

```bash
export PARKOUR_ROOT=/pfs/users/lxy/latency-sensitive-bench
export PARKOUR_ENV=/pfs/users/lxy/conda/envs/extreme-parkour
export PARKOUR_PY="${PARKOUR_ENV}/bin/python"

conda create -p "${PARKOUR_ENV}" python=3.11 -y
"${PARKOUR_PY}" -m pip install --upgrade pip
"${PARKOUR_PY}" -m pip install setuptools==80.9.0
"${PARKOUR_PY}" -m pip install --no-build-isolation flatdict==4.0.1
"${PARKOUR_PY}" -m pip install \
  torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
"${PARKOUR_PY}" -m pip install 'isaacsim[all,extscache]==5.0.0' \
  --extra-index-url https://pypi.nvidia.com

TERM=xterm OMNI_KIT_ACCEPT_EULA=YES conda run -p "${PARKOUR_ENV}" \
  bash -lc "cd '${PARKOUR_ROOT}/third_party/IsaacLab' && ./isaaclab.sh -i rsl_rl"
"${PARKOUR_PY}" -m pip install --no-build-isolation --no-deps -e \
  "${PARKOUR_ROOT}/third_party/Isaaclab_Parkour"
"${PARKOUR_PY}" -m pip install --no-build-isolation -e \
  "${PARKOUR_ROOT}/third_party/Isaaclab_Parkour/parkour_tasks"
conda install -p "${PARKOUR_ENV}" libglu=9.0.3 -y
"${PARKOUR_PY}" -m pip install \
  click==8.1.7 psutil==5.9.8 typing_extensions==4.12.2
```

Isaac Sim 首次启动需要接受 NVIDIA EULA，并可能下载 extension cache。非交互运行前：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

不要用系统 Python 3.14，也不要让 `uv` 重新解析为其他 Torch/Isaac Sim 版本。

## 6. Preflight 和 smoke

### 6.1 Python/GPU

```bash
export PARKOUR_ROOT=/pfs/users/lxy/latency-sensitive-bench
export PARKOUR_PY=/pfs/users/lxy/conda/envs/extreme-parkour/bin/python
export OMNI_KIT_ACCEPT_EULA=YES

"${PARKOUR_PY}" - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
PY

"${PARKOUR_PY}" -m compileall -q \
  "${PARKOUR_ROOT}/third_party/Isaaclab_Parkour/parkour_isaaclab" \
  "${PARKOUR_ROOT}/third_party/Isaaclab_Parkour/parkour_tasks" \
  "${PARKOUR_ROOT}/third_party/Isaaclab_Parkour/scripts/rsl_rl/modules"
```

### 6.2 单卡训练 smoke

从 Parkour submodule 根目录各跑 1 iteration：

```bash
cd "${PARKOUR_ROOT}/third_party/Isaaclab_Parkour"

OMNI_KIT_ACCEPT_EULA=YES "${PARKOUR_PY}" scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 \
  --num_envs 8 --max_iterations 1 --seed 1 \
  --run_name smoke_single --headless

LD_PRELOAD="${PARKOUR_ENV}/lib/libGLU.so.1" \
OMNI_KIT_ACCEPT_EULA=YES "${PARKOUR_PY}" scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 \
  --num_envs 1 --max_iterations 1 --seed 1 --enable_cameras \
  --load_run 2026-09-01_16-27-46_smoke_single --checkpoint model_0.pt \
  --run_name smoke_single_student --headless
```

### 6.3 双卡同步 smoke

当前 torch cu128 wheel 自带 NCCL 2.26.2，在本机 CUDA 13/Blackwell 上会在两 rank communicator
初始化时挂起。预加载系统 NCCL 2.28.3 后，独立 `all_reduce` 和两类训练均通过。因此本节点上的
所有 DDP 命令都必须显式使用 `/lib/x86_64-linux-gnu/libnccl.so.2`。

teacher 跑 1 iteration：

```bash
cd "${PARKOUR_ROOT}/third_party/Isaaclab_Parkour"

LD_PRELOAD=/lib/x86_64-linux-gnu/libnccl.so.2 NCCL_DEBUG=WARN \
OMNI_KIT_ACCEPT_EULA=YES "${PARKOUR_PY}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 \
  --distributed --headless --seed 1 \
  --num_envs 8 --max_iterations 1 \
  --run_name smoke_ddp_teacher
```

student 显式加载这个 teacher checkpoint，也跑 1 iteration：

```bash
LD_PRELOAD="${PARKOUR_ENV}/lib/libGLU.so.1:/lib/x86_64-linux-gnu/libnccl.so.2" \
NCCL_DEBUG=WARN OMNI_KIT_ACCEPT_EULA=YES \
"${PARKOUR_PY}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 \
  --distributed --headless --seed 1 --enable_cameras \
  --num_envs 1 --max_iterations 1 \
  --load_run 2026-09-01_16-35-39_smoke_ddp_teacher --checkpoint model_0.pt \
  --run_name smoke_ddp_student
```

smoke 只验证两卡 environment、NCCL、camera、broadcast/all-reduce 和 checkpoint；不能当成正式模型。

### 6.4 目标容量短 profile

正式训练前各跑 5 iterations，丢弃 iteration 0，以 iterations 1--4 的中位数记录稳态。
A/B 保持相同 DDP 布局、总 env 数、rollout horizon、优化器、PPO epochs/minibatches、physics/control
和 10 Hz depth 语义：

| phase | env/rank | baseline run / FPS | optimized run / FPS | 提升 | optimized collection | learning | 锁定预算 ETA |
|---|---:|---|---|---:|---:|---:|---:|
| teacher | 3072 | `2026-09-01_17-31-16_profile_teacher_ddp_3072` / 54,056 | `2026-09-01_19-35-52_profile_teacher_strict_opt_final_ddp_3072` / 63,390 | 17.3% | 2.130 s | 0.207 s | 9.74 h / 15k |
| student | 96 | `2026-09-01_17-32-59_profile_student_ddp_96` / 6,192 | `2026-09-01_19-36-56_profile_student_strict_opt_final_ddp_96` / 6,958.5 | 12.4% | 3.204 s | 0.107 s | 4.60 h / 5k |

保留的改动只删除等价热路工作：GPU terrain mask 取代每步 NumPy/CPU 往返，metrics
保持在 GPU 并在 reset 时只做一次 host sync，zero-delay action 绕过无用 FIFO，静态 privileged
latent 缓存，history 就地滑动，depth 批处理，camera update 与已有 10 Hz 消费节拍对齐，
并在同一节拍复用固定 teacher 的 scan latent。height scanner 保持 20 ms；纯张量新旧路径
逐元素等价检查已通过。

### 6.5 第二轮 free-lunch 筛选

筛选门槛固定为双卡端到端稳态提升至少 1%；低于门槛的实验代码全部回退：

| candidate | 语义检查 | 目标容量结果 | 结论 |
|---|---|---|---|
| height scanner `update_period` 20 ms → 100 ms | 当前 `lazy_sensor_update=True`，只有 observation 每 5 步读取时才 raycast | 单卡 collection 2.076 s → 2.117 s | 恢复 20 ms；没有省掉 80% raycast |
| goal/env 张量清理、actor concat、teacher scan cache、日志同步合并 | strict-equivalence probes 逐值通过 | teacher DDP 2.057 s → 2.061 s（-0.18%） | 未过门槛，全部回退 |
| 最小 student clone/host-sync 清理 | 日志值和顺序逐值一致 | 两组 DDP 端到端 FPS 6971 → 6922、6945 → 6912（-0.70%、-0.48%） | 未过门槛，回退 |
| `nn.Module.compile` actor | 非 exact；最大绝对误差 `1.16e-4`，默认 `allclose` 失败 | actor 284.4 us → 257.6 us；actor-only +9.4%，预计端到端不足 1% | 不进入训练路径 |
| `torch.backends.cudnn.benchmark=True` | forward exact；backward 最大绝对误差 `7.96e-13`，默认 `allclose` 通过 | student DDP 端到端 +1.26%～+1.64% | 启用 |

height scanner 恢复 20 ms；cuDNN benchmark 已启用，接受 backward 约 `8e-13` 的最大绝对差异。

20-iteration DDP 证据 runs 分别为 teacher `2026-09-01_20-32-25_strict_ddp_baseline_teacher_3072`
/ `2026-09-01_20-33-53_strict_ddp_candidate_teacher_3072`，student baseline
`2026-09-01_20-35-44_strict_ddp_baseline_student_96`、`2026-09-01_20-43-26_strict_ddp_baseline2_student_96`，
minimal candidate `2026-09-01_20-41-03_strict_ddp_minimal_student_96`、
`2026-09-01_20-45-23_strict_ddp_minimal2_student_96`，以及 cuDNN
`2026-09-01_20-48-58_cudnn_benchmark_student_96`。

### 6.6 checkpoint resume smoke

Teacher/Student 均已用优化 profile 的 `model_4.pt` 完成双卡精确 resume smoke，
分别由 runs `2026-09-01_19-38-22_smoke_resume_teacher_strict_opt` 和
`2026-09-01_19-39-08_smoke_resume_student_strict_opt` 从 iteration 4 继续训练。

## 7. 正式双卡训练

### 7.1 Teacher

Teacher PPO 配置固定为 `learning_rate=2e-4`、`schedule="fixed"`。下面的 15k run 同时是
fixed-LR 候选和通过门禁后的正式 run，不再额外重复一次相同训练：

```bash
cd "${PARKOUR_ROOT}/third_party/Isaaclab_Parkour"

LD_PRELOAD=/lib/x86_64-linux-gnu/libnccl.so.2 NCCL_DEBUG=WARN \
WANDB_MODE=online OMNI_KIT_ACCEPT_EULA=YES \
"${PARKOUR_PY}" -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=2 --master_port=29542 \
  scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 \
  --distributed --headless --seed 1 \
  --num_envs 3072 --max_iterations 15000 \
  --run_name teacher_zero_delay_fixed_lr_ddp_seed1_15k \
  --logger wandb --log_project_name isaaclab-parkour
```

门禁使用 W&B 未舍入指标：

- iteration 100：不能连续 20 iterations 同时出现 mean reward 接近零、mean episode length < 20；
- iteration 300：最后 50 iterations 的 median reward > 0.05、median episode length > 100；
- iteration 1000：上述状态保持，无 NaN/Inf，`Loss/learning_rate` 始终为 `2e-4`。

任一门禁失败即停止，不自动改 reward、curriculum、Dagger、PhysX partitions 或其他参数。通过
iteration 1000 后，按最近 100 iterations 的中位耗时估算剩余 ETA，并进入事件驱动等待：正常
训练睡到预计完成时间；进程退出、Traceback、NCCL/CUDA/NaN 或再次出现持续零奖励 collapse 时
提前唤醒。完成后记录 rank 0 输出的确切 run directory 和最终 checkpoint，不使用目录 glob
或 `latest`。

### 7.2 Student

```bash
export TEACHER_RUN='2026-09-02_00-49-22_teacher_zero_delay_fixed_lr_ddp_seed1_15k'
export TEACHER_CKPT='model_14999.pt'

LD_PRELOAD="${PARKOUR_ENV}/lib/libGLU.so.1:/lib/x86_64-linux-gnu/libnccl.so.2" \
NCCL_DEBUG=WARN WANDB_MODE=online OMNI_KIT_ACCEPT_EULA=YES \
"${PARKOUR_PY}" -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=2 --master_port=29543 \
  scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 \
  --distributed --headless --seed 1 --enable_cameras \
  --num_envs 96 --max_iterations 5000 \
  --load_run "${TEACHER_RUN}" --checkpoint "${TEACHER_CKPT}" \
  --run_name student_zero_delay_ddp_seed1_5k \
  --logger wandb --log_project_name isaaclab-parkour
```

student runner 因 distillation 类型会加载 teacher model；这里不传 `--resume`，避免把 teacher
optimizer 状态解释成 student resume。

iteration 100 门禁已通过：total loss 从 iteration 0 的 `4.9011` 降至 `2.9893`，depth actor
loss `2.1700`、yaw loss `0.8193`，吞吐 `5559 steps/s`，无 NaN/Inf 或进程错误。最近 100
iterations 中位耗时为 `4.220s`，预计约 `2026-09-02 08:03 UTC` 完成；之后使用与 Teacher
相同的事件驱动等待，正常结束或新出现 Traceback、NCCL/CUDA/NaN、worker 退出时唤醒。

首次 Student 进程在 iteration 2286 后于 `2026-09-02 04:52 UTC` 被外部硬终止：最后指标
total loss `1.0828`，没有 traceback、OOM、NCCL/CUDA 错误或 wrapper exit marker；系统和
GPU 资源均充足，tmux server 与其他 session 正常，仅 Student session 消失。最后完整保存点
是 `model_2200.pt`，其 Student 网络、depth encoder/actor、optimizer 和 iteration 均完整。
续跑使用 `--resume --load_run 2026-09-02_10-10-50_student_zero_delay_ddp_seed1_5k
--checkpoint model_2200.pt --max_iterations 2800`；runner 从 iteration 2200 计到 4999，并创建
run `2026-09-02_13-02-16_student_zero_delay_ddp_seed1_5k_resume2200`。恢复后的 iteration
2200 total loss `0.9842`，iteration 2220 为 `1.1273`；最近 20 iterations 中位耗时
`4.045s`，新 ETA 为 `2026-09-02 08:11 UTC`。

## 8. 固定评测

正式评测继续使用上游 `evaluation.py` 的 256 envs、1500 steps，以及四项输出：

```text
mean reward
mean episode length
normalized waypoint progress
edge violation
```

teacher 与 student 可分别占用 GPU 0/1 并行执行；正式模型均使用 training seed 1，每个
checkpoint 只运行一次 `evaluation.py`，不做额外 evaluation seed sweep。当前 evaluator 没有
`--seed` 参数；单次运行会启动 256 个并行 env、执行 1500 steps，并汇总期间完成的多个
episode。不要用 A1 论文图表或 Robot Parkour 五技能数值作为 Go2 gate。

student evaluation 同样需要 `--enable_cameras`，并在进程环境中预加载
`${PARKOUR_ENV}/lib/libGLU.so.1`；所有 checkpoint 都写出精确绝对路径，不使用 `latest`。

本轮通过条件：

- teacher 15k / student 5k runs 正常结束；
- rank 0 checkpoints 可由对应 play/evaluation 入口显式加载；
- reward、episode length、waypoint progress、edge violation 全部 finite；
- waypoint progress 非零；
- action runtime probe 证明同拍施加；
- depth runtime probe 证明每 5 步读取当次 buffer 最新帧。

## 9. 实际运行记录

| 项目 | 状态/结果 |
|---|---|
| worktree | `/pfs/users/lxy/latency-sensitive-bench`；`master@37641dc8` |
| GPU | 2× 97,887 MiB Blackwell，compute capability 12.0，driver 580.159.04；P2P read/write OK |
| Isaac Lab / Parkour | `46dff135...` pinned；Parkour `0fc54118...`，optimization base `4265e66...` |
| Python/Torch/Isaac Sim | Python 3.11；Torch 2.7.0+cu128；Isaac Sim 5.0.0；安装完成 |
| fork branches | `fix/extreme-parkour-ddp`=`e3fe6e3d...`；`research/zero-delay-go2`=`44dc7c8...`；`master`=`0fc54118...`；均已发布 |
| upstream PR | compare/create form 已就绪；GitHub App 对 upstream 只有 read 权限，API create PR 返回 403 |
| 单卡 teacher/student smoke | 通过；runs `2026-09-01_16-27-46_smoke_single` / `2026-09-01_16-42-32_smoke_single_student` |
| teacher DDP smoke | 通过；run `2026-09-01_16-35-39_smoke_ddp_teacher`；384 timesteps |
| student DDP smoke | 通过；run `2026-09-01_16-43-57_smoke_ddp_student`；48 timesteps |
| optimized teacher/student DDP smoke | 通过；runs `2026-09-01_19-27-20_smoke_strict_opt_teacher_ddp` / `2026-09-01_19-28-07_smoke_strict_opt_student_ddp` |
| teacher 3072 env/rank capacity smoke | 通过；run `2026-09-01_16-47-41_capacity_ddp_teacher_3072`；1 iteration、147,456 timesteps、42,506 FPS；`model_0.pt` 可加载 |
| teacher 3072 env/rank profile | baseline 54,056 FPS；optimized run `2026-09-01_19-35-52_profile_teacher_strict_opt_final_ddp_3072` 63,390 FPS，+17.3% |
| student 96 env/rank profile | baseline 6,192 FPS；optimized run `2026-09-01_19-36-56_profile_student_strict_opt_final_ddp_96` 6,958.5 FPS，+12.4% |
| teacher/student exact resume | 通过；优化 profile 的 `model_4.pt` 均从 iteration 4 继续训练 1 iteration |
| camera teardown | current keyboard unsubscribe API 单卡实跑通过，无退出 `AttributeError` |
| 优化代码 pin | base `4265e66...`；scanner/cuDNN/partitions follow-up 位于 Parkour `0fc54118...` |
| 第二轮 strict free lunch | teacher/student 双卡均未稳定超过 1% 端到端门槛；实验代码已回退 |
| allclose candidates | compile 因默认 `allclose` 失败而拒绝；cuDNN benchmark +1.26%～+1.64%，已启用，backward 最大绝对差异 `7.96e-13` |
| teacher adaptive 15k | **failed/collapsed**；run `2026-09-01_23-53-57_teacher_zero_delay_ddp_seed1_15k`；[W&B `vjsucm5x`](https://wandb.ai/saberrr-zju/isaaclab-parkour/runs/vjsucm5x)；console `logs/teacher_zero_delay_ddp_seed1_15k.console.log`；LR iteration 5 达 `1e-2`，value loss iteration 12 达 `9.6582`，reward iteration 24 峰值 `0.2307` 后下降并从约 iteration 80 持续为零；iteration 1418 人工停止，episode length `11.09` |
| teacher fixed 15k | **completed；exit 0**；run `2026-09-02_00-49-22_teacher_zero_delay_fixed_lr_ddp_seed1_15k`；checkpoint `model_14999.pt`；[W&B `8lpfqa6a`](https://wandb.ai/saberrr-zju/isaaclab-parkour/runs/8lpfqa6a)；console `logs/teacher_zero_delay_fixed_lr_ddp_seed1_15k.console.log`；保存配置已确认 `schedule=fixed`、LR `2e-4`；gates 100/300/1000 均通过；iteration 14999 reward `19.59`、episode length `710.00`、terrain level `6.0593`、value loss `0.0049`；训练于 `2026-09-02 02:07 UTC` 正常结束，未复现 collapse |
| student 5k | **running；resumed from model 2200**；original run `2026-09-02_10-10-50_student_zero_delay_ddp_seed1_5k` / [W&B `bhzgjid1`](https://wandb.ai/saberrr-zju/isaaclab-parkour/runs/bhzgjid1) / console `logs/student_zero_delay_ddp_seed1_5k.console.log`；iteration 2286 后被外部硬终止，loss 正常、无程序错误；resume run `2026-09-02_13-02-16_student_zero_delay_ddp_seed1_5k_resume2200` / [W&B `g3i1d4dh`](https://wandb.ai/saberrr-zju/isaaclab-parkour/runs/g3i1d4dh) / console `logs/student_zero_delay_ddp_seed1_5k_resume2200.console.log`；精确恢复 `model_2200.pt` 的网络和 optimizer，iteration 2220 total loss `1.1273`；最近 20 iterations 中位 `4.045s`，预计 `2026-09-02 08:11 UTC` 完成 |
| teacher fixed evaluation | **passed**；checkpoint `model_14999.pt`；console `logs/eval_teacher_fixed_lr_seed1_model_14999.console.log`；reward `23.14±6.02`、episode length `907.95±296.15`、normalized waypoint progress `1.00±0.01`、edge violation `0.19±0.42` |
| student fixed evaluation | pending |

## 10. 本轮明确 deferred

- 通用 `latency_bench` EnvAdapter 与 scheduler/executor 接入；Parkour VLA 已直接复用
  `PolicyRunner` 和 `ProcessInferencePool` 作为模型边界；
- fixed/profile/distribution/temporal latency sweep；
- 上游 action FIFO、旧 depth frame 或 latency-aware retraining；
- Robot Parkour Learning 五技能、官方 A1 Isaac Gym 复现；
- sim2sim、sim2real 和真机部署；
- 更长稳态 profile、Nsight kernel trace 和 Student camera pipeline 进一步深挖；
- `empirical_normalization=True` 的 teacher-to-student 双 normalizer 数据流；当前两个正式配置均关闭；
- 为 future backend、robot 或第三张 GPU 增加抽象。

## 11. 34D perception VLA zero-delay bootstrap

这一步是进入 DAgger 前的额外 bootstrap，不是原 paper 的训练阶段。它先冻结
`model_14999.pt` 的 teacher actor，将官方 GR00T-N1.7-3B 放在原 depth encoder 的位置：

```text
robot RGB 256x150 @ 10 Hz + current proprio 53D + fixed prompt
  -> swappable PolicyRunner
  -> terrain latent 32D + normalized current/next yaw 2D, H1
  -> fixed teacher actor @ 50 Hz
  -> raw Go2 joint action 12D
```

固定 prompt 为 `Traverse the parkour course and follow each waypoint.`。34D label 的前 32D
来自同一 teacher checkpoint 的 `actor.infer_scandots_latent(obs)`，后 2D 是 teacher actor
使用的 current/next yaw 除以 `1.5`。proprio 的 yaw 两维在送入 VLA 前置零，避免 oracle
waypoint 从 state 旁路进入模型。

VLA RGB 数据契约固定为：相机输出 `256x150`，采样率 `10 Hz`，水平/垂直视场角约为
`87°x58°`，保持原 depth 相机的光学视野比例。相机的 `clipping_range` 为 `0.1--100 m`；
RGB 保留真实远景，不用原 depth raycaster 的 `2 m` 最大距离模拟 RGB。Isaac Sim 5.0 /
Isaac Lab 2.2 没有 camera scene partition，因此采集器采用两遍 tiled render：第一遍同时
取得 RGB 和 `instance_segmentation_fast`，第二遍只在 Fabric 中隐藏 17 个共享 Go2 mesh
prototype；最终用第一遍中本 env 的 robot surface pixel 覆盖无机器人背景。输出保留自身
可见机体表面，但移除自身及其他 env 机器人的阴影、反射和几何，不改 robot root state，
也不引入自定义 CUDA kernel。强制重叠的 2-env smoke 验证了邻居消失、物理状态逐位不变且
visibility 恢复；50-env 相机路径中位耗时由约 `23 ms` 增至 `53 ms`（`2.32x`）。实际
collector smoke 已完成 1 个 8-env episode，共写出 85 张 `256x150` RGB frame 和 85 行
manifest；正式 250/25 数据仍须在本修复后重新采集。

### 11.1 Canonical teacher 与 oracle gate

新任务 `Isaac-Extreme-Parkour-VLA-Unitree-Go2-v0` 只为采集/评测增加 robot-mounted tiled
RGB camera；原 teacher/student 训练配置不变。先在相同 seed 2、50 个初始 env 上分别运行
50 Hz privileged teacher 与 10 Hz privileged-latent oracle：

```bash
export REPO_ROOT=/pfs/users/lxy/latency-sensitive-bench
export PARKOUR_ROOT="${REPO_ROOT}/third_party/Isaaclab_Parkour"
export TEACHER_CKPT="${PARKOUR_ROOT}/logs/rsl_rl/unitree_go2_parkour/2026-09-02_00-49-22_teacher_zero_delay_fixed_lr_ddp_seed1_15k/model_14999.pt"

cd "${PARKOUR_ROOT}"
PYTHONPATH="${REPO_ROOT}" OMNI_KIT_ACCEPT_EULA=YES \
  "${PARKOUR_PY}" scripts/rsl_rl/parkour_vla.py teacher-eval \
  --headless --seed 2 --num_envs 50 --checkpoint "${TEACHER_CKPT}" \
  --output_dir "${REPO_ROOT}/outputs/official_gr00t/extreme_parkour_go2/eval"

PYTHONPATH="${REPO_ROOT}" OMNI_KIT_ACCEPT_EULA=YES \
  "${PARKOUR_PY}" scripts/rsl_rl/parkour_vla.py oracle-eval \
  --headless --seed 2 --num_envs 50 --checkpoint "${TEACHER_CKPT}" \
  --output_dir "${REPO_ROOT}/outputs/official_gr00t/extreme_parkour_go2/eval"
```

oracle 每次更新 privileged 32D latent 后严格持有 5 个 control steps。若其 mean normalized
waypoint progress 低于同场景 teacher 的 50%，停止后续采集；不自动改成 direct 12D BC。

### 11.2 250/25 数据与 LeRobot 转换

collector 继续运行 oracle episode，仅提交触发 `cur_goal_idx >= num_goals` success termination
的轨迹，不把终点前摔倒或 timeout 当成功。它写 10 Hz PNG 与原子 JSONL episode shards，按
split seed 0 得到精确 250 train / 25 validation episodes：

修复前生成的 `outputs/official_gr00t/extreme_parkour_go2/raw` 和
`outputs/official_gr00t/extreme_parkour_go2/lerobot` 均视为污染数据，必须整套重建，不能
继续转换、混用或用于训练。collector 要求 `--output_dir` 是空的、不存在的目录；完成相机
smoke 后再执行以下采集和转换命令。训练进程与 checkpoint 不属于本节数据修复范围。

```bash
cd "${PARKOUR_ROOT}"
PYTHONPATH="${REPO_ROOT}" OMNI_KIT_ACCEPT_EULA=YES \
  "${PARKOUR_PY}" scripts/rsl_rl/parkour_vla.py collect \
  --headless --seed 1 --num_envs 50 --checkpoint "${TEACHER_CKPT}" \
  --train_episodes 250 --val_episodes 25 --split_seed 0 \
  --output_dir "${REPO_ROOT}/outputs/official_gr00t/extreme_parkour_go2/raw"

cd "${REPO_ROOT}"
uv run --project . python scripts/official_gr00t/prepare_parkour_dataset.py \
  outputs/official_gr00t/extreme_parkour_go2/raw \
  outputs/official_gr00t/extreme_parkour_go2/lerobot
```

finalizer 流式生成 canonical parquet、train/validation LeRobot videos、53D state/34D action
modality metadata，并对两套 dataset 运行 `gr00t.data.stats`。缺字段、错误 action layout 或错误
model output dimension 直接失败，不使用别名、零 state 或其他 fallback。

### 11.3 GR00T 5k 与最终闭环 gate

正式训练固定 fresh `GR00T-N1.7-3B`、2 GPUs、global batch 128、LR `1e-4`、H1、5k steps、
one-step regression、vision tuning、BF16、action/state dropout 0：

```bash
cd "${REPO_ROOT}"
bash scripts/bash_scripts/train_go2_parkour_vla_5k.sh
```

最终 evaluator 在 Isaac GPU 与独立 GR00T Python worker 之间复用现有 inference pool；50 个
环境按 batch 8 同步预测，全部 34D perception 输出返回前仿真不前进，因此该阶段定义为
zero-delay。actor 连续 5 个 control steps 使用同一 latent/yaw，并将预测 yaw 乘回 `1.5`：

```bash
cd "${PARKOUR_ROOT}"
PYTHONPATH="${REPO_ROOT}" OMNI_KIT_ACCEPT_EULA=YES \
  "${PARKOUR_PY}" scripts/rsl_rl/parkour_vla.py vla-eval \
  --headless --seed 2 --num_envs 50 --checkpoint "${TEACHER_CKPT}" \
  --policy_config "${REPO_ROOT}/configs/examples/extreme_parkour/official_gr00t_eval.yaml" \
  --inference_device cuda:1 --inference_batch_size 8 \
  --output_dir "${REPO_ROOT}/outputs/official_gr00t/extreme_parkour_go2/eval"
```

输出仍为 reward、episode length、normalized waypoint progress、edge violation。四项必须
finite，且 GR00T mean normalized waypoint progress 至少达到同一 50-env teacher 的 50%。
在该 gate 通过前，不启动正式 DAgger run、latency scheduler、profile、raw-action track 或
HAIC VLA。此前记录的 clean SFT checkpoint-5000 progress `0.6850±0.3393` 来自旧 evaluator：
每 5 个 control steps 只有第 1 步使用 VLA 预测 yaw，后 4 步错误复用了 observation 中的
privileged yaw，因此该结果无效。commit `658cd53` 修复为 5 步持续使用预测 yaw 后，同一
checkpoint 在 seed 2、50 env、batch 32 下重测 progress 为 `0.2325±0.2165`，未通过 SFT
gate；结果见 `outputs/official_gr00t/extreme_parkour_go2_dagger/round_00/eval_before/vla-eval.json`。

## 12. 固定 teacher actor 的 latent-only DAgger

DAgger 只更新已经通过 SFT gate 的 VLA，固定同一个 teacher actor 用于 rollout 和评测：

```text
RGB + masked proprio -> trainable VLA -> latent32 + normalized yaw2
                                      -> fixed teacher actor -> joint action12 (rollout only)
                                      -> fixed joint-position/PD controller -> Go2
```

SFT collection、DAgger collection 和 evaluation 全部使用
`Isaac-Extreme-Parkour-VLA-Unitree-Go2-v0` 这个 canonical env 1。DAgger 唯一改变的是由当前
VLA 控制机器人，并把并行环境数从 50 覆盖为 192；不切换到原 depth-student env 2。

Actor 与 history encoder 的唯一实现位于 Isaaclab_Parkour 的
`parkour_isaaclab/actor.py`，该模块只依赖 PyTorch；Isaac collector/evaluator 使用它，GR00T
训练不导入或更新它，不再由 latency-sensitive-bench 复制网络。保存后的 GR00T checkpoint
是标准 34D VLA checkpoint，输出走通用 processor 反归一化；部署不需要 teacher checkpoint、
Isaaclab_Parkour 或 latency-sensitive-bench 的 Actor 代码。

训练只使用标准 34D SFT latent/yaw loss：32D terrain latent 加 2D normalized current/next
yaw；不添加 12D action loss，也不训练或保存 student actor。固定 normalization stats 与
bootstrap 数据一致。

### 12.1 有界 DAgger 预算

原配置每个 iteration 是 `192 envs × 120 control steps = 23,040` 个 50 Hz transitions；depth
每 5 步更新一次，所以等价为 `4,608` 个 10 Hz perception samples。5,000 iterations 总计：

| quantity | 每轮 | 10 轮总计 |
|---|---:|---:|
| 10 Hz VLA rows | 64,000 | 640,000 |
| 50 Hz rollout control steps | 320,000 | 3,200,000 |
| optimizer updates | 500 | 5,000 |

GR00T 使用 global batch `128`、gradient accumulation `1`，因此每轮执行
`500 × 128 = 64,000` 个 supervised samples。每轮新增 64,000 行 current-VLA rollout，每行携带一个
oracle 34D latent/yaw target；训练数据使用 bootstrap LeRobot 加所有已完成 DAgger round 和当前
round 的 LeRobot 聚合数据。collector 和 GR00T shard 均为 1,000 rows，每轮 64 shards；模型、
optimizer 和 scheduler 从上轮 checkpoint 继续，teacher actor 始终从固定 teacher checkpoint
加载。

### 12.2 数据与 round driver

DAgger collector 使用一个 canonical env 1 进程、192 camera envs 和 seed 1；与 SFT/eval
相比只改变 rollout policy 和并行环境数。每个原子
shard 是 `episode.npz + episode.mp4`，不落逐帧 PNG；numeric payload 保存
`state[53]`、`action[34]` 和 `termination[5]`。每行保存一个 oracle 34D latent/yaw target，
包括 failure/timeout 标记处；
`termination` 只写入 parquet 供诊断，不进入标准 34D loss、不屏蔽 loss，也不重置 5-step
cadence。LeRobot parquet 直接保留这些诊断列，
并逐字节复制由 bootstrap 数据现有 fingerprint-aware stats 流程生成的固定 normalization
stats，DAgger round 不重新计算 stats。

完成 250/25 bootstrap 数据和上节 5k SFT 后，以下入口先在 env 1 执行 round-0
step-0/50-env baseline，再依次执行 10 轮 env 1 的 192-env 采集、转换、500-step
latent-only 训练和 env 1 的 seed-2/50-env 固定评测：

```bash
cd /pfs/users/lxy/latency-sensitive-bench
uv run --project . python scripts/official_gr00t/train_parkour_dagger.py
```

step-0 结果保存为 `round_00/eval_before/vla-eval.json`。每次 round eval 都把 `reward`、`episode_length`、`normalized_waypoint_progress` 和
`edge_violation` 的 mean/std 写入同一个 W&B run 的 `dagger_eval/*` namespace，并以
`train/global_step=0, 500, ..., 5000` 对齐，因此完整 run 一共 11 个 eval 点。round 边界中断
后用 `--start-round N`（零起始）继续；目标
`checkpoint-{(N+1)*500}/.dagger-round-complete` 已由成功返回的训练子进程原子写入时不会重复训练该
round。round 内采集中断不能只按 row count
续采，因为重新启动 Isaac 会回到相同 seed 的初始状态、从而重复 deterministic prefix；程序
会明确失败，必须由操作者先保留或清理该 round 的 partial directory 后从空目录重采。
每轮 eval 子进程正常结束后保留 checkpoint、eval 结果、policy 配置、`dataset_manifest.json`、
`training_datasets.json` 和该轮 `lerobot/`；只删除该轮 `raw/`。失败路径不清理，便于定位和续跑。
旧 current-only DAgger 输出缺少可复用的 round `lerobot/` 和聚合训练 manifest，不能作为该流程的
resume 根目录；重新跑时使用新的 output root 或先清掉旧 round/checkpoint。
默认输出根目录是 `outputs/official_gr00t/extreme_parkour_go2_dagger`，最终 VLA checkpoint 是
`training/checkpoint-5000`。

2026-09-03 的真实 smoke 已验证：SFT `checkpoint-5000` 可直接初始化
`Gr00tPolicy`，旧的 `haic_dagger` processor 参数错误不再出现；collector 随后以
`UnitreeGo2ParkourVlaEnvCfg` 创建 192 个 canonical env 1，并成功写出一个 192-row shard。
该 shard 包含 finite 的 `state[192,53]`、`action[192,34]`、
`termination[192,5]` 以及 MP4。尚未启动正式 640k-row / 3.2M-control-step run。
