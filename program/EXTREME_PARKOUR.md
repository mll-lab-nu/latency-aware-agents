# Extreme Parkour Isaac Lab / Go2 zero-delay baseline 执行手册

> 任务：`Isaac-Extreme-Parkour-{Teacher,Student}-Unitree-Go2-v0`
>
> Extreme Parkour fork：优化 base commit `4265e663627a3f5e50a076508b78cfdad05b2b39`；
> 当前 follow-up diff 恢复 student height scanner 的 20 ms `update_period`，并启用 cuDNN benchmark；
> zero-delay commit `991c4b02aa08a802e57d01924ef2f3826d0d7334`；upstream base
> `d6766d8883cffe427f2d4c087717d4eca1f23d8c`
>
> Isaac Lab：v2.2.0，commit `46dff135f44683f031edf346e544fcfd8456b2bb`
>
> Isaac Sim：5.0.0；PyTorch：2.7.0 + cu128
>
> 状态：依赖和优化 base commit 已固定；teacher/student 双卡 smoke、目标容量 A/B、精确 resume
> 和第二轮 free-lunch 筛选已完成。follow-up 尚待 review/commit；未启动正式训练。

## 1. 目标与实验口径

本轮只建立一个 baseline：

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
export PARKOUR_ROOT=/pfs/users/lxy/latency-sensitive-bench-parkour
cd "${PARKOUR_ROOT}"

git submodule sync --recursive
git submodule update --init --recursive
git -C third_party/IsaacLab rev-parse HEAD
git -C third_party/Isaaclab_Parkour rev-parse HEAD
```

期望输出分别为：

```text
46dff135f44683f031edf346e544fcfd8456b2bb
4265e663627a3f5e50a076508b78cfdad05b2b39
```

Parkour submodule URL 固定为 `git@github.com:mindorigin150/Isaaclab_Parkour.git`。fresh checkout
不再执行 `git apply`；commit 不匹配时直接检查 submodule URL/gitlink，不静默改用其他 revision。
当前 review worktree 相对 Parkour base 只允许存在 student height-scanner 100 ms → 20 ms、启用
cuDNN benchmark 两处一行 follow-up；正式启动前必须先提交并同步更新本节 revision 与父仓 gitlink。

## 5. 环境安装

环境独立位于 `/pfs/users/lxy/conda/envs/extreme-parkour`：

```bash
export PARKOUR_ROOT=/pfs/users/lxy/latency-sensitive-bench-parkour
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
export PARKOUR_ROOT=/pfs/users/lxy/latency-sensitive-bench-parkour
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

```bash
cd "${PARKOUR_ROOT}/third_party/Isaaclab_Parkour"

LD_PRELOAD=/lib/x86_64-linux-gnu/libnccl.so.2 NCCL_DEBUG=WARN \
OMNI_KIT_ACCEPT_EULA=YES "${PARKOUR_PY}" -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=2 --master_port=29542 \
  scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 \
  --distributed --headless --seed 1 \
  --num_envs 3072 --max_iterations 15000 \
  --run_name teacher_zero_delay_ddp_seed1_15k
```

完成后记录 rank 0 输出的确切 run directory 和最终 checkpoint。不要用目录 glob 或 `latest`。

### 7.2 Student

下面两个 placeholder 必须替换成上一步的精确值：

```bash
export TEACHER_RUN='<exact teacher run directory name>'
export TEACHER_CKPT='<exact teacher checkpoint filename>'

LD_PRELOAD="${PARKOUR_ENV}/lib/libGLU.so.1:/lib/x86_64-linux-gnu/libnccl.so.2" \
NCCL_DEBUG=WARN OMNI_KIT_ACCEPT_EULA=YES \
"${PARKOUR_PY}" -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=2 --master_port=29543 \
  scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 \
  --distributed --headless --seed 1 --enable_cameras \
  --num_envs 96 --max_iterations 5000 \
  --load_run "${TEACHER_RUN}" --checkpoint "${TEACHER_CKPT}" \
  --run_name student_zero_delay_ddp_seed1_5k
```

student runner 因 distillation 类型会加载 teacher model；这里不传 `--resume`，避免把 teacher
optimizer 状态解释成 student resume。

## 8. 固定评测

正式评测继续使用上游 `evaluation.py` 的 256 envs、1500 steps，以及四项输出：

```text
mean reward
mean episode length
normalized waypoint progress
edge violation
```

teacher 与 student 可分别占用 GPU 0/1 并行执行；每个命令都传确切 checkpoint 和 seed，运行
seeds 1、2、3。不要用 A1 论文图表或 Robot Parkour 五技能数值作为 Go2 gate。

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
| worktree | `/pfs/users/lxy/latency-sensitive-bench-parkour`；branch `research/extreme-parkour-lab`；base `ff70f83f` |
| GPU | 2× 97,887 MiB Blackwell，compute capability 12.0，driver 580.159.04；P2P read/write OK |
| Isaac Lab / Parkour | `46dff135...` pinned；Parkour optimization `4265e66...` pinned，upstream base `d6766d88...` |
| Python/Torch/Isaac Sim | Python 3.11；Torch 2.7.0+cu128；Isaac Sim 5.0.0；安装完成 |
| fork branches | `fix/extreme-parkour-ddp`=`e3fe6e3d...`；`research/zero-delay-go2`=`44dc7c8...`；`master`=`4265e66...`；均已发布 |
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
| 优化代码 pin | base `4265e66...` 已 commit/push，父仓 `ff70f83...` 已固定；scanner/cuDNN follow-up 待 review/commit |
| 第二轮 strict free lunch | teacher/student 双卡均未稳定超过 1% 端到端门槛；实验代码已回退 |
| allclose candidates | compile 因默认 `allclose` 失败而拒绝；cuDNN benchmark +1.26%～+1.64%，已启用，backward 最大绝对差异 `7.96e-13` |
| teacher 15k | 未启动（按要求停在 preflight） |
| student 5k | pending |
| fixed evaluation | pending |

## 10. 本轮明确 deferred

- `latency_bench` EnvAdapter、PolicyRunner、scheduler/executor 接入；
- fixed/profile/distribution/temporal latency sweep；
- 上游 action FIFO、旧 depth frame 或 latency-aware retraining；
- Robot Parkour Learning 五技能、官方 A1 Isaac Gym 复现；
- 新 evaluator、sim2sim、sim2real 和真机部署；
- 更长稳态 profile、Nsight kernel trace 和 Student camera pipeline 进一步深挖；
- `empirical_normalization=True` 的 teacher-to-student 双 normalizer 数据流；当前两个正式配置均关闭；
- 为 future backend、robot 或第三张 GPU 增加抽象。
