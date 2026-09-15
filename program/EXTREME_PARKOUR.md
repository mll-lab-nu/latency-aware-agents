# Extreme Parkour / Go2 teacher 与早期 NVIDIA GR00T 实验记录

> NVIDIA GR00T 独立运行时和入口已移除。当前统一使用 StarVLA；运行入口见
> [SCALE_MODELS](SCALE_MODELS.md)，产物和评测状态见 [PROGRESS](PROGRESS.md)。
> 下文只保留历史实验事实，命令不再是可执行手册。

新机器的完整环境安装以 [QUICK_START.md](../docs/QUICK_START.md) 为准。本文的服务器路径与驱动修复保留为运行记录；四模型连续 state H1 流水线见 [SCALE_MODELS.md](SCALE_MODELS.md)。

> 任务：`Isaac-Extreme-Parkour-{Teacher,Student}-Unitree-Go2-v0`
>
> Isaac Sim：5.0.0；PyTorch：2.7.0 + cu128
>
> 首个 adaptive-LR teacher正式 run 已确认 collapse，并在 iteration 1418 停止；fixed-LR teacher 已通过 iteration
> 100/300/1000 门禁并正常完成 15k；Teacher evaluation 已通过。
> 34D terrain-latent+yaw VLA 的 RGB collector、LeRobot/GR00T 数据转换、5k 训练入口和
> zero-delay evaluator 已实现；固定-LR teacher 已通过，旧 latent-only bootstrap/SFT/DAgger
> 仅作为历史证据。action+yaw objective、完整 5×50 Hz actor observation、fresh bootstrap
> 和固定 50/50 replay 已实现；当前 task-owned 入口默认执行 SFT 及 round 0--9。

读者复现入口见 [`docs/extreme_parkour/cookbook.md`](../docs/extreme_parkour/cookbook.md)，跨任务阶段契约见
[`docs/standard_pipeline.md`](../docs/standard_pipeline.md)。

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
- latency integration：task-owned eval entrypoint 已注入本仓库 scheduler/executor。拟合样本从
  `env.observe()` 到 inference parent 收到 34D（32D terrain latent + 2D yaw）action chunk；冻结
  actor 生成 12D motor action及后续 simulator step 不进入 profile。

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
（teacher 9.74 h + student 4.60 h）；

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

## 5. 环境安装

新机器按 [QUICK_START.md §5](../docs/QUICK_START.md#5-extreme-parkour) 安装命名环境 `extreme-parkour`。
固定组合是 Python 3.11、Torch 2.7.0+cu128、Isaac Sim 5.0.0 和仓库内 IsaacLab v2.2.0。
HAIC 使用独立的 IsaacLab v2.3.2 checkout，不要为 HAIC 切换这里的 editable 源码版本。

以下历史运行记录使用 `/pfs/users/lxy/conda/envs/extreme-parkour`；新机器将
`PARKOUR_ROOT`、`PARKOUR_ENV` 和 `PARKOUR_PY` 替换为自己的安装路径。

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

## 9. 实际运行记录

| 项目 | 状态/结果 |
|---|---|
| Python/Torch/Isaac Sim | Python 3.11；Torch 2.7.0+cu128；Isaac Sim 5.0.0；安装完成 |
| teacher 3072 env/rank profile | baseline 54,056 FPS；optimized run `2026-09-01_19-35-52_profile_teacher_strict_opt_final_ddp_3072` 63,390 FPS，+17.3% |
| student 96 env/rank profile | baseline 6,192 FPS；optimized run `2026-09-01_19-36-56_profile_student_strict_opt_final_ddp_96` 6,958.5 FPS，+12.4% |
| teacher/student exact resume | 通过；优化 profile 的 `model_4.pt` 均从 iteration 4 继续训练 1 iteration |
| camera teardown | current keyboard unsubscribe API 单卡实跑通过，无退出 `AttributeError` |
| allclose candidates | compile 因默认 `allclose` 失败而拒绝；cuDNN benchmark +1.26%～+1.64%，已启用，backward 最大绝对差异 `7.96e-13` |
| teacher adaptive 15k | **failed/collapsed**；run `2026-09-01_23-53-57_teacher_zero_delay_ddp_seed1_15k`；[W&B `vjsucm5x`](https://wandb.ai/saberrr-zju/isaaclab-parkour/runs/vjsucm5x)；console `logs/teacher_zero_delay_ddp_seed1_15k.console.log`；LR iteration 5 达 `1e-2`，value loss iteration 12 达 `9.6582`，reward iteration 24 峰值 `0.2307` 后下降并从约 iteration 80 持续为零；iteration 1418 人工停止，episode length `11.09` |
| teacher fixed 15k | **completed；exit 0**；run `2026-09-02_00-49-22_teacher_zero_delay_fixed_lr_ddp_seed1_15k`；checkpoint `model_14999.pt`；[W&B `8lpfqa6a`](https://wandb.ai/saberrr-zju/isaaclab-parkour/runs/8lpfqa6a)；console `logs/teacher_zero_delay_fixed_lr_ddp_seed1_15k.console.log`；保存配置已确认 `schedule=fixed`、LR `2e-4`；gates 100/300/1000 均通过；iteration 14999 reward `19.59`、episode length `710.00`、terrain level `6.0593`、value loss `0.0049`；训练于 `2026-09-02 02:07 UTC` 正常结束，未复现 collapse |
| student 5k | **running；resumed from model 2200**；original run `2026-09-02_10-10-50_student_zero_delay_ddp_seed1_5k` / [W&B `bhzgjid1`](https://wandb.ai/saberrr-zju/isaaclab-parkour/runs/bhzgjid1) / console `logs/student_zero_delay_ddp_seed1_5k.console.log`；iteration 2286 后被外部硬终止，loss 正常、无程序错误；resume run `2026-09-02_13-02-16_student_zero_delay_ddp_seed1_5k_resume2200` / [W&B `g3i1d4dh`](https://wandb.ai/saberrr-zju/isaaclab-parkour/runs/g3i1d4dh) / console `logs/student_zero_delay_ddp_seed1_5k_resume2200.console.log`；精确恢复 `model_2200.pt` 的网络和 optimizer，iteration 2220 total loss `1.1273`；最近 20 iterations 中位 `4.045s`，预计 `2026-09-02 08:11 UTC` 完成 |
| teacher fixed evaluation | **passed**；checkpoint `model_14999.pt`；console `logs/eval_teacher_fixed_lr_seed1_model_14999.console.log`；reward `23.14±6.02`、episode length `907.95±296.15`、normalized waypoint progress `1.00±0.01`、edge violation `0.19±0.42` |
| student fixed evaluation | pending |
| VLA zero-delay common-executor smoke（bootstrap flow-aux SFT checkpoint-5000） | **completed；公共 summary 已生成（2026-09-05 01:11 UTC）**；`latency-eval --eval-config outputs/official_gr00t/extreme_parkour_go2_action_yaw_h40_official_flow_aux_dagger/bootstrap/vla_eval/eval_config.yaml`（100 episodes / 50 slots / seed 2 / `--inference_batch_size 32` / runtime `model_14999.pt`）；输出 `bootstrap/vla_eval/queue_eval_latency_0.json` 与 `queue_eval_results.jsonl`：逐步 100 episodes、mean_length `848.43`、mean_return `10.57±5.63`、`normalized_waypoint_progress` `0.385±0.248`、`edge_violation` count `84,843`。该 run 是统一链路 smoke，不作为策略质量结论 |


## 11. 34D perception VLA zero-delay bootstrap

这一步是进入 DAgger 前的额外 bootstrap，不是原 paper 的训练阶段。它先冻结
`model_14999.pt` 的 teacher actor，将官方 GR00T-N1.7-3B 放在原 depth encoder 的位置：

```text
robot RGB 256x150 @ 10 Hz + current proprio 53D + fixed prompt
  -> swappable PolicyRunner
  -> [40,34] terrain-latent+yaw chunk at 50 Hz resolution
  -> consume chunk[0..4], fixed teacher actor @ 50 Hz, then replan
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
的轨迹，不把终点前摔倒或 timeout 当成功。它在内存中暂存 episode，并以原子 NPZ+MP4
shards 写出 10 Hz RGB、5×50 Hz actor observation、termination 和 reward；按 split seed 0
得到精确 250 train / 25 validation episodes：

修复前生成的 `outputs/official_gr00t/extreme_parkour_go2/raw` 和
`outputs/official_gr00t/extreme_parkour_go2/lerobot` 缺少 action objective 所需的 5×50 Hz
actor observation，不能用于 replay；必须在新目录整套重建，不能
继续转换、混用或用于训练。collector 要求 `--output_dir` 是空的、不存在的目录；完成相机
smoke 后再执行以下采集和转换命令。训练进程与 checkpoint 不属于本节数据修复范围。

```bash
cd "${PARKOUR_ROOT}"
PYTHONPATH="${REPO_ROOT}" OMNI_KIT_ACCEPT_EULA=YES \
  "${PARKOUR_PY}" scripts/rsl_rl/parkour_vla.py collect \
  --headless --seed 1 --num_envs 50 --checkpoint "${TEACHER_CKPT}" \
  --train_episodes 250 --val_episodes 25 --split_seed 0 \
  --output_dir "${REPO_ROOT}/outputs/official_gr00t/extreme_parkour_go2_action_data/raw"

cd "${REPO_ROOT}"
uv run --project . python scripts/official_gr00t/prepare_parkour_dataset.py \
  outputs/official_gr00t/extreme_parkour_go2_action_data/raw \
  outputs/official_gr00t/extreme_parkour_go2_action_data/lerobot_h40_semantic_stats \
  --video-shards
```

finalizer 直接从 NPZ+MP4 shards 生成 train/validation LeRobot parquet/videos、53D state/34D action、
`actor_observation[5,753]` 和 `termination[5]`，并对两套 dataset 运行 `gr00t.data.stats`。缺字段、错误 action layout 或错误
model output dimension 直接失败，不使用别名、零 state 或其他 fallback。

### 11.3 旧 latent-only GR00T 5k 与最终闭环评测

已完成的 latent bootstrap 使用 fresh `GR00T-N1.7-3B`、2 GPUs、global batch 128、LR `1e-4`、
H1、5k steps、one-step regression、vision tuning、BF16、action/state dropout 0。该结果只保留
作历史证据，section 12 不再加载或续训该 checkpoint；旧 shell wrapper 已删除，当前入口为：

```bash
cd "${REPO_ROOT}"
bash scripts/tasks/extreme_parkour_go2/train_vla.sh \
  /absolute/path/to/model_14999.pt --end-round 0 \
  --bootstrap-raw outputs/official_gr00t/extreme_parkour_go2_action_data/raw
```

最终 latency evaluator 在 Isaac GPU 与独立 GR00T Python worker 之间复用公共 inference pool；
同一个 task-owned vector backend 同时服务 measured、zero 和 profile-replay。50 个
环境按 batch 8 同步预测，全部 34D perception 输出返回前仿真不前进，因此 zero 模式定义为
zero-delay。VLA 每次返回 H40 chunk，actor 依次使用 `chunk[0]` 到 `chunk[4]`，然后用最新
observation 重新规划；五个位置的训练 label 相同，但 runtime 不强制复制 `chunk[0]`。
预测 yaw 在进入 actor 前乘回 `1.5`。
公共 executor 在记录 terminal step 后按 slot reset，直到累计恰好 100 个 episode；每个 episode 独立最多 1,500
个 50 Hz control steps（30 秒）。初始 terrain assignment 固定后，eval reset 不再按被测 policy
表现升降 terrain level，因此各 checkpoint 使用同一评测分布：

```bash
cd "${PARKOUR_ROOT}"
PYTHONPATH="${REPO_ROOT}" OMNI_KIT_ACCEPT_EULA=YES \
  "${PARKOUR_PY}" scripts/rsl_rl/parkour_vla.py latency-eval \
  --headless --seed 2 --num_envs 50 --checkpoint "${TEACHER_CKPT}" \
  --eval_episodes 100 --max_episode_steps 1500 \
  --eval-config "${REPO_ROOT}/configs/examples/extreme_parkour/official_gr00t_eval.yaml" \
  --inference_device cuda:1 --inference_batch_size 8 \
  --output_dir "${REPO_ROOT}/outputs/official_gr00t/extreme_parkour_go2/eval"
```

输出仍为 reward、episode length、normalized waypoint progress、edge violation。四项必须
finite。此前记录的 clean SFT checkpoint-5000 progress `0.6850±0.3393` 来自旧 evaluator：
每 5 个 control steps 只有第 1 步使用 VLA 预测 yaw，后 4 步错误复用了 observation 中的
privileged yaw，因此该结果无效。commit `658cd53` 修复为 5 步持续使用预测 yaw 后，同一
checkpoint 在 seed 2、50 env、batch 32 下重测 progress 为 `0.2325±0.2165`；该结果来自旧的
50-episode evaluator，仅保留为历史诊断值。

## 12. 固定 teacher actor 的 action + yaw DAgger

DAgger 从 fresh action+yaw SFT VLA 开始迭代，固定同一个 teacher actor 用于 bootstrap、rollout
和评测：

```text
RGB + masked proprio -> trainable VLA -> latent32 + normalized yaw2
                                      -> paper MTS yaw -> fixed teacher actor
                                      -> joint action12 (rollout only)
                                      -> fixed joint-position/PD controller -> Go2
```

SFT collection、DAgger collection 和 evaluation 全部使用
`Isaac-Extreme-Parkour-VLA-Unitree-Go2-v0` 这个 canonical env 1。DAgger 唯一改变的是由当前
VLA 控制机器人，并把并行环境数从 50 覆盖为 192；不切换到原 depth-student env 2。

bootstrap 与 DAgger 使用完全相同的训练目标：不再监督 32D latent 本身，只保留两个与闭环直接
一致的 loss：

- `action_loss`：VLA 的 40 个位置分别对应 40 个 50 Hz actor observation；每 5 个位置共享
  同一个 10 Hz oracle latent/yaw label。在冻结 actor 输出上计算 student/teacher 12D motor
  action 的向量 L2；第一个 terminal action 有效，terminal 后位置由累计 mask 排除；
- `yaw_loss`：40 个位置的 predicted current/next yaw 与对应 oracle yaw 在同一 radian 单位上
  计算向量 L2，并使用相同 terminal mask。

teacher motor action 由同一个冻结 actor、原始 actor observation 和 oracle terrain latent 在线
算出，不在数据里重复保存。actor 只作为训练期 loss 函数加载，不进入 optimizer、GR00T model
或 checkpoint；保存结果仍是标准 34D VLA。VLA 输入中的 yaw 两维继续置零。rollout 按论文 MTS
比较 predicted current yaw 与 GT current yaw 的环形误差：误差小于 `0.6 rad` 时使用整对
predicted yaw，否则使用整对 GT yaw；最终 eval 始终使用 predicted yaw，不使用 MTS fallback。

这与原 Parkour 官方 Phase 2 实现的实际启用路径一致：总 loss 是 motor-action L2 加 direction/yaw
L2；仓库中虽保留 latent loss 函数，但 runner 调用被注释，没有启用 latent matching。本项目冻结
actor 是为隔离 VLA 修正而采用的简化版本，不等同于论文中可训练的完整 student actor。

GR00T action head 固定输出 H40。训练使用官方 random-t velocity-MSE，只调用一次 denoiser，
并在该次预测的 clean endpoint 上增加冻结 actor action L2 与 yaw L2；总 loss 为两者之和。
inference 从 fresh Gaussian 出发调用官方默认 4-step Euler sampler，一次 policy call 只编码一次
RGB/state，四次 denoise 复用同一组 backbone features。

### 12.1 有界 DAgger 预算

原配置每个 iteration 是 `192 envs × 120 control steps = 23,040` 个 50 Hz transitions；depth
每 5 步更新一次，所以等价为 `4,608` 个 10 Hz perception samples。5,000 iterations 总计：

| quantity | 每轮 | 10 轮总计 |
|---|---:|---:|
| 10 Hz VLA rows | 64,000 | 640,000 |
| 50 Hz rollout control steps | 320,000 | 3,200,000 |
| optimizer updates | 500 | 5,000 |

bootstrap 从 fresh `GR00T-N1.7-3B` 开始，在 action-rich 250/25 episode 数据上使用 global
batch `128`、gradient accumulation `1`、peak LR `5e-5` 训练 5,000 updates，不混入 replay。
DAgger 使用相同 batch、gradient accumulation 和 peak LR，因此每轮执行
`500 × 128 = 64,000` 个 supervised samples。每轮新增 64,000 行 current-VLA rollout，每行携带一个
oracle 34D latent/yaw target。采样权重固定为 50% 新采集的 action-rich bootstrap replay，
另 50% 在所有已完成 DAgger round 与当前 round 间按数据量分配；bootstrap 不会随着 round 增多
而被稀释。collector 和 GR00T shard 均为 1,000 rows，每轮 64 shards；模型、
optimizer 和 scheduler 从上轮 checkpoint 继续，teacher actor 始终从固定 teacher checkpoint
加载。

### 12.2 数据与 round driver

DAgger collector 使用一个 canonical env 1 进程、192 camera envs 和 seed 1；与 SFT/eval
相比只改变 rollout policy 和并行环境数。每个原子
shard 是 `episode.npz + episode.mp4`，不落逐帧 PNG；每个 shard 只包含单一 env、单一
episode 的连续 segment，reset 或 1,000-row 上限处切分；segment 尾部最后一个 control
位置标为 sequence boundary，使 padding 不会产生伪 future supervision。numeric payload 保存
`state[53]`、`action[34]`、`actor_observation[5,753]` 和 `termination[5]`。每行保存一个 oracle 34D latent/yaw target，
包括 failure/timeout 标记处；
loader 使用 `delta_indices[j] = j // 5` 将连续 8 行组装成 40 个 50 Hz target，并用
`row[i+j//5].actor_observation[j%5]` 取得对应 actor state。`termination` 同时控制 action/yaw
loss mask。每个环境执行 5 个 chunk 位置后刷新 VLA；reset 立即丢弃旧 chunk 并从 `k=0` 重启。
reset observation 同时从新 root pose 和新 goals 立即重算 wrapped GT yaw，不复用上一 episode 缓存。
LeRobot parquet 直接保留这些诊断列，
并逐字节复制 bootstrap 的固定 normalization stats，DAgger round 不重新计算 stats。Parkour
action 使用语义范围：32D terrain latent 为 `[-1,1]`，两维 yaw 为
`[-pi/1.5,pi/1.5]`；不再用 bootstrap 样本偶然观察到的 yaw min/max 裁剪标签。

默认仍由 GPU 0 运行 Isaac Sim、GPU 1 运行一个 GR00T worker。双 worker A/B 显式重复
`--inference-device`，例如 `--inference-device cuda:0 --inference-device cuda:1`；collector
按全局 env slot 固定 worker affinity，不在不同请求或 reset 后迁移 slot。正式使用双 worker
前先以相同 row budget 比较外部 wall time，未达到 20% 提升则保留默认单 worker。

旧 latent-only `checkpoint-5000` 不再作为初始化。当前读者入口先用上节重新采集的 action-rich
bootstrap dataset，从 GR00T base fresh 训练 5k action+yaw bootstrap；随后在 env 1 执行 step-0
baseline，并默认执行 round 0--9 的 192-env 采集、转换、500-step action+yaw 训练和 seed-2
固定评测。每个 eval 使用 50 个并行环境、严格统计 100 个 episode，单 episode 最多 1,500
control steps：

```bash
cd /pfs/users/lxy/latency-sensitive-bench
bash scripts/tasks/extreme_parkour_go2/train_vla.sh \
  /absolute/path/to/model_14999.pt \
  --bootstrap-raw outputs/official_gr00t/extreme_parkour_go2_action_data/raw
```

step-0 结果保存为 `bootstrap/vla_eval/queue_eval_latency_0.json`。每次 round eval 都把 `reward`、`episode_length`、`normalized_waypoint_progress` 和
`edge_violation` 的 mean/std 写入同一个 W&B run 的 `dagger_eval/*` namespace，并以
`train/global_step=0, 500, ..., 5000` 对齐。默认执行 `--end-round 10`；仅做 SFT 时使用
`--end-round 0`，续跑时用
task-owned wrapper 使用 `--resume --start-round N --end-round 10` 启动剩余轮次。round 边界中断
后用 `--start-round N`（零起始）继续；目标
`checkpoint-{(N+1)*500}/.dagger-round-complete` 已由成功返回的训练子进程原子写入时不会重复训练该
round。round 内采集中断不能只按 row count
续采，因为重新启动 Isaac 会回到相同 seed 的初始状态、从而重复 deterministic prefix；程序
会明确失败，必须由操作者先保留或清理该 round 的 partial directory 后从空目录重采。
每轮 eval 子进程正常结束后保留 checkpoint、eval 结果、policy 配置、`dataset_manifest.json`、
`training_datasets.json` 和该轮 `lerobot/`；只删除该轮 `raw/`。失败路径不清理，便于定位和续跑。
旧 current-only DAgger 输出缺少可复用的 round `lerobot/` 和聚合训练 manifest，不能作为该流程的
resume 根目录；重新跑时使用新的 output root 或先清掉旧 round/checkpoint。
默认输出根目录是 `outputs/official_gr00t/extreme_parkour_go2_action_yaw_h40_official_flow_aux_dagger`；fresh bootstrap
checkpoint 是 `bootstrap/training/<wandb-run-id>_sft/checkpoint-5000`，round-0 checkpoint 是
`training/<wandb-run-id>/checkpoint-500`。`--bootstrap-raw` 会跳过 bootstrap gate/重采，直接从已有 raw
重建 H40 dataset 后训练；已有 fresh bootstrap 时可省略 `--run-bootstrap`，或用
`--bootstrap-checkpoint` 显式指定；旧 latent-only checkpoint 不得传入。

2026-09-03 的旧 DAgger 与 statsfix run 均未使用论文 MTS，且 collector 存在 reset phase
锁相；旧 normalization run 还会裁剪超出 bootstrap 范围的 yaw。旧 run 的 round-0 progress
分别从 `0.2225` 降至 `0.0700`、从 `0.2300` 降至 `0.1475`，全部数据和 checkpoint 作废，
不能用于 resume 或后续聚合训练。
随后仍采用 latent-only loss 的一次 run 又从 `0.245` 降到 `0.088`；该结果进一步说明修复
rollout/eval plumbing 并没有修复监督目标，相关 checkpoint 同样只保留作失败证据。

2026-09-04 的 H40 4-step 可微 sampler run 在约 step 3380 后出现持续梯度爆炸，progress 从
checkpoint-3000 的 `0.5225` 降至 checkpoint-3500 的 `0.4075`、checkpoint-4000 的
`0.2925`。该 run 已停止并原样保留；不能 resume 到新的 random-t flow + auxiliary 路径。
