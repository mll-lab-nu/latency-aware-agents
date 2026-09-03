# MIKASA / InterceptGrabFast H1 zero-delay baseline 执行手册

> 任务：`InterceptGrabFast-VLA-v0`
>
> MIKASA：v1.0.0，commit `16634db18bef08128ed79346469c86fc12169aed`
>
> StarVLA：commit `823cd2cdd4fc3934c9d89ac96c03c23b07cd8004`
>
> 状态：zero-delay PPO teacher、275 条成功轨迹、250/25 LeRobot、5k QwenOFT SFT 和
> canonical VLA 验收均已完成；当前复现入口见 [`docs/mikasa/cookbook.md`](../docs/mikasa/cookbook.md)。

## 1. 目标与不可更改的实验口径

本轮只跑通一个 zero-delay baseline：

```text
privileged-state PPO teacher
  -> 275 successful GrabFast trajectories
  -> QwenOFT H1 SFT
  -> canonical zero-delay evaluation
```

这是一条 **task-specific PPO distillation** 路线：先在 `InterceptGrabFast-VLA-v0` 上训练
privileged-state PPO，再用它生成 GrabFast demonstrations。它不是复刻“用 InterceptMedium
数据训练后直接 zero-shot 到 GrabFast”的官方表格，因此不能把那种 transfer setting 下可能出现的
0 分当作本路线的预期结果。

锁定以下设置，不在执行中静默替换：

- 环境：`InterceptGrabFast-VLA-v0`。
- latency：zero-delay；本轮不做 fixed-delay、realtime profile 或 latency-aware training。
- 控制：H1、7D `pd_ee_delta_pose`，官方 20 Hz（每个 control step 50 ms）；每次只预测当前一帧 action。
- teacher：49D privileged state PPO，不是 VLA teacher。
- dataset：275 条成功轨迹，前 250 条 train、后 25 条 validation。
- VLA：当前仓库的 `QwenOFT`，即 Qwen3-VL-4B + MLP action head；不是 upstream
  OpenVLA-7B。
- SFT：单卡、DeepSpeed ZeRO-2、global batch 64、5,000 optimizer steps。
- 最终 teacher gate：官方 trainer gate 加独立 canonical 100-episode gate，两者都要通过。
- 最终 VLA baseline gate：canonical zero-delay 50 episodes，至少 25 次 `success_once`。

本手册记录底层实验命令和历史证据；读者复现使用 task-owned
`scripts/tasks/mikasa_intercept_grab_fast/` 入口，避免重复拼接环境变量。

## 2. 已核对的关键事实

### 2.1 PPO teacher 与 reward

teacher 使用 49D privileged state。当前 live observation 的组成是：

```text
qpos 9 + qvel 9 + TCP pose 7 + ball pose 7 + initial velocity/oracle 3
+ grasp flag 1 + ball linear velocity 3 + ball angular velocity 3
+ TCP-to-ball 3 + gripper width 1 + relative velocity 3 = 49
```

actor 和 critic 分离，各自是 3 层、每层 256 hidden units 的 Tanh MLP；actor 输出 7D
Gaussian action。训练 rollout 时采样，dataset 和 evaluation 时使用 deterministic mean action。

`normalized_dense` reward 保持官方实现：

- reach：`10 * (1 - tanh(3 * tcp_to_ball_distance))`；
- ball in gripper region：`+10`；
- very close：最多 `+30`；
- 合适的 gripper width/closing：最多 `+90`；
- 已尝试闭合 gripper：`+90`；
- 已抓住 ball：`+200`；
- success 时总 reward 直接覆盖为 `300`；最后整体除以 `300`。

success 定义为 `is_ball_grasped AND robot_is_static(0.2)`。`static_reward` 等中间量会写入
reward diagnostics，但当前没有加到 shaped reward。训练和验收以 `success_once` 为主，不能用
return 代替 success rate。

### 2.2 官方 PPO 默认参数

```text
seed=123                       reward_mode=normalized_dense
num_envs=1024                  num_eval_envs=16
num_steps=60                   num_eval_steps=60
learning_rate=3e-4             gamma=0.99
gae_lambda=0.95                num_minibatches=32
update_epochs=4                norm_adv=true
clip_coef=0.2                  clip_vloss=false
ent_coef=0.0                   vf_coef=0.5
max_grad_norm=0.5              target_kl=0.2
anneal_lr=false                finite_horizon_gae=true
eval_freq=25                   torch_deterministic=true
```

每轮 PPO batch 是 `1024 * 60 = 61,440` transitions，minibatch 是 1,920。因为 trainer 对
iteration 数向下取整：

```text
5M request   -> 81 iterations   -> 4,976,640 actual steps
50M request  -> 813 iterations  -> 49,950,720 actual steps
150M request -> 2441 iterations -> 149,975,040 actual steps
```

eval 在 iteration 1、26、51……执行；除第一次训练前 eval 外，相邻 eval 约隔 1.536M
environment steps。

官方 threshold 是 `success_once >= 0.95`，但默认每次 eval 只有 16 个完整 episode：

```text
15 / 16 = 0.9375 < 0.95
16 / 16 = 1.0
```

因此 trainer gate 实际要求当轮 `16/16`，通过后保存 `final_success_ckpt.pt` 并立即退出。它不是
固定 canonical seeds 上的统计 gate，必须再做独立 100-episode 验收。

### 2.3 SFT 模型与 batch

当前 `openvla` 名称实际映射到：

```text
top RGB + wrist RGB + instruction
  -> Qwen3-VL-4B-Instruct
  -> MLP action head
  -> one 7D action
```

配置固定为：

```text
framework.name=QwenOFT
action_dim=7                  action_env_dim=7
action_horizon=1             future_action_window_size=0
past_action_window_size=0    include_state=false
loss_type=l1                 bf16=true
```

LeRobot 数据中保留 7D proprio 只为 dataset contract/statistics；它不会输入 QwenOFT。

单卡时：

```text
global batch = per-device batch * gradient accumulation
```

所以 `global batch=64`、`5,000 steps` 一共呈现 320,000 个训练样本。若 250 条 train
trajectory 仍约有 12,440 transitions，则约为 `320000 / 12440 = 25.72 epochs`；这次明确以
5,000 optimizer steps 为停止条件，不再称为 13 epochs。最终 epoch 必须用本次真实 train
transition 数重新计算并记录。

## 3. Phase 0：同步服务器、密钥和环境

### 3.1 已知服务器事实

用户原先给出的 `/pfs/users/lxy/latency-sensitive-vla` 在当前节点不存在。实际可见文件是：

- `/pfs/users/lxy/latency-sensitive-bench/servers.md`：只记录
  `ssh xinyuan@204.12.163.107`；
- `/pfs/users/lxy/latency-sensitive-bench/.env`：包含 `WANDB_API_KEY`、`WANDB_ENTITY`、
  `HF_TOKEN` 等变量。

服务器地址已更正为 `204.12.163.107`，当前节点可以直接 SSH 连接；硬件与环境实测结果见
“实际运行记录”。

### 3.2 rsync 当前未提交工作树

在能连接服务器的源机器上执行：

```bash
LOCAL_ROOT=/pfs/users/lxy/latency-sensitive-bench-mikasa-h1
REMOTE_HOST=xinyuan@204.12.163.107
REMOTE_DIR=latency-sensitive-bench-mikasa-h1

git -C "${LOCAL_ROOT}" submodule update --init \
  third_party/flappy-bird-gymnasium \
  third_party/sample-factory
git -C "${LOCAL_ROOT}/third_party/MIKASA-Robo" submodule update --init --recursive \
  utils/convert_npz_to_rlds/rlds_dataset_builder
git -C "${LOCAL_ROOT}/third_party/MIKASA-Robo/utils/convert_npz_to_rlds/rlds_dataset_builder" \
  checkout --detach 7b2f4f38a7773b792f6e690f5c838aaea39754f6

ssh "${REMOTE_HOST}" "mkdir -p ~/${REMOTE_DIR}"

rsync -a --info=progress2 \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='outputs' \
  --exclude='results' \
  --exclude='wandb' \
  "${LOCAL_ROOT}/" "${REMOTE_HOST}:~/${REMOTE_DIR}/"

scp /pfs/users/lxy/latency-sensitive-bench/.env \
  "${REMOTE_HOST}:~/${REMOTE_DIR}/.env"
ssh "${REMOTE_HOST}" "chmod 600 ~/${REMOTE_DIR}/.env"
```

不要打印 `.env`，不要把它加入 Git/W&B artifact。当前 superproject 已固定包含集成代码的
MIKASA/StarVLA submodule revision，不需要再应用 vendor patch；也不要做与上述依赖无关的广义
submodule update。StarVLA bootstrap 还会直接安装上面两个固定 top-level submodule，所以它们也
必须在 rsync 前初始化。若从 fresh clone 开始，额外先初始化两个 vendor submodule：

```bash
git submodule update --init third_party/MIKASA-Robo third_party/starVLA
```

然后执行本节前面的 top-level dependency 与 RLDS builder 初始化命令。

### 3.3 远端 preflight

登录后：

```bash
ssh xinyuan@204.12.163.107
cd ~/latency-sensitive-bench-mikasa-h1
export REMOTE_ROOT="$PWD"

set -a
source .env
set +a

nvidia-smi
df -h .
which uv
which conda
```

若远端没有 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"
```

若远端没有 Conda：

```bash
curl -fsSL \
  https://repo.anaconda.com/miniconda/Miniconda3-py310_26.1.1-1-Linux-x86_64.sh \
  -o /tmp/miniconda-py310.sh
bash /tmp/miniconda-py310.sh -b -p "${HOME}/miniconda3"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
```

记录以下信息到本文的“实际运行记录”：GPU 型号和数量、每卡显存、driver、CUDA、可用磁盘、
远端 hostname。确认 Hugging Face 下载与 W&B 均可联网，但不要把 token 输出到终端。

本轮 PPO 和 SFT 都只使用一张卡。PPO 的 1,024 parallel envs 在一张 GPU 内运行；SFT 显式
设置 `CUDA_VISIBLE_DEVICES=0` 和 `--num_processes 1`。

### 3.4 MIKASA 环境

```bash
cd "${REMOTE_ROOT}"
bash scripts/mikasa/install.sh

export MIKASA_PY="${REMOTE_ROOT}/third_party/MIKASA-Robo/.venv/bin/python"
"${MIKASA_PY}" -c \
  'import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())'
```

如果 capability 是 `(12, 0)`，项目锁定的 Torch 2.2.1 不支持该 GPU。只在确认这一事实后执行：

```bash
uv pip install \
  --python "${MIKASA_PY}" \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.7.1'
```

该服务器的系统包策略禁止安装 NVIDIA GL/Vulkan 包，但 kernel driver 是 `580.173.02`。使用同版本
Ubuntu userspace package，不修改 apt pin 或内核驱动：

```bash
export NVIDIA_VULKAN_ROOT="${HOME}/.local/nvidia-580.173.02"
cd /tmp
apt-get download 'libnvidia-gl-580=580.173.02-0ubuntu0.24.04.1'
mkdir -p "${NVIDIA_VULKAN_ROOT}"
dpkg-deb -x \
  /tmp/libnvidia-gl-580_580.173.02-0ubuntu0.24.04.1_amd64.deb \
  "${NVIDIA_VULKAN_ROOT}"

export LD_LIBRARY_PATH="${NVIDIA_VULKAN_ROOT}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VK_DRIVER_FILES="${NVIDIA_VULKAN_ROOT}/usr/share/vulkan/icd.d/nvidia_icd.json"
```

覆盖 Torch 后，本文所有 MIKASA 命令都直接调用 `${MIKASA_PY}`，不要再用会重新同步
Torch 2.2.1 的 `uv run`。每个新的 MIKASA shell 都必须重新 export 上述两个 Vulkan runtime
变量。

执行最小 runtime smoke：

```bash
cd "${REMOTE_ROOT}"
PYTHONPATH="${REMOTE_ROOT}:${REMOTE_ROOT}/third_party/MIKASA-Robo" \
"${MIKASA_PY}" - <<'PY'
import gymnasium as gym
import torch
import mikasa_robo_suite.vla.memory_envs  # noqa: F401

env = gym.make(
    "InterceptGrabFast-VLA-v0",
    num_envs=1,
    obs_mode="state",
    control_mode="pd_ee_delta_pose",
    render_mode="all",
    sim_backend="gpu",
    reward_mode="normalized_dense",
)
obs, _ = env.reset(seed=123)
action = torch.zeros((1, 7), device=env.unwrapped.device)
env.step(action)
print("runtime smoke OK", env.action_space.shape)
env.close()
PY
```

验收：reset 和一次 step 成功、GPU kernel 无 architecture error、action 最后一维为 7。

### 3.5 StarVLA/QwenOFT 环境与权重

优先复用服务器上已经存在且能通过 import smoke 的 `starvla_rl_games_openvla`。若不存在，使用
仓库已有 bootstrap，而不是手写另一套 requirements：

```bash
cd "${REMOTE_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
LATENCY_BENCH_ROOT="${REMOTE_ROOT}" \
  bash third_party/starVLA/examples/rl_games/install/bootstrap.sh \
  --tier dev --model openvla
```

激活环境。若服务器用 `/venv` prefix，则激活对应 prefix；否则使用命名环境：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ -d /venv/starvla_rl_games_openvla ]]; then
  conda activate /venv/starvla_rl_games_openvla
else
  conda activate starvla_rl_games_openvla
fi
cd "${REMOTE_ROOT}/third_party/starVLA"

python -c 'import torch, deepspeed, accelerate, flash_attn; print(torch.__version__)'
```

下载 base VLM 和 Bridge/RT-1 initialization checkpoint：

```bash
hf download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct

hf download StarVLA/Qwen3VL-OFT-Bridge-RT-1 \
  --local-dir playground/Pretrained_models/Qwen3VL-OFT-Bridge-RT-1

test -f playground/Pretrained_models/Qwen3VL-OFT-Bridge-RT-1/checkpoints/steps_5000_pytorch_model.pt
```

## 4. Phase 1：训练并严格验收 PPO teacher

### 4.1 W&B 与通用入口

PPO trainer 默认 `track=false`。正式运行直接调用底层 Tyro CLI：

```bash
cd "${REMOTE_ROOT}"
set -a
source .env
set +a

export PYTHONPATH="${REMOTE_ROOT}:${REMOTE_ROOT}/third_party/MIKASA-Robo"
export PPO_ENTRY="${REMOTE_ROOT}/third_party/MIKASA-Robo/mikasa_robo_suite/vla/dataset_collectors/get_dataset_collectors_ckpt.py"
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VULKAN_ROOT="${HOME}/.local/nvidia-580.173.02"
export LD_LIBRARY_PATH="${NVIDIA_VULKAN_ROOT}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VK_DRIVER_FILES="${NVIDIA_VULKAN_ROOT}/usr/share/vulkan/icd.d/nvidia_icd.json"
```

PPO 项目名会被官方代码固定为 `MIKASA-Robo-dataset-collectors`；run name 包含任务、seed 和
timestamp。W&B entity 从 `.env` 读取。

### 4.2 5M engineering smoke

```bash
"${MIKASA_PY}" "${PPO_ENTRY}" \
  --env-id InterceptGrabFast-VLA-v0 \
  --total-timesteps 5000000 \
  --track \
  --wandb-entity "${WANDB_ENTITY}"
```

smoke 只检查：能训练完、显存稳定、rollout/update 吞吐正常、loss/return 为有限值、eval 与 W&B
持续写入。无论 smoke 是否碰巧成功，其权重都不作为最终 teacher；正式 run 从头开始。

### 4.3 默认正式 run

```bash
"${MIKASA_PY}" "${PPO_ENTRY}" \
  --env-id InterceptGrabFast-VLA-v0 \
  --total-timesteps 150000000 \
  --track \
  --wandb-entity "${WANDB_ENTITY}"
```

不要覆盖其他参数。通过官方 gate 时 checkpoint 位于：

```text
oracle_checkpoints/ppo_memtasks/state/normalized_dense/InterceptGrabFast-VLA-v0/
  <run>/<timestamp>/final_success_ckpt.pt
```

若跑满仍未通过，只会得到 `final_ckpt.pt`。当前 trainer 没有 periodic model checkpoint；不要把
手工 kill 当成可续训 checkpoint。`--checkpoint` 只加载 model weights，不恢复 optimizer 或
global step。

### 4.4 独立 strict gate

每个 `final_success_ckpt.pt` 都只是候选。对候选执行固定 seeds、deterministic 100 episodes：

```bash
export PPO_CKPT=/absolute/path/to/final_success_ckpt.pt
export PPO_EVAL_DIR="${REMOTE_ROOT}/outputs/mikasa/ppo_teacher_strict_eval"

cd "${REMOTE_ROOT}"
"${MIKASA_PY}" scripts/mikasa/evaluate.py \
  --policy ppo \
  --checkpoint "${PPO_CKPT}" \
  --episodes 100 \
  --output-dir "${PPO_EVAL_DIR}"

"${MIKASA_PY}" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); n=sum(d["successes"]); print(f"{n}/100"); assert n >= 95' \
  "${PPO_EVAL_DIR}/summary.json"
```

canonical seeds 是从 `4242424242` 连续递增的 100 个 seed。只有同时满足以下条件才能进入
Phase 2：

1. 官方 trainer 至少一次 `16/16`，产出 `final_success_ckpt.pt`；
2. 独立 canonical evaluation 至少 `95/100`；
3. checkpoint、W&B run 和 `summary.json` 都已记录到本文。

### 4.5 默认参数失败后的有限调参顺序

不改变 state、reward、success、action、network、seed、rollout length 或 control mode。按以下
顺序单因素调参：

1. `learning_rate=1e-4`，先跑 50M；
2. `learning_rate=6e-4`，先跑 50M；
3. 用 canonical 100-episode SR 排名，平手时比较 mean return；
4. 对更好的 learning rate 尝试 `update_epochs=8`，先跑 50M；
5. 选择严格 eval 最好的配置，从头跑最多 150M。

50M screening 命令是在默认正式命令上增加：

```text
--total-timesteps 50000000 --learning-rate 0.0001
--total-timesteps 50000000 --learning-rate 0.0006
--total-timesteps 50000000 --learning-rate <best_lr> --update-epochs 8
```

每个 run 一旦产生 `final_success_ckpt.pt` 就立即做 100-episode strict eval。禁止降低 0.95 gate，
也禁止为了过 gate 修改 reward。如果所有这些配置都失败，停止 goal，汇总 W&B 曲线、每个配置的
best trainer eval 和 canonical SR；不要带着不合格 teacher 继续采集。

## 5. Phase 2：采集并转换 250/25 dataset

### 5.1 隔离已验收 checkpoint

collector 会自动选择 checkpoint root 下 mtime 最新的 `final_success_ckpt.pt`。为了避免误用其他
候选，把唯一通过 strict gate 的 checkpoint 放入独立 root：

```bash
export STRICT_CKPT_ROOT="${REMOTE_ROOT}/outputs/mikasa/strict_teacher"
mkdir -p "${STRICT_CKPT_ROOT}/oracle_checkpoints/InterceptGrabFast-VLA-v0"
cp "${PPO_CKPT}" \
  "${STRICT_CKPT_ROOT}/oracle_checkpoints/InterceptGrabFast-VLA-v0/final_success_ckpt.pt"
```

### 5.2 收集恰好 275 条成功轨迹

```bash
export DATASET_ROOT="${REMOTE_ROOT}/outputs/mikasa/grabfast_h1_dataset"
export NVIDIA_VULKAN_ROOT="${HOME}/.local/nvidia-580.173.02"
export LD_LIBRARY_PATH="${NVIDIA_VULKAN_ROOT}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VK_DRIVER_FILES="${NVIDIA_VULKAN_ROOT}/usr/share/vulkan/icd.d/nvidia_icd.json"

cd "${REMOTE_ROOT}"
PYTHONPATH="${REMOTE_ROOT}:${REMOTE_ROOT}/third_party/MIKASA-Robo" \
"${MIKASA_PY}" \
  third_party/MIKASA-Robo/mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py \
  --env-id InterceptGrabFast-VLA-v0 \
  --path-to-save-data "${DATASET_ROOT}" \
  --ckpt-dir "${STRICT_CKPT_ROOT}" \
  --num-train-data 275
```

collector 使用 deterministic PPO，同时 step state env 和 RGB env。它只接受整批全部成功的
batch；失败 episode、混合失败 batch 和长度小于 10 的 episode 都不计数。保存的 action 会
clip 到 `[-1,1]`，与 controller 实际收到的 action 一致；轨迹在第一次 success/done 处截断，
最大长度 60。

### 5.3 NPZ audit

输入目录应为：

```text
${DATASET_ROOT}/data_npz/intercept_grab_fast_vla_v0/
```

执行一次完整 audit：

```bash
export NPZ_TASK_DIR="${DATASET_ROOT}/data_npz/intercept_grab_fast_vla_v0"

"${MIKASA_PY}" - "${NPZ_TASK_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
metadata = json.loads((root / "metadata.json").read_text())
files = list(root.glob("train_data_*.npz"))
assert metadata["num_episodes"] == 275
assert len(files) == 275
assert len(set(metadata["episode_seeds"])) == 275
assert all(metadata["success_once"])
assert all(10 <= n <= 60 for n in metadata["episode_lengths"])

transitions = 0
for path in files:
    episode = np.load(path)
    t = episode["action"].shape[0]
    assert episode["action"].shape == (t, 7)
    assert episode["proprio"].shape == (t, 7)
    assert episode["rgb"].shape[0] == t
    assert episode["reward"].shape[0] == t
    assert np.max(np.abs(episode["action"])) <= 1.0
    assert bool(episode["success_once"])
    transitions += t

print({"episodes": len(files), "transitions": transitions})
PY
```

把实际 total transitions、平均/最小/最大 episode length 写入本文。参考官方 GrabFast 250 条
数据约有 12,440 transitions，平均 49.76；本次实际值可以不同。

### 5.4 NPZ -> RLDS -> LeRobot

RLDS builder 只生成一个 `train` split，因此先用 275 episodes 构建一次 RLDS，再用 TFDS split
expression 导出两个 LeRobot dataset：

```bash
cd "${REMOTE_ROOT}/third_party/MIKASA-Robo"

uv sync --project utils/convert_npz_to_rlds/rlds_dataset_builder
uv run --project utils/convert_npz_to_rlds/rlds_dataset_builder \
  python utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py \
  --task InterceptGrabFast-VLA-v0 \
  --repo-root "$PWD" \
  --npz-root "${DATASET_ROOT}/data_npz" \
  --rlds-root "${DATASET_ROOT}/data_rlds"

uv sync --project utils/convert_rlds_to_lerobot

uv run --project utils/convert_rlds_to_lerobot \
  python utils/convert_rlds_to_lerobot/convert_rlds_to_lerobot.py \
  --task InterceptGrabFast-VLA-v0 \
  --repo-root "$PWD" \
  --rlds-root "${DATASET_ROOT}/data_rlds" \
  --lerobot-root "${DATASET_ROOT}/data_lerobot" \
  --repo-id-template 'local/{task}_h1_train' \
  --fps 20 \
  --split 'train[:250]'

uv run --project utils/convert_rlds_to_lerobot \
  python utils/convert_rlds_to_lerobot/convert_rlds_to_lerobot.py \
  --task InterceptGrabFast-VLA-v0 \
  --repo-root "$PWD" \
  --rlds-root "${DATASET_ROOT}/data_rlds" \
  --lerobot-root "${DATASET_ROOT}/data_lerobot" \
  --repo-id-template 'local/{task}_h1_val' \
  --fps 20 \
  --split 'train[250:275]'
```

输出名固定为：

```text
local/intercept_grab_fast_vla_v0_h1_train  # 250 episodes
local/intercept_grab_fast_vla_v0_h1_val    # 25 episodes
```

为 StarVLA 写 modality 和 mixture：

```bash
cd "${REMOTE_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ -d /venv/starvla_rl_games_openvla ]]; then
  conda activate /venv/starvla_rl_games_openvla
else
  conda activate starvla_rl_games_openvla
fi

python scripts/mikasa/prepare_starvla_dataset.py \
  --data-root "${DATASET_ROOT}/data_lerobot" \
  --train-name local/intercept_grab_fast_vla_v0_h1_train \
  --val-name local/intercept_grab_fast_vla_v0_h1_val
```

Phase 2 gate：train/val 分别恰好 250/25 episodes；两个 dataset 都有 `meta/info.json`、
`meta/stats.json`、`meta/modality.json`；mixture JSON 同时引用这两个准确名称。

## 6. Phase 3：单卡 DeepSpeed ZeRO-2 SFT

### 6.1 固定训练预算

- 单卡，`CUDA_VISIBLE_DEVICES=0`，`--num_processes 1`。
- 必须使用 `starVLA/config/deepseeds/deepspeed_zero2.yaml`；不能改成裸 Python/裸
  Accelerator，也不能切 ZeRO-3。
- global batch 始终为 64。
- optimizer steps 始终为 5,000。
- held-out action-loss eval 每 500 steps。
- 训练结束保存 final model `.pt` 和 training state；不额外保留五份周期 checkpoint。
- W&B project：`starvla_mikasa`；entity 从 `.env` 读取。

### 6.2 OOM profile 顺序

先尽量增大真实 per-device batch，并保持 accumulation 1：

| 尝试 | per-device batch | accumulation | global batch |
|---|---:|---:|---:|
| 1 | 64 | 1 | 64 |
| 2 | 32 | 2 | 64 |
| 3 | 16 | 4 | 64 |
| 4 | 8 | 8 | 64 |
| 5 | 4 | 16 | 64 |

一次尝试只有在完成 model load、forward、backward、optimizer step 和第一次日志后才算不 OOM。
OOM 后结束失败进程，确认 GPU memory 已释放，用新的 W&B run id 从 step 0 重启；不要续不完整
checkpoint，也不要把 global batch 降到 64 以下。

### 6.3 训练命令

从 `MICRO_BATCH=64`、`ACC_STEPS=1` 开始；每次 OOM 按上表修改两个变量：

```bash
cd "${REMOTE_ROOT}"
set -a
source .env
set +a

source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ -d /venv/starvla_rl_games_openvla ]]; then
  conda activate /venv/starvla_rl_games_openvla
else
  conda activate starvla_rl_games_openvla
fi

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MICRO_BATCH=64
export ACC_STEPS=1
export TRAIN_NAME=local/intercept_grab_fast_vla_v0_h1_train
export VAL_NAME=local/intercept_grab_fast_vla_v0_h1_val
export VLA_ROOT="${REMOTE_ROOT}/third_party/starVLA"
export VLA_DATA_ROOT="${DATASET_ROOT}/data_lerobot"
export INIT_CKPT="${VLA_ROOT}/playground/Pretrained_models/Qwen3VL-OFT-Bridge-RT-1/checkpoints/steps_5000_pytorch_model.pt"
export RUN_ID="mikasa_grabfast_h1_gb64_mb${MICRO_BATCH}_ga${ACC_STEPS}"

cd "${VLA_ROOT}"
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/MIKASA/train_files/starvla_qwenoft_intercept_grab_fast_h1.yaml \
  --run_id "${RUN_ID}" \
  --wandb_project starvla_mikasa \
  --wandb_entity "${WANDB_ENTITY}" \
  --datasets.vla_data.data_root_dir "${VLA_DATA_ROOT}" \
  --datasets.vla_data.data_mix "${TRAIN_NAME}" \
  --datasets.vla_data.eval_data_mix "${VAL_NAME}" \
  --datasets.vla_data.custom_mixtures_path "${VLA_DATA_ROOT}/mikasa_intercept_grab_fast_h1.json" \
  --datasets.vla_data.per_device_batch_size "${MICRO_BATCH}" \
  --trainer.gradient_accumulation_steps "${ACC_STEPS}" \
  --trainer.max_train_steps 5000 \
  --trainer.profile_timing.enabled true \
  --trainer.pretrained_checkpoint "${INIT_CKPT}" \
  --checkpoint.save_training_state true \
  --checkpoint.save_pt_file true
```

`trainer.eval_interval=500` 是 held-out action loss，不是环境 success rate。
`trainer.profile_timing.enabled=true` 用于让 W&B 记录吞吐；global batch 仍必须由 run ID、resolved
config 和 stdout 共同确认。W&B 至少应记录 train/eval loss、learning rate、真实 global batch、
optimizer step、epoch 和吞吐。

训练结束后必须存在：

```text
third_party/starVLA/results/Checkpoints/${RUN_ID}/config.full.yaml
third_party/starVLA/results/Checkpoints/${RUN_ID}/checkpoints/steps_5000_state/
third_party/starVLA/results/Checkpoints/${RUN_ID}/checkpoints/steps_5000_pytorch_model.pt
```

计算并记录实际 epoch：

```text
actual epochs = 5000 * 64 / actual train transitions
```

### 6.4 canonical zero-delay VLA eval

训练完成后启动 policy server：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ -d /venv/starvla_rl_games_openvla ]]; then
  conda activate /venv/starvla_rl_games_openvla
else
  conda activate starvla_rl_games_openvla
fi
cd "${REMOTE_ROOT}/third_party/starVLA"

export VLA_CKPT="${VLA_ROOT}/results/Checkpoints/${RUN_ID}/checkpoints/steps_5000_pytorch_model.pt"
python deployment/model_server/server_policy.py \
  --ckpt_path "${VLA_CKPT}" \
  --port 10093 \
  --use_bf16
```

另一个 shell 使用 MIKASA env 跑 canonical 50 seeds：

```bash
cd "${REMOTE_ROOT}"
export MIKASA_PY="${REMOTE_ROOT}/third_party/MIKASA-Robo/.venv/bin/python"
export PYTHONPATH="${REMOTE_ROOT}:${REMOTE_ROOT}/third_party/MIKASA-Robo"
export VLA_EVAL_DIR="${REMOTE_ROOT}/outputs/mikasa/vla_h1_zero_delay_eval"
export NVIDIA_VULKAN_ROOT="${HOME}/.local/nvidia-580.173.02"
export LD_LIBRARY_PATH="${NVIDIA_VULKAN_ROOT}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VK_DRIVER_FILES="${NVIDIA_VULKAN_ROOT}/usr/share/vulkan/icd.d/nvidia_icd.json"

"${MIKASA_PY}" scripts/mikasa/evaluate.py \
  --policy vla \
  --host 127.0.0.1 \
  --port 10093 \
  --episodes 50 \
  --output-dir "${VLA_EVAL_DIR}"

"${MIKASA_PY}" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); n=sum(d["successes"]); print(f"{n}/50"); assert n >= 25' \
  "${VLA_EVAL_DIR}/summary.json"
```

Phase 3 gate：5,000 optimizer steps 完成；global batch 被日志确认是 64；final model/state、W&B
run 和 50-episode JSON 均存在；canonical `success_once >= 25/50`。未达到 50% 时如实报告
baseline failure，不把 action-loss eval 冒充环境成功率。

## 7. 当前复现入口

完整的读者路径和跨任务阶段契约见
[`docs/standard_pipeline.md`](../docs/standard_pipeline.md)。本任务的两个入口是：

```bash
bash scripts/tasks/mikasa_intercept_grab_fast/train_small_policy.sh
bash scripts/tasks/mikasa_intercept_grab_fast/train_vla.sh /absolute/path/to/final_success_ckpt.pt
```

第二条命令从显式验收的 teacher checkpoint 开始，完成数据采集/转换、SFT 和
canonical zero-delay eval。底层命令仍保留在上文，便于定位环境或数据问题。

## 8. 实际运行记录

| 项目 | 状态/结果 |
|---|---|
| 本地同步准备 | RLDS builder `7b2f4f38...`、Flappy `cd1c7d5...`、Sample Factory `d056356...` ready；两个 uv lock check、bash syntax、MIKASA tests `3/3` passed |
| 远端 hostname / GPU / driver | `computeinstance-u00mv1dams6a8gc51r`；1× NVIDIA RTX PRO 6000 Blackwell Server Edition 97,887 MiB；driver `580.173.02`；CUDA 13.0；磁盘可用 469 GB |
| MIKASA runtime smoke | passed；Python 3.10.21，Torch `2.7.1+cu128`，capability `(12,0)`，user-space Vulkan `580.173.02`，reset + one step OK |
| StarVLA import/model smoke | passed；Torch `2.11.0+cu128`，DeepSpeed `0.16.9`，Accelerate `1.5.2`，FlashAttention `2.8.3.post1`，capability `(12,0)`；Qwen3-VL-4B base 与 Bridge/RT-1 initialization（含 `steps_5000_pytorch_model.pt`）下载完整 |
| PPO 5M W&B run | passed；`EXIT_STATUS=0`，约 16.4k SPS，显存稳定，末次 16-episode eval `success_once=0.0625`；W&B run [`o67hxo1f`](https://wandb.ai/saberrr-zju/MIKASA-Robo-dataset-collectors/runs/o67hxo1f)；smoke checkpoint 不作 teacher |
| PPO 正式配置与 W&B run | default run；`150M` 上限、seed 123、1024 envs、batch 61,440、无 checkpoint；在 24,576,000 steps 触发官方早停；W&B run [`ek4jtsoc`](https://wandb.ai/saberrr-zju/MIKASA-Robo-dataset-collectors/runs/ek4jtsoc) |
| trainer `16/16` checkpoint | tuned final passed at epoch 201 / 12,288,000 steps；`oracle_checkpoints/ppo_memtasks/state/normalized_dense/InterceptGrabFast-VLA-v0/ppo-mlp-state-dense-intercept-grab-fast-vla-v0__123__state__20260831_203629/20260831_203629/final_success_ckpt.pt` |
| PPO canonical 100-episode SR | tuned final passed `100/100`，mean return `36.819`；fixed seeds `4242424242..4242424341`；`outputs/mikasa/ppo_teacher_strict_eval_tuned_final/summary.json`（default candidate was `72/100`） |
| PPO tuning screens | `learning_rate=1e-4` candidate at 15,360,000 steps，canonical `97/100`、mean return `32.400`，W&B [`4jyud8v7`](https://wandb.ai/saberrr-zju/MIKASA-Robo-dataset-collectors/runs/4jyud8v7)，summary `outputs/mikasa/ppo_teacher_strict_eval_lr1e4/summary.json`；`6e-4` candidate at 27,648,000 steps，canonical `91/100`、mean return `27.547`，W&B [`xwn3xyk2`](https://wandb.ai/saberrr-zju/MIKASA-Robo-dataset-collectors/runs/xwn3xyk2)，summary `outputs/mikasa/ppo_teacher_strict_eval_lr6e4/summary.json`；`1e-4` + `update_epochs=8` candidate at 12,288,000 steps，canonical `100/100`、mean return `36.819`，W&B [`7c89wxd8`](https://wandb.ai/saberrr-zju/MIKASA-Robo-dataset-collectors/runs/7c89wxd8)，summary `outputs/mikasa/ppo_teacher_strict_eval_lr1e4_ep8/summary.json`；best config `1e-4` + 8 epochs |
| PPO tuned final run | passed；from scratch，`learning_rate=1e-4`，`update_epochs=8`，12,288,000 steps early stop；W&B [`a789mz9i`](https://wandb.ai/saberrr-zju/MIKASA-Robo-dataset-collectors/runs/a789mz9i) |
| dataset episodes/transitions/length | NPZ audit passed；275 episodes，9,859 transitions，mean/min/max length `35.851/24/57`，275 unique seeds，all success，actions clipped to `[-1,1]` |
| LeRobot train/val counts | Phase 2 gate passed；train 250 episodes / 8,911 frames，val 25 / 948 frames，20 FPS；两套 `info/stats/modality.json` 与 `mikasa_intercept_grab_fast_h1.json` mixture verified |
| SFT microbatch/accumulation/global batch | OOM profile selected `8/8/64` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；mb64/32 true forward OOM，mb16 later OOM at step 88；selected mb8 run passed beyond that point |
| SFT W&B run / final checkpoint | passed；run id `mikasa_grabfast_h1_gb64_mb8_ga8_expandable`；W&B [`3qt4k7nc`](https://wandb.ai/saberrr-zju/starvla_mikasa/runs/3qt4k7nc) 已同步；5,000 steps，actual epochs `35.91`；resolved config 与 stdout 共同确认 `per_device_batch_size=8`、accumulation `8`、total/global batch `64` |
| SFT completed runtime | `5000/5000`，elapsed `7:36:30`，mean `5.48 s/step`；训练日志 `logs/mikasa/sft_mb8_ga8_expandable.log` 以 `EXIT_STATUS=0` 结束，无 OOM、Traceback 或 ERROR |
| SFT held-out action loss | step `500..5000` 每 500 steps 均完成 800-sample action-loss 与 classification eval；train loss `0.072997/0.063220/0.045567/0.040993/0.037294/0.026909/0.023618/0.018658/0.012922/0.008939`；eval loss `0.115920/0.111115/0.105102/0.092906/0.095080/0.091598/0.094032/0.098793/0.095458/0.093039`；所有值 finite |
| SFT final artifacts | passed；`third_party/starVLA/results/Checkpoints/mikasa_grabfast_h1_gb64_mb8_ga8_expandable/config.full.yaml`、同目录 `checkpoints/steps_5000_state/` 与 `checkpoints/steps_5000_pytorch_model.pt`（9,785,060,075 bytes）均存在 |
| VLA canonical 50-episode SR | passed `27/50`（`54%`），mean return `22.326`；fixed seeds `4242424242..4242424291`，returns all finite；summary `outputs/mikasa/vla_h1_zero_delay_eval/summary.json`；部署端在既有 MIKASA data registry 注册 checkpoint 的 train mixture 后，strict follow-through 写出 `FOLLOWTHROUGH_EXIT_STATUS=0` |

## 9. 本轮明确 deferred

- fixed/realtime latency sweep、latency profile 和 delayed PPO teacher；
- H8/K8 action chunk、future action labels 或 chunk-aware loss；
- VLA state、delay、queue 或 applied-action 输入；
- VLA online RL；
- upstream OpenVLA-7B、MuVLA 或其他 backbone；
- InterceptMedium-to-GrabFast transfer reproduction。

这些工作只有在本手册的 zero-delay baseline 全部通过后再单独立项。
