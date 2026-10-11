# 环境配置快速指南

本页面向 Linux x86_64 + NVIDIA GPU，从仓库根目录执行命令。覆盖共享 benchmark、Balance/MIKASA 模拟器、StarVLA 和 MIKASA 数据转换环境。各任务的输入、预算与命令见[标准流程](standard_pipeline.md)。

## 1. 获取代码与主机准备

```bash
git lfs install
git clone --filter=blob:none https://github.com/ZihanWang314/latency-sensitive-bench.git
cd latency-sensitive-bench
export BENCH_ROOT="$PWD"
nvidia-smi
```

第三方源码随主仓库检出，版本由主仓库 commit 固定。直接修改 `third_party/` 中的源码并在主仓库提交。上游来源、资源裁剪规则和更新步骤见 [第三方源码维护](../third_party/README.md)。

主机需有 Git、Git LFS、编译工具、可用的 NVIDIA 驱动，以及支持离屏渲染的 NVIDIA EGL/Vulkan runtime。`nvidia-smi` 成功只说明 CUDA 驱动可见，不能代替渲染检查。MIKASA 还需要与主机驱动匹配的 Vulkan 用户态库。安装后执行下方的模拟器检查。

`install.sh` 可安装 Miniconda 和 uv，并安装环境内 CUDA Toolkit；不会安装主机驱动或系统图形库。共享与 StarVLA 栈固定为 Python 3.10、CUDA Toolkit 12.8.1、Torch 2.11.0+cu128。

## 2. 环境对应关系

| 用途 | 环境 / Python | 安装入口 |
| --- | --- | --- |
| benchmark、Balance、常规游戏、共享数据准备 | Conda `latency`，3.10 | 根目录 `install.sh`；Balance 补充见下文 |
| 四种 StarVLA 模型训练与独立推理进程 | Conda `starvla_rl_games_openvla`，3.10 | `install.sh --model openvla --tier dev` |
| 原有 RL-games 模型独立栈 | Conda `starvla_rl_games_{pi0,pi05,gr00t,wan_oft}`，3.10 | 对应 `--model`，或 `--model all` |
| MIKASA / ManiSkill | `third_party/MIKASA-Robo/.venv`，3.10 | `scripts/mikasa/install.sh` |
| MIKASA H8 / DAgger 的 LeRobot 数据打包 | converter 目录内 `.venv`，3.10 | 本页第 4 节 |

`--model all` 只表示所有模型环境加共享 `latency`，不包含 MIKASA 模拟器和 LeRobot 数据打包环境。`qwenoft/qwenpi_v3/qwengr00t/wanoft` 是统一 StarVLA 流水线模型名，四者默认共用 `starvla_rl_games_openvla`。

## 3. 共享环境、Balance 与模型环境

```bash
bash install.sh --model openvla --tier dev
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate latency
python -m pip install -e '.[dev,sonic]'
python -m pip install --no-deps -e third_party/humanoid-bench
conda install -n latency -c conda-forge 'ffmpeg=7' -y
export MUJOCO_GL=egl
python -c 'import torch, humanoid_bench, mujoco, onnxruntime; print(torch.__version__, torch.cuda.is_available())'
conda deactivate
```

Balance 需要这里的 HumanoidBench 与 SONIC 场景资产依赖；`--model gr00t` 或 `--model all` 会自动安装它们。训练需要 `--tier dev` 中的 DeepSpeed 和数据依赖。

```bash
conda activate starvla_rl_games_openvla
python -c 'import torch, transformers, diffusers, deepspeed, accelerate, flash_attn, pyarrow; print(torch.__version__, transformers.__version__, diffusers.__version__)'
conda deactivate
```

当前 StarVLA common requirements 已包含 Diffusers、Qwen 工具与动作头依赖，四模型不需要分别装四套环境。代码位于 `third_party/starVLA`，无需额外的 sibling checkout。

需要原有全部模型栈时执行：

```bash
bash install.sh --model all --tier dev
```

## 4. MIKASA 与数据打包环境

```bash
cd "$BENCH_ROOT"
bash scripts/mikasa/install.sh
export MIKASA_PY="$BENCH_ROOT/third_party/MIKASA-Robo/.venv/bin/python"
"$MIKASA_PY" -c 'import torch, mani_skill; print(torch.__version__, torch.cuda.get_device_capability())'
```

MIKASA 锁定 Torch 2.2.1。RTX PRO 6000 等 `(12, 0)` GPU 的已验证运行方式是安装后覆盖为 CUDA 12.8 的 Torch 2.7.1：

```bash
uv pip install --python "$MIKASA_PY" \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1
```

覆盖后直接调用 `$MIKASA_PY`，再次 `uv sync` / `uv run` 会按锁文件恢复旧 Torch。MIKASA `train_small_policy.sh` 会再次调用安装脚本；在覆盖 Torch 的 Blackwell 主机上，优先使用 `train_qwenoft.sh` 等共享启动器，它们不会重新同步模拟器环境。不要在该环境安装根项目的完整依赖，它们会覆盖 MIKASA 的 NumPy/Torch 约束；任务启动器通过 `PYTHONPATH` 引入 benchmark。

当前共享 H1 使用已发布的 LeRobot 数据。
当前 conditioned H8 和 DAgger 将采集的 NPZ 直接打包成 LeRobot，只需要下面的打包环境。
环境目录沿用 `convert_rlds_to_lerobot` 名称，实际运行的是
`latency_bench.integrations.starvla.mikasa_h8_export`。
`scripts/mikasa/install.sh` 不会安装该打包环境：

```bash
uv sync --project third_party/MIKASA-Robo/utils/convert_rlds_to_lerobot --python 3.10
third_party/MIKASA-Robo/utils/convert_rlds_to_lerobot/.venv/bin/python \
  -c 'import lerobot; print(lerobot.__file__)'
```

<a id="mikasa-runtime-check"></a>
### 模拟器检查

在仓库根目录执行。检查当前设备能否完成 reset 和一次 step：

```bash
PYTHONPATH="$BENCH_ROOT:$BENCH_ROOT/third_party/MIKASA-Robo" \
"$MIKASA_PY" - <<'PYTHON'
import gymnasium as gym
import torch
import mikasa_robo_suite.vla.memory_envs

env = gym.make(
    "InterceptGrabFast-VLA-v0",
    num_envs=1,
    obs_mode="state",
    control_mode="pd_ee_delta_pose",
    render_mode="all",
    sim_backend="gpu",
    reward_mode="normalized_dense",
)
env.reset(seed=123)
env.step(torch.zeros((1, 7), device=env.unwrapped.device))
print("runtime smoke OK", env.action_space.shape)
env.close()
PYTHON
```

reset、step 和 GPU kernel 均须成功，动作最后一维应为 7。
此检查不证明训练收敛或视觉推理性能。


## 5. 认证与首次端到端检查

```bash
conda activate latency
hf auth login
wandb login
export MUJOCO_GL=egl
```

也可通过 `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` 和 `WANDB_API_KEY` 设置凭据。脚本不会自动加载根目录 `.env`。共享流水线会下载教师、backbone 和可复用数据。部分来源使用默认分支，精确复现须提供实验记录中的固定版本。正式训练使用 `--no-publish` 可保持本地运行；自动发布需要目标仓库写权限。发布目标见 [artifacts.md](artifacts.md)。发布的模型和数据会记录实际解析到的 revision。

先验证命令入口，再运行一组真实 smoke：

```bash
python -m scripts.starvla.train_task --help
CUDA_VISIBLE_DEVICES=0,1 bash scripts/tasks/mikasa_intercept_grab_fast/train_qwenoft.sh \
  --smoke --run-id setup-smoke
```

默认训练使用两张 GPU，评测时模拟器使用 `cuda:0`，推理使用 `cuda:1`。已有 2×96 GB GPU 验收记录；不是所有模型都能在单张卡运行。环境名称与表格一致时自动查找解释器；自定义路径使用 `--vla-python /absolute/path/bin/python --task-python /absolute/path/bin/python`。shell 中仍需能找到 `conda`。

Balance 的输入要求见 [Balance cookbook](humanoidbench/cookbook.md)；不能只替换任务目录。模型脚本后缀可替换为 `qwenoft`、`qwenpi_v3`、`qwengr00t`、`wanoft`。smoke 会执行两次 bootstrap 更新、保存并重载模型、两集 L0 评测。输出位于 `outputs/starvla/smoke/`，不会上传。

将 `--smoke` 替换为 `--no-publish` 可运行完整预算并保留本地产物。省略 `--no-publish` 会自动发布。详细命令和预算见[标准流程](standard_pipeline.md)。

MIKASA 的 `train_{qwenoft,qwenpi_v3,qwengr00t,wanoft}.sh` 调用 `scripts/starvla/train_task.py`；通过 `--task-python` 显式指向对应模拟器环境。

Atari / ViZDoom / Flappy 的依赖由根安装脚本处理；无需把 Isaac 或 ManiSkill 安装进去。
