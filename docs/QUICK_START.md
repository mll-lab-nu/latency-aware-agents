# 环境配置快速指南

本页面向 Linux x86_64 + NVIDIA GPU，从仓库根目录执行命令。覆盖共享 benchmark、四任务模拟器、StarVLA 和 MIKASA 数据转换环境。四任务 × 四模型的实际 smoke 记录见 [SCALE_MODELS.md](../program/SCALE_MODELS.md#smoke-acceptance--acceptance-v1)。

## 1. 获取代码与主机准备

```bash
git lfs install
git clone --filter=blob:none https://github.com/ZihanWang314/latency-sensitive-bench.git
cd latency-sensitive-bench
git submodule update --init --depth 1 --jobs 8
export BENCH_ROOT="$PWD"
nvidia-smi
```

根仓库保留完整 commit/tree 历史，旧版本文件内容按需下载；父仓库记录的 8 个固定
顶层 submodule commit 以 depth 1 检出，不递归初始化第三方仓库内部的任务专用依赖。
需要修改某个顶层 submodule 并查看其完整历史时，执行
`git -C third_party/<name> fetch --unshallow`。HAIC、Isaaclab_Parkour 的 submodule URL
使用 GitHub SSH，需要对应仓库的访问权限及 SSH key。

已有 checkout 更新子模块使用 `git submodule update --init`，新初始化的顶层子模块会
遵循 `.gitmodules` 中的浅克隆设置；根目录安装脚本也会执行此命令。该设置不会清理已经下载
的完整子模块历史。需要回收已有 checkout 的磁盘空间时，先保存子模块中的修改，再重新 clone，
不要直接删除 `.git/modules`。

主机需有 Git、Git LFS、编译工具、可用的 NVIDIA 驱动，以及支持离屏渲染的 NVIDIA EGL/Vulkan runtime。`nvidia-smi` 成功只说明 CUDA 驱动可见，不能代替渲染检查。Isaac Sim 首次启动还需要下载资源；接受 NVIDIA EULA 后设置 `OMNI_KIT_ACCEPT_EULA=YES`。MIKASA 的 Vulkan 主机诊断见 [MIKASA §3.4](../program/MIKASA.md#34-mikasa-环境)，其中特定驱动版本的修复是历史服务器操作，不能直接套用到另一台机器。

`install.sh` 可安装 Miniconda 和 uv，并安装环境内 CUDA Toolkit；不会安装主机驱动或系统图形库。共享与 StarVLA 栈固定为 Python 3.10、CUDA Toolkit 12.8.1、Torch 2.11.0+cu128。

## 2. 环境对应关系

| 用途 | 环境 / Python | 安装入口 |
| --- | --- | --- |
| benchmark、Balance、常规游戏、共享数据准备 | Conda `latency`，3.10 | 根目录 `install.sh`；Balance 补充见下文 |
| 四个 H1 模型训练与独立推理进程 | Conda `starvla_rl_games_openvla`，3.10 | `install.sh --model openvla --tier dev` |
| 原有 RL-games 模型独立栈 | Conda `starvla_rl_games_{pi0,pi05,gr00t,wan_oft}`，3.10 | 对应 `--model`，或 `--model all` |
| MIKASA / ManiSkill | `third_party/MIKASA-Robo/.venv`，3.10 | `scripts/mikasa/install.sh` |
| MIKASA NPZ → RLDS | builder 目录内 `.venv`，3.9 | 本页第 4 节 |
| MIKASA RLDS → LeRobot | converter 目录内 `.venv`，3.10 | 本页第 4 节 |
| Extreme Parkour | Conda `extreme-parkour`，3.11 | Isaac Sim 5.0.0 + 仓库 IsaacLab v2.2.0 |
| HAIC | Conda `haic`，3.11 | Isaac Sim 5.1.0 + **单独**的 IsaacLab v2.3.2 checkout |

`--model all` 只表示所有模型环境加共享 `latency`，不包含后三类模拟器或 MIKASA 转换环境。`qwenoft/qwenpi_v3/qwengr00t/wanoft` 是统一 StarVLA 流水线模型名，四者默认共用 `starvla_rl_games_openvla`。

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

Balance 需要这里的 HumanoidBench/SONIC 依赖；`--model gr00t` 或 `--model all` 会自动安装它们。训练需要 `--tier dev` 中的 DeepSpeed 和数据依赖。

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

## 4. MIKASA 与两个转换环境

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

覆盖后直接调用 `$MIKASA_PY`，再次 `uv sync` / `uv run` 会按锁文件恢复旧 Torch。原有 MIKASA `train_small_policy.sh` / `train_vla.sh` 会再次调用安装脚本；在覆盖 Torch 的 Blackwell 主机上，使用 runbook 的直接 Python 分阶段命令。新的四模型 H1 启动器不会重新同步模拟器环境。不要在该环境安装根项目的完整依赖，它们会覆盖 MIKASA 的 NumPy/Torch 约束；任务启动器通过 `PYTHONPATH` 引入 benchmark。

需要自己采集 MIKASA 数据时，另外安装这两个环境；`scripts/mikasa/install.sh` 不会安装它们：

```bash
uv sync --project third_party/MIKASA-Robo/utils/convert_npz_to_rlds/rlds_dataset_builder --python 3.9
uv sync --project third_party/MIKASA-Robo/utils/convert_rlds_to_lerobot --python 3.10
third_party/MIKASA-Robo/utils/convert_npz_to_rlds/rlds_dataset_builder/.venv/bin/python \
  -c 'import tensorflow, tensorflow_datasets; print(tensorflow.__version__)'
third_party/MIKASA-Robo/utils/convert_rlds_to_lerobot/.venv/bin/python \
  -c 'import lerobot, tensorflow; print(tensorflow.__version__)'
```

只复用已发布的 H1 bootstrap 数据时不会经过 NPZ/RLDS 转换。完整模拟器 reset/step 检查见 [MIKASA §3.4](../program/MIKASA.md#34-mikasa-环境)。

## 5. Extreme Parkour

```bash
cd "$BENCH_ROOT"
conda create -n extreme-parkour python=3.11 -y
conda activate extreme-parkour
python -m pip install --upgrade pip
python -m pip install setuptools==80.9.0
python -m pip install --no-build-isolation flatdict==4.0.1
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'isaacsim[all,extscache]==5.0.0' --extra-index-url https://pypi.nvidia.com
export OMNI_KIT_ACCEPT_EULA=YES
(cd third_party/IsaacLab && TERM=xterm ./isaaclab.sh -i rsl_rl)
python -m pip install --no-build-isolation --no-deps -e third_party/Isaaclab_Parkour
python -m pip install --no-build-isolation -e third_party/Isaaclab_Parkour/parkour_tasks
conda install -n extreme-parkour -c conda-forge libglu=9.0.3 -y
python -m pip install click==8.1.7 psutil==5.9.8 typing_extensions==4.12.2
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name())'
conda deactivate
```

保持仓库记录的 `third_party/IsaacLab` v2.2.0。运行命令、GPU preflight 和 teacher 配置见 [Parkour runbook](../program/EXTREME_PARKOUR.md#6-preflight-和-smoke)。

## 6. HAIC

HAIC 使用另一个 Isaac Sim/Lab 版本。将 Lab v2.3.2 放在独立目录，避免切换 Parkour 已安装的 editable 源码：

```bash
cd "$BENCH_ROOT"
export HAIC_LAB_ROOT="$(dirname "$BENCH_ROOT")/IsaacLab-v2.3.2"
git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git "$HAIC_LAB_ROOT"
conda create -n haic python=3.11 -y
conda activate haic
python -m pip install --upgrade pip
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
export OMNI_KIT_ACCEPT_EULA=YES
(cd "$HAIC_LAB_ROOT" && TERM=xterm ./isaaclab.sh -i none)
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e third_party/HAIC
python -c 'import torch, torchrl, tensordict, active_adaptation; print(torch.__version__)'
conda deactivate
```

HAIC `setup.py` 固定 Torch 2.7.0、TorchRL 0.7.0、TensorDict 0.7.0；先选择与本机已验证环境一致的 cu128 wheel，再安装 HAIC。资产清单见 [HAIC assets](../third_party/HAIC/active_adaptation/assets/README.md)，任务启动与 runtime smoke 见 [HAIC runbook](../program/HAIC.md#phase-0预检与-runtime-smoke)。H1 流水线会在 `haic` 环境导出 frozen actor，在 StarVLA 环境执行训练。

## 7. 认证与首次端到端检查

```bash
conda activate latency
hf auth login
wandb login
export MUJOCO_GL=egl
export OMNI_KIT_ACCEPT_EULA=YES
```

也可通过 `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` 和 `WANDB_API_KEY` 设置凭据。脚本不会自动加载根目录 `.env`。H1 流水线从各模型和数据仓库的默认分支下载小策略、backbone 和可复用数据；正式运行还需要组织模型仓库、数据集仓库的写权限，发布目标见 [artifacts.md](artifacts.md)。发布的模型和数据会记录实际解析到的 revision。

先验证命令入口，再运行一组真实 smoke：

```bash
python -m scripts.starvla.train_task --help
CUDA_VISIBLE_DEVICES=0,1 bash scripts/tasks/humanoidbench_balance_simple/train_qwenoft.sh \
  --smoke --run-id setup-smoke
```

默认训练使用两张 GPU，评测时模拟器使用 `cuda:0`，推理使用 `cuda:1`。已有 2×96 GB GPU 验收记录；不是所有模型都能在单张卡运行。环境名称与表格一致时自动查找解释器；自定义路径使用 `--vla-python /absolute/path/bin/python --task-python /absolute/path/bin/python`。shell 中仍需能找到 `conda`。

其余任务将路径替换为 `extreme_parkour_go2`、`mikasa_intercept_grab_fast`、`haic_pull_cart`；模型脚本后缀可替换为 `qwenoft`、`qwenpi_v3`、`qwengr00t`、`wanoft`。smoke 会执行两次 bootstrap 更新、保存并重载模型、两集 L0 评测；Parkour/HAIC 另跑一个小 DAgger round。输出位于 `outputs/starvla/smoke/`，不会上传。

去掉 `--smoke` 即启动完整预算并自动发布数据和最终模型，详细命令和预算见 [SCALE_MODELS.md](../program/SCALE_MODELS.md)。

四任务的 `train_{qwenoft,qwenpi_v3,qwengr00t,wanoft}.sh` 都调用 `scripts/starvla/train_task.py`；通过 `--task-python` 显式指向对应模拟器环境。

可选 MyoChallenge 任务，在实际执行该任务的 benchmark 环境安装 `python -m pip install -e '.[myochallenge]'`。原有 Atari / ViZDoom / Flappy 的依赖由根安装脚本处理；无需把 Isaac 或 ManiSkill 安装进去。
