# 安装与环境检查

安装目标为 Linux x86_64、Python 3.10、CUDA Toolkit 12.8.1 和
Torch 2.11.0+cu128，并安装 FFmpeg 7。主机需要 Git、Git LFS、C++ 编译工具和 NVIDIA 驱动。
图像模拟器还需要与驱动匹配的 NVIDIA EGL/Vulkan 用户态库。
安装脚本安装环境内 CUDA Toolkit，不安装主机驱动和系统图形库。

## 获取代码和安装基础环境

```bash
git lfs install
git clone --filter=blob:none https://github.com/ZihanWang314/latency-sensitive-bench.git
cd latency-sensitive-bench
nvidia-smi
bash install.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate latency
```

默认安装公共运行、图像、数据采集、Parquet、转换、profiling 依赖和
仓库内维护的 Sample Factory 源码。第三方源码由主仓库 revision 固定，
无需额外 clone 上游框架。维护约定见[第三方源码说明](../third_party/README.md)。

安装开发测试依赖时使用 `bash install.sh --tier dev`。
安装脚本最后执行依赖导入和基础 GPU 运行检查，不启动训练。
`--skip-validate` 跳过最终基础与模型检查。

## 按 task 安装模拟器

```bash
bash install.sh --env flappy
bash install.sh --env air_raid,ant,mikasa_intercept_grab_fast
bash install.sh --env all --tier dev
```

`--env` 接受逗号分隔的任务名称。同类任务共用的依赖只安装一次。
`all` 安装全部内置任务所需的模拟器。

| 任务名称 | 安装内容 |
| --- | --- |
| `flappy` | 固定 Flappy 源码、pygame 和 Flappy CUDA 扩展 |
| `air_raid,demon_attack,asterix,atlantis` | ALE、Gymnasium Atari 注册、ROM 和 Atari CUDA 扩展 |
| `deadly_corridor` | ViZDoom |
| `ant,half_cheetah,hopper,humanoid,inverted_pendulum,swimmer,walker2d` | MuJoCo |
| `walker2d_continuous,walker2d_continuous_torque` | MuJoCo，保留已有连续动作入口 |
| `pendulum_discrete,pendulum_continuous` | 使用基础环境已有的 Gymnasium 和 pygame |
| `lunar_lander_discrete` | Box2D 和构建工具 |
| `g1_sonic_balance_simple` | MuJoCo、dm-control 和固定 HumanoidBench 源码 |
| `mikasa_intercept_grab_fast` | 固定 MIKASA 源码、ManiSkill 和 SAPIEN |

Atari 安装沿用 `AutoROM --accept-license` 下载 ROM。
Balance 的场景资产通过已有入口准备：

```bash
conda activate latency
python -m latency_bench.envs.tasks.humanoidbench.assets \
  --cache-dir results/humanoidbench_g1_sonic_assets
```

MIKASA 安装到 `latency`，使用统一的 Torch 和 NumPy 版本。
激活 `latency` 后，`bash scripts/mikasa/install.sh` 也可以单独补装 MIKASA。
PPO、采集、转换和模拟器 eval 使用该解释器。
MIKASA 启动器默认使用当前 `python`，需要指定路径时设置 `TASK_PY`。

## 安装 StarVLA

```bash
bash install.sh --env flappy --model openvla --tier dev
```

这条命令同时安装 `latency` 和 `starvla_rl_games_openvla`，并为两者安装
Flappy。训练期间的模型 eval 在模型解释器内运行，也需要所选模拟器。

`openvla` 是沿用的安装环境名称。当前 `qwenoft`、`qwenpi_v3`、
`qwengr00t` 和 `wanoft` 共用 `starvla_rl_games_openvla`。
原有独立安装目标 `pi0,pi05,gr00t,wan_oft` 和 `--model all` 保留。
`--model all` 创建全部已有模型环境，模拟器仍由 `--env` 选择。
`--tier dev` 安装训练所需的 DeepSpeed 等依赖。
Flash Attention 使用已有固定 wheel，`--skip-flash-attn` 可跳过安装及检查。

基础环境和模型环境都安装 LeRobot 0.4.4，供数据导出使用。
项目先安装数据导出需要的依赖，再单独安装 LeRobot：

```bash
python -m pip install --no-deps 'lerobot==0.4.4'
```

LeRobot 使用 `--no-deps` 单独安装。Gymnasium 0.29.1、Torch 2.11.0 和其他数据依赖由项目管理。
LeRobot 的上游版本声明与这套版本存在冲突，因此 `pip check` 会报告这些冲突。
安装验收以导入、真实数据导出和训练加载检查为准。
MIKASA H8 打包可把 `--lerobot-python` 指向 `latency` 的 Python。

## 检查与运行

```bash
conda activate latency
python -m scripts.install.validate base
python -m latency_bench eval --help
python -m latency_bench collect --help
python -m latency_bench convert --help
python -m latency_bench profile --help
```

需要获取 HF artifact 或记录 W&B 时执行 `hf auth login`、`wandb login`。
也可以设置 `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` 和 `WANDB_API_KEY`。
根安装脚本不自动加载 `.env`。

训练与 eval 命令见[标准流程](standard_pipeline.md)、
[Balance cookbook](humanoidbench/cookbook.md)和[MIKASA cookbook](mikasa/cookbook.md)。
任务启动器支持显式 `--task-python` 和 `--vla-python`。
共享 MIKASA 训练入口默认查找 Conda `latency` 和 `starvla_rl_games_openvla`。
使用现有 `--smoke` 执行短训练时，模型会更新参数、保存并重载 checkpoint，
再执行 eval。短测结果只验证运行流程，完整训练预算和结果协议由任务配方定义。
