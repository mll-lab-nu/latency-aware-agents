# Install LAGEN

Choose the installation that matches your task. Run checkout commands from the repository root.
LAGEN provides evaluation, collection, conversion, and profiling through one command line interface.
See the [standard pipeline](workflow.md) for the next step.

## Get the source

Install Git and Git LFS. Clone the repository:

```bash
git lfs install
git clone --filter=blob:none https://github.com/ZihanWang314/latency-sensitive-bench.git
cd latency-sensitive-bench
```

The checkout contains the pinned third-party sources. The repository revision identifies their versions.
See [third-party sources](../third_party/README.md) for upstream references.

## Use a CPU environment or your own components

The [MountainCar example](guides/custom_components.md) uses a CPU environment and a constant policy.
It does not require an NVIDIA GPU or a reference training framework.
Install the Python package in a Python 3.10 environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[runtime]"
```

For LeRobot conversion, install the converter separately:

```bash
python -m pip install --no-deps 'lerobot==0.4.4'
```

Install `.[runtime,data]` for the data dependencies used by this converter.
LeRobot's upstream dependency constraints conflict with the project's pinned Gymnasium and Torch versions.
Therefore, `pip check` reports these conflicts.
Use the installation validation results to assess the supported combination.

## Install the reference GPU environment

Conda must already be installed and available in your shell.

The reference installer targets Linux x86_64, Python 3.10, CUDA Toolkit 12.8.1, Torch 2.11.0+cu128, and FFmpeg 7.
Install C++ build tools and an NVIDIA driver on the host.
GPU renderers also need NVIDIA EGL or Vulkan libraries that match the driver.
The installer supplies environment dependencies. It does not install host drivers or system graphics libraries.

```bash
nvidia-smi
bash install.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate latency
```

The default installs runtime, image, data, profiling, and pinned Sample Factory dependencies.
Use `--tier dev` to add development dependencies.
The installer checks imports and basic CUDA execution. It does not train a model.
Use `--skip-validate` to skip its final checks.

## Select simulators

Pass concrete task names to `--env`:

```bash
bash install.sh --env flappy
bash install.sh --env air_raid,ant,mikasa_intercept_grab_fast
bash install.sh --env all --tier dev
```

The installer installs shared simulator dependencies once per environment.
Omitting `--env` installs only the base environment.

| Task names | Simulator dependencies |
| --- | --- |
| `flappy` | Pinned Flappy source, pygame, Flappy CUDA extension |
| `air_raid,demon_attack,asterix,atlantis` | ALE, Atari registration, ROMs, Atari CUDA extension |
| `deadly_corridor` | ViZDoom |
| `ant,half_cheetah,hopper,humanoid,inverted_pendulum,swimmer,walker2d` | MuJoCo |
| `walker2d_continuous,walker2d_continuous_torque` | MuJoCo |
| `pendulum_discrete,pendulum_continuous` | Base Gymnasium and pygame dependencies |
| `lunar_lander_discrete` | Box2D and build tools |
| `g1_sonic_balance_simple` | MuJoCo, dm-control, pinned HumanoidBench source |
| `mikasa_intercept_grab_fast` | Pinned MIKASA source, ManiSkill, SAPIEN |

Atari installation runs `AutoROM --accept-license` to download ROMs.
Balance also requires scene assets:

```bash
conda activate latency
python -m latency_bench.envs.tasks.humanoidbench.assets \
  --cache-dir results/humanoidbench_g1_sonic_assets
```

MIKASA uses the `latency` interpreter with the shared Torch and NumPy versions.
Its PPO, collection, conversion, and simulator evaluation commands use that interpreter.
To add MIKASA to an active environment, run:

```bash
conda activate latency
bash scripts/mikasa/install.sh
```

Task launchers use the active interpreter by default. Set `TASK_PY` when a launcher needs an explicit interpreter.

## Select a model environment

Install the simulator in the model environment as well when model evaluation runs there:

```bash
bash install.sh --env mikasa_intercept_grab_fast --model openvla --tier dev
```

This command creates `latency` and `starvla_rl_games_openvla` with MIKASA dependencies.
`openvla` is the installation target name for the shared StarVLA environment.
QwenOFT, QwenPI v3, QwenGR00T, and WanOFT use that environment.
`--tier dev` includes training dependencies such as DeepSpeed.

The installer also accepts `pi0`, `pi05`, `gr00t`, `wan_oft`, and `--model all` for the original model stacks.
`--model all` creates those model environments. Select their simulators through `--env`.
`--skip-flash-attn` skips the pinned Flash Attention wheel and its check.

Reference environments include LeRobot 0.4.4 installed with `--no-deps`.
MIKASA H8 conversion can use the `latency` interpreter as `--lerobot-python`.

## Check the installation

For the reference GPU environment, run its existing validation command:

```bash
conda activate latency
python -m scripts.install.validate base
```

Inspect the public commands in either installation:

```bash
python -m latency_bench eval --help
python -m latency_bench collect --help
python -m latency_bench convert --help
python -m latency_bench profile --help
```

Follow the [CPU example](guides/custom_components.md) to exercise your own components.
Use the [MIKASA cookbook](cookbooks/mikasa.md) or [Balance cookbook](cookbooks/balance.md) for reference tasks.
A short training run checks updates, checkpoint loading, and evaluation. It does not establish model quality.

## Access remote inputs and optional logs

Run `hf auth login` only when the selected input artifact requires authentication.
You can also set `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`.
The installer does not load `.env`.
Use the [artifact guide](artifacts.md) to select model and dataset revisions.

Training and experiment commands save local outputs.
Task launchers disable W&B by default.
To enable it in a task launcher, pass `--wandb-project YOUR_PROJECT` and optionally `--wandb-entity YOUR_TEAM`.
Authenticate with `wandb login` or `WANDB_API_KEY` before that run.
For Sample Factory recipes, set `logging.wandb_project` and your optional `logging.wandb_entity` in a local overlay.
