# MIKASA / QwenOFT realtime profile 工作记录

> 2026-09-14 核查：当前产物、具体评测数值和未核验事项见 [PROGRESS](PROGRESS.md)；下文旧运行状态按原日期阅读。

> 更新时间：2026-09-04
>
> workload：`mikasa_intercept_grab_fast`
>
> model：`qwenoft`，即 StarVLA Qwen3-VL-4B + OFT action head，不是 upstream OpenVLA-7B
>
> 状态：2× Blackwell 实机的 5 sessions × 100 episodes 已完成、校验并发布

## 1. 当前结论

MIKASA `InterceptGrabFast-VLA-v0` 的第一份 canonical realtime profile 已完成：

```text
PROFILE_REF=qwenoft/1x-rpbzzz6/mikasa_intercept_grab_fast/instance_e906b84a82f9e086
RUN_ID=20260904T073201011403Z
```

完整 run 共 500 episodes、30,000 observation attempts；16,473 个 request 被单个 inference
worker 接收，13,527 个因 worker busy 被 capacity drop。profile 已发布到 Hugging Face dataset
`latency-sensitive-bench/profiles`，远端同时存在 94 个 raw-run 文件。

这次 realtime rollout 的 task success 是 `0/500`，mean return `0.35156`，每个 episode 都跑满
60 steps。作为对照，同一训练模型的 canonical zero-delay 结果是 `27/50` success（54%，见
[MIKASA.md](MIKASA.md)）。这不影响 latency samples 的定义或完整性，但说明约 100 ms 的单 worker
部署延迟和 45.09% capacity drop 已经足以让当前 policy 完全失效；后续 sim2real 实验必须解释这
个差距，不能只报告 latency marginal。

同一 profile 的 temporal replay 也已完成 50 episodes：`0/50` success，mean return
`0.34424`，1,300 次 inference 的注入延迟均值 `94.62 ms`、最大 `252.17 ms`，并产生 1,700 次
capacity drop。这里不要求重新得到逐 episode 完全相同的 zero-delay 轨迹；验收条件是 zero 和
temporal 经过同一 executor/backend/policy 链路，且无延迟结果不存在系统性偏离。独立的 seeded
zero 检查为 `31/50`，与旧 `27/50` 属于可接受的随机波动，因此不再把精确复刻旧 baseline 当作
blocker。

## 2. 锁定的测量口径

本次部署使用两张可见 GPU：logical `cuda:0` 跑 MIKASA/SAPIEN，logical `cuda:1` 跑 StarVLA
inference。profile coordinate 只描述 inference deployment，因此名称是 `1x-rpbzzz6`，不是
`2x-rpbzzz6`。`hardware.json` 仍记录两张可见卡。

每个 20 Hz observation opportunity 都写入 `observation_attempts.jsonl`：

```text
submitted=true
  -> worker_id
  -> observation_to_action_latency_ms
  -> worker_service_time_ms

submitted=false
  -> capacity drop；不伪造 latency 或 service time
```

`observation_to_action_latency` 从 observation capture 开始，到 inference parent 收到 7D action
结束。后续 ordering、scheduler enqueue、frame application、ManiSkill controller 和 simulator
`env.step()` 不计入该值。profile builder 只使用完整 observation-attempt sequence；IID 和 temporal
是同一 sequence 的不同消费方式，capacity drop 由 service time 与 50 ms observation cadence
自然产生。

当前 workload 的固定合同是：

```text
env_fps=20                  obs_fps=20
frame_ms=50                max_steps=60
session_count=5            episodes_per_session=100
evaluation_seed=4242424242 scheduler=latest_ready
simulation_gpu=cuda:0      inference_device=cuda:1
```

## 3. 新机器从零执行

以下命令对应本次成功路径。仓库根目录必须是当前工作目录；不要在
`third_party/starVLA` 内启动 profile，因为训练 checkpoint 的 `base_vlm` 是相对仓库根目录的
`./playground/Pretrained_models/Qwen3-VL-4B-Instruct`。

### 3.1 仓库与硬件

```bash
cd /path/to/latency-sensitive-bench
export BENCH_ROOT="$PWD"

git submodule update --init \
  third_party/MIKASA-Robo \
  third_party/starVLA \
  third_party/flappy-bird-gymnasium \
  third_party/sample-factory

source "$(conda info --base)/etc/profile.d/conda.sh"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
df -h .
```

新 GPU 名称必须先在 `configs/profiles.yaml` 的 `gpu_models` 中注册；本次机器返回
`NVIDIA RPBZZZ6`，仓库已经有对应 alias。

### 3.2 准备 Python 环境

运行 profile 前先激活已经安装好项目和任务依赖的 Python 环境；profile 脚本沿用当前环境，
不会自动创建、激活或修改环境。

```bash
cd "${BENCH_ROOT}"
python -c \
  'import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())'
```

请在当前环境中自行完成所需的 Torch、StarVLA 和 simulator 依赖安装，并确认 Hugging Face
认证可用。profile 脚本不会自动寻找或切换 Conda/uv 环境。

### 3.3 下载 base VLM 与已训练模型

训练 checkpoint 只保存 fine-tuned StarVLA 权重，inference 仍需要 Qwen base VLM：

```bash
cd "${BENCH_ROOT}"
mkdir -p playground/Pretrained_models
hf download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct
```

如果同一文件已经位于 `third_party/starVLA/playground/Pretrained_models/`，可像本次运行一样在
仓库根目录创建 symlink；不要复制第二份 8.3 GB 权重：

```bash
mkdir -p playground/Pretrained_models
ln -s "$(realpath third_party/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct)" \
  playground/Pretrained_models/Qwen3-VL-4B-Instruct
```

已训练 MIKASA 模型由 `configs/profiles.yaml` 精确解析：

```text
repo:       latency-sensitive-bench/benchmark-models
path:       zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenoft-v1
checkpoint: checkpoints/steps_5000_pytorch_model.pt
```

若注册的 `local_root` 不存在，`profile run` 会从该路径下载当前 artifact。

若 `local_root` 已存在，resolver 不会偷偷回退到 Hugging Face；目录中缺 checkpoint 会直接失败。
要 profile 自己的本地模型，在 run 命令中增加 `--model-path /absolute/model/root`，该 root 必须
包含注册的相对 checkpoint 和 model config。

### 3.4 RGB smoke 与正式 profile

先确认 MIKASA GPU RGB 路径能创建双视角 observation：

```bash
cd "${BENCH_ROOT}"
CUDA_VISIBLE_DEVICES=0 "${MIKASA_PY}" - <<'PY'
from scripts.mikasa.evaluate import _make_vla_env

env = _make_vla_env()
observation, _ = env.reset(seed=4242424242)
print(observation["rgb"].shape)
env.close()
PY
```

本次输出是 `torch.Size([1, 128, 128, 6])`。然后运行正式 profile：

```bash
cd "${BENCH_ROOT}"
mkdir -p logs
CUDA_VISIBLE_DEVICES=0,1 bash scripts/profile.sh run \
  --model qwenoft \
  --workload mikasa_intercept_grab_fast \
  |& tee logs/mikasa-blackwell-profile.log
```

任务 launcher 和公共 profile 入口都沿用当前已激活的 Python 环境。

中断后用完全相同的 artifact、解释器、entrypoint 和 config 恢复：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/profile.sh run \
  --model qwenoft \
  --workload mikasa_intercept_grab_fast \
  --resume RUN_ID
```

`run_contract.json` 不一致时 resume 会直接拒绝；已完成 session 会跳过，未完成 session 会重跑。

### 3.5 本地验收、发布与远端回读

成功 run 必须打印 `PROFILE_REF`、`PROFILE_PATH`、`RUN_ID`、`GPU_CLASS` 和 `INSTANCE_ID`。发布
前至少确认五个 session 的 `DONE`、每个 `observation_attempts.jsonl` 非空，以及当前 profile 的
六个 canonical 文件：

```bash
export PROFILE_REF=qwenoft/1x-rpbzzz6/mikasa_intercept_grab_fast/instance_e906b84a82f9e086
export RUN_ID=20260904T073201011403Z
export RUN_DIR="results/profiling/runs/${PROFILE_REF}/${RUN_ID}"
export PROFILE_DIR="results/profiling/profiles/${PROFILE_REF}"

for session in 0 1 2 3 4; do
  test -f "${RUN_DIR}/sessions/${session}/DONE"
  test -s "${RUN_DIR}/sessions/${session}/observation_attempts.jsonl"
done

for artifact in profile.json latency_profile.png latency_distribution.json latency_burst_model.json hardware.json DONE; do
  test -f "${PROFILE_DIR}/${artifact}"
done

jq '{model_id, workload_id, n_observation_attempts, n_admitted_observations,
     n_capacity_drops, per_slot_summary, provenance}' \
  "${PROFILE_DIR}/profile.json"
```

本地 gate 通过后再发布：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/profile.sh publish \
  --model qwenoft \
  --workload mikasa_intercept_grab_fast
```

发布后直接按 profile 坐标回读：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/profile.sh resolve \
  --model qwenoft \
  --workload mikasa_intercept_grab_fast \
  --gpu 1x-rpbzzz6 \
  --instance instance_e906b84a82f9e086
```

远端布局是：

```text
profiles/<PROFILE_REF>/{profile.json,latency_profile.png,latency_distribution.json,
                        latency_burst_model.json,hardware.json,DONE}
runs/<PROFILE_REF>/<RUN_ID>/...
```

raw run 不上传本机专用的 `generated.yaml`、`run_contract.json` 和 run-level hardware/config
文件；完整 measurement JSONL、episode metrics、pipeline summaries、warmup diagnostics 和图表会
上传。

## 4. 2026-09-04 实际运行结果

### 4.1 机器与软件

| 项目 | 实测值 |
|---|---|
| 可见 GPU | 2× `NVIDIA RPBZZZ6`，每张 97,887 MiB |
| compute capability | 12.0 |
| driver | `580.159.04` |
| simulator/inference | 使用调用 profile 时已激活的 Python 环境 |
| profile coordinate | `1x-rpbzzz6/instance_e906b84a82f9e086` |
| base VLM | Qwen3-VL-4B-Instruct，现有 8.3 GB 目录通过 root symlink 复用 |
| trained model source | Hugging Face model artifact |

SAPIEN 报告系统没有可发现的 Vulkan ICD，但其 fallback 在真实 RGB smoke 和全部 500 episodes 中
正常工作，因此没有改系统 driver 或安装额外 Vulkan package。`pynvml` deprecation、
`pkg_resources` deprecation 和 Albumentations update notice 也只是运行时 warning。

### 4.2 Profile 数值

| metric | value |
|---|---:|
| observation attempts | 30,000 |
| admitted | 16,473（54.91%） |
| capacity drops | 13,527（45.09%） |
| latency mean / std | 99.563 / 32.537 ms |
| latency p50 / p90 | 99.886 / 146.202 ms |
| latency p95 / p99 | 168.197 / 219.746 ms |
| latency max | 664.269 ms |
| worker service mean | 99.157 ms |
| worker service p95 / p99 | 167.814 / 219.315 ms |
| probability > 1 / 2 / 3 frames | 99.994% / 19.098% / 7.066% |
| realtime task success | 0 / 500 |
| mean return | 0.35156 |

五个 session 都各有 100 episodes、6,000 attempts 和 `DONE`。每轮 admitted 数依次为
`3452, 3101, 3143, 3484, 3293`，合计与 profile manifest 一致。

五次 stress warmup 都达到 100 samples，但状态均为 `unstable/drifting`；协议保留 warning 后继续
正式测量。对应 trimmed-mean span 为 `30.44, 4.18, 34.22, 4.32, 10.31 ms`。这份 profile
因此可以用于当前实机的 canonical replay，但后续若换独占节点、功耗策略或 driver，应重新测量并
比较，而不是假设跨机器不变。

本地 artifact：

```text
results/profiling/runs/qwenoft/1x-rpbzzz6/mikasa_intercept_grab_fast/
  instance_e906b84a82f9e086/20260904T073201011403Z/       114 MB
results/profiling/profiles/qwenoft/1x-rpbzzz6/mikasa_intercept_grab_fast/
  instance_e906b84a82f9e086/                             130 KB
```

发布后检查结果：canonical 文件 `6/6` 本地/远端 SHA-256 一致，远端 profile 6 files，raw run
94 files。失败的诊断 run `20260904T072621212271Z` 没有 `DONE`，保留在本地用于定位问题，没有
发布。

## 5. 本次修复的三个根因

### 5.1 multiprocessing spawn 跨解释器重载

MIKASA session 由 native venv 启动，但 inference worker 使用 StarVLA Conda Python；spawn 会在
worker interpreter 中把 `scripts/mikasa/evaluate.py` 作为 `__mp_main__` 重新加载。原文件顶层
import `mani_skill`、`mikasa_robo_suite` 和 `baselines`，而 StarVLA env 没有这些 native package，
所以第一个 run 在模型测量前失败。

修复是把 native-only import 移到真正创建 MIKASA env/PPO policy 的函数中。公共类型和 WebSocket
client 仍保持顶层 import，没有增加 fallback 或复制依赖。新增回归测试会以 `__mp_main__` 启动
脚本，并主动拒绝上述三个 package 的 import。

### 5.2 Hugging Face checkpoint symlink 丢失 `.pt`

Hugging Face snapshot 中的 `steps_5000_pytorch_model.pt` 是指向 `blobs/<hash>` 的 symlink。
`Path.resolve()` 会把 checkpoint 变成没有 `.pt` suffix 的 blob path，随后 StarVLA 按 suffix 验证
时拒绝它。这个问题存在两层，因此两层都必须修：

- `latency_bench/cross_device/profile_resolver.py` 保留 snapshot 内 checkpoint path，只用
  `stat()` 确认目标存在；
- `third_party/starVLA/starVLA/model/framework/share_tools.py` 使用 `os.path.abspath()` 规范化
  `..`，但不 dereference symlink。

只修任意一层都会在下一层再次丢失 suffix。config YAML 可以从 extensionless blob 正常加载，
不需要扩展这个改动。

### 5.3 Qwen base VLM 相对路径

checkpoint config 保存的是 `./playground/Pretrained_models/Qwen3-VL-4B-Instruct`。本次机器已有
`third_party/starVLA/playground/...`，但仓库 root 没有对应路径；创建 root symlink 后模型正常
加载。fresh machine 直接下载到 root 的 `playground/` 即可。

## 6. 验证与未完成事项

本次代码验证：

```text
main targeted tests:     71 passed
profile plot/sync tests:  24 passed
StarVLA path tests:       5 passed
main changed-file ruff:   passed
root + submodule diff:    git diff --check passed
real RGB smoke:           [1, 128, 128, 6]
full profile:             5/5 sessions DONE
remote round trip:        6/6 canonical hashes match
```

StarVLA 是 git submodule。`share_tools.py` 和它的 regression test 当前位于 nested worktree；正式
提交时必须先在 `third_party/starVLA` 提交这两个文件，再更新 superproject gitlink，否则主仓库
patch 只会显示 submodule dirty，必要的第二层 symlink 修复会丢失。

profile replay 已统一到 task-native `latency-eval --eval-config` 与公共 `run_from_config()`。
`realtime` 只接受 `latency.method=measured`；`zero`、`temporal`、`iid` 等只由 simulated
executor 消费，因此不会再把“真实 inference wall time”误叫作 zero latency。MIKASA canonical
zero 与 temporal replay 命令见 [MIKASA.md](MIKASA.md#64-canonical-zero-delay-与-profile-replay)。

剩余事项：

1. 若要把 Blackwell profile 恢复为 task-owned 一条命令，修改 MIKASA launcher，避免每次 run
   都重新 sync 并降级 Torch；在此之前继续使用本文公共入口。
2. 若换 GPU、driver、功耗策略、物理 GPU UUID 或 worker 数，重新 profile；不要复用当前
   `instance_e906b84a82f9e086`。
3. 不再保留旧文档中的“尚未运行完整 5×100”、旧 action-only artifact 问题和重复 schema
   设计待办；canonical contract 已由当前代码、
   [profiling cookbook](../docs/profiling/cookbook.md) 和本次发布 artifact 共同定义。
