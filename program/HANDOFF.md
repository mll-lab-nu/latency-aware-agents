# 五环境历史资产接收与评测快照

本文保留 Flappy Bird、Demon Attack、Deadly Corridor、Atlantis 和 Asterix 原稿记载的
接收、评测及发布状态，以及原稿中的评测配置。源仓库、checkpoint 路径、
revision 和中央目标路径见 [ARTIFACT_HANDOFF.md](ARTIFACT_HANDOFF.md)。

**快照归属：** 2026-09-30 收录的[实施前本地原稿](../outputs/refactor/batch-02/before-docs/HANDOFF.md)，
对应 HEAD `51ed017d` 加当时工作树；原稿未跟踪，HEAD 本身不能标识其内容。
收录日期是文档快照日期，原稿没有记录状态核查日期或逐行报告/checkpoint revision，
表中数值的具体来源仍待核实。“正在评测”等措辞仅作为原稿记录保留，不建立当前运行状态。

以下状态、数量和分数原样保留供追溯。只有确认 checkpoint、模型输入、seed、FPS 与 horizon
一致后，才可与其他结果表对照；本页分数不能直接替换 RESULTS 的正式值。新版决策历史模型的
独立结果与发布身份见 [DECISION_HISTORY_VLA_H8](DECISION_HISTORY_VLA_H8.md)，
Atlantis/Asterix H8 见 [ATLANTIS_ASTERIX_H8](ATLANTIS_ASTERIX_H8.md)；正式结果索引见
[RESULTS](RESULTS.md)。不同版本的完成状态不回填到原稿资产表；训练、正式评测、验收与发布
仍分别由各资产的证据确认。

缩写：O = OpenVLA，G = GR00T，π = π0.5。

状态：

- ✅：资产已接收，正式评测和中央发布已完成。
- 🟡：资产已接收或已暂存，但评测、打包、发布或发布后回读尚未完成。
- ⚠️：资产已经发布，但现有评测协议需要复核或重跑。
- ❌：按当前决策不进入正式 accepted 路径。
- —：该环境没有这一类资产。

## 5×5 原稿接收状态（逐行来源待核实）

| 环境 | 0-latency small | 0-latency VLA | Latency profile | Profiled small | Profiled VLA |
|---|---|---|---|---|---|
| Flappy Bird | ✅ 已发布；100 episodes 为 **444.60** | 🟡 三族已完成：O **439.15**、G **19.37**、π **46.60**；待打包、发布和远端回读 | 🟡 三族 canonical 文件完整；mean/p95 ms：O **75.87/81.32**、G **137.73/147.21**、π **200.61/213.87**；待中央 model provenance 最终修正 | ⚠️ 已发布：O **5.137**、G **4.162**、π **2.000**；结果异常低，profile replay 协议仍需审计 | ❌ 当前无正式接收项；O 缺完整 matching-profile 闭环，G/π 已知差 |
| Demon Attack | ⚠️ 模型已发布；旧 100-episode 结果 **30.64** 使用了错误 episode 协议，必须重评 | 🟡 O/G/π 正在使用修正后的完整游戏协议重评 | 🟡 三族 canonical 文件完整；O **71.59/75.24**、G **129.16/136.17**、π **192.75/203.18** ms；待 provenance 最终修正 | ⚠️ 已发布的 O **341.70**、G **313.25**、π **270.15** 来自旧协议，需要使用修正后的 14400 horizon/full-game episode 重评 | ❌ 旧数据 O 明确差；新数据未发布；G/π 无新模型 |
| Deadly Corridor | ✅ 已发布；100 episodes 为 **2116.79** | 🟡 三族已完成：O **360.90**、G **160.08**、π **191.59**；待打包、发布和远端回读 | 🟡 三族 canonical 文件完整；O **73.69/78.02**、G **131.66/139.50**、π **193.88/206.22** ms；待 provenance 最终修正 | ✅ 已发布：O **686.03**、G **566.55**、π **347.67** | ❌ 现有三族均不进入正式路径 |
| Atlantis | ✅ 已发布；100 episodes 为 **29271** | 🟡 O 已完成 **25023**；G/π 正在评测；三族均待统一发布 | — | — | — |
| Asterix | ✅ 已发布；100 episodes 为 **4099** | 🟡 三族已完成：O **450**、G **380**、π **283**；待打包、发布和远端回读 | — | — | — |

快照数量：

- 0-latency small：5/5 checkpoint bundle 已发布；Demon Attack 的正式分数待重评。
- 0-latency VLA：15/15 checkpoint/config/statistics 已暂存；10/15 的 100-episode
  评测已经完成，Demon Attack 三族和 Atlantis G/π 仍在运行；当前尚未统一发布。
- RTX 3090 latency profile：9/9 canonical profile 可读；最终 provenance 修正和 HF
  revision 回读未完成。
- Profiled small：9/9 bundle 已发布；Flappy 结果需审计，Demon Attack 三项需按修正
  协议重评。
- Profiled VLA：0 项 accepted，这是当前明确决策，不是迁移遗漏。

## 正式评测公共合同

所有可接受的正式评测必须满足：

- 每个 checkpoint 运行 100 episodes，确定性推理。
- 逻辑 case IDs 为 `0..99`，目标 seeds 为 `42..141`。
- VLA 实际环境 seeds 为 `42..141`。Sample Factory evaluator 当前存在既有的
  `+104729` reset-seed 映射，small policy 的实际 reset seeds 为 `104771..104870`；
  结果文件必须同时记录逻辑 ID 和实际 seed。
- zero-latency 使用 simulated virtual time，执行 H100/H200 不改变延迟条件。
- zero-VLA 使用训练数据中的完整 latency-0 prompt，`prompt_mode: raw`，不得使用
  `latency_neutral`。prompt 必须包含 `Current action latency is 0 raw frames (0.00 ms)`
  及对应环境的 env/observation FPS。
- VLA `action_prefix.mode: none`，policy 在 CUDA 上推理；图像最终输入尺寸为 224×224。
- VLA 每进程固定 Torch intra-op 线程 8、inter-op 线程 4；`OMP_NUM_THREADS`、
  `MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`NUMEXPR_NUM_THREADS` 均为 8。
- 输出报告 episode 数、mean/std、Student-t 95% CI、episode length、horizon、并行度和
  逐 episode 原始记录，并指向 exact checkpoint。

## 各评测类型配置

### 0-latency small

入口配置：

- Flappy：`configs/experiments/flappy/sf_eval_zero_latency.yaml`
- Demon Attack：`configs/experiments/demon_attack/sf_eval_zero_latency.yaml`
- Deadly Corridor：`configs/experiments/deadly_corridor/sf_eval_zero_latency.yaml`
- Atlantis：`configs/experiments/atlantis/sf_eval_zero_latency.yaml`
- Asterix：`configs/experiments/asterix/sf_eval_zero_latency.yaml`

共同关键项：

```yaml
latency:
  method: zero
policy:
  deterministic: true
evaluation:
  eval_episodes: 100
  eval_deterministic: true
```

环境差异：

| 环境 | env/obs FPS | frame stack | 并行度 | horizon | 关键 episode/reward 配置 |
|---|---:|---:|---:|---:|---|
| Flappy | 30/30 | 4 | 100 | 3600 | `eval_raw_reward: false`；GPU simulator |
| Demon Attack | 60/15 | 4 | 100 | 14400 | `terminal_on_life_loss: false`；`eval_raw_reward: true`；`demon_attack_gpu_batched` |
| Deadly Corridor | 35/8.75 | 4 | 16 | 30000 | Sample Factory VizDoom；raw reward |
| Atlantis | 60/15 | 4 | 32 | 5000 | `gymnasium_sf`；zero latency |
| Asterix | 60/15 | 4 | 32 | 5000 | `gymnasium_sf`；zero latency |

Demon Attack 先前使用 `terminal_on_life_loss: true`、horizon 3600 得到的 30.64 不是
当前正式协议结果。修正协议要求掉命不结束 episode，并恢复历史 evaluator 的 14400
horizon。

### Profiled small

Profiled small 与各环境的 0-latency small 配置相同，只替换 latency sampler：

```yaml
executor:
  simulated_worker_capacity: 1
latency:
  method: iid
  profile_path: results/profiling/profiles/<family>/1x-rtx3090/<task>/instance_a5037b165aa0cedc/profile.json
  profile_worker_slot: 0
  seed: 42
```

每个 checkpoint 使用其训练时对应的 O/G/π profile，不交叉替换。Demon Attack 的现有
正式 profile 配置为：

- `configs/experiments/demon_attack/sf_eval_openvla_rtx3090_profile.yaml`
- `configs/experiments/demon_attack/sf_eval_gr00t_rtx3090_profile.yaml`
- `configs/experiments/demon_attack/sf_eval_pi05_rtx3090_profile.yaml`

Flappy 和 Deadly Corridor 使用同样的 IID override，并分别替换为各自 task/profile 路径。
Profile sampler 将 observation-to-action latency 的毫秒值按任务 `env_fps` 向上量化为 raw
frames；不是把毫秒直接当 observation steps。

### 0-latency VLA

入口配置：

- Flappy：`configs/experiments/flappy/starvla_eval_zero_latency.yaml`
- Demon Attack：`configs/experiments/demon_attack/starvla_eval_zero_latency.yaml`
- Deadly Corridor：`configs/experiments/deadly_corridor/starvla_eval_zero_latency.yaml`
- Atlantis：`configs/experiments/atlantis/starvla_eval_zero_latency.yaml`
- Asterix：`configs/experiments/asterix/starvla_eval_zero_latency.yaml`

三族在同一任务上共用下面的环境和评测配置，只替换 exact checkpoint、模型 config 和
dataset statistics。

| 环境 | env/obs FPS | obs stride | env frame stack / 模型输入 | backend | 并行度 | horizon | scheduler |
|---|---:|---:|---|---|---:|---:|---|
| Flappy | 30/30 | 1 raw frame | 4 / 最新单帧 | `flappy_gpu_batched` | 10 | 3600 | `one_frame_then_noop`、`latest_ready` |
| Demon Attack | 60/15 | 4 raw frames | 4 / 最新单帧 | `demon_attack_gpu_batched` | 10 | 14400 | `hold`、`latest_ready` |
| Deadly Corridor | 35/8.75 | 4 raw frames | 4 / 最新单帧 | VizDoom MultiBinary CPU env + CUDA VLA | 4 | 30000 | `hold`、`latest_ready` |
| Atlantis | 60/15 | 4 raw frames | 1 / 单帧 | `atlantis_gpu_batched` | 10 | 5000 | `hold`、`issue_order_fifo` |
| Asterix | 60/15 | 4 raw frames | 1 / 单帧 | `asterix_gpu_batched` | 10 | 5000 | `hold`、`issue_order_fifo` |

逐环境附加配置：

#### Flappy

```yaml
env:
  gym_id: FlappyBird-v0
  env_fps: 30
  obs_fps: 30
  frame_stack: 4
  obs_resize: [224, 224]
  simulator: gpu
  use_gpu_render: false
policy:
  prompt_mode: raw
  actions: [noop, flap]
evaluation:
  eval_parallel_envs: 10
  latency_bench_env_backend: flappy_gpu_batched
  eval_max_steps: 3600
  eval_raw_reward: true
```

#### Demon Attack

```yaml
env:
  env_id: ALE/DemonAttack-v5
  env_fps: 60
  obs_fps: 15
  frame_stack: 4
  obs_resize: [224, 224]
  noop_max: 30
  reset_mode: fire
  terminal_on_life_loss: false
  clip_reward: false
  repeat_action_probability: 0.0
  full_action_space: false
policy:
  prompt_mode: raw
  actions: [NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE]
evaluation:
  eval_parallel_envs: 10
  latency_bench_env_backend: demon_attack_gpu_batched
  eval_max_steps: 14400
  eval_raw_reward: true
```

#### Deadly Corridor

```yaml
env:
  env_fps: 35
  obs_fps: 8.75
  frame_stack: 4
  res_w: 128
  res_h: 72
  obs_resize: [224, 224]
policy:
  prompt_mode: raw
  action_layout: multibinary_7
  multibinary_threshold: 0.0
  actions: [MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT, ATTACK]
evaluation:
  eval_parallel_envs: 4
  eval_max_steps: 30000
  eval_raw_reward: true
```

StarVLA 使用 `VizdoomDeadlyCorridor-MultiBinary` 环境；配置中的 7 维语义按钮在执行前映射
到 live VizDoom button order。

#### Atlantis

```yaml
env:
  base_env_id: ALE/Atlantis-v5
  env_fps: 60
  obs_fps: 15
  frame_stack: 1
  obs_resize: [224, 224]
  noop_max: 30
  terminal_on_life_loss: false
  clip_reward: false
  repeat_action_probability: 0.0
  full_action_space: false
policy:
  prompt_mode: raw
  actions: [noop, fire, rightfire, leftfire]
evaluation:
  eval_parallel_envs: 10
  latency_bench_env_backend: atlantis_gpu_batched
  eval_max_steps: 5000
  eval_raw_reward: true
```

#### Asterix

```yaml
env:
  base_env_id: ALE/Asterix-v5
  env_fps: 60
  obs_fps: 15
  frame_stack: 1
  obs_resize: [224, 224]
  noop_max: 30
  terminal_on_life_loss: false
  clip_reward: false
  repeat_action_probability: 0.0
  full_action_space: false
policy:
  prompt_mode: raw
  action_layout: factorized_6
  actions: [noop, up, right, left, down, upright, upleft, downright, downleft]
evaluation:
  eval_parallel_envs: 10
  latency_bench_env_backend: asterix_gpu_batched
  eval_max_steps: 5000
  eval_raw_reward: true
```

Atlantis/Asterix 的 ALE native `frameskip` 为 1；4-raw-frame observation cadence 由
latency-bench scheduler 负责，避免 native frameskip 与 scheduler 重复计算。

### Latency profile

九项 profile 均来自同一 RTX 3090 instance：`instance_a5037b165aa0cedc`。当前评测不重新
profiling，只消费已有 canonical distribution。

| 环境 | O mean/p95 ms | G mean/p95 ms | π mean/p95 ms |
|---|---:|---:|---:|
| Flappy | 75.87 / 81.32 | 137.73 / 147.21 | 200.61 / 213.87 |
| Demon Attack | 71.59 / 75.24 | 129.16 / 136.17 | 192.75 / 203.18 |
| Deadly Corridor | 73.69 / 78.02 | 131.66 / 139.50 | 193.88 / 206.22 |

### Profiled VLA

当前不运行统一正式 profiled-VLA sweep：

- Flappy O 只有在 exact config、statistics、task contract 齐全后才运行 matching-profile
  评测；G/π 已知差。
- Demon Attack 旧数据 O 已知差；等待新数据和新模型，G/π 当前无新模型。
- Deadly Corridor 现有 O/G/π 均明确差，不重复消耗评测资源。
- Atlantis/Asterix 没有 latency profile 或 profiled-VLA 接收项。

## 发布完成条件

某项只有同时满足以下条件才算完成：

1. checkpoint、config、statistics 和源 revision 可重新下载。
2. exact checkpoint 通过真实 inference smoke。
3. 正式评测包含恰好 100 个约定 case IDs/seeds，并使用本文件记录的协议。
4. `eval_l0.json` 或 `eval_profile.json` 指向 exact checkpoint。
5. bundle 发布到中央 HF 路径后重新下载并完成哈希、配置、provenance 和评测回读。
6. latency profile 的 model path 和 revision 能解析到已发布中央模型。

仅 checkpoint 已下载、评测进程已启动或文件已经上传，都不能单独视为完成。
