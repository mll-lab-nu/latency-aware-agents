# Humanoid 延迟研究提案

[English](./humanoidbench.md) | [中文](./humanoidbench.zh.md)

> 状态：讨论中的提案
>
> 最后更新：2026-08-26

## 结论

默认路线为：

```text
HumanoidBench 任务/场景/reward
        +
与官方 SONIC 对齐的 Unitree G1
        ↓
逐任务 FastTD3 teacher：proprioception + privileged task state → 64D body token + 14D hands
        ↓
50 Hz frozen SONIC decoder → 29D body position target + 14D hand target
        ↓
经过质量门控的 rollout：RGB + language + proprioception → 实际执行的 78D command
        ↓
StarVLA 训练基础设施 + NVIDIA 官方 GR00T N1.7 model/processor
        ↓
约 2.5 Hz VLA → 40 ×（64D body token + 14D hands）
        ↓
50 Hz SONIC，由 benchmark 注入 VLA observation/inference latency
```

此前四任务实验使用连续 `Box(78)` teacher contract：前 64 维是自由 SONIC body token，后 14 维是可学习 hand target。Hurdle 的 10M 结果表明这个接口只复用了 decoder，没有真正使用 motion prior，因此 Hurdle 下一轮改为官方 Run reference token 加 64D FastTD3 residual；其他 task 的旧 baseline 保持不变。

这条路线主动放弃与原生 HumanoidBench 的可比性。目标是为我们自己的 embodied VLA pipeline 构建受控 latency benchmark，而不是在原始机器人和 actuator 定义下提交排行榜结果。

## 目标与非目标

### 目标

- 研究视觉 observation 延迟和 VLA inference 延迟如何影响全身任务表现。
- 生成对 VLA 真正有用的 teacher data，而不只是最大化原始标量 reward。
- 在 locomotion、balance 和 basketball 之间复用同一种 body command representation。
- 所有实验都在 MuJoCo simulation 中完成。
- 尽可能保留当前高效的 off-policy 训练路径。

## 现有实验已经说明了什么

### Flat joint-space RL 能学会 task reward，但不一定学到有用动作

第一轮 Sample Factory APPO baseline 使用 32 个 worker、rollout 64、batch 2048、一个 epoch 和 10M environment steps：

| 任务 | Return | Episode length |
|---|---:|---:|
| Balance Hard | 33.74 | 37.9 |
| Hurdle | 14.43 | 76.7 |
| Run | 12.86 | 93.5 |
| Basketball | 12.12 | 31.5 |
| High Bar Hard | 2.81 | 56.8 |

运行配置：[baseline experiment summary](../results/humanoidbench_10m_baseline/generated/summary.json)。

对齐官方设置的 APPO 实验把优化日程改为 rollout 256、batch 512、十个 epoch、每个 epoch 16 个 batch、learning rate `3e-5`，训练量仍为 10M steps：

| 任务 | Return | Episode length |
|---|---:|---:|
| Balance Hard | 73.20 | 92.3 |
| Hurdle | 12.64 | 56.0 |
| Run | 20.25 | 96.3 |
| Basketball | 31.46 | 60.5 |
| High Bar Hard | 2.57 | 25.4 |

运行配置：[official-aligned experiment summary](../results/humanoidbench_sf_official_10m_wandb/generated/summary.json)。High Bar 没有 queue summary，因此直接从其[十个 evaluation episode](../results/humanoidbench_sf_official_10m_wandb/h1strong_highbar_hard/eval_best_l0/eval_latency_0/episode_metrics.jsonl)聚合。

增加 optimizer work 对部分任务有帮助，但没有解决整个任务族。完成的 flat-action FastTD3 实验强得多：

| 任务 | Return | Episode length | Training FPS |
|---|---:|---:|---:|
| Run | 907.21 ± 4.72 | 1000 | 4109 |
| Hurdle | 503.72 ± 163.57 | 930.8 | 4864 |

运行配置：[FastTD3 experiment summary](../results/humanoidbench_fasttd3/generated/summary.json)。

但人工检查 evaluation video 后发现动作明显不自然。我们还没有计算量化的 motion-quality metric，所以这只是定性观察，并非测量过的 human-likeness score：

- [Run evaluation video](../results/humanoidbench_fasttd3/h1hand_run/eval_best_l0/eval_latency_0/videos/bin_00_episode_000003.mp4)
- [Hurdle evaluation video](../results/humanoidbench_fasttd3/h1hand_hurdle/eval_best_l0/eval_latency_0/videos/bin_00_episode_000007.mp4)

这与任务定义一致：原始 reward 可以鼓励向前运动、保持平衡和完成任务，却没有规定自然手臂姿态、步态对称性、脚底打滑、落地冲击或动作平滑度。[FastTD3 的 HumanoidBench 报告](https://younggyo.me/fast_td3/)和 [MuJoCo MPC 实验](https://arxiv.org/abs/2408.00342)都报告了相同的 reward-design failure mode。

### MuJoCo environment 不是 flat RL 的主要吞吐瓶颈

快速、单次测量的 benchmark 结果为：

| 任务 | Raw env，1 env | Raw env，32 envs | SF aggregate，32 envs | SF steady 10 s，32 envs |
|---|---:|---:|---:|---:|
| Balance Hard | 471 | 15024 | 4103 | 7373 |
| Hurdle | 532 | 16617 | 4007 | 6758 |
| Run | 692 | 22466 | 4815 | 7782 |
| Basketball | 632 | 20412 | 4714 | 7782 |
| High Bar Hard | 201 | 6445 | 2189 | 5325 |

原始测量：[throughput CSV](../results/humanoidbench_throughput/quick/throughput.csv)。

Raw environment 可以随并行度扩展，flat FastTD3 能达到约 4k–5k FPS。SONIC decoder 会引入新的开销，因此不能把这个结果当作 token-space training 仍能保持相同吞吐的承诺。

此前的内存问题也不是 `/dev/shm` 限制：host 暴露约 1.84 TiB 内存，而 job cgroup 的限制和观测峰值约为 483 GB。


### FastTD3 + SONIC Run 合同 smoke

第一版仅适配 Run，使用固定版本的 SONIC v1.1 decoder 和完全对应的 29-DoF G1 MuJoCo asset。FastTD3 看到连续 `Box(64)` action；环境构造官方 994D decoder input，在一个 50 Hz control interval 内保持 decoded 29D body target，并在十个 2 ms MuJoCo substep 中重新计算 PD control；14D hand target 固定为 relaxed pose。

仓库侧 contract check 已通过：deterministic seeded reset、decoder history layout 与 zero padding、IsaacLab-to-MuJoCo joint mapping、64D action 与 99D privileged observation space、单步执行、RGB render，以及 1000-step zero-token rollout。本地 dynamics 有意采用 deployment-style `0.002 × 10`，而不是 SONIC training simulator 的 `0.005 × 4`；二者 policy interval 都是 50 Hz，但 dynamics trajectory 并不完全相同。

一次 100k-step FastTD3 smoke 已完成整个链路，并生成 final、best、periodic checkpoint 和 deterministic eval artifact。它只证明 pipeline 可用，不是最终学习结果：

| 测量项 | 结果 |
|---|---:|
| 采集的 environment steps | 100,224 |
| 包含 startup 与 compilation 的端到端 wall-clock FPS | 865.7 |
| Best checkpoint 的 10-episode eval return | 16.60 ± 1.31 |
| Eval 平均 episode length | 132.5 |
| Eval video | [MP4](../results/humanoidbench_sonic_fasttd3/smoke_100k/g1_sonic_run/eval_best_l0/eval_latency_0/videos/bin_00_episode_000002.mp4) |

该短程 policy 仍会摔倒，尚不能作为 teacher，但早期学习信号很强。约 100k steps 时，原始 61D FastTD3 Run 的 rolling train return 为 1.72，而 SONIC 已达到 6.56。原始 run 到约 649k steps 才达到相近的 6.63。原始 rolling return 首次超过 16.6 是约 2.65M steps，但该值不能与 SONIC deterministic eval 的 16.6 直接等同。现有证据支持在性能优化后继续训练，但尚不能证明最终 task success 或自然 motion。

由于这是短程 debug，W&B 被关闭；生成的正式训练配置会启用已有的 `latency-sensitive-bench` project 和 `humanoidbench-sonic-fasttd3` group。

当前每个 environment 使用独立、单线程 CPU ONNX Runtime decoder；每种数量只测一次，结果如下：

| Environment 数量 | Aggregate environment FPS | 单 environment FPS |
|---:|---:|---:|
| 1 | 149.9 | 149.9 |
| 4 | 441.0 | 110.3 |
| 8 | 890.3 | 111.3 |
| 16 | 1515.7 | 94.7 |
| 32 | 2519.3 | 78.7 |

这足以验证设计，但不足以启动 10M-step run。在当时仅适配 Run 的历史实验中，优化路径把一个支持动态 batch 的 ONNX decoder 移入 Sample Factory inference worker；actor/replay 保存 64D token，只有解码后的 29D target 发送给 MuJoCo。后续 100k 验证的稳态吞吐达到 5.06k–5.38k FPS；由于 128 个 worker 启动约需 40 秒，包含启动时间的总 FPS 为 1.52k。下面的五任务修正已用可学习 hand 和完整 78D replay 取代这份 action contract。

随后 1M-step W&B run 完整成功：

| 测量 | 结果 |
|---|---:|
| Collected environment steps | 1,000,192 |
| 包含 startup/shutdown 的总 FPS | 4,329.6 |
| 成熟阶段 60-second FPS | 5,495.5 |
| Best-checkpoint deterministic return，10 episodes | 139.16 ± 20.51 |
| Mean episode length | 922.8 / 1000 |
| 跑满 horizon 的 episodes | 3 / 10 |
| W&B | [run](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/humanoidbench_sonic_fasttd3_gpu_decoder_g1_sonic_run) |
| Evaluation video | [MP4](../results/humanoidbench_sonic_fasttd3_gpu_decoder_1m/g1_sonic_run/eval_best_l0/eval_latency_0/videos/bin_00_episode_000005.mp4) |

这证明 Run 上可以快速学习 token-space policy，但 motion 是否适合作为 teacher data 仍需看视频。单次结果本身既不支持直接跑 10M，也不能外推到另外四个任务。

### 长程训练前的历史 parity 修正

旧 G1 对比同时换了机器人、简化场景和本地近似 reward，因此 H1/G1 差距不可解释；继续调 optimizer 不能回答原问题。修正后的本地 candidate：

- 原样复用 HumanoidBench 官方 floor、hurdle、balance 和 basketball task assets；
- 直接调用官方 `Run`、`Hurdle`、`BalanceHard` 和 `Basketball` reward/reset/termination class；
- 只改变 robot/controller、G1 keyframe、observation/action 维度和 task class 必需的 site/sensor/camera；
- 暴露 raw `43D = 29 body + 14 hands` 与 SONIC `78D = 64 token + 14 hands`；
- raw action 使用 MuJoCo joint order，只在 SONIC decoder output 边界执行 Isaac-to-MuJoCo permutation；
- 保留原始 `0.002 × 10 = 20 ms` timing 和 task-specific tracking camera。

历史 candidate XML 均可加载和渲染，八个 family contract test 全部通过。两张 GPU 上并发、关闭 W&B 的 Hurdle smoke 分别让两个 variant 经过 65,792 steps 和真实 learner update：raw43 总 FPS 为 1,201.7，sonic78 为 1,197.1；共享 host CPU 时二者成熟阶段 10-second FPS 约为 1.6k 与 1.7k。这只是 pipeline check，不是后续独立 throughput benchmark。

历史 gate 是人工检查 H1/G1 配对视频；该 candidate 没有发布 corrected asset revision。

此前公开的 [`latency-sensitive-bench/g1-sonic-assets`](https://huggingface.co/datasets/latency-sensitive-bench/g1-sonic-assets) revision 对历史对比无效：它使用简化本地 scene，而不是官方 HumanoidBench task assets 和逻辑。当前支持的替代 bundle 明确不包含两个 High Bar task。Decoder 继续单独固定到 NVIDIA `nvidia/GEAR-SONIC` revision `1157dbafbae49ba17b8b53ace04cd8191b6a6697`。

### Hurdle 10M 结果与 residual 修复

同一 Hurdle scene 的 best-checkpoint evaluation 为：

| Interface | Return | Episode length |
|---|---:|---:|
| raw43 | 160.21 ± 62.74 | 863.7 |
| sonic78 | 303.35 ± 92.18 | 1000.0 |

SONIC 确实改善了稳定性，但视频中的动作仍然不对。主要问题不是 decoder 数值 bug，而是训练合同没有真正使用 pretrained motion prior：FastTD3 可以在整个连续 64D cube 内自由输出，约 49.3% 的 token 分量绝对值不小于 0.95，约 33.0% 不小于 0.99；官方 encoder/FSQ 产生的 motion manifold 没有约束它。

同时存在以下问题：

- 最主要：RL 学习目标没有适配 SONIC。自由 token 只经过 decoder，不等于从参考动作编码得到的 motion prior。
- Reward 只优化任务表现，不能充分约束自然动作；是否增加倒地、碰杆和 motion-quality 项留给独立 reward audit，本轮不修改 reward、termination 或 best-checkpoint metric。
- Hurdle reset 没有使用 SONIC 的中立位。
- Decoder 已经收到完整 930D history/state；真正缺失的是 actor、critic 和 replay 看不到这些状态，也看不到参考 token。
- 14D hand 没有 motion prior，Hurdle 又不需要手部控制，容易产生无意义动作。
- 本地 `0.002 × 10` 与 SONIC training simulator 的 `0.005 × 4` 都是 50 Hz policy interval，但 dynamics 不完全相同；目前只记录为剩余风险。

直接修复方案是：离线用官方 Run planner 生成 30 Hz qpos，重采样到 50 Hz，再经过同 revision 的官方 encoder（其输出已包含 FSQ）得到 `1001 × 64` reference token cache。Hurdle FastTD3 输出连续 64D residual：

```text
token[t] = reference_token[t] + (1 / 16) * residual[t]
```

这里 `1/16` 是官方 FSQ 相邻量化 level 的间距；组合后不再 quantize，也不 clip。Actor、critic 和 replay 统一看到 `task observation + 930D decoder state + 64D reference token`。Decoder 仍接收组合后的 token 和原始 930D state。Reset 使用 SONIC default joints、pelvis `z=0.76`、清空 history/last action/token cursor；14D hands 固定 relaxed zero target。第一版只为 Hurdle 增加 `sonic64_residual`，保留 raw43/sonic78 作为历史 baseline。

## 为什么首选 SONIC 作为 body interface

[SONIC](https://arxiv.org/html/2511.07820) 是面向 Unitree G1 的 learned whole-body controller。其公开 VLA workflow 暴露紧凑的 motion-token command，由底层 policy 处理高频 body action。对本项目而言，它有三点价值：

1. Teacher 和 VLA 可以使用同一种 body command representation。
2. Learned whole-body prior 可能在引入 imitation 机制前就避免一部分退化的 raw-joint 解。
3. 可以在 command-chunk 边界研究 VLA latency，同时让 low-level controller 继续以 50 Hz 运行。

SONIC 本身并不保证任务成功、动作像人或手指自然。这些仍然是 teacher、task reward 和 hand command 的实验属性。

### Body command 是连续 64D vector

SONIC 在训练内部使用 finite scalar quantization，但官方部署 client 发送的是 checkpoint-specific 64D `float32` motion token。它还会对 command 做线性混合；如果 runtime interface 强制使用 token ID，这种操作就不成立。因此：

- 旧 baseline 的 SONIC body prefix 保持连续 `Box(64)`；加上可学习 hands 后为 `Box(78)`。
- Hurdle residual actor 也输出连续 `Box(64)`，但语义是对官方 FSQ reference token 的一个 level residual，不再负责从零生成完整 token。
- 不增加 quantization wrapper，也不切换到 MultiDiscrete APPO。
- 把 64 个维度视为 checkpoint-specific latent contract，而不是各自有独立物理含义的 control。

公开合同见 [SONIC VLA workflow](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html)、[inference guide](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_inference.html)和 [`features_sonic_vla.py`](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/data/features_sonic_vla.py)。

### 历史 parity 方案：所有任务都先从自由 64D 层开始

旧 raw43/sonic78 baseline 的 teacher contract 为：

```text
observation: robot proprioception + privileged task/object state
action:      64D continuous SONIC body token + 14D hand targets
algorithm:   FastTD3
decoder:     frozen SONIC policy
```

这不表示所有任务共享同一个 actor，而是各任务 actor 共享一种 action contract、replay format、decoder 和最终 VLA target。Hurdle residual 已不再遵循这个 `64D + 14D` 合同：它学习 64D residual，并固定 hands。

某个任务失败时，首先检查：

1. G1 scene 和 reset state；
2. reward 与 termination 语义；
3. token scaling、history、proprioception 和 SONIC checkpoint 兼容性；
4. frozen decoder 是否能表达任务所需动作；
5. 训练稳定性和 update 数量是否充足。

完成这些检查后，才能为该任务重新考虑更高层 planner、reference tracker 或 skill interface。现在就增加这一层会制造第二种未经验证的 command language，也让 teacher/VLA contract 更难解释。

## 机器人与任务适配

### 使用与 SONIC 官方完全一致的 G1

原生 HumanoidBench G1 与公开 SONIC controller 的 wire contract 不兼容：

| 属性 | 原生 HumanoidBench G1 | 与官方 SONIC 对齐的 G1 |
|---|---:|---:|
| Body DoF | 23 | 29 |
| Hand DoF | 14 | 14 |
| Body actuation | torque | 通过 SONIC/PD contract 输出 position target |
| 预期 command | raw actuator action | 64D motion token |

既然不要求原生分数可比，最干净的方案就是把官方 SONIC-aligned G1 model 和 controller contract 引入 HumanoidBench scene。

HumanoidBench upstream 有 32 个 task，但本 SONIC G1 adaptation 只支持 30 个。`highbar_simple` 和 `highbar_hard` 明确不支持：不注册、不生成、不做 comparison、不训练。其余 task 复用同一份官方 G1/controller 定义，不为每个任务 fork 一套机器人。

### 首批四个任务（历史 raw43/sonic78 parity）

| 任务 | Scene adaptation | 初始 hand policy |
|---|---|---|
| Balance Hard | 使用官方 task asset/class；只替换 robot 和 G1 keyframe | 可学习 14D target |
| Run | 使用官方 floor 和 `Run` task class；只替换 robot/keyframe | 可学习 14D target |
| Hurdle | 使用官方 hurdle assets 和 `Hurdle` task class；只替换 robot/keyframe | 历史 baseline 可学习 14D；residual 固定 relaxed zero |
| Basketball | 使用官方 court、ball reset、reward、termination；只替换 robot/keyframe | 可学习 14D target |

这轮历史 parity 不允许自定义 reward、termination、obstacle、ball、board、bar geometry、camera semantics 或 timing。Hurdle residual 继续保持相同 task contract，只改变 reset 与 controller interface。由于 robot/controller 不同，原生 HumanoidBench 数字仍只作为历史 smoke reference，不能直接比较。

### 决策：继续长程训练前先对齐支持的 30-task family

初始修正暴露了几项必须一次解决的 family contract：

- HumanoidBench 的 `robot.dof` 表示 robot `qpos` 长度，不是 actuator 数量。因此所有 task 看到的 SONIC G1 必须是 `dof=50`、`nv=49`、43 个 actuator。继续使用 native G1 的 44 会让 observation 和 reward 把最后六个 robot coordinate 错当成 task state。
- Policy observation 必须使用每个官方 `task.get_obs()`。统一拼接 model state 会丢失 Reach、Push、Cube、Package、Bookshelf、Spoon 的 goal 或 phase，还会让 Kitchen reward 使用过期状态。
- 共享 robot asset 必须一次提供 HumanoidBench 完整的 root、site、sensor 和八个 camera contract；这些名称属于 robot adapter，不应变成 30 个 task patch。
- 初态 parity 指 task-space parity，而不是让 H1 与 G1 joint vector 数值相等。Task object、root placement、feet、palm、contact、COM 与 camera 语义对齐，各本体仍使用自身合法 joint coordinate。

因此实现用同一套 raw `43D` 和 SONIC `78D` interface 注册支持的 30 个 task key。每个支持的 native G1 scene 使用其 keyframe 作为本体初态种子。Room 是 reset 例外：task object 使用 fork 中互不重叠且不穿地的布局，SONIC G1 保留 candidate 的稳定本体姿态，只继承 reference scene 的场景位姿。这是按 joint semantics retarget，不是按 index 粘贴：free root 和 12 个腿关节直接对应；native `torso_joint` 对应 SONIC `waist_yaw_joint`；elbow pitch/roll 对应 elbow/wrist roll；native hand zero–two 对应 thumb，three–four 对应 index，five–six 对应 middle。SONIC 独有的 waist roll/pitch 与 wrist pitch/yaw 保持 neutral；native 44-qpos robot 边界之后的 task-object qpos 则逐值保留。

当前 gate 结果为 30/30 个支持 task 的 task-space 初态通过。High Bar Simple 和 High Bar Hard 明确不在 SONIC G1 支持边界内，因为 G1 手部 morphology 没有经过验证的等价 grasp contract。两者没有 SONIC registry entry、scene、comparison 或 training job。

通用上游 bug 已在 [`mindorigin150/humanoid-bench`](https://github.com/mindorigin150/humanoid-bench) 中分别提交：Kitchen 的 MuJoCo 3.3 default nesting、G1 `free_base` contract、Room 互不重叠且不穿地的 object reset、Truck episode bookkeeping、Truck package density 遗漏、Spoon target phase reset、High Bar reset noise 顺序，以及 Package randomized destination 的 stale observation。主仓库固定该 fork 为 submodule。上游 [issue #62](https://github.com/carlosferrazza/humanoid-bench/issues/62) 独立报告了完全相同的 G1 Truck `free_base` failure；[issue #11](https://github.com/carlosferrazza/humanoid-bench/issues/11) 也记录了 `robot.dof` 包含 free-root qpos、并不等于 actuator 数量这一反复造成误解的语义。我们采纳公开的 [Truck density issue #60](https://github.com/carlosferrazza/humanoid-bench/issues/60) 作为明确的物理合同修复：Truck 十个可搬运 package 的 visual/collision geom 都使用 `density="5"`，与现有 Package task 一致，不再使用 MuJoCo 默认密度。它会改变 Truck 物理，因此此前 Truck 训练结果失效；该修复与 robot/reset parity 分开追踪。

本地使用是显式且方便的，不会污染 HumanoidBench native registry：导入 `latency_bench.envs.humanoidbench_sonic`，选择 `G1_SONIC_ENV_SPECS["g1_sonic_<task>"]`，再把对应生成资产的 `scene_path` 传给 `gym.make`。Raw control 使用 `raw_body_target`；旧 SONIC baseline 使用 `sonic_token`；Hurdle residual 使用 `sonic_token_residual`。三者都固定对应版本的 controller/decoder 资产。

## 手部是独立的控制问题

官方 VLA action 为：

```text
64D body motion token + 7D left hand + 7D right hand = 78D
```

64D body token 不约束 14 个 hand joint。如果 flat RL actor 在 Run 或 Hurdle 中输出不受约束的手指动作，即使 body motion 改善，手仍然可能很怪。公开 deployment path 把 hands 作为独立 command stream；训练时可选的 hand decoder 不会改变这个外部 contract。

旧 raw43/sonic78 task-parity baseline 让四个 actor 都学习完整 14D hand stream。Hurdle residual 是明确例外：该任务不需要手部控制，64D residual actor 之外的 14 个 hand joints 固定为 relaxed zero target；Basketball 等操作任务仍保留可学习 hands。

旧 baseline 的 rollout dataset 和 replay 必须保存完整、实际执行的 78D command；Hurdle residual replay 保存 64D residual。如果其他 task 的 locomotion 已成功但无约束手指让 teacher data 不可用，最小后续方案是显式 hand smoothness/neutral-pose cost，并与 raw task reward 分开报告。

## Sample Factory 集成

### 最小 learner 边界

FastTD3 继续负责 actor、critic、replay 和 optimization。旧 HumanoidBench G1 SONIC baseline 暴露 `Box(78)`，raw baseline 暴露 `Box(43)`；Hurdle residual variant 暴露 `Box(64)`。

旧 SONIC replay 保存完整 78D learnable action。Hurdle residual replay 保存原始 64D residual；inference worker 先与 observation 中的 base token 组合，再把 decoded 29D body target 注入 MuJoCo。两种路径都不能用 decoded 29D target 替代 actor action。

### 吞吐约束

官方 C++/ONNX MuJoCo demo 适合验证正确性。Eval 和 nonzero-latency run 为每个 environment 使用一个进程内、单线程 CPU ONNX Runtime session，因为 decoder 必须处于被延迟的 environment boundary 内。

Zero-latency FastTD3 training 改用 Sample Factory inference worker 内的单个 CUDA ONNX Runtime session。旧 baseline 直接解码 78D actor action 的前 64 维；Hurdle residual 先计算 `base + residual / 16`。两者再与 930D controller state 拼接并解码为 29D body target，而 replay 始终保留 actor 原始 action。部署 graph 已实际通过 batch 1、8、32 和 64；GPU 与 CPU 输出的最大绝对误差小于 `0.01`。

Decoder 必须在每个 50 Hz policy step 接收真实 controller history 和 proprioception。该 target 在本 step 的十个 2 ms MuJoCo substep 中保持不变是预期行为；跨多个 policy step 复用才会改变 SONIC control contract。

长程训练前分别报告：

- raw MuJoCo FPS；
- SONIC decode time 和 batch size；
- end-to-end environment FPS；
- learner FPS 和 GPU utilization。

优化后的 100k 验证达到 5.06k–5.38k 稳态 FPS，超过 3.5k 门槛。短 run 中包含 startup 的 FPS 会被 128 个 environment process 约 40 秒的初始化明显拉低，因此长训报告必须同时给出稳态 FPS 和总 wall-clock FPS。

## Teacher 质量与 rollout data

### 成功是必要条件，但不充分

Teacher checkpoint 必须同时满足以下条件才可使用：

1. 达到 task success 或 raw-return threshold；
2. 达到 motion-quality threshold。

第一轮 token-space 实验至少记录：

- 官方/raw task return 和 episode length；
- 如果引入 shaping，则同时记录 shaped return；
- token magnitude 和 temporal difference；
- decoded target velocity/acceleration；
- actuator effort 或 power；
- locomotion 的 foot slip、contact impact 和 non-foot contact；
- joint-limit margin；
- task-specific failure reason。

这些指标能诊断动作，但不能证明动作分布与人类对齐。真正的 human-like motion 最终需要外部 reference distribution。

### 何时简单 regularization 已经足够

VLA teacher 不一定必须“看起来与真人完全相同”。如果 decoded trajectory 满足以下条件，简单 regularization 就足够：

- 动力学稳定且能完成任务；
- 足够平滑，不会教给 student 抖动或 controller abuse；
- 没有明显利用 contact 漏洞或滥用 joint limit；
- 对任务而言具有足够多样性；
- 面对 closed-loop state deviation 仍然稳健。

SONIC 已经提供 motion prior，因此第一轮 token-space 实验不应再加入 AMP 或 motion tracking。如果高 reward token policy 仍然利用 scene 漏洞，再增加最小的 task-aware penalty，并始终单独保留 raw reward 供报告使用。

### Dataset contract

每个 training sample 应包含：

| 字段 | 来源 | Student 可见？ |
|---|---|---|
| RGB observation | simulated camera | 是 |
| language instruction | task template | 是 |
| robot proprioception | 官方 G1 state contract | 是 |
| 未来 40 × 64D body tokens | teacher execution | target |
| 未来 40 × 14D hand targets | fixed/synergy/task hand policy | target |
| privileged task/object state | simulator | 否；仅用于 debug 和 quality gating |
| raw reward、success、quality metrics | simulator/evaluator | 否；仅用于 filtering 和 analysis |
| timing 和 command-age metadata | latency wrapper | 否；用于 analysis，未来可选作 conditioning |

对于依赖物体和场景的决策，image 必须保持必要性。Privileged object state 可以帮助 teacher，但不能泄漏到 VLA input。

先做 offline behavior cloning。只有 closed-loop evaluation 证明出现 distribution shift 后才增加 DAgger-style data aggregation，不提前搭建。

## 在 StarVLA 中实现 VLA

StarVLA 继续作为 training shell：负责 dataset loading、distributed launch、trainer、W&B logging、checkpointing 和 evaluation。Model 和 processor 应通过 Hugging Face/Isaac-GR00T interface 直接导入 NVIDIA 官方 GR00T N1.7 实现。

不要扩展当前 `QwenGR00T.py` 路径后把它标记成 N1.7。该文件是从早期 N1.5 design 衍生的轻量 Qwen-VL + flow head；当前 HumanoidBench 配置还是 horizon 1、61D action、177D state。N1.7 使用不同的 model、processor、modality configuration 和 checkpoint contract。

预期边界是：

```text
StarVLA data/training/evaluation
        导入
官方 AutoModel/processor + Gr00tN1d7 configuration
        加载
官方 base checkpoint + UNITREE_G1_SONIC post-training initialization
        预测
40 × 78D SONIC-compatible action chunk
```

保留官方 processor、normalization statistics、modality configuration 和 checkpoint layout。除非官方 loading path 经实际尝试后证明不可集成，否则不要复制 model source、重命名 state-dict key 或创建私有 checkpoint conversion。

G1 SONIC post-training checkpoint 只负责对齐 body/action representation，不是这五个任务的 zero-shot solver。Task behavior 仍然来自我们模拟得到的 teacher data。

参考资料：

- [Isaac-GR00T repository](https://github.com/NVIDIA/Isaac-GR00T)
- [Official GR00T policy interface](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/policy/gr00t_policy.py)
- [SONIC model card](https://nvlabs.github.io/GR00T-WholeBodyControl/model_card.html)
- [SONIC quickstart](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/quickstart.html)

## Latency 语义

Benchmark 在 VLA observation/inference boundary 注入 latency，而不是在 50 Hz low-level controller 内部注入。

```text
t0：采集 RGB + proprioception
    VLA inference 尚未完成
    SONIC 继续以 50 Hz 执行上一条 chunk
t1：新的 40-step chunk 到达
    根据 t1 - t0 丢弃已经过时的 prefix
    开始执行剩余 suffix
    如果没有有效 suffix，记录 underflow/fallback event
```

这与公开 SONIC client 的行为一致：删除已经过去的 prefix，而不是重新执行本应在过去发生的 command。40-step chunk 在 50 Hz 下覆盖约 0.8 秒；VLA nominal request rate 约为 2.5 Hz，因此 request 与 low-level execution 在时间上重叠。

测量：

- observation-to-chunk latency；
- 每个 token 执行时的 command age；
- stale-prefix length；
- chunk underflow 次数和持续时间；
- SONIC decode/control latency，与 VLA latency 分开记录；
- task return、success 和 motion quality 随 latency 的变化。

Fallback action 必须在实验定义中明确指定，不能由 adapter 静默隐藏。

## 分阶段实现与决策门

### Gate 0 — contract smoke test

- 导入完全一致的官方 G1 MuJoCo asset 和 SONIC checkpoint。
- 让同一组 token/proprioception sequence 分别经过官方 reference 和仓库集成。
- 比较 decoded target 和短程 MuJoCo rollout。
- 确认历史 baseline 的 64D body + 14D hand contract，以及 Hurdle residual 的 64D residual + fixed-hands contract 和 state history。

只有本地集成与官方 contract 足够一致、能复现 deterministic smoke trajectory 后才能继续。

### Gate 1 — 单任务 token RL proof

- 在 Hurdle 上使用离线官方 Run reference token cache。
- 使用 `Box(64)` FastTD3 residual 和 fixed relaxed hands。
- 验证任务学习、motion video、controller metric 和 end-to-end FPS。

Token RL 能学到有用 locomotion 且吞吐可接受时才继续。如果失败，在增加另一层 abstraction 前先判断瓶颈是 scene mapping、controller expressivity 还是 optimization。

### Gate 2 — 历史四任务 direct-token teacher suite（Hurdle 已由 residual 替代）

- 复用同一个 robot、decoder、action space 和 replay contract。
- 为 Balance、Run、Hurdle 和 Basketball 分别训练 checkpoint。
- 历史 baseline 每个任务都保持相同的可学习 14D hand suffix；当前 Hurdle residual 不适用。
- 同时根据 task success 和 motion quality 选择 checkpoint。

只升级失败的任务。非 Hurdle task 只有在 direct 64D training 已经充分训练，并且完成 contract check 后仍可稳定复现失败，才值得增加 task-specific higher layer。

### Gate 3 — teacher data 与 VLA fine-tuning

- 旧 baseline 导出通过质量门控的 78D rollout；Hurdle residual 导出 64D residual，并显式携带 reference contract。两者都包含 RGB、language 和 proprioception。
- 在 StarVLA 内通过 offline behavior cloning fine-tune 官方 N1.7。
- 在没有 latency 时评估 open-loop chunk prediction 和 closed-loop task success。

只有 closed-loop state 离开 teacher-data distribution 时才增加 data aggregation。

### Gate 4 — latency study

- 注入受控的 observation/inference delay。
- SONIC 保持 50 Hz 执行，同时使用 stale-prefix removal。
- 比较不同 latency distribution 下的 task success 和 motion degradation。
- 报告中分离 VLA delay、controller compute 和 environment throughput。

## 当前不实现的路线

| 路线 | 为什么现在不做 | 何时重启 |
|---|---|---|
| VLA 直接输出 raw 43D/61D joints | 重复学习 low-level control，且已复现不自然的 flat-action incentive | 证明 SONIC token control 本身是限制因素 |
| 在旧 flat actor 上增加 reward regularization | 可以缓解症状，但 teacher 与目标 VLA/SONIC interface 仍不对齐 | 作为 diagnostic 或 fallback baseline |
| 在线 per-task planner/skill selection layer | 训练时引入 774 MB planner 和新的在线状态机没有必要 | 固定离线 Run reference 无法覆盖所需 motion |

## 决策日志

| 日期 | 决策 | 理由/证据 | 放弃的替代方案 | 重启条件 |
|---|---|---|---|---|
| 2026-08-26 | 支持 30 个 task，并明确排除两个 High Bar task | SONIC G1 没有经过验证的等价 High Bar grasp contract | 把 blocked High Bar scene 留在 registry 或训练矩阵 | 批准独立的 robot/task contract |
| 2026-08-26 | 用 task-space 而不是 qpos equality 定义 H1/G1 parity | 两种本体 joint 数量和运动学不同，但 task object 与 contact 可比较 | 按索引复制 H1 或 native G1 qpos | 下游方法明确需要逐 joint retarget demonstration |
| 2026-08-26 | Truck 十个可搬运 package geom 采用 `density="5"` | issue #60 指出 density 遗漏，且 Package task 已有相同项目约定 | 保留 MuJoCo 默认密度的 Truck 物理 | 证据显示目标密度不同，或必须保持未修改物理 |
| 2026-08-26 | 通用 HumanoidBench bug 在项目 fork 中逐问题独立提交 | SONIC adaptation 留在本仓库，同时让 Kitchen、Truck、Room、Spoon、Package、High Bar reset 修复可复用 | 在 asset generator 维护重复 patch | 修复被上游接受，或不再使用该 fork |
| 2026-08-25 | 使用与 SONIC 官方完全一致的 G1 | 避免 23-DoF torque 与 29-DoF position-target contract mismatch | 现在就为原生 G1 或 H1 retarget/retrain SONIC | 官方 G1 无法表达必要任务 |
| 2026-08-25 | 历史决策，已被 Hurdle residual 替代：每个 teacher 首次尝试都使用 direct continuous 64D FastTD3 | 最小的共享 body interface，且与 deployment token 一致 | quantized/MultiDiscrete policy 或直接增加 high-level planner | 非 Hurdle task 仍可保留为 baseline |
| 2026-08-25 | 允许 per-task teacher checkpoint，但保持一种 action contract | 任务不同；共享 wire format 才是有价值的复用 | 没有证据就假设一个 universal actor | multi-task sharing 成为明确实验 |
| 2026-08-25 | Hand 保持在 64D body token 之外 | 官方 VLA wire format 有独立的 7D + 7D hands | 所有任务都输出不受约束的 14D fingers | Basketball 需要超出初始 hand synergy 的能力 |
| 2026-08-25 | 历史决策，已被替代：Balance/Run/Hurdle 使用 fixed relaxed hands，High Bar 使用 fixed grasp，Basketball 使用 scripted grasp | 它简化了第一版 Run-only proof，但不再保留完整 task controllability | 从一开始就训练完整 hand policy | 已在 2026-08-26 重启；parity 实验现在学习全部 14 个 hand joint |
| 2026-08-25 | 历史 baseline 的 teacher data 保存完整、实际执行的 78D action | Student 必须学习真正发送给 controller 的 command | 只保存 actor output 或 decoded 29D target | Hurdle residual 已改为保存 64D residual 与 reference contract |
| 2026-08-25 | 在 StarVLA 中复用官方 N1.7，不扩展 `QwenGR00T.py` | 保留 pretrained model/processor/action contract | 本地复制或近似 N1.7 | 实际尝试后仍无法集成官方 loading path |
| 2026-08-25 | 在 VLA chunk boundary 定义 latency，并移除 stale prefix | 与官方 deployment logic 一致，也能隔离 high-level latency | inference 时冻结整个 controller，或执行 stale command | 选择不同 deployment architecture |
| 2026-08-26 | 第一版只实现 Run，使用 fixed relaxed hands，并让每个 environment 使用一个 CPU ONNX Runtime session | 最小路径已经足以验证 64D token、decoder、controller、FastTD3 checkpoint 和 deterministic video pipeline | 未测量前就构建 centralized decoder service | profiling 证明 decoder execution 或 duplication 是主要瓶颈 |
| 2026-08-26 | Latency benchmark 使用 `0.002 × 10` dynamics | 对齐 500 Hz deployment control loop，同时保留 SONIC 的 50 Hz policy interval | 复制 training simulator 的 `0.005 × 4` dynamics | deterministic comparison 证明该 dynamics 选择妨碍 SONIC 产生有效行为 |
| 2026-08-26 | Zero-latency training 在 Sample Factory inference worker 中用 CUDA batch 解码 SONIC | 单 decoder session 和 GPU I/O binding 把稳态吞吐提升到 5.06k–5.38k FPS；首个 Run-only 实验保存 64D，修正后保存 78D | 每个 environment 一个 CPU session，或独立 decoder service | nonzero latency 要求 decoder 在 environment boundary 内执行 |
| 2026-08-26 | 历史决策，部分保留：G1 资产放入固定 revision 的公开 Hugging Face dataset，官方 decoder 单独固定 revision | 存储方式仍合理，但首个 asset revision 无效 | 在 Git 中 vendor 资产，或仅为存储资产 fork HumanoidBench | corrected local assets 通过 review 后替换 dataset 默认 revision |
| 2026-08-26 | 历史 parity 对比 raw `43D` 与 SONIC `78D`，两边都学习相同 14 个 hand joint | 保留官方 task controllability，避免 ignored/scripted action，同时只隔离 29D-body 与 64D-token 差异 | fixed hands 或仅 Basketball 使用 scripted synergy | Hurdle residual 已改用 fixed hands |
| 2026-08-26 | 历史决策，已被 Hurdle residual 替代：四任务统一使用 direct continuous 64D body-token control | Hurdle 10M 虽优于 raw43，但动作仍不对且 token 大量饱和 | 现在就加 task planner 或 high-level skill interface | 非 Hurdle task 仍可保留为 baseline |
| 2026-08-26 | Hurdle 使用离线官方 Run planner → encoder/FSQ reference token，加 `1/16` FastTD3 residual | 让 decoder 接收实际 motion-manifold reference，并让 actor/critic/replay 看见 decoder state 与 reference token | 继续在完整 64D cube 内从零学习；在线运行 774 MB planner | residual 仍无法跨杆时，重新审计 reference motion 或 reward |

## 待回答的实验问题

这些是待测量的问题，不是新基础设施承诺：

1. Hurdle 的 offline Run reference + 64D residual 能否同时改善跨杆表现与 motion quality？
2. Basketball 能否仅凭官方 sparse task signal 学会完整 14D hand stream？
3. 官方 task definition 能否按计划对 G1 完全不改，还是某个任务会暴露 morphology-specific impossibility？
4. 使用 simulated RGB 和 privileged-teacher rollout 时，offline N1.7 fine-tuning 是否稳定？
5. 每个任务在多大的 command age 或 stale-prefix length 下失败？

每个 gate 失败后，都应先用证据更新决策日志，再引入新算法或 abstraction。
