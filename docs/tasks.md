# Latency-Sensitive Embodied Tasks 调研

**状态：** Research snapshot

**调研日期：** 2026-08-22

**范围：** 适合 latency-sensitive benchmark 的 embodied task、environment suite、benchmark 与 simulator 候选。

---

## 1. 结论

应该先选 task，再决定 simulator。适合本项目的任务必须满足：action 的应用时机会改变任务结果，而不是只让 episode 晚一点完成。最明确的 latency-sensitive机制包括：

1. 外部物体或障碍物自主运动；
2. interception、release、impact、handover 等短暂动作窗口；
3. locomotion 或 whole-body balance 的不稳定动力学；
4. deformable、granular、tactile 或 force-sensitive contact；
5. policy 推理期间其他 agent 仍然运动的 multi-agent interaction。

目前没有一个现成 benchmark 同时满足：

```text
轻量安装
+ 社区广泛使用
+ 内在的高速动态
+ 标准 VLA action space
+ 官方 language-labelled demonstrations
```

因此，不同实验目标应由不同任务承担：

- **Latency headline：** MyoChallenge Table Tennis、DeepMind Control Suite moving-ball tasks，以及可选的 DynamicVLA。
- **Whole-body / locomotion：** HumanoidBench、LocoMuJoCo。
- **Dexterous / VLA pipeline：** Gymnasium-Robotics Adroit/Shadow Hand、 robosuite。
- **Dynamic navigation：** HighwayEnv；PyFlyt 作为 aerial-control 扩展。
- **High-fidelity optional track：** 精选 Isaac Lab task，而不是让 Isaac 成为所有 embodied task 的必选依赖。

如果首先实现一个轻量 embodied backend，候选任务明显收敛到 **continuous Gymnasium + MuJoCo**。Gymnasium 应作为接入边界，MuJoCo 只是其背后的一个 simulator。只有当具体任务的科学价值足够高时，才额外启用 Isaac Lab plugin。

当前 [`gymnasium_adapter.md`](gymnasium_adapter.md) 的 release path 已支持
native `Discrete` 和 bounded continuous `Box` action。本文大部分候选使用
continuous action；只要环境暴露稳定的 Box/vector 与 policy-compatible
observation，就可以沿用 generic Gymnasium adapter、latency executor 和
continuous rollout/export contract。structured action、action chunk 和后续
VLA model/data support 仍需按任务单独确认。

---

## 2. Simulator、framework、task 与 benchmark

| 层级 | 职责 | 示例 |
| --- | --- | --- |
| Simulator/runtime | 物理、渲染、传感器和场景状态 | MuJoCo、PyBullet、Isaac Sim、SAPIEN |
| Environment framework/API | `reset`/`step`、vectorization、task authoring、wrapper | Gymnasium、dm_control、Isaac Lab、PettingZoo |
| Task suite | 机器人、场景、reward、termination 和注册任务 | MyoSuite、HumanoidBench、LocoMuJoCo、Gymnasium-Robotics |
| Benchmark | 固定任务、数据、指标、split 和 evaluation protocol | MyoChallenge、DOMINO、DynamicVLA DOM、D4RL/Minari |

例如：

```text
MyoChallenge Table Tennis    benchmark/task
MyoSuite                     task suite
MuJoCo                       simulator
Gym-style API                environment interface
```

```text
DynamicVLA DOM               benchmark/task package
Isaac Lab                    environment framework
Isaac Sim + PhysX            simulator/runtime
```

Gymnasium 不是 simulator，而是让不同 task package 和 simulator 可以被同一
executor 使用的接口标准。

---

## 3. 筛选标准

| 维度 | 核心问题 |
| --- | --- |
| Intrinsic latency sensitivity | delay 是否改变 success、safety、collision、drop 或 force，而不只是 completion time？ |
| Environment dynamics | 与任务相关的状态是否会在 agent 不动作时继续变化？ |
| VLA suitability | 是否有 RGB observation 和可处理的 action interface？ |
| Demonstration readiness | 是否有 expert data、checkpoint、baseline 或 replay 工具？ |
| Integration cost | 是否 Gymnasium-compatible，且无需大型 simulator application？ |
| Maturity | 是否仍在维护，并且有单一研究 prototype 之外的使用者？ |

文中的 High / Medium / Low 表示：

- **High：** latency sensitivity 是 task 核心机制的一部分；
- **Medium：** 有合理机制，但必须通过 latency sweep 验证；
- **Low：** 很可能只体现为 command lag 或更长 completion time。

不能因为环境包含 physics 或 contact 就直接纳入 benchmark。所有正式 task 都
必须通过[第 11 节](#11-latency-sensitivity-screening-gate)的实验筛选。

---

## 4. 推荐候选总表

| 优先级 | Task / suite | Runtime/API | Timing-sensitive 机制 | Data / VLA readiness | 建议定位 |
| --- | --- | --- | --- | --- | --- |
| S | MyoChallenge Table Tennis P2 | MyoSuite、MuJoCo、Gym-style | 随机来球具有很短的 interception window | 有 baseline 和渲染；muscle action 维度很高 | Latency headline |
| S | dm_control `dog.fetch` / `quadruped.fetch` | MuJoCo、dm_env；可用 Shimmy | 被抛出或移动的球在 agent 运动和接触期间持续变化 | 经典 control suite；需自行训练 teacher | Moving-object control |
| S | dm_control `ball_in_cup.catch` | MuJoCo、dm_env | 20 ms control task，使用 sparse catch success | 小、稳定、容易复现 | Reference/smoke benchmark |
| A | HumanoidBench balance、hurdle、run、basketball | MuJoCo、Gymnasium | 跌倒、步态相位、跨栏和 release timing 不可逆 | 支持 proprio/image/tactile，有训练代码和 low-level skill | Whole-body benchmark |
| A | MyoSuite Baoding Balls / Pen Twirl | MuJoCo、Gym-style | 多点接触和物体运动需要持续相位对齐 | 有 challenge policy 和 baseline | Dexterous dynamics |
| A | LocoMuJoCo H1/G1 run 或 gait imitation | MuJoCo/MJX、Gymnasium wrapper | delay 导致 gait/reference phase error 和跌倒 | 大量 motion data 和 IL baseline | Locomotion demonstrations |
| A- | Adroit Hammer/Pen、Shadow Hand touch | MuJoCo、Gymnasium | impact、reorientation、tactile feedback、drop prevention | Adroit 有 D4RL/Minari trajectory | Dexterous/VLA pipeline |
| A- | robosuite TwoArmHandover/Wipe | MuJoCo、Gym wrapper | handover 与 force/contact timing 可能改变 failure/safety | RGB/depth、OSC action、demo/replay 工具 | VLA-compatible manipulation |
| A- | HighwayEnv Intersection/Merge/Roundabout | 2D dynamics、Gymnasium | 其他车辆持续运动，安全决策窗口会关闭 | 有 RGB，Gym 生态成熟 | Dynamic navigation |
| B | PyFlyt Rocket Landing/Fixed-Wing Dogfight | PyBullet、Gymnasium/PettingZoo | 飞行状态不稳定，地面和对手持续接近 | 可渲染 RGB，缺少成熟 demo 数据 | Aerial extension |
| Optional | DynamicVLA DOM | Isaac Lab on Isaac Sim | dynamic object 和 target configuration 在 policy 执行时持续变化 | 有 DOM 数据、场景、模型和 VLA evaluation | High-fidelity VLA headline |
| Optional | SoftVTBench | Isaac Lab on Isaac Sim | slip、drop、deformation 依赖接触时机 | 1,628 demonstrations，visual+tactile | Contact-safety headline |

这里的排序只表示对本项目 setting 的适合度，不代表 S-tier task 更容易训练。
MyoChallenge 和 HumanoidBench 都有高维 action space，可能需要 hierarchical
controller 或 task-specific VLA action head。

---

## 5. 轻量 MuJoCo / Gymnasium 候选

### 5.1 MyoSuite / MyoChallenge

[MyoSuite](https://github.com/MyoHub/myosuite) 是 Apache-2.0 的 MuJoCo task
suite，使用 Gym-style API，并持续用于 NeurIPS MyoChallenge。

#### Table Tennis

推荐 ID：

```text
myoChallengeTableTennisP1-v0
myoChallengeTableTennisP2-v0
```

P2 会改变来球位置和速度、球拍质量及摩擦。成功要求球拍只击球一次，并让球
落在对面球台；episode 最长三秒。几十毫秒的 late action 可以直接把有效回球
变成 miss 或 out-of-bounds，而不是简单地推迟成功时间。因此它是目前最明确的
轻量 latency headline。

来源：[MyoChallenge 2025 文档](https://myosuite.readthedocs.io/en/stable/challenge-doc2025.html)。

主要限制是 action interface：policy 控制数百维 muscle activation，而不是标准
6/7-DoF end-effector command。它暴露的是可直接保留的 native continuous Box，
因此适合 privileged RL teacher；标准 VLA 仍可能需要 hierarchical action
representation 或额外的 continuous action head。

#### Soccer Shootout

推荐 ID：

```text
myoChallengeSoccerP1-v0
myoChallengeSoccerP2-v0
```

任务同时包含接近足球、whole-body balance、击球时机和 moving goalkeeper。
它同样有很强的 timing sensitivity，也同样存在高维 muscle action 与标准 VLA
action 不匹配的问题。

#### Dexterous tasks

推荐 ID：

```text
myoChallengeBaodingP2-v1
myoHandPenTwirlRandom-v0
myoChallengeDieReorientP2-v1
myoChallengeChaseTagP1-v0
```

Baoding Balls 最值得优先考虑：两个自由球必须持续相对旋转且不能掉落。
Pen Twirl 和 Die Reorientation 是单物体、间歇接触的替代任务。
[MyoSuite task catalogue](https://myosuite.readthedocs.io/en/stable/suite.html)
还提供 rough、hilly、stair locomotion。

#### Baoding Balls

推荐先探测并冻结以下两个 variant：

```text
myoChallengeBaodingP1-v1
myoChallengeBaodingP2-v1
```

它们都可以通过 `LatencyBench/MyoSuite-v0` 的共享 RGB/lifecycle wrapper
接入；P1 主要改变球的位置，P2 还引入目标、尺寸和摩擦的变化。动作、观测
shape、实际 Box bounds、no-op、native horizon 和 MuJoCo clock 必须从安装的
MyoSuite 版本 probe 后写入 task contract。不要从文档中的 `[0, 1]` 描述推断
运行时 bounds，也不要把 sparse/dense reward 或成功指标放进共享 wrapper。

部分年份有公开 winning policy 和训练代码，例如
[2022 Baoding Ball solution](https://github.com/amathislab/myochallenge)。这些
可以作为 teacher 起点，但不是 language-labelled VLA demonstrations。

### 5.2 DeepMind Control Suite

[dm_control](https://github.com/google-deepmind/dm_control) 是经典的
Apache-2.0 MuJoCo control suite。原生 API 是 `dm_env`，但可通过
[Shimmy](https://shimmy.farama.org/environments/dm_control/) 转为 Gymnasium。

#### Ball in Cup

```python
suite.load("ball_in_cup", "catch")
```

该 task 的 control timestep 为 20 ms，球的初始位置随机，并在球进入 cup 时
给出 sparse reward。它足够小，适合成为 continuous action delay、stale
observation、action hold 和 jitter 的 reference implementation。

来源：[`ball_in_cup.py`](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/ball_in_cup.py)。

#### Quadruped / Dog Fetch

```python
suite.load("quadruped", "fetch")
suite.load("dog", "fetch")
```

球带初始速度并继续运动，机器人需要接近、接触并把球带向目标。相较于平地
velocity tracking，这类 task 有不可通过延长 episode 弥补的 object interaction
window，因此更适合正式 latency claim。

来源：
[quadruped task](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/quadruped.py)、
[dog task](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/dog.py)。

#### Soccer / locomotion

dm_control 还提供
[multi-agent soccer](https://github.com/google-deepmind/dm_control/blob/main/dm_control/locomotion/soccer/README.md)、
humanoid run 和 terrain escape。Soccer 的科学价值高，但 multi-agent protocol
和集成面都比单智能体 suite task 更复杂。

这些 task 没有随包提供 VLA demonstrations，需要训练 privileged small teacher，
并在 rollout 时渲染 RGB。

### 5.3 HumanoidBench

[HumanoidBench](https://github.com/carlosferrazza/humanoid-bench) 是 MuJoCo
whole-body benchmark，包含 15 个 manipulation task 和 12 个 locomotion task。
它使用 Gymnasium，并支持 privileged state、egocentric image 和 tactile
observation；仓库提供 PPO、SAC、Dreamer、TD-MPC2 和部分 low-level skill。

推荐 task：

```text
h1hand-balance_hard-v0
h1hand-hurdle-v0
h1hand-run-v0
h1hand-basketball-v0
h1strong-highbar_hard-v0
```

- **Balance hard：** delay 会造成 fall 或不可恢复的 support error；
- **Hurdle：** foot-contact phase 必须与跨越障碍对齐；
- **Run：** 高速 gait 比普通 walk 更敏感；
- **Basketball：** release timing 改变 ballistic trajectory；
- **High bar：** grasp、swing 和 body phase 必须同步。

HumanoidBench 显式支持 image/tactile，因此比 MyoChallenge 更适合作为 VLA
输入端。它的 whole-body action 仍然维度很高，而且完整 H1/G1 模型比标准
Gymnasium MuJoCo task 更昂贵。

来源：
[official repository](https://github.com/carlosferrazza/humanoid-bench)、
[project page](https://humanoid-bench.github.io/)。

### 5.4 LocoMuJoCo

[LocoMuJoCo](https://github.com/robfiras/loco-mujoco) 是基于 MuJoCo、MJX、
MjWarp 的 imitation-learning benchmark，提供 Gymnasium interface、humanoid /
quadruped model、motion retargeting 和 PPO/GAIL/AMP/DeepMimic 示例。

推荐 task：

```text
Unitree H1 walk/run
Unitree G1 walk/run
Unitree Go2 locomotion
ANYmal / Spot locomotion
```

它对本项目最大的价值是数据：提供 default、LAFAN1、AMASS motion source，
并为部分 environment 提供带 action 的 perfect/suboptimal trajectory。Latency
sensitivity 来自 gait phase、contact transition、reference trajectory alignment
和 fall termination。

reference trajectory 必须按 environment time 推进。如果 wrapper 把 reference
和 action 一起延迟，就会把真正需要测量的 phase error 消掉。

来源：
[official repository](https://github.com/robfiras/loco-mujoco)、
[documentation](https://loco-mujoco.readthedocs.io/en/latest/)。

### 5.5 Gymnasium-Robotics

[Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)
由 Farama 维护并使用原生 Gymnasium API，是接入成本最低的 dexterous 方案。

推荐 task family：

```text
AdroitHandHammer
AdroitHandPen
HandManipulateEgg + continuous touch sensors
HandManipulatePen + continuous touch sensors
```

Hammer 提供 impact 和 nail/contact state；Pen 和 Shadow Hand 提供 in-hand
reorientation 与 drop failure；Shadow Hand touch variant 还提供 continuous
contact-force sensing，可单独研究 tactile/observation latency。

来源：[Gymnasium-Robotics task catalogue](https://robotics.farama.org/)。

Adroit 在 Farama
[Minari D4RL collection](https://huggingface.co/datasets/farama-minari/D4RL)
中有 human、expert 和 cloned trajectory。这些数据适合作为 baseline，但不是
language-and-image VLA data。

这些任务的外部世界基本静止，必须先证明 delay 会稳定影响 success、drop 或
force，才能用于正式 latency claim。

### 5.6 robosuite

[robosuite](https://github.com/ARISE-Initiative/robosuite) 是 MuJoCo
manipulation framework，提供 RGB、depth、proprioception、operational-space
controller、demonstration recording 和 Gym wrapper。

推荐 task：

```text
TwoArmHandover
Wipe
TwoArmPegInHole
ToolHang
```

`TwoArmHandover` 有 ownership-transfer window，`Wipe` 可以体现 force/contact
quality。其 OSC action 比 MyoChallenge/HumanoidBench 的原始 action 更接近常见
VLA 接口。但外部场景仍然较静态，所以更适合证明 VLA data pipeline，而不是
单独承担 latency headline。

### 5.7 Standard Gymnasium MuJoCo

安装方式：

```bash
pip install "gymnasium[mujoco]"
```

适合做 control baseline 的 task：

```text
Ant-v5
Humanoid-v5
Walker2d-v5
Hopper-v5
InvertedPendulum-v5
```

它们容易训练、容易复现，适合验证 continuous action queue 和 fall
termination，但不包含 moving object、language 或任务多样性，不应作为
embodied/VLA headline。

来源：[Gymnasium MuJoCo environments](https://gymnasium.farama.org/main/environments/mujoco/)。

### 5.8 MuJoCo Playground

[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
是基于 MJX/MuJoCo Warp 的 GPU/JAX task framework，包含 quadruped、humanoid、
dexterous hand、arm 和 vision task，例如 rough-terrain locomotion 与 in-hand
reorientation。

它远轻于 Isaac Sim，但原生 API 不是 Gymnasium。其 functional JAX state 和
batched action queue 应由专门的 JAX backend 支持，不应该强行塞入同步 NumPy
Gym adapter，因此属于后续扩展而不是第一批接入。

---

## 6. 轻量 navigation / aerial 候选

### 6.1 HighwayEnv

[HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) 是成熟、MIT
licensed 的 Gymnasium autonomous-driving environment。

推荐：

```text
intersection-v0
merge-v0
roundabout-v0
highway-v0
```

其他车辆会持续运动，delayed action 可能错过 merge gap、造成 intersection
collision 或迫使车辆停车。simulation frequency 和 policy frequency 可以分别
配置，因此 action hold / delay 语义容易审计。

它很轻并支持 RGB，但 vehicle model 和可视化是 2D。适合作为 dynamic-navigation
和 protocol task，不是 high-fidelity robotics benchmark。

### 6.2 PyFlyt / gym-pybullet-drones

[PyFlyt](https://github.com/jjshoots/PyFlyt) 提供 PyBullet-based Gymnasium /
PettingZoo environment，覆盖 quadrotor、fixed-wing、dogfight 和 rocket landing。
Rocket Landing 与 Fixed-Wing Dogfight 中 vehicle、ground 和 opponent 都持续变化，
因此 action timing 很重要。

[gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) 是已有
IROS 论文的 task framework，显式区分 physics/control frequency，并支持
RGB/depth/segmentation、multi-drone、downwash 和 firmware/SITL。现成 learning
task 主要是 hover/velocity tracking；若要承担强 latency claim，需要再选或增加
dynamic target / racing task。

### 6.3 VMAS

[VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) 是 PyTorch
向量化 2D simulator，提供 Gymnasium、RLlib、TorchRL 和 PettingZoo interface。
相关 scenario 包括 football、road traffic、flocking、transport、ball trajectory
和 give-way。

它适合 batched multi-agent timing 和 stale-observation 实验，但仍偏 synthetic
multi-agent environment。GPL-3.0 license 也要求单独审查 release boundary。

### 6.4 Crowd navigation

[CrowdNav](https://github.com/vita-epfl/CrowdNav) 与
[gym-collision-avoidance](https://github.com/mit-acl/gym-collision-avoidance)
提供轻量 dynamic-human/multi-agent collision avoidance。科学问题相关，但软件栈
较旧，输入主要是 structured 2D state。第一版 navigation 集成优先 HighwayEnv。

---

## 7. Isaac 生态候选

Isaac Sim 是 simulator/runtime；Isaac Lab 是提供 environment abstraction、GPU
vectorization、task manager 和 RL/IL tooling 的 robot-learning framework。一个
项目可以在 teaser 中写 Isaac Sim，同时使用 Isaac Lab 实现 task。

Isaac Lab 官方 catalogue 通过 `Isaac-*` 与 `IsaacContrib-*` 区分 core task 和
支持级别较低的 contributed task：
[official environment list](https://isaac-sim.github.io/IsaacLab/develop/source/overview/environments.html)。

### 7.1 DynamicVLA DOM

[DynamicVLA](https://github.com/hzxie/DynamicVLA) 是使用 Isaac Lab
`ManagerBasedRLEnv` API、运行在 Isaac Sim 4.5 上的外部 task package，不是
NVIDIA 内置 task。

优势：

- 目标是 dynamic-object manipulation，而不是普通静态 pick/place；
- 有 DOM train/test data、scene、object 和 pretrained model；
- 有 RGB observation 和 continuous robot action；
- simulation 与 inference 已分进程。

本项目应保留 task、asset、reset 和 metric，并替换它自己的 policy scheduler。
target-device inference 期间 environment 必须继续推进。

代码使用限制商业用途的
[S-Lab License 1.0](https://github.com/hzxie/DynamicVLA/blob/master/LICENSE)，
发布和商业使用前必须审查。

### 7.2 SoftVTBench

[SoftVTBench](https://github.com/TuojingAI/SoftVTBench) 是外部 Isaac Lab
visuo-tactile benchmark，覆盖 rigid/deformable manipulation。当前 release 包含
四类 task、1,628 demonstrations、33 assets、20 Hz synchronized data，并把
Goal Success 与 Safety Success 分开报告。

它的价值在于：即使 nominal goal 达成，delay 仍可能造成 slip、drop 或
over-deformation。源码是 Apache-2.0，但 dataset、USD asset 和 tactile runtime
asset 有独立条款。

### 7.3 IsaacLabEvalTasks

[IsaacLabEvalTasks](https://github.com/isaac-sim/IsaacLabEvalTasks) 是 NVIDIA
Isaac organization 提供的外部 industrial task package：

- **Nut Pouring：** 抓 beaker、把 nut 倒入 bowl、再把 bowl 放到 scale；
- **Exhaust Pipe Sorting：** 双手 transfer pipe，并放入 bin。

它包含 first-person RGB、dataset、checkpoint、GR00T post-training 和 closed-loop
evaluation。Pouring 与 bimanual transfer 有合理的 timing-sensitive 机制，但任务
只有两个，并且没有官方 latency protocol。

### 7.4 Official Isaac Lab tasks

推荐 screening 的原生 task：

```text
Isaac-Shadow-Handover / Isaac-Shadow-Handover-Direct
Shadow / Allegro in-hand reorientation camera variants
IsaacContrib-Navigation-3DObstacles-ARL-Robot-1
IsaacContrib-Tracking-LocoManip-Digit
Isaac-Velocity-Rough-UnitreeGo2/G1/H1
Isaac-Lift-Cloth-Franka-Camera
Isaac-Lift-Cable-Franka-Camera
IsaacContrib-Franka-Pour
```

- Shadow handover/reorientation 有 object drop 和 contact timing；
- rough locomotion / Digit loco-manip 有 balance 和 gait phase；
- ARL drone 有快速 relative motion 和 depth perception，但障碍物本身静止；
- cloth、cable、granular task 有持续 object dynamics，但部分依赖 experimental
  physics backend。

Factory、AutoMate、FORGE 容易使用且 contact-rich，但外部世界静止。它们可以
进入 screening pool，不能未经验证就成为 headline。

### 7.5 Isaac Lab-Arena ecosystem

[IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) 在 Isaac Lab 上增加
scene/task/embodiment/evaluation 组合层，包含 G1 loco-manip、open microwave、
refrigerator sequence 等 example。Arena 明确是 alpha software，API 不稳定；
核心 latency plugin 应依赖 Isaac Lab，仅提供可选的 thin Arena bridge。

[LW-BenchHub](https://github.com/placeholder/LW-BenchHub) 和
[RoboLab](https://github.com/NVlabs/RoboLab) 提供大量 language/kitchen task 和
有用的 policy server/client interface，但大多数任务是静态 rearrangement。任务
数量不能替代 latency sensitivity。

---

## 8. Migration candidates

### 8.1 DOMINO

[DOMINO](https://github.com/H-EmbodVis/DOMINO) 是本次调研中动态 manipulation
覆盖最广的 benchmark：35 tasks、110K+ expert trajectories、moving target、
history、prediction 和多个 dynamic level。

它基于 SAPIEN/RoboTwin，而不是 Isaac Lab 或 MuJoCo。通过独立 SAPIEN backend
运行官方环境，比声称可以直接 wrap 为 Isaac task 风险更低。若要 port，必须
重写 actor、controller、physics-step motion、reset 和 success logic。

官方
[RoboTwin IsaacLab-Arena branch](https://github.com/RoboTwin-Platform/RoboTwin/tree/IsaacLab-Arena)
可以作为 migration template，但当前只有部分静态任务，没有 DOMINO dynamic
workflow。

### 8.2 Bi-DexHands / Dexterous Gym

[Bi-DexHands](https://github.com/PKU-MARL/DexterousHands) 是成熟的 Isaac Gym
benchmark，包含 catch、throw、handover 和 two-object exchange。
`ShadowHandCatchUnderarm` 是最值得单独迁移的 task，因为 late action 会直接错过
flying object。迁移到 Isaac Lab 需要 DirectRLEnv rewrite，不是 wrapper-only
integration。

[Dexterous Gym](https://github.com/henrycharlesworth/dexterous-gym) 包含
MuJoCo/Gym PenSpin、handover、underarm/overarm catch 和 two-object catch。
任务语义非常匹配，但代码依赖 Gym 0.15 和 `mujoco-py`，用户规模较小，也没有
清晰的 repository license。它适合作为 task reference，不适合作为依赖。

### 8.3 DynamicVLA 与 DOMINO 对比

| 属性 | DynamicVLA DOM | DOMINO |
| --- | --- | --- |
| Runtime | Isaac Lab on Isaac Sim | SAPIEN/RoboTwin |
| 当前接入方式 | 外部 package 已使用原生 Isaac Lab API | 需要独立 backend 或完整 port |
| 数据 | DOM data、scene、object、model | 35 tasks、110K+ expert trajectories |
| 主要价值 | 低迁移成本的 high-fidelity dynamic VLA task | 动态 manipulation 语义覆盖广 |
| 主要风险 | non-commercial license、runtime 重 | migration cost 高 |

---

## 9. 已有 delay benchmark：Real-World RL Suite

Google
[Real-World RL Suite](https://github.com/google-research/realworldrl_suite)
已经在 dm_control task 上实现：

- action、observation、reward delay；
- dropped、stuck、repeated action/observation；
- non-stationary physical perturbation。

它把 action delay 定义为 action 发出到执行之间相隔的 environment step 数。它是
必须引用的 related work，也可以作为 delay semantics 参考，但不应成为新的
runtime dependency：

- 使用配置的 step delay，而不是 target-device 实测 latency distribution；
- 目标是 RL robustness，不包含 demonstration regeneration；
- 没有 VLA fine-tuning pipeline；
- repository 已于 2026-04-19 archived，example 还包含旧 TensorFlow 依赖。

本项目的区别是：

```text
Real-World RL Suite
  configured step delay
  -> evaluate RL robustness

This benchmark
  target-device latency profile
  -> train a latency-exposed small teacher
  -> regenerate demonstrations under the profile
  -> fine-tune a VLA
  -> evaluate whether the data produces latency-aware behavior
```

这个区别必须通过 zero-latency re-rollout control 证明，不能只写在论文文字里。

---

## 10. 不应承担主 claim 的候选

### Meta-World、LIBERO、RoboCasa、ordinary pick-and-place

它们适合 language/multi-task manipulation，但大多数场景外部静止。即使 task
名字是 basketball 或 soccer，通常也只是抓取或推动初始静止的球。除非 delay
sweep 证明存在不可逆 timing failure，否则只能作为 compatibility/control task。

### Standard ManiSkill tabletop tasks

ManiSkill 适合 GPU visual data generation，但多数标准 tabletop task 静态。
RollBall、PushT 和 dexterous reorientation 可以进入 screening；普通 pick/place
不能仅因使用 physics 就被称为 latency-sensitive。

### Safety Gymnasium

这是成熟的 safe-RL benchmark，但大多数 hazard/goal 静止。它适合 cost-aware
control，不自动等于 dynamic latency benchmark。

### MiniGrid / PettingZoo

Dynamic Obstacles、Waterworld、Pursuit、Pistonball 是轻量 timing test，但仍偏
2D/game-like，并与现有 gaming track 重叠。适合 CI 和 multi-agent scheduler
测试，不适合 embodied headline。

### Habitat、AirSim、Webots、RLBench

它们需要大量 scene asset、独立 simulator application、Unreal/Coppelia/desktop
runtime 或大量 task-specific integration，无法解决本次调研强调的 lightweight
dependency 问题。

### Legacy Brax environments

旧 `brax.envs` 已不是 Google 推荐的机器人环境路径。新的 GPU/JAX 工作应基于
MuJoCo Playground、MJX 或 MuJoCo Warp，不应新增 legacy Brax backend。

---

## 11. Latency-sensitivity screening gate

每个候选都必须通过相同 screening，才可以进入正式 benchmark。

### 固定条件

- 同一个 high-performing teacher checkpoint；
- 固定 episode seed、initial state、object trajectory 和 task manifest；
- 固定 observation/control rate；
- 相同 action representation 和 low-level controller；
- 相同 episode time limit 和 success definition。

### 测试条件

至少包含：

```text
zero latency
one or more fixed delays
jittered delay
measured target-device profile
```

inference pending 时 environment timeline 必须继续推进。等待模型返回后才调用
下一次 `env.step()` 的同步 loop 不构成 latency-sensitive evaluation。

### 接受证据

delay 必须稳定、可重复地影响至少一个 task-semantic outcome：

- success/catch rate；
- fall、collision、drop、missed-interception rate；
- safety success / excessive deformation；
- contact/impact force / tactile safety；
- landing region、dynamic clearance 等 completion-quality metric。

如果唯一稳定变化只是 wall-clock 或 simulated completion time 变长，该 task
不能作为 latency headline。

最终 learning claim 至少比较：

```text
M_official  official demonstrations（存在时）
M_zero      同一 small teacher 在 zero latency 下重新 rollout
M_latency   同一 teacher family 在 target profile 下重新训练/rollout
```

三者使用相同 VLA initialization、data budget、seed 和 target-device evaluation
profile。核心比较是 target profile 下的 `M_latency` versus `M_zero`。

---

## 12. 推荐 release task pack

### Lightweight core

| Track | 首选 task | 原因 |
| --- | --- | --- |
| Reference | dm_control `ball_in_cup.catch` | 小、20 ms control step、sparse catch success |
| Dynamic object | dm_control `dog.fetch` / `quadruped.fetch` | moving ball + locomotion + contact |
| Sports headline | MyoChallenge Table Tennis P2 | 随机短 interception window，success 清晰 |
| Whole-body | HumanoidBench balance hard / hurdle | 不可逆 fall/contact failure，支持 image |
| Locomotion data | LocoMuJoCo Unitree H1 run | 有 motion data，gait-phase sensitive |
| Dexterous data/VLA | Adroit Hammer / Shadow Hand touch | Gymnasium-native，有 expert trajectory |
| Dynamic navigation | HighwayEnv Intersection | 其他 agent 持续移动，decision gap 会关闭 |

### Optional high-fidelity track

| Track | Task | 原因 |
| --- | --- | --- |
| Dynamic VLA | DynamicVLA DOM | 外部 Isaac Lab native API task，有 VLA asset |
| Deformable/tactile | SoftVTBench | safety metric 可体现 slip/drop/deformation |
| Industrial | IsaacLabEvalTasks Nut Pouring | granular + bimanual interaction，有 GR00T workflow |

### Later migration

1. Bi-DexHands `ShadowHandCatchUnderarm`，补充 flying-object dexterity；
2. 如果战略上必须统一到 Isaac，再迁移小规模 DOMINO Level-1 subset；
3. 为 GPU-parallel training 新增独立 MJX/MuJoCo Playground backend。

---

## 13. 对当前 codebase 的接入含义

轻量 embodied 路径应为：

```text
task package registers/exposes an environment
  -> continuous Gymnasium compatibility boundary
  -> latency-aware executor and action queue
  -> raw simulator/control ticks
  -> canonical rollout and demonstration export
```

task adapter 只负责：

- task/environment ID 与 pinned version；
- registration import 和 construction argument；
- observation/render extraction；
- native continuous action spec 和 controller level；
- success/failure/task-quality metric；
- fixed episode manifest。

共享 benchmark 负责：

- observation issue、inference completion、action-ready、action-apply timestamp；
- per-environment action queue 与 hold policy；
- fixed、jittered、measured-profile latency source；
- environment 独立于 inference 推进的 realtime execution；
- canonical data/evaluation metadata。

对于 hierarchical environment，delay 必须作用在 policy 所控制的层级：

- locomotion controller 之前的 high-level velocity command；
- IK/OSC/WBC 之前的 end-effector target；
- low-level actuator/muscle controller 之前的 VLA action。

如果 delay 施加到 controller 之后的 actuator command，测量的是 controller/bus
latency，而不是 VLA inference latency。

当前 repository 已经分离 realtime environment/inference worker，这是正确的
execution model。Gymnasium boundary 已支持 native continuous Box/vector；正式
接入更多任务仍需逐项确认 structured action、first-class action chunk、
continuous StarVLA model/data support，以及 gaming path 之外的 task metric。
这些能力应在 Gymnasium boundary 统一实现，而不是按 simulator 重复实现。

---

## 14. 官方来源索引

### Lightweight ecosystems

- [Gymnasium MuJoCo](https://gymnasium.farama.org/main/environments/mujoco/)
- [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)
- [Minari D4RL datasets](https://huggingface.co/datasets/farama-minari/D4RL)
- [dm_control](https://github.com/google-deepmind/dm_control)
- [Real-World RL Suite](https://github.com/google-research/realworldrl_suite)
- [MyoSuite](https://github.com/MyoHub/myosuite)
- [MyoChallenge 2025](https://myosuite.readthedocs.io/en/stable/challenge-doc2025.html)
- [HumanoidBench](https://github.com/carlosferrazza/humanoid-bench)
- [LocoMuJoCo](https://github.com/robfiras/loco-mujoco)
- [robosuite](https://github.com/ARISE-Initiative/robosuite)
- [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)
- [PyFlyt](https://github.com/jjshoots/PyFlyt)
- [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)

### Isaac ecosystems

- [Isaac Lab environment catalogue](https://isaac-sim.github.io/IsaacLab/develop/source/overview/environments.html)
- [Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena)
- [DynamicVLA](https://github.com/hzxie/DynamicVLA)
- [SoftVTBench](https://github.com/TuojingAI/SoftVTBench)
- [IsaacLabEvalTasks](https://github.com/isaac-sim/IsaacLabEvalTasks)
- [LW-BenchHub](https://github.com/placeholder/LW-BenchHub)
- [RoboLab](https://github.com/NVlabs/RoboLab)

### Migration references

- [DOMINO](https://github.com/H-EmbodVis/DOMINO)
- [RoboTwin IsaacLab-Arena branch](https://github.com/RoboTwin-Platform/RoboTwin/tree/IsaacLab-Arena)
- [Bi-DexHands](https://github.com/PKU-MARL/DexterousHands)
- [Dexterous Gym](https://github.com/henrycharlesworth/dexterous-gym)
