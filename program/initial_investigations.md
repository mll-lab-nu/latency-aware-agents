# Latency-Sensitive Robot Benchmark 调研

更新日期：2026-09-01

## 1. 调研目标与筛选标准

我们要找的是任何与 latency 高度相关的机器人任务，包括 manipulation、locomotion、loco-manipulation 和 aerial navigation。

第一梯队候选需要尽量满足：

1. 任务成功明显依赖闭环反应时间，latency 不能仅靠“整体做慢一点”规避。
2. 能在 simulator environment 内加入 observation delay、action queue 或 actuator delay。
3. 有在线 RL，优先 PPO / privileged teacher，能够在加入 latency 后重新训练 teacher。
4. 能由 latency-aware teacher 自动生成 rollout 或 demonstration，不依赖重新采集 teleoperation。
5. 代码、训练入口和任务定义已公开，能够立即验证。
6. 与已经选定的 MIKASA Robo `InterceptGrab` 形成互补，而不是重复“预测移动目标并在正确时刻拦截/抓取”的定位。

因此：

- **MIKASA Robo `InterceptGrab`** 保留为移动目标拦截与抓取的核心任务。
- **HumanoidArena** 不进入第一梯队：主要 demonstration 来自 teleoperation，无法自然获得 latency-aware demonstration。
- **DynamicVLA / DOM 和 DOMINO** 放入第二梯队：任务定位与 `InterceptGrab` 重合。
- DQ-Bench、Catch It、dynamic handover、足球和乒乓球等任务即使训练管线合适，也因核心仍是移动物体轨迹预测与接触时机而放入第二梯队。

## 2. 结论

除 MIKASA `InterceptGrab` 外，目前最值得投入的三个正交方向是：

1. **HAIC PullCart**：持续接触、物体惯性、全身平衡和受力耦合。
2. **Robot Parkour Learning Leap**：视觉预测、起跳时机、落脚和高速 locomotion。
3. **Click-and-Traverse G1CatPri**：全身避碰、跨越、下蹲和侧身穿越。

如果接受飞行机器人，再加入：

4. **FlightBench maze/racing**：高速 ego-vision navigation 和碰撞规避。

这四类任务的 latency failure mode 与 `InterceptGrab` 基本正交。

| 梯队 | Benchmark / task | Latency 失效机制 | Online teacher / demo | 立即可用性 |
| --- | --- | --- | --- | --- |
| 第一 | HAIC PullCart / PushCart | 持续接触力、物体惯性、全身失衡 | Privileged PPO，可重训 | 高 |
| 第一 | Robot Parkour Leap / Climb / Crawl | 起跳与落脚决策过期、视觉状态滞后 | PPO specialist → DAgger depth student | 高 |
| 第一 | Click-and-Traverse G1CatPri | 碰撞、错过落脚点、姿态切换过晚 | Privileged PPO specialists → DAgger generalist | 高 |
| 第一 | FlightBench maze/racing | 航迹误差、高速碰撞 | State MAPPO teacher → visual DAgger student | 中，工程栈较重 |
| 第二 | VIRAL / GR00T-VisualSim2Real | delayed RGB 管线完整，但任务时间压力偏弱 | PPO teacher → RGB DAgger student | 高 |
| 第二 | KinDER | 动态物理推理，但现成 3D PPO 覆盖有限 | PPO / SAC state baseline | 中 |
| 第二 | SkillBench | whole-body 稳定与搬运，但缺少强时间约束 | PPO / SkillBlender | 中 |
| 第二 | Neverwhere | 视觉 parkour evaluation 很强 | teacher training 依赖外部组件 | 中低 |
| 第二 | DynamicVLA / DOM、DOMINO | 与 InterceptGrab 的移动目标定位重合 | 以 scripted / imitation 数据为主 | 高 |
| 第二 | HumanoidArena | whole-body VLA 完整 | 依赖 teleop demonstration | 高，但不适配 latency teacher |

## 3. 第一梯队

### 3.1 HAIC：首选的新 benchmark

HAIC 是 RSS 2026 的 Unitree G1 whole-body object interaction 项目，包含：

- `pull_cart`
- `push_cart`
- `push_wheelchair`
- `box_slope` / `box_stair`
- `skateboard`

官方直接提供 PPO teacher 和 student/adaptation 训练入口：

```bash
python scripts/train.py algo=ppo_haic_train task=G1/haic/skateboard
python scripts/train.py algo=ppo_haic_finetune task=G1/haic/skateboard checkpoint_path=...
```

PullCart 的 privileged observation 包含物体关节位置、速度、力矩、接触几何和动力学参数。其核心难点是机器人、接触物和下肢平衡之间的持续动力学耦合，而不是一次性的抓取时机。

HAIC 论文确实考虑了 delay，但其定位是 **sim-to-real domain randomization**，不是论文的主要研究变量：

- “Sim-to-Real Transfer”部分说明使用大规模 domain randomization 缩小 reality gap。
- 附录 Table XI 在 Actuation 类别中列出 `Action Delay ~ U[40, 120] ms`。
- 论文没有把不同 latency 下的性能或 latency-aware teacher 作为独立实验；这里的 delay 主要用于提高真机泛化和鲁棒性。
- 附录还报告 HAIC 方法本身的平均 inference latency 只有 0.17 ms，低于 20 ms control period 的 1%，强调 world model 没有引入显著计算延迟。

公开代码也已经实现 action-delay buffer：

- policy step：20 ms；
- Isaac physics step：5 ms；
- release 配置为 `min_delay=2, max_delay=6`，并包含 action smoothing；
- buffer 支持跨 control-step 的 delay，可扩大到更高 latency 档位。

需要注意，当前 release 配置的 `2–6` 与论文报告的 `40–120 ms` 不能直接按字面对应；正式实验前需要核对代码版本和 delay 单位。尽管如此，论文和代码都证明该 PPO teacher pipeline 已经把 action delay 放在 environment 内训练。我们的区别是将 latency 从 sim-to-real nuisance randomization 提升为显式研究变量，并据此训练 latency-aware teacher 和生成 demonstrations。

推荐顺序：

1. `pull_cart`
2. `push_cart`
3. `box_stair` / `box_slope`

风险与缺口：

- 使用 motion/reference data，但策略仍是在线 PPO，因此 reference 本身不需要重新采集 latency demonstration。
- 没有现成 RGB VLA 数据层，需要从 latency-aware teacher rollout 中渲染 RGB 并建立数据接口。
- 仓库根目录目前未看到明确 license 文件，正式复用前需要确认。

来源：

- [HAIC 官方仓库](https://github.com/ldt29/HAIC)
- [HAIC 论文 PDF](https://haic-humanoid.github.io/static/haic.pdf)
- [PullCart task 配置](https://github.com/ldt29/HAIC/blob/main/cfg/task/G1/haic/pull_cart.yaml)
- [默认 latency 配置](https://github.com/ldt29/HAIC/blob/main/cfg/task/base/tracking-base.yaml)
- [Action delay 实现](https://github.com/ldt29/HAIC/blob/main/active_adaptation/envs/mdp/action.py)
- [项目主页](https://haic-humanoid.github.io/)

### 3.2 Robot Parkour Learning：最贴合 latency 假设的 locomotion benchmark

Robot Parkour Learning 的任务包括：

- Climb
- Leap over gaps
- Crawl beneath barriers
- Tilt / squeeze through slits
- Run

训练管线为：

1. 使用 privileged terrain 和 physics information，通过 PPO 训练多个 specialist。
2. 使用 DAgger 将 specialist 蒸馏为单一 egocentric depth policy。
3. 在真机上仅使用 onboard perception 和 proprioception。

它不依赖 reference motion 或 teleoperation。论文原生随机化：

- forward depth latency：200–260 ms；
- proprioception latency：37.5–47.5 ms；
- camera pose、FOV、friction 和 motor strength。

这已经直接验证了高感知延迟下的 locomotion。我们的改造只需进一步在 environment 中加入 inference/action FIFO，并在同样的 delayed dynamics 下重新训练 privileged PPO teacher。

推荐顺序：

1. `Leap`：延迟改变何时起跳，无法仅靠放慢动作规避。
2. `Climb`：延迟影响接近、前肢接触和后肢推进之间的相位。
3. `Crawl`：延迟影响高度调整和前进速度的闭环协调。

风险与缺口：

- 项目是 CoRL 2023，不是最新 simulator stack。
- 原生 latency 主要覆盖视觉和 proprioception；action queue 需要自行加入。
- 基于 Isaac Gym，后续维护成本高于 Isaac Lab。

来源：

- [Robot Parkour Learning 项目页](https://robot-parkour.github.io/)
- [训练代码与 checkpoints](https://github.com/ZiwenZhuang/parkour)
- [论文 PDF](https://robot-parkour.github.io/resources/Robot_Parkour_Learning.pdf)

### 3.3 Click-and-Traverse：新的 humanoid traversal 候选

Click-and-Traverse 是 2026 年的 Unitree G1 cluttered traversal 项目，场景同时包含地面、侧面和头顶障碍，要求机器人：

- 跨越地面障碍；
- 蹲下通过低矮空间；
- 侧身穿过狭缝；
- 在不规则室内几何中完成全身避碰。

官方已发布：

- `G1CatPri` privileged PPO task；
- specialist PPO training；
- specialist-to-generalist DAgger；
- generalist checkpoint；
- held-out random scene generator；
- MuJoCo batch evaluation。

仓库的 G1 locomotion environment 已经有：

- `enable_ctrl_delay`；
- `ctrl_delay_range=[0, 2]`；
- motor-target FIFO；
- 20 ms control step。

CAT task 的 Humanoid Potential Field 默认每 5 个 control step 更新一次，相当于 100 ms sample-and-hold perception。当前 CAT 配置尚未完整启用 control-delay 路径，但同仓库已有实现，可以直接接入，而不需要重新设计 latency mechanism。

风险与缺口：

- DAgger generalist 的 observation 是 HumanoidPF，不是 RGB。
- 需要在 rollout 中额外渲染 egocentric RGB，供我们的 VLA 学习。
- task 中部分 control-delay 代码目前默认关闭或未接通，需要做一次小范围整合。

来源：

- [Click-and-Traverse 官方仓库](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse)
- [论文](https://arxiv.org/abs/2601.16035)
- [G1 control-delay 实现](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/main/cat_ppo/envs/g1/env_loco.py)
- [CAT perception update](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/main/cat_ppo/envs/g1/env_cat.py)

### 3.4 FlightBench：飞行机器人的高 latency 候选

FlightBench 是 Flightmare 上的 ego-vision quadrotor navigation benchmark，提供 forest、maze 和 racing/multi-waypoint 场景，以及 RGBD、odometry 和 IMU。

其 learning pipeline 与我们的需求高度一致：

1. 训练 state-based perception-aware MAPPO teacher。
2. 使用 teacher 收集 offline rollout。
3. 预训练视觉 encoder 和 action network。
4. 使用 DAgger 在 student visited states 上继续蒸馏。

底层 action 是四维 collective thrust + body rates。高速飞行时，延迟会直接转化为航迹误差和碰撞，而不是只降低完成速度。可以在 Gym `step` 边界加入 observation/action queue 后重新训练 teacher。

风险与缺口：

- Ubuntu 20.04、ROS Noetic、Unity 2020 和 Flightmare，安装与维护成本较高。
- 项目始于 2024 年，不是最新 benchmark。
- 默认 benchmark 场景以静态障碍为主，应该优先使用 maze/racing，并固定较高目标速度。

来源：

- [FlightBench 官方仓库](https://github.com/thu-uav/FlightBench)
- [完整文档与 teacher/student 训练说明](https://anonymous314159265358.github.io/FlightBench/)

## 4. 第二梯队

### 4.1 VIRAL / GR00T-VisualSim2Real

VIRAL 是最好的 latency-aware teacher/student 工程母版之一：

- Unitree G1，Isaac Sim 5.1 + Isaac Lab；
- privileged-state PPO teacher；
- RGB DAgger student；
- `rgb_image_delayed`；
- 固定或 per-environment random delay；
- reset 时重新采样 delay；
- RGB history buffer；
- teacher/student checkpoints 和 evaluation logs。

但其 walk/stand/place/grasp/turn 和 DoorMan 任务大多没有不可逆的时间窗口。VLA 可能通过降低动作速度规避 latency，因此更适合作为 delayed-RGB 管线参考，而不是 latency 论文的主任务。

来源：

- [GR00T-VisualSim2Real 官方仓库](https://github.com/NVlabs/GR00T-VisualSim2Real)
- [Delayed RGB 配置](https://github.com/NVlabs/GR00T-VisualSim2Real/blob/main/gr00t/rl/config/obs/loco_manip/obs_walk_stand_place_grasp_turn_homie_distill_rgb_delay.yaml)

### 4.2 KinDER

KinDER 是 RSS 2026 physical-reasoning benchmark，包含 25 个 procedural environments，支持 Gymnasium、PyBullet、MuJoCo 和 pygame，并提供 PPO、SAC、MBRL、checkpoint 和 demonstration datasets。

较相关的任务包括：

- `DynScoopPour2D`
- `DynPushPullHook2D`
- `BalanceBeam3D`
- `Dynamo3D`
- `ScoopPour3D`
- `SweepIntoDrawer3D`
- `Tossing3D`

其中 `DynScoopPour2D` 的多物体物理与工具接触比较适合 latency；`BalanceBeam3D` 测试物体放置造成的旋转力矩；`Dynamo3D` 测试移动障碍物后的 navigation。

没有进入第一梯队的原因：

- 现成 PPO 主要覆盖动态 2D、BaseMotion3D、Transport3D 和少量 TidyBot3D task。
- 最有潜力的 dynamic 3D task 并非都有成熟 PPO config/result。
- 没有现成 privileged state teacher → RGB student 管线。
- 部分任务是 sparse reward 和 long horizon，teacher 训练风险高于 MIKASA。

来源：

- [KinDER benchmark](https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden)
- [KinDER baselines](https://github.com/Princeton-Robot-Planning-and-Learning/kinder-baselines)
- [项目主页](https://prpl-group.com/kinder-site/)

### 4.3 SkillBench / SkillBlender

SkillBench 包含 H1、G1 和 H1-2，以及八个 whole-body task：

- FarReach
- ButtonPress
- CabinetClose
- FootballShoot
- BoxPush
- PackageLift
- BoxTransfer
- PackageCarry

低层 primitive 和高层 baseline 均有 PPO，官方提供 checkpoint。最值得考虑的是 `PackageCarry`、`BoxPush` 和 `BoxTransfer`，因为它们涉及 locomotion、接触和负载稳定。

没有进入第一梯队的原因：

- task 缺少明显不可逆时间窗口。
- 现有 action-delay-like 机制主要是当前动作与上一动作随机混合，不是严格 FIFO latency。
- 没有完整 state teacher → visual student pipeline。

来源：

- [SkillBlender / SkillBench 项目页](https://psi-lab.ai/SkillBlender-web/)
- [官方代码](https://github.com/Humanoid-SkillBlender/SkillBlender)

### 4.4 Neverwhere

Neverwhere 是 IROS 2026 hyper-realistic visual parkour benchmark，包含 60+ Gaussian Splatting scenes 和四类任务：hurdle、gap、stairs、ramp。

它提供 Gaussian Splat、collision mesh、waypoint、RGB/depth/point-cloud/segmentation wrapper。evaluation script 已经暴露独立的 visual delay 和 action delay，并使用 action/visual buffer。

没有进入第一梯队的原因：

- 项目主要定位为 closed-loop evaluation。
- 当前 teacher/student learning code 依赖 LucidSim、`cxx` policy modules 和分布式训练组件，仓库本身不是完整、独立的 PPO teacher pipeline。
- 更适合后期为 Robot Parkour policy 增加 photorealistic visual evaluation。

来源：

- [Neverwhere 项目页](https://ziyc.github.io/neverwhere-bench/)
- [官方代码](https://github.com/embodied-intelligence/neverwhere)
- [Delay evaluation 脚本](https://github.com/embodied-intelligence/neverwhere/blob/main/neverwhere_baselines/debug_unroll.py)

### 4.5 与 InterceptGrab 定位重合的动态任务

以下项目本身可能很强，但统一放入第二梯队，因为主要研究信号仍是移动物体轨迹预测和精确接触时机：

- [DynamicVLA / DOM](https://github.com/hzxie/DynamicVLA)
- [DOMINO](https://github.com/H-EmbodVis/DOMINO)
- [DQ-Bench / DQ-Net](https://github.com/YoungYNG/DQ_WBC)
- [Catch It!](https://github.com/hang0610/Catch_It)
- [Dynamic Handover](https://github.com/cypypccpy/dynamic_handover)
- humanoid soccer striker
- HOPE / PACE humanoid table tennis
- Bi-DexHands catching tasks

这些项目可以作为第二组 related-work 或补充实验，但不应该抢占 HAIC / Parkour / CAT 的首轮工程资源。

### 4.6 HumanoidArena

HumanoidArena 有完整 whole-body VLA、LeRobot dataset、checkpoint 和 evaluation pipeline，但 demonstration 主要来自 teleoperation。将 latency 加入 environment 并不能把原有 teleop trajectories 自动变成 latency-aware demonstrations。

除非后续为其任务重新训练 simulator RL teacher，否则它保持第二梯队，更适合作为 VLA infrastructure 参考。

来源：

- [HumanoidArena 官方仓库](https://github.com/RL2-humanoid-VLA/HumanoidArena)
- [项目主页](https://humanoidarena.github.io/)

## 5. 值得跟踪但暂时不能直接用

### Light-Loco-Parkour

Light-Loco-Parkour 几乎满足理想条件：privileged teachers、multi-expert DAgger、recurrent depth student、climb/step down/vault，以及 27–33 Hz camera、camera noise、dropout 和 30–60 ms latency。

但截至 2026-09-01，官方只发布项目页和论文，没有完整训练代码，因此不能列入可立即开工的第一梯队。

来源：[Light-Loco-Parkour 项目页](https://light-loco-parkour.github.io/)

### RPL / Perceptive Humanoid Parkour

两者都有 privileged expert → depth student 的强 pipeline，也显式考虑 sensor latency 或硬件时序，但官方训练代码仍为 coming soon，暂时只跟踪：

- [RPL](https://rpl-humanoid.github.io/)
- [Perceptive Humanoid Parkour](https://php-parkour.github.io/)

### FALCON

FALCON 的 payload transport、cart pulling 和 door opening 与 latency 下的受力交互高度相关，并使用双 agent PPO。但仓库仍缺少可靠的完整训练、sim2sim、sim2real release 和 checkpoint，因此暂不作为首轮候选。

来源：

- [FALCON 官方仓库](https://github.com/LeCAR-Lab/FALCON)
- [项目主页](https://lecar-lab.github.io/falcon-humanoid/)

## 6. 推荐实验组合

首轮只验证三个任务，不继续堆叠相似 benchmark：

1. **MIKASA InterceptGrab**
   - 代表 moving-object interception。
   - 已经验证 teacher 训练效果好。

2. **HAIC PullCart**
   - 代表 placeholderained contact、object inertia 和 humanoid balance。
   - 直接复用已有 action-delay buffer，并在高 latency 下重训 PPO teacher。

3. **Robot Parkour Leap**
   - 代表 visual anticipation、commit timing 和 locomotion。
   - 复用已有 perception-latency randomization，再加入 action queue。

若第三个 simulator stack 不易复现，则用 **Click-and-Traverse G1CatPri** 替代 Robot Parkour；如果需要额外的高速非接触机器人场景，再加入 **FlightBench maze/racing**。

这个组合覆盖三种不同的 latency failure mode：

| Task | 延迟造成的主要失败 |
| --- | --- |
| InterceptGrab | 对移动目标估计过期，错过拦截与抓取窗口 |
| HAIC PullCart | 持续接触控制滞后，产生力振荡、轨迹误差或失衡 |
| Robot Parkour Leap | 起跳决策过期，错过落脚点或无法跨越 gap |
| Click-and-Traverse | 全身避碰和姿态切换过晚，发生碰撞 |
| FlightBench | 高速航迹修正滞后，碰撞或偏离航路 |

相较继续增加 catch、soccer 或 ping-pong task，这组任务更能证明 latency-aware teacher / demonstration 的方法具有跨 manipulation 与 locomotion 的普适性。
