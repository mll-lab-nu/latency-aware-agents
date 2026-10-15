# bodies/ — 机器人资源库

RoboOlympics 的框架仍然是四部分：**players / sports / core / tools**。`bodies/` 不是第五部分，而是各运动共用的
**机器人资源库**（2026-10-08 架构决定 "BPKG"）：机器人模型、装在手腕上的附件、唯一可信的限位 / 伺服路径、可选的
控制库，以及准入用的一致性测试。core 不导入它，也不因它改变；运动从这里取机器人，生成器把运动用到的那部分复制进
任务的模拟器目录。

## 1. 里面有什么

| 路径 | 内容 |
|---|---|
| `__init__.py` | 登记表：`BODIES`（有模型的机器人）、`ARMS`（只有限位的机械臂条目）、`CONTROLLERS`；`load_body(name)`、`assemble(...)`、`planner(name)` |
| `limits.py` | 机械臂限位的原语（S 曲线、`JerkLimiter`、`TorqueRateLimiter`、`check_rows`、测量统计）。击剑在用，**行为未改** |
| `servo.py` | **唯一可信的限位 / 伺服路径**：`SetpointGenerator`（关节设定点，限速度、加速度、加加速度）、`PlanarBase`（平面底盘，按机器人自身坐标系逐轴或按合矢量限）、`Actuators`（输出级：力矩变化率、电机扭矩-转速曲线、力矩上限，唯一写 `d.ctrl` 的地方）、`WalkCommand`（行走指令：真机摇杆范围 + 变化率）、`Servo`（把以上组合起来）、`check_row`（10 ms 目标行的检查） |
| `sources.py` | 声明规则（只用标准库）：每个限位必须有来源；servo、一致性测试和 CI 共用 |
| `selfcollision.py` | **自碰撞**（§4.1）：按 `body.toml [self_collision]` 生成 MuJoCo 接触对、比赛中记录自碰撞（`Monitor`）、检查声明和模型是否相符 |
| `contacts.py` | **机器人之间的接触**（§4.2）：`between(model, ["A_", "B_"])` 让几台机器人互相碰、碰到对方的器具（剑、拍），不碰自己 |
| `assembly.py` | 装配配置：机器人 + 附件 + 控制库 → 带版本哈希的配置 |
| `conformance.py` | 准入一致性测试（每个机器人同一套检查，见 §6） |
| `lod.py` | 选手画面用的网格级别（`meshes_25k/`）的生成工具（开发者步骤，比赛从不运行） |
| `control/` | 可选控制库：`wholebody.py`（全身规划器）、`arbiter.py`（腿式机器人的关节组仲裁 + 关节目标计划接口）、`kinematics.py`（低层玩家用的 FK / IK 库，独立文件；模型带自碰撞对时 IK 默认避开自碰撞，§4.1） |
| `hands/` | 附件：`hands.toml` + 每个附件一个 MJCF 片段（`paddle` 已用；`racket`、`board`、`suction`、`epee`、`gripper` 是草稿，只有几何；`epee` 另有带出处的质量分布：0.455 kg，标准件重剑） |
| `wheeled_humanoid/` | Rainbow Robotics **RB-Y1 M**（轮式人形；MuJoCo Menagerie，Apache-2.0） |
| `unitree_g1/` | **Unitree G1** 29 自由度（腿式；Menagerie，BSD-3-Clause），含 Unitree 行走策略、可选的 FALCON / AGILE 行走策略（`falcon.py`，MIT；`agile.py`，Apache-2.0；见 §5 第 6 条）和踢球 `kick.py` |
| `unitree_h1/` | **Unitree H1-2**（腿式；unitree_rl_gym，BSD-3-Clause），含 Unitree H1-2 行走策略 |
| `galaxea_r1pro/` | **Galaxea R1 Pro**（轮式人形；GalaxeaManipSim，Apache-2.0） |
| `tianji_marvin_pro/` | **天机 Marvin Pro**（固定底座双臂）——**仅供测试**，见 §8 |
| `abb_irb1100/` | **ABB IRB 1100-4/0.58 + 两轴龙门**（DeepMind 乒乓机器人的配置）；没有许可干净的上游模型，**按 ABB 公开尺寸自建的近似模型**（运动学精确、简单几何、质量估计），见 `abb_irb1100/README.md` |
| `franka_panda/`、`franka_fr3/`、`kuka_iiwa14/` | 只有限位的机械臂条目（`body.toml`）：各运动场景自带模型（击剑用 Panda） |
| `booster_t1/` | **Booster T1** 只有限位的腿式条目（`body.toml`，登记在 `LEGS`）：足球场景自带模型（sai-mujoco 下半身）和行走策略；运动在 `sport.toml` 写 `[legs] body = "booster_t1"`，腿经 `servo.Actuators`（`[motors]` 电机曲线 + 力矩上限）和 `servo.WalkCommand` 执行 |

每个有模型的机器人目录：`body.toml`（自由度、限位及来源、伺服增益、静止姿态、挂载点）、`body.py`（`mount` 装进场景、
`rig` 驱动、`robot_only_xml` 导出单独机器人）、`model.xml`（生成的，不要手改）、`meshes/`（仿真用网格，在 git 里）、
`LICENSE`、`tools/make_model.py`（来源与改动记录，也负责生成 `meshes_fine/`）。网格有三个级别：`meshes/`（仿真，
在 git 里）、**`meshes_25k/`（选手画面用的约 2.5 万面级别，在 git 里**，2026-10-08 决定 D1；`python -m bodies.lod build
<robot>` 从精细网格抽稀生成，`lod.json` 记录工具版本、面数和每个文件的 sha256，可复现；`mount(..., lod="25k")`）、
`meshes_fine/`（上游原始网格，**不进 git**，`.gitignore`；没有它时 `lod="fine"` 自动退回 `meshes/`）。

## 2. 三个概念（加一个"装配配置"）

1. **机器人本体**（在库里）：模型、关节、执行器、**真实限位**。真实限位由厂商控制器执行的值决定（SDK 优先于
   URDF），绝不"无限"；没公布的值写成标注过的估计。
2. **控制能力**（可选的共享代码，运动自己决定是否 import）：行走策略（随机器人提供，属于本体的一部分，因为真机
   出厂就带）、IK、全身规划器、控制权仲裁。放在 `control/`（以及机器人自带的技能，如 G1 的 `kick.py`）。没有任何
   强制的生命周期接口。
3. **模式**（由运动声明，不由机器人声明）：高层 / 低层、program / direct 等在运动的工具包清单（toolkit manifest）和
   `sport.toml` 里声明；机器人只提供能力，不规定怎么玩。

**装配配置** = 机器人 + 附件（手或工具）+ 运动声明用到的控制库。它有自己的版本哈希（`assembly.py`）：机器人的
`body.toml`、`model.xml`、代码、仿真网格、策略权重，附件的声明和模型，控制库，以及限位 / 伺服路径（`limits.py`、
`servo.py`）任何一个变了，哈希就变。任务的 `task.toml` 记 `robot_configuration = "<id>"`，模拟器目录里有
`bodies/configuration.json`；运动可以用 `bodies.assembly.recorded(sim_dir)` 把它放进自己的 `/spec` 和成绩。

```python
from bodies import assemble
cfg = assemble("unitree_g1", "paddle", controllers=["wholebody"])
cfg.id      # "unitree_g1@g1-2+paddle+wholebody#<12 位哈希>"
```

## 3. 真实限位（2026-10-08 审计，已批准的数值）

来源写在各自 `body.toml` 的 `[limits.<组>.source]` 里（`S<k>` = `[sources]` 的条目；`ESTIMATE:` = 估计并说明方法；
`DERIVED:` = 由公开数据推算；`NOT PUBLISHED:` = 没人公布、不限，并说明原因）。

| 机器人（版本） | 关节 | 底盘 / 行走 | 输出级 |
|---|---|---|---|
| RB-Y1（wh-3） | 手臂速度 URDF；**加速度 15 rad/s²**（URDF 10 × 控制器最大倍率 1.5）；躯干、升降：URDF 5 × 1.5；加加速度 = 50 × 加速度（估） | 合速度 1.5 m/s（产品页）、2.0 m/s²（推算）；**偏航加速度 1.0 rad/s²**（SDK 默认）；加加速度 25 ×（估） | URDF 力矩 |
| Galaxea R1 Pro（r1p-2） | **手臂 SDK {3,3,3,3,5,5,5} rad/s，加速度 / 加加速度 = 1.5 × 速度**；**躯干 1.5 rad/s**，加速度 / 加加速度 2.25 | **逐轴**（机器人坐标系）：1.5 / 1.5 m/s，**x 2.5、y 1.0 m/s²**，偏航 3 rad/s、1.0 rad/s²（文档矛盾，取小值） | 数据表 / URDF 力矩 |
| Unitree G1（g1-2） | URDF 速度；加速度、加加速度：NOT PUBLISHED（Unitree SDK 不限，电机曲线约束） | **行走指令截到 Unitree 摇杆上限 [0.8, 0.5, 1.57]**，变化率 [2, 2, 4] /s（估） | **电机扭矩-转速曲线**（unitree_rl_lab）+ URDF 力矩，所有 29 个关节 |
| Unitree H1-2（h1-2） | 同 G1 | **[0.8, 0.5, 1.57]** | **电机扭矩-转速曲线（估）**：Unitree 没有公开 H1-2 的电机分配，按 G1 公开曲线的形状估计（X2 = URDF 速度 / 0.9，X1 = 0.6 X2，Y = URDF 力矩），所有 27 个关节 |
| Marvin Pro（mp-2） | 180°/s；**加速度 450 / 900 °/s²（SDK，已执行）**；加加速度 50 ×（估）；力矩 URDF | 无底座（任何底座运动被拒绝） | URDF 力矩 |
| Franka Panda / FR3 | 厂商全套（速度、加速度、加加速度、力矩、力矩变化率） | — | 击剑的伺服 |
| ABB IRB 1100 + 龙门（abb-1） | 速度 ABB（460/360/280/560/420/750 °/s）；**加速度：轴 1–3 由 ABB 的 0 类停止数据推算（30.9 / 20.3 / 22.2 rad/s²），轴 4–6 估计**；加加速度 50 ×（估）；力矩 DeepMind 仿真的值 | **龙门逐轴**：2 m/s（DeepMind），8 m/s²、200 m/s³（估，同乒乓 Franka 导轨），驱动力 4102 / 1499 N（DeepMind 仿真） | 力矩 / 驱动力上限 |
| Booster T1（t1-1，只有限位） | 腿：URDF 速度和力矩（45 / 30 / 30 / 60 / 24 / 15 N·m）；加速度、加加速度：NOT PUBLISHED | 行走指令 [1, 1, 1]（策略训练范围），变化率 [2, 2, 4] /s（估） | **电机扭矩-转速曲线（估）**：URDF 速度的 80% 以前满力矩，到 URDF 速度降到 0；URDF 力矩 |
| KUKA iiwa 14（iiwa14-limits-2） | 速度 KUKA；**加速度 = 10 × 速度（Air Hockey Challenge）**；加加速度 50 ×（估）；力矩变化率 NOT PUBLISHED | — | — |

## 4. 限位与伺服：唯一路径，不能绕过

每个机器人的 `body.py` 在 `servo.py` 里构建自己的伺服（`wheeled_layout` / `legged_layout`）；运动通过机器人的 rig
（`rig.begin_step`、`rig.apply`）驱动，或者（只有限位的机械臂）用 `limits.JerkLimiter` + `limits.TorqueRateLimiter`
/ `servo.Servo`。三层：

1. **指令层** `rig.check(u, u_prev, u_prev2)`：范围、每步速度；给了 `u_prev2` 还查加速度（底盘按其模式逐轴或合矢量）。
   行走指令截到真机摇杆范围并限变化率。
2. **设定点层**：每 1 ms 物理步，10 ms 目标插值后的参考进 `SetpointGenerator` / `PlanarBase`，设定点的速度、加速度、
   加加速度永远不超过声明值（离散步的超调已留余量）。它在 `limits.JerkLimiter` 的级联上加了"按加速度和加加速度一起
   刹车"的位置环：R1 Pro 这种加加速度很小（1 s 才到满加速度）的 SDK 值，原来的平方根律会过冲振荡。
3. **输出级** `Actuators.write`：力矩变化率（有公布值时）→ 电机扭矩-转速曲线（有公布值时）→ 力矩上限 → `d.ctrl`。
   这是库里唯一写执行器的地方。

不改变的：碰撞、外力造成的真实运动（审计 D9：记录，不阻止）；MJCF 的关节范围。

### 4.1 自碰撞（B0 的一条，2026-10-09）

真机的手臂不能穿过自己的身体，仿真里也不能。起因：宣传片里 G1 打乒乓时手臂穿进躯干（最深约 12 cm），因为资源库的
机器人几何体都是 contype 4 / conaffinity 0（彼此不碰，包括自己）。现在每个机器人的 `body.toml [self_collision]` 声明
哪些部分互相碰，`selfcollision.py` 在 `mount()` 里把它变成显式的 MuJoCo `<pair>`（名字 `<前缀>selfcol_<i>`），
不改 contype / conaffinity，所以两台机器人之间、机器人和场地之间的接触都不变。

- **部分**：按 body 名（去掉前缀）的正则分成 `left_arm` / `right_arm`（单臂：`arm`）和 `body`（躯干、腰、头、骨盆、腿、
  底盘和轮子、基座）；没匹配上的 body 跟父 body 走，挂在手臂下面的就是这只手臂的**工具**（拍、剑、球拍、板）；
  固定底座的机械臂站在上面的东西（基座、龙门滑台、导轨小车）算 `body`。
- **规则**：手臂 × 身体（含腿）、左臂 × 右臂（含工具）、工具 × 自己手臂的连杆；Franka 这种单臂还开 `same_arm`（隔两个及以上
  关节的连杆互碰，和 MuJoCo 自己的父子过滤一致）；腿式机器人的腿（`left_leg` / `right_leg`）开 `leg_leg`（左腿 × 右腿）和
  `same_leg`（同一条腿隔一个关节以上，如小腿 × 脚；2026-10-10 Changhe 审计：行走策略横移时两腿互穿）。`ignore_geoms`
  排除感应体（G1、H1-2 踢球用的 strike 盒子）。
- **排除**：自动的两类不列出——同一个刚体组（连杆、它的安装座、它的工具）和只隔一个关节的两个刚体组（关节壳体本来就重叠；
  MuJoCo 自己的父子过滤也这么做）；其余写进 `[[self_collision.exclude]]`，每条两个 body 正则 + 理由。目前的清单：

  | 机器人 | 启用的对（配拍子） | 声明的排除 | 理由 |
  |---|---|---|---|
  | G1（g1-4） | 544（腿 × 腿 125） | 无 | 网格在站姿和随机姿态下都不结构性重叠；躯干 × 肩 pitch 是父子（自动） |
  | H1-2（h1-4） | 349（腿 × 腿 56） | 躯干 × 肩 roll | 肩的第二个连杆，零位（行走策略训练时的手臂姿态）下网格正好贴着躯干（0.0 mm），否则站着就永久记一次自碰撞；上臂（肩 yaw）起照常碰 |
  | RB-Y1（wh-4） | 223 | 胸（torso_5）× 肩 arm_1 | 肩部壳体：胸部拟合胶囊（半径 0.149 m）盖住了肩窝，29% 的合法姿态里重叠到 3 cm（渲染网格在安装处也重叠 1.8 cm，模型问题）；上臂 arm_2 起照常碰 |
  | R1 Pro（r1p-3） | 189 | 胸（torso_link4）× 肩 link2；肩支架 × 同侧 link2 | 肩部壳体：link2 在支架里转，拟合的盒子 / 胶囊在 52–66% 的合法姿态里重叠到 1.9 cm；上臂 link3 起照常碰 |
  | Marvin Pro（mp-3，仅测试） | 121 | 支架（base）× 肩 arm_1 | 支架胸部的盒子盖住两肩，站姿就重叠 0.6 / 1.3 cm，80% 的合法姿态重叠 |
  | Franka Panda（panda-limits-2） | 每个运动的场景 | 无 | 只有自动的（父子、法兰 × 工具、link0 × 基座） |
  | ABB IRB 1100 + 龙门（abb-1） | 74 | 无（`same_arm` 关） | 手臂连杆 1–6 和拍子对龙门（框架、横梁、滑台）和手臂底座、拍子对手臂；连杆对连杆不查：ABB 的轴范围就是机械设计，近似几何体在肘、腕处比真机粗，会挡住真机能到的姿态 |

  RB-Y1 的另一只手（`[rest] other_arm`）肩 roll 0.30 → 0.40：原来上臂的碰撞胶囊在站姿就嵌进胸部 0.4 cm，现在离 1.2 cm。
- **后果**：物理上挡住（接触参数：MuJoCo 默认的 condim 3、摩擦 1，solref 0.005 1——外壳是硬的，默认 0.02 会让推到力矩上限的
  伺服把手臂压进身体 2 cm）。伺服不知道（真机控制器也不知道）：手臂顶在自己身上，PD 力矩升到上限。
- **记录**：每个 rig 的 `apply` 每个物理步调一次 `Monitor.update`；`rig.self_collisions()` 给出事件数（一对几何体开始接触，
  20 ms 内的反复接触算一次）、接触时间、峰值法向力和 **10 ms 平均力的峰值**、冲量、最深穿透、最严重的几对。10 ms 平均力
  超过 `warn_force_n`（20 N：Franka 碰撞反射的默认阈值，libfranka；其他机器人没有公开值，用同一个数，标为估计）算一次
  **警告**。**只记录，不判罚**：怎么判由用户决定（建议见 PR）。运动把这份记录放进比赛记录。
- **检查**：`python -m bodies.selfcollision check`（每个机器人、每种手：站姿下没有启用的对相交；200 个随机合法姿态里没有
  哪对在 95% 以上都相交——那是安装处的结构性重叠，必须写成排除；没有用不上的排除）；一致性测试第 9 项另外验证"挡得住"：
  手臂按速度上限冲向身体里面 8–11 cm 的目标，开自碰撞后穿透 < 1.5 cm（实测 0.1–0.6 cm），关掉时 ≥ 3 cm；
  `ci_checks/check-robot-limits.py` L4（声明，标准库）、L5（几何，CI 装 mujoco）、S4（运动自己的场景里的生成块和声明一致）。
- **运动自己场景里的机械臂**（Franka）：`sport.toml [self_collision]` 写 `body` 和每个场景的名字映射（前缀、后缀、属于基座的
  世界几何体），`python -m bodies.selfcollision scene sports/<运动>` 把接触对写进场景文件的生成块；CI 的 S4 核对。
- **避开自碰撞的 IK**：`bodies/control/kinematics.py` 的 `solve()` 在模型带自碰撞对时默认让被求解关节带动的每一对至少相距
  `self_min_dist`（2 cm）：5 cm 内的对给一行距离约束（最近点、分离法向、`mj_jac` 点雅可比），步长按最小改动投影到约束上，
  再在真实距离上做线搜索（不让任何一对近于 5 mm，长步长不会穿过细连杆）；够不到的目标留下位置误差，而不是让手臂穿过胸口。
  `avoid_self_collision=False` 是原来的求解器（逐位一致）。测试 `python -m bodies.control.test_selfcollision_ik`（G1、H1-2）。
- 关掉：`mount(..., self_collision=False)`、`robot_only_xml(..., self_collision=False)`（对比实验用；正式比赛不允许）。


CI（`ci_checks/check-robot-limits.py`，只用标准库）检查：每个机器人的限位都有来源；每个机器人的伺服建在 `servo.py`
里、库里没有别处写 `.ctrl`；每个引用库机器人的运动（`[arm] body`、`[legs] body` 或 `[body] name`；`[legs]` 要用
`servo.Actuators` 和 `servo.WalkCommand`）都导入并使用这条路径，不自定义
同名限位器、不给 `limits` / `servo` 模块或 rig 的伺服打补丁、不用 `scaled(k>1)` 放宽限位。场景里有库中机器人却没有
引用库的运动只给警告（目前：乒乓、网球、羽毛球、篮球、空气曲棍球——审计 G1，留给各运动的 PR）。

### 4.2 机器人之间的接触（2026-10-11）

库里的机器人默认彼此不碰（§4.1：contype 4 / conaffinity 0），两台机器人会互相穿过，剑尖也会穿过对方身体。需要机器人
相碰的运动（击剑、篮球、足球）在编译后调用 `contacts.between(model, 前缀列表)`：第 k 台机器人的碰撞几何体加上它的
"身体位"，手上的 contact 几何体再加"器具位"（从第 16 位起，每台两位，不占运动和库自己用的低位）；每台机器人碰别人的
身体和器具，不通过这些位碰自己（自碰撞仍走 §4.1 的接触对）；原来能碰的都还能碰（只加位不减位）；画图用的几何体和
感应盒（G1 踢球的 `*_strike`）不动。

**坑**：MuJoCo 的宽相先按 `body_contype` / `body_conaffinity`（编译时由刚体上几何体的位按位或得到）过滤刚体对，
编译后只改几何体的位不够，机器人照样穿过；`between()` 会一起重算（击剑、篮球筛选时都踩到过）。检查：conformance
第 2 项"robots touch each other"（两台机器人重叠放置：库默认 0 个接触，调用后 > 0，自身接触不变），
`python bodies/tests/test_contacts.py`。

## 5. 运动怎么用一个机器人 + 附件

1. `sports/<sport>/sport.toml`：

   ```toml
   [body]
   name = "wheeled_humanoid"
   hand = "board"                 # bodies/hands/hands.toml
   arm = "right"
   controllers = ["wholebody"]    # 可选：用到的 bodies/control/ 库
   ```

   `tools/generate.py` 把机器人（不含 `meshes_fine/`、`tools/`）、`hands/`、`limits.py`、`servo.py`、`sources.py`、
   `assembly.py`、声明的控制库复制进任务的 `environment/sim/bodies/`，写 `configuration.json`，`task.toml` 记
   `robot_configuration`。`tools/check` 第 1 阶段检查机器人、附件、控制库都存在，且 `status = "test-use"` 的机器人
   不能用在 `publish = true` 的运动里。

2. 建场景：从自己的世界 MJCF 开始，每个选手挂一个机器人：

   ```python
   import sys; from pathlib import Path
   ROOT = Path(__file__).resolve().parents[1]             # 任务里是 sim/，仓库里是 sports/
   sys.path.insert(0, str(ROOT if (ROOT / "bodies").exists() else ROOT.parent))
   from bodies import load_body
   import xml.etree.ElementTree as ET, math, mujoco
   B = load_body("wheeled_humanoid")
   world = ET.parse("scene/my_world.xml").getroot()
   for seat, (pos, yaw) in {"A": ((0, -3, 0), math.pi / 2), "B": ((0, 3, 0), -math.pi / 2)}.items():
       info = B.mount(world, f"{seat}_", pos, yaw, hand="board", accent=(0.1, 0.3, 0.9, 1))
   model = mujoco.MjModel.from_xml_string(ET.tostring(world, encoding="unicode"))
   ```

   机器人几何体 contype 4 / conaffinity 0（彼此不碰）；想让机器人碰到的世界几何体给 conaffinity 第 4 位；附件的
   contact 几何体 contype 1 / conaffinity 1（运动自己加 `<pair>`）。腿式机器人的场景必须声明
   `<compiler angle="radian">`，地面给 conaffinity 4。

3. 驱动：`rig = B.rig(model, "A_", hand="board")`；`rig.place(data, u)` 摆到初始姿态（伺服设定点同步）；每个 10 ms
   控制步算下一个目标 `u`（`rig.check(u, u_prev, u_prev2)` 说明哪里超限），先 `rig.begin_step(data, u)`，再对 10 个
   物理步各调一次 `rig.apply(data, u_prev, u, j)` 然后 `mujoco.mj_step`。不要自己写这个机器人的 `data.ctrl`。
   受控向量 `u`：轮式 = `[base_x, base_y, base_yaw, lift, waist, arm_0..6]`；腿式 = `[walk_vx, walk_vy, walk_wz,
   waist_pitch, waist_yaw, arm_0..6]`（腿归行走策略）。

4. 高层：`bodies/control/wholebody.py` 的 `WholeBodyPlanner(B, hand).plan(...)`：给"n 步后工具在某位姿、某速度"，
   出 100 Hz 的 `u` 行。它按速度上限规划，加速度和加加速度由伺服在执行中保证；预演（rehearse）用的是同一个伺服。
   腿式机器人在它的平面副本上规划。

5. 低层：`B.robot_only_xml(hand=...)` 是单独的机器人模型（低层座位用）；腿式机器人的关节组仲裁和"关节目标计划"
   接口在 `bodies/control/arbiter.py`（G1、H1-2 都能用；踢球技能只有 G1），IK 库是 `bodies/control/kinematics.py`。
   仲裁器的参数（2026-10-08 决定 D3 / D4，`body.toml [lowlevel]`）：每 20 ms 最多一个计划，每行保持 10 ms（零阶保持），
   最多 300 行，腿的增益上限为估计值；踢球技能结束后不经过渐变直接交还行走策略，计划结束后经过 0.3 s 渐变交还；
   2 s 内没有确认稳定就放行玩家的行走指令（记录，不拒绝）。组可以任意组合，例如 G1 乒乓：腿交给行走策略
   （`walk`），玩家规划腰和手臂（`plan(d, ["waist", "arms"], Q)`）。所有腿式机器人共用这套代码、摇杆上限和电机曲线；
   它是运动的动作层可选用的共享代码，不是机器人必须实现的生命周期。

6. G1 的行走策略可以换：默认是 Unitree 的策略（只管腿）；可选 **FALCON**（LeCAR-Lab，MIT，arXiv:2505.06776：在
   IsaacGym 里用整台 29 自由度 G1 训练，训练时手臂在做动作、手上受外力；`falcon.py`，权重 `policy/falcon_g1_29dof.npz`，
   来源、增益和参数在 `body.toml [walk.falcon]`）或 **AGILE**（NVIDIA WBC-AGILE，Apache-2.0，arXiv:2603.20147：在 Isaac
   Lab 里训练，只管腿，训练时站着的机器人手臂和腰 yaw 随机动；`agile.py`，权重 `policy/agile_g1_velocity_height.npz`，
   来源、增益和参数在 `body.toml [walk.agile]`）。运动自己选：`rig = B.rig(model, "A_", hand="paddle", walker="falcon")`
   （或 `walker="agile"`）。`u`、`begin_step` / `apply` / `check` / `kick` 不变，所有力矩照样过伺服的输出级（电机曲线、
   力矩上限）。FALCON 不同的地方：
   - 腿用 FALCON 的 PD 增益；腰和手臂仍归运动（`u`），FALCON 把手臂目标和腰的 yaw 当作指令（训练里腰的 roll / pitch
     指令恒为 0，这里也给 0）；两只手腕的 PD 目标叠加 FALCON 的平衡残差（最多 0.05 rad，每 20 ms 更新，不经过设定点
     生成器；没有它，站着的机器人几秒内就滑走）。
   - 行走指令先截到摇杆上限，再截到 FALCON 的训练范围 ±1（偏航最多 1 rad/s）；速度低于 0.2 m/s 算 0，再没有转向（< 0.2 rad/s）就站着不踏步。
   - 手臂在整个关节范围内按速度上限乱挥时，FALCON 比 Unitree 的策略更容易倒（±0.6 rad 内挥动两者都不倒）。
   - 仲裁器（`control/arbiter.py`）也支持：`Arbiter(B.rig(..., walker="falcon"))`，腿归 FALCON（`walk`），腰和手臂照常
     `plan` / hold，FALCON 看当步的手臂目标和腰的 yaw；手腕残差照样叠加（计划中的手腕也加）；交还腿时渐变到 FALCON 的
     站姿并重置它的观测历史。
   - AGILE：腿用它的 PD 增益（与 FALCON 相同）；腰和手臂仍归运动（`u`），AGILE 只读 29 个关节的实测位置和速度，不看
     手臂目标，也没有残差（不需要：站 60 s 漂 4.5 cm，几乎都在前 5 s）；指令先截到摇杆上限，再截到训练范围（vx −0.5…1.5、
     vy ±0.5、偏航 ±1 rad/s），范数低于 0.1 算 0，高度指令 0.72 m（训练的行走高度）；重置后第一帧填满观测历史（同训练）。
     训练里走路时手臂和腰都在零位，所以带着别的手臂姿势（包括库的 rest 姿势）走路已在训练之外（实测照样走得稳）。
     仲裁器里走和 FALCON 同一条代码路径。
   - 一致性测试 3b 对每个可选策略各查一遍：输出与 ONNX 一致、站稳、按指令行走、确定性；在仲裁器下再查站稳、按指令行走、
     边走边快速挥两臂、腿计划后交还、确定性。
   - 两者比较（仲裁器下，G1 不带手；括号里是左手拿 0.5 kg 的棍、手臂在持杆姿势）：站 60 s 漂移 FALCON 2.2 cm（27 cm，
     偏航 1.7 rad）、AGILE 4.5 cm（3.8 cm）；vx 0.8 → 0.77（0.79，同时侧移 0.08 m/s、偏航 0.22 rad/s）对 0.79（0.81）；
     vy ±0.3 → 0.23 / −0.25（0.15 / −0.34）对 ±0.31（0.31 / −0.27）；偏航 ±1 → 1.00 / −0.94（0.60 / −1.01）对 ±0.97
     （0.95 / −0.96）；快速挥两臂（两肩 ±1.4 rad，0.25 s）后保持，之后 3 s 漂移最多 0.02（0.42）对 0.01（0.006）m/s；
     手臂在整个关节范围内按速度上限乱挥（站 20 次 + 走 20 次）摔倒 8/40（15/40）对 0/40（2/40）。AGILE 的弱点：原地
     转向低于约 0.7 rad/s 基本转不动（0.35 rad/s 时第一秒后只剩 0.03–0.05 rad/s；边走边转时 0.2 m/s 以上正常），站立
     时骨盆 0.67–0.70 m，比 FALCON（0.74 m）低约 5 cm（Unitree 的策略 0.78 m）。默认仍是 Unitree 的策略，main 上的
     行为不变。
   改用 FALCON / AGILE 的运动在自己的 PR 里重录金标准。

## 6. 怎么加一个机器人（准入）

1. **来源与许可**：模型和网格必须能再分发（许可证文件放在机器人目录里，`NOTICE` 加一条）；`tools/make_model.py`
   记录从哪个仓库哪个提交来、改了什么；精细网格不进 git，由它生成。许可不清楚的只能标 `status = "test-use"`，
   README 顶部写明。
2. **限位与来源**：`body.toml` 的每个限位组（关节组、底盘、行走）都要有速度、加速度、加加速度、力矩（底盘：速度、
   加速度、加加速度、偏航对应三项；行走：指令范围、变化率），每项在 `[limits.<组>.source]` 里写来源；没公布的写
   `ESTIMATE:`（并说明方法）或 `NOT PUBLISHED:`（并说明为什么不限）。SDK / 控制器执行的值优先于 URDF。
3. **伺服**：`body.py` 用 `servo.wheeled_layout` / `servo.legged_layout`（或直接 `servo.Servo` + `servo.Actuators`）
   构建伺服，`rig.apply` 只通过它写执行器；API 与现有机器人一致（`mount`、`rig`、`robot_only_xml`、`place`、
   `sync`、`begin_step`、`apply`、`check`、`tool`）。在 `__init__.py` 的 `BODIES` 登记。
4. **一致性测试**：`python -m bodies.conformance <name>` 全部 PASS（在运动镜像里跑，需要 mujoco）：
   1 限位有来源、许可、来源工具；2 两台机器人同场编译，导出模型与挂载后运动学一致；3 伺服到达目标（腿式：在真机
   指令上限内行走不倒）；4 激进参考下设定点和力矩都在限位内、`check()` 拒绝超限行；5 选手画面级别 `meshes_25k/` 已提交且与 `lod.json` 一致，
   它和精细网格都与仿真运动学一致（精细网格缺失则只比较 25k）；6 确定性重放；7 装配哈希稳定；8 可选控制库能在它上面运行（报告）。
5. **自碰撞**：`body.toml [self_collision]`（§4.1）：部分、规则、`warn_force_n` 及来源、排除清单（每条写理由）；
   `python -m bodies.selfcollision check <name>` 通过。
6. `ci_checks/run_all.sh` 通过（包括 `check-robot-limits.py`）。

2026-10-08 的结果：五个机器人和三个机械臂条目全部 PASS。记录下来的差异（WARN，不影响运动学）：几个机器人的
精细网格和抽稀网格（仿真级别、25k 级别）形状不同（例如 RB-Y1 的 `LINK_2_0`、`LINK_3_0`、`m_base_5` 差 15–18 cm：
上游 OBJ 里有几片孤立的大三角形，抽稀时被去掉；其余是细小特征被合并）；KUKA 新的加速度下 `limits.JerkLimiter`（击剑用的原语）的速度会超
0.7%，`SetpointGenerator` 不会。

## 7. 维护者

库：Chi Gui（@CHIGUI0）。每个机器人的维护者是把它带进库的人；改 `body.toml` 的数值要升 `revision`，并重跑一致性
测试。用到某个机器人的运动升级它时，要在运动自己的 PR 里重录金标准。

## 8. 天机 Marvin Pro：仅供测试

`tianji_marvin_pro` 的 `body.toml` 是 `status = "test-use"`。模型来自厂商相关的 RoboTwin 分支
（KLMmotion/marvin_pro_robotwin，MIT）；**公开发布前需要天机的书面确认**。在那之前它只能用在 `publish = false`
的运动里（`tools/check` 会拒绝）。

## 9. 已知缺口

- 只有击剑通过这条路径执行机械臂限位；乒乓、网球、羽毛球、篮球、空气曲棍球还没有接入（审计 G1–G5，各运动的 PR）。
- 全身规划器只按速度上限规划；R1 Pro 的 SDK 加速度（4.5 rad/s²）下挥拍会被伺服明显放慢，快速项目基本打不了
  （审计 D5：慢项目身体）。RB-Y1 乒乓在 15 rad/s² 下要重新测（审计 §7.3 #10）。
- H1-2 的电机曲线是估计值（Unitree 没有公开它的电机分配）。电机曲线的制动侧按 unitree_actuators.py 的示意图保持 Y2 不随速度下降（它的代码也让制动力矩衰减，会让被推过 X2 的关节失去制动）。
- 附件都没有执行器限位（审计 G10）：第一个真正驱动夹爪 / 吸盘的运动要补上真实夹爪的数值。
- 关节的实际速度（物理，不是设定点）在冲击和 PD 跟踪下可能超过限位，按审计 D9 只记录。
- 自碰撞（§4.1）：KUKA iiwa 14（空气曲棍球）还没有 `[self_collision]`；用 Franka 的运动（乒乓、网球、羽毛球、击剑、篮球）
  要在各自的 `sport.toml [self_collision]` 里接入，接入前 CI 只给警告。碰撞几何是上游的拟合胶囊 / 盒子 / 凸包，比真机外壳粗，
  排除清单里的肩部壳体就是这么来的；更细的凸分解以后再做。
