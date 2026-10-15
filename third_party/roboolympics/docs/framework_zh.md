# 框架说明

版本 v0.2（2026-10-06；10-08 加入两种模式的统一定义和共享的机器人资源库；10-09 加入能力和"比赛智能"的对应；10-11 加入保密官方种子和连续分数，核心 0.2.14，6.4 节）。本文说明框架分成哪几部分、各部分做什么、彼此通过什么约定相连，以及怎样加一项运动。物理引擎、渲染器这些基础概念见单独的[《背景：模拟器、物理引擎与渲染器》](?doc=background)。

**怎么读**：想了解全貌，读第 1–3 节；要加一项运动，读第 4.2 节和第 8 节；要接入一个新模型，读第 4.1 节和第 3 节的"选手"；要做分析或报表，读第 4.3 节和第 7 节；想知道两种模式（Agent as Athlete / Agent as Robotics Engineer）各测什么、给什么工具，读第 6.3 节；九项能力和运动科学里"比赛智能"（读比赛）的对应，在 6.3.2 节末尾。

## 1. 总览

<svg viewBox="0 0 920 540" width="100%" role="img" aria-label="框架的四个部分、三份约定、机器人资源库和下面的引擎" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.h{fill:var(--fg,#16211c);font-size:15px;font-weight:600}.s{fill:var(--muted,#56665e);font-size:12px}.m{fill:var(--fg,#16211c);font-size:11.5px;font-family:ui-monospace,Menlo,monospace}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.fpl{fill:rgba(76,170,124,.06)}.fsp{fill:rgba(236,146,82,.06)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.ln{stroke:var(--muted,#56665e);stroke-width:1.2;fill:none}</style>
<defs><marker id="m80217" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<polygon points="24,100 236,100 300,136 300,258 236,294 24,294" class="fpl"/>
<polygon points="896,100 684,100 620,136 620,258 684,294 896,294" class="fsp"/>
<text x="130" y="52" text-anchor="middle" class="h">选手 players/</text>
<text x="130" y="74" text-anchor="middle" class="s">模型 + harness + 玩法配置</text>
<rect x="32" y="110" width="196" height="36" rx="7" class="pl" stroke-width="1.3"/>
<text x="130" y="134" text-anchor="middle" class="t">Opus 5.5 · Claude Code</text>
<rect x="32" y="156" width="196" height="36" rx="7" class="pl" stroke-width="1.3"/>
<text x="130" y="180" text-anchor="middle" class="t">Fable 5.1 · Claude Code</text>
<rect x="32" y="202" width="196" height="36" rx="7" class="pl" stroke-width="1.3"/>
<text x="130" y="226" text-anchor="middle" class="t">GPT-6 · Codex</text>
<rect x="32" y="248" width="196" height="36" rx="7" class="pl" stroke-width="1.3" style="stroke-dasharray:5 4"/>
<text x="130" y="272" text-anchor="middle" class="s">… 随时可加</text>
<text x="790" y="52" text-anchor="middle" class="h">运动 sports/</text>
<text x="790" y="74" text-anchor="middle" class="s">世界 + 动作 + 裁判 + 观测</text>
<rect x="692" y="110" width="196" height="36" rx="7" class="sp" stroke-width="1.3"/>
<text x="790" y="134" text-anchor="middle" class="t">乒乓球</text>
<rect x="692" y="156" width="196" height="36" rx="7" class="sp" stroke-width="1.3"/>
<text x="790" y="180" text-anchor="middle" class="t">足球 5v5</text>
<rect x="692" y="202" width="196" height="36" rx="7" class="sp" stroke-width="1.3"/>
<text x="790" y="226" text-anchor="middle" class="t">击剑</text>
<rect x="692" y="248" width="196" height="36" rx="7" class="sp" stroke-width="1.3" style="stroke-dasharray:5 4"/>
<text x="790" y="272" text-anchor="middle" class="s">… 随时可加</text>
<rect x="692" y="304" width="196" height="40" rx="7" class="nu" stroke-width="1.3" style="stroke-dasharray:4 3"/>
<text x="790" y="320" text-anchor="middle" class="s">共享资源：机器人资源库 bodies/</text>
<text x="790" y="336" text-anchor="middle" class="s">运动可以引用，不是框架的部分</text>
<rect x="300" y="124" width="320" height="198" rx="12" fill="none" stroke="rgba(98,148,222,.7)" stroke-width="1.6"/>
<rect x="312" y="136" width="86" height="122" rx="8" class="co" stroke-width="1.3"/>
<text x="355" y="192" text-anchor="middle" class="t b">①</text>
<text x="355" y="214" text-anchor="middle" class="t b">选手协议</text>
<rect x="406" y="136" width="108" height="122" rx="8" class="co" stroke-width="1.3"/>
<text x="460" y="180" text-anchor="middle" class="t b">核心 core/</text>
<text x="460" y="202" text-anchor="middle" class="t">流程 · 规则</text>
<text x="460" y="224" text-anchor="middle" class="t">记录 · 判定</text>
<rect x="522" y="136" width="86" height="122" rx="8" class="co" stroke-width="1.3"/>
<text x="565" y="192" text-anchor="middle" class="t b">②</text>
<text x="565" y="214" text-anchor="middle" class="t b">运动接口</text>
<rect x="312" y="266" width="296" height="44" rx="8" class="co" stroke-width="1.3"/>
<text x="460" y="294" text-anchor="middle" class="t b">③ 比赛记录</text>
<path d="M460 322L460 356" class="ar" marker-end="url(#m80217)"/>
<rect x="30" y="358" width="860" height="84" rx="10" class="to" stroke-width="1.3"/>
<text x="50" y="380" class="t b">工具 tools/</text>
<text x="130" y="380" class="s">只读写文件，不进入比赛</text>
<rect x="50" y="392" width="190" height="36" rx="7" class="to" stroke-width="1.3"/>
<text x="145" y="416" text-anchor="middle" class="t">生成与检查</text>
<rect x="260" y="392" width="190" height="36" rx="7" class="to" stroke-width="1.3"/>
<text x="355" y="416" text-anchor="middle" class="t">运行与调度</text>
<rect x="470" y="392" width="190" height="36" rx="7" class="to" stroke-width="1.3"/>
<text x="565" y="416" text-anchor="middle" class="t">观看</text>
<rect x="680" y="392" width="190" height="36" rx="7" class="to" stroke-width="1.3"/>
<text x="775" y="416" text-anchor="middle" class="t">分析</text>
<rect x="30" y="462" width="860" height="66" rx="10" class="gr" stroke-width="1.3"/>
<text x="50" y="484" class="t b">引擎</text>
<rect x="50" y="494" width="400" height="26" rx="6" class="nu" stroke-width="1.3"/>
<text x="250" y="512" text-anchor="middle" class="s">物理引擎：MuJoCo · pooltool …（可扩展）</text>
<rect x="470" y="494" width="400" height="26" rx="6" class="nu" stroke-width="1.3"/>
<text x="670" y="512" text-anchor="middle" class="s">渲染器：MuJoCo · Isaac · Cycles …（可扩展）</text>
</svg>

**图 1　框架的四个部分、三份约定，以及下面的引擎；运动可以引用共享的机器人资源库（虚线框）。**

整个框架由四个部分组成：
- **选手**（左）：被评测的一方。一个选手 = 一个模型 + 它使用的 harness（例如 Claude Code、Codex）+ 玩法配置（写代码，还是不写代码）。
- **运动**（右）：比赛的一方。一项运动 = 世界（场地、球、机器人和物理）+ 动作（选手能下什么命令）+ 裁判（怎么算分）+ 观测（选手能看到什么）。
- **核心**（中）：主持每一场比赛。它负责比赛的流程（计时、轮次、复盘）、执行比赛规则（比赛形式、模式）、把发生的一切记录下来，并判定胜负和这场比赛是否有效。
- **工具**（下）：把运动和选手组织成任务、比赛和赛季，再把比赛记录变成直播和报表。工具只读写文件，不进入比赛。

运动还可以引用一个共享的**机器人资源库**（`bodies/`，右下虚线框）：真实机器人的模型、真实限位和可选的控制代码库。它是运动之间共用的资源，不是框架的第五个部分，核心也不知道它（见第 3 节和[机器人资源库](../bodies/)）。

四个部分之间只通过**三份约定**相连：
- **① 选手协议**：选手和核心之间怎么通信。
- **② 运动接口**：核心怎么调用一项运动。
- **③ 比赛记录**：核心产出哪些文件，交给工具使用。

四个部分下面还有一层**引擎**：物理引擎负责"世界怎么动"，渲染器负责"世界长什么样"。它们是被各部分调用的通用组件，两种都可以扩展：新运动可以接入别的物理引擎，也可以加新的渲染器。区别在于物理引擎由每项运动定死，渲染器可以在每场比赛和每次重渲染时选择（见 3.1 节）。

这个结构像一个沙漏：中间（核心和三份约定）窄，所有人都遵守、很少改、每次改都升版本；两边（选手和运动）宽，可以随时增加，互不影响。所以每个人只需要了解中间的一小块：加运动的人看 ②，接入模型的人看 ①，做报表的人看 ③。打个比方，这就像一个真实的联赛：运动是场馆和规则书，选手是运动员，核心是裁判组，工具是联赛办公室、转播和数据统计。

## 2. 设计原则

框架按下面八条原则设计，后文的每个决定都可以对照它们。

| 原则 | 意思 |
|---|---|
| 一句话说清 | 每部分都能用一句话说清职责；名字用"运动、选手、比赛、赛季"这种领域里的词 |
| 接口小，里面深 | 别人需要知道的东西很少，复杂的部分由核心吸收，不推给贡献者 |
| 一种改动只碰一个地方 | 加运动、接模型、开赛季、加报表、改规则，各自只改一个目录 |
| 正交，能自由组合 | 运动、比赛形式、模式、选手互相独立；核心里没有针对某项运动的特殊处理 |
| 边界都能自动检查 | 每份约定都有机器检查，违反时自动报错，而不是靠人发现 |
| 对的事容易做 | 可复现、公平、不泄露信息是默认行为，不靠作者记得 |
| 只抽象真正的共性 | 各项运动的物理各不相同，不强行统一 |
| 版本清楚，结果可比 | 所有影响结果的东西都有版本号，结果记下全部版本 |

## 3. 四个部分

| 部分 | 一句话 | 里面有什么 | 谁来改 | 不能做什么 |
|---|---|---|---|---|
| **选手** `players/` | 一个模型加它的工作方式 | 各 harness 的适配（Claude Code、Codex）；玩法配置；选手清单 | 接入新模型或新 harness 的人 | 不认识任何运动；碰不到比赛内部状态 |
| **运动** `sports/<运动>/` | 定义一个世界和它的规则 | 世界、动作、裁判、观测、内置选手、规格文件 `sport.toml` | 贡献新运动的人 | 不引用别的运动；不改核心；观测不给白名单以外的东西 |
| **核心** `core/` | 按统一规则主持每一场比赛 | 流程、规则、记录、判定（见第 5 节） | 少数维护者，单独提交，升版本 | 不出现任何运动名或 harness 名；不算物理 |
| **工具** `tools/` | 把运动和选手组织成任务、比赛、赛季，再把记录变成直播和报表 | 生成与检查、运行与调度、观看、分析 | 任何人 | 不 import 模拟器；不影响比赛结果；不写死路径和密钥 |

**机器人资源库**（`bodies/`）：运动之间共享的机器人资源，不是框架的部分；运动可以引用，也可以自带机器（赛车、无人机、桌上足球杆）。分清三件事：

| 概念 | 是什么 | 放在哪里 |
|---|---|---|
| 机器人本体 | 模型、关节、电机、带出处的真实限位 | 资源库 |
| 控制能力 | 行走策略、IK 库、全身运动规划器、控制权仲裁 | 资源库里**可选**的共享代码库；运动用不用自己定，没有必须实现的统一生命周期接口 |
| 模式 | 高层提供什么、底层开放什么 | 框架定义模式（6.3 节），每项运动在自己的配置里声明；工具包清单属于运动，**不属于机器人** |

- **机器人和装配配置分开**：例如"G1 + 乒乓球拍"和"G1 + 网球拍"是两个装配配置，共用同一个 G1 基础模型。每个装配配置（机器人 + 手或工具 + 控制器）有自己的版本 hash，记进结果。换手臂是换一个机器人模型，不是换配置。
- **限位只走一个模块**：真实限位由资源库里唯一可信的共享限位和伺服模块执行，用资源库机器人的运动必须调用它、不能绕开；CI 对每个这样的运动检查（见第 9 节）。
- **先复用资源，再复用代码**：模型、网格、限位优先共享；控制代码只有多项运动真的共用时才放进资源库。现有的全身运动规划器（`bodies/wholebody.py`）、G1 的控制权仲裁、IK 库都作为可选的共享库保留。
- 现有的机器人（机械臂、轮式人形、腿式人形）见[机器人资源库](../bodies/)。

**选手的玩法配置**：写代码时，选手正常使用 harness 的全部工具，自己写程序、在比赛中运行、复盘时修改。不写代码时，用的仍是同一个 harness，只是关掉执行代码的能力，而且要有结构上的保证：Claude Code 只开放 `game` 工具（工具白名单）；Codex 由网关在执行前拦下所有不合规的调用。另有一个自写的最小循环（看图 → 调一次模型 → 出一个动作），作为备选配置。（2026-10-10：main 上只保留写代码玩法；不写代码玩法的代码、任务和测试存档在分支 `archive/direct-play`，本节和 6.2 节关于不写代码的内容保留作设计记录。）

### 3.1 引擎：物理引擎和渲染器

<svg viewBox="0 0 920 312" width="100%" role="img" aria-label="渲染器：选手和观众的画面共用" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.h{fill:var(--fg,#16211c);font-size:15px;font-weight:600}.s{fill:var(--muted,#56665e);font-size:12px}.m{fill:var(--fg,#16211c);font-size:11.5px;font-family:ui-monospace,Menlo,monospace}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.fpl{fill:rgba(76,170,124,.06)}.fsp{fill:rgba(236,146,82,.06)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.ln{stroke:var(--muted,#56665e);stroke-width:1.2;fill:none}</style>
<defs><marker id="m95351" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<rect x="20" y="60" width="190" height="170" rx="8" class="sp" stroke-width="1.3"/>
<text x="115" y="111" text-anchor="middle" class="t b csp">运动提供</text>
<text x="115" y="133" text-anchor="middle" class="s">场景（世界的几何）</text>
<text x="115" y="152" text-anchor="middle" class="s">相机定义</text>
<text x="115" y="171" text-anchor="middle" class="s">（位置、视角、尺寸、帧数）</text>
<text x="115" y="190" text-anchor="middle" class="s">可选：展示外观、展示场景</text>
<path d="M212 145L268 145" class="ar" marker-end="url(#m95351)"/>
<rect x="270" y="20" width="300" height="250" rx="10" class="nu" stroke-width="1.3"/>
<text x="420" y="44" text-anchor="middle" class="t b">渲染器 renderers/（插件）</text>
<rect x="286" y="58" width="268" height="38" rx="6" class="gr" stroke-width="1.3"/>
<text x="420" y="82" text-anchor="middle" class="s">MuJoCo　默认 · 实时 · CPU 或 GPU</text>
<rect x="286" y="104" width="268" height="38" rx="6" class="gr" stroke-width="1.3"/>
<text x="420" y="128" text-anchor="middle" class="s">MuJoCo 增强　阴影、材质 · 实时</text>
<rect x="286" y="150" width="268" height="38" rx="6" class="gr" stroke-width="1.3"/>
<text x="420" y="174" text-anchor="middle" class="s">Isaac Sim RTX　实时光追或路径追踪 · GPU</text>
<rect x="286" y="196" width="268" height="38" rx="6" class="gr" stroke-width="1.3"/>
<text x="420" y="220" text-anchor="middle" class="s">Blender Cycles　路径追踪 · 离线</text>
<rect x="286" y="242" width="268" height="20" rx="6" class="gr" stroke-width="1.3" style="stroke-dasharray:5 4"/>
<text x="420" y="258" text-anchor="middle" class="s">… 随时可加</text>
<path d="M572 100L628 80" class="ar" marker-end="url(#m95351)"/>
<path d="M572 190L628 210" class="ar" marker-end="url(#m95351)"/>
<rect x="630" y="20" width="270" height="120" rx="8" class="co" stroke-width="1.3"/>
<text x="765" y="56" text-anchor="middle" class="t b cco">选手的画面（比赛中）</text>
<text x="765" y="78" text-anchor="middle" class="s">由核心调用，按任务配置选</text>
<text x="765" y="96" text-anchor="middle" class="s">必须满足每帧的时间预算</text>
<text x="765" y="116" text-anchor="middle" class="s">结果里记下用的渲染器</text>
<rect x="630" y="150" width="270" height="120" rx="8" class="to" stroke-width="1.3"/>
<text x="765" y="186" text-anchor="middle" class="t b cto">观众的画面（直播、宣传）</text>
<text x="765" y="208" text-anchor="middle" class="s">由观看工具调用，随时可换</text>
<text x="765" y="226" text-anchor="middle" class="s">读比赛记录重放后再拍</text>
<text x="765" y="246" text-anchor="middle" class="s">每一帧标出用的渲染器</text>
<text x="20" y="296" class="s">所有渲染器拍的是同一份场景：比赛区域的几何、相机位置在所有渲染器里一致，只有灯光、材质、画质不同；渲染器不改物理。</text>
</svg>

**图 2　渲染器以插件形式接入，选手的画面和观众的画面都可以自由选择用哪一个。**

四个部分下面有两种引擎。它们不是我们为这个框架写的代码，而是被调用的通用组件，就像一个联赛要用的场地设施和摄像设备：
- **物理引擎**：每项运动自己选定一个，写在 `sport.toml` 里。现在大多数运动用 MuJoCo，台球用专门的台球物理引擎 pooltool；以后写新运动的人也可以接入别的物理引擎，核心不关心用的是哪一个。唯一的要求是能一步步推进，而且同样的输入永远得到同样的输出，这样比赛才能重放、能验证。换物理引擎会改变比赛结果，所以它由运动定死，不让选手或赛季去选。
- **渲染器**：负责把某一刻的世界"拍成图像"。它不改变物理，所以可以自由选择，以插件的形式接入 `renderers/`：

- **它从哪里拿东西**：运动提供场景（世界的几何）和相机定义（位置、视角、尺寸、帧数），可以另外提供展示外观：灯光和材质的预设（`Scene.look`），以及只给观众看的展示场景（见下）；渲染器读场景和每一帧的状态，按相机定义出图。
- **选手的画面**：比赛中，核心按任务配置调用渲染器，给选手拍相机画面。默认是 MuJoCo 自带的渲染器；也可以选更好看的实时渲染器（例如 MuJoCo 增强、Isaac Sim 的实时光追）。用哪个渲染器是任务的一个参数，和慢放倍数一样；拍一帧的时间要跟得上比赛，所以路径追踪这类慢渲染器只用在观众这边，需要 GPU 的渲染器在任务里注明。
- **观众的画面**：直播和宣传片里，观看工具读比赛记录、重放比赛，再用任意渲染器重新拍，包括又慢又逼真的路径追踪。
- **换渲染器不换比赛**：所有渲染器拍的是同一份场景，比赛区域的几何和相机位置一致，只有灯光、材质和画质不同。所以换观众的渲染器不影响任何结果；换选手的渲染器会改变选手看到的画面，这时成绩按渲染器分开比较。
- **展示场景（只给观众）**：比赛场景里的机器人通常画的是碰撞网格，不像真实的机器人。运动可以在 `sports/<运动>/presentation/presentation.toml` 里声明一份展示场景：在比赛场景的**同一批刚体**上挂上精细的外观网格（例如 MuJoCo Menagerie 的 Franka、KUKA iiwa），这些网格只看不碰、没有质量，原来的碰撞网格留在模型里，只是不画（移到第 3 组）。渲染器画展示场景之前自动核对"运动学相同"：刚体、关节、坐标系、惯量、标记点和相机都和比赛场景一样，比赛场景的每个几何体形状、位置、碰撞参数不变（只能改外观或隐藏），新增的只能是不参与碰撞的可见几何体；不一致就拒绝出图。所以重放同一场比赛，每个刚体的位姿逐位相同，展示场景只改变观众看到的样子。它由观众这边使用：核心的回放视频（有展示场景就用，`RO_VIDEO_SCENE=match` 可关掉）、重新渲染和宣传图；每一帧都标出"presentation meshes, spectators only"。只给观众时，展示场景的文件不进模拟器目录，所以 `source_sha`、标准回放和任务目录都不变；第三方模型文件不进仓库，构建时按固定的上游提交下载并逐个核对 sha256（`python3 -m renderers.presentation.assets fetch`），只读取数据，保留原许可证并在 NOTICE 里注明出处。
- **选手的画面**：默认用 `Sport.scene` 返回的比赛场景，核心拿到只给观众的展示场景会直接报错。运动可以在 `presentation.toml` 里声明 `players = "25k"`（或 `"full"`），让选手的画面也用展示场景（核心 0.2.9）：运动学照样在出图前自动核对；生成器把声明和模型文件（按声明的网格精度，逐个核对 sha256，带上许可证）打包进模拟器目录（`sport/presentation/`），所以比赛时不用联网、不用缓存，这些文件也算进 `source_sha`；这时的画面就是比赛画面，不再标"spectators only"。精细网格在 CPU 上画得更慢（实测完整的 Menagerie 网格让一个观察慢 1.3 到 6.3 倍：8 个画面、两台 Franka 时每个观察多 300 到 470 毫秒，给到 8 到 16 个线程仍多 250 到 290 毫秒，因为 llvmpipe 的顶点处理基本不随线程数加速），所以选手用的是减面版本：`renderers/presentation/lod.json` 里的细节层级（例如每台机器人约 2.5 万个三角形），由 `renderers/presentation/lod.py` 从固定提交的上游文件生成，体积小（Franka 0.8 MB），提交进仓库，许可证不变。运动在 `[observation]` 里声明 `render_threads`（每个渲染进程的 llvmpipe 线程数，生成器写成 `LP_NUM_THREADS`）和 `sim_cpus`（模拟器容器的 CPU 数），帧预算由 tools/check 量（运动自己出图的也量它自己的出图路径）。换成精细网格会改变选手看到的画面：运动要升版本，并重新验证参考选手。
- **加一个新渲染器**：在 `renderers/` 里实现"读场景和每一帧的状态 → 按相机定义出图"即可，不用改任何运动。

### 3.2 运行层：Harbor

模型参加的比赛由 [Harbor](https://github.com/CHIGUI0/harbor)（通用的 agent 评测执行器，我们用自己的 fork，README 里锁定了具体提交）运行。分工是：**框架定义"这场比赛是什么"，Harbor 负责"把它跑起来"**。

| | Harbor 负责 | 框架负责 |
|---|---|---|
| 内容 | 建容器、隔离网络、安装和启动 agent（Claude Code、Codex）、计时、收集产物、在单独的容器里运行评分、保存结果 | 世界、动作、裁判、观测（运动）；流程、规则、记录、判定（核心）；选手接入（选手）；生成、检查、分析（工具） |
| 交接 | 读取标准的任务目录：`task.toml`、`instruction.md`、`environment/`、`tests/` | `tools/generate.py` 把每项运动的 `sport.toml` 展开成这样的任务目录，放在 `tasks/` 下 |

一场模型比赛：Harbor 拉起一个 **sim 容器**（核心加运动插件组成的比赛服务，以及统计思考时间的计量代理）和每个选手一个**座位容器**（装着 harness，只能经内部网络访问 sim）；选手看图、写程序、发命令；比赛结束后，Harbor 在独立的评分容器里运行核心的评分：重放比赛记录、核对哈希、判定胜负，写出 `reward.json`。对战需要一场比赛有多个座位容器、每个座位有自己的网络和账号，这是我们在 fork 里加的。

不经过 Harbor 的部分：`tools/check` 的规格检查、通用测试、标准回放和公平检查，以及内置选手之间的对局，都在单个容器里直接跑；实跑阶段默认用 `tools/live.py` 在一个容器里模拟 Harbor 的流程，加 `--harbor` 才真正走 Harbor。以后换别的执行器，只需要换生成器的输出，运动和核心不用改。

## 4. 三份约定

### 4.1 ① 选手协议

<svg viewBox="0 0 920 340" width="100%" role="img" aria-label="选手协议：一场比赛里选手调用的端点" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.h{fill:var(--fg,#16211c);font-size:15px;font-weight:600}.s{fill:var(--muted,#56665e);font-size:12px}.m{fill:var(--fg,#16211c);font-size:11.5px;font-family:ui-monospace,Menlo,monospace}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.fpl{fill:rgba(76,170,124,.06)}.fsp{fill:rgba(236,146,82,.06)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.ln{stroke:var(--muted,#56665e);stroke-width:1.2;fill:none}</style>
<defs><marker id="m54497" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<rect x="20" y="20" width="150" height="300" rx="8" class="pl" stroke-width="1.3"/>
<text x="95" y="126" text-anchor="middle" class="t b">选手</text>
<text x="95" y="148" text-anchor="middle" class="s">它写的程序</text>
<text x="95" y="168" text-anchor="middle" class="s">或模型本人</text>
<text x="95" y="186" text-anchor="middle" class="s"></text>
<text x="95" y="206" text-anchor="middle" class="s">用 game 命令</text>
<text x="95" y="224" text-anchor="middle" class="s">或直接发请求</text>
<path d="M172 170L196 170" class="ar" marker-end="url(#m54497)"/>
<rect x="200" y="20" width="430" height="300" rx="10" class="co" stroke-width="1.3"/>
<text x="415" y="44" text-anchor="middle" class="t b">一场比赛里依次调用的端点（所有运动相同）</text>
<rect x="216" y="64" width="398" height="34" rx="6" class="nu" stroke-width="1.3"/>
<text x="230" y="86" class="m">GET /spec</text>
<text x="356" y="86" class="s">读说明书：动作类型和范围、相机、单位</text>
<rect x="216" y="106" width="398" height="34" rx="6" class="nu" stroke-width="1.3"/>
<text x="230" y="128" class="m">POST /start</text>
<text x="356" y="128" class="s">准备好了，开赛</text>
<rect x="216" y="148" width="398" height="114" rx="8" fill="none" stroke="rgba(98,148,222,.6)" stroke-width="1.2" stroke-dasharray="5 4"/>
<text x="230" y="166" class="s">比赛进行中，反复调用：</text>
<text x="230" y="190" class="m">POST /observe</text>
<text x="356" y="190" class="s">看相机画面、自己的状态、裁判信息</text>
<text x="230" y="217" class="m">POST /act</text>
<text x="356" y="217" class="s">出手：一个动作命令（内容见右边）</text>
<text x="230" y="244" class="m">POST /wait</text>
<text x="356" y="244" class="s">等到某个时刻或事件</text>
<rect x="216" y="270" width="398" height="34" rx="6" class="nu" stroke-width="1.3"/>
<text x="230" y="292" class="m">POST /ready</text>
<text x="356" y="292" class="s">一个回合结束后：复盘完，继续</text>
<path d="M632 205L656 205" class="ar" marker-end="url(#m54497)"/>
<rect x="660" y="20" width="240" height="300" rx="10" class="sp" stroke-width="1.3"/>
<text x="780" y="44" text-anchor="middle" class="t b">动作的内容由运动决定</text>
<text x="676" y="76" class="t">乒乓球 · 高层</text>
<text x="676" y="96" class="m">{"type":"act", t, pos,</text>
<text x="676" y="113" class="m"> normal, vel}</text>
<text x="676" y="144" class="t">乒乓球 · 底层</text>
<text x="676" y="164" class="m">{"type":"traj", k0, q}</text>
<text x="676" y="195" class="t">足球</text>
<text x="676" y="215" class="m">{"type":"walk", v}</text>
<text x="676" y="250" class="s">类型和范围写在 /spec 里</text>
<text x="676" y="274" class="s">出错统一：400 不合法</text>
<text x="676" y="292" class="s">409 冻结中 · 429 未完成</text>
</svg>

**图 3　选手在一场比赛里调用的端点；动作的内容由运动决定。**

选手和比赛之间只有一条通道：比赛容器里的一个 HTTP 服务。选手（它写的程序，或者不写代码时的模型本人）用 `game` 命令行或直接发请求，按图中的顺序调用：先读说明书、开赛，比赛中反复看画面、出手、等待，每个回合结束后复盘完再继续。**协议只规定"信封"**，也就是有哪些端点、回复是什么格式、出错怎么报，不规定动作的内容。动作里装什么，由运动当前生效的动作层决定，并在 `/spec` 里公布（图右是三个真实的例子）。所以不同运动、不同模式的动作各不相同，协议本身不用改。另外，选手的所有模型调用都经过核心的计量通道，思考时间、token 和 API 错误由核心统一记录。

### 4.2 ② 运动接口

<svg viewBox="0 0 920 362" width="100%" role="img" aria-label="一项运动的组成" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.h{fill:var(--fg,#16211c);font-size:15px;font-weight:600}.s{fill:var(--muted,#56665e);font-size:12px}.m{fill:var(--fg,#16211c);font-size:11.5px;font-family:ui-monospace,Menlo,monospace}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.fpl{fill:rgba(76,170,124,.06)}.fsp{fill:rgba(236,146,82,.06)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.ln{stroke:var(--muted,#56665e);stroke-width:1.2;fill:none}</style>
<defs><marker id="m7609" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<rect x="20" y="20" width="130" height="250" rx="8" class="co" stroke-width="1.3"/>
<text x="85" y="140" text-anchor="middle" class="t b">核心</text>
<text x="85" y="162" text-anchor="middle" class="s">只调用这些函数</text>
<path d="M152 145L186 145" class="ar" marker-end="url(#m7609)"/>
<rect x="190" y="20" width="350" height="78" rx="8" class="sp" stroke-width="1.3"/>
<text x="204" y="44" class="t b">世界</text>
<text x="204" y="66" class="s">场地、球、机器人、物理参数</text>
<text x="204" y="85" class="m">build · step · state_digest</text>
<rect x="550" y="20" width="350" height="78" rx="8" class="sp" stroke-width="1.3"/>
<text x="564" y="44" class="t b">动作</text>
<text x="564" y="66" class="s">底层必需；高层插件可选</text>
<text x="564" y="85" class="m">schema · apply · [lower]</text>
<rect x="190" y="106" width="350" height="78" rx="8" class="sp" stroke-width="1.3"/>
<text x="204" y="130" class="t b">裁判</text>
<text x="204" y="152" class="s">只读物理事件，判得分和胜负</text>
<text x="204" y="171" class="m">phase · scores · tiebreaks</text>
<rect x="550" y="106" width="350" height="78" rx="8" class="sp" stroke-width="1.3"/>
<text x="564" y="130" class="t b">观测</text>
<text x="564" y="152" class="s">相机 + 自己的身体 + 裁判信息</text>
<text x="564" y="171" class="m">observe · cameras</text>
<rect x="190" y="192" width="710" height="78" rx="8" class="sp" stroke-width="1.3"/>
<text x="204" y="216" class="t b">内置选手</text>
<text x="204" y="238" class="s">参考选手（只看相机，证明能赢）· 脚本对手 L1–L3 · 检查用的脚本策略</text>
<text x="204" y="257" class="s">和模型选手走同一个动作接口</text>
<rect x="20" y="284" width="880" height="64" rx="8" class="sp" stroke-width="1.3" style="stroke-dasharray:5 4"/>
<text x="34" y="310" class="t b">规格 sport.toml（数据）</text>
<text x="34" y="332" class="s">规则书与偏离 · 比赛形式 · 动作层级 · 观测白名单 · 决胜顺序 · 漏洞清单；提示词和 README 由它生成</text>
</svg>

**图 4　一项运动由四组必需的部分、内置选手和一份规格文件组成，核心只调用图中列出的函数。**

一项运动要回答四个问题，每个问题对应一组函数：

| 组 | 回答的问题 | 必需的函数 | 可选的函数 |
|---|---|---|---|
| 世界 | 有什么东西，物理怎么推进？ | `build(配置)` · `step()` → 物理事件 · `state_digest()` | — |
| 动作 | 选手能下什么命令？ | `schema()` · `apply(座位, 底层命令)` | 高层插件：`lower(座位, 高层命令)` → 底层命令 |
| 裁判 | 什么算得分、犯规，怎么分出胜负？ | `phase()` · `scores()` · `tiebreaks()` | 回合制：`to_move()`；`forfeit()`、`resume()` |
| 观测 | 选手能看到什么？ | `observe(座位)`（自身 + 裁判信息）· `cameras(座位)`（相机定义，由渲染器拍） | `reply_views(座位, 命令, 结果)`（动作回复里的结果图像）；`broadcast_frame()`（给直播）；展示外观 |

另外两样：
- **内置选手**：我们为这项运动写的选手。参考选手只看相机，用来证明这项运动打得赢；脚本对手是单人赛里 L1–L3 难度的对手；还有做可行性和公平检查用的脚本策略。它们和模型选手走同一个动作接口。
- **规格文件 `sport.toml`**：这项运动的所有声明都写在这里，是数据而不是代码。生成器、评分、文档和网站都读它，提示词和 README 里的规则说明也由它生成，所以只写一处。

**动作回复里的图像**（核心 0.2.8，可选）：有的运动要让选手在动作的回复里直接看到这个动作的结果，方便复盘和分析，比如射箭每一箭后的靶面特写（一组的第 3 箭射完箭就拔了，之后再也看不到）、飞镖每一镖后的镖盘特写（本轮已投的几镖）、篮球投篮后球最接近篮筐前后的几帧。运动实现 `reply_views(座位, 底层命令, apply 的返回)`，按动作声明要拍哪些相机，还可以给一小串帧（每帧自己的渲染状态和标签，按"逐帧"或"逐相机"排列）；不实现就没有回复图像。核心在动作生效后、仍在锁内时立刻取这份快照，然后**在锁外**用任务的选手渲染器画出来（和观测图像同一条路：同一个渲染进程或相机工作器、只用比赛场景、同一帧时间预算），以 `observation`（`image_png_b64` + `tiles`，和 `/observe` 的格式相同，帧多时每个图块另标 `frame`、`label`）附在回复里。画图的时间不算选手的棋钟（核心在画图期间停住这个座位的钟），实时项目里也不算进动作的计时（`wall_s_after_request` 在画图前取，画图时间单独报 `render_wall_s`），也不挡其他座位的请求；比赛日志里另记一行 `reply_images`（图块数、字节数、画图时间，只作信息，不参与重放）。图像只由比赛状态和这一步的命令决定，所以重放可以原样画出来（`python -m core.episode 插件 目录 --reply N`），标准回放的逐位核对不受影响。规格里用 `[observation] reply_images` 一句话写明回复里有什么图，生成的提示词据此说明；`tools/check` 检查声明和实现一致，通用测试检查它不改状态、不超帧时间预算。回合制里核心还把"谁正在复盘"告诉运动（`Match.awaiting_review`）：只在轮到自己时给图的运动，用它在行动方复盘期间不给另一方图像。

规划进程、渲染进程这类性能细节由核心提供通用工具，运动只需标出哪个函数比较慢。世界和裁判在概念上分开：世界只提供物理事实，裁判只根据物理事件做判断。

### 4.3 ③ 比赛记录

每场比赛产出下面几份文件，工具只读这些文件：

| 文件 | 内容 |
|---|---|
| `episode.jsonl` | 比赛日志：每条生效的命令、生效的控制步、状态哈希。可以在新进程里完整重放 |
| `think_calls*.jsonl` | 每次模型调用的起止时间、token、错误 |
| `final.json` / `reward.json` | 比分、决胜方式、技术故障类别，以及 `success`、`won`、`valid_trial`、`deterministic` 等字段；连续分数 `<座位>_continuous_score`（单方形式另有 `continuous_score`）和保密种子的核验结果 `hidden_seed`、`hidden_seed_index`（核心 0.2.14，6.4 节；保密种子的比赛里 `seed` 为 -1，任何记录里都没有种子） |
| `broadcast.jsonl` | 直播用的画面状态，只给观看工具读 |
| 选手程序快照（计划） | 每次复盘时选手程序的版本和改动，用来分析"边打边改程序" |

每条结果都记下**四个版本号**（协议版本、核心版本、运动规则版本、选手配置版本）和**选手画面用的渲染器**。版本或渲染器不同的结果不会合在一张榜里比较。

**排行榜和统计读两个数**：胜负（`<座位>_won`，单人形式的 `success`）和连续分数（`<座位>_continuous_score`，越大越好）。连续分数怎么算由运动定义，字段名和方向由核心统一（6.4 节）。

## 5. 核心怎样主持一场比赛

<svg viewBox="0 0 920 296" width="100%" role="img" aria-label="一场比赛的生命周期和核心的工作" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.h{fill:var(--fg,#16211c);font-size:15px;font-weight:600}.s{fill:var(--muted,#56665e);font-size:12px}.m{fill:var(--fg,#16211c);font-size:11.5px;font-family:ui-monospace,Menlo,monospace}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.fpl{fill:rgba(76,170,124,.06)}.fsp{fill:rgba(236,146,82,.06)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.ln{stroke:var(--muted,#56665e);stroke-width:1.2;fill:none}</style>
<defs><marker id="m43565" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<rect x="20" y="40" width="160" height="96" rx="8" class="nu" stroke-width="1.3"/>
<text x="100" y="73" text-anchor="middle" class="t b">① 准备</text>
<text x="100" y="95" text-anchor="middle" class="s">世界冻结</text>
<text x="100" y="114" text-anchor="middle" class="s">读说明、写程序</text>
<rect x="20" y="176" width="160" height="80" rx="8" class="co" stroke-width="1.3"/>
<text x="100" y="210" text-anchor="middle" class="t b cco">规则</text>
<text x="100" y="232" text-anchor="middle" class="s">安排座位和先手</text>
<path d="M100 138V174" class="ar" stroke-dasharray="4 3"/>
<path d="M182 88L198 88" class="ar" marker-end="url(#m43565)"/>
<rect x="200" y="40" width="160" height="96" rx="8" class="nu" stroke-width="1.3"/>
<text x="280" y="73" text-anchor="middle" class="t b">② 进行中</text>
<text x="280" y="95" text-anchor="middle" class="s">世界在跑（×S 慢放）</text>
<text x="280" y="114" text-anchor="middle" class="s">看、出手、等</text>
<rect x="200" y="176" width="160" height="80" rx="8" class="co" stroke-width="1.3"/>
<text x="280" y="201" text-anchor="middle" class="t b cco">流程 · 记录</text>
<text x="280" y="223" text-anchor="middle" class="s">计时、一次一个动作</text>
<text x="280" y="242" text-anchor="middle" class="s">记日志和计量</text>
<path d="M280 138V174" class="ar" stroke-dasharray="4 3"/>
<path d="M362 88L378 88" class="ar" marker-end="url(#m43565)"/>
<rect x="380" y="40" width="160" height="96" rx="8" class="nu" stroke-width="1.3"/>
<text x="460" y="73" text-anchor="middle" class="t b">③ 回合结束</text>
<text x="460" y="95" text-anchor="middle" class="s">冻结，进入复盘</text>
<text x="460" y="114" text-anchor="middle" class="s">改程序后 /ready</text>
<rect x="380" y="176" width="160" height="80" rx="8" class="co" stroke-width="1.3"/>
<text x="460" y="201" text-anchor="middle" class="t b cco">流程</text>
<text x="460" y="223" text-anchor="middle" class="s">复盘时限</text>
<text x="460" y="242" text-anchor="middle" class="s">超时判负</text>
<path d="M460 138V174" class="ar" stroke-dasharray="4 3"/>
<path d="M542 88L558 88" class="ar" marker-end="url(#m43565)"/>
<rect x="560" y="40" width="160" height="96" rx="8" class="nu" stroke-width="1.3"/>
<text x="640" y="73" text-anchor="middle" class="t b">④ 比赛结束</text>
<text x="640" y="95" text-anchor="middle" class="s">按决胜顺序</text>
<text x="640" y="114" text-anchor="middle" class="s">一定分出胜负</text>
<rect x="560" y="176" width="160" height="80" rx="8" class="co" stroke-width="1.3"/>
<text x="640" y="201" text-anchor="middle" class="t b cco">判定</text>
<text x="640" y="223" text-anchor="middle" class="s">决胜：规则书</text>
<text x="640" y="242" text-anchor="middle" class="s">→ 加赛 → 兜底</text>
<path d="M640 138V174" class="ar" stroke-dasharray="4 3"/>
<path d="M722 88L738 88" class="ar" marker-end="url(#m43565)"/>
<rect x="740" y="40" width="160" height="96" rx="8" class="nu" stroke-width="1.3"/>
<text x="820" y="73" text-anchor="middle" class="t b">⑤ 验证</text>
<text x="820" y="95" text-anchor="middle" class="s">新进程重放日志</text>
<text x="820" y="114" text-anchor="middle" class="s">哈希一致才算数</text>
<rect x="740" y="176" width="160" height="80" rx="8" class="co" stroke-width="1.3"/>
<text x="820" y="201" text-anchor="middle" class="t b cco">判定</text>
<text x="820" y="223" text-anchor="middle" class="s">有效性：故障作废</text>
<text x="820" y="242" text-anchor="middle" class="s">自己的问题算输</text>
<path d="M820 138V174" class="ar" stroke-dasharray="4 3"/>
<path d="M460 38C460 14 280 14 280 36" class="ar" marker-end="url(#m43565)"/>
<text x="370" y="14" text-anchor="middle" class="s">还没结束：双方 /ready 后回到 ②</text>
<text x="20" y="282" class="s">上排是比赛的阶段，下排是核心在这个阶段做的事（流程、规则、记录、判定四类）。</text>
</svg>

**图 5　一场比赛的五个阶段（上排），以及核心在每个阶段做的事（下排）。**

1. **准备**：世界冻结，选手读说明、写程序；核心按比赛形式安排座位和先手。双方都 `/start` 后开赛；一方开赛后，另一方超过开赛时限、或者会话结束时还没开赛，就判负（§9 核心规则 5）。
2. **进行中**：世界在跑。实时项目按 1/S 的速度推进（S 是慢放倍数，第一赛季为 10）；回合制项目轮到谁，谁的棋钟就走。选手的思考也会耗掉比赛时间。核心负责计时、一次只处理一个动作，并记下所有命令和模型调用。
3. **回合结束**：裁判判出一分、一投或一局后，世界冻结，进入复盘。选手可以改程序，然后 `/ready`；双方都准备好就回到第 2 步。对战中复盘超时判负。
4. **比赛结束**：核心按决胜顺序得出结果，一定分出胜负。单人形式开赛后会话提前结束，算有效的输（§9 核心规则 6）。
5. **验证**：隔离的验证器在新进程里从头重放比赛日志，比分和状态哈希一致才算数，再判定这场比赛是否有效。

所以核心的工作分四类：**流程**（计时、座位与轮次、回合与复盘）、**规则**（比赛形式、模式）、**记录**（比赛日志、回放、计量）、**判定**（胜负、有效性）。它对所有运动一视同仁，并保证四件事：

| 保证 | 怎么做到 |
|---|---|
| 可重放 | 所有命令进日志；随机数按名字从核心取；状态哈希覆盖所有影响比分的东西。重放对不上就算技术故障，作废重打 |
| 公平 | 所有选手用同一个时钟；一次只处理一个请求；先手、发球、发车位这类天然不对称，由比赛形式轮换 |
| 一定分出胜负 | 决胜顺序固定：规则书比分 → 规则书决胜 → 有上限的加赛 → 兜底规则；兜底规则写进提示词 |
| 有效性分明 | 我们这边的故障（模拟器崩溃、回放不一致、API 错误超过 5 次等）作废重打；选手自己的问题（超时、违规、程序崩溃）算有效的输 |

## 6. 任务、比赛与赛季

<svg viewBox="0 0 920 132" width="100%" role="img" aria-label="任务、比赛、赛季的关系" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.h{fill:var(--fg,#16211c);font-size:15px;font-weight:600}.s{fill:var(--muted,#56665e);font-size:12px}.m{fill:var(--fg,#16211c);font-size:11.5px;font-family:ui-monospace,Menlo,monospace}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.fpl{fill:rgba(76,170,124,.06)}.fsp{fill:rgba(236,146,82,.06)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.ln{stroke:var(--muted,#56665e);stroke-width:1.2;fill:none}</style>
<defs><marker id="m76698" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<text x="20" y="34" class="t b">任务</text>
<text x="60.0" y="34" class="t">=</text>
<text x="81.8" y="34" class="t b csp">运动</text>
<text x="121.8" y="34" class="t">×</text>
<text x="143.6" y="34" class="t b cco">比赛形式</text>
<text x="209.6" y="34" class="t">×</text>
<text x="231.4" y="34" class="t b cco">模式</text>
<text x="271.4" y="34" class="t">×</text>
<text x="293.2" y="34" class="t b csp">等级</text>
<text x="430" y="34" class="s">生成器从运动规格里声明的组合产出</text>
<text x="20" y="74" class="t b">比赛</text>
<text x="60.0" y="74" class="t">=</text>
<text x="81.8" y="74" class="t b">任务</text>
<text x="121.8" y="74" class="t">×</text>
<text x="143.6" y="74" class="t b cpl">选手们</text>
<text x="196.6" y="74" class="t">×</text>
<text x="218.4" y="74" class="t b">种子</text>
<text x="258.4" y="74" class="t">×</text>
<text x="280.2" y="74" class="t b cco">先手顺序</text>
<text x="430" y="74" class="s">运行工具组织；先手顺序按比赛形式轮换</text>
<text x="20" y="114" class="t b">赛季</text>
<text x="60.0" y="114" class="t">=</text>
<text x="81.8" y="114" class="t">冻结在某个版本的一组比赛</text>
<text x="430" y="114" class="s">结果只和同版本的结果一起排名</text>
</svg>

**图 6　任务、比赛、赛季的关系。**

- **任务**是一项运动在某种比赛形式、某种模式、某个难度等级下的具体设置，由生成器从运动规格里声明的组合自动产出，没有手写的任务。
- **比赛**是一个任务加上具体的选手、随机种子和先手顺序。
- **赛季**是冻结在某个版本下的一组比赛，结果只和同版本的结果放在一起排名。

### 6.1 比赛形式

<svg viewBox="0 0 920 290" width="100%" role="img" aria-label="四种比赛形式" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.h{fill:var(--fg,#16211c);font-size:15px;font-weight:600}.s{fill:var(--muted,#56665e);font-size:12px}.m{fill:var(--fg,#16211c);font-size:11.5px;font-family:ui-monospace,Menlo,monospace}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.fpl{fill:rgba(76,170,124,.06)}.fsp{fill:rgba(236,146,82,.06)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.ln{stroke:var(--muted,#56665e);stroke-width:1.2;fill:none}</style>
<defs><marker id="m50425" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<rect x="20" y="20" width="205" height="250" rx="10" class="nu" stroke-width="1.3"/>
<text x="122" y="46" text-anchor="middle" class="t b">单人</text>
<circle cx="92" cy="86" r="16" fill="rgba(120,120,120,0.55)"/><text x="92" y="90" text-anchor="middle" class="s" style="fill:#fff;font-weight:600">A</text>
<rect x="132" y="72" width="42" height="28" rx="6" class="gr" stroke-width="1.3"/>
<text x="153" y="92" text-anchor="middle" class="s">脚本</text>
<text x="36" y="136" class="s">1 个座位</text>
<text x="36" y="166" class="s">对脚本对手 L1–L3，或没有对手</text>
<text x="36" y="196" class="s">例：射箭、保龄球、高尔夫</text>
<text x="36" y="226" class="s">排名：对各等级的成绩</text>
<rect x="245" y="20" width="205" height="250" rx="10" class="nu" stroke-width="1.3"/>
<text x="347" y="46" text-anchor="middle" class="t b">对战</text>
<circle cx="312" cy="86" r="16" fill="rgba(120,120,120,0.55)"/><text x="312" y="90" text-anchor="middle" class="s" style="fill:#fff;font-weight:600">A</text>
<circle cx="382" cy="86" r="16" fill="rgba(120,120,120,0.3)"/><text x="382" y="90" text-anchor="middle" class="s" style="fill:#fff;font-weight:600">B</text>
<text x="347" y="90" text-anchor="middle" class="s">vs</text>
<text x="261" y="136" class="s">2 方 × 每方 1 座位</text>
<text x="261" y="166" class="s">两个选手正面对抗</text>
<text x="261" y="196" class="s">例：乒乓、网球、击剑、台球</text>
<text x="261" y="226" class="s">排名：胜负 → Elo</text>
<rect x="470" y="20" width="205" height="250" rx="10" class="nu" stroke-width="1.3"/>
<text x="572" y="46" text-anchor="middle" class="t b">团队</text>
<circle cx="512" cy="76" r="9" fill="rgba(120,120,120,0.55)"/>
<circle cx="534" cy="76" r="9" fill="rgba(120,120,120,0.55)"/>
<circle cx="523" cy="96" r="9" fill="rgba(120,120,120,0.55)"/>
<circle cx="610" cy="76" r="9" fill="rgba(120,120,120,0.3)"/>
<circle cx="632" cy="76" r="9" fill="rgba(120,120,120,0.3)"/>
<circle cx="621" cy="96" r="9" fill="rgba(120,120,120,0.3)"/>
<text x="572" y="90" text-anchor="middle" class="s">vs</text>
<text x="486" y="136" class="s">2 方 × 每方 k 座位</text>
<text x="486" y="166" class="s">每个机器人一个选手</text>
<text x="486" y="196" class="s">例：足球 5v5、篮球 3x3</text>
<text x="486" y="226" class="s">排名：按队计胜负 → Elo</text>
<rect x="695" y="20" width="205" height="250" rx="10" class="nu" stroke-width="1.3"/>
<text x="797" y="46" text-anchor="middle" class="t b">多方</text>
<circle cx="749" cy="86" r="12" fill="rgba(120,120,120,0.6)"/>
<circle cx="781" cy="86" r="12" fill="rgba(120,120,120,0.45)"/>
<circle cx="813" cy="86" r="12" fill="rgba(120,120,120,0.32)"/>
<circle cx="845" cy="86" r="12" fill="rgba(120,120,120,0.2)"/>
<text x="711" y="136" class="s">N 方 × 每方 1 座位</text>
<text x="711" y="166" class="s">同场同时比</text>
<text x="711" y="196" class="s">例：多车赛车</text>
<text x="711" y="226" class="s">排名：名次 → 积分</text>
</svg>

**图 7　四种比赛形式。**

比赛形式只回答"谁和谁比、每方有几个座位、怎么轮换、怎么排名"，不包含任何运动规则：几局几分、怎么得分都属于运动的裁判。每项运动在 `sport.toml` 里声明自己支持哪些形式，例如乒乓支持单人和对战，足球支持单人、对战和团队，赛车支持单人、对战和多方。选手之间没有交互的项目只保留单人形式：射箭、篮球投篮、飞盘高尔夫、推杆高尔夫、保龄球；飞镖保留对战，因为"抢先打完"有策略可做。

团队项目还要选**控制布局**：教练模式下，一个选手指挥全队（形式上就是对战），测全局视野下的战术和分工；每人一个选手时（形式上是团队），每个选手只控制自己、只看自己的相机，靠限长限速、对手看不到的队内频道沟通，测分散决策下的协作。任务参数 `RO_TEAM_CHANNEL=off` 可以关掉队内频道（队伍和规则都不变，只是队友不能说话），用来对比"能沟通"和"不能沟通"（核心 0.2.5）。核心 0.2.20 起，读频道不再"取走"消息：每条消息有本方的序号 `seq`，`observe` / `status` / `wait` 带 `team_since`（读过的最大序号）就只给更新的消息，不带就给最近 2 秒比赛时间的消息，并附 `team_seq`（本方最新序号）；`POST /<seat>/team {"since": n}` 可以翻看全部记录。同一座位上 agent 自己、它的程序、监视脚本都能各自读到同一批消息（0.2.20 以前每条消息只交给第一个来读的客户端，agent 查状态会把程序的消息"抢走"）。多座位格式（对战、团队）的每个座位的 `/app/output` 也按座位收进 `artifacts/seats/<座位>/app/output`。

### 6.2 模式

<svg viewBox="0 0 920 306" width="100%" role="img" aria-label="两个开关组成的四种组合" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.h{fill:var(--fg,#16211c);font-size:15px;font-weight:600}.s{fill:var(--muted,#56665e);font-size:12px}.m{fill:var(--fg,#16211c);font-size:11.5px;font-family:ui-monospace,Menlo,monospace}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.fpl{fill:rgba(76,170,124,.06)}.fsp{fill:rgba(236,146,82,.06)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.ln{stroke:var(--muted,#56665e);stroke-width:1.2;fill:none}</style>
<defs><marker id="m97086" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<text x="300" y="30" text-anchor="middle" class="t b">写代码</text>
<text x="300" y="48" text-anchor="middle" class="s">选手写程序，在比赛中运行</text>
<text x="680" y="30" text-anchor="middle" class="t b">不写代码</text>
<text x="680" y="48" text-anchor="middle" class="s">模型本人每次回复出一个动作</text>
<text x="60" y="124" text-anchor="middle" class="t b">高层</text>
<text x="60" y="244" text-anchor="middle" class="t b">底层</text>
<rect x="120" y="66" width="360" height="104" rx="8" class="nu" stroke-width="1.3"/>
<text x="300" y="103" text-anchor="middle" class="t b">Agent as Athlete</text>
<text x="300" y="125" text-anchor="middle" class="s">选手做竞技决策；运动技能由平台提供</text>
<text x="300" y="144" text-anchor="middle" class="s">×10 · 所有运动都支持</text>
<rect x="500" y="66" width="360" height="104" rx="8" class="nu" stroke-width="1.3"/>
<text x="680" y="103" text-anchor="middle" class="t b">不写代码</text>
<text x="680" y="125" text-anchor="middle" class="s">同一个 harness 关掉执行代码</text>
<text x="680" y="144" text-anchor="middle" class="s">大幅放慢 · 适合回合制和部分实时项目</text>
<rect x="120" y="186" width="360" height="104" rx="8" class="nu" stroke-width="1.3"/>
<text x="300" y="223" text-anchor="middle" class="t b">Agent as Robotics Engineer</text>
<text x="300" y="245" text-anchor="middle" class="s">选手用公开的基础工具自己开发运动技能</text>
<text x="300" y="264" text-anchor="middle" class="s">实现高层的技能 API · 不是每项运动都有</text>
<rect x="500" y="186" width="360" height="104" rx="8" class="gr" stroke-width="1.3" style="stroke-dasharray:5 4"/>
<text x="680" y="223" text-anchor="middle" class="t b">不提供</text>
<text x="680" y="245" text-anchor="middle" class="s">让模型逐步说出关节目标：</text>
<text x="680" y="264" text-anchor="middle" class="s">延迟跟不上，也没有意义</text>
</svg>

**图 8　模式由两个独立的开关组成，共四种组合。**

模式是任务的两个开关：
- **动作层级**（高层 Agent as Athlete / 底层 Agent as Robotics Engineer）：选手是直接用平台的运动技能去比赛，还是先用公开的基础工具自己开发运动技能，见 6.3 节。
- **玩法**（写代码 / 不写代码）：选手是写程序在比赛中运行，还是模型本人每次回复出一个动作。

其中"不写代码 + 底层"不提供。2026-10-10 起，"不写代码"整个玩法从 main 移到分支 `archive/direct-play`，main 上的任务都是写代码玩法。模型思考、选手程序和物理各有自己的频率：模型几秒想一次，程序每秒运行几十到几百次，物理每秒推进几百到上千步。让语言模型逐步说出关节目标，延迟跟不上，也不是它该做的事。

<svg viewBox="0 0 920 266" width="100%" role="img" aria-label="两个开关分别落在哪个部分" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.h{fill:var(--fg,#16211c);font-size:15px;font-weight:600}.s{fill:var(--muted,#56665e);font-size:12px}.m{fill:var(--fg,#16211c);font-size:11.5px;font-family:ui-monospace,Menlo,monospace}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.fpl{fill:rgba(76,170,124,.06)}.fsp{fill:rgba(236,146,82,.06)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.ln{stroke:var(--muted,#56665e);stroke-width:1.2;fill:none}</style>
<defs><marker id="m78160" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<text x="20" y="26" class="s">开关只改下面这几处；世界、裁判、相机和计分都不变。</text>
<rect x="20" y="44" width="130" height="96" rx="8" class="nu" stroke-width="1.3"/>
<text x="85" y="86" text-anchor="middle" class="t b">动作层级</text>
<text x="85" y="108" text-anchor="middle" class="s">高层 / 底层</text>
<rect x="170" y="44" width="360" height="96" rx="8" class="sp" stroke-width="1.3"/>
<text x="350" y="77" text-anchor="middle" class="t b csp">运动</text>
<text x="350" y="99" text-anchor="middle" class="s">声明提供哪几层：底层必需，高层插件可选</text>
<text x="350" y="118" text-anchor="middle" class="s">底层模式另给运动声明的工具包（清单公开）</text>
<rect x="550" y="44" width="350" height="96" rx="8" class="to" stroke-width="1.3"/>
<text x="725" y="77" text-anchor="middle" class="t b cto">工具 · 生成</text>
<text x="725" y="99" text-anchor="middle" class="s">任务 id 加 -lowlevel</text>
<text x="725" y="118" text-anchor="middle" class="s">说明书换成对应的动作说明</text>
<rect x="20" y="156" width="130" height="96" rx="8" class="nu" stroke-width="1.3"/>
<text x="85" y="198" text-anchor="middle" class="t b">玩法</text>
<text x="85" y="220" text-anchor="middle" class="s">写代码 / 不写代码</text>
<rect x="170" y="156" width="235" height="96" rx="8" class="pl" stroke-width="1.3"/>
<text x="288" y="189" text-anchor="middle" class="t b cpl">选手</text>
<text x="288" y="211" text-anchor="middle" class="s">不写代码时关掉执行代码</text>
<text x="288" y="230" text-anchor="middle" class="s">Claude 白名单 · Codex 网关</text>
<rect x="415" y="156" width="235" height="96" rx="8" class="co" stroke-width="1.3"/>
<text x="532" y="189" text-anchor="middle" class="t b cco">核心</text>
<text x="532" y="211" text-anchor="middle" class="s">每次回复最多一个动作</text>
<text x="532" y="230" text-anchor="middle" class="s">赛后审计，违规判负</text>
<rect x="660" y="156" width="240" height="96" rx="8" class="to" stroke-width="1.3"/>
<text x="780" y="189" text-anchor="middle" class="t b cto">工具 · 生成</text>
<text x="780" y="211" text-anchor="middle" class="s">任务 id 加 -direct-</text>
<text x="780" y="230" text-anchor="middle" class="s">说明书换成"直接玩"一节</text>
</svg>

**图 9　两个开关分别落在哪个部分。**

动作层级开关主要落在运动上：运动声明自己提供哪几层动作，底层模式另外按运动自己的工具包清单给出机器人自身的模型和基础工具。玩法开关落在选手和核心上：选手的 harness 关掉执行代码，核心执行"每次回复最多一个动作"并做赛后审计。生成器相应地改任务名和说明书。开关只改这几处，世界、裁判、相机和计分永远不变，所以同一项运动在不同模式下的结果可以直接比较。比较两种模式时，只变一个开关，对手、种子、慢放倍数都相同。哪些模式进入赛季，等赛季和任务清单确定后再定。

### 6.3 两种模式：Agent as Athlete / Agent as Robotics Engineer

**两档统一的是评测能力，不是控制算法。** 高层把运动技能交给平台，测竞技智能；底层给与机器人相适应的基础工具，把项目的核心运动技能留给 agent 自己开发。不要求所有机器人拿到相同的工具，只要求同一档测同一种能力。（2026-10-08 采纳）

#### 6.3.1 定义

**High-Level — Agent as Athlete**\
Agent 拥有完成该运动所需的基本运动技能。\
主要评测感知、预测、规划、实时决策和竞技策略。\
机器人如何移动关节、维持平衡、完成具体运动动作，由平台提供的固定控制系统负责。

**Low-Level — Agent as Robotics Engineer**\
Agent 获得机器人的物理模型、底层控制接口和必要的通用基础控制能力。\
主要评测运动技能开发、轨迹规划、控制器设计、工具复用、集成和调试。\
Agent 可以自行实现、组合或改进控制方法，开发完成体育项目所需的运动技能。

| | 高层：Agent as Athlete | 底层：Agent as Robotics Engineer |
|---|---|---|
| 平台提供 | 这项运动的技能 API（例：乒乓 hit、足球 walk / kick） | 机器人模型、关节控制接口、通用基础能力（例：行走策略、IK 库） |
| Agent 交付 | 比赛程序：看、预测、决策、策略、复盘改进 | 比赛程序，加技能 API 的实现（例：自己实现 hit），满足同一份接口规格，放在固定路径 |
| 怎么测 | 比赛：单人对脚本对手、对战 | 比赛照常（自己的策略 + 自己的实现）；赛后取最终实现单独评测：API 测试（时刻、位置、朝向、速度误差，隐藏请求）+ 比赛测试（我们固定的参考策略装在 agent 的实现上打 L1–L4） |
| 和谁比 | 不同模型之间 | 我们自己的实现（参考解），以及不同模型之间 |

<svg viewBox="0 0 920 460" width="100%" role="img" aria-label="控制链与两种模式的分工（以乒乓为例）" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.h{fill:var(--fg,#16211c);font-size:15px;font-weight:600}.s{fill:var(--muted,#56665e);font-size:12px}.m{fill:var(--fg,#16211c);font-size:11.5px;font-family:ui-monospace,Menlo,monospace}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.fpl{fill:rgba(76,170,124,.06)}.fsp{fill:rgba(236,146,82,.06)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.ln{stroke:var(--muted,#56665e);stroke-width:1.2;fill:none}</style>
<defs><marker id="m43201" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<rect x="300" y="20" width="400" height="52" rx="8" class="gr" stroke-width="1.3" style="stroke-dasharray:5 4"/>
<text x="500" y="40" text-anchor="middle" class="t">"打一个正手"</text>
<text x="500" y="62" text-anchor="middle" class="s">太高层，难的判断都被做掉了：不提供</text>
<rect x="300" y="92" width="400" height="66" rx="8" class="pl" stroke-width="1.3"/>
<text x="500" y="120" text-anchor="middle" class="t b">① 击球决策</text>
<text x="500" y="142" text-anchor="middle" class="s">0.24 s 后，拍子以 pose X、速度 V 击球</text>
<rect x="300" y="178" width="400" height="52" rx="8" class="nu" stroke-width="1.3"/>
<text x="500" y="198" text-anchor="middle" class="t b">② 轨迹生成</text>
<text x="500" y="220" text-anchor="middle" class="s">拍子随时间怎么走</text>
<rect x="300" y="244" width="400" height="52" rx="8" class="nu" stroke-width="1.3"/>
<text x="500" y="264" text-anchor="middle" class="t b">③ 逆运动学（IK）</text>
<text x="500" y="286" text-anchor="middle" class="s">拍子位姿 → 7 个关节角</text>
<rect x="300" y="310" width="400" height="52" rx="8" class="nu" stroke-width="1.3"/>
<text x="500" y="330" text-anchor="middle" class="t b">④ 控制与跟踪</text>
<text x="500" y="352" text-anchor="middle" class="s">连续发关节目标，按反馈修正</text>
<rect x="300" y="390" width="400" height="52" rx="8" class="gr" stroke-width="1.3"/>
<text x="500" y="410" text-anchor="middle" class="t b">⑤ 关节伺服 + MuJoCo 物理</text>
<text x="500" y="432" text-anchor="middle" class="s">两种模式都一样；限位由共享的限位模块执行</text>
<path d="M500 72L500 90" class="ar" marker-end="url(#m43201)"/>
<path d="M500 158L500 176" class="ar" marker-end="url(#m43201)"/>
<path d="M500 230L500 242" class="ar" marker-end="url(#m43201)"/>
<path d="M500 296L500 308" class="ar" marker-end="url(#m43201)"/>
<path d="M500 362L500 388" class="ar" marker-end="url(#m43201)"/>
<path d="M290 92H272V158H290" fill="none" stroke="rgba(76,170,124,.9)" stroke-width="2"/>
<text x="262" y="116" text-anchor="end" class="t b cpl">高层 Athlete</text>
<text x="262" y="134" text-anchor="end" class="s">选手负责 ①</text>
<path d="M290 166H246V362H290" fill="none" stroke="rgba(76,170,124,.9)" stroke-width="2" stroke-dasharray="6 4"/>
<text x="236" y="254" text-anchor="end" class="t b cpl">底层 Engineer</text>
<text x="236" y="272" text-anchor="end" class="s">选手还负责 ②–④</text>
<text x="236" y="290" text-anchor="end" class="s">可用公开的 IK 库等工具</text>
<text x="262" y="152" text-anchor="end" class="s">②–④ 由平台的运动技能提供</text>
<path d="M702 350C760 350 760 200 704 200" class="ar" marker-end="url(#m43201)" stroke-dasharray="4 3"/>
<text x="772" y="262" class="s">底层模式另有</text>
<text x="772" y="280" class="s">调试循环：</text>
<text x="772" y="298" class="s">离线测、看失败、</text>
<text x="772" y="316" class="s">调参数</text>
</svg>

**图 10　控制链（以乒乓为例），以及两种模式下选手负责的部分。**

"打一个正手"太高层，难的判断都被做掉了，所以不提供。高层下，选手只负责 ①：什么时刻、以什么拍面姿态和速度击球。底层下，选手负责 ①–④：自己设计挥拍、发关节目标、离线测试和调参；IK 这类通用计算可以用工具里的库。⑤ 两档都一样。

#### 6.3.2 两档各测什么

| 能力 | 高层 | 底层 |
|---|---|---|
| 看：从相机画面估计球的位置、速度、旋转 | ✓ | ✓ |
| 预测：球何时、何处、以什么状态到达 | ✓ | ✓ |
| 竞技决策：击球时刻、拍面姿态和速度；走位 | ✓ | ✓ |
| 可行性判断：够不够得到、来不来得及 | ✓（系统会提示） | ✓（要自己判断） |
| 策略和适应：对手的习惯、比分局面、复盘后改打法 | ✓ | ✓ |
| 技能开发：轨迹规划、控制器设计、工具复用、集成与调试 | — | ✓ |

##### 和运动科学的对应：读比赛

运动科学里的比赛智能（game intelligence），又叫感知—认知能力（perceptual-cognitive skills；Williams & Ericsson 2005；Mann et al. 2007 的荟萃分析），我们叫"读比赛"：看懂场上发生了什么，预判接下来会怎样，据此选动作。它分成七块，常用 Endsley 的"感知 → 理解 → 预测"三层概括。定义、例子和测法见站点的《读比赛》页，这里只给和能力编号的对应。

| 读比赛的一块 | 对应能力 |
|---|---|
| 态势感知 | 从画面重建全场是 A1；看懂局面是 B2、B3 |
| 模式识别 | B2 |
| 预判（对手下一步做什么） | B2 |
| 读对手 | B2 |
| 读队友和空间 | B3 |
| 比赛管理 | B2 |
| 时间压力下的决策 | B2 |

**边界**：预测球的物理轨迹属于 A2，把球打到位属于 A3，都不算读比赛。读比赛针对人的行为和局面，对应 B2、B3。C1、C2 是把它写成程序、在复盘里修改的方式。

**两种模式对应两层。**高层只考感知—认知：动作由平台的技能 API 完成，选手决定看什么、预判什么、做什么。底层再加感知—动作：选手还要自己开发技能，把决定变成关节动作。6.3.2 开头的表里，前五行是感知—认知，最后一行"技能开发"是感知—动作。

**和传统 policy 的区别**

| | 传统 policy（强化学习、模仿学习、VLA） | 这里的选手（coding agent） |
|---|---|---|
| 能力怎么获得 | 赛前训练：大量试错或示范数据 | 不为这项运动训练：读说明书、看画面、写程序 |
| 存在哪里 | 网络权重 | 模型的通用知识 + 为这场比赛写的程序（可读、可改） |
| 比赛中怎么适应 | 权重固定；要适应就重新训练或微调 | 每分之后复盘，改程序或打法（C2） |
| 对手 | 训练环境里的对手或自我对弈；机器人策略的评测多是没有对手的固定任务 | 陌生的脚本对手或另一个模型，会有习惯，也会变 |
| 时间尺度 | 网络每个控制步都输出动作 | 两个频率：模型几秒想一次，程序每秒跑几十到几百次 |
| 评测看什么 | 任务成功率 | 比赛胜负，加上分项诊断（得分阶梯、读比赛诊断、底层的 API 测试） |

**限定**：这里只借用运动科学的分类框架来组织能力，不声称测出了与人可比的比赛智能。

文献：A. M. Williams, K. A. Ericsson (2005). Perceptual-cognitive expertise in sport: Some considerations when applying the expert performance approach. *Human Movement Science* 24(3), 283–307. [doi:10.1016/j.humov.2005.06.002](https://doi.org/10.1016/j.humov.2005.06.002)。D. T. Y. Mann, A. M. Williams, P. Ward, C. M. Janelle (2007). Perceptual-cognitive expertise in sport: A meta-analysis. *Journal of Sport & Exercise Psychology* 29(4), 457–478. [doi:10.1123/jsep.29.4.457](https://doi.org/10.1123/jsep.29.4.457)。

#### 6.3.3 哪些项目设哪一档

| 项目 | 高层 | 底层 | 理由 |
|---|---|---|---|
| 乒乓、网球、羽毛球 | ✓ | ✓ | 给了 IK 和行走，挥拍和击球仍要自己开发 |
| 足球 | ✓ | — | 给了行走策略以后，没有剩下要开发的核心运动技能，底层和高层差不多 |
| 短跑、跨栏 | — | ✓ | 给奔跑策略就等于把这项运动做完了；底层的参考解本身就是一项不小的工作 |
| 出手装置类（冰壶、保龄球）和机器类项目 | ✓ | 一般不设 | 除非底层还剩核心技能要开发 |
| 其他项目 | 按检验定 | 按检验定 | 用核心技能边界检验（6.3.7）决定 |

规则：设的每一档都要有可测的能力，并且有我们只用公开工具做出的参考解。

#### 6.3.4 机器人在两档里的角色

| | 高层 | 底层 |
|---|---|---|
| 机器人 | **固定**：每项运动一个最可靠的装配配置，所有 agent 相同，主榜只比竞技智能 | **研究变量**：先在同一个机器人上比 agent，再换机器人重复，看跨形态的泛化 |
| 为什么 | 换机器人会带进和 agent 无关的因素：同一个高层规划器，参考选手在 RB-Y1 上 24/25，在 G1 上 0/14 | 不同机器人的工程问题不同（Franka：IK、轨迹、击球速度；RB-Y1：底盘和手臂协调；G1：接入行走策略、平衡、挥拍；H1-2：适应不同的运动学和执行器），这正是要测的 |
| 怎么选、按什么顺序 | 过 B9、参考选手胜率高、已实现 B0、许可清楚、画面清楚 | Franka → RB-Y1 → G1；H1-2、R1 Pro 以后 |
| 怎么报告 | 换机器人只作可选扩展，单独报告，不进主榜 | 按机器人分别报告，不同机器人的成功率不直接比较 |

身体意识两档都测：高层的机器人虽然固定，agent 仍要从规格里理解它的能力（底盘速度、够得到的范围）。研究问题：**"Can coding agents not only compete as athletes using predefined motor skills, but also act as robotics engineers who develop those skills across different robot embodiments?"**

#### 6.3.5 底层给什么、不给什么

- **统一控制权，不统一控制器**：关节接口始终可用；现成的基础策略（如行走）可用、可改、可替换，也可以不用。
- **不同机器人给不同的基础工具**，但要预先公开、固定，同一项运动的所有 agent 一致。
- **不给项目的核心技能**：例如短跑不给奔跑策略，乒乓不给挥拍。
- **给 IK 库**（可选工具），前提是核心技能边界检验通过（6.3.7）。
- **工具清单属于运动**：每项运动为每个装配配置写一份公开清单（策略和权重的来源、许可证、sha256，IK 库版本，接口和频率，禁止的技能），它是任务版本的一部分。

| 机器人 · 项目 | 高层给的运动技能 | 底层的工具 | 底层要 agent 自己开发 |
|---|---|---|---|
| 固定机械臂 · 乒乓 | hit（平台规划挥拍、解 IK、跟踪） | 模型 + 关节接口 + IK 库 | 挥拍轨迹、击球时机、拍速和拍面 |
| 轮式人形 RB-Y1 / R1 Pro · 乒乓 | hit（全身规划器协调底盘、升降、腰、手臂） | 模型 + 关节接口 + 底盘速度接口 + IK 库 | 站位、底盘和手臂的协调、挥拍 |
| 腿式人形 G1 · 乒乓 | 行走 + hit | 模型 + 关节接口 + 官方行走策略 + IK 库 | 挥拍，以及上半身和平衡的协调 |
| 腿式人形 H1-2 · 乒乓 | 行走 + hit | 同 G1 | 同 G1，适应不同的运动学和执行器 |
| 固定双臂 Marvin Pro · 飞镖、射箭等 | 投放技能（按项目） | 模型 + 双臂关节接口 + IK 库 | 挥臂轨迹、出手时机和速度 |

<svg viewBox="0 0 920 232" width="100%" role="img" aria-label="底层模式给什么、自己开发什么、不给什么" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.h{fill:var(--fg,#16211c);font-size:15px;font-weight:600}.s{fill:var(--muted,#56665e);font-size:12px}.m{fill:var(--fg,#16211c);font-size:11.5px;font-family:ui-monospace,Menlo,monospace}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.fpl{fill:rgba(76,170,124,.06)}.fsp{fill:rgba(236,146,82,.06)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.ln{stroke:var(--muted,#56665e);stroke-width:1.2;fill:none}</style>
<defs><marker id="m4769" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<rect x="20" y="20" width="280" height="170" rx="8" class="pl" stroke-width="1.3"/>
<text x="160" y="56" text-anchor="middle" class="t b">① 工具包（清单公开）</text>
<text x="160" y="80" text-anchor="middle" class="s">机器人自身的模型（含真实限位）</text>
<text x="160" y="99" text-anchor="middle" class="s">关节接口：目标角度 / 速度 / 力矩</text>
<text x="160" y="118" text-anchor="middle" class="s">IK 库；按机器人的基础策略（如行走）</text>
<text x="160" y="137" text-anchor="middle" class="s">MuJoCo、NumPy、SciPy、练习模拟器</text>
<text x="160" y="156" text-anchor="middle" class="s">相当于厂家给的说明书和 SDK</text>
<rect x="320" y="20" width="280" height="170" rx="8" class="nu" stroke-width="1.3"/>
<text x="460" y="56" text-anchor="middle" class="t b">② 选手自己开发</text>
<text x="460" y="80" text-anchor="middle" class="s">项目的核心技能（挥拍、出手）</text>
<text x="460" y="99" text-anchor="middle" class="s">轨迹和时机</text>
<text x="460" y="118" text-anchor="middle" class="s">控制器、跟踪和调参</text>
<text x="460" y="137" text-anchor="middle" class="s">工具和自己程序的集成、调试</text>
<text x="460" y="156" text-anchor="middle" class="s">（可在 CPU 上自己训练）</text>
<rect x="620" y="20" width="280" height="170" rx="8" class="gr" stroke-width="1.3"/>
<text x="760" y="56" text-anchor="middle" class="t b">③ 不给</text>
<text x="760" y="156" text-anchor="middle" class="s">做好的运动技能（hit、kick）</text>
<text x="760" y="80" text-anchor="middle" class="s">球的精确位置和速度</text>
<text x="760" y="99" text-anchor="middle" class="s">对手的内部状态</text>
<text x="760" y="118" text-anchor="middle" class="s">球未来的轨迹</text>
<text x="760" y="137" text-anchor="middle" class="s">比赛模拟器的完整状态</text>
<text x="20" y="216" class="s">选手容器里只有机器人自己的模型和工具包（没有球、球台、对手）；比赛模拟器在另一个容器里，只能通过 observe 和 act 交互。</text>
</svg>

**图 11　底层模式里，选手拿到什么、要自己开发什么、拿不到什么。**

#### 6.3.6 实现要点

| 要点 | 做法 |
|---|---|
| 控制权仲裁 | 运动的动作层按关节组（腿、腰、手臂、底盘）仲裁，可用资源库的共享仲裁代码。同一组同一时刻只有一个控制源；交接平滑、守限位；行走策略收回时重新初始化，确认站稳才算交回；每步记下控制源。agent 自己控制过以后，行走策略可能处在训练没见过的状态而失效，风险由 agent 承担 |
| 高频控制 | agent 提交一段未来的关节目标计划（每行一个控制步），共享伺服模块的 PD 按时间表执行；新计划替换旧计划剩下的部分。把 agent 的代码放进模拟器运行是以后的阶段 |
| 隐藏状态 | 模拟器里的基础策略只读机器人自己的本体，不读球和对手 |
| 确定性 | 接口以下（伺服、仲裁、行走策略）在 numpy 里确定性运行，重放逐位一致；agent 自己训练的控制器只在选手容器里跑（只用 CPU，时间算进会话） |

#### 6.3.7 怎么评和排名

底层的任务是**实现我们高层用的技能 API**（例：乒乓的 `hit(t, pos, normal, vel)`），满足同一份接口规格。

**底层的流程**（2026-10-08 定）：
1. **比赛照常**：agent 用自己的策略和自己实现的技能 API 打一场正常的比赛，流程和高层一样。
2. **每分可改**：每分之后的复盘里，agent 可以同时改进策略和技能实现。
3. **赛后取最终实现单独评测**：比赛结束后，平台从固定的交付路径（例：`/app/output/skill/hit.py`）取 agent 最终的技能实现，在另一个隔离容器里跑 API 测试和比赛测试。隐藏请求只在赛后使用。交付约定见第 9 节机器人与模式规则第 10 条。
4. **三个数**：一场底层比赛得到三个数：比赛结果（端到端）、API 测试、比赛测试。

| 项 | 内容 |
|---|---|
| 比赛（两档） | 单人对脚本对手、对战。底层用 agent 自己的策略和自己的实现，得到端到端的比赛结果 |
| API 测试（底层，赛后） | 击球请求从真实的参考比赛里采样，按真实限位筛掉做不到的，作为隐藏请求，只在赛后使用。报：触球时刻、位置、拍面朝向、速度的误差，球是否回到对方台面，超限次数，摔倒次数；另报开发成本（token、时间、迭代次数、训练 CPU 时间） |
| 比赛测试（底层，赛后） | 我们固定的高层参考选手装在 agent 的最终实现上，打 L1–L4；和"参考选手 + 我们的实现"对比，比的只是实现的好坏 |
| 核心技能边界检验 | 只用工具、不开发技能的最简实现（如用 IK 把拍子挡在球路上），API 测试应 ≈ 0。如果得分，就收紧成功条件（目标区域、出球速度），不拿掉 IK。我们只用工具做出的参考实现必须能过 |
| 排名 | 按"运动 × 装配配置 × 模式"分开排；高层主榜每项运动一个机器人；底层的三个数（比赛结果、API 测试、比赛测试）分别报告，API 测试单列；B9 没过的组合只作挑战组 |

扩展（可选）：**全栈**变体，agent 自己设计接口并直接比赛，单独报告。

**实现**：两档的共同部分在共享库 [`skilleval/`](../skilleval/README.md)（和 `bodies/` 一样不是框架的第五部分）：技能合同的基类、工具包清单的格式和 sha256 检查、隐藏请求和公开练习集、在单独进程里运行选手技能的沙箱，以及赛后评测器（`tools/skill_eval.py` 起隔离容器，跑 API 测试和比赛测试）。每项运动在 `sport.toml` 的 `[modes.high]` / `[modes.low]` 里声明，`tools/check` 第 1 步查声明、第 6 步跑平台实现、参考解、边界基线和隔离探针。怎样给一项运动加两档见 CONTRIBUTING.md「如何给一项运动加两档」，最小的例子是 `sports/hello`。

#### 6.3.8 常见问题

| 问题 | 回答 |
|---|---|
| 底层必须自己写 IK、全身控制或强化学习吗 | 不必须，技术路线 agent 自己选 |
| 底层必须用行走策略吗 | 不必须；可以用、改或替换（固定预算、限位内） |
| 能自己训练控制器吗 | 能，只用 CPU，时间算进会话 |
| 底层也比赛吗 | 比，比赛照常：agent 用自己的策略和自己的实现打，每分之后可以改两者。赛后另用我们固定的高层参考选手驱动 agent 的最终实现（比赛测试），这一项比的只是实现的好坏；agent 自己设计接口并比赛是可选的全栈扩展 |
| 比赛中改了实现，评哪一版 | 比赛结束时固定路径上的最终版。建议每次复盘存一份快照，用来分析技能在比赛中怎么变好 |
| 比赛中能看到 API 测试的请求吗 | 不能，隐藏请求只在赛后使用 |
| 以前的设计怎么处理 | 单独的"电机层"并入底层的关节接口；乒乓底层改为提供 IK 库，之前的底层比赛按演示处理、重跑 |

**注**：
- 高层接口保持 `hit(t, pos, normal, vel)`（某时刻拍面的位置、朝向和速度），它考预测、物理推理和精确的击球规划。更抽象的 `hit(落点, 旋转, 时机)` 会把最难的击球计算交给平台，只记为可能的变体，不采用。
- 我们的全身规划器 `bodies/wholebody.py` 是我们自己的 coding agent 在搭平台时写的（不限时、反复迭代、有人的反馈），是"coding agent 能做这类工程"的动机例子，**不是基准结果**。它是运动学规划器（选站位、IK、速度分配、挥拍、预演），不是基于动力学的全身控制；腿的平衡来自 Unitree 的行走策略。

### 6.4 保密官方种子和连续分数（核心 0.2.14，2026-10-11）

对 21 项运动的审核发现两个共同问题：官方种子公开且固定，最优动作可以记住；单人项目的二值成功线贴近能力上界，强弱选手都能过线。所以正式评测用**保密的种子集**，结果另给一个**连续分数**。两件事都由核心统一做，各运动只做声明，不各自实现。

#### 6.4.1 保密官方种子

**做法**：承诺—揭示（commit–reveal）。冻结时生成种子集，仓库里只放承诺；比赛时核心把一个种子交给运动；验证器按登记的 sha256 认可。**种子集按"形式 × 等级"分格**：每个有等级的单人形式每级一格（`match/L1`、`match/L2`……），其余每个形式一格（`duel`、`team`……）；每格有自己的种子，格与格之间不共用。

| 东西 | 内容 | 放在哪里 | 谁能看到 |
|---|---|---|---|
| 私密种子集 | 每格一组 `{"seed", "salt"}`（6 位种子 + 128 位随机盐；格式 `ro-hidden-seeds-private/2`） | 仓库外，只在正式比赛的主机（mll-4090-1）的受限目录里（`RO_SEEDS_DIR`，目录 0700、文件 0600），由用户保管并备份；`.gitignore` 挡住 `*.hidden_seeds.json`，CI 发现私密格式的文件就报错 | 用户（正式比赛的主机） |
| 承诺 | 每个种子一个：`sha256("ro-hidden-seeds/1\|<运动>\|<种子>\|<盐>")`，按格、按顺序 | 公开：`sports/<运动>/seed_commitments.json` | 所有人；有盐，推不出种子 |
| 登记 | 所有格的承诺的 sha256，每格的种子个数 | 公开：`sport.toml` 的 `[official.hidden_seeds]`（`sha256`、`counts = { "match/L1" = 20, …, duel = 20 }`） | 所有人 |
| 一场比赛的种子 | 这场所在格的第 I 个种子和它的盐 | 运行脚本在本机解析 `--seed hidden:I`，只通过环境变量交给比赛容器（sim）：`RCB_SEED`/`DUEL_SEED`、`RCB_SEED_SALT`、`RCB_SEED_INDEX` | 只有 sim 容器；**任何记录里都没有** |

**流程**：
1. **冻结时**（运动负责人在 mll-4090-1 上，或把生成交给用户）：`RO_SEEDS_DIR=<受限目录> python3 tools/seeds.py new <运动> --count N`（每格 N 个；`--counts match/L3=40` 单独改某格）。它生成私密种子集（避开公开的开发种子）、写出公开的承诺文件、把 `sha256` 和 `counts` 写进 `sport.toml`；然后重新生成任务。`tools/seeds.py check <运动>` 核对三者一致。
2. **运行时**：赛季脚本（`plan.py`、`run_round.py`）对登记了种子集的运动按格排 `hidden:0` … `hidden:N-1`，第 r 轮用这场所在格的 `hidden:r-1`；`run_single.sh` 等脚本用 `tools/seeds.py resolve <任务目录> I [--level L]` 在本机取出种子和盐（先核对私密集就是登记的那一份）。作业名里只写 `hI`。核心照常把 `cfg.seed` 交给运动的 `build(cfg)`，运动不知道种子是不是保密的。
3. **不写种子**：这样的比赛里核心在任何输出里都不写种子和盐：比赛日志的表头、`final.json`、`replay_check.json`、直播日志里种子写成 -1，只记承诺（`final.json` 的 `seed_proof`：`commitment` 和 `index`）；运动自己的字段（`final_fields`、直播帧）和给选手的每个回复（`/spec`、观测、动作回复）都经过核心的清洗：键名是 `seed` 或 `*_seed` 的值、等于种子的整数和字符串、等于盐的字符串，都换成 -1。盐也不写：有了盐，6 位种子可以从承诺暴力反推。新进程重放从同一组环境变量取种子，并核对它给出的承诺。种子的唯一记录是私密种子集加上格和序号。
4. **验证**：验证器（`tests/grade.py` = `core/grading.py`，加上 `tests/seeds.py` = `core/seeds.py`）先按登记的 sha256 和个数核对 `contract.json` 里的承诺，再在这场比赛的形式和等级所在的格里找 `seed_proof` 的承诺（重放给出的承诺必须相同）。找到了：`hidden_seed = 1`、`hidden_seed_index = 序号`；否则 0 和 -1。别的格的种子不算。登记了种子集的运动，`official_config = 1` 要求 `hidden_seed = 1`；公开的 `[official] seeds` 只用于开发和检查，合作者开发时用它们。`reward.json` 的 `seed` 在保密种子的比赛里是 -1。
5. **离线重放**（重渲染、复查）：只能在有私密集的主机上做：`read RO_SEED RO_SEED_SALT < <(python3 tools/seeds.py resolve <任务目录> I [--level L])`，再照常运行 `python -m core.episode`；没有这两个变量或对不上承诺时，重放拒绝运行。
6. **赛季结束后**：可以公开私密文件；任何人都能用 `tools/seeds.py check <运动> --private 文件` 重算出登记的 sha256。

**为什么这样设计**：
- 只公开整个集合的 sha256 时，验证器要确认"这个种子在集合里"就得拿到整个集合，而验证器的输入（任务的 `tests/`、作业目录）都不保密。逐个加盐承诺后，验证器只需要这场比赛的承诺，记录里连这一场的种子也没有。
- 盐有 128 位，承诺公开也推不出种子；盐不进任何记录，所以承诺也没法被暴力反推。
- 按格分开：每一级对手、每种形式各有自己的种子，结果可以按格独立统计，一格的种子用完不影响别格。
- 种子只走比赛容器的环境变量：现有的隔离（选手和 sim 分容器、选手网络只通 sim 的 HTTP）已经保证选手看不到，不需要新的挂载。
- 运动代码不变：核心在运动外面完成交接、记录和清洗，"随机数只从核心按名字取"这条规则不受影响。通用测试 `seed_private` 会指出运动自己在哪里显示了种子（只作提示：保密比赛里核心会清洗）。

#### 6.4.2 连续分数

**原则（所有运动相同）**：每场比赛给每一方一个数，**越大越好，方向和胜负一致**（赢的一方通常数也更大）；判负（没到场、中途结束、超时、退出）的一方得一个**固定的 `forfeit` 值**，取定义允许的最差值。在这个原则下，每项运动的负责人从 `share`、`margin`、`own`、`hook` 里选一种（下面的建议表），写进 `sport.toml` 的 `[score]`：

```toml
[score]
name = "point_share"                  # 标识（小写字母、数字、下划线）
text = "points won / points played"   # 一句话定义（README、网站）
from = "share"                        # share | margin | own | hook
forfeit = 0.0                         # 判负的一方得这个值：定义允许的最差值
range = [0, 1]                        # 可选：取值范围，通用测试会检查
```

| `from` | 怎么算 | 适合 |
|---|---|---|
| `share` | 己方得分 / 各方得分之和（`scores()`；都为 0 时 0.5） | `scores()` 记的是分数的对抗项目 |
| `margin` | 己方得分 − 其他方的最高得分 | 进球类 |
| `own` | 己方得分 | 对着成功线、越高越好的单人项目 |
| `hook` | 运动实现 `Match.continuous_score() -> {方: 数}` | `scores()` 给不出的量，或者方向是越小越好的计数 |

**各运动的建议**（负责人从中选定，写进自己的 `sport.toml`）：

| 运动 | `scores()` 是什么 | 建议 |
|---|---|---|
| 乒乓 | 一局的分数（单人含 `opponent`） | `share`（得分占比），`forfeit = 0`，`range = [0, 1]` |
| 羽毛球 | 分数 | `share` |
| 网球 | 分数 | `share`（如果 `scores()` 记的是局或盘，改用 `hook` 记分数） |
| 足球 | 进球 | `margin`（净胜球），`forfeit` 取固定下限（例如规则书的弃权比分 0:3 → −3） |
| 飞镖 | **局数** | `hook`：平均每镖得分，`forfeit = 0` |
| 空气曲棍球、桌上足球、击剑、冰壶 | 分数 / 进球 | `share` 或 `margin` |
| 台球 | **剩余球数（越少越好）** | `hook`（例如己方已进球数占比）；不能用 `share` |
| 射箭、篮球 | 总环数 / 命中数（单人） | `own`（或 `hook` 归一化到每箭、每投） |
| 高尔夫推杆、飞盘高尔夫、保龄球、棒球单人、无人机、赛车 | 杆数等，方向不一定是越大越好 | `hook`，保证越大越好（例如负杆数） |
| hello（示例） | 总分 | `hook`：平均每球得分（0–10） |

- **核心做的**：比赛结束时按声明算出每一方的值（判负或退出的一方取 `forfeit`，防止"领先时放弃"保住好看的数），写进 `final.json` 的 `continuous`（`name`、`from`、`higher_is_better: true`、`by_side`、`forfeited`）；新进程重放时再算一次，验证器要求两次一致；`reward.json` 里每个座位一个 `<座位>_continuous_score`，单方形式（单人、对脚本对手）另有 `continuous_score`。值不是有限数（运动的错误）算技术故障。
- **没有声明的运动**：没有这些字段；`tools/check` 第 1 步给警告。**登记了保密种子（已冻结）的运动必须声明**，否则第 1 步不通过。
- **检查**：`tools/check` 第 1 步核对声明（`from = "hook"` 当且仅当插件实现了 `continuous_score`）；通用测试 `continuous_score` 检查所有测试比赛的值是有限数、在 `range` 内、不低于 `forfeit`，以及没到场的一方得到 `forfeit`。

## 7. 工具

<svg viewBox="0 0 920 282" width="100%" role="img" aria-label="从运动规格到赛季报表" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.h{fill:var(--fg,#16211c);font-size:15px;font-weight:600}.s{fill:var(--muted,#56665e);font-size:12px}.m{fill:var(--fg,#16211c);font-size:11.5px;font-family:ui-monospace,Menlo,monospace}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.fpl{fill:rgba(76,170,124,.06)}.fsp{fill:rgba(236,146,82,.06)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.ln{stroke:var(--muted,#56665e);stroke-width:1.2;fill:none}</style>
<defs><marker id="m80835" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<rect x="20" y="30" width="160" height="66" rx="8" class="sp" stroke-width="1.3"/>
<text x="100" y="58" text-anchor="middle" class="t b">运动规格</text>
<text x="100" y="80" text-anchor="middle" class="s">sport.toml</text>
<path d="M182 63L198 63" class="ar" marker-end="url(#m80835)"/>
<rect x="200" y="30" width="160" height="66" rx="8" class="to" stroke-width="1.3"/>
<text x="280" y="58" text-anchor="middle" class="t b">生成与检查</text>
<text x="280" y="80" text-anchor="middle" class="s">展开成任务</text>
<path d="M362 63L378 63" class="ar" marker-end="url(#m80835)"/>
<rect x="380" y="30" width="160" height="66" rx="8" class="nu" stroke-width="1.3"/>
<text x="460" y="58" text-anchor="middle" class="t b">任务目录</text>
<text x="460" y="80" text-anchor="middle" class="s">自包含，可直接跑</text>
<path d="M542 63L558 63" class="ar" marker-end="url(#m80835)"/>
<rect x="560" y="30" width="160" height="66" rx="8" class="to" stroke-width="1.3"/>
<text x="640" y="58" text-anchor="middle" class="t b">运行与调度</text>
<text x="640" y="80" text-anchor="middle" class="s">任务 × 选手 → 比赛</text>
<path d="M722 63L738 63" class="ar" marker-end="url(#m80835)"/>
<rect x="740" y="30" width="160" height="66" rx="8" class="co" stroke-width="1.3"/>
<text x="820" y="58" text-anchor="middle" class="t b">比赛</text>
<text x="820" y="80" text-anchor="middle" class="s">核心 + 运动</text>
<rect x="380" y="130" width="160" height="122" rx="8" class="nu" stroke-width="1.3"/>
<text x="460" y="157" text-anchor="middle" class="t b">配置（代码之外）</text>
<text x="460" y="179" text-anchor="middle" class="s">赛季配置</text>
<text x="460" y="198" text-anchor="middle" class="s">选手清单</text>
<text x="460" y="217" text-anchor="middle" class="s">账户（仓库外）</text>
<text x="460" y="236" text-anchor="middle" class="s">本机路径（不入库）</text>
<path d="M542 191H640V100" class="ar" marker-end="url(#m80835)" stroke-dasharray="4 3"/>
<text x="652" y="160" class="s">运行时读取</text>
<path d="M820 98L820 128" class="ar" marker-end="url(#m80835)"/>
<rect x="740" y="130" width="160" height="56" rx="8" class="co" stroke-width="1.3"/>
<text x="820" y="152" text-anchor="middle" class="t b">比赛记录</text>
<text x="820" y="174" text-anchor="middle" class="s">日志 · 结果 · 版本号</text>
<path d="M820 188L820 212" class="ar" marker-end="url(#m80835)"/>
<path d="M770 188C770 202 640 200 640 212" class="ar" marker-end="url(#m80835)"/>
<rect x="560" y="214" width="160" height="54" rx="8" class="to" stroke-width="1.3"/>
<text x="640" y="236" text-anchor="middle" class="t b">观看</text>
<text x="640" y="258" text-anchor="middle" class="s">直播 · 重渲染</text>
<rect x="740" y="214" width="160" height="54" rx="8" class="to" stroke-width="1.3"/>
<text x="820" y="236" text-anchor="middle" class="t b">分析</text>
<text x="820" y="258" text-anchor="middle" class="s">赛季表 · 指标</text>
</svg>

**图 12　从运动规格到赛季报表。**

生成器把每项运动的规格展开成任务目录；运行工具读取配置（赛季配置、选手清单、账户、本机路径），把任务和选手配成比赛，按账户额度调度运行；每场比赛产出比赛记录；观看和分析工具读这些记录，做直播、重渲染、赛季表和各项指标。配置都放在代码之外，换机器、换账户不用改代码。

| 工具 | 做什么 | 读 | 写 |
|---|---|---|---|
| 生成与检查 | 把运动规格展开成任务目录；`tools/check <运动>` 一条命令跑完所有检查 | `sport.toml`、运动代码、核心 | `tasks/<id>/`（只有生成器能写） |
| 运行与调度 | 跑一场或一轮赛季；两种先手、缺失补跑；按账户额度调度 | 任务、选手清单、赛季配置、账户 | 作业目录、结果 |
| 观看 | 直播（自动导播、慢放、字幕）、赛后重渲染（渲染器任选，见 3.1 节）、实时看板 | 直播日志、比赛日志 | 视频、网页 |
| 分析 | 赛季表和 Elo、座位偏差、得分阶梯、思考重叠、适应性、能力曲线、用量 | 结果、比赛日志、模型调用记录 | 报表、网站数据 |

## 8. 怎样加一项运动

<svg viewBox="0 0 920 186" width="100%" role="img" aria-label="加一项运动的流程" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.h{fill:var(--fg,#16211c);font-size:15px;font-weight:600}.s{fill:var(--muted,#56665e);font-size:12px}.m{fill:var(--fg,#16211c);font-size:11.5px;font-family:ui-monospace,Menlo,monospace}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.fpl{fill:rgba(76,170,124,.06)}.fsp{fill:rgba(236,146,82,.06)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.ln{stroke:var(--muted,#56665e);stroke-width:1.2;fill:none}</style>
<defs><marker id="m20221" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<rect x="20" y="20" width="164" height="150" rx="8" class="sp" stroke-width="1.3"/>
<text x="102" y="80" text-anchor="middle" class="t b">① 复制示例运动</text>
<text x="102" y="102" text-anchor="middle" class="s">从仓库里最小的</text>
<text x="102" y="121" text-anchor="middle" class="s">示例运动开始</text>
<path d="M186 95L198 95" class="ar" marker-end="url(#m20221)"/>
<rect x="200" y="20" width="164" height="150" rx="8" class="sp" stroke-width="1.3"/>
<text x="282" y="61" text-anchor="middle" class="t b">② 写 5 样必需</text>
<text x="282" y="83" text-anchor="middle" class="s">规格 sport.toml</text>
<text x="282" y="102" text-anchor="middle" class="s">世界 + 裁判</text>
<text x="282" y="121" text-anchor="middle" class="s">底层动作 · 观测</text>
<text x="282" y="140" text-anchor="middle" class="s">参考选手</text>
<path d="M366 95L378 95" class="ar" marker-end="url(#m20221)"/>
<rect x="380" y="20" width="164" height="150" rx="8" class="sp" stroke-width="1.3"/>
<text x="462" y="61" text-anchor="middle" class="t b">③ 按需加可选</text>
<text x="462" y="83" text-anchor="middle" class="s">高层插件</text>
<text x="462" y="102" text-anchor="middle" class="s">脚本对手（有单人赛）</text>
<text x="462" y="121" text-anchor="middle" class="s">工具包清单（有底层）</text>
<text x="462" y="140" text-anchor="middle" class="s">直播适配</text>
<path d="M546 95L558 95" class="ar" marker-end="url(#m20221)"/>
<rect x="560" y="20" width="164" height="150" rx="8" class="to" stroke-width="1.3"/>
<text x="642" y="70" text-anchor="middle" class="t b">④ tools/check</text>
<text x="642" y="92" text-anchor="middle" class="s">规格 → 通用测试</text>
<text x="642" y="112" text-anchor="middle" class="s">→ 标准回放</text>
<text x="642" y="130" text-anchor="middle" class="s">→ 公平检查 → 实跑</text>
<path d="M726 95L738 95" class="ar" marker-end="url(#m20221)"/>
<rect x="740" y="20" width="164" height="150" rx="8" class="nu" stroke-width="1.3"/>
<text x="822" y="80" text-anchor="middle" class="t b">⑤ 提交</text>
<text x="822" y="102" text-anchor="middle" class="s">只改</text>
<text x="822" y="121" text-anchor="middle" class="s">sports/&lt;运动&gt;/</text>
</svg>

**图 13　加一项运动的五步。**

一项运动必需的只有五样：规格文件、世界和裁判、底层动作、观测、参考选手。其余按需要添加：有高层模式时写高层插件，有单人赛时写脚本对手，提供底层模式时写好工具包清单（6.3 节），并附上只用清单里的工具做出的参考解（两档的声明、合同、清单和赛后评测都用共享库 `skilleval/`，步骤见 CONTRIBUTING.md「如何给一项运动加两档」）。写完后运行 `tools/check <运动>`，它依次做规格检查、通用测试、标准回放、公平检查和实跑，全部通过就可以提交。仓库里有一个最小的示例运动 `sports/hello/`，新人可以复制它开始写。

**动手之前先过五道门槛**（写进 PR 说明）：
1. **物理可信**：引擎能真实模拟这项运动（游泳、花滑在这一步被排除）；
2. **模型有判断可做**：成绩不能只由底层控制器决定；
3. **只靠相机能玩**；
4. **有官方规则书**，裁判能按规则判；
5. **参考选手能赢，对战两边公平**；每个提供的"运动 × 装配配置 × 模式"都有只用公开工具做出的参考解（9 节机器人与模式规则）。

PR 里还要写明它带来了什么：一种新物理、一项覆盖不足的能力，或者替换哪一项表演赛项目。

**能力声明**：`sport.toml` 里的 `[capabilities]` 一节给出这项运动考察的七项能力（A1 3D 空间理解、A2 物理动力学理解、A3 具身与执行、A4 时间与延迟、B1 规则、B2 策略与读对手、B3 团队协作，各标主要 / 次要 / 不涉及；C1 实时编程和 C2 在比赛中学习每项运动都考，不用声明），以及"考的物理"（从固定词表里选，例如弹道与风、旋转与弹跳、摩擦与滚动、碰撞）。`tools/check` 检查它必须填；网站上的"能力 × 运动"矩阵和每个模型的能力画像都从这里生成。能力的定义见《能力体系》。评测设置（模式开关、减速倍数）不引入新能力。

**五条贡献路径，每种改动只碰一个地方：**

| 想做的事 | 改哪里 | 不用碰 | 自动检查 |
|---|---|---|---|
| 加一项运动 | `sports/<新运动>/` | 核心、选手、工具 | `tools/check <运动>` |
| 接入一个新模型或 harness | `players/`（适配 + 清单里一行） | 运动、核心 | 冒烟比赛 + 不写代码的红队测试 |
| 开一个赛季 | `seasons/<名字>.toml` | 所有代码 | 版本冻结检查 |
| 加一种观看方式或报表 | `tools/` 下对应目录 | 比赛相关的一切 | 用已有记录跑通 |
| 改比赛规则（协议、形式、有效性） | `core/`（单独提交，升版本） | 各项运动 | 所有运动的标准回放 |

这套结构能装下的新场景包括：计分单人项目（田径投掷）、团队项目（沙排、冰球）、多方同场（短跑、多车赛）、复杂轮次（冰壶、台球、棒球）、只有底层的项目（人形短跑）、换用别的物理引擎（只要能逐步推进、结果可复现）、新的 harness、人类和模型对打，以及"读比赛"的球探报告。团队项目和多方同场需要先给核心补两个接口（见第 10 节）。

## 9. 规则

规则按"在哪里检查"分组。旧的字母编号保留为别名，以前的引用继续有效。

**运动规则**（由 `tools/check <运动>` 检查）
1. 观测只给相机画面、自己的身体和裁判信息；字段写成白名单，出现白名单以外的东西就报错。动作回复里的结果图像（核心 0.2.8，`reply_views`）和观测图像同样对待：同一个选手渲染器、只用比赛场景（展示场景不进选手的图）、同一帧时间预算，只拍这个座位在这一刻本来就有权看到的东西；画图时间不算选手的时间。
2. 裁判只根据物理事件判罚，不能直接改物理状态来制造结果。
3. 高层动作必须降成底层命令；比赛日志里只记底层命令。
4. 随机数只从核心按名字取；状态哈希覆盖所有影响比分的东西。
5. 决胜顺序写在 `sport.toml` 里，一定分出胜负；不来自规则书的兜底规则要写进提示词。
6. 所有座位、各方都公平：镜像对称测试，加偏差检查（用内置选手两种先手各打 N 场，算出因位置、先手或出发顺序获得的胜率优势及其 95% 置信区间：整个区间都落在 ±5 个百分点以外才判不公平；区间比 ±5 个百分点还宽说明场数不够：回合制项目判不通过（打得快，必须打够场数），实时项目给出警告；结果写明用了多少场）；同一方的座位拿到的观测和动作权限相同，角色差别只能来自规则书。
7. 规则有出处：写明规则书和版本，每条偏离都有编号和条文号；影响胜负的偏离写进提示词。
8. 漏洞清单里的每一条都有测试，例如"拖延不能得利""越界命令被拒绝"。
9. 参考选手只看相机，每种格式都能把比赛正常打完（端到端，报告合法回球率等，不设数值门槛）；对脚本对手的格式（以及带等级的单方格式，如棒球击球、无人机竞速）另做对手标定：真值感知的参考选手赢最低一级、胜率随等级下降，并且有"只看画面也能赢"的证据（相机参考选手赢下最低一级的多数，或第 ⑤ 步模型只看相机赢过）（2026-10-11，`docs/screening/README.md`"Reference check"）；什么都不做的选手（nop）得 0 分。
10. 只改自己的目录；需要共用的物理工具先单独提升到工具箱。
11. 执行散布（对战）：散布保留，它代表真机的离散。同一场里两个选手的散布序列互相独立；一对对战（交换先手、同一种子）的两场交换这两条序列，所以一对比赛下来两边的运气相同。风这类环境条件仍然共用。原因：每个动作共用同一个散布时，后出手的一方能读出对手的偏差并提前补偿（飞镖上实测到过）。

**选手规则**（接入时检查）
1. 选手不认识任何运动，也碰不到比赛内部状态：选手容器隔离，比赛中没有外网。
2. 不写代码时必须有结构上的保证：Claude Code 用工具白名单，Codex 由网关在执行前拦截不合规的调用。拦截只返回一句固定提示，不算违规；如果真的执行了不该执行的东西，赛后审计判负。审计从选手开赛（`game start`）时开始。
3. 选手配置（模型、harness、版本、玩法、禁用的工具）只定义一次，带版本号。
4. 账户和密钥不进仓库，仓库里只有变量名。

**核心规则**（修改核心时检查）
1. 核心里不出现任何运动名或 harness 名。
2. 所有选手用同一个时钟，一次只处理一个请求。
3. 技术故障作废重打，选手自己的问题算有效的输；API 临时错误每个座位最多容忍 5 次。
4. 修改核心要单独提交、升版本，并且所有运动的标准回放都要通过。
5. 开赛时限：所有座位都 `/start`（实时项目也可以是第一次 `/ready`）后才开赛；第一个座位开赛后，其余座位有 `START_TIMEOUT_S` 的时间。两方及以上的比赛里，时限到了还没开赛的一方判负（`no_show`），比赛有效，到场的一方获胜；一方只要有一个座位没开赛，就算没到场；多于两方时，没到场的一方退出，其余各方接着比（运动不允许退出时比赛到此结束）。所有方都没到场，比赛作废（技术故障，无人获胜）。单人形式没有对手可等，时限从比赛服务启动时算起；时限到了还没开赛，算有效的输（`no_show`，success = 0）。时限没到、还没开赛会话就结束了（验证器来收比赛时这一方还没开赛），和时限到了一样处理，不再等时限：单人形式算有效的输（`no_show`，success = 0）；两方及以上的比赛里，收比赛时还没开赛的每一方都判负（`no_show`），比赛有效，两方时到场的一方获胜；所有方都没开赛的，比赛仍不算数。（2026-10-06 起，核心 0.2.2；收比赛时按没到场处理自 2026-10-07 起，核心 0.2.7；还没迁移的旧项目沿用原来的规则，迁移后统一。）
6. 中途结束会话：单人形式（只有一方：独自计分，或对手是脚本）开赛以后、比赛结束以前，选手的会话结束了（验证器来收比赛时比赛还没完），算有效的输（`abandoned`，success = 0，`reward.json` 里 `abandoned` = 1）。实时项目和回合制项目一样，回合制也不管收比赛时是不是轮到选手（比如正在等复盘、或者运动暂时不让任何一方行动）。提前结束会话（选手以为已经打完、放弃、到了 agent 时限、上下文用完）是选手自己的问题，正是我们要测的东西；旧任务也一直算有效的输。API 和基础设施故障不归这条管：Harbor 把这类试次记为 agent 错误，重跑。开赛前会话就结束的，按第 5 条算没到场（`no_show`，核心 0.2.7）。两方及以上的比赛不变：回合制收比赛时轮到的一方判负（`abandoned`）；实时项目里会话先结束的一方停在场上不动，到它该复盘时超过 `READY_TIMEOUT_S` 判负（`ready_timeout`），所有方的会话都结束时比赛还没完，就是没打完的无效比赛（`collected`）。（2026-10-07 起，核心 0.2.6。）

**机器人与模式规则**（运动引用资源库里的机器人、以及每个"运动 × 装配配置 × 模式"进正式榜之前检查；2026-10-08 起）
1. **真实限位（B0，最高原则）**：所有机器人、两种模式都遵守真机的限制（关节位置、速度、加速度、加加速度、力矩、力矩变化率、底盘速度和加速度）；手臂不能穿过自己的身体（自碰撞：手臂和躯干、头、底座，左右臂之间，手持器具和身体之间都有接触，排除清单写明理由；2026-10-09 起，`bodies/selfcollision.py`，bodies/README.md §4.1）。厂商 SDK 和 URDF 不一致时用 SDK；没有官方值的用最接近真机的来源并写明推算方法；不允许不限。平台提供的技能（高层的 hit、kick，底层的行走策略）同样要守，和选手的命令过同一道检查，超出的被拒绝或截断，并记进比赛日志。限位由资源库里唯一可信的共享限位和伺服模块执行，用资源库机器人的运动必须调用它、不能绕开，CI 对每个这样的运动检查。
2. **真机可行性（B9）**：按"运动 × 装配配置"检验（装上的手和工具会改变质量和重心，原装 G1 上稳定的行走策略，装配后不默认稳定）。进正式榜前，用上界选手（知道真实状态、搜索所有可行动作）在真实限位下检验这项运动的基本动作（接发、回球、射门等）。球拍类的门槛：合法发球上界可回 ≥ 90%；其他项目按项目定。达不到的组合要么改运动设置，要么只作挑战组单独报告，不和其他机器人同榜。
3. **工具包清单**：清单属于运动，每项运动为它用的每个装配配置写一份公开清单，写明两种模式各提供什么（控制器、策略权重及其来源、许可证和 sha256、IK 库及其版本、关节接口和频率、权限）和禁止什么（运动专用技能、外部权重）。清单是任务版本的一部分：改清单要升运动版本、重验参考解；同一项运动的所有选手拿到同一份清单。
4. **参考解**：每个提供的"运动 × 装配配置 × 模式"都有一份我们只用清单里的工具做出的参考解，证明这个组合可解（第 2 条证明物理上可行，这一条证明用这套工具可行）。
5. **核心技能边界**：底层清单里的工具不能已经解决这项运动的核心运动挑战。只用工具、不开发运动技能的最简实现，在底层的 API 测试上应接近 0（门槛写在清单里）；做不到就收紧成功条件（目标区域、出球速度）；IK 库照常提供。
6. **控制权仲裁**：运动的动作层（可用资源库里共享的仲裁代码）按关节组仲裁控制权，同一组关节同一时刻只有一个控制源；交接平滑、守限位；行走策略收回控制时重新初始化，确认站稳后才算交回；每个控制步记下每组的控制源，重放时核对。
7. **工具不越界**：模拟器里运行的控制器和基础策略只读机器人自己的本体状态和选手给的指令，不读球、对手和场地。
8. **确定性**：动作接口以下的一切（伺服、仲裁、行走策略）在 numpy 里确定性运行，内部状态进状态哈希，重放逐位一致；GPU 推理不进入可信路径。选手自己训练的控制器只在选手容器里运行（只用 CPU，训练时间算进会话）。
9. **装配配置有版本**：每个装配配置（机器人 + 手或工具 + 控制器）有自己的版本 hash，记进结果；换手臂是换机器人模型。
10. **底层交付约定**：底层的技能实现放在 `/app/output/` 下的固定路径（例：`/app/output/skill/hit.py`），函数签名固定（和高层的技能 API 同一份接口规格）；只用清单里的底层工具；不依赖比赛进程的状态；确定性运行。比赛结束时这个路径上的版本就是最终实现，平台在另一个隔离容器里对它做 API 测试和比赛测试；隐藏请求只在赛后使用。建议每次复盘存一份快照，用来分析技能在比赛中怎么变好。

**渲染器规则**（接入渲染器时检查）
1. 渲染器不改物理，也不改比赛状态。
2. 同一份场景：比赛区域的几何和相机位置与默认渲染器一致，渲染前自动核对（展示场景见第 6 条：外观可以不同，运动学必须相同）。
3. 用于选手画面时，每帧时间必须满足任务的预算；需要 GPU 的要声明。
4. 选手画面用的渲染器记进结果；不同渲染器的成绩不合在一张榜里。
5. 重新渲染的视频每一帧都标出渲染器名字。
6. 展示场景默认只给观众，运动在 `presentation.toml` 里声明 `players` 后也用于选手的画面。两种情况都要和比赛场景运动学相同（刚体、关节、坐标系、惯量、相机一致；比赛场景的几何体只能改外观或隐藏，新增的只能是不碰撞的可见几何体），渲染前自动核对（`renderers/scene_check.same_kinematics`）。只给观众时：选手的画面不能用，每一帧标出用的是展示网格，文件不进模拟器目录。用于选手的画面时：声明的网格（完整或减面）连同许可证打包进模拟器目录，算进 `source_sha`，按 sha256 核对；满足帧预算（第 3 条），容器的 CPU 和渲染线程数写在 `[observation]`；运动升版本并重新验证参考选手。第三方模型按固定提交和 sha256 下载，完整模型不进仓库；减面版本由固定的上游文件生成，体积小，提交进仓库（`renderers/presentation/lod/`）；都保留许可证并注明出处。（2026-10-06 起，核心 0.2.4；用于选手的画面：2026-10-07 起，核心 0.2.9。）

**工具规则**
1. 不 import 模拟器，不影响比赛结果；直播日志只写不读。
2. 不写死本机路径和机器名。
3. 结果记下四个版本号；版本不同的结果不混在一张榜里。
4. 赛季冻结：赛季期间不改任务，补跑用同一个版本。正式比赛用登记过的保密种子集（6.4 节），私密种子集不进仓库。

## 10. 迁移计划

现有的 18 项运动按下面的顺序迁到新结构。每一步都保持所有任务通过检查：每个任务先用参考选手录一场比赛，改造后用新代码重放，比分和状态哈希必须完全一致。

| 阶段 | 内容 |
|---|---|
| 0 | 补齐核心：一方多个座位、多方和名次、动作延后生效、带参数的观测、队内频道、无对手的计分单人、直播和录像；用足球验证最难的情况；放一个最小示例运动；建立"选手"部分（harness 适配和不写代码的结构限制挪进来）；建立渲染器插件（运动只提供场景和相机定义，比赛中和赛后都能选渲染器；现有的 MuJoCo、Cycles、Isaac 三种接进来） |
| 1 | 保龄球、推杆高尔夫、飞盘高尔夫、冰壶、台球、射箭、篮球投篮、飞镖单人 |
| 2 | 空气曲棍球、桌上足球、无人机、乒乓单人 |
| 3 | 击剑、网球、羽毛球、赛车 |
| 4 | 棒球、足球 |

新的运动（田径、冰球、沙排等）直接按新结构写。目前这套接口只在乒乓对战和飞镖对战上验证过（两项运动的模拟器和裁判代码一行没改，标准回放逐位一致），足球、棒球、赛车最可能需要新接口，所以阶段 0 先用足球验证。

讨论过程（包括从旧的九层结构到现在的对应）见文档列表里的讨论稿。
