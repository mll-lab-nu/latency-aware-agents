# 构建与筛选：一项运动怎样进入 benchmark

版本 v2（2026-10-10）。本文先说明 benchmark 由哪些部分组成、设计时要回答哪些问题、一项运动按什么流程进入 benchmark，再逐一详细说明每个步骤。框架的代码结构见[框架说明](framework_zh.md)，各条规则的完整条文见框架说明第 9 节。

**怎么读**：
- 想了解全貌，读第 1 节；
- 想知道为什么这样设计、和已有工作有什么区别，读第 2 节；
- 想了解一项运动怎样进入 benchmark，读第 3 节；要筛选或加入一项运动，再按第 4–9 节逐步进行；
- 关心模拟是否符合真实物理和真实机器人，读第 5 节；
- 关心底层模式给 agent 什么控制能力，读第 6 节；
- 关心比赛的安全与公平，读第 7 节；
- 关心九项能力怎样评测、能得出什么结论，读第 9 节。

## 总览

<svg viewBox="0 0 920 440" width="100%" role="img" aria-label="benchmark 的组成：被测的 coding agent、一项运动的正式配置、核心和评测产出" style="max-width:920px;font-family:inherit">
<style>.bd{fill:rgb(206,112,58);font-size:11px;font-weight:600}.t{fill:var(--fg,#16211c);font-size:13px}.h{fill:var(--fg,#16211c);font-size:15px;font-weight:600}.s{fill:var(--muted,#56665e);font-size:12px}.m{fill:var(--fg,#16211c);font-size:11.5px;font-family:ui-monospace,Menlo,monospace}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}</style>
<defs><marker id="m81001" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<rect x="20" y="20" width="220" height="330" rx="10" class="pl" stroke-width="1.3"/>
<text x="130" y="48" text-anchor="middle" class="h cpl">被测对象：coding agent</text>
<text x="130" y="72" text-anchor="middle" class="s">模型 + harness</text>
<text x="130" y="91" text-anchor="middle" class="s">（如 Claude Code、Codex）</text>
<rect x="34" y="108" width="192" height="92" rx="8" class="nu" stroke-width="1.2"/>
<text x="130" y="133" text-anchor="middle" class="t b">Athlete（高层）</text>
<text x="130" y="156" text-anchor="middle" class="s">平台提供运动技能</text>
<text x="130" y="176" text-anchor="middle" class="s">agent 编写竞技程序</text>
<rect x="34" y="212" width="192" height="92" rx="8" class="nu" stroke-width="1.2"/>
<text x="130" y="237" text-anchor="middle" class="t b">Engineer（底层）</text>
<text x="130" y="260" text-anchor="middle" class="s">平台提供机器人和工具包</text>
<text x="130" y="280" text-anchor="middle" class="s">agent 实现同一套技能</text>
<text x="130" y="330" text-anchor="middle" class="s">比赛中编写、运行、修改程序</text>
<path d="M243 185L277 185" class="ar" marker-start="url(#m81001)" marker-end="url(#m81001)"/>
<rect x="280" y="20" width="340" height="250" rx="10" class="sp" stroke-width="1.3"/>
<text x="450" y="46" text-anchor="middle" class="h csp">一项运动的正式配置</text>
<text x="450" y="66" text-anchor="middle" class="s">经筛选确定并冻结，对所有 agent 相同</text>
<rect x="296" y="80" width="308" height="30" rx="6" class="nu" stroke-width="1.1"/>
<text x="310" y="100" class="t">规则：官方规则书 + 偏差清单</text>
<text x="594" y="100" text-anchor="end" class="bd">第 ① 步</text>
<rect x="296" y="116" width="308" height="30" rx="6" class="nu" stroke-width="1.1"/>
<text x="310" y="136" class="t">机器人：真实限位，真机可行</text>
<text x="594" y="136" text-anchor="end" class="bd">第 ② 步</text>
<rect x="296" y="152" width="308" height="30" rx="6" class="nu" stroke-width="1.1"/>
<text x="310" y="172" class="t">场地：标准尺寸，或声明的适配版本</text>
<text x="594" y="172" text-anchor="end" class="bd">第 ② 步</text>
<rect x="296" y="188" width="308" height="30" rx="6" class="nu" stroke-width="1.1"/>
<text x="310" y="208" class="t">接口：技能 API（高层）/ 工具包（底层）</text>
<text x="594" y="208" text-anchor="end" class="bd">第 ③ 步</text>
<rect x="296" y="224" width="308" height="30" rx="6" class="nu" stroke-width="1.1"/>
<text x="310" y="244" class="t">对手、种子与计分</text>
<text x="594" y="244" text-anchor="end" class="bd">第 ① 步</text>
<rect x="280" y="284" width="340" height="66" rx="10" class="co" stroke-width="1.3"/>
<text x="450" y="309" text-anchor="middle" class="t b cco">核心：主持比赛</text>
<text x="450" y="331" text-anchor="middle" class="s">统一时钟 · 观测白名单 · 可重放的记录</text>
<path d="M623 185L657 185" class="ar" marker-end="url(#m81001)"/>
<rect x="660" y="20" width="240" height="330" rx="10" class="to" stroke-width="1.3"/>
<text x="780" y="46" text-anchor="middle" class="h cto">评测产出</text>
<text x="676" y="78" class="t b">正式成绩</text>
<text x="676" y="98" class="s">胜负、得分、排名</text>
<text x="676" y="130" class="t b">底层技能测试</text>
<text x="676" y="150" class="s">隐藏请求测试、</text>
<text x="676" y="168" class="s">固定参考策略下的比赛测试</text>
<text x="676" y="200" class="t b">过程诊断</text>
<text x="676" y="220" class="s">经过效度检验的指标</text>
<text x="676" y="252" class="t b">受控实验</text>
<text x="676" y="272" class="s">只改变一个条件的对照</text>
<text x="676" y="304" class="t b">开发成本</text>
<text x="676" y="324" class="s">token、时间、修改次数</text>
<rect x="20" y="368" width="880" height="56" rx="10" class="gr" stroke-width="1.2"/>
<text x="460" y="392" text-anchor="middle" class="t">单人、对脚本对手 → <tspan class="b">First Exam</tspan>　｜　对战、团队、多方（模型对模型）→ <tspan class="b">Arena</tspan></text>
<text x="460" y="413" text-anchor="middle" class="s">任务 = 运动 × 比赛形式 × 模式 × 等级；比赛 = 任务 × 选手 × 种子 × 先手顺序</text>
</svg>

**图 1　benchmark 的组成。左：被测对象；中：一项运动的正式配置（右侧标出它在构建流程的哪一步确定）和主持比赛的核心；右：评测产出。**

benchmark 让 coding agent 编写程序，在遵守真实物理约束的体育比赛中上场，评测它开发**竞技策略**和**机器人运动技能**的能力。

- **被测对象**：coding agent，即一个模型加上它使用的 harness。
  - 它不逐帧决定动作，而是编写一个选手程序上场，在复盘时修改程序。
  - 按平台提供什么，分为两种模式（见第 1.1 节）。
- **一项运动的正式配置**：
  - 包括规则、机器人、场地、接口、对手、种子和计分；
  - 由第 3 节的筛选流程确定，确定后冻结，对所有 agent 相同。
- **评测产出**：
  - 正式成绩只看比赛结果；
  - 底层另有技能测试；
  - 过程诊断和受控实验用于分析能力，不进入正式成绩。
- **两类比赛**：
  - 单人和对脚本对手的比赛用于 First Exam；
  - 模型对模型的对战、团队和多方比赛构成 Arena。

**统一什么，不统一什么。** benchmark 统一能力目标、设计原则和验收流程，不统一机器人、控制算法和场地。不同运动可以使用不同的机器人；每项运动找到一套通过全部检验的配置即可进入 benchmark。

### 两种模式

| | Agent as Athlete（高层） | Agent as Robotics Engineer（底层） |
|---|---|---|
| 研究问题 | 机器人已具备基本运动技能时，agent 能否编写并持续改进一个在动态物理环境中有效竞技的程序 | 给定机器人、控制接口和工程资源，agent 能否自行开发、集成和调试这项运动所需的运动技能 |
| 平台提供 | 运动技能 API | 机器人模型、关节接口和公开的工具包 |
| agent 开发 | 竞技程序 | 竞技程序 + 同一套技能 API 的实现 |
| 评价 | 比赛成绩 | 比赛成绩、隐藏请求测试、固定参考策略下的比赛测试，分别报告 |

两种模式回答不同的研究问题，而不是同一问题的两个难度等级。

### 每项运动设哪种模式

**不是每项运动都要两种模式。** 一种模式只有在它能测出东西时才设：两种都设可以，只设一种也可以。

**模式是配置的结果，不是运动固有的属性。** 同一项运动换一种机器人或技能设计，结论可能不同：
- 冰壶、保龄球目前使用出手机器，命令就是出手参数，没有底层可开发的技能；换成机器人身体来投掷，就有了底层。
- 篮球 3x3 的高层技能怎样划分（粗到几个，细到十几个），目前还没有定论。

所以设哪种模式、技能怎样划分，都是起点"列出候选"时搜索的一部分，可以让 agent 参与探索：
1. 提出候选的技能集合；
2. 实现并跑参考选手；
3. 用第 ④ 步的检验比较不同方案。

**不适合设高层的情况**：
- **平台技能会把运动本身做完**：一旦提供了这个技能，剩下的竞技判断微不足道。例如短跑：给了奔跑技能，成绩只由一个速度参数决定。
- **平台做不出可靠的技能**：没有可靠的控制方法实现这个技能，参考选手无法完成比赛（违反 P4）。这时可以先开放能可靠实现的部分技能，其余暂缓。
- **核心技能可以绕开**：不用核心技能也能得分，而且无法用规则堵住。

**不适合设底层的情况**：
- **当前配置下没有要开发的运动技能**：
  - 出手机器的命令就是出手参数，"实现技能"和"下命令"是同一件事；
  - 赛车、无人机、桌上足球的动作本身就是机器的控制量。
- **基础控制已经完成了核心技能**：基础控制是让任务可完成所必须提供的，例如腿式人形的行走策略。如果给了它之后，剩下的运动技能已经由它完成，底层就和高层没有区别。
- **做不出参考解**：在规定的机器人、工具和算力下，没有可靠的参考实现。
- **测不出区分度**：隐藏请求测试和比赛测试分不出实现的好坏，或者所有模型都得 0。

**由每项运动的负责人审核决定**：
- 负责人按本文的流程，决定这项运动设哪种模式、用什么机器人、技能怎样划分；
- 高层要通过 P4、P5 和技能必要性检验，底层要通过第 6.7 节的四条标准；
- 结论和证据写进这项运动的筛选报告和规格文件。

现有的实现只是当前的状态。例如：
- 乒乓的两种模式都已实现；
- 出手类项目目前只有高层；
- 短跑、跨栏目前只有底层。

这些都可以随新的机器人、技能设计和检验结果改变。

## 设计 benchmark 要回答的问题

设计一个 benchmark，要先回答一组问题：测什么，环境是否真实，任务能不能完成，agent 解决的是不是我们想测的问题，成绩是否反映能力，能否区分不同的 agent，结果是否可信。本节先说明这个 benchmark 和已有工作的区别，再逐个回答这些问题。

这些问题分三组：
- **共同的底座**：和角色无关，每项运动都要先回答；
- **两条线**：运动员（Athlete，高层）和工程师（Engineer，底层）分别回答同一组问题，回答的方式不同；
- **共同的收尾**：结果是否可信。

第 3 节的六个步骤就是为回答这些问题而设的；本节最后用一张图把问题和步骤对应起来。

### 和已有 benchmark 的区别

和这个 benchmark 最接近的工作有三类：
- 让机器人或模拟人体做体育运动，被测的是强化学习算法；
- 让模型直接输出动作或离线写程序来比赛；
- 让编码 agent 当参赛者，但比的不是物理任务。

我们和它们的区别，正是对上面那组问题的回答不同：

| 核心问题 | 已有工作 | 这个 benchmark |
|---|---|---|
| 测谁、测什么 | SMPLOlympics、DeepMind 的乒乓和足球：强化学习算法。LLM Colosseum、Orak：模型直接出动作。CodeClash：编码 agent，但不是物理任务。ArtifactArena：模型离线写机器人本体和控制器。NVIDIA Robo Olympics：编码 agent 是搭建演示的人，不是被测对象 | 编码 agent 本身，分**运动员**和**工程师**两种角色，在同一套物理任务里分开测竞技决策和运动技能开发 |
| 环境是否真实 | SMPLOlympics 用模拟人体；ArtifactArena 的本体和规则都是自己设计的 | 真实机器人，遵守厂商给出的限位；真实规则书（ITTF、FIE、BWF 等），偏离逐条声明；物理真实性审计和真机可行性检验 |
| 实时与在环 | 实时游戏这条线研究过"思考耗时"，但不在物理世界里；其他物理任务的工作不实时 | 思考和计算消耗比赛时间；只给相机画面；比赛进行中可以改程序；模型对模型 |
| 成绩是否反映能力 | 通常报告比赛成绩或演示结果 | 每项运动进入 benchmark 前，要用参考解、简单基线和能力敏感性证明成绩反映声称的能力，筛选报告公开 |

<svg viewBox="0 0 920 472" width="100%" role="img" aria-label="和最接近的工作在九个方面的对比矩阵" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.s{fill:var(--muted,#56665e);font-size:12px}.b{font-weight:600}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.rl{stroke:var(--line,#d7e0db);stroke-width:1}.cf{fill:rgb(62,150,106)}.ce{fill:none;stroke:var(--muted,#56665e);stroke-width:1.3}.na{stroke:var(--muted,#56665e);stroke-width:1.3;opacity:.5}.csp{fill:rgb(206,112,58)}.ar{fill:rgba(98,148,222,.07)}</style>
<rect x="270" y="14" width="105" height="418" rx="8" class="sp" stroke-width="1.3"/>
<text x="322" y="44" text-anchor="middle" class="t b csp">我们</text><text x="427" y="44" text-anchor="middle" class="s">ArtifactArena</text><text x="532" y="36" text-anchor="middle" class="s">NVIDIA</text><text x="532" y="54" text-anchor="middle" class="s">Robo Olympics</text><text x="637" y="44" text-anchor="middle" class="s">SMPLOlympics</text><text x="742" y="44" text-anchor="middle" class="s">CodeClash</text><text x="847" y="36" text-anchor="middle" class="s">LLM Colosseum</text><text x="847" y="54" text-anchor="middle" class="s">/ Orak</text>
<line x1="20" y1="70" x2="900" y2="70" class="rl"/>
<text x="20" y="96" class="t">被测的是编码 agent 本身</text>
<circle cx="322" cy="92" r="7" class="cf"/><circle cx="427" cy="92" r="7" class="ce"/><path d="M427 85A7 7 0 0 1 427 99Z" class="cf"/><circle cx="532" cy="92" r="7" class="ce"/><circle cx="637" cy="92" r="7" class="ce"/><circle cx="742" cy="92" r="7" class="cf"/><circle cx="847" cy="92" r="7" class="ce"/>
<text x="20" y="132" class="t">测竞技决策（运动员）</text>
<line x1="20" y1="110" x2="900" y2="110" class="rl"/>
<circle cx="322" cy="128" r="7" class="cf"/><circle cx="427" cy="128" r="7" class="ce"/><path d="M427 121A7 7 0 0 1 427 135Z" class="cf"/><circle cx="532" cy="128" r="7" class="ce"/><circle cx="637" cy="128" r="7" class="ce"/><circle cx="742" cy="128" r="7" class="cf"/><circle cx="847" cy="128" r="7" class="cf"/>
<text x="20" y="168" class="t">测运动技能开发（工程师）</text>
<line x1="20" y1="146" x2="900" y2="146" class="rl"/>
<circle cx="322" cy="164" r="7" class="cf"/><circle cx="427" cy="164" r="7" class="cf"/><circle cx="532" cy="164" r="7" class="ce"/><path d="M532 157A7 7 0 0 1 532 171Z" class="cf"/><circle cx="637" cy="164" r="7" class="ce"/><line x1="735" y1="164" x2="749" y2="164" class="na"/><line x1="840" y1="164" x2="854" y2="164" class="na"/>
<text x="20" y="204" class="t">真实机器人（厂商限位）</text>
<line x1="20" y1="182" x2="900" y2="182" class="rl"/>
<circle cx="322" cy="200" r="7" class="cf"/><circle cx="427" cy="200" r="7" class="ce"/><circle cx="532" cy="200" r="7" class="ce"/><path d="M532 193A7 7 0 0 1 532 207Z" class="cf"/><circle cx="637" cy="200" r="7" class="ce"/><line x1="735" y1="200" x2="749" y2="200" class="na"/><line x1="840" y1="200" x2="854" y2="200" class="na"/>
<text x="20" y="240" class="t">真实规则书</text>
<line x1="20" y1="218" x2="900" y2="218" class="rl"/>
<circle cx="322" cy="236" r="7" class="cf"/><circle cx="427" cy="236" r="7" class="ce"/><circle cx="532" cy="236" r="7" class="ce"/><circle cx="637" cy="236" r="7" class="ce"/><path d="M637 229A7 7 0 0 1 637 243Z" class="cf"/><line x1="735" y1="236" x2="749" y2="236" class="na"/><line x1="840" y1="236" x2="854" y2="236" class="na"/>
<text x="20" y="276" class="t">实时：思考消耗比赛时间</text>
<line x1="20" y1="254" x2="900" y2="254" class="rl"/>
<circle cx="322" cy="272" r="7" class="cf"/><circle cx="427" cy="272" r="7" class="ce"/><circle cx="532" cy="272" r="7" class="ce"/><circle cx="637" cy="272" r="7" class="ce"/><circle cx="742" cy="272" r="7" class="ce"/><circle cx="847" cy="272" r="7" class="cf"/>
<text x="20" y="312" class="t">比赛进行中改程序</text>
<line x1="20" y1="290" x2="900" y2="290" class="rl"/>
<circle cx="322" cy="308" r="7" class="cf"/><circle cx="427" cy="308" r="7" class="ce"/><line x1="525" y1="308" x2="539" y2="308" class="na"/><line x1="630" y1="308" x2="644" y2="308" class="na"/><circle cx="742" cy="308" r="7" class="ce"/><line x1="840" y1="308" x2="854" y2="308" class="na"/>
<text x="20" y="348" class="t">只给相机画面</text>
<line x1="20" y1="326" x2="900" y2="326" class="rl"/>
<circle cx="322" cy="344" r="7" class="cf"/><circle cx="427" cy="344" r="7" class="ce"/><circle cx="532" cy="344" r="7" class="ce"/><path d="M532 337A7 7 0 0 1 532 351Z" class="cf"/><circle cx="637" cy="344" r="7" class="ce"/><circle cx="742" cy="344" r="7" class="ce"/><circle cx="847" cy="344" r="7" class="ce"/><path d="M847 337A7 7 0 0 1 847 351Z" class="cf"/>
<text x="20" y="384" class="t">模型对模型</text>
<line x1="20" y1="362" x2="900" y2="362" class="rl"/>
<circle cx="322" cy="380" r="7" class="cf"/><circle cx="427" cy="380" r="7" class="cf"/><circle cx="532" cy="380" r="7" class="ce"/><circle cx="637" cy="380" r="7" class="ce"/><path d="M637 373A7 7 0 0 1 637 387Z" class="cf"/><circle cx="742" cy="380" r="7" class="cf"/><circle cx="847" cy="380" r="7" class="cf"/>
<circle cx="290" cy="454" r="6" class="cf"/><text x="302" y="458" class="s">是</text><circle cx="350" cy="454" r="6" class="ce"/><path d="M350 448A6 6 0 0 1 350 460Z" class="cf"/><text x="362" y="458" class="s">部分</text><circle cx="422" cy="454" r="6" class="ce"/><text x="434" y="458" class="s">否</text><line x1="476" y1="454" x2="488" y2="454" class="na"/><text x="494" y="458" class="s">不适用（不是物理任务，或没有这个环节）</text>
</svg>

**图 2　和最接近的工作对比。已有工作大多只测其中一种角色：CodeClash、LLM Colosseum 测的是竞技决策，ArtifactArena 测的是本体和控制器的设计，NVIDIA 的 agent 做的是工程师的工作，但不是被测对象。每一项单独都有人做过；同时满足这几项、并在同一套任务里分开测两种角色的，据我们所知只有这个 benchmark。各项依据和更多工作见[相关工作](http://mll-4090-1.cs.northwestern.edu:8042/agent-sports-site/related/)。**

### 共同的底座

这四个问题和角色无关，每项运动都要先回答。

#### 问题 1：测什么？

**为什么重要。** 关于这个 benchmark 的讨论里，同时出现过很多意见：要不要统一成手柄接口，不同运动为什么用不同机器人，跑步和篮球的技能怎么定义。每条都有道理，但如果"benchmark 到底测什么能力"没有统一的答案，就无法判断"统一手柄"或"统一机器人"是好是坏。

**我们的回答。**
- 测编码 agent 在真实物理约束下的体育比赛中的能力，分三类：
  - 物理世界理解：三维感知、动力学、具身、时间与延迟；
  - 比赛理解：规则、策略与读对手、团队协作；
  - 编码 agent 能力：实时编程、在比赛中学习。
- 分两种角色：
  - **运动员**：机器人已经具备运动技能时，agent 能否写出并持续改进一个有效竞技的程序；
  - **工程师**：给定机器人和工程资源，agent 能否自己开发出这项运动所需的运动技能；
  - 两种角色回答不同的研究问题，不是同一问题的两个难度。
- **统一测量原则，不统一机器人和接口**：不同运动用不同的机器人和控制方式是正常的。

**怎么检验。** 第 ① 步写下每项运动保留的核心挑战和声称考察的能力；后面的步骤逐项检验这份声明。

#### 问题 2：这项运动适合哪种角色？

**为什么重要。** 不是每项运动都能同时测两种角色。例如短跑：如果平台提供现成的奔跑技能，agent 只剩"开始、停下"，运动员这条线就测不出东西。反过来，冰壶用出手机器时，命令就是出手参数，工程师也没有技能可开发。

**我们的回答。** 一种角色只有在能测出东西时才设；判断标准见第 1.2 节。设哪种角色、技能怎样划分，是候选搜索的一部分，可以让 agent 参与探索，由每项运动的负责人决定。

**怎么检验。** 第 ① 步提出，第 ③ 步确定；第 ④ 步的检验不通过时可以撤掉一种角色。

#### 问题 3：环境是否真实？

**为什么重要。** 不真实的环境会奖励真实世界中不存在的打法。例如羽毛球早期把球拍增益设成 20，球速约是真实球拍的 5 倍，agent 学到的打法在真实世界里不可能出现。benchmark 的可信度来自物理真实。

**我们的回答。** 真实性分层：物理真实和机器人真实是底线，不能为了让任务可完成而修改；运动配置可以为机器人适配，但必须声明，适配版本和标准版本是不同的任务。

<svg viewBox="0 0 920 300" width="100%" role="img" aria-label="真实性分层：物理真实和机器人真实是底线，运动配置可以适配但要声明" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.s{fill:var(--muted,#56665e);font-size:12px}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}</style>
<rect x="20" y="20" width="500" height="262" rx="12" class="sp" stroke-width="1.3"/>
<text x="38" y="46" class="t b csp">运动配置：可以适配，必须声明</text>
<text x="38" y="66" class="s">场地尺寸、局长、发球分布、对手等级</text>
<rect x="44" y="84" width="452" height="182" rx="10" class="co" stroke-width="1.3"/>
<text x="62" y="110" class="t b cco">机器人真实：底线，不可改</text>
<text x="62" y="130" class="s">厂商给出的速度、加速度、加加速度、力矩限位</text>
<text x="62" y="148" class="s">对应真实存在的机器；真机可行性检验</text>
<rect x="68" y="166" width="404" height="84" rx="8" class="pl" stroke-width="1.3"/>
<text x="86" y="192" class="t b cpl">物理真实：底线，不可改</text>
<text x="86" y="212" class="s">接触、摩擦、空气阻力由引擎计算</text>
<text x="86" y="230" class="s">参数公开、有出处；不关闭接触和自碰撞</text>
<rect x="540" y="20" width="360" height="124" rx="10" class="nu" stroke-width="1.3"/>
<text x="558" y="46" class="t b cpl">允许（要声明）</text>
<text x="558" y="70" class="s">羽毛球场地缩小到 0.4 倍，网高不变</text>
<text x="558" y="92" class="s">乒乓一局 5 分（控制比赛时长）</text>
<text x="558" y="114" class="s">单臂机器人用抛球装置代替空手抛球</text>
<rect x="540" y="158" width="360" height="124" rx="10" class="nu" stroke-width="1.3"/>
<text x="558" y="184" class="t b csp">不允许</text>
<text x="558" y="208" class="s">放大关节速度或力矩，让机器人够得着</text>
<text x="558" y="230" class="s">把球拍增益调到真实球拍的数倍</text>
<text x="558" y="252" class="s">关闭接触或自碰撞，让动作做得出来</text>
</svg>

**图 3　真实性分层。里面两层是底线；外层可以为机器人适配，但每一处偏离都要写进规则偏离表，适配版本和标准版本的成绩不直接比较。**

**怎么检验。** 第 ② 步：真实限位有出处、有真实对应的机器、物理一致（第 5 节的六条原则和物理真实性审计）。

#### 问题 4：物理上能不能完成？

**为什么重要。** 如果对方发过来的球机器人根本接不到，比赛就测不出任何东西。例如乒乓第一赛季用固定底座的 Franka 机械臂，选手发的短球几乎接不到。

**我们的回答。**
- **机器人和场地在设计阶段一起选**：有的机器人需要适配场地才能完成运动，而场地是否合适又取决于机器人。先固定其中一个再选另一个会形成循环依赖，所以两者放在同一步，在评测前一起冻结。
- **以运动为主**：先从官方配置出发，找能做到的机器人；找不到，再在说明理由的前提下适配场地。我们的目标是用体育测 agent，不是研究某台机器人能参加多少项运动。
- **每项运动找到一套可行的配置即可**，不穷举机器人和场地的所有组合。
- **用上界检验可行性**：让一个"知道一切"的选手（知道球的真实状态，搜索所有可行的击球）在真实限位下打，看能完成多少。连它都做不到，任何 agent 都做不到。

<svg viewBox="0 0 920 300" width="100%" role="img" aria-label="乒乓候选机器人的真机可行性：合法发球中能被接回的比例，门槛 90%" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.s{fill:var(--muted,#56665e);font-size:12px}.b{font-weight:600}.bp{fill:rgba(76,170,124,.55)}.bf{fill:rgba(140,140,140,.35)}.ax{stroke:var(--line,#d7e0db);stroke-width:1}.th{stroke:rgb(206,112,58);stroke-width:1.6;stroke-dasharray:5 4}.csp{fill:rgb(206,112,58)}.cpl{fill:rgb(62,150,106)}</style>
<line x1="260" y1="30" x2="260" y2="262" class="ax"/>
<line x1="860" y1="30" x2="860" y2="262" class="ax"/>
<text x="260" y="282" text-anchor="middle" class="s">0%</text>
<text x="560" y="282" text-anchor="middle" class="s">50%</text>
<text x="860" y="282" text-anchor="middle" class="s">100%</text>
<line x1="560" y1="30" x2="560" y2="262" class="ax"/>
<line x1="800" y1="22" x2="800" y2="262" class="th"/>
<text x="800" y="16" text-anchor="middle" class="s csp">门槛 90%</text>
<text x="250" y="52" text-anchor="end" class="t b">Franka Panda + 导轨</text>
<rect x="260" y="36" width="568" height="24" rx="3" class="bp"/>
<text x="834" y="53" class="t b cpl">94.7%</text>
<text x="250" y="90" text-anchor="end" class="t">ABB IRB 1100 + 龙门</text>
<rect x="260" y="74" width="443" height="24" rx="3" class="bf"/>
<text x="709" y="91" class="t">73.9%</text>
<text x="250" y="128" text-anchor="end" class="t">RB-Y1（轮式人形）</text>
<rect x="260" y="112" width="412" height="24" rx="3" class="bf"/>
<text x="678" y="129" class="t">68.7%</text>
<text x="250" y="166" text-anchor="end" class="t">Unitree G1（腿式人形）</text>
<rect x="260" y="150" width="203" height="24" rx="3" class="bf"/>
<text x="469" y="167" class="t">33.8%</text>
<text x="250" y="204" text-anchor="end" class="t">Unitree H1-2（腿式人形）</text>
<rect x="260" y="188" width="137" height="24" rx="3" class="bf"/>
<text x="403" y="205" class="t">22.9%</text>
<text x="250" y="242" text-anchor="end" class="t">Galaxea R1 Pro（轮式）</text>
<rect x="260" y="226" width="2" height="24" class="bf"/>
<text x="268" y="243" class="t">0%（加加速度限位太小，挥不动拍）</text>
</svg>

**图 4　乒乓的真机可行性检验。每种机器人都按厂商限位运动，标准球台；纵向是候选机器人，横向是 284 个合法发球中，"知道一切"的接球者从最佳准备位置能接回的比例。只有 Franka + 导轨超过 90% 的门槛，所以乒乓选它。**

**怎么检验。** 第 ② 步的真机可行性检验（门槛按运动类型定，球拍类为 90%）；第 ④ 步再用参考选手确认在真实比赛里能赢。

### 两条线：运动员与工程师

两种角色要回答同一组问题：平台给什么、能不能完成、会不会太简单、测到的是不是声称的能力、能不能区分 agent。回答的逻辑相同，检验的方式不同。

先看两种角色的分工。一项运动的控制链从相机画面开始，到物理世界结束；两种角色的区别，是 agent 负责到哪一段：

<svg viewBox="0 0 920 372" width="100%" role="img" aria-label="控制链与分工：运动员和工程师两种角色下 agent 与平台各负责哪一段；技能给动作和给结果的区别" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.s{fill:var(--muted,#56665e);font-size:12px}.m{fill:var(--fg,#16211c);font-size:11px;font-family:ui-monospace,Menlo,monospace}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.10);stroke:rgba(140,140,140,.45)}.ag{fill:rgba(76,170,124,.30)}.pf{fill:rgba(140,140,140,.22)}.sm{fill:rgba(140,140,140,.08);stroke:rgba(140,140,140,.45);stroke-dasharray:4 3}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.hl{fill:none;stroke:rgb(206,112,58);stroke-width:1.6;stroke-dasharray:5 4}</style>
<defs><marker id="m81011" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<text x="344" y="22" text-anchor="middle" class="s csp">若技能接收"落点"，这两段由平台完成</text>
<rect x="238" y="30" width="212" height="80" rx="10" class="hl"/>
<rect x="20" y="40" width="98" height="60" rx="8" class="nu" stroke-width="1.2"/><text x="69" y="64" text-anchor="middle" class="t b">相机画面</text><text x="69" y="84" text-anchor="middle" class="s">几帧图像</text>
<rect x="130" y="40" width="98" height="60" rx="8" class="nu" stroke-width="1.2"/><text x="179" y="64" text-anchor="middle" class="t b">感知</text><text x="179" y="84" text-anchor="middle" class="s">球在哪里</text>
<rect x="240" y="40" width="98" height="60" rx="8" class="nu" stroke-width="1.2"/><text x="289" y="64" text-anchor="middle" class="t b">预测</text><text x="289" y="84" text-anchor="middle" class="s">球会怎么飞</text>
<rect x="350" y="40" width="98" height="60" rx="8" class="nu" stroke-width="1.2"/><text x="399" y="64" text-anchor="middle" class="t b">决策</text><text x="399" y="84" text-anchor="middle" class="s">何时何处打向哪</text>
<rect x="460" y="40" width="98" height="60" rx="8" class="nu" stroke-width="1.2"/><text x="509" y="64" text-anchor="middle" class="t b">技能命令</text><text x="509" y="84" text-anchor="middle" class="m">hit(…)</text>
<rect x="570" y="40" width="98" height="60" rx="8" class="nu" stroke-width="1.2"/><text x="619" y="64" text-anchor="middle" class="t b">技能实现</text><text x="619" y="84" text-anchor="middle" class="s">IK、挥拍轨迹</text>
<rect x="680" y="40" width="98" height="60" rx="8" class="nu" stroke-width="1.2"/><text x="729" y="64" text-anchor="middle" class="t b">关节控制</text><text x="729" y="84" text-anchor="middle" class="s">伺服、真实限位</text>
<rect x="790" y="40" width="110" height="60" rx="8" class="nu" stroke-width="1.2"/><text x="845" y="64" text-anchor="middle" class="t b">物理世界</text><text x="845" y="84" text-anchor="middle" class="s">引擎计算</text>
<path d="M118 70L128 70" class="ar" marker-end="url(#m81011)"/><path d="M228 70L238 70" class="ar" marker-end="url(#m81011)"/><path d="M338 70L348 70" class="ar" marker-end="url(#m81011)"/><path d="M448 70L458 70" class="ar" marker-end="url(#m81011)"/><path d="M558 70L568 70" class="ar" marker-end="url(#m81011)"/><path d="M668 70L678 70" class="ar" marker-end="url(#m81011)"/><path d="M778 70L788 70" class="ar" marker-end="url(#m81011)"/>
<text x="20" y="138" class="t b">运动员（Athlete，高层）</text>
<rect x="20" y="146" width="538" height="30" rx="6" class="ag"/><text x="289" y="166" text-anchor="middle" class="t">agent 写的比赛程序</text>
<rect x="570" y="146" width="208" height="30" rx="6" class="pf"/><text x="674" y="166" text-anchor="middle" class="t">平台提供的技能</text>
<rect x="790" y="146" width="110" height="30" rx="6" class="sm"/><text x="845" y="166" text-anchor="middle" class="s">模拟器</text>
<text x="20" y="204" class="t b">工程师（Engineer，底层）</text>
<rect x="20" y="212" width="648" height="30" rx="6" class="ag"/><text x="344" y="232" text-anchor="middle" class="t">agent 写的比赛程序 + 自己实现的技能</text>
<rect x="680" y="212" width="98" height="30" rx="6" class="pf"/><text x="729" y="232" text-anchor="middle" class="s">关节接口+工具包</text>
<rect x="790" y="212" width="110" height="30" rx="6" class="sm"/><text x="845" y="232" text-anchor="middle" class="s">模拟器</text>
<rect x="20" y="262" width="430" height="96" rx="10" class="pl" stroke-width="1.3"/>
<text x="38" y="288" class="t b cpl">给动作：hit(t, pos, normal, vel)</text>
<text x="38" y="310" class="s">技能接收"球拍在什么时刻、什么位置、以什么朝向</text>
<text x="38" y="328" class="s">和速度击球"。预测和瞄准都由 agent 自己做；</text>
<text x="38" y="346" class="s">球最后落在哪里由物理决定。</text>
<rect x="470" y="262" width="430" height="96" rx="10" class="sp" stroke-width="1.3"/>
<text x="488" y="288" class="t b csp">给结果：hit(落点)</text>
<text x="488" y="310" class="s">平台要替 agent 预测球路、反算球拍状态、瞄准，</text>
<text x="488" y="328" class="s">预测和决策被平台做掉了，这部分能力就测不到。</text>
<text x="488" y="346" class="s">所以技能必须给动作，不给结果（P1）。</text>
</svg>

**图 5　控制链与分工（以乒乓为例）。绿色是 agent 负责的部分，灰色是平台提供的部分。运动员到"技能命令"为止；工程师还要自己实现技能。**

两条线逐项对照如下：

<svg viewBox="0 0 920 420" width="100%" role="img" aria-label="运动员和工程师两条线的检验对照：同一组问题，不同的检验方式" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.s{fill:var(--muted,#56665e);font-size:12px}.b{font-weight:600}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}</style>
<text x="185" y="40" text-anchor="middle" class="t b cco">运动员（Athlete，高层）</text>
<text x="460" y="40" text-anchor="middle" class="t b">问题</text>
<text x="735" y="40" text-anchor="middle" class="t b cto">工程师（Engineer，底层）</text>
<rect x="20" y="58" width="330" height="50" rx="8" class="co" stroke-width="1.2"/><text x="185" y="79" text-anchor="middle" class="t">运动技能 API，按 P1–P6 设计</text><text x="185" y="97" text-anchor="middle" class="s">给动作，不给结果；不替 agent 做决策</text>
<rect x="362" y="58" width="196" height="50" rx="8" class="gr" stroke-width="1.2"/><text x="460" y="88" text-anchor="middle" class="t b">平台给什么</text>
<rect x="570" y="58" width="330" height="50" rx="8" class="to" stroke-width="1.2"/><text x="735" y="79" text-anchor="middle" class="t">控制基础（五种方案选一）+ 工具包</text><text x="735" y="97" text-anchor="middle" class="s">按可行、不平凡、可扩展、可测量选</text>
<rect x="20" y="116" width="330" height="50" rx="8" class="co" stroke-width="1.2"/><text x="185" y="137" text-anchor="middle" class="t">参考选手只用平台技能</text><text x="185" y="155" text-anchor="middle" class="s">能赢最低一级的脚本对手</text>
<rect x="362" y="116" width="196" height="50" rx="8" class="gr" stroke-width="1.2"/><text x="460" y="146" text-anchor="middle" class="t b">能不能完成</text>
<rect x="570" y="116" width="330" height="50" rx="8" class="to" stroke-width="1.2"/><text x="735" y="137" text-anchor="middle" class="t">只用工具包写成的参考技能</text><text x="735" y="155" text-anchor="middle" class="s">达到技能测试的门槛</text>
<rect x="20" y="174" width="330" height="50" rx="8" class="co" stroke-width="1.2"/><text x="185" y="195" text-anchor="middle" class="t">简单程序必须输</text><text x="185" y="213" text-anchor="middle" class="s">不用核心技能拿不到分</text>
<rect x="362" y="174" width="196" height="50" rx="8" class="gr" stroke-width="1.2"/><text x="460" y="204" text-anchor="middle" class="t b">会不会太简单</text>
<rect x="570" y="174" width="330" height="50" rx="8" class="to" stroke-width="1.2"/><text x="735" y="195" text-anchor="middle" class="t">边界实现（只有 IK、不规划挥拍）</text><text x="735" y="213" text-anchor="middle" class="s">得分接近 0</text>
<rect x="20" y="232" width="330" height="50" rx="8" class="co" stroke-width="1.2"/><text x="185" y="253" text-anchor="middle" class="t">削弱参考选手的某项能力</text><text x="185" y="271" text-anchor="middle" class="s">成绩必须下降（能力敏感性）</text>
<rect x="362" y="232" width="196" height="50" rx="8" class="gr" stroke-width="1.2"/><text x="460" y="252" text-anchor="middle" class="t b">测到的是不是</text><text x="460" y="270" text-anchor="middle" class="t b">声称的能力</text>
<rect x="570" y="232" width="330" height="50" rx="8" class="to" stroke-width="1.2"/><text x="735" y="253" text-anchor="middle" class="t">隐藏请求测试 + 固定策略比赛</text><text x="735" y="271" text-anchor="middle" class="s">把技能从策略里单独分出来评</text>
<rect x="20" y="290" width="330" height="50" rx="8" class="co" stroke-width="1.2"/><text x="185" y="311" text-anchor="middle" class="t">技能由平台执行</text><text x="185" y="329" text-anchor="middle" class="s">不存在这个问题</text>
<rect x="362" y="290" width="196" height="50" rx="8" class="gr" stroke-width="1.2"/><text x="460" y="310" text-anchor="middle" class="t b">测的是不是</text><text x="460" y="328" text-anchor="middle" class="t b">真正用过的技能</text>
<rect x="570" y="290" width="330" height="50" rx="8" class="to" stroke-width="1.2"/><text x="735" y="311" text-anchor="middle" class="t">技能在环：比赛中的动作</text><text x="735" y="329" text-anchor="middle" class="s">必须经过 agent 自己的实现</text>
<rect x="20" y="348" width="330" height="50" rx="8" class="co" stroke-width="1.2"/><text x="185" y="369" text-anchor="middle" class="t">不同模型的比赛成绩拉得开</text><text x="185" y="387" text-anchor="middle" class="s">没有集中在 0 分或满分</text>
<rect x="362" y="348" width="196" height="50" rx="8" class="gr" stroke-width="1.2"/><text x="460" y="378" text-anchor="middle" class="t b">能不能区分 agent</text>
<rect x="570" y="348" width="330" height="50" rx="8" class="to" stroke-width="1.2"/><text x="735" y="369" text-anchor="middle" class="t">不同模型的技能测试分数拉得开</text><text x="735" y="387" text-anchor="middle" class="s">没有全部为 0</text>
</svg>

**图 6　两条线的对照。中间是同一组问题，两边是各自的检验方式。比如"会不会太简单"：运动员这条线要求简单程序必须输，工程师这条线要求只有 IK 的边界实现得分接近 0。**

#### 问题 5：平台给什么，边界在哪？

**为什么重要。** 平台给得太多，agent 没有东西可做，也就测不出能力；给得太少，任务根本完成不了。两条线的风险不同：
- **运动员**：平台负责把 agent 的命令变成关节动作。如果平台在这个过程中替 agent 选击球点、预测球路或瞄准，agent 的成绩里就有平台的功劳。
- **工程师**：只给关节接口最"干净"，但很多运动会因此做不了，比如腿式人形机器人连站稳都需要训练好的策略。而且训练好的策略不一定能拆开用：一个只观测下半身的行走策略看不到上半身挥拍带来的重心变化，把 agent 的手臂指令和它的腿部指令拼在一起，不能保证边走边挥拍不摔倒。

**我们的回答。**
- **运动员**：运动技能按六条原则 P1–P6 设计。核心是给动作、不给结果，不替 agent 做竞技决策，并写明控制器的权限和失败处理（第 6 节"高层：技能设计原则"）。
- **工程师**：分开两个变量：
  - 平台给多少控制基础：原始控制、机器人工具库、行走 + 独立操作、协调的全身控制、封闭的全身策略，五种方案选一；
  - 资源从哪里来：默认用固定工具包；让 agent 自己上网寻找和部署作为单独的实验条件，因为它会把资源能否获得、软件能否安装和运动控制混在一起；
  - 不论选哪种，都按四条标准检查：可行、不平凡、可扩展、可测量。标准统一，方案按机器人定：固定机械臂可能用工具库，人形机器人可能需要行走策略或全身控制（第 6 节"底层"各小节）。

**怎么检验。** 第 ③ 步的接口审查。

#### 问题 6：agent 用这些能不能完成比赛？

**为什么重要。** 平台给的东西不够用，任何 agent 都赢不了，成绩全是 0，测不出差别。但"能完成"在两种比赛格式里含义不同：
- **对战**（两个 agent 对打）：双方条件完全一样，输赢由双方水平决定。只需要保证比赛打得起来：只看画面、通过平台接口、在真实延迟下，能正常击球、形成回合。
- **对脚本对手**（单人赛）：对手是我们写的固定程序。还要保证最低一级对手是正常程序能赢的；否则所有 agent 都是 0 分，这一级分不出高下。这是对手标定的问题。

**我们的回答。** 先写一个能完成比赛的参考：
- **运动员**：参考选手只用平台技能，和 agent 用同样的接口和规则。它是"称职工程师水平"的程序，不是最优解；物理上的上界由第 ② 步的真机可行性检验给出。
- **工程师**：参考技能只用工具包里的东西写成，在技能测试中必须达到门槛。

参考选手的检验分两部分：
- **对手标定**（只针对脚本对手的格式）：用真值感知的参考选手跑大样本，确认它能赢最低一级、胜率随等级下降；再要一项"只看画面也能赢"的证据：相机参考选手赢下最低一级的多数，或者第 ⑤ 步里模型只看相机赢过。
- **端到端检查**（所有格式）：真机可行性检验保证物理上做得到，但它的上界选手知道真实状态、不走平台接口、不考虑延迟。"看得见、接口能用、来得及"这三点，由相机参考选手和模型试跑的比赛来证明：报告相机参考选手的合法回球率，明显偏低时查清原因，沿回退路径修改。不设单独的数值门槛。

**怎么检验。** 第 ④ 步的参考解检验，加上第 ⑤ 步的模型试跑。

#### 问题 7：会不会太简单？

**为什么重要。** 如果一个不懂这项运动的简单程序也能赢，成绩就不反映能力。例如羽毛球早期的最低一级对手，"永远打中路"的程序就能赢下全部比赛：对手的反应规则使得任何快球都直接得分，落点无关紧要。

**我们的回答。**
- **运动员**：
  - 简单程序（只打中路、只反射、固定动作、随机动作等）对每一级对手都必须输掉多数比赛；
  - 核心技能必须用得上。分两种情况：
    - **必需型**：不用这个技能就没有办法得分，例如乒乓、羽毛球不击球就不可能得分。检验：不用核心技能的程序（只发球、只移动）拿不到分。
    - **价值型**：这项运动本身就有别的正当得分办法，技能是其中一个决策。例如足球里带球走进门和射门都是正当打法，但踢出去的球更快、更难拦截。检验：在同一组种子上，会用这个技能的参考选手，在每个正式等级都要明显强于"决策相同、但从不用这个技能"的版本。放着技能不用会明显吃亏，agent 就不会因为不用技能而拿到好成绩；同时也不用为此改规则书。
- **工程师**：边界实现（例如只有 IK、不规划挥拍）在技能测试中得分接近 0，说明工具包本身没有把技能做完。

**怎么检验。** 第 ④ 步的技能必要性和简单基线检验。

#### 问题 8：成绩反映的是声称的能力吗？

**为什么重要。** 一项运动声称考察九项能力中的几项，但比赛只给出胜负和比分。

**只看成功率，无法判断能力。** 成绩好，可能是因为看得准、算得准、反应快，也可能是策略好；同样的胜率背后，可能是完全不同的能力组合。两个例子：
- 乒乓：把参考选手分别削弱一项能力，只看胜分率，认出被削弱的是哪一项，准确率只有 35%（五类，随机猜是 20%）；
- 冰壶：会根据反馈校准冰面摩擦的参考选手，和不校准的版本胜率相同。只看胜率，这项能力完全看不出来。

**但也不能为了测量而改变比赛。** 如果在比赛里插入结构化的环节，例如要求 agent 先报告它估计的球位，就会限制 agent 写程序的自由，测到的也就不再是原来的东西。

**其他 benchmark 的做法：**
- EmbodiedBench 按能力设计任务子集；
- LIBERO-PRO 改变外部条件，看成绩怎样变化；
- AgentBoard 记录过程中完成了哪些步骤；
- BEHAVIOR-1K 由环境评价结果，不规定 agent 怎样实现。

共同点是：不只凭最终成绩还原能力，而是结合任务设计、过程记录和受控对照，逐步建立证据。**结构化的是评测器，不是 agent。**

**我们的回答。** 九项能力是诊断分类，不是九个独立的分数。我们不承诺每一项都能单独测出来，而是用四种方法逐步建立证据。

1. **发布前：能力敏感性（第 ④ 步）。** 把参考选手声称考察的某项能力故意削弱，例如给球位加噪声、去掉空气阻力、不补偿延迟、不看对手，成绩必须下降。下降说明这项能力确实影响成绩；不下降，就不能声称考察它。
2. **每场比赛：记录过程指标，事后分析（行为诊断）。** 评测器在模拟器一侧记录，不改变比赛，也不要求 agent 输出任何额外信息。记录三类东西，每个指标用于报告前都要先过效度检验：
   - **过程事件**：每个回合的事件链，例如触球 → 过网 → 落台 → 得分，看失分停在哪一环；
   - **命令里隐含的判断**：技能给的是动作，所以每条命令都包含 agent 的预测。`hit(t, pos, …)` 里的 t 和 pos 就是它对"球何时在何处"的判断，和真实球路一比就得到预测误差。
     - 同一个来球的几条命令来回跳动，对应感知（A1）；
     - 朝同一方向偏移，对应物理模型（A2）；
     - 最后一条命令的时间偏差，对应延迟补偿（A4）。
   - **反事实重放**：模拟器是确定的，可以从任意时刻分叉重放，例如改成"完美执行""晚 50 ms 执行""换一个击球目标"，判断失分来自预测、执行还是决策；
   - **效度检验**：每个指标要先确认它只对对应的那项能力敏感，才能用于报告。乒乓中，只用通过检验的指标，单场认出被削弱能力的准确率是 81%，只看胜分率是 35%。
3. **研究分析：受控实验。** 每次只改变一个条件，其余相同，用配对种子比较。可以改的条件包括隐藏的物理参数、慢放倍数、相机条件、队内通信、对手风格、能否修改程序。
4. **工程师：技能单独评测。** 用隐藏请求测试和"我们固定的参考策略 + agent 的技能"打比赛，把技能的好坏从策略里分出来。

正式成绩只来自比赛结果；过程指标和受控实验用于分析，不进入正式成绩。哪些能力已经能测、哪些暂时测不了，按证据如实报告（第 9.5 节）。

<svg viewBox="0 0 920 330" width="100%" role="img" aria-label="从一场比赛到能力证据：结果、过程记录、受控实验三条线；乒乓中加上过程指标后认出被削弱能力的准确率从 35% 提高到 81%" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.s{fill:var(--muted,#56665e);font-size:12px}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.cpl{fill:rgb(62,150,106)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}.csp{fill:rgb(206,112,58)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.bg{fill:rgba(140,140,140,.35)}.bp{fill:rgba(76,170,124,.55)}.ax{stroke:var(--line,#d7e0db);stroke-width:1}.th{stroke:rgb(206,112,58);stroke-width:1.4;stroke-dasharray:4 3}</style>
<defs><marker id="m81031" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<rect x="20" y="40" width="130" height="250" rx="8" class="gr" stroke-width="1.3"/>
<text x="85" y="70" text-anchor="middle" class="t b">一场比赛</text>
<text x="85" y="104" text-anchor="middle" class="s">agent 自由写程序</text>
<text x="85" y="126" text-anchor="middle" class="s">评测器在模拟器</text>
<text x="85" y="144" text-anchor="middle" class="s">一侧记录</text>
<text x="85" y="176" text-anchor="middle" class="s">不要求 agent</text>
<text x="85" y="194" text-anchor="middle" class="s">额外输出</text>
<rect x="170" y="40" width="190" height="70" rx="8" class="pl" stroke-width="1.3"/>
<text x="265" y="68" text-anchor="middle" class="t b cpl">结果</text>
<text x="265" y="90" text-anchor="middle" class="s">胜负、比分、得分率</text>
<rect x="380" y="40" width="210" height="70" rx="8" class="nu" stroke-width="1.3"/>
<text x="485" y="68" text-anchor="middle" class="t b">正式成绩（排名）</text>
<text x="485" y="90" text-anchor="middle" class="s">说明好不好，不说明为什么</text>
<rect x="170" y="130" width="190" height="70" rx="8" class="co" stroke-width="1.3"/>
<text x="265" y="154" text-anchor="middle" class="t b cco">过程记录</text>
<text x="265" y="174" text-anchor="middle" class="s">事件链、命令里的预测</text>
<text x="265" y="191" text-anchor="middle" class="s">反事实重放</text>
<rect x="380" y="130" width="210" height="70" rx="8" class="nu" stroke-width="1.3"/>
<text x="485" y="154" text-anchor="middle" class="t b">过程指标（过效度检验）</text>
<text x="485" y="174" text-anchor="middle" class="s">预测误差 → A2，抖动 → A1</text>
<text x="485" y="191" text-anchor="middle" class="s">时间偏差 → A4，失分环节</text>
<rect x="170" y="220" width="190" height="70" rx="8" class="to" stroke-width="1.3"/>
<text x="265" y="244" text-anchor="middle" class="t b cto">受控实验</text>
<text x="265" y="264" text-anchor="middle" class="s">只改一个条件</text>
<text x="265" y="281" text-anchor="middle" class="s">配对种子比较</text>
<rect x="380" y="220" width="210" height="70" rx="8" class="nu" stroke-width="1.3"/>
<text x="485" y="244" text-anchor="middle" class="t b">条件对成绩的影响</text>
<text x="485" y="264" text-anchor="middle" class="s">隐藏参数 → A2，慢放 → A4</text>
<text x="485" y="281" text-anchor="middle" class="s">队内通信 → B3</text>
<path d="M151 75L168 75" class="ar" marker-end="url(#m81031)"/><path d="M151 165L168 165" class="ar" marker-end="url(#m81031)"/><path d="M151 255L168 255" class="ar" marker-end="url(#m81031)"/>
<path d="M361 75L378 75" class="ar" marker-end="url(#m81031)"/><path d="M361 165L378 165" class="ar" marker-end="url(#m81031)"/><path d="M361 255L378 255" class="ar" marker-end="url(#m81031)"/>
<rect x="610" y="40" width="290" height="250" rx="8" class="nu" stroke-width="1.3"/>
<text x="755" y="66" text-anchor="middle" class="t b">乒乓：单场认出被削弱的能力</text>
<text x="755" y="86" text-anchor="middle" class="s">五类，每类一个只削弱一项的参考选手</text>
<text x="630" y="122" class="t">只看胜分率</text>
<rect x="630" y="130" width="85" height="24" rx="3" class="bg"/>
<text x="721" y="147" class="t">35%</text>
<text x="630" y="190" class="t">加上过程指标</text>
<rect x="630" y="198" width="198" height="24" rx="3" class="bp"/>
<text x="834" y="215" class="t b cpl">81%</text>
<line x1="630" y1="236" x2="875" y2="236" class="ax"/>
<text x="630" y="254" text-anchor="middle" class="s">0</text>
<text x="875" y="254" text-anchor="middle" class="s">100%</text>
<line x1="679" y1="127" x2="679" y2="158" class="th"/><line x1="679" y1="195" x2="679" y2="236" class="th"/>
<text x="679" y="272" text-anchor="middle" class="s csp">随机猜 20%</text>
</svg>

**图 7　从一场比赛到能力证据。左：评测器从同一场比赛得到三类证据，agent 不需要做任何额外的事；只有"结果"进入正式成绩。右：乒乓的效度检验中，只看胜分率很难认出是哪一项能力出了问题，加上通过检验的过程指标后准确率明显提高。**

**怎么检验。** 第 ④ 步的能力敏感性检验；第 ⑥ 步的正式评测（第 9 节）。

#### 问题 9：测的是比赛中真正用过的技能吗？（工程师）

**为什么重要。** 如果赛后评测的技能实现和比赛中实际使用的不是同一份，agent 可以另交一份专门应付测试的实现，技能测试就和比赛脱节。

**我们的回答。** 技能在环：比赛中 agent 的动作必须经过它自己的技能实现；赛后评测的就是这份实现（第 6 节"技能在环"）。运动员这条线的技能由平台执行，不存在这个问题。

**怎么检验。** 第 ③ 步设计；第 ⑥ 步赛后评测。

#### 问题 10：能区分不同的 agent 吗？

**为什么重要。** 所有模型都是 0 分，或者都是满分，排名就没有意义。参考选手是我们打磨过的程序，模型写的第一版程序往往远不如它；只有让真实的 agent 上场，才知道分数是否拉得开，环境里有没有被它们找到的捷径。

**我们的回答。** 少量前沿模型实际比赛：
- **运动员**看比赛成绩；
- **工程师**看技能测试分数。

分数不能集中在 0 分或满分；发现捷径就修复，并回到第 ④ 步重新检验。

**怎么检验。** 第 ⑤ 步的区分度检验。

### 共同的收尾

#### 问题 11：结果是否可信、可复现、公平？

**为什么重要。** 结果不能复现，就无法比较；环境有漏洞，agent 可以绕过比赛本身取胜；评测中途按某个模型的表现修改配置，成绩就不公平。

**我们的回答。**
- **可复现**：同一程序、同一种子，比赛逐位相同；用标准回放检查。
- **公平**：对战交换先后手和场地；脚本对手和 agent 遵守同样的物理限制。
- **安全**：选手程序在隔离的容器里运行，看不到对手、种子和评测代码。
- **冻结**：正式评测前冻结机器人、场地、接口、对手、种子和计分，对所有 agent 相同；冻结后不按任何模型的成绩修改，修改即发布新版本。
- **统计**：用足够多的种子和置信区间报告成绩。

**怎么检验。** 第 ④ 步的工程、公平和安全检验；第 ⑥ 步冻结。

### 问题和步骤的对应

<svg viewBox="0 0 920 498" width="100%" role="img" aria-label="十一个问题分别在六个步骤中的哪一步检验" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.s{fill:var(--muted,#56665e);font-size:12px}.b{font-weight:600}.cl{fill:rgba(140,140,140,.06)}.rl{stroke:var(--line,#d7e0db);stroke-width:1}.dsp{fill:rgb(206,112,58)}.dco{fill:rgb(78,124,196)}.dpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}.cpl{fill:rgb(62,150,106)}</style>
<rect x="480" y="20" width="70" height="470" class="cl"/><rect x="620" y="20" width="70" height="470" class="cl"/><rect x="760" y="20" width="70" height="470" class="cl"/>
<text x="515" y="44" text-anchor="middle" class="t b">①</text><text x="515" y="62" text-anchor="middle" class="s">定义运动</text>
<text x="585" y="44" text-anchor="middle" class="t b">②</text><text x="585" y="62" text-anchor="middle" class="s">机器人场地</text>
<text x="655" y="44" text-anchor="middle" class="t b">③</text><text x="655" y="62" text-anchor="middle" class="s">设计接口</text>
<text x="725" y="44" text-anchor="middle" class="t b">④</text><text x="725" y="62" text-anchor="middle" class="s">无模型验证</text>
<text x="795" y="44" text-anchor="middle" class="t b">⑤</text><text x="795" y="62" text-anchor="middle" class="s">模型试跑</text>
<text x="865" y="44" text-anchor="middle" class="t b">⑥</text><text x="865" y="62" text-anchor="middle" class="s">冻结与评测</text>
<line x1="20" y1="74" x2="900" y2="74" class="rl"/>
<text x="20" y="94" class="t b csp">共同的底座</text>
<text x="34" y="119" class="t">1　测什么？</text>
<text x="34" y="149" class="t">2　这项运动适合哪种角色？</text>
<text x="34" y="179" class="t">3　环境是否真实？</text>
<text x="34" y="209" class="t">4　物理上能不能完成？</text>
<circle cx="515" cy="115" r="6" class="dsp"/>
<circle cx="515" cy="145" r="6" class="dsp"/><circle cx="655" cy="145" r="6" class="dsp"/>
<circle cx="585" cy="175" r="6" class="dsp"/>
<circle cx="585" cy="205" r="6" class="dsp"/><circle cx="725" cy="205" r="6" class="dsp"/>
<line x1="20" y1="222" x2="900" y2="222" class="rl"/>
<text x="20" y="242" class="t b cco">两条线（运动员、工程师各回答一遍）</text>
<text x="34" y="267" class="t">5　平台给什么，边界在哪？</text>
<text x="34" y="297" class="t">6　agent 用这些能不能完成比赛？</text>
<text x="34" y="327" class="t">7　会不会太简单？</text>
<text x="34" y="357" class="t">8　成绩反映的是声称的能力吗？</text>
<text x="34" y="387" class="t">9　测的是真正用过的技能吗？（工程师）</text>
<text x="34" y="417" class="t">10　能区分不同的 agent 吗？</text>
<circle cx="655" cy="263" r="6" class="dco"/>
<circle cx="725" cy="293" r="6" class="dco"/>
<circle cx="725" cy="323" r="6" class="dco"/>
<circle cx="725" cy="353" r="6" class="dco"/><circle cx="865" cy="353" r="6" class="dco"/>
<circle cx="655" cy="383" r="6" class="dco"/><circle cx="865" cy="383" r="6" class="dco"/>
<circle cx="795" cy="413" r="6" class="dco"/>
<line x1="20" y1="430" x2="900" y2="430" class="rl"/>
<text x="20" y="450" class="t b cpl">共同的收尾</text>
<text x="34" y="475" class="t">11　结果是否可信、可复现、公平？</text>
<circle cx="725" cy="471" r="6" class="dpl"/><circle cx="865" cy="471" r="6" class="dpl"/>
</svg>

**图 8　每个问题在哪一步检验。每一步都对应至少一个问题：步骤是为回答问题而设的。**

## 构建流程

<svg viewBox="0 0 920 462" width="100%" role="img" aria-label="一项运动的构建流程：共同的定义运动、机器人与场地两步，在设计接口处分成运动员和工程师两条线，各自经过无模型验证和模型试跑，最后一起冻结" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.s{fill:var(--muted,#56665e);font-size:12px}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.bk{stroke:rgb(206,112,58);stroke-width:1.3;fill:none;stroke-dasharray:5 4}.fz{stroke:rgb(78,124,196);stroke-width:1.6;fill:none;stroke-dasharray:3 4}.cb{fill:rgb(206,112,58);font-size:12px}</style>
<defs><marker id="m81021" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker><marker id="m81022" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="rgb(206,112,58)"/></marker></defs>
<text x="20" y="28" class="s">每一步：做什么 → 这一步的检验（白框）。第 ③–⑤ 步，运动员和工程师两条线各走一遍。</text>
<rect x="20" y="50" width="110" height="280" rx="8" class="gr" stroke-width="1.3"/>
<text x="75" y="76" text-anchor="middle" class="t b">起点：候选</text>
<text x="75" y="110" text-anchor="middle" class="s">机器人、场地</text>
<text x="75" y="132" text-anchor="middle" class="s">角色、技能集合</text>
<text x="75" y="154" text-anchor="middle" class="s">对手、计分</text>
<text x="75" y="176" text-anchor="middle" class="s">已有结果剪枝</text>
<rect x="145" y="50" width="110" height="280" rx="8" class="sp" stroke-width="1.3"/>
<text x="200" y="76" text-anchor="middle" class="t b">① 定义运动</text>
<text x="200" y="110" text-anchor="middle" class="s">核心挑战</text>
<text x="200" y="132" text-anchor="middle" class="s">规则与偏离</text>
<text x="200" y="154" text-anchor="middle" class="s">考察的能力</text>
<text x="200" y="176" text-anchor="middle" class="s">设哪种角色</text>
<text x="200" y="198" text-anchor="middle" class="s">对手、种子</text>
<rect x="153" y="286" width="94" height="30" rx="6" class="nu" stroke-width="1.1"/>
<text x="200" y="306" text-anchor="middle" class="t b csp">入选条件</text>
<rect x="270" y="50" width="110" height="280" rx="8" class="sp" stroke-width="1.3"/>
<text x="325" y="76" text-anchor="middle" class="t b">② 机器人场地</text>
<text x="325" y="110" text-anchor="middle" class="s">真实限位</text>
<text x="325" y="132" text-anchor="middle" class="s">真实对应</text>
<text x="325" y="154" text-anchor="middle" class="s">真机可行</text>
<text x="325" y="176" text-anchor="middle" class="s">物理一致</text>
<rect x="278" y="286" width="94" height="30" rx="6" class="nu" stroke-width="1.1"/>
<text x="325" y="306" text-anchor="middle" class="t b csp">物理真实性</text>
<rect x="395" y="50" width="110" height="130" rx="8" class="co" stroke-width="1.3"/>
<text x="450" y="74" text-anchor="middle" class="t b cco">③ 运动员接口</text>
<text x="450" y="98" text-anchor="middle" class="s">运动技能 API</text>
<text x="450" y="118" text-anchor="middle" class="s">P1–P6</text>
<rect x="403" y="138" width="94" height="30" rx="6" class="nu" stroke-width="1.1"/>
<text x="450" y="158" text-anchor="middle" class="t b cco">接口审查</text>
<rect x="520" y="50" width="110" height="130" rx="8" class="co" stroke-width="1.3"/>
<text x="575" y="74" text-anchor="middle" class="t b cco">④ 无模型验证</text>
<text x="575" y="98" text-anchor="middle" class="s">参考选手、基线</text>
<text x="575" y="118" text-anchor="middle" class="s">能力敏感性</text>
<rect x="528" y="138" width="94" height="30" rx="6" class="nu" stroke-width="1.1"/>
<text x="575" y="158" text-anchor="middle" class="t b cco">六项检验</text>
<rect x="645" y="50" width="110" height="130" rx="8" class="co" stroke-width="1.3"/>
<text x="700" y="74" text-anchor="middle" class="t b cco">⑤ 模型试跑</text>
<text x="700" y="98" text-anchor="middle" class="s">比赛成绩</text>
<text x="700" y="118" text-anchor="middle" class="s">捷径与漏洞</text>
<rect x="653" y="138" width="94" height="30" rx="6" class="nu" stroke-width="1.1"/>
<text x="700" y="158" text-anchor="middle" class="t b cco">区分度</text>
<rect x="395" y="200" width="110" height="130" rx="8" class="to" stroke-width="1.3"/>
<text x="450" y="224" text-anchor="middle" class="t b cto">③ 工程师接口</text>
<text x="450" y="248" text-anchor="middle" class="s">控制基础、工具包</text>
<text x="450" y="268" text-anchor="middle" class="s">技能在环</text>
<rect x="403" y="288" width="94" height="30" rx="6" class="nu" stroke-width="1.1"/>
<text x="450" y="308" text-anchor="middle" class="t b cto">四条标准</text>
<rect x="520" y="200" width="110" height="130" rx="8" class="to" stroke-width="1.3"/>
<text x="575" y="224" text-anchor="middle" class="t b cto">④ 无模型验证</text>
<text x="575" y="248" text-anchor="middle" class="s">参考技能</text>
<text x="575" y="268" text-anchor="middle" class="s">边界实现</text>
<rect x="528" y="288" width="94" height="30" rx="6" class="nu" stroke-width="1.1"/>
<text x="575" y="308" text-anchor="middle" class="t b cto">技能测试门槛</text>
<rect x="645" y="200" width="110" height="130" rx="8" class="to" stroke-width="1.3"/>
<text x="700" y="224" text-anchor="middle" class="t b cto">⑤ 模型试跑</text>
<text x="700" y="248" text-anchor="middle" class="s">技能测试分数</text>
<text x="700" y="268" text-anchor="middle" class="s">比赛成绩</text>
<rect x="653" y="288" width="94" height="30" rx="6" class="nu" stroke-width="1.1"/>
<text x="700" y="308" text-anchor="middle" class="t b cto">区分度</text>
<rect x="790" y="50" width="110" height="280" rx="8" class="pl" stroke-width="1.3"/>
<text x="845" y="76" text-anchor="middle" class="t b">⑥ 冻结与评测</text>
<text x="845" y="110" text-anchor="middle" class="s">记录全部版本</text>
<text x="845" y="132" text-anchor="middle" class="s">保密种子集</text>
<text x="845" y="154" text-anchor="middle" class="s">正式比赛</text>
<text x="845" y="176" text-anchor="middle" class="s">技能测试</text>
<text x="845" y="198" text-anchor="middle" class="s">诊断与实验</text>
<rect x="798" y="286" width="94" height="30" rx="6" class="nu" stroke-width="1.1"/>
<text x="845" y="306" text-anchor="middle" class="t b cpl">冻结</text>
<path d="M131 190L143 190" class="ar" marker-end="url(#m81021)"/>
<path d="M256 190L268 190" class="ar" marker-end="url(#m81021)"/>
<path d="M381 190L387 190L387 115L393 115" class="ar" marker-end="url(#m81021)"/>
<path d="M387 190L387 265L393 265" class="ar" marker-end="url(#m81021)"/>
<path d="M506 115L518 115" class="ar" marker-end="url(#m81021)"/>
<path d="M631 115L643 115" class="ar" marker-end="url(#m81021)"/>
<path d="M756 115L788 115" class="ar" marker-end="url(#m81021)"/>
<path d="M506 265L518 265" class="ar" marker-end="url(#m81021)"/>
<path d="M631 265L643 265" class="ar" marker-end="url(#m81021)"/>
<path d="M756 265L788 265" class="ar" marker-end="url(#m81021)"/>
<path d="M772 38L772 342" class="fz"/>
<path d="M325 331L325 358L75 358L75 333" class="bk" marker-end="url(#m81022)"/>
<text x="200" y="376" text-anchor="middle" class="cb">不可行：换机器人或场地</text>
<path d="M575 331L575 396L450 396L450 333" class="bk" marker-end="url(#m81022)"/>
<text x="512" y="414" text-anchor="middle" class="cb">不通过：改接口、对手或计分</text>
<path d="M700 331L700 432L610 432L610 333" class="bk" marker-end="url(#m81022)"/>
<text x="655" y="450" text-anchor="middle" class="cb">无区分度：调难度或修漏洞</text>
<text x="900" y="360" text-anchor="end" class="s">蓝色虚线：冻结</text>
<text x="900" y="379" text-anchor="end" class="s">冻结后不按成绩修改；</text>
<text x="900" y="398" text-anchor="end" class="s">修改即发布新版本，</text>
<text x="900" y="417" text-anchor="end" class="s">重新通过第 ④、⑤ 步</text>
</svg>

**图 9　一项运动的构建流程。前两步两种角色共用；第 ③ 步分成运动员（蓝）和工程师（紫）两条线，各自经过第 ④、⑤ 步；第 ⑥ 步一起冻结。橙色虚线是开发阶段的回退路径（两条线相同，图中只画在下方）；蓝色虚线之后配置冻结。只设一种角色的运动只走其中一条线。**

一项运动从候选到正式比赛，依次经过六个步骤。每一步结束时有一项检验，不通过就沿回退路径修改，再重新检验。

| 步骤 | 做什么 | 回答第 2 节的哪些问题 | 检验 |
|---|---|---|---|
| ① 定义运动 | 确定要保留的核心挑战、规则、考察的能力、设哪种角色，以及对手、种子和计分 | 1 测什么、2 适合哪种角色 | 入选条件 |
| ② 机器人与场地 | 选择真实存在的机器人和场地，确认真实条件下能完成这项运动 | 3 环境是否真实、4 物理上能不能完成 | 物理真实性检验 |
| ③ 设计接口 | 运动员：设计运动技能；工程师：选择控制基础和工具包、设计技能在环；两者都要定观测和时间模型 | 2 适合哪种角色、5 平台给什么、9 真正用过的技能 | 接口审查 |
| ④ 无模型验证 | 用参考选手、参考技能和脚本程序检验 | 4 能不能完成、6 能否完成比赛、7 会不会太简单、8 成绩是否反映能力、11 可信与公平 | 六项检验 |
| ⑤ 模型试跑 | 少量前沿模型实际比赛 | 10 能否区分 agent | 区分度检验 |
| ⑥ 冻结与正式评测 | 冻结全部版本，进行正式比赛、技能测试、诊断和受控实验 | 8 三个证据层次、9 赛后技能测试、11 冻结 | — |

### 为什么这样安排

- **先搜索，再验证**：
  - 一项运动的配置是在"机器人 × 场地 × 接口 × 对手"中搜索出来的；
  - 起点是列出候选，已有的实验结果（包括失败的结果）用来排除不可行的组合；
  - 候选不只是机器人和场地，也包括设哪种模式、技能怎样划分；这些可以让 agent 参与探索；
  - 一项运动找到一套通过全部检验的配置就可以进入 benchmark，找到多套更好。
- **每项运动由负责人审核**：负责人按这个流程做出选择，并把结论和证据写进筛选报告。
- **机器人和场地一起选**：两者互相决定，所以放在同一步（见第 2 节问题 4）。
- **先低成本，后高成本**：
  - 第 ①–④ 步只用参考选手和脚本，不需要模型；
  - 第 ⑤ 步才运行模型，而且只在前面全部通过之后。
- **开发期可以回退，冻结后不再修改**：
  - 开发期间可以沿回退路径反复调整；
  - 冻结后不根据任何模型的成绩调整配置，修改即发布新版本，重新通过第 ④、⑤ 步。
- **统一流程，不统一配置**：不同运动可以用不同的机器人、控制方式和场地；统一的是能力目标、原则和这六个步骤的检验。

## 第一步：定义运动

**这一步回答**：第 2 节的问题 1（测什么）和问题 2（适合哪种角色）。

这一步确定这项运动"考什么"。后面的每一步都以这里写下的核心挑战和能力声明为依据。

### 这一步确定什么

| 项 | 内容 |
|---|---|
| 核心挑战 | 这项运动必须保留、不能被平台或规则简化掉的挑战，例如乒乓中预测带旋转的来球并在正确的时刻以正确的拍面击球 |
| 规则 | 官方规则书和版本；每条偏差编号，注明对应条文；影响胜负的偏差写进给 agent 的说明 |
| 考察的能力 | 这项运动声称考察的能力（能力分类见第 9.2 节）；第 ④ 步会逐项检验 |
| 可调整的参数 | 哪些环境参数允许为机器人调整（场地尺寸、球、规则细节），哪些不能动 |
| 对手、种子和计分 | 脚本对手的等级划分；比赛种子的分布；计分方式 |

### 为什么对手、种子和计分也在这一步定

对 21 项运动的审核发现，技能接口本身基本合格，比赛不能区分能力的原因集中在三处：
- 脚本对手分级失效，简单策略就能战胜高等级对手；
- 官方种子公开且固定，最优动作可以记住；
- 二值成功线贴近能力上界，强弱选手都能过线。

所以对手、种子和计分要和机器人一样，从一开始就设计，并在最后一起冻结：
- **对手**：等级单调，并且对 agent 的落点和战术有反应；
- **种子**：正式评测用保密的多种子集；
- **计分**：单人项目用连续分，不用贴近上界的二值成功线。

### 入选条件

五项都满足才继续：
1. 物理引擎能可信地模拟这项运动；
2. 成绩不只由底层控制器决定，agent 有判断可做；
3. 只凭允许的感知（相机画面、自身状态、裁判信号）就能比赛；
4. 有官方规则书，裁判能按规则判罚；
5. 能做出参考选手。

## 第二步：选择机器人与场地

**这一步回答**：第 2 节的问题 3（环境是否真实）和问题 4（物理上能不能完成）。

这一步确定使用哪个机器人和装配配置（机器人 + 手或工具 + 控制器），以及使用标准场地还是适配场地。

### 为什么物理真实性是底线

**要求**：模拟器里允许的每一个动作，真实的机器人或机器也必须做得到。比赛要在物理上真实，而不只是看起来真实。否则 agent 学到的打法依赖于真实世界不存在的能力，成绩就失去意义。

物理和机器人是底线，运动可以适配。场地、机器人和控制辅助是三件不同的事，要分开决定：真实尺度不等于困难，缩小场地也不等于简单。

### 六条原则

| 原则 | 要求 | 例子 |
|---|---|---|
| 真实限位 | 关节位置、速度、加速度、加加速度、力矩和力矩变化率都来自厂商 SDK、数据手册或 URDF（不一致时以 SDK 为准），由共享的限位和伺服模块执行，并且逐项检查，不只检查速度；不允许不设限；推算的值标为估计 | Franka 的关节限位；宇树电机的扭矩-转速曲线 |
| 有真实对应 | 每个机器人或出手机构都对应一台真实存在的机器，参数在它公开的范围内 | 冰壶对应冰壶机器人 Curly；保龄球对应 USBC 的出球机 E.A.R.L. |
| 真机可行 | 上界选手（知道真实状态、搜索全部合法动作）在真实限位和真实接触规律下能完成这项运动的基本动作，并且能出现真实的回合；物理修正后要重做 | 球拍类：合法发球的可回率 ≥ 90% |
| 改设定，不改物理 | 运动在真实限位下不可行时，缩小场地、换球或调整规则（如同国际网联为少儿网球缩小场地、换低压球），不放宽机器人 | 羽毛球采用标准尺寸 0.4 倍的场地 |
| 真实的感知和时间 | 只给相机画面，不给特权状态；控制频率和延迟与真机一致 | 观测按白名单提供；动作有生效延迟 |
| 物理一致 | 没有自身穿透（手臂不能穿过躯干或底座）；接触不能被全局关闭；机器人之间有接触 | 自碰撞检查覆盖手臂 × 躯干、左右臂、器具 × 身体、腿 × 腿 |

这六条对平台提供的技能和策略同样适用。

### 检验

| 检验 | 方法 | 通过标准 |
|---|---|---|
| 真实限位 | 关节位置、速度、加速度、力矩等每个限值都有出处（厂商 SDK 优先于 URDF），由共享的限位模块执行；平台提供的技能和行走策略同样受限 | 全部有出处并执行 |
| 真实对应 | 核对机器人或出手机构对应的真实机器，以及参数是否在其公开范围内 | 有对应的真实机器，参数在范围内；没有的写明是估计 |
| 真机可行性 | 上界选手（知道真实状态、搜索全部可行动作）在真实限位下完成这项运动的基本动作。只检验物理任务（接发球、防守、空门射门等）；上界选手对脚本对手的胜率属于第 ④ 步的对手标定，不算在这里 | 达到运动的门槛；球拍类为合法发球的可回率 ≥ 90% |
| 物理一致 | 物理真实性审计（第 5.4 节）：自身穿透、接触是否被关闭、出手速度是否有真实机器对应 | 没有"过宽"的发现；"偏严"的发现已声明 |

**不通过**：更换机器人或场地，或者回到第 ① 步重新考虑适配方式。

适配场地是正当的选择，但必须声明。例：真实物理下，机械臂在标准羽毛球场上打不出后场球，因此羽毛球采用标准尺寸 0.4 倍的场地，并在规则偏差中写明。

### 物理真实性审计

审计检查一件事：模拟器是否允许真实机器做不到的动作。第一次审计由 Changhe 完成（2026-10-09），对象是当时的主分支和自碰撞改动（PR #44）。

**范围**：
- **自碰撞**：人形机器人和机械臂；
- **真实限位**：网球、无人机、赛车；
- **出手机构的速度上限**：飞镖、飞盘、桌上足球、棒球、保龄球；
- 其他不真实的行为也在范围内。

**每条发现给出两个数**：
- 仿真中的实测值，并说明测量方法；
- 真实机器的值，来自厂商规格或论文，并注明出处。

没有出处的数值一律标为"估计"。

**方法**：
- **两轮核验**：先由一个独立的 agent 做第一轮审查，再由人工在代码和同版本的 MuJoCo 中复核最重要的结论。
- **自碰撞**：
  - 在检查模型中打开同一机器人所有几何体之间的接触，保留父子关节的过滤；
  - 每 10 ms 把真实比赛中的姿态放进模型做碰撞检测，记录每对部件的最大穿透深度（凸包深度）；
  - 覆盖全部标准回放比赛、新跑的比赛和 1000 个随机合法姿态；
  - 初始姿态就已重叠的部件对视为建模造成的结构重叠，予以扣除；
  - 区分两种情况：在真实比赛中发生，或只在随机姿态中可能发生。
- **限位**：
  - 从比赛记录中按 1 ms 统计关节速度、加速度、加加速度和力矩变化率，与机器人的规格比较；
  - 导轨和底盘的速度、加速度与市售产品比较。
- **出手机构**：出手速度与对应真实机器的公开参数比较。
- **仿真 ÷ 真实的比值**：把每个量画在对数尺度上，大于 1 表示仿真允许的比真实更多。

**分级与处理**：

| 级别 | 含义 | 处理 |
|---|---|---|
| 过宽 | 模拟器允许真实机器做不到的事 | 修正：执行限位、截速、打开接触；或者写成声明的偏差，并评估对成绩的影响 |
| 未知 | 需要精细网格或真机测试才能判断 | 补充核验 |
| 偏严 | 比真实机器保守，或和规则规定的尺寸不一致 | 比赛仍可进行，写明声明 |
| 符合 | 在真实机器的能力范围内 | — |

**第一次审计的结果**：共 39 条，其中过宽 15 条、未知 4 条、偏严 3 条、符合 17 条。

| 运动或部分 | 级别 | 发现 | 建议 |
|---|---|---|---|
| 羽毛球 | 过宽，在真实比赛中发生 | 平台自己的击球规划让手腕和拍柄穿进 1.0 m 高的底座，最深 15 cm；5 场 97 拍中 21 拍穿透超过 1 cm。场景关闭了接触，所以一直没被发现 | 打开接触，采用自碰撞检查，改击球路径，重做真机可行性检验 |
| 网球（球拍） | 过宽，未声明 | 刚性拍面加增益使出球速度达到拍头速度的 9.1 倍，真实球拍约 1.4 倍；换成真实球拍规律后，23 次标准击球没有一次过网；底层抽击可达 87–94 m/s，人类发球纪录为 73 m/s | 写成声明的偏差，接触时的拍头速度截到 3.2 m/s；长期改用真实球拍规律和少儿网球的小场地 |
| 网球（手臂和导轨） | 过宽 | 只检查了每 10 ms 的目标速度：手臂加速度为限值的 2.9 倍，加加速度 3.2 倍，力矩变化率 2.9 倍；导轨声明 4 m/s、8 m/s²，没有对应产品，底层实测达到 6.15 m/s、37.7 m/s² | 改用执行全部限位的共享伺服，增加加加速度检查，在长行程慢导轨和小场地之间二选一 |
| 网球（真机可行性） | 过宽 | 上界接发 98.8% 只是因为球拍增益；换成真实球拍，机械臂达不到标准场地的最低合法发球速度 | 在真实接触规律和小场地上重做检验 |
| 资源库人形机器人 | 过宽 | G1 的行走策略在指令极限时两脚互相穿过最深 5.8 cm（H1-2 为 2.1 cm），真机不可能出现；两台 G1 相距 0.15 m 时有 50 对几何体重叠，但接触为 0 | 增加左右腿之间的碰撞对，打开机器人之间的接触 |
| 空气曲棍球 | 过宽 | 击球槌中心最低到台面下 11.1 cm，4.2% 的控制步低于台面 1 cm 以上；真实系统要求末端高度在 ±2 cm 以内 | 打开槌和台面的接触，或在逆运动学和伺服中强制 ±2 cm |
| 棒球（击球） | 过宽 | 挥棒最高 34 m/s（职业联赛平均 32 m/s），参考选手 5 挥 5 本垒打；没有找到能达到 30 m/s 的整根球棒挥棒机 | 声明为挥棒机，并确定速度上限（本垒打约需 26 m/s） |
| 击剑、网球（随机姿态） | 过宽，只在模型中 | 合法姿态下剑可以穿过自己的底座 15.5 cm；网球 3.4% 的随机姿态中球拍穿过底座 | 打开相应部件的碰撞 |
| 桌上足球 | 符合，需声明 | 杆实测 2.71 m/s、101 rad/s，超过自己规格写的 2.5 m/s、80 rad/s（多 8%、26%） | 声明对应的真实机器，并执行规格上限 |
| 无人机 | 动力学符合；观测过宽；场地偏严 | 观测没有噪声和延迟，而真实竞速无人机依靠视觉惯性里程计，误差约 ±0.1 m；门和赛道小于国际航联的规定 | 声明为动作捕捉场地，或加入相应噪声；写明场地偏小 |
| 赛车 | 动力学偏严；车体与规则不符 | 最高速度 3.9 m/s，比真车保守；车宽 0.45 m，超出规则规定的 0.326 m | 写明 4 m/s 是主办方的限速；声明车体，或按标准底盘重建 |
| 乒乓 | 符合 | 28 对碰撞中 27 对是真实的物理碰撞 | 真实限位版本上线后复查 |
| 保龄球、飞镖、投球、飞盘 | 符合 | 出手速度在对应真实机器或人类的范围内；飞盘的飞行距离可能偏长 15–30%（估计） | 飞盘按实测数据校准升力和阻力系数 |

**仿真 ÷ 真实的比值**（大于 1 表示仿真允许的更多）：

| 量 | 比值 |
|---|---|
| 网球出球速度 ÷ 拍头速度 | 6.50 |
| 网球导轨加速度 | 4.71 |
| 网球手臂加加速度 | 3.20 |
| 网球手臂加速度 | 2.90 |
| 网球抽击时的关节速度 | 1.80 |
| 桌上足球杆速 | 1.08 |
| 棒球挥棒速度 | 1.06 |
| 赛车最大侧向加速度 | 0.92 |
| 无人机推重比 | 0.81 |
| 赛车最高速度 | 0.37 |
| 无人机最高速度 | 0.32 |

**审计带来的三条经验**，已并入上面的原则：
1. **限位要查全**：只检查速度不够，加速度、加加速度和力矩变化率都要检查。
2. **检查接触是否打开**：场景里全局关闭接触时，显式声明的碰撞对也会一起失效。所以自碰撞检查不能只确认碰撞对存在，还要确认接触确实打开了。
3. **可行性检验要在真实物理下做**：网球的高可回率来自不真实的球拍规律，修正物理之后必须重做。

**审计中尚未确定的部分**，标为未知或估计：
- 部分轮式人形的穿透可能只是碰撞外壳过厚，需要用精细网格或真机确认；
- 真实机械臂受到击球冲击时是否触发碰撞保护，需要真机测试；
- 无人机、赛车在加入噪声或修改上限后还没有重跑。

**审计是持续的**：
- 运动或机器人升版本时，重新审计受影响的部分；
- 能自动化的项目写进检查脚本：限位必须经过共享模块、接触不能被全局关闭、比赛中的穿透深度不超过阈值、出手速度不超过声明的机器上限。

### 例：乒乓的机器人选择

下表是乒乓在这一步的候选和检验结果。检验方法是：上界选手在真实限位下，能接回多少合法发球。

| 候选（机器人 + 标准球台） | 合法发球的可回率 | 结果 |
|---|---|---|
| Franka Panda + 导轨 | 94.7% | 通过，作为高层的机器人 |
| ABB IRB 1100 + 两轴龙门 | 73.9%（加长球拍 88.4%） | 不通过：臂展短，够不到网前短球 |
| RB-Y1（轮式人形） | 69% | 不通过 |
| G1（腿式人形） | 34% | 不通过，只用于底层研究 |
| H1-2（腿式人形） | 23% | 不通过 |
| R1 Pro（轮式人形） | 0% | 不通过：按官方 SDK 限位无法完成挥拍 |

通过的配置接着进入第 ③–⑤ 步：
- **高层**：使用 `hit(t, pos, normal, vel)`，符合 P1、P2。
- **底层**：提供关节接口和 IK 库。
  - 参考实现的隐藏请求测试为 79/82；
  - 只用 IK 挡球、不开发挥拍的程序为 0；
  - 前沿模型达到 77–80/82，接近满分。真实限位版本上线后要重新检验区分度。

## 第三步：设计接口

**这一步回答**：第 2 节的问题 5（平台给什么）和问题 9（真正用过的技能），并最终确定问题 2（设哪种角色）。

这一步决定 agent 和比赛世界之间的接口：agent 能看到什么、能下什么命令、平台替它做了什么。接口决定 agent 要解决什么问题：给得太多，没有东西可测；给得太少，任务无法完成。

这一步也决定这项运动设哪种模式：高层、底层，或两者都设。判断标准见第 1.2 节。

### 这一步确定什么

| 项 | 内容 |
|---|---|
| 高层技能合同 | 按 P1–P6 设计。合同写明输入（坐标系、单位、范围）、执行时序（何时执行、能否打断）、反馈、控制器读取什么和能纠正什么、失败处理 |
| 底层工具包 | 机器人模型、关节接口、可选的基础控制（如 IK 库、行走策略），列入公开清单，注明来源、许可证和 sha256；不包含这项运动的核心技能。控制基础怎样选择见第 5.3–5.7 节 |
| 观测 | 白名单：相机画面、自身状态、裁判信号；不提供世界的真实三维状态 |
| 时间模型 | 实时项目的慢放倍数，或回合制项目的棋钟 |
| 算力 | 底层可以自行训练控制器，目前只允许使用 CPU，训练时间计入比赛会话 |
| 技能在环 | 见第 6.8 节 |

### 高层：技能设计原则

| 编号 | 原则 | 要求 |
|---|---|---|
| P1 | 动作导向 | 技能接收身体动作目标（姿态、速度、出手参数），不接收比赛结果（"打到某点""射门"） |
| P2 | 不替代竞技决策 | 控制器可以修正执行误差，不能替 agent 预测物体轨迹或选择策略 |
| P3 | 物理可行 | 技能遵守真实限位；不可行的命令不保证成功 |
| P4 | 任务完备 | 只使用平台技能的参考选手能够完成有意义的比赛 |
| P5 | 测评有效 | 技能不能替 agent 完成要评测的推理；由第 7 节的检验证明 |
| P6 | 接口透明 | 明确输入、执行时序、反馈、控制器的权限和失败处理；所有 agent 使用同一份 |

例：乒乓提供 `hit(t, pos, normal, vel)`，由 agent 决定拍子在何时、何处、以什么朝向和速度击球；不提供 `hit(落点)`，否则"怎样把球打到指定位置"这一计算就由平台完成了。不同运动的技能 API 可以不同，统一的是合同格式和验收流程。

### 底层：平台提供多少控制能力

底层模式要回答的问题是：**给 agent 多少控制能力。**

- **给得太少**，任务无法完成。例如腿式人形只给关节力矩时，agent 首先要解决站立平衡，所有运动都退化成"能不能站稳"，成绩全部接近 0，测不出差别。
- **给得太多**，这项运动的核心运动技能已经由平台完成，agent 只需调用接口，没有工程可测。

这个问题包含三个相互独立的变量，分开决定：

| 变量 | 问题 | 当前设定 |
|---|---|---|
| 控制基础 | 平台提供多少底层控制能力 | 按机器人和运动选择，用第 6.7 节的四条标准检验 |
| 资源来源 | 这些控制能力从哪里来 | 平台提供固定的工具包 |
| 算力 | agent 能否自己训练控制器，用多少算力 | 可以训练，只用 CPU，训练时间计入比赛会话 |

### 控制基础的五种方案

<svg viewBox="0 0 920 270" width="100%" role="img" aria-label="底层控制基础从少到多：五种方案和两端的风险" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.h{fill:var(--fg,#16211c);font-size:15px;font-weight:600}.s{fill:var(--muted,#56665e);font-size:12px}.m{fill:var(--fg,#16211c);font-size:11.5px;font-family:ui-monospace,Menlo,monospace}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.gr{fill:rgba(140,140,140,.07);stroke:rgba(140,140,140,.45)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}.cco{fill:rgb(78,124,196)}.cto{fill:rgb(142,110,190)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.bk{stroke:rgb(206,112,58);stroke-width:1.3;fill:none;stroke-dasharray:5 4}.fz{stroke:rgb(78,124,196);stroke-width:1.6;fill:none;stroke-dasharray:3 4}.cb{fill:rgb(206,112,58);font-size:12px}.bd{fill:rgb(206,112,58);font-size:11px;font-weight:600}</style>
<defs><marker id="m81005" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<rect x="20" y="20" width="200" height="54" rx="8" class="sp" stroke-width="1.2"/>
<text x="120" y="42" text-anchor="middle" class="t b csp">给得太少</text>
<text x="120" y="62" text-anchor="middle" class="s">任务无法完成，成绩接近 0</text>
<rect x="700" y="20" width="200" height="54" rx="8" class="sp" stroke-width="1.2"/>
<text x="800" y="42" text-anchor="middle" class="t b csp">给得太多</text>
<text x="800" y="62" text-anchor="middle" class="s">核心技能已由平台完成，无可测</text>
<rect x="240" y="20" width="440" height="54" rx="8" class="pl" stroke-width="1.2"/>
<text x="460" y="42" text-anchor="middle" class="t b cpl">可选范围</text>
<text x="460" y="62" text-anchor="middle" class="s">通过可行、不平凡、可扩展、可测量四条标准（第 6.7 节）</text>
<rect x="20" y="100" width="166" height="62" rx="8" class="nu" stroke-width="1.2"/>
<text x="103" y="125" text-anchor="middle" class="t b">A 原始控制</text>
<text x="103" y="147" text-anchor="middle" class="s">关节接口、伺服</text>
<rect x="198" y="100" width="166" height="62" rx="8" class="nu" stroke-width="1.2"/>
<text x="281" y="125" text-anchor="middle" class="t b">B 机器人工具库</text>
<text x="281" y="147" text-anchor="middle" class="s">+ 正逆运动学等</text>
<rect x="376" y="100" width="166" height="62" rx="8" class="nu" stroke-width="1.2"/>
<text x="459" y="125" text-anchor="middle" class="t b">C 行走 + 独立操作</text>
<text x="459" y="147" text-anchor="middle" class="s">行走策略管腿</text>
<rect x="554" y="100" width="166" height="62" rx="8" class="nu" stroke-width="1.2"/>
<text x="637" y="125" text-anchor="middle" class="t b">D 协调的全身控制</text>
<text x="637" y="147" text-anchor="middle" class="s">跟踪动作目标</text>
<rect x="732" y="100" width="166" height="62" rx="8" class="nu" stroke-width="1.2"/>
<text x="815" y="125" text-anchor="middle" class="t b">E 封闭的全身策略</text>
<text x="815" y="147" text-anchor="middle" class="s">没有扩展接口</text>
<path d="M20 182L898 182" class="ar" marker-end="url(#m81005)"/>
<text x="460" y="202" text-anchor="middle" class="s">平台提供的控制能力：少 → 多</text>
<text x="281" y="232" text-anchor="middle" class="t b cco">固定机械臂、轮式人形</text>
<text x="459" y="232" text-anchor="middle" class="t b cco">腿式人形打球、踢球</text>
<text x="637" y="232" text-anchor="middle" class="t b cco">短跑、跨栏（待验证）</text>
<text x="460" y="256" text-anchor="middle" class="s">蓝字：各类机器人的起点（仍须通过四条标准）</text>
</svg>

**图 10　底层控制基础从少到多。两端都不可取：太少则任务无法完成，太多则没有工程可测。**

| 方案 | 平台提供 | 主要测量的工程能力 | 主要风险 |
|---|---|---|---|
| A 原始控制 | 关节接口、基础伺服 | 底层控制、动力学、稳定性 | 难度过高，可能完全无法完成运动 |
| B 机器人工具库 | A + 正逆运动学、优化器等工具 | 控制设计、轨迹规划、工具集成 | 工具可能过强（直接解决核心技能）或过弱 |
| C 行走策略 + 独立操作 | 预训练的行走策略控制腿，agent 控制上身 | 运动技能开发、上下身协调 | 控制权分开，不代表动力学上协调 |
| D 协调的全身控制 | 能跟踪动作目标（参考动作）的全身控制策略 | 生成动作参考、开发技能、系统集成 | 受限于基础策略的能力边界，需要适配 |
| E 封闭的全身策略 | 训练好的整体策略，没有合适的动作扩展接口 | 主要是部署已有策略 | 可能没有留下技能开发的空间 |

这五种是**候选的控制基础**，不是五个要分别实现的赛道。每个"运动 × 机器人"选择其中一种，选择依据是第 6.7 节的四条标准；不同运动可以选择不同的方案，不要求统一。

### 预训练策略要先检查什么

**一个训练好的行走策略，不一定支持 agent 独立控制手臂。** 即使它只输出腿部关节目标，可以和手臂指令拼在一起，也不代表机器人能稳定地边走边挥拍。因此，使用任何预训练策略之前，要逐项检查：

| 检查项 | 问题 |
|---|---|
| 观测 | 策略读取哪些状态？是否读取手臂的关节状态或上一步动作？手臂由 agent 控制时，这些输入是否超出训练分布 |
| 动作 | 策略输出哪些关节的目标 |
| 控制关节 | 哪些关节可以交给 agent，交接时策略是否需要重新初始化 |
| 可扩展性 | 策略是否接受外部的动作目标；agent 快速挥动手臂时，下肢能否保持平衡 |
| 训练条件 | 训练时手臂的负载、动作分布和执行器模型，是否和这项运动一致（例如拿着球拍） |
| 稳定性 | 在目标运动和装配配置上实测：摔倒次数、骨盆漂移、末端跟踪误差 |

我们在 G1 人形机器人上的实测说明了这项检查的必要性：

- **宇树官方行走策略 + agent 控制手臂（方案 C）**：原地开环挥拍时，骨盆漂移 28–68 cm。
- **FALCON（上下肢联合训练的公开策略）**：
  - 站得更稳，挥拍时不摔倒；
  - 但手臂跟踪误差 16–23 cm，击球时拍速只有计划值的 20–85%；
  - 把手臂换成独立伺服、只保留它的下肢部分时，机器人 10 秒内走开 7.8 米：上下身无法分开使用。

**策略的名称不能说明它能做什么，必须在目标运动和装配配置上实测。**

另一方面，上下身之间的耦合本身就是真实的工程内容。在方案 C 上：
- agent 编写的击球实现在隐藏请求测试中达到 21–28/29，平台自己的实现只有 8/29；
- 只用工具、不开发技能的程序为 0/29，比赛中摔倒 9 次。

### 资源来源

| 方案 | 含义 | 额外测量的能力 | 代价 |
|---|---|---|---|
| 固定工具包 | 平台提供固定的工具和策略 | 在给定资源下开发 | — |
| 开放资源 | agent 自己搜索、部署和集成外部资源 | 技术调研、选型、迁移、依赖管理 | 混入外部资源是否可得、软件能否安装等与运动控制无关的因素；复现困难；需要开放网络，增加安全隔离的成本 |
| 混合 | 平台提供基础工具，允许 agent 搜索替换 | 工程开发 + 自主选择资源 | 同上 |

**决定**：默认使用固定工具包。开放资源有研究价值，作为单独的实验条件，不替代默认设置。选手容器默认没有外网（见第 7.1 节）。

**固定工具包的要求**：
- **公开清单**：工具包列入公开清单，写明每个组件的来源、许可证、版本和 sha256。清单属于运动，是任务版本的一部分，所有 agent 使用同一份。
- **复用资源库**：同一个机器人尽量复用资源库中有版本号的模型、限位和控制代码；开放哪些能力，按运动决定。
- **工具不读取外部世界**：工具只读取机器人自身的状态，不读取球、对手和场地。
- **不含核心技能**：这项运动的核心技能（例如乒乓的挥拍击球）不放进工具包。

### 选择控制基础的四条标准

无论选择哪种控制基础，都只检查四件事。标准统一，具体方案不要求统一。

| 标准 | 问题 | 在筛选流程中的检验 | 通过标准 |
|---|---|---|---|
| 可行（Feasible） | 在规定的机器人、工具和算力下，是否有可靠的参考实现 | 第 ④ 步：参考解 | 只用工具包做出的参考实现，隐藏请求测试达到下限 |
| 不平凡（Non-trivial） | 给了基础能力之后，agent 是否仍需真正开发核心运动技能 | 第 ④ 步：技能必要性 | 只用工具、不开发技能的实现得分接近 0 |
| 可扩展（Extensible） | agent 是否有足够的控制权和开发空间 | 第 ③ 步：接口审查 | agent 能直接控制核心技能涉及的关节；控制权按关节组仲裁，交接平滑；基础策略可以使用、修改或替换 |
| 可测量（Measurable） | 能否用隐藏请求测试和参考策略比赛测试客观评价 | 第 ④、⑤ 步 | 两项测试能区分实现的好坏；模型的分数不全在 0，也不全在满分 |

**按机器人类型的起点**（仍需通过上面四条标准）：

| 机器人类型 | 起点 | 理由 |
|---|---|---|
| 固定机械臂（例：乒乓的 Franka） | B：关节接口 + IK 库 | 乒乓上已验证：参考实现隐藏请求测试 79/82，只用 IK 挡球的程序为 0 |
| 轮式人形 | B + 底盘速度接口 | 底盘和手臂的协调留给 agent |
| 腿式人形（球拍类、足球） | C：官方行走策略 + 关节组仲裁 | 真机同样随附厂商的行走控制；平衡和手臂的耦合留下了工程空间；D 待验证 |
| 腿式人形（短跑、跨栏） | D：动作跟踪的全身控制 | 给行走或奔跑策略就等于完成了这项运动本身 |
| 出手机器、车辆、无人机 | 按各自的控制方式 | 出手机器的命令就是出手参数，底层没有可开发的技能；如果改用机器人身体投掷，按上面对应的机器人类型选择 |
| 任何机器人 | A 只作扩展条件 | 只给关节力矩时，腿式人形的成绩预计全部接近 0 |

### 技能在环

<svg viewBox="0 0 920 214" width="100%" role="img" aria-label="底层的技能在环：比赛和赛后测试使用同一份技能实现" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.h{fill:var(--fg,#16211c);font-size:15px;font-weight:600}.s{fill:var(--muted,#56665e);font-size:12px}.m{fill:var(--fg,#16211c);font-size:11.5px;font-family:ui-monospace,Menlo,monospace}.b{font-weight:600}.pl{fill:rgba(76,170,124,.12);stroke:rgba(76,170,124,.55)}.sp{fill:rgba(236,146,82,.12);stroke:rgba(236,146,82,.55)}.co{fill:rgba(98,148,222,.12);stroke:rgba(98,148,222,.55)}.to{fill:rgba(160,128,208,.11);stroke:rgba(160,128,208,.55)}.nu{fill:var(--card,#ffffff);stroke:var(--line,#d7e0db)}.ar{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none}.dl{stroke:var(--muted,#56665e);stroke-width:1.3;fill:none;stroke-dasharray:5 4}</style>
<defs><marker id="m81004" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted,#56665e)"/></marker></defs>
<rect x="20" y="24" width="190" height="112" rx="8" class="pl" stroke-width="1.3"/>
<text x="115" y="52" text-anchor="middle" class="t b">agent 的竞技程序</text>
<text x="115" y="76" text-anchor="middle" class="s">决定何时、何处、多快</text>
<text x="115" y="96" text-anchor="middle" class="s">发出技能调用</text>
<text x="115" y="118" text-anchor="middle" class="m">hit(t, pos, normal, vel)</text>
<path d="M212 80L248 80" class="ar" marker-end="url(#m81004)"/>
<rect x="250" y="24" width="250" height="112" rx="8" class="nu" stroke-width="1.3"/>
<text x="375" y="52" text-anchor="middle" class="t b">agent 的技能实现</text>
<text x="375" y="76" text-anchor="middle" class="m">/app/output/skill/hit.py</text>
<text x="375" y="98" text-anchor="middle" class="s">平台在沙箱中运行，只提供机器人状态</text>
<text x="375" y="118" text-anchor="middle" class="s">每次调用记录文件哈希</text>
<path d="M502 80L538 80" class="ar" marker-end="url(#m81004)"/>
<rect x="540" y="24" width="160" height="112" rx="8" class="co" stroke-width="1.3"/>
<text x="620" y="52" text-anchor="middle" class="t b">比赛模拟器</text>
<text x="620" y="76" text-anchor="middle" class="s">核心动作只接受</text>
<text x="620" y="96" text-anchor="middle" class="s">由技能生成的</text>
<text x="620" y="116" text-anchor="middle" class="s">关节命令</text>
<rect x="740" y="24" width="160" height="112" rx="8" class="to" stroke-width="1.3"/>
<text x="820" y="52" text-anchor="middle" class="t b">赛后测试</text>
<text x="820" y="76" text-anchor="middle" class="s">同一份文件</text>
<text x="820" y="96" text-anchor="middle" class="s">隐藏请求测试</text>
<text x="820" y="116" text-anchor="middle" class="s">+ 参考策略比赛测试</text>
<path d="M375 137L375 152L820 152L820 140" class="dl" marker-end="url(#m81004)"/>
<text x="598" y="168" text-anchor="middle" class="s">比赛结束时的最终版本（哈希一致）</text>
<text x="20" y="196" class="s">复盘时可以修改竞技程序和技能实现；每个版本的哈希及其参与的回合都写入比赛记录。</text>
</svg>

**图 11　技能在环（底层）：比赛中使用的技能实现，就是赛后测试的技能实现。**

**技能在环**保证核心技能确实参与了比赛：

| 模式 | 要求 | 做法 |
|---|---|---|
| 高层 | 核心技能必须用得上 | 动作只有平台技能；第 ④ 步的技能必要性检验确认：必需型运动中只用非核心动作的程序得分接近 0；价值型运动中不用核心技能的版本明显弱于参考选手 |
| 底层 | 比赛由 agent 自己的技能实现完成 | 竞技程序发出技能调用，平台在沙箱中运行 agent 的技能文件，生成关节命令；核心动作不接受绕过技能的命令；每次调用记录文件哈希，赛后测试使用同一份文件。生成的任务里每个座位一个技能容器（没有网络、只读、看不到模拟器，`sport.toml [modes.low.inloop]`），技能的计算时间计入比赛时间（世界不等它，算完的下一个控制步生效） |

### 接口审查

逐条核对 P1–P6，确认合同、工具包清单和观测白名单完整。

## 第四步：无模型验证

**这一步回答**：第 2 节的问题 6（能否完成比赛）、问题 7（会不会太简单）、问题 8（成绩是否反映能力），以及问题 11 中的工程、公平和安全部分。

这一步只使用参考选手和脚本程序，不消耗模型额度。大部分不合适的配置在这里被排除。

**跑法的统一规定**：
- **延迟和慢放**：门槛在默认慢放倍数下判，并带上和正式比赛一致的命令生效延迟；A4（时间与延迟）的能力敏感性另在较低的慢放倍数（×1–×5）下测。"一致"的定义（core 0.2.18）：实时运动在 `sport.toml [timing] reaction_delay_steps` 声明反应延迟 D，正式服务里命令在 max(下一个控制步边界, 该座位最近一次拿到的世界状态的步 + D) 生效，双方一样；筛选框架 `tools/screen/play.py` 直接读这个 D（hooks 里不再写 `DELAY`），所以只要程序在 D 之内算完，筛选和比赛按定义一致，不受机器负载影响。D 取正式慢放下正常程序实测延迟的中位数，和标定一起定（CONTRIBUTING.md "Declare the reaction delay"）。
- **发球**：简单基线统一使用参考选手的发球，免得基线因发球延误直接判负、门槛形同虚设。
- **成本**：大样本的检验（简单基线、能力敏感性）在真值感知或不渲染画面的快速模式下跑；渲染相机画面的比赛只做确认，每级 5–10 场。对手每改一次，先重跑参考选手和简单基线；能力敏感性和第 ⑤ 步只在最终配置上跑。
- **底层**：这一轮底层只要求参考技能达到隐藏请求测试的门槛、边界实现得分接近 0；隐藏请求测试和比赛测试结论不一致时以哪个为准，另行规定。

| 检验 | 问题 | 方法 | 通过标准 |
|---|---|---|---|
| 参考解 | 用平台提供的东西能否完成任务（P4） | 只用平台技能（高层）或只用工具包（底层）编写参考选手。对手标定用真值感知跑大样本；另用相机参考选手和模型试跑的比赛做端到端检查（见第 2 节问题 6） | 高层：真值参考选手能赢最低一级、胜率随等级下降，且有"只看画面也能赢"的证据（相机参考选手赢下多数，或模型只看相机赢过）；报告相机参考选手的合法回球率。底层：参考实现的隐藏请求测试达到下限 |
| 技能必要性 | 核心技能是否用得上 | 高层必需型（不用技能就无法得分，如乒乓）：只用非核心动作的程序；高层价值型（有别的正当得分办法，如足球的带球）：参考选手和"决策相同、从不用该技能"的版本在相同种子上配对比较；底层：只用工具、不开发技能 | 必需型、底层：得分接近 0；价值型：不用技能的版本在每个正式等级都明显弱于参考选手（配对差值区间在 0 以下） |
| 简单基线 | 决策空间是否足够 | 固定动作、随机、只做反射、只打中路等简单程序。先用真值感知跑；某个基线在某一级用真值赢了，就只把这一项改用和 agent 相同的相机感知重跑 | 对每个正式对手等级都输掉多数比赛（每级至少 20 个种子，同时报告胜率）；改用相机才输掉的等级，报告里标明"这一级主要考感知"；最高正式等级上，读真值的基线也必须输掉多数；单人项目明显低于参考选手 |
| 能力敏感性 | 声称考察的能力，比赛是否真的需要 | 在参考选手中逐项减弱一项能力，技能保持不变，每项设两档强度，与原版在相同种子上配对比较（见下文"能力敏感性怎么做"） | 强档的成绩明显低于原版，弱档不高于噪声；每级对手都满足 |
| 工程检查 | 比赛能否可靠、公平地进行 | 确定性和重放、已知漏洞测试、公平性检查（先手和座位优势）、参考选手得 1 分而不动的选手得 0 分 | 全部通过；先手和座位优势在 ±5 个百分点以内 |
| 安全检查 | 选手能否越过信息边界 | 隔离探针尝试读取模拟器、写文件、联网 | 全部失败 |

**比赛要有意义**（和参考解一起检查）：
- 比赛长度和复盘次数要让第一版程序有机会改进。例：羽毛球一局 5 分时，模型第一版程序全部 0:5，每场只复盘 4 次；改成一局 11 分后才有区分度；
- 团队运动给出每场进球数的下限：两支像样的队几乎总是 0:0 时，比赛比不出高下（足球 5v5 的人形版就停在这里）。

**简单基线和能力敏感性两项检验最常不通过。** 在对 21 项运动的审核中，技能接口本身基本合格；比赛不能区分能力的原因集中在三处：
- 脚本对手分级失效，简单策略就能战胜高等级对手；
- 官方种子公开且固定，最优动作可以记住；
- 二值成功线贴近能力上界。

这也是第 ① 步要把对手、种子和计分一起设计的原因。

### 能力敏感性怎么做

能力敏感性回答的是：**我们声称这项运动考察某项能力，这项能力是否真的影响成绩？** 做法是用参考选手做对照实验：故意把某一项能力弄差，其他完全不变，看成绩会不会下降。

**1. 每项声称的能力，做弱、强两档"削弱版"参考选手。** 原则是**削弱能力，不削弱技能**：削弱版照样能正常出手、正常比赛，只是某一项判断变差了。把技能拿掉，成绩当然会下降，但这说明不了任何能力。以乒乓为例：

| 能力 | 弱档 | 强档 |
|---|---|---|
| A1 三维感知 | 画面中球的位置加 1 像素噪声 | 加 2 像素噪声 |
| A2 物理理解 | 预测球路时忽略空气阻力 | 用错误的弹跳系数 |
| A4 时间与延迟 | 少补偿 20 ms 延迟 | 少补偿 30 ms 延迟 |
| B2 策略与读对手 | 不看对手，随机选边 | 不看对手，永远打同一个角 |

**2. 与原版在相同条件下配对比较。**
- 同一组种子、同样的对手等级，每级至少 20 个配对种子；
- **对正式的对手等级测**，不对同水平的参考选手测：同水平的两方对打几乎全是平局（足球 20 场里 17 场打平），削弱什么都测不出；
- 比较得分占比（我方得分 / 双方总分），它比胜负更细；
- 对每个种子计算"削弱版 − 原版"的差值，再对差值求 95% 置信区间。配对能消掉种子本身难易带来的差别。

**3. 先测噪声。** 真实物理的比赛是混沌的：乒乓中一个数改动 0.1 mm，几拍之后比赛就完全不同了。所以要做几个"复本"，只改动无关紧要的细节（例如命令的数值多保留一位小数），看什么都不改时成绩会波动多少。没有这个基准，就分不清成绩下降是因为能力被削弱，还是随机波动。

**4. 判定。** 目前的判据：
- 强档的配对差值区间在 0 以下；
- 弱档不能比原版高出噪声以上；
- 每一级对手都满足，最多允许一级例外。

<svg viewBox="0 0 920 352" width="100%" role="img" aria-label="能力敏感性的判定示意：复本给出噪声带；能力甲的削弱版明显低于原版，判为敏感；能力乙的削弱版落在噪声带内，判为不敏感" style="max-width:920px;font-family:inherit">
<style>.t{fill:var(--fg,#16211c);font-size:13px}.s{fill:var(--muted,#56665e);font-size:12px}.b{font-weight:600}.nb{fill:rgba(140,140,140,.14)}.ax{stroke:var(--muted,#56665e);stroke-width:1}.z{stroke:var(--fg,#16211c);stroke-width:1.2}.gl{stroke:var(--line,#d7e0db);stroke-width:1;stroke-dasharray:3 4}.cg{stroke:rgb(140,140,140);stroke-width:2.2}.fg{fill:rgb(140,140,140)}.cpl2{stroke:rgb(62,150,106);stroke-width:2.2}.fpl{fill:rgb(62,150,106)}.csp2{stroke:rgb(206,112,58);stroke-width:2.2}.fsp{fill:rgb(206,112,58)}.cpl{fill:rgb(62,150,106)}.csp{fill:rgb(206,112,58)}</style>
<rect x="664" y="44" width="112" height="244" class="nb"/>
<text x="720" y="36" text-anchor="middle" class="s">复本的噪声范围</text>
<line x1="440" y1="44" x2="440" y2="300" class="gl"/><line x1="580" y1="44" x2="580" y2="300" class="gl"/><line x1="300" y1="44" x2="300" y2="300" class="gl"/><line x1="860" y1="44" x2="860" y2="300" class="gl"/>
<line x1="720" y1="44" x2="720" y2="300" class="z"/>
<text x="290" y="74" text-anchor="end" class="t">复本（只改无关细节）</text>
<line x1="678" y1="70" x2="762" y2="70" class="cg"/><circle cx="720" cy="70" r="5" class="fg"/>
<text x="290" y="119" text-anchor="end" class="t">能力甲 · 弱档</text>
<line x1="531" y1="115" x2="699" y2="115" class="cpl2"/><circle cx="615" cy="115" r="5" class="fpl"/>
<text x="290" y="154" text-anchor="end" class="t">能力甲 · 强档</text>
<line x1="342" y1="150" x2="538" y2="150" class="cpl2"/><circle cx="440" cy="150" r="5" class="fpl"/>
<text x="300" y="184" class="t b cpl">能力甲：两档都低于原版，强档区间在 0 以下 → 敏感，可以声称</text>
<text x="290" y="219" text-anchor="end" class="t">能力乙 · 弱档</text>
<line x1="657" y1="215" x2="804" y2="215" class="csp2"/><circle cx="734" cy="215" r="5" class="fsp"/>
<text x="290" y="254" text-anchor="end" class="t">能力乙 · 强档</text>
<line x1="629" y1="250" x2="776" y2="250" class="csp2"/><circle cx="699" cy="250" r="5" class="fsp"/>
<text x="300" y="280" class="t b csp">能力乙：两档都落在噪声范围里 → 不敏感，不能声称</text>
<line x1="300" y1="300" x2="860" y2="300" class="ax"/>
<text x="300" y="318" text-anchor="middle" class="s">−0.6</text><text x="440" y="318" text-anchor="middle" class="s">−0.4</text><text x="580" y="318" text-anchor="middle" class="s">−0.2</text><text x="720" y="318" text-anchor="middle" class="s">0</text><text x="860" y="318" text-anchor="middle" class="s">+0.2</text>
<text x="580" y="342" text-anchor="middle" class="s">得分占比的配对差值（削弱版 − 原版）；点是平均值，横线是 95% 置信区间</text>
</svg>

**图 12　能力敏感性的判定（示意，不是实测数据）。灰色带是复本给出的噪声范围。能力甲被削弱后成绩明显下降，说明比赛确实奖励这项能力；能力乙被削弱后成绩没有超出噪声，就不能声称这项运动考察它。**

**结果怎么处理。**
- **成绩下降**：这项能力可以写进这项运动的能力声明。
- **成绩不下降**：不能声称考察它。要么从声明中删除，要么修改运动让它值分，最常见的是修改对手，然后重新检验。例如乒乓和羽毛球的早期对手每次击球后都会回到原位，"打离对手远的一侧"和随机选边没有区别，读对手（B2）就测不出来；羽毛球的对手改成击球后慢慢回位之后，B2 才开始敏感。因此：**声明 B2 的运动，脚本对手必须有可以利用的习惯**（例如回位慢、会预判你的习惯）；**场地大小和机器人速度的比例也要让站位有意义**。羽毛球的场地缩小到 0.4 倍，而底座的导轨 5 m/s，横穿场地只要半秒，站在哪里、往哪边打都无所谓，所以要换更慢的真实底座。

**它能说明什么、不能说明什么。**
- **它检验的是 benchmark，不是模型**：它证明"这项运动会奖励这项能力"，是发布前的检验。某个模型在哪项能力上强，要靠比赛中的过程指标和受控实验（第 9 节）。
- **依赖对手**：读对手这类能力，对手必须有可以利用的习惯才测得出来。
- **有些能力不用这种方法**：
  - 高层的 A3（具身执行）由平台完成，不归于 agent；
  - C1、C2 是编码 agent 自身的能力，参考选手里没有可以削弱的对应部分。C2 用程序快照在相同种子上离线重赛来测（第 9 节"受控实验"）。
- **样本量要够**：真实物理的比赛噪声大。乒乓中单场得分占比的波动约 ±0.4，每级 10 个种子只能检出 0.15 左右的差别，所以弱档的削弱常常检不出来。差别小的能力需要更多种子。


### 安全与公平

工程检查和安全检查依据下列措施。它们大多由框架统一实现，每项运动接入时自动获得，第 ④ 步确认它们在这项运动中确实生效。

| 方面 | 措施 |
|---|---|
| 信息边界 | 观测按白名单提供；平台的控制器和行走策略只读取机器人自身状态，不读取球和对手；种子不公开；底层的隐藏请求只保存在模拟器一侧，只在赛后使用 |
| 网络隔离 | 选手容器位于内部网络，准备阶段结束后断开外网；网关隔离，无法访问宿主机上的服务；去除原始套接字权限 |
| 模型访问 | 计量代理只转发固定路径，每个座位锁定模型，去除服务器端工具（网页搜索、抓取） |
| 请求处理 | 格式错误的请求一律记为选手的无效动作，不会导致比赛作废 |
| 比赛结构 | 没有重置和暂停接口；所有选手共用一个时钟；一次只处理一个请求；队内频道只在同队之间传递 |
| 公平 | 镜像对称测试；先手和座位优势检查；对战双方的执行误差序列相互独立，一对比赛中交换使用；所有 agent 使用同一份配置和工具包 |
| 技能测试隔离 | 无网络、只读文件系统、降低权限；隔离探针确认技能代码无法读取模拟器或联网 |
| 有效性 | 技术故障的比赛作废重赛；选手自身原因（未到场、中途退出、复盘超时）记为有效的负场 |
| 审计 | 比赛记录可逐位重放，状态哈希覆盖所有影响比分的量；每次复盘保存程序快照 |

## 第五步：模型试跑

**这一步回答**：第 2 节的问题 10（能否区分不同的 agent）。

这是唯一需要运行模型的步骤：少量前沿模型（至少两个）各比赛一到两场。

**区分度检验**：
- 分数不能全部接近 0，也不能全部接近满分；
- 模型发现的捷径和漏洞要修复，修复后回到第 ④ 步重新检验；
- 模型找到的有效固定打法，要固化成简单基线，回到第 ④ 步重测。例：羽毛球里 Astra 找到"短宽、左右交替"，写成程序后打穿了 L1、L2，于是给对手加了预判。

**试跑前先估算复盘节奏**：确认 agent 在复盘时限内来得及回应。足球的第一场对战因为双方都没在 300 秒内回应复盘，判为不完整；复盘时限改成 600 秒后重跑。

**为什么要单独一步**：参考选手和脚本只能证明任务可解、不平凡；真实 agent 会以设计者没有想到的方式利用环境。例如在乒乓和羽毛球中，模型找到过利用对手出拍延迟和回合加速规则得分的打法。这类捷径只有在模型实际比赛后才会暴露。

## 第六步：冻结与正式评测

**这一步回答**：第 2 节问题 11 中的冻结部分；问题 8 的三个证据层次和问题 9 的赛后技能测试在这一步实施。

通过前五步之后，配置冻结，进入正式评测。正式成绩只来自比赛结果；对九项能力的分析来自行为诊断和受控实验。

### 冻结

**记录的版本**：
- 运动的版本；
- 装配配置的哈希；
- 工具包的 sha256；
- 核心版本。

**冻结的内容**：种子集保密，对手等级和计分方式固定。
- **保密种子由核心统一实现**：冻结时生成种子集，只公开它的 sha256；比赛运行时由核心把种子交给运动，验证器按 sha256 认可。各运动不各自实现。
- **连续分数由各运动定义，核心统一字段**：连续分数是什么因运动而异（例如乒乓、羽毛球用得分占比，飞镖用平均每镖得分）；核心在比赛结果里提供统一的字段，排行榜和统计按这个字段读取。

**怎么做**（核心 0.2.14，框架说明 6.4 节，CONTRIBUTING.md "Freeze a sport"）：
1. 在 `sport.toml [score]` 声明连续分数。原则对所有运动相同：越大越好、和胜负同方向，判负的一方得固定的 `forfeit` 值；负责人按建议表从 `share`（得分占比）、`margin`（净胜）、`own`（己方得分）、`hook`（插件的 `continuous_score()`）里选一种；
2. 在 mll-4090-1 上用 `tools/seeds.py new <运动> --count N` 生成保密种子集，按"形式 × 等级"分格、每格各自的种子。私密文件只放在这台机器的受限目录（例如 `/data/guichi/secrets/seeds/`），由用户保管和备份；仓库里只有承诺和 sha256（`[official.hidden_seeds]`）。合作者开发用公开种子；
3. 重新生成任务，`tools/check <运动> --stages 1` 通过。此后只有对应格里的保密种子算正式配置，正式比赛在 mll-4090-1 上用 `--seed hidden:I` 运行；任何记录里都不出现真实种子。

赛季期间不修改任务，补赛使用同一版本。冻结后不根据任何模型的成绩调整配置，修改即发布新版本，并重新通过第 ④、⑤ 步。

### 能力分类

benchmark 用九项能力描述"一个好的运动员程序需要什么"：

| 领域 | 能力 | 含义 |
|---|---|---|
| A 物理世界理解 | A1 3D 空间 | 从相机画面恢复位置、深度和朝向，在三维中推理角度、线路和坡面 |
| | A2 物理动力学 | 预测物体的运动（飞行、阻力、旋转、弹跳、摩擦、碰撞、风），并从观测反推物理参数 |
| | A3 具身与执行 | 知道自己的机器人能做到什么（可达范围、速度、力矩），把意图变成准确的动作；底层模式下还要自己搭建控制 |
| | A4 时间与延迟 | 知道画面是多久以前的、计算和执行要花多久，按"动作生效时世界的样子"提前补偿 |
| B 比赛理解 | B1 规则 | 知道怎样得分、什么犯规，并利用规则 |
| | B2 策略与读对手 | 判断局面，在风险和收益间取舍；识别并利用对手的习惯 |
| | B3 团队协作 | 分工、跑位、传球，通过队内频道沟通 |
| C 编码 agent | C1 实时编程 | 把感知、预测和策略写成能实时运行的程序，比赛中调试和修复 |
| | C2 比赛中迭代改进 | 根据比赛反馈修改程序或打法，并且越改越好 |

**九项能力是诊断用的分类，不是九个独立分数。** 一场比赛的胜负同时受九项能力影响，不能把胜率直接解释成其中某一项的测量结果。也不要求每项运动覆盖九项能力：每项运动只声明它真正考察的能力，并通过第 ④ 步的能力敏感性检验。

### 三个证据层次

| 层次 | 回答的问题 | 来源 | 是否进入正式成绩 |
|---|---|---|---|
| ① 结果 | agent 表现如何 | 胜负、得分、合法动作完成率 | 是，正式成绩只来自这一层 |
| ② 行为诊断 | 失败出现在哪个环节 | 模拟器的真实状态、物理事件、动作轨迹和比赛重放 | 否，作为每场比赛的附带分析 |
| ③ 受控实验 | 哪些条件影响表现，某项能力对成绩的因果贡献 | 只改变一个条件的配对比较 | 否，用于论文中的能力分析 |

### ① 结果：正式成绩

- **分开排名**：按"运动 × 机器人配置 × 模式"分开排名。
  - 高层每项运动只用一个机器人；
  - 没有通过真机可行性检验的组合只作为挑战组单独报告。
- **Arena**：模型对模型的对战、团队和多方比赛，采用配对种子并交换先手。
- **First Exam**：单人和对脚本对手的比赛，按各等级的成绩报告。
- **底层的三个结果**：比赛成绩、隐藏请求测试、参考策略下的比赛测试分别报告，不合成单一分数；同时报告开发成本（token、时间、修改次数、训练用的 CPU 时间）。

### ② 行为诊断

行为诊断在每场比赛后从比赛记录自动计算，不改变比赛，也不要求 agent 输出额外信息。它有三个来源：

| 来源 | 原理 | 乒乓中的例子 |
|---|---|---|
| 过程事件 | 模拟器真值和裁判事件构成每个回合的事件链 | 触球 → 过网 → 落台 → 合法回球 → 得分；失分停在哪一环（没碰到、没过网、没落台） |
| 命令隐含的判断 | 技能是动作导向的，每条命令都包含 agent 对未来的预测 | `hit(t, pos, …)` 的 t 和 pos 就是 agent 对"球何时在何处"的预测，与真实球路比较即得预测误差 |
| 反事实重放 | 模拟器是确定的，可以从任意状态分叉重放 | 同一时刻改为"完美执行""晚 50 ms 执行""换一个击球目标"，判断失分来自预测、执行还是决策 |

**误差怎样对应到能力**（乒乓上的检验结果）：
- 同一个来球的几条命令之间来回跳动：对应感知（A1）；
- 同一个来球的命令朝同一方向偏移：对应物理模型（A2）；
- 最后一条命令的时间偏差：对应延迟补偿（A4）。

**效度检验**：一个诊断指标要先通过效度检验，才能用于报告。
- **方法**：构造一组参考选手，每个只削弱一项能力，检查指标是否只对相应的那项能力敏感；
- **基线**：用多个未削弱的复本估计噪声水平；
- **通过标准**：对目标能力的效应明显大于噪声，并且远大于其他能力被削弱时的效应。

乒乓的检验中，只用通过检验的指标，就能在单场比赛中认出是哪一项能力被削弱（准确率 81%）；只看胜分率时是 35%（五类，随机水平 20%）。

**目前各项能力的可测情况**：

| 能力 | 现状 | 原因 |
|---|---|---|
| A1、A2、A4、C2 | 可以测量 | 有通过效度检验的指标或实验 |
| B1、B2 | 暂时无法测量 | 现有脚本对手下，削弱这两项能力成绩并不下降，比赛没有奖励它们；要先改进对手 |
| B3 | 只能测过程 | 分工、跑位、沟通可以从轨迹和消息测量，但现有足球规则下分工并不带来胜利 |
| A3（高层） | 不归于 agent | 高层的执行由平台完成，执行误差是平台的属性 |
| C1 | 只作描述 | 命令频率等指标同时受预测质量影响，还没有专门的检验 |

### ③ 受控实验

每次只改变一个条件，其余全部相同，用配对种子比较。不需要每项运动都做，论文中选择两到三项有代表性的实验：

| 实验 | 改变什么 | 考察的能力 | 已有结果 |
|---|---|---|---|
| 隐藏物理参数 | 摩擦、弹性等参数按种子在一定范围内变化，不告诉 agent | A2：从反馈中辨识物理参数 | 冰壶：会根据反馈校准的参考选手，停点误差随出手次数收敛，不校准的版本不收敛；两者胜率相同，只看胜率无法发现差别 |
| 时间预算 | 慢放倍数 ×20 → ×10 → ×5 → ×1 | A4：时间压力下的表现 | — |
| 视觉条件 | 相机画面与真实状态对比，或改变相机位置和光照 | A1 | — |
| 队内通信 | 关闭队内频道，队伍和规则不变 | B3：沟通的作用 | 开关已实现 |
| 对手风格 | 对手带有一个可利用的弱点，与无弱点的对手对比 | B2：读对手 | 要先证明这个弱点确实值分 |
| 能否修改程序 | 冻结程序与允许修改对比，用程序快照在相同种子上离线重赛 | C2：比赛中学习 | 乒乓：两场有明显学习；比赛内"修改前后"的朴素对比与离线结果几乎无关（相关 0.09），因此必须离线比较 |

### 评测器的原则

**评测器可以高度结构化，但不规定 agent 的程序结构，也不要求 agent 输出内部推理变量。** 例如：
- 乒乓的触球、过网、落台、合法回球和得分，都从物理记录中自动检测；
- 不要求 agent 报告自己预测的球路，它的预测已经包含在 `hit` 命令的参数里。

理由有两个：
1. **自由度是被测的一部分**：agent 选择什么算法、怎样组织程序，本身就是被测能力的一部分。
2. **要求报告会改变行为**：在乒乓中要求 agent 记录对手位置后，模型改为专门往对手另一侧打。报告要求本身就像一条提示。

### program play 测的是什么

- **测的是程序**：program play 测的是 agent 能否**编写出**具备这些能力的程序，例如用颜色阈值检测球、三角测量位置、拟合轨迹。这不等于语言模型本身在逐帧做视觉理解和物理推理，论文中要写明这一点。
- **A1 的准确表述**：agent 只能使用允许的外部感知（相机画面、裁判信号）和自身状态，无法获得世界的真实三维状态。
- **A2 的两个部分**：
  - 已知物理规律时预测物体运动；
  - 从观测中辨识未知参数。
  
  后者只有在参数隐藏且变化时才能测到，见上面的"隐藏物理参数"实验。

### 统计

- **配对**：配对种子，两种先手顺序都打，所有结果报告置信区间。
- **场数**：比较胜率要较多场次（检测中等差异约需 40 场以上）；诊断指标噪声小，每组 10–15 场即可。
- **噪声**：混沌的模拟器中，微小的差别就会让比赛分叉，因此基线要用多个复本估计噪声。

## 当前进展与计划

| 顺序 | 内容 |
|---|---|
| 1 | 在乒乓上完整走一遍筛选流程（高层和底层），实现底层的技能在环 |
| 2 | 用同一流程筛选羽毛球、网球和足球 2v2，检验流程是否通用 |
| 3 | 把各项检验写成脚本，整理检查清单和报告模板，供合作者筛选其他运动 |

**尚未决定**：
- 每项运动最终设哪种模式（取决于第 ④、⑤ 步的结果）；
- 是否允许 agent 自行搜索和使用外部资源；
- 是否允许 GPU 训练；
- 不同指标如何合成排行榜；
- 是否另做全尺寸的真实版本。
