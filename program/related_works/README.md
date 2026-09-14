# Delay RL / asymmetric critic references

此前已做 FastTD3 critic-only transport augmentation；当前新增了 [capacity-aware DCAC 实验](../../scripts/experiments/dcac/README.md)，
同时增广 actor/value 输入并进行因果前缀重采样。不把有限历史增广等同于无偏或完整 Markov 状态。

| 文献 | 解决的问题 | 与当前实验的关系 |
|---|---|---|
| [Revisiting State Augmentation](stochastic_delay_state_augmentation.md) | 随机延迟下的信息状态与策略学习 | 原方案也增广 policy，不能当作 critic-only 保证 |
| [Delay-Correcting Actor-Critic](dcac.md) | 随机延迟下的 credit assignment / 样本效率 | 新实验移植因果前缀重采样，并适配 capacity-one 的 busy admission |
| [Unbiased Asymmetric RL](unbiased_asymmetric_rl.md) | 部分可观测环境中的非对称价值估计偏差 | 历史与状态有不同职责，特权状态不自动消除偏差 |



### 1. latency setting 下微调 VLA
https://arxiv.org/pdf/2607.24008
https://arxiv.org/pdf/2608.23831


### 2. latency setting 下对 small teacher model 进行 RL (或者说单纯研究 realtime RL 的)
https://arxiv.org/pdf/2010.02966
https://arxiv.org/pdf/1911.04448


### 3. small model as teacher / expert -> transfer to VLA
https://arxiv.org/pdf/2412.09858
https://arxiv.org/pdf/2406.19972
https://arxiv.org/pdf/2510.03142v1
https://openreview.net/pdf?id=sciA_xgYofB

### 4. RL with action chunking
https://arxiv.org/html/2507.07969


5. 用一个公式说清楚什么叫“可 transfer”

设真实环境 state 是：

$$ x_t. $$

small teacher 看到：

$$ h_t^T $$

VLA 看到：

$$ h_t^S. $$

teacher：

$$ a_t^T\sim \pi_T(a\mid h_t^T). $$

我们希望通过 BC 找到：

$$ \pi_S(a\mid h_t^S). $$

最理想的情况是存在某个函数：

$$ f $$

使得：

$$ \boxed{ a_t^T \approx f(h_t^S) } $$

也就是说：

$$ H(A_T\mid H_S) $$

足够小。

一个失败例子

假设 teacher 能看到：

$$ z_t\in\{\text{ball moving left},\text{ball moving right}\}. $$

于是：

$$ z_t=L \Rightarrow a_t=-1 $$ $$ z_t=R \Rightarrow a_t=+1. $$

但 VLA 输入里看不到 \(z_t\)。

对 VLA 来说两个 sample 都是：

$$ h_t^S=h. $$

dataset 于是变成：

$$ h\rightarrow -1 $$

同时：

$$ h\rightarrow +1. $$

那么不管 VLA 是 1B、7B 还是 100B：

$$ \boxed{\text{它都不可能知道该选哪个}} $$

L1/MSE 最后可能学成：

$$ a\approx0. $$

diffusion/flow 可以学成双峰：

$$ p(a\mid h) $$

但它也只是随机选左右，并不能知道真实 ball 在哪边。

所以：

更大的 model capacity 解决不了缺失的信息。

6. 这正好解释了为什么 “same observation” 不是正确条件

same observation：

$$ h_T=h_S $$

当然是一个非常方便的 sufficient condition。

但不是 necessary condition。

例如 teacher 看到：

$$ \text{exact object }(x,y,z) $$

student 看到：

$$ \text{RGB}. $$

只要 RGB 足够清楚，使得：

$$ (x,y,z) \approx g(I), $$

那么：

$$ \pi_T(x,y,z) $$

原则上就能变成：

$$ \pi_S(I). $$

HumanVLA 和 RLDG 都是这种情况。

所以更精确的条件应该叫：

$$ \boxed{ \textbf{student-observable sufficiency} } $$

而不是：

$$ \text{observation equality}. $$
7. 但这恰好让你们 latency setting 变得危险

现在假设你们 small RL policy 学：

$$ a_t^T = \pi_T( s_t^{exec}, a_{t-K:t-1}, d_t ). $$

它可以看到真正 execution-time state。

但 VLA 最后只输入传统：

$$ \pi_{\rm VLA} ( I_{t-d}, p_{t-d}, L ). $$

那么：

$$ \boxed{ H( A_T \mid I_{t-d}, p_{t-d}, L ) } $$

可能非常大。

因为：

$$ I_{t-d},p_{t-d} $$

之后发生了什么，student 根本不知道。

于是 small policy 越厉害，反而越可能成为一个：

$$ \boxed{\text{Impossibly Good Expert}} $$
8. 一个 latency-specific 例子

假设 VLA 在 \(t=0\) 拍了一张图：

ball
  ↓
robot

然后 inference 200 ms。

在这期间存在两个 rollout。

Trajectory A

球往左：

$$ s_{200ms}^{A} $$

teacher：

$$ a_T=\text{reach left}. $$
Trajectory B

球往右：

$$ s_{200ms}^{B} $$

teacher：

$$ a_T=\text{reach right}. $$

但 VLA training input 在两种情况下都是：

$$ (I_0,p_0,L). $$

于是你的 BC 数据本质上：

$$ (I_0,p_0,L) \rightarrow \{\text{left,right}\}. $$

这不是 SFT tuning 不好。

是 supervision 本身不可实现。

这也是为什么你前面讨论 FutureRTC / ARLI 时，加：

$$ a_{\rm history} $$

或：

$$ s_{\rm fresh} $$

是一个理论问题，而不只是 architecture trick。

9. 所以你们的 small model teacher 设计不能随便用 privileged state

这一点我现在觉得特别重要。

你可能原来想：

small RL 要效率高，所以直接给 simulator state，先把 latency-aware policy 训出来；然后 VLA 去 BC。

这个想法只有在：

$$ \boxed{ \text{teacher 使用的 task-relevant information 能被 VLA reconstruct} } $$

时才可靠。

否则 teacher 会过度利用 privileged state。

最近甚至有专门研究这个问题的工作。

Student-Informed Teacher Training，ICLR 2025 Spotlight 就说：独立训练 privileged teacher 时，teacher 很容易发展出 student 无法 imitation 的行为，因此他们把 teacher/student 联合训练，并在 teacher reward 里加入 teacher-student action discrepancy penalty，让 teacher 主动变得 imitable。

2026 年又有 Teacher-Student Representational Alignment for RL-Driven Imitation Learning，直接把这种现象称为 irreducible imitation gap，并试图让 teacher 只依赖 teacher/student shared representation。

这两个和你们的问题非常相关。

10. 我现在会把 transfer 条件总结成这张表
条件	是否必须相同	对你们的含义
Model size	❌	small → large 完全没问题
Input modality	❌	state teacher → image VLA 已有大量 precedent
Language input	❌	specialist 可按 task 分开训，rollout 时附 prompt
Task / reward semantics	✅	teacher 和 VLA 必须解决同一个任务
Action semantics	基本 ✅	joint / EE frame / normalization / control mode 必须对齐
Action temporal semantics	✅	\(a_t\) 到底作用在哪个 wall-clock state 必须一致
Student-observable information	✅	student 必须有足够信息推断 teacher decision
Same raw observation	❌	sufficient，但远非 necessary
Same closed-loop state distribution	不完全，但很重要	否则 BC covariate shift
Dataset coverage	✅	最好 DAgger/on-policy distillation
Same latency	对你们几乎 ✅	至少 teacher 必须在 target VLA latency dynamics 下学

最后这一条，我认为是你们项目最值得重点抓的。

11. “same latency”到底是什么意思？

不是说：

small model 也必须真的运行 200 ms。

完全不需要。

small policy 可以 0.2 ms 算完。

但是environment wrapper 必须让它生活在和 VLA 相同的 delayed control process 中。

例如目标 VLA：

$$ d_t\sim P_{\rm VLA}(d) $$

平均 150 ms。

那么 small teacher 虽然瞬间算完：

$$ a_t=\pi_T(o_t) $$

你也故意让 action：

$$ 150ms $$

以后才真正进入 actuator，或者让 controller 在这期间继续执行 committed action。

也就是：

small teacher
     │
  1 ms compute
     │
     ▼
delay emulator ───── 150 ms ─────► execution

而不是：

small teacher ─── 1 ms ───► execution

否则你训练得到的其实是：

$$ \pi_T^{1ms}. $$

然后却要求 VLA 学：

$$ \pi^{150ms}. $$

两者对应的是不同的 closed-loop MDP。

12. RLDG 在这里其实是一个很好的 warning

RLDG 中：

$$ RL=10Hz, \qquad OpenVLA=4Hz $$

居然还能 transfer。

这说明：

control frequency 不完全一致并不自动导致 transfer failure。

但它们的主要任务是 connector insertion / contact-rich manipulation，环境变化相对慢，而且底层还有：

$$ 1kHz $$

impedance controller。

所以这个结果绝对不能推导出：

“在 interception / locomotion / balance 这种 latency-sensitive dynamic task 上，10Hz teacher → 4Hz VLA 也没问题。”

恰恰这是你们应该验证的东西。

13. 你们最好不要直接做 “full privileged teacher → vanilla VLA”

我现在更推荐一个 information-matched teacher。

Teacher actor：

$$ \pi_T( z_t^{shared} ) $$

其中：

$$ z_t^{shared} = [ \text{VLA-recoverable state}, a_{\rm history}, d_t ]. $$

而 critic 可以继续使用完整 simulator privileged state：

$$ V( s_t^{priv} ). $$

也就是经典 asymmetric actor-critic：

$$ \boxed{ \text{actor：student-compatible information} } $$ $$ \boxed{ \text{critic：full privileged information} } $$

这样你仍然可以利用 simulator state 加速 RL，但是不会让 teacher actor 学会依赖 VLA 永远不可能知道的变量。

14. 对你们甚至可以做三级 teacher

这会成为一个非常漂亮的 ablation。

Oracle Teacher
$$ T_{\rm oracle} : s_t^{true} \rightarrow a_t $$

所有 execution-time privileged state 都能看到。

它可能 performance 最高，但：

$$ \text{imitability lowest}. $$
Shared-information Teacher
$$ T_{\rm shared} : [ o_{\rm stale}^{state}, a_{\rm hist}, d ] \rightarrow a_t. $$

只使用 VLA 理论上能够获得的信息。

Belief Teacher
$$ T_{\rm belief} : [ o_{\rm stale}, a_{\rm hist}, o_{\rm hist}, d ] \rightarrow a_t. $$

加入和最终 VLA 一样的 memory structure。

然后分别 distill 到 VLA。

如果得到：

$$ R(T_{\rm oracle}) > R(T_{\rm shared}) $$

但是：

$$ R(VLA\leftarrow T_{\rm oracle}) < R(VLA\leftarrow T_{\rm shared}), $$

这是一个特别漂亮的结果：

Better RL teacher does not necessarily make a better VLA teacher.

因为：

$$ \boxed{ \text{teacher optimality} \neq \text{teacher imitability}. } $$

这基本就是 SITT / Impossibly Good Expert 的 insight 在 latency-VLA 上的新 manifestation。

15. 第二个条件：action interface 比 observation interface 更需要一致

这个经常容易被忽略。

RLDG 能这么直接：

$$ RL\rightarrow OpenVLA $$

一个非常重要的原因就是所有 policy 共享：

$$ \boxed{ \text{same 6D wrist-frame action space} } $$

如果你们 small policy 是：

$$ a_t^{small}\in\mathbb R^{43} $$

50 Hz joint target；

而 VLA 是：

$$ A_t^{VLA} = [a_t,\ldots,a_{t+39}] $$

2.5 Hz 产生一个 40-step action chunk，

那么你不能简单说：

每个 small action 都作为一个 SFT target。

真正需要匹配的是：

$$ \boxed{ \text{action abstraction} + \text{control rate} + \text{chunk timing} + \text{execution scheduler}. } $$

否则 teacher 和 student 实际上在解两个不同 decision problem。

16. 对 action chunk VLA，我会这样生成 teacher data

假设 student 每次输出：

$$ A_t = [ a_t, a_{t+1}, \dots, a_{t+H-1} ]. $$

那么 small policy rollout 时最好收集：

$$ a_t^T, a_{t+1}^T, \dots, a_{t+H-1}^T $$

作为：

$$ A_t^T. $$

但注意：

不应该简单从一个 zero-latency rollout 随便截 H 个 action。

它们应该来自和 VLA deployment scheduler 相同的 delayed rollout。

换句话说，dataset 单位最好是：

$$ \boxed{ ( o_{\rm capture}, \text{latency context}, A_{\rm committed}, A_{\rm target}^{teacher} ) } $$

而不只是：

$$ (o_t,a_t). $$
17. 第三个问题：offline SFT 很可能不够

HumanVLA 已经明确发现 offline BC 出现严重 covariate shift，所以改成 DAgger。

今年的 VLA-OPD 又是同样的故事：

传统 offline SFT 只训练：

$$ s\sim d_{\rm teacher}. $$

但 deployment 时：

$$ s\sim d_{\rm VLA}. $$

一点误差累积以后 VLA 就进入 dataset 没有的 state。

所以他们让 VLA 自己 rollout，在：

$$ s\sim d_{\pi_{\rm VLA}} $$

这些 state 上再请求 teacher supervision，也就是 on-policy distillation。

对于你们 latency task，这个问题应该比普通 manipulation 更严重。

因为一个迟 100 ms 的小误差可能很快导致：

$$ \text{phase shift} \rightarrow \text{larger state mismatch} \rightarrow \text{fall / miss}. $$

所以我会把：

$$ \boxed{\text{SFT → DAgger}} $$

视为你们 pipeline 的重要升级路径，而不是 optional trick。

18. DreamMimic 又给了你另一个很相关的方向

刚出的 DreamMimic (2026-08) 做的是：

$$ \text{privileged teacher} \rightarrow \text{vision-based humanoid student}. $$

他们直接说难点来自：

partial observability；
contact-rich dynamics；
long-horizon behavior。

于是没有只做 vanilla BC，而是加入 world-model latent：

$$ h_{t+1} = f(h_t,o_t,a_t) $$

并用：

action-conditioned multi-step prediction；
privileged state prediction；
contact prediction；
object state prediction；
reward prediction；

帮助 student 学出能够携带历史状态的 predictive representation。

这和你们刚才讨论 latency memory 的联系非常直接：

当 single VLA observation 不足以解释 privileged teacher behavior 时，history/belief model 可以把本来不可直接观察的信息变成可推断的信息。

19. 甚至已经有 dynamic future-information distillation

最近的 ForeTime-VLA 做 conveyor-belt moving-object manipulation。

它 offline 使用可以看到未来的 teacher，得到 future-aware representation；deployment student 当然不能看 future，于是用一个 8-frame causal history encoder 去预测这个 future-aware target。最终在 moving belt 上显著提升抓取表现。

它不是：

$$ \text{small RL teacher} $$

而是 future-aware / WAM-style teacher，

但概念上非常接近：

$$ \boxed{ \text{privileged future information} \rightarrow \text{causal history-based VLA} } $$

这进一步说明：

history 能不能让 student reconstruct teacher information，本身就是 transfer 成败的核心。

20. 所以目前 literature landscape 我会这样看
Work	Teacher	Student	Obs 是否相同	怎么解决 gap
HumanVLA 2024	privileged state RL	vision-language-action humanoid	❌	DAgger + active rendering + last action
RLDG 2025	specialist RL	OpenVLA / Octo	❌	RL rollout SFT，共享 action space
MM-Nav 2025	privileged RL experts	multi-view VLA	❌	iterative/online expert data
SITT 2025	privileged RL teacher	visual student	❌	让 teacher 主动变得 imitable
VLA-OPD 2026	expert	pretrained VLA	可不同	student on-policy distillation
DreamMimic 2026	privileged humanoid teacher	visual humanoid policy	❌	recurrent world model / predictive latent
ForeTime-VLA 2026	future-aware teacher	causal VLA	❌	temporal history reconstruct future info

所以这个领域已经非常明确：

$$ \boxed{ \text{RL teacher → VLA is feasible} } $$

但：

$$ \boxed{ \text{privileged behavior → limited-observation student} } $$

是否 feasible，是一个独立且很深的问题。

21. 对你们而言，我认为最重要的是一个 “transferability diagnostic”

甚至在开始训 VLA 之前，你们就能检测这个假设。

收集 small teacher rollout：

$$ D= \{ h_t^{VLA},a_t^T \}. $$

然后训练一个很小的 probe：

$$ q_\phi(a_t^T\mid h_t^{VLA}). $$

依次比较：

$$ h_0= [I_t,p_t,L] $$ $$ h_1= [I_t,p_t,L,d_t] $$ $$ h_2= [I_t,p_t,L,d_t,a_{t-K:t-1}] $$ $$ h_3= [h_2,\text{obs history}] $$ $$ h_4= [h_2,p_{\rm fresh}] $$

看 held-out teacher-action prediction error。

如果：

$$ \mathcal E(h_0) \gg \mathcal E(h_2), $$

你就有很强的 evidence：

traditional VLA observation is insufficient for transferring latency-aware behavior; action memory makes the teacher substantially more imitable.

这个实验便宜得离谱，但是理论意义很好。

甚至可以定义：

$$ \boxed{ \text{Imitability Gap} = \mathcal L(a_T\mid h^{student}) } $$

作为你们 paper 里的 diagnostic metric。

22. 我现在对你们 pipeline 的推荐版本

不要：

$$ \text{full-state zero-latency RL} \rightarrow \text{vanilla VLA SFT}. $$

而是：

$$ \boxed{ \text{target-latency small RL} } $$

↓

small teacher 在和目标 VLA 一样的 delayed scheduler中训练；

↓

actor 尽量使用：

$$ \boxed{ \text{student-recoverable information} } $$

如：

$$ [ s_{\rm stale}, a_{\rm history}, d ] $$

privileged information 可以给 critic；

↓

收集和 VLA observation 同步的：

$$ [ I_{\rm stale}, p_{\rm stale}, L, a_{\rm history}, d ] $$

以及 teacher action chunks；

↓

先 SFT；

↓

再：

$$ \boxed{\text{DAgger / on-policy distillation}} $$

在 VLA 真正产生的 delayed states 上让 small teacher relabel。

整个结构我会概括成：

$$ \boxed{ \textbf{Latency-Equivalent Teacher Distillation} } $$

而不是简单：

$$ \text{RL data → VLA}. $$
23. 最后：这个 gap 我目前没看到被直接做掉

我重点搜了：

specialist RL → VLA；
privileged RL → VLA；
teacher/student observation mismatch；
inference latency + distillation；
VLA latency + RL teacher；

目前没有找到一篇把下面四件事完整连起来的工作：

$$ \boxed{ \begin{aligned} &1.\ \text{small RL teacher learns under target inference latency}\\ &2.\ \text{explicitly studies teacher→VLA information mismatch}\\ &3.\ \text{distills latency-aware behavior into a slow VLA}\\ &4.\ \text{tests whether the behavior survives actual VLA wall-clock latency} \end{aligned} } $$

RLDG 有：

$$ \text{small RL}\rightarrow\text{VLA}, $$

但没有系统研究 latency-sensitive behavior。

FutureRTC / ARLI 有：

$$ \text{VLA latency}, $$

但不是你们这种：

$$ \text{small latency-aware RL} \rightarrow \text{VLA}. $$

HumanVLA / SITT / DreamMimic 有：

$$ \text{teacher-student information asymmetry}, $$

但没有 inference-latency MDP。

所以我现在反而觉得，你们真正值得抓的并不是“small model 可以 transfer 到 VLA”这个 claim——这个已经有人证明很多次了；而是“latency-aware behavior 在什么 information/timing conditions 下才可 transfer”，以及怎么构造一个 latency-equivalent、VLA-imitable 的 small teacher。

这比一开始单纯说“先 RL small model，然后 BC VLA”强很多，而且可以直接连上我们前面三条 literature：

$$ \boxed{ \text{Random-delay MDP} } \rightarrow \boxed{ \text{Impossibly Good Expert / SITT} } \rightarrow \boxed{ \text{RLDG / HumanVLA} } \rightarrow \boxed{ \text{FutureRTC / ARLI} } $$

这已经开始形成一条挺完整的方法论故事了。

这批工作更新非常快；如果你愿意，我也可以持续监控 arXiv 上 RL→VLA distillation / latency-aware VLA / privileged-to-visual policy distillation 的新论文，有真正撞你们路线的就提醒你。