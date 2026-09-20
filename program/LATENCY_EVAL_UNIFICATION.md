# Profile latency 注入与统一 eval 链路工作记录

> 2026-09-14 核查：当前产物、具体评测数值和未核验事项见 [PROGRESS](PROGRESS.md)；下文旧运行状态按原日期阅读。

> 日期：2026-09-04（2026-09-06 review 更正）
>
> 状态：二次 pass 的共享容量/共享 sampler 结论已撤回，现恢复每 replica 独立模拟资源。
> 公共定向回归 127 passed；四任务 command-RL 尚未完成，见 [四任务中文交接](LATENCY_RL_HANDOFF.zh-CN.md)。
> 下文既有 native 实测仅对应当时版本，不构成新 command-RL 的验收。
>
> 相关文档：[PROFILE.md](PROFILE.md)、[MIKASA.md](MIKASA.md)、
> [HAIC.md](HAIC.md)、[EXTREME_PARKOUR.md](EXTREME_PARKOUR.md)、
> [profiling cookbook](../docs/profiling/cookbook.md)

## 1. 目标

将实机 profile 得到的 inference latency 注入 eval，同时保证 zero 和 profile replay 只改变
latency source，不再分别维护 WebSocket baseline evaluator、profile evaluator 和 task-local
latency loop。

最终入口统一为：

```text
scripts/profile.sh simulate
  -> cross_device protocol / profile resolver
  -> task-owned `latency-eval --eval-config`
  -> common `run_from_config()`
  -> common simulated/realtime executor
  -> task-owned EnvStepBackend
```

公共层负责 episode 生命周期、latency sampler、worker capacity、action scheduler、inference
pool、seed、logging 和 summary。任务层只负责原生 simulator 的 reset / observe / step、动作转换和
task metrics。

## 2. 为什么旧 no-latency WebSocket 与 `latency.method=zero` 不等价

旧 MIKASA VLA baseline 是独立的同步循环：env 直接通过 WebSocket 请求 server，收到 action 后
立刻 step。公共 zero evaluator 则经过 inference worker、worker capacity、action chunk、result
timeline 和 action scheduler。虽然两者都不人为增加等待帧，但原实现仍有这些结构差异：

- policy 位于不同进程，生命周期和随机数状态不同；
- diffusion 初始噪声曾依赖请求顺序及 batch composition；
- vector slot 的 reset/seed 曾错误复用，导致任务初态发生系统性变化；
- task-local loop 与公共 executor 对 action chunk、soft reset 和 metrics 的处理并非同一实现。

小实验给出了区分随机波动和系统偏差的依据：

| 路径 | MIKASA success |
|---|---:|
| 旧 canonical WebSocket baseline | `27/50` |
| 初始错误 vector seed 实验 | `49/50` |
| 修正 slot/episode seed 后的公共 zero 实验 | `31/50` |

`49/50` 相对 `27/50` 是系统性抬升，根因是 seed/reset 语义，不是 zero latency 本身。修正后
`31/50` 与旧 `27/50` 的四次成功差异可由环境与 diffusion RNG 波动解释，因此不要求逐 episode
或逐 bit 复刻旧 baseline。验收标准是同一配置语义下没有系统性偏离。

## 3. 关键决策

### 3.1 `measured` 与 `zero` 分开

- `executor.mode=realtime` 只接受 `latency.method=measured`；它记录真实 observation-to-action
  wall time，并允许真实 worker busy / capacity drop。
- `zero`、`temporal`、`iid`、`fixed`、`normal` 只由 simulated executor 消费。
- zero 表示 action 在虚拟时间中没有人为增加 ready frame；模型计算仍真实执行，但其 wall time
  不推进 simulator clock。

这样不再把“真实 inference wall time”命名为 zero latency，也不会让同一个配置因机器速度而改变
zero 的实验含义。

### 3.2 只保留一套 eval 调度实现

删除或停止使用以下重复路径：

- MIKASA WebSocket VLA eval / `_MikasaProfileEnv`；
- HAIC `dagger-eval` 的第二套 episode loop；
- Parkour `vla-eval` / `profile` 的第二套 episode loop。

PPO teacher、oracle 和 DAgger collection 保留，因为它们产生不同训练数据或 privileged-policy
指标，不是 latency eval 的重复实现。

### 3.3 task-native backend，不复制 simulator

- MIKASA 使用原生 ManiSkill GPU vector env，输出双 RGB view 和 7D action。
- HAIC 使用原生 TorchRL/Isaac backend，并在 task backend 内完成 256D latent 到 23D motor
  action 的冻结 actor 转换。
- Extreme Parkour 使用原生 Isaac Lab vector env；`step_no_reset()` 将 episode reset 所有权交给
  common executor，并保留 34D latent+yaw action contract。

同一个 backend 同时供 measured、zero 和 profile replay 使用。realtime 单 slot 只通过一个很薄的
adapter 接入既有 wall-clock executor；simulated 即使只有一个 slot 也走 batched executor，避免
MIKASA 再产生第三套逻辑。

### 3.4 temporal 是机器状态时间线，不按 episode 重置

IID 和参数分布按 episode 派生独立随机流；temporal burst model 表示一台机器连续变化的隐藏状态。
每个 vector slot 是独立实验副本，拥有独立 sampler 和随机流；同一副本的慢 regime/dwell
跨 episode 保留。episode reset 清空该副本的 pending commands、容量和 policy history，
不清除 temporal regime。物理 GPU batching 不构成副本间的模拟资源竞争。

### 3.5 区分 simulator seed 与 action-noise seed

历史 NVIDIA GR00T evaluator 曾为每个 observation 显式传入 `action_noise_seed`，使输出不依赖
batch 分组或 worker 完成顺序；该独立运行时现已移除。

Parkour DAgger 的 simulator 会 auto-reset，collector 并不掌握每次 reset 的独立 simulator seed；
因此不能把用于 diffusion 的 seed 冒充 `episode_seed`。字段已改名，剩余 `episode_seed` 只用于真实
episode/logging 语义。

### 3.6 profile 必须固定完整 artifact 身份

profile replay 从 profile provenance 读取 model/runtime 的 repo、path 和 checkpoint 坐标。
HF 下载使用当前仓库内容，不做版本固定；`--model-path` 与 `--hf-artifacts` 仍互斥。

GPU 分工也进入生成配置：`simulation_gpu` 注入 `env.simulator_device`；MIKASA 将其明确传给
ManiSkill 的 `sim_backend` 和 `render_backend`，inference devices 从剩余可见 GPU 分配。

## 4. 实现记录

公共实现的主要变化：

- `run_from_config()` 和 executor factory 接受 task-native `env_backend`；
- batched simulated executor 支持公共 `ProcessInferencePool.predict_batch()`；
- slot 到 inference worker 的映射稳定，physical batch size 由 `inference_batch_size` 控制；
- episode/soft reset 同时 reset worker 内 policy state；
- task metrics 和 step-weighted metric moments 进入公共 summary；
- realtime profile/eval 与 simulation replay 统一使用 `eval_episodes`，并统一使用 `eval_entrypoint`；
- profile replay 支持 `zero`、`temporal`、`iid`、`fixed`、`normal`。

任务配置：

| workload | eval episodes | simulator slots | inference batch |
|---|---:|---:|---:|
| MIKASA InterceptGrabFast | 100 | 1 | 1 |
| HAIC PullCart | 100 | 100 | 32 |
| Extreme Parkour Go2 | 100 | 50 | 8 |

MIKASA 保持单 simulator slot，是为了避免把 vector physics 差异混入本次 latency 对照；公共 batched
executor 仍被复用。HAIC / Parkour 使用它们原生的 GPU vectorization。

### 4.1 二次 pass 与后续 review 更正

2026-09-06 更正：realtime 在 episode 前 drain，再向全部物理 worker 清理该逻辑 slot。

- **撤回跨 replica 的 burst-entry 概率缓存**：每个副本独立 `HiddenRegimeSampler`，不再
  为交错调用维护 episode 概率表；同一副本保留跨 episode 的 temporal 状态。
- **撤回共享 worker 占用和固定模拟 worker 映射**：每个副本独立容量，选择 lowest-idle
  模拟 worker，episode/soft reset 清空占用。物理推理路由与 profile worker 身份分开。
- **全 burst slot 的 steady 分布回退到机器级 calm 池**（`latency_bench/latency/burst_profile.py`）。
  `steady: None` 时不再留给 sampler 崩溃：slot 自身无 calm 样本时用机器级 calm 池拟合；
  整个 profile 无任何 calm 样本时在 build 阶段报错（无效 artifact 不允许生成）。
- **profile artifact 路径 containment**（`latency_bench/cross_device/profile_resolver.py`）。
  `_download_profile_artifact` 的 `path_in_repo` / `checkpoint` / `model_config` 拼接前做
  absolute + `..` 逃逸校验，与非受信远端 profile 的来源一致。
- native eval entrypoint 统一使用调用者当前激活的 Python 环境，不在 registry 或 launcher 中切换解释器。
- **replay 使用 profile 记录的 `model_config`**（`profile_resolver.py`）。原来依据当前 registry
  spec 判断是否带 `model_config`；现在以 provenance 为准，registry 日后改名不再影响历史 replay。
- **HAIC contract test 的 torchrl 隔离修好**（`third_party/HAIC/tests/test_haic_vla.py`）。
  根因是 `patch.dict(sys.modules, ...)` 退出时按 clear 语义恢复，会把刚导入的
  `active_adaptation.vla.backend` 一并逐出 sys.modules，测试内再 import 时重新绑定真 torchrl 的
  `step_mdp`（torchrl 0.7 要求真 TensorDict API，fake TD 崩溃）。现在无条件注入 torchrl-free
  shim 并把 backend 模块重新注册，测试不再依赖环境中 torchrl 的版本行为；15/15 通过。
- **Parkour 终点 goal 索引不再静默 clamp**（`third_party/Isaaclab_Parkour/.../parkour_event.py`）。
  超出终端 padding band 的索引直接报错，保留最后 goal 的 padding band 作为设计内终端状态。
- **Parkour `_step` 的 active mask 恢复改为 try/finally**（`parkour_manager_based_rl_env.py`），
  异常路径不再残留 stale mask。

## 5. 实机结果

使用 profile：

```text
PROFILE_REF=qwenoft/1x-rpbzzz6/mikasa_intercept_grab_fast/instance_e906b84a82f9e086
RUN_ID=20260904T073201011403Z
```

temporal replay 完成 50 episodes：

| metric | result |
|---|---:|
| success | `0/50` |
| mean return | `0.34424` |
| inference completions | `1,300` |
| injected latency mean | `94.62 ms` |
| p50 / p90 / p95 / p99 | `92.61 / 133.67 / 152.39 / 203.29 ms` |
| max latency | `252.17 ms` |
| submitted / dropped observations | `1,300 / 1,700` |

结果目录：

```text
results/profiling/simulations/qwenoft/1x-rpbzzz6/
  mikasa_intercept_grab_fast/instance_e906b84a82f9e086/
  20260904T073201011403Z/temporal/0/
```

profile 本身的 realtime 结果为 `0/500`，旧 canonical zero 为 `27/50`。temporal replay 的
`0/50` 与约两帧的平均 latency、1,700 次 capacity drop 一致，说明 latency 已实际进入 eval，
而不是只写进 config 或 summary。

本次没有再跑一次完整 scalar zero。修正 seed 后的 `31/50` 已排除系统性 baseline 偏移；继续追求
`27/50` 的精确复现不会增加结论可信度。未完成的 scalar zero 目录和三个诊断目录保留在本地，未
删除、未作为正式结果发布。

## 6. 验证

历史二次 pass 全量：**1095 passed / 9 failed / 78 skipped / 124 deselected**。9 个 failure 已在 HEAD
detached worktree 上复现，全部为改动前即失败（flappy/cule 对齐、sample_factory composite
space、sweep-config 三个与本次改动无关的既有问题）；本次改动未引入新的 root 回归。
其中固化共享 worker/sampler 行为的测试已更正。2026-09-06 公共定向回归 127 passed；
factory suite 55 passed、1 项缺失可选 `flappy_bird_gymnasium` 依赖而失败。

native 验证：

- **Parkour 零延迟 smoke 已跑完**：`latency-eval --eval-config .../bootstrap/vla_eval/eval_config.yaml`
  （100 episodes / 50 slots / 50 Hz，checkpoint `bootstrap/.../flow_aux_5k_sft/checkpoint-5000`，
  runtime `model_14999.pt`）于 2026-09-05 01:11 UTC 完成，公共 summary
  `queue_eval_latency_0.json` / `queue_eval_results.jsonl` 已生成：逐步 100 episodes、
  mean_length 848.43、mean_return 10.57、`normalized_waypoint_progress` mean 0.385、
  `edge_violation` count 84,843。该 run 是 smoke，不评价策略质量。
- **HAIC remote 已同步完整 change set**（2026-09-05）：远端 `204.12.163.107:/home/xinyuan/
  latency-sensitive-bench-mikasa-h1/third_party/HAIC` 的 12 个变更文件已从本地 worktree
  覆盖（此前远端只有 09-04 14:40 的旧 `backend.py`，缺 matching base-env 改动，即
  `AttributeError: SimpleEnv has no attribute set_episode_seed` 的根因）。远端 native
  smoke **尚未执行**：按 HAIC.md 的 gate，代码经用户 review/commit 后再跑正式命令。
- HAIC contract tests **15/15 通过**（`third_party/HAIC/tests`，haic env + parent PYTHONPATH）；
  Parkour contract tests **12/12 通过**（`test_actor.py` + `test_vla_eval.py`；`test_camera.py`
  /`test_terrain_generator.py` 需要 Isaac EULA 交互与 omni 环境，不在本机无头环境跑）。
- `git diff --check` 主仓库与两个改动子模块均通过。

`flappy_bird_gymnasium` 未安装时，依赖该可选环境的 Flappy render equivalence 测试会在收集阶段
跳过；这不影响跨设备 artifact resolver 的回归测试。

## 7. 当前边界

- temporal replay 当前只保存在本地，没有代替用户发布结果；需要正式论文 artifact 时再使用
  `--publish`。
- 不承诺不同 simulator slot 数下轨迹逐 bit 相同；承诺 seed/reset contract 正确，且 zero 对照无
  系统性偏移。
- **native inactive-slot physics 未隔离**：HAIC 与 Parkour 的 vector env 都还是在单 scene 内
  全向量 step（IsaacLab `sim.step()` 全局，`write_data_to_sim`/`scene.update` 接受
  env_ids 但物理步进无法只推进子集）。active mask 保护的是计数/终止/奖励/观察噪声/每-slot
  RNG 与机器人状态历史；终端后尚未重启的 slot 会带着最后一次 action 继续被推进（不计入
  metrics）。真正的按 slot 物理隔离需要每-slot 独立 scene（MIKASA 的单 slot 架构），与
  "HAIC / Parkour 使用原生 GPU vectorization" 的目标冲突；当前结论是账户隔离已完成、
  物理隔离为已知边界。二次 pass 后不再把"只推进 active slots"写成已完成语义。
- **HAIC 远端 smoke 与 profile/temporal 正式 run 待执行**：HAIC native smoke 尚未验证
  "summary 已生成"；MIKASA 的 corrected zero（31/50）与 temporal replay（0/50）仍是
  语义检查，不是发布 artifact。
- 更换 GPU、driver、功耗策略、物理 GPU UUID 或 worker 数后必须重新 profile。
- submodule 改动（HAIC 12 文件、Isaaclab_Parkour 11 文件）需要分别 review/commit，**必须整树
  提交**（staged index 是旧迭代：只有 worktree 状态是自洽的），之后主仓库才能更新 gitlink；
  不能只提交 dirty gitlink。


## 固定 profile 均值 prefix

四任务共用策略请求合同，适用于单环境/批量 simulated、realtime 本地/环境子进程，
以及 Balance 的 DAgger 学生推理、development、held-out 和独立评估。

- `policy.action_prefix.mode: none`：不传显式 prefix；baseline 既有 state/action context 保留。
- `profile_mean` 必须提供独立的 `policy.action_prefix.profile_path`。
  配置入口从原始 `summary.mean_ms` 计算 `K=ceil(mean_ms*env_fps/1000)`，要求 `0≤K<H`。
  保存解析后的 mean_ms、env_fps、frames、action_horizon、action_dim 和 profile_path；
  同一运行及其恢复使用这份冻结配置，不因采样延迟或实时预热重新选择 K。
- 实际 simulated d 仍由 `latency` 配置采样；realtime d 为实际完成耗时。
  参考 profile 不替代实际延迟来源。K=0 使用全 false mask；K≥H 不截断、不自动扩展模型。
- prefix 只根据观测帧时控制器已收到的动作投影，复用 scheduler 的 startup/hold/ordering。
  不访问未返回 timeline；到达时仍按 observation timestamp 跳过过期 head，不额外等待 K。
- realtime 在环境侧固定输入，prefix 在共享内存模板建立前加入；父进程仅转发。
  预热与正式请求输入结构一致，episode 开始重置调度状态。
- Balance 增广 state 的公共环境适配保留旧 chunk 与当前位置；模拟训练 rollout 继续复用
  原来的任务包装。MuJoCo EGL 在初始化前设置；其他任务保持原渲染器。

统一独立入口：`python -m scripts.starvla.evaluate --task TASK --bundle BUNDLE
--environment ENV_CONFIG --output NEW_DIRECTORY --prefix-mode none
--vla-python VLA_PYTHON --backbone BACKBONE`。
conditioned 使用 `--prefix-mode profile_mean --prefix-profile PROFILE`。
通用任务由 `--task-python` 指定原生任务环境 Python；默认是当前 Python。
原生配置决定 executor 模式；realtime 的 `latency.method=measured`、单环境设置沿用已有合同。
Balance 的固定均值补评测 seeds/FPS/指标参见 [BALANCE_CAUSAL_EVAL](BALANCE_CAUSAL_EVAL.md)。

新 Balance 训练显式指定 `--horizon 8` 或 `--horizon 16`。conditioned teacher 私有分支先
回放与学生请求相同的 K 个动作，再生成 suffix；prefix 不计入 loss，终止后不提供监督。
真实随机 d 仅影响在线轨迹及执行后的日志，不进入学生输入或决定标签 mask。
这条标签分支是反事实监督，不要求与真实执行轨迹逐帧相等。

旧数据、checkpoint 与实验结果保持原样。旧模型可在新输出目录重评；旧 run 不原地恢复
为新协议，也不复用旧 evaluation/history 做新 best 比较。新 run 的 environment.json 与
数据 label_provenance、导出的 task_contract 与 rollout 结果均保存 action_prefix 合同；
checkpoint bundle 沿用同一 task_contract，恢复沿用原配置。此次迁移本身不表示重训、
正式评测或发布完成。

实现验证：[固定 K 公共合同回归记录](../outputs/tasks/humanoidbench_balance_simple/fixed_k_unification_20260920/README.md)。
