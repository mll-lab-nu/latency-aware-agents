# Pi0.5 pipeline 支持审计

2026-09-13。范围：AirRaid、InvertedPendulum、Hopper、HalfCheetah、Walker2d、Ant、Humanoid。只检查现有代码、配置和本地测试；没有启动训练、GPU profiling、数据采集或上传。主仓库 `bfd0560a24c9bab9fcef63ff4d632c64876a7889`，StarVLA `f364fdf080434aea14bb2a19931e017f0af32d87`。保留当前工作区已有修改。

结论：**基础组件多数具备，但还没有可直接执行这 7 个环境完整 pipeline 的 Pi0.5 配方。缺口集中在模型身份、初始化与原生维度适配、环境入口、profiling 配置，以及真实训练/闭环验收。**

## 1. 当前 Pi0.5 path 是什么

`model=pi05` → `rl_games.model_alias=pi-0.5` → `QwenPI_v3`。这是仓库现有的 StarVLA 实现：Qwen3-VL-4B、逐层视觉语言特征投影、LayerwiseFM flow-matching 动作头。当前别名不代表已经接入官方 OpenPI 原生 π0.5。

配置引用的初始化模型是 `StarVLA/Qwen3VL-PI_v3-Bridge-RT_1`，文件为 `checkpoints/steps_50000_pytorch_model.pt`。此次没有联网下载或核验该模型当前 revision/实际 tensor shapes，因此这是**配置中的来源**，不是本次已验收的 checkpoint。

现有两条相关路径：

- **RL-games 路径**：`third_party/starVLA/examples/rl_games/scripts/launch_train.py` + `config/model/pi05.yaml`。有旧游戏的训练命令、Bridge 初始化、共享部署 wrapper 和 profiling preset。
- **H1 原生路径**：`scripts/starvla/model_config.py::make_model_config` + `scripts/starvla/train_task.py`。明确支持 `qwenpi_v3`、任意指定的 state/action 维度、连续 state encoder、单步动作和 L0 eval。完整脚本目前只接受 balance / parkour / mikasa / haic，数据合同是 `task_contract.json`，不能直接当作这 7 个环境的 launcher。

源代码：[别名映射](../../../third_party/starVLA/starVLA/training/rl_games/alias.py)、[Pi05 配置](../../../third_party/starVLA/examples/rl_games/config/model/pi05.yaml)、[模型实现](../../../third_party/starVLA/starVLA/model/framework/VLM4A/QwenPI_v3.py)、[H1 配置构造](../../../scripts/starvla/model_config.py)。

## 2. 对目标 pipeline 的逐阶段支持

| 阶段 | 当前支持程度 | 尚需完成的工作 |
| --- | --- | --- |
| adaptation | 7 个环境已有通用 adapter / action / state / RGB 合同，可复用 | 固定 Pi05 使用的同一环境与数据合同；无需重写环境 |
| zero-latency small model training & eval | Sample Factory 通用训练、zero latency 和评估已具备，独立于 VLA 家族 | 相同 clock / seed / reward / gate 下可以复用已有 L0 teacher 证据；如要求重跑，使用同一配方 |
| zero-latency data rollout | 通用 exporter、筛选、RGB+state、native/discrete LeRobot 转换已具备 | 检查已有 L0 数据是否与 Pi05 输入与时序一致；模型名称变化本身不要求重新采集 |
| zero-latency Pi05 training | 框架 forward/loss、通用 trainer/loader 已实现；7 环境完整配方未就绪 | 定义兼容的初始化、action/state 维度、state encoding、预算；补通用 Pi05 入口 |
| zero-latency Pi05 eval | 共享 `policy.type=starvla` 支持模型加载及离散/连续解码 | 用真正训练出的 Pi05 bundle 跑 matched seeds 闭环；现有 generic env preset 默认关掉训练内环境评估，需外部显式执行 |
| profiling | `pi05` 已注册；QwenPI_v3 有 backbone/projector/sampler 分段计时；canonical 测量框架可复用 | 这 7 个环境均没有 Pi05 workload preset，必须绑定新 L0 bundle 后实测 |
| baseline Pi05 eval | measured realtime executor 和部署接口具备 | 在目标 GPU、clock、seed、horizon、renderer 下显式运行；不能用 OpenVLA latency/return 替代 |
| profile-latency small model training & eval | 已有 IID training、profile handoff、simulated worker-pool eval | 必须用新 Pi05 profile；保持 action-delay training 与 worker-occupancy eval 的区别 |
| profile-latency rollout | 通用 exporter 支持显式 latency config | 新 teacher 的质量判断、有限 probe、门槛、采集与 replay 验证重新执行 |
| profile-latency Pi05 training & eval | 可复用前述训练/评估组件 | 使用本次数据和统计，固定 L0/profile 两侧模型配方；完成 checkpoint reload、最终 measured eval 和证据保存 |

共享执行依据：[现有 runbook](../profile-latency-new-plan.md)。已有 OpenVLA profile-latency teacher/data 与其测量分布绑定，不能直接当成 Pi05 profile-latency 数据。

## 3. 直接切换模型会碰到的具体问题

### 初始化入口没有覆盖原生 7 环境

`train_gymnasium_openvla.sh` 把 native 数据映射为 `init=scratch`；把 model 名改为 Pi05 会在 `validate_rl_games_config` 被 `pi-0.5 scratch is not supported` 拒绝。

`init=bridge` 固定 `action_carrier=bridge`、7D action、默认 7D state、H=1。实际调用 `apply_action_spec` 后，Ant(8D) 和 Humanoid(17D) 明确报错。其余环境维度不超过 7 也只代表通过这一项检查，不代表 native 数据能直接喂给 7D Bridge head。

| 环境 | 当前数据 active action 维度 | state 维度 | 直接套用旧 Bridge 配方 |
| --- | ---: | ---: | --- |
| AirRaid | 6（离散动作分数） | 不使用 | 要匹配 action carrier，并显式禁用无意义 state |
| InvertedPendulum | 5（当前离散化动作合同） | 4 | 需要保留已有离散化定义，并适配 state/action |
| Hopper | 3（连续） | 11 | 需要适配原生 action/state |
| HalfCheetah | 6（连续） | 17 | 同上 |
| Walker2d | 6（连续） | 17 | 同上 |
| Ant | 8（连续） | 27 | 7D carrier 直接拒绝 |
| Humanoid | 17（连续） | 376 | 7D carrier 直接拒绝 |

现有 `backbone_bridge_factorized11` 提供了可复用机制：只加载 `qwen_vl_interface`，重新初始化 projector 和 action head。但该 preset 自带 DeadlyCorridor 的 11D 设置，不能原样套用。选择“只加载 VLM”“保留可兼容 projector/DiT”“完整 Bridge 初始化并显式转换 carrier”会改变科学配方，正式训练前必须写明，不能偷偷跳过不兼容 tensor。

源代码：[action spec](../../../third_party/starVLA/starVLA/training/rl_games/action_spec.py)、[初始化验证](../../../third_party/starVLA/starVLA/training/rl_games/config_validation.py)、[部分加载 preset](../../../third_party/starVLA/examples/rl_games/config/init/backbone_bridge_factorized11.yaml)。实际配置检查结果：[contract-check.json](contract-check.json)。

### 模型维度、数据维度和归一化必须一起绑定

QwenPI_v3 默认可配置 `action_dim=32`，但 generic Gymnasium loader 不会自动把 native 数据补为 32D。LayerwiseFM 虽然只对 `action_env_dim` 计算 loss，但动作编码器仍要求输入宽度等于 `action_dim`；连续动作 inverse normalization 也要求输出宽度与统计一致。因此 active-dim masking 不能代替 carrier 适配。

state 可以走默认 `discretized_text`，也可以走 action head 的连续 encoder；两条均有实现。7 环境要显式绑定 `uses_state/include_state`、state_dim、manifest 中的归一化和 `dataset_statistics.json`。改变 state/action encoder 宽度还涉及预训练 checkpoint 形状，不是 `strict=False` 就会自动兼容。

依据：[generic data registry](../../../third_party/starVLA/examples/rl_games/train_files/data_registry/data_config.py)、[LayerwiseFM](../../../third_party/starVLA/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py)、[norm processor](../../../third_party/starVLA/deployment/model_server/policy_norm_processor.py)。

### 现有动作和 latency 语义

旧 RL-games Pi05 Bridge 配方强制 H=1；虽然模型本身能输出 action chunk，当前 generic `StarVlaPolicyRunner` 只消费 `actions[index, 0]`。沿用已有逐动作 pipeline 可使用 H=1；如要执行完整 chunk，需要另外定义调度合同，不能仅把模型 horizon 改为 16。

现有 Pi05 profiling 只注册 flappy / demon_attack / deadly_corridor，且 `calibrate_env_fps=true`。如本轮仍固定 10 FPS，需禁用自动校准并同时设置 env/obs clocks。`num_inference_timesteps=4` 直接影响每次推理的 sampler 成本；不能套用 OpenVLA 实测值。模型 forward 还硬编码 `repeated_diffusion_steps=2`，修改同名配置并不会改变该训练重复次数。

依据：[profiles.yaml](../../../configs/profiles.yaml)、[policy decode](../../../latency_bench/policy/starvla.py)、[QwenPI_v3 forward](../../../third_party/starVLA/starVLA/model/framework/VLM4A/QwenPI_v3.py)。

## 4. 本次验证与边界

- 实际执行了模型别名、Bridge action-spec 和 scratch validation 检查：Ant/Humanoid 的 Bridge 失败可复现；7 环境 Pi05 preset 均不存在。
- `test_config_validation.py`：8 passed。
- `test_starvla_model_config.py` + `test_starvla_train_task.py`：9 passed。覆盖 H1 配置/调度合同，不能代表真实训练成功。
- generic dataset loader + policy wrapper：53 passed、1 failed。失败是旧 `from_pretrained` mock 未接受现实现传入的 `backbone_path`。
- generic live policy：12 passed、1 failed。失败是测试 mock 仍返回 2 个 decode 对象，而实现现在解包 3 个。
- profile protocol：21 passed、4 failed。失败为旧测试的 episode-count / FPS calibration 期望与当前 registry 不一致。
- 普通 `.venv` 下 `test_starvla_policy.py` 因缺 OmegaConf 整模块跳过；临时引用现有 Anaconda 的纯 Python OmegaConf/ANTLR 后，10 passed、32 failed，失败入口为缺 `torchvision`。没有安装或修改项目依赖。这批结果不构成 Pi05 模型行为失败的证据。
- 完整 Hydra Pi05 composition/SFT 测试及真实 forward/backward/模型重载未完成；本机缺 Hydra 和完整 StarVLA 依赖。没有进行 CUDA、3090/H100 runtime 或 Hub 已训练模型的验证。旧配置中的 checkpoint 路径只能证明已有实验入口，不能据此声称这 7 个环境已经训练成功。

## 5. 补齐顺序

1. 明确继续使用当前 `QwenPI_v3` 别名实现，还是要求官方 OpenPI π0.5；后者需要另一条集成路径。
2. 为 native Gymnasium 定义一套明确的初始化策略，复用已有模块级加载机制；对齐 state/action/carrier/H=1/图像/统计，不引入第二套环境 adapter 或 exporter。
3. 给现有 generic training 入口增加 Pi05 模型选择，固定 7 环境 manifests 与训练配方；修复相关过时测试并跑配置合同检查。
4. 在目标 GPU 上完成一个真实 batch 的 forward/backward、checkpoint 保存/重载及短闭环，分别覆盖连续动作和离散动作；这一步决定是否具备正式执行条件。
5. L0 模型验收后补 Pi05 的 7 个 profiling workload，并按现有 runbook 执行其余阶段。

这次只交付审计，不修改上述实现、不启动新实验、不提交或推送 Git。
