# 五环境历史资产交接

本文是 Flappy Bird、Demon Attack、Deadly Corridor、Atlantis 和 Asterix 历史
资产迁入中央 Hugging Face 仓库的长期账本。它区分“源资产可下载”“正式评测完成”
和“中央发布后回读通过”；训练期间的 checkpoint 选择值不当作独立评测成绩。

缩写：O = OpenVLA，G = GR00T，π = π0.5。

中央仓库：

- 模型：[`placeholder/benchmark-models`](https://huggingface.co/placeholder/benchmark-models)
- 数据：[`placeholder/benchmark-datasets`](https://huggingface.co/datasets/placeholder/benchmark-datasets)
- Profile：[`placeholder/profiles`](https://huggingface.co/datasets/placeholder/profiles)

## 交接总表

| 环境 | 0-latency small | 0-latency VLA | Latency profile | Profiled small | Profiled VLA |
|---|---|---|---|---|---|
| Flappy Bird | 正式 100 局 mean 444.600；Published | 三族保留；历史 O 438.7、π 252.3、G 221.7；全部重评 | 三族 canonical 文件完整；mean/p95 ms：O 75.9/81.3、G 137.7/147.2、π 200.6/213.9 | 历史已发布；训练/评测容量不一致，成绩失效 | O 缺完整正式评测/配置，保持待评；G、π 已知差；当前均不发布 |
| Demon Attack | 正式 100 局 raw game-score mean 8727.150（14400 steps）；Published | 历史 O 6557.8、π 3848.8、G 3612.8；全部重评 | 三族完整；O 71.6/75.2、G 129.2/136.2、π 192.8/203.2 | 历史已发布；训练/评测容量不一致，成绩失效 | 旧数据 O 为 255.25/200 episodes，明确差；新数据未发布；G/π 无新模型 |
| Deadly Corridor | 正式 100 局 mean 2116.794；Published | 历史 O 686.5、π 252.6、G 241.9；全部重评 | 三族完整；O 73.7/78.0、G 131.7/139.5、π 193.9/206.2 | 历史已发布；训练/评测容量不一致，成绩失效 | 现有三族均明确差：G 约 380.4、O 约 305.8、π约 198.6；不发布 |
| Atlantis | 正式 100 局 mean 29271.000；Published | 三族公开候选齐全；历史 π 19700/20、G 18456/80、O 17727.5/80；全部重评 | 无 | 无 | 无 |
| Asterix | 正式 100 局 mean 4099.000；Published | 三族公开候选齐全；历史 G 3152.5/20、O 1816.9/80、π 1801.9/80；全部重评 | 无 | 无 | 无 |

历史 VLA 数值仅作对照。中央正式结果只接受下面统一协议产生、且能指回 exact
checkpoint 的新评测 JSON。不同模型族并列报告，不按模型族之间的 reward 高低删除模型。

## 正式迁移集合

### 0-latency small：5 项

| 环境 | 源仓库与 revision | 源路径 | 中央路径 | 当前状态 |
|---|---|---|---|---|
| Flappy Bird | `placeholder/paper-experiment-models@8d576e64dd9f7370e06abfebff37b3d50dfdf8d1` | `jitter_destructiveness/real_profile/latency0/teachers/flappy/flappy_teacher_fixed_l0_obs30_seed0/`，选择 `best_000047376_24281088_reward_2078.738.pth` | `zero-latency/flappy/small-policy/sample-factory-v1/` | Published；100 局 mean 444.600；central readback `98b693409c2f86c10b7a11e511fa60ff9e93e7ec` |
| Demon Attack | 同上 | `jitter_destructiveness/real_profile/latency0/teachers/demon_attack/demon_attack_teacher_fixed_l0_obs15_seed0/`，选择 `best_000048704_24936448_reward_27118.050.pth` | `zero-latency/demon-attack/small-policy/sample-factory-v1/` | Published；14400-step 100 局 raw game-score mean 8727.150；central eval/README readback `7a6d7af88f921c6244aafe03078f1d30d94cc9b7`；provenance `dd81908094a43d9c68e9c1349ad0a7cb697d5cf5` |
| Deadly Corridor | 同上 | `jitter_destructiveness/real_profile/latency0/teachers/deadly_corridor/deadly_corridor_teacher_fixed_l0_obs8p75_seed0/`，选择 `best_000040896_20938752_reward_2260.797.pth` | `zero-latency/deadly-corridor/small-policy/sample-factory-v1/` | Published；100 局 mean 2116.794；central readback `98b693409c2f86c10b7a11e511fa60ff9e93e7ec` |
| Atlantis | `placeholder/benchmark-models@56b2c095d76ea5b10dd304e6e03c2ca81801630b` | `zero-latency/atlantis/small-policy/sample-factory-v1/` | 原路径补齐正式 bundle | Published；100 局 mean 29271.000；central readback `98b693409c2f86c10b7a11e511fa60ff9e93e7ec` |
| Asterix | 同上 | `zero-latency/asterix/small-policy/sample-factory-v1/` | 原路径补齐正式 bundle | Published；100 局 mean 4099.000；central readback `98b693409c2f86c10b7a11e511fa60ff9e93e7ec` |

Small bundle 只包含所选 checkpoint、原始 `config.json`、`eval_l0.json` 或
`eval_profile.json`、`provenance.json` 和 `README.md`。不复制 W&B、训练日志或其他
retained checkpoints。

### 0-latency VLA：15 项

| 环境 | O 源 | G 源 | π 源 | 中央目标 |
|---|---|---|---|---|
| Flappy Bird | `placeholder/openvla_flappy_fix_latency_0@06f64d1a03905ad0f04a447c57afb970c29a559e` | `placeholder/gr00t_bridge_flappy_single_latency_clean_data_exp2@4e9f72bc7f74384a1155caecc62f1dab380dcd24` | `placeholder/pi05_fix_latency@ae2c1f25b927b27debb8db2ca5903ef4f3910321` 下 `pi05_flappy_fix_latency_0_200ep/checkpoints/steps_5000_state` | `zero-latency/flappy/vla/starvla-{openvla,gr00t,pi05}-v1/` |
| Demon Attack | `placeholder/openvla_demon_attack_200ep@04fece441aa5b6f5a6a015264a6aa5df3dc4c29a` 下 `demon_attack_fix_latency_0_200ep/` | `placeholder/gr00t_bridge_demon_attack_single_latency_clean_data_exp3_action_1e-5_backbone_1e-6@fb0b8c4ef4d76b8c7ad97edf4990341f2b93cf70` | `placeholder/pi05_demon_attack_fix_latency_0@c797dd4d141e48fee2af1ae4fa1801d1aa307ed6` | `zero-latency/demon-attack/vla/starvla-{openvla,gr00t,pi05}-v1/` |
| Deadly Corridor | `placeholder/openvla_bridge_deadly_corridor_single_latency_clean_data_bce_exp2@31983a233371ba1e755e28b38de3b0b72ded39f7` | `placeholder/gr00t_bridge_deadly_corridor_single_latency_clean_data_bce_exp2@e93458b10c98dd1f4c712071a74f46dda2fa905d` | `placeholder/pi05_bridge_deadly_corridor_single_latency_clean_data_exp2@4d80b52ab5160e20327c843d7737c25ecfcb5ec7` | `zero-latency/deadly-corridor/vla/starvla-{openvla,gr00t,pi05}-v1/` |
| Atlantis | `placeholder/openvla_bridge_atlantis_single_latency_clean_data_exp1_1gpu_r1@08662fc9b1c181e3a96895e66de76a66525bb4e5` | `placeholder/gr00t_bridge_atlantis_single_latency_clean_data_exp1_1gpu_r1@d3fe75a8671b1819598339a23a9277b5e6211b78` | `placeholder/pi05_bridge_atlantis_single_latency_clean_data_exp1_1gpu_r1@7a5026531a0128c9d071e7a7e9a26f7ce88b94e3` | `zero-latency/atlantis/vla/starvla-{openvla,gr00t,pi05}-v1/` |
| Asterix | `placeholder/openvla_bridge_asterix_factorized6_single_latency_clean_data_exp1_1gpu_r1@cacfa287a2c36d96ffe8601874d9ae3ba0a7ba59` | `placeholder/gr00t_bridge_asterix_factorized6_single_latency_clean_data_exp1_1gpu_r1@c9185c81f689413079e59556d68747172568dc7f` | `placeholder/pi05_bridge_asterix_factorized6_single_latency_clean_data_exp1_1gpu_r1@5aa5ec259c0f16e7b6ccaff3b12c478cf7cbd2bb` | `zero-latency/asterix/vla/starvla-{openvla,gr00t,pi05}-v1/` |

每个 VLA bundle 使用现有正式格式：`checkpoints/model.safetensors`、`config.yaml`、
`config.full.yaml`、`dataset_statistics.json`、`task_contract.json`、`eval_l0.json`、
`provenance.json` 和 `README.md`。不得换成相似训练配置或另一 checkpoint。optimizer、random
state、W&B 和训练日志不迁移。

Flappy π 使用上表已确认公开的 5000-step state。3000/5000 step 不拆成两个模型角色，
也不因 step 差异重新 profiling。

### RTX 3090 latency profile：9 项

全部来自 `placeholder/profiles@48d13d3dfa2b77c2d10fbbbcc5210d8af9fc8749`
并保留原目录：

| 环境 | O mean/p95 ms | G mean/p95 ms | π mean/p95 ms |
|---|---:|---:|---:|
| Flappy Bird | 75.87 / 81.32 | 137.73 / 147.21 | 200.61 / 213.87 |
| Demon Attack | 71.59 / 75.24 | 129.16 / 136.17 | 192.75 / 203.18 |
| Deadly Corridor | 73.69 / 78.02 | 131.66 / 139.50 | 193.88 / 206.22 |

路径为
`profiles/{openvla|gr00t|pi05}/1x-rtx3090/<task>/instance_a5037b165aa0cedc/`。
九项都已有完整 canonical profile 文件。模型发布后只修改 profile 的 model provenance，
使其指向对应中央模型路径和发布 revision；不重跑 profiling，不改采样数据。

### Profiled small：9 项

本节九项旧 checkpoint 和旧评测仅保留作审计证据，已因训练/评测 worker-capacity
契约不一致而失效，不能继续作为正式结果或发布候选。除下表 Flappy O/G 的
2026-09-22 replacement 外，源仓库均为
`placeholder/paper-experiment-models@8d576e64dd9f7370e06abfebff37b3d50dfdf8d1`。

| 环境 | O checkpoint | G checkpoint | π checkpoint |
|---|---|---|---|
| Flappy Bird | `results/checkpoints/flappy/profiled_small_qwen_10fps_25m/flappy_profile_qwenoft_rtx3090_temporal_25m_obs10_seed42/checkpoint_p0/best_000046048_23592960_reward_1867.668.pth` | `results/checkpoints/flappy/profiled_small_qwen_10fps_25m/flappy_profile_qwengr00t_rtx3090_temporal_25m_obs10_seed42/checkpoint_p0/best_000048624_24903680_reward_1812.752.pth` | `pi05_rtx3090_profile_25m...seed11/...reward_111.747.pth`（本轮未替换） |
| Demon Attack | `openvla_rtx3090_profile_25m...seed10/...reward_13578.800.pth` | `gr00t_rtx3090_profile_25m...seed13/...reward_7958.400.pth` | `pi05_rtx3090_profile_25m...seed14/...reward_4591.800.pth` |
| Deadly Corridor | `openvla_rtx3090_profile_10m...seed13/...reward_2059.584.pth` | `gr00t_rtx3090_profile_10m...seed13/...reward_1877.202.pth` | `pi05_rtx3090_profile_10m...seed10/...reward_1524.672.pth` |

表中 `...` 是同一行模型的
`memory/small_model/<task>/<model>_rtx3090_profile_{25m|10m}_fs4_1min_eval/train/`
前缀。`provenance.json` 保存完整源路径，不能用省略写法。中央路径为
`latency-aware/<task>/small-policy/sample-factory-{openvla|gr00t|pi05}-rtx3090-v1/`。
九项历史 reward 只用于审计，不设置淘汰门槛，也不据此选择 corrected checkpoint。

旧 Flappy O/G 于 2026-09-22 使用 temporal `HiddenRegimeSampler`、10/10 FPS、
seed 42、25M 训练产物强制替换；100 局 best-checkpoint eval（horizon 18000）mean/std/
mean length 分别为 O 1177.629/803.195/9519.83、G 6.687/3.507/71.21。由于训练时未执行
`capacity=1` observation admission，这两个结果和旧 Flappy π、Demon Attack、Deadly
Corridor 三族结果一起失效。原子替换回读
revision 为 `d9034095ab140c1bf1369c3dc236499518e343d2`，最终 provenance revision 为
`fa25969ae7db7bc7f42fb6d19eccbbcfb18ab6c7`。Flappy π 本轮未上传，中央路径仍保留
此前 IID bundle（mean 2.000）。其余正式 100 局 profile replay 结果中，Demon Attack
的 14400-step raw game-score mean 为
O 5632.750、G 3168.300、π 1097.850；Deadly Corridor mean 为 O 686.030、
G 566.550、π 347.668。
Deadly Corridor 的历史回读 revision 为
`98b693409c2f86c10b7a11e511fa60ff9e93e7ec`；Demon Attack 的 soft-reset 修正版
eval/README 回读 revision 为 `7a6d7af88f921c6244aafe03078f1d30d94cc9b7`，provenance
revision 为 `dd81908094a43d9c68e9c1349ad0a7cb697d5cf5`。旧的 Demon Attack
3600-step JSON 以及 14400-step clipped-reward 中间结果均已被新正式结果取代，不再作为
可比较成绩。

Corrected 协议为全新 `capacity1_seed42` 实验：Flappy 使用 temporal profile、10/10 FPS、
25M steps 和 18000 horizon；Demon Attack 使用 IID profile、60/15 FPS、25M steps 和
14400 horizon；Deadly Corridor 使用 IID profile、60/15 FPS、10M steps 和 3600 horizon。
三族都固定训练 seed 42，评测 deterministic 100 局、`capacity=1`，并绑定各自训练内部
产生的 `best_*.pth`。corrected checkpoint、日志与评测只写挂载盘；明确审阅批准前不修改
任何 HF bundle。

## 不进入正式模型路径的资产

- 当前所有 profiled VLA 均不发布。Deadly Corridor 三族、Flappy G/π、Demon Attack
  旧数据 O 已有明确差结果，只在本账本保留 provenance。
- Flappy profiled O 只有在 exact config、dataset statistics 和 task contract 全部找回后
  才做 matching-profile 评测；不以相似配置代替。
- Demon Attack 新数据发布后另立交接项；现在不以旧数据模型冒充新数据结果。
- [`profile-rollouts-v2`](https://huggingface.co/datasets/placeholder/profile-rollouts-v2)
  已知导致差结果，不复制到中央 dataset 仓库。
- 不建立 `experimental/` 或 `legacy/` 模型目录。

## 历史迁移评测合同

下列 profile-small 评测因训练容量契约不一致而失效；corrected 协议见上文。

- 每个 checkpoint 100 episodes，逻辑 case IDs/目标环境 seeds 为 `42..141`，确定性评测。
- 当前 evaluator 对 Sample Factory policy 有既有 `+104729` reset-seed 映射。因此 small
  policy 的实际环境 reset seeds 为 `104771..104870`；评测 JSON 必须同时记录逻辑 case
  IDs 和实际 `episode_seed`，不得把两者都写成 `42..141`。VLA 的实际 seeds 为
  `42..141`。
- horizon：Flappy 3600、Demon Attack 14400、Deadly Corridor 30000、Atlantis/Asterix 5000。
- 5 个 zero-small 用 zero latency；9 个 profiled small 用各自 O/G/π 的
  `instance_a5037b165aa0cedc` IID profile replay；15 个 zero-VLA 全部用 zero latency。
- 15 个 zero-VLA 使用训练数据中的完整 latency-0 prompt，`prompt_mode: raw`，不得使用
  `latency_neutral` 截断 timing suffix。prompt 中的 env/observation FPS 分别为 Flappy
  `30/30`、Demon Attack `60/15`、Deadly Corridor `35/8.75`、Atlantis/Asterix
  `60/15`；正式启动前逐字校验生成 prompt 与源 parquet 或 task manifest 一致。
- Demon Attack 显式设置 `terminal_on_life_loss: true` 以检测掉命；掉命后 CuLE backend
  执行 soft reset 并继续同一 formal episode，只在 `lives=0` true game over 或 14400-step
  horizon 结束。`eval_raw_reward: true` 只关闭 reward clipping。
- Demon Attack 的正式 return 字段为原始 ALE `game_score`。GPU backend 配置必须显式
  `clip_reward: false`，且验收时逐局要求 `episode_return_env == game_score`；clipped-reward
  汇总不得作为历史 raw-score 的可比成绩。
- 并行度：small 的 Flappy/Demon Attack 为 100，Atlantis/Asterix 为 32，Deadly Corridor
  为 16；VLA 的 Flappy/Demon Attack/Atlantis/Asterix 为 10，Deadly Corridor 为 4。
  每个并行槽绑定自己的 episode ID、环境 seed 和 action-noise seed。结果按 episode ID
  排序，并行不会改变虚拟时间或 profile replay 语义。
- 每个 VLA evaluator 进程固定 Torch intra-op 线程为 8、inter-op 线程为 4，并将
  `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`NUMEXPR_NUM_THREADS`
  均设为 8；GPU inference/backend 不再由每进程数百个默认 CPU 线程互相争抢。
- 每项 `eval_l0.json`/`eval_profile.json` 报告 episode 数、逻辑 IDs、实际 seeds、
  mean/std、Student-t 95% CI、episode length、horizon、并行度以及逐 episode 原始记录。

## 发布验收

只有同时满足以下条件才从 `Pending` 改成 `Published`：

1. 源 checkpoint/config/revision 可按记录重新下载。
2. exact checkpoint 能加载并完成一次真实 inference smoke。
3. 正式评测恰好包含 100 个指定 case IDs，且每条记录指向 exact checkpoint。
4. 九个 profile 的 canonical 文件仍完整，sampler 可读，model path/revision 可解析。
5. 发布后从中央 HF revision 回读 checkpoint、config、评测 JSON、provenance 和 README。
6. `docs/artifacts.md` 写入回读通过的中央 revision；随后才更新 `program/PROGRESS.md`。

下载、调度、GPU 分配和迁移过程不写入 `PROGRESS.md`；失败与运行时间线写到
`program/logs/`，原始评测和 bundle staging 写到 `outputs/`。
