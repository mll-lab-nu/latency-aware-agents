# Hugging Face 仓库合并交接

更新日期：2026-10-08。公开仓库合并完成，本轮删除 41 个 model 源和 39 个 dataset 源。公开仓库只剩三个中央仓库；认证后另有三个私有仓库，均保持原状。Figure 2 六份游戏权重已发布，15 条曲线及 95 行 CSV 数值未改动。

## 用户目标与执行要求

- 正式 release 和实验 archive 统一到组织 `latency-sensitive-bench` 的三个仓库：model `benchmark-models`、dataset `benchmark-datasets`、dataset `profiles`。继续按论文实验归属合并其他组织仓库。
- 先读论文实验描述、代码、配置和来源记录，再确定正式用途或 archive。原始轨迹、逐局结果、日志、配置和来源缺口必须保留。
- 优先让 HF 复用已有对象。同仓改目录使用 Git tree/commit 元数据；LFS 使用服务端复制。不要批量下载重传已有内容，也不要默认找云端机器中转。
- 当前 HF SDK 的普通 Git 文件复制会下载再上传。这是 SDK 行为，不是服务端能力的完整结论。跨仓纯引用试验曾因目标缺少 Git 对象被拒绝；先检查目标已有对象，需要传输时只传缺失对象，使用压缩 Git 传输，避免批量 Base64 请求。
- 仓库整理使用远端 Git/LFS 身份、大小和固定引用核验。不要把完整 checkpoint 下载、模型加载、重训或重评加成整理前置条件；没有验证的运行能力须明确记录。
- 删除源仓库前，核对完整文件清单、固定目标和全部当前消费者。共享源只迁移部分内容时不能整仓删除。个人上游仓库保留。

## 已完成与主要证据

- 用户于 2026-10-08 决定删除 benchmark-datasets 根目录 `archive/` 全部内容，包含 task-transfer，不迁移或另存旧记录。仅删除根目录；嵌套 `experiments/*/archive/` 保留。后续迁移不得恢复该根目录。
- 同内容数据改用已有 canonical 位置；无同内容替代的旧训练入口退役。执行清单、固定引用和存储回执见 [根目录 archive 退役报告](../outputs/hf_cleanup/root-archive-retirement-20261008/REPORT.md)。

### Figure 2：评测与六份游戏权重均已发布

- [固定实验入口](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/d1014aaaa1dd7f40ae21c60070dc0659fa265422/experiments/latency-degradation)：15 条曲线、CSV 95 行、主图 72 个不同数据点；数值和原图未改动。六份游戏权重均有中央固定引用。
- [评测 revision efeac490](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/efeac490144ca9c94660e2410163415e0a8809a9/evaluations/mikasa-intercept-grab-fast/latency-degradation)：477 个证据文件身份核对通过，MIKASA 30 条件的 6000 局完整回读通过，七份游戏 JSON 的 55 个条件各 20 局。
- [模型来源元数据](https://huggingface.co/MLL-Lab/LAGEN-models/tree/3f10fc76b7cbaf0f88d600b57427ebc19c50ad0f/reports/latency-degradation-20261007)：八份原配置/统计已回读；六份游戏权重已发布到中央模型仓库，SHA256、大小和 Xet hash 均与源文件相同。六份 MIKASA 模型复用既有固定 revisions。
- Flappy π 图用 step 5000，与 model `pi05_fix_latency` 的 Flappy 权重 SHA256 相同。Demon GR00T 图用低学习率 step 5000。两条游戏 OpenVLA 曲线的 checkpoint 绑定未确认；Flappy GR00T 图用 step 3000，但源 main 和 34 个历史提交只有 best 300/600/1800/2400，不能替换。
- model `pi05_fix_latency` 和 Demon GR00T 低学习率源已在全部资产覆盖及消费者核验后删除。个人上游保留。部分游戏训练数据精确 revision 未恢复，保留缺失状态。
- [本批报告](../outputs/hf_cleanup/latency-degradation-20261007/REPORT.md)记录原阻塞和恢复核验。用户于 2026-10-08 明确恢复这六份权重工作，指定服务器中转并保持原文件内容。六份均经标准 Xet 上传完成，仍归入 benchmark-models。后续源仓库清理见下方完成报告。

此前 HF `lfs-files/duplicate` 返回 `target already has sha256 stored without xetHash`。
本次按用户明确要求，下载固定源文件并经 `HfApi.upload_file` / `hf_xet` 原样上传，六份均成功。
[恢复核验](../outputs/hf_cleanup/latency-degradation-20261007/weight_reupload_identity_verification.json)记录六个固定下载入口的原 SHA256、大小和 Xet hash。
不把普通 LFS 上传申请当作只读探测；下载重传仍不是其他批次的默认方案。三个中央仓库历史不重写。

- [中央仓库存储精简](../outputs/hf_cleanup/storage-prune-20261007/REPORT.md)：HAIC 和
  Extreme Parkour 已从发布目录移除；保留选优权重，清理 optimizer 和 trainer state。
  当前引用、原始来源记录和实际释放的存储分别核验。后续迁移不得恢复这些废弃对象。
  MIKASA phase 发布仅上传推理权重和来源信息；训练恢复状态留在训练机器。

Task transfer 已删除 **2 个 model 源、6 个 dataset 源**，删除后均返回 404。
252 个文件身份匹配，128 个固定引用可解析；46 个 Parquet 样本读取、5 个图像解码通过；31 项相关测试通过。

| 分类 | Run | Checkpoint step |
|---|---|---:|
| 正式 | `demon_attack_fix_latency_0_200ep` | 7000 |
| 正式 | `openvla_bridge_demon_attack_latency_mixed_024_exp1` | 5000 |

- [Task transfer 固定入口](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/e983a3a72e042c6fab5575b847cdf6f4432cddaa/experiments/task-transfer)。正式模型在 `zero-latency/demon-attack/vla/`、`latency-aware/demon-attack/vla/`；根目录 task-transfer 数据归档按用户决定退役。
- [本轮报告](../outputs/hf_cleanup/task-transfer-20261007/REPORT.md)。同目录的 `copy_manifest.json`、`post_retirement_verification.json`、`completion.json`、`consumer_changes.json` 和 `consumer_task.diff` 是接手证据。
- [其他已完成批次](../outputs/hf_cleanup/)：memory、latency-transfer、latency-in-prompt、fine-tuning-scope、factor-unification 等。先查已有映射，避免重复收录。
- [Sim2Real 报告](../outputs/hf_cleanup/sim2real-calibration-20261007/REPORT.md)：30-case 正式实验与 ICLR archive 已归入三主仓。该批次清理仓内目录，没有删除独立仓库。
- [HF 远端目录操作回执](../outputs/hf_cleanup/sim2real-calibration-20261007/metadata_only_profiles_copy.json)：13,202 个路径引用约 6.82 GB 内容，只发送约 710 KB 元数据。跨仓限制见同目录 `metadata_only_crossrepo_response.json`。

两项既有 converter 断言失败已记录在本轮报告，不要为通过无关断言修改转换逻辑。
完整 framework 加载未重复验证。Mixed Demon 原始训练配置、统计与精确数据版本缺失，评测与权重的 digest 绑定仍未确认。
论文 mixed Demon L0 为 5812，公开评测为 5832.25；论文 raw-frame 标签与 corrected 配置也有差异：label 4 = 266.67 ms，在 60 env FPS / 15 obs FPS 下等于 16 raw frames。保留差异，不修改论文数值，不补造来源。

## 公开仓库合并结果与私有仓库核对

- [完整报告](../outputs/hf_cleanup/remaining-repos-20261008/REPORT.md)及[中央来源索引](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/a6e93d4a9a1277c5cd5a7145e0295e3e190374e1/reports/source-repository-retirement-20261008)。本轮先删除 36 个明确退役模型，再覆盖并删除五个共享模型源、39 个数据源。每个源删除后均返回 404。
- 39 个数据源的 9,115 个文件全部有保留去向。16,140 个原始 Git/LFS 路径引用身份一致；61 个小型 G1 Git 二进制对象转换为 LFS 后内容一致。70 份已有模型表示核验通过，包括 65 个 Standard-Pipeline 选优推理 bundle。33 个未选 checkpoint、训练状态或 W&B 二进制按既有保留政策退役。
- Standard-Pipeline 当前文件、两个已引用旧版本及其他 215 个文档旧版本证据保留。训练/评测进入 benchmark-datasets，硬件 profiling 进入 profiles。更新 391 个固定文档链接，并保留原始来源映射。当前读取和上传入口已核对；数据目录、latency 筛选和原配置名保留。
- `pi05_fix_latency` 的 Flappy step 5000 与 Demon GR00T 低学习率 step 5000 均在中央模型目录；八份可取得的原配置/统计已进入对应目录。源 model/dataset `humanoidbench`、`mikasa-robo`、Standard-Pipeline 和 G1 场景资源均完成合并。
- 认证后的组织清单为三个 model、三个 dataset。其中公开的只有 benchmark-models、benchmark-datasets、profiles。

三个私有仓库未迁移、未公开、未删除：

| 私有仓库 | 核对结论 | 当前处理 |
|---|---|---|
| model `latency-checkpoints` | 当前论文量化绑定未确认；两个 Demon Attack 工具仍引用 `checkpoints_old/demon_attack_sf_same_latency_l0_seed0` | 保留；不是孤儿 |
| model `starvla-rl-games-results` | 历史 joint54 LoRA 结果；未确认当前 arxiv 正文使用，旧结果尚无精确权重绑定 | 保留历史资产，不能据此判为孤儿 |
| dataset `results` | 旧延迟/跨 GPU 记录；当前图表使用中央 profiles。论文中的本地 `results/` 路径不证明这个 HF repo 是来源 | 保留，来源归属未完成 |

[私有仓库核对报告](../outputs/hf_cleanup/remaining-repos-20261008/PRIVATE_REPOSITORIES.md)只保存在本地，没有发布到公开中央仓库。
个人上游仓库保持原状。完整模型加载、重训及重评没有作为本次整理前置条件。

## 工作区与注意事项

- 工作树及 StarVLA 子仓有大量其他修改。保留现状，按本批次记录前后差异，不执行全仓 reset、checkout 或覆盖。
- 访问 HF 使用官方 `https://huggingface.co`。认证已有 `.env` 的 `HF_TOKEN`；不要输出 token，也不要发给镜像。model 和 dataset 同名时按 `repo_type + repo_id` 分别处理。
- 更新消费者时保留精确数据选择。本轮曾把 Deadly L2/L4/L6/L8 错改为 L0，已修复并增加回归用例，见 `exact_dataset_selection_repairs.json`。
- 发布根 README 时保留其他 agent 的并发更新。基于最新远端提交做局部修改，检查 Git 接收结果，不能仅凭 HTTP 200 判成功；不 force-push、不压缩中央仓库历史。
- 一次性工具在仓库外，结束后删除；保留清单和核验回执。本轮及 Sim2Real 的临时脚本、环境、下载目录已删除，不能依赖旧 `/tmp` 脚本。
- 遵守 AGENTS.md：不要新增防御性检查、隐藏 fallback 或过度抽象。HF 整理记录放 `outputs/hf_cleanup/`，不要把过程日志写入 `program/PROGRESS.md`。
- 旧文档中保留独立 release 仓库或禁止中央 archive 收录原始资产的说法，按本次用户的三主仓要求更新。
