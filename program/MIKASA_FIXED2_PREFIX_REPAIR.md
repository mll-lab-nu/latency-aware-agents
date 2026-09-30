# MIKASA Fixed-2：63D prefix 合同与重训

本文保留 2026-09-24 的修复方案与旧数据复用决定。后续重采集、阶段结束发布和当前
启动参数以 [prefix63 重跑协议](MIKASA_PREFIX63_RESTART.md) 为准。

适用版本为 **63D v3、H8×7、K=2**。本页的旧数据复用决定属于 2026-09-24 修复方案；
2026-09-25 接续协议明确重新采集 Fixed-1/2/3/4。运行时应选择与所用数据和模型对应的
完整协议，最新训练、评测、验收和发布状态见 [PROGRESS](PROGRESS.md)。

本轮只重训 QwenOFT Fixed-2，保留旧 71D checkpoint 和评测证据。旧模型的常量统计
会抹掉跨延迟请求中的部分旧计划信息，因此旧五档成功数 `1/1/177/0/0`（各 200 局）
不能单独归因于模型的延迟泛化能力。调度和成功率指标维持原定义。

## 输入与执行

- 新合同 `latency_bench_starvla_h8_v3`：7D proprio + 8×7 个观察时刻对齐的 prefix
  槽位，共 63D。K=2，仅两格承诺动作非零，其余六格填零。
- `encode_prefix_state` 同时供原始数据打包和在线请求使用；离线只读取 teacher chunk
  的 prefix mask 部分，suffix 标签不进入 state。删除旧计划原始 head 布局和 one-hot。
- proprio 使用原 bootstrap 七维统计；56 个动作输入统一固定 min/max 为 -1/+1，
  写入同一合同和 `stats_gr00t.json`。动作输出统计及 suffix loss mask 不变。
- QwenOFT 现有 continuous state projector 输入为 63；旧 71D 权重不用于新模型初始化。
- H8×7、20/20Hz、单 worker、忙时丢请求、输出端硬覆盖 K=2；到达时从
  `当前帧−观测帧` 选 head，不加 K。启动 noop、耗尽保持末动作，所有 eval latency 保持 K=2。

## 数据与预算

复用 `conditioned-h8-seed-reset-20260922/fixed-2/raw`：train 11,570 行 / 64,368 个
有效 suffix 标签，validation 1,250 行 / 6,943 个；图像、划分和标签不变。
新数据、run ID、运行目录独立保存，DAgger delta 统一冻结新 bootstrap normalization。

底座为 Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`。
SFT 5,000 optimizer steps 后，以新 SFT step-5000 权重和新优化器执行 DAgger 10×500。
roll-in 仅使用 L2，每轮至少 16,000 个有效 suffix 标签；首轮 bootstrap/current 为
50/50，之后 bootstrap/current/history 为 40/40/20。两阶段均为双卡、每卡 batch 64、
global 128、accumulation 1、seed 0、gradient checkpointing 关闭；OOM 不缩 batch。
每 500 steps 保存模型及完整 trainer state。

## 评测和选优

DAgger step 0、500…5000 共 11 个边界，每个冻结 checkpoint 依次评测 L0–L4，
各 100 局，seeds `4242424442–4242424541`。每格独立保存，恢复仅补缺失格。
全部五格完成才产生该 step 的选优记录，随后采集下一轮 L2 数据并恢复 trainer。
best 只按 L2 成功数、mean return、较晚 step 依次选择，包含 step 0 的 SFT baseline。

最终只对同一个 best 评测 L0–L4，各 200 局，seeds `4242424242–4242424441`。
每局 60 个实际 transitions、10 个并行环境、inference batch 8、deterministic inference。
报告保存 success、mean return、总体 SD、checkpoint、K 和执行审计；生成五条 periodic
曲线与旧 71D / 新 63D 对比的 PDF 和 300-DPI PNG。修复验收不要求成功率曲线单调。

## 执行与恢复

当前入口 `scripts/tasks/mikasa_intercept_grab_fast/run_fixed_prefix63.sh` 接收 condition、
唯一 run ID、物理 GPU 对、SFT/DAgger 端口；Fixed-2 对照可额外传入旧五格 JSON。
当前版本按重跑协议在 run ID 对应目录采集数据，不再接收旧数据目录参数。
要求环境变量 `VLA_PY`、`UV`。原修复实验在独立运行目录内执行，
第二台 H200 物理 GPU 0–1 训练；resident 边界两个 rank 清理未用 CUDA cache 后，
模拟器共用 GPU 0，模拟器退出才恢复训练。独立评测为 GPU 0 模拟、GPU 1 推理。

本地测试通过后同步源代码，直接执行正式流程，不运行额外 GPU 短流程。入口重跑会
复用完整数据和已完成 SFT；未完成 SFT 从完整 trainer state 恢复，DAgger 从完整双 rank
state 恢复，并复用已完成评测单元格。数据打包中断的半成品应保留到独立目录后重打包。

运行状态与路径见 [操作记录](logs/mikasa-fixed2-prefix63-2026-09-24.md)。训练完成、
终评完成、性能判断与发布分别记录；本轮不自动发布或恢复其他 Fixed 条件的训练。

## 扩展 Fixed-1、3、4（用户批准）

沿用同一 63D 合同，分别固定 K=1/3/4。每路从相同固定 Qwen 底座开始，
SFT 5k + DAgger 10×500；每卡 batch 64/global128/accumulation1/seed0/GC关闭，
保留每500次保存、无 OOM 缩 batch。旧对应条件的原始 teacher 轨迹已完整采集，
分别重打包，保留原划分和 suffix 标签；不复用 Fixed-2 的标签或训练权重。

每路 roll-in 使用自身训练 latency，首轮50/50、后续40/40/20，冻结自身bootstrap统计。
DAgger step0及每500 steps都评测L0–L4各100局，仍使用原开发seeds；选优分别按
matched L1/L3/L4 的成功数、mean return、较晚step；best五档各200局正式seeds不变。
产出每路五条periodic曲线、五格终评和JSON；Fixed-1/3/4没有旧71D正式列可比较，
因此报告自身终评，不构造旧模型对照。原 L2 控制脚本和运行目录保持其已部署版本。

当前入口示例：`bash scripts/tasks/mikasa_intercept_grab_fast/run_fixed_prefix63.sh fixed-3 RUN_ID 2,3 29936 29937`。
三路部署、卡分配与当前状态见 [扩展记录](logs/mikasa-prefix63-expansion-2026-09-24.md)。
