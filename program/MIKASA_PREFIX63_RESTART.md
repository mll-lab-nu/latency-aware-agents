# MIKASA prefix63：2026-09-25 重跑协议

本协议记录 2026-09-25 批准的 **63D v3、H8×7** 五档 periodic eval 与阶段结束上传方案。Fixed-1/2/3/4 均从零重新
采集 teacher 数据、训练 SFT，再接 DAgger；不依赖已丢失的临时盘产物。

它接续 [Fixed-2 输入修复](MIKASA_FIXED2_PREFIX_REPAIR.md)，与
[7D baseline / 71D v2 历史合同](MIKASA_ACTION_CHUNK_H8.md) 分别对应自己的模型和输入尺度。
下文部署与进度描述是 2026-09-25 的记录；最新核验状态见 [PROGRESS](PROGRESS.md)，
接续背景见 [Fixed-2 修复与部署记录](logs/mikasa-fixed2-prefix63-2026-09-24.md)及
[故障记录](logs/mikasa-prefix63-failure-2026-09-25.md)。

## 配方

- QwenOFT、H8×7、63D 当前时刻对齐 prefix；每路固定 K=N，动作输入 min/max 为 -1/+1。
- 20/20Hz、单 worker、busy-drop、按观察时刻选 head、输出硬覆盖 prefix、耗尽 hold-last。
- 每路 train/validation 至少64288/6884个有效suffix标签，种子划分不交叉；同一固定PPO teacher。
- Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`；seed0。
- SFT5000 + DAgger10×500；每卡batch64、global128、accumulation1、gradient checkpointing关闭。
- 每500次仅在服务器保存模型和完整恢复状态。OOM不自动缩batch。
- DAgger每轮至少16000有效suffix标签，冻结bootstrap归一化；首轮50/50，之后40/40/20。

## 评测

DAgger step0、500…5000 **评L0–L4五档**，每档100局，
seeds `4242424442–4242424541`。仍只按matched LN的成功数、mean return、较晚step选优，包含step0。
生成五条开发曲线；完整跑满5k，不按开发或测试结果追加训练。

最终同一best统一评测L0–L4，各200局，seeds `4242424242–4242424441`，
60个实际transitions、10个并行环境、inference batch8、deterministic inference。
K保持训练条件不变。五档报告保留成功率、return/总体SD、权重和执行审计。

## 发布与存储

训练期间不上传checkpoint。SFT阶段结束后直接从H200发布step5000模型、合同和恢复状态；
DAgger结束并完成终评后发布最终恢复状态、选中的best模型及报告。上传后按返回revision
逐文件核对路径和大小，失败不记录成功。失败时另外发布控制器和**实际trainer**日志。

目标仓库 `latency-sensitive-bench/benchmark-models`，每路独立目录：
`latency-aware/mikasa-intercept-grab-fast/vla/starvla-qwenoft-h8/fixedN-prefix63-restart-20260925`。
训练数据、checkpoint、原始日志和图都在服务器；不向用户笔记本下载实验产物。

## 运行

从仓库根目录执行，为每路实验指定 run ID、设备与端口；模型解释器和 W&B
entity 由调用方提供。数据和训练输出按 run ID 保存在 `outputs/starvla/` 下。

```bash
export VLA_PY=/path/to/model/python UV=/path/to/uv WANDB_ENTITY=your-entity
bash scripts/tasks/mikasa_intercept_grab_fast/run_fixed_prefix63.sh \
  fixed-2 RUN_ID 0,1 29704 29705
```

共同的采集、打包、数据审计和 SFT 阶段复用 `run_conditioned_h8.sh`。
DAgger 按本页的固定预算和五档评测协议接续；成功发布记录直接复用确切版本。
当前结果及完成状态见 [PROGRESS](PROGRESS.md)，过程证据见本页开头的记录链接。
