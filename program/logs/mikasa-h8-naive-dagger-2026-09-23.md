# MIKASA naive H8 DAgger 运行记录

可复用合同见 [MIKASA_ACTION_CHUNK_H8](../MIKASA_ACTION_CHUNK_H8.md#naive-h8-dagger-对照)。
本文件记录队列与训练的运行状态；正式指标核验后再写入 `program/PROGRESS.md`。

## 准备与运行

- QwenOFT、QwenPI v3、QwenGR00T 从已有 zero-latency H8 SFT step-5000 权重
  出发。第二台 H200 已从第一台同步三份权重、训练配置、评测配置与原 H8 数据。
- zero-latency collector 用当前 student rollout，private PPO branch 从当时主环境
  状态生成完整有效 H8 标签；增量数据保持 7D proprio 和原 SFT normalization。
  双卡训练的 OFT/PI/GR00T micro batch 与 accumulation 分别为 32/2、64/1、
  32/2，global batch 均为 128。
- 调度器读取 SSH JSON 时将 GPU 键转回整数；资源选择测试覆盖旧六路的训练卡
  声明及 6/7 上模拟器与新训练共用的条件。第一台相关 H8 和 resident 测试
  `27 passed`，第二台可用的 H8 测试 `25 passed`。
- 队列位于
  `outputs/starvla/runs/mikasa_intercept_grab_fast/h8-naive-dagger-20260923/queue/`，
  顺序为 OFT、PI、GR00T。2026-09-23 已直接提交正式 OFT 至第一台 GPU 6/7、
  PI 至第二台 GPU 6/7；两路 simulator 均选 GPU 6，模拟器进程沿用 GPU lock。
  GR00T 仍在队列中。现有 Codex heartbeat `mikasa-h8-dagger-15` 每 15 分钟
  检查六路 conditioned 状态和三路队列。

三路正式终点、独立评测与验收均尚待核验，不以控制器启动代替训练完成。
