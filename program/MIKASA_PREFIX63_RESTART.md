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

## 2026-09-25 部署入口

W&B 使用用户指定项目 `zihanwang-ai-northwestern-university/starvla_tasks`；
SFT 显式设置 entity，DAgger 继承 SFT 的 entity/project。原 `saberrr-zju/starvla_tasks`
返回403，已验证新项目能创建并同步run。L1/L2无已完成optimizer更新，保留数据并归档
失败启动目录后重启；L3/L4当前数据子进程完成后切换新控制器，不丢弃正在处理的数据。

主机 `34.131.223.74`，独立源码 `/home/ubuntu/codex/mikasa-prefix63-restart-20260925`。
四路物理卡分别为L1=0,1、L2=2,3、L3=4,5、L4=6,7；模拟/独立推理使用各自逻辑0/1。

```bash
VLA_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_openvla/bin/python \
UV=/home/ubuntu/.local/bin/uv \
bash scripts/tasks/mikasa_intercept_grab_fast/run_fixed_prefix63.sh \
  fixed-2 fixed2-prefix63-restart-20260925 2,3 29704 29705
```

服务器上 `outputs/tasks/mikasa_intercept_grab_fast/restart-20260925/launch.json` 保存四路命令，
各run目录的`phase`及`controller.log`提供实时状态。L1/L2改用新W&B项目后已确认完成optimizer更新；
L3/L4保留当前数据处理，完成后自动换用新控制器。训练与终评尚未完成。
远端CPU回归152项通过；原三路trainer失败的具体异常因日志丢失、W&B读取被403拒绝仍未确认。

## 五档恢复与时间估计（9月25日16:25）

已按用户最新指令恢复五档periodic。L1/L2在首轮准备阶段切换controller，复用已完成的
matched评测和采集结果，补齐其余档；尚未进入DAgger的任务直接采用新代码。
完成且已核验的SFT publication receipt复用，避免控制器重入时重复上传。

预计从16:25起，四路全部完成训练、终评及发布还需8–10小时，主要等待L4；
目标9月26日00:30–02:30，保守排期至03:00。该估计不是完成承诺。
依据：历史L2从DAgger启动到终评完成5h52m；首轮采集/打包L2约5.35m，
L1约6.24m、L3约8.19m、L4约17.46m。当前每500更新约13–16m，
五档resident评测约8–10m/边界。模型质量、CPU并发和上传影响实际耗时。
