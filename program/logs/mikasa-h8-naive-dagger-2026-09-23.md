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
  PI 至第二台 GPU 6/7，GR00T 至第一台 GPU 2/3；simulator 分别选 GPU 6、6、2，
  模拟器进程沿用 GPU lock。GR00T 启动时 Fixed-2 OFT 已完成并释放 2/3；
  调度器已允许 simulator 与任一新 trainer pair 同卡。现有 Codex heartbeat
  `mikasa-h8-dagger-15` 每 15 分钟检查六路 conditioned 状态和三路训练。

三路正式终点、独立评测与验收均尚待核验，不以控制器启动代替训练完成。

## 首轮元数据修复（2026-09-23）

OFT 与 PI 的首轮 student rollout、16k-head 数据打包及 step-0 评测完成后，
两 rank trainer 均在第一个 optimizer step 前退出。StarVLA 合并 bootstrap 与增量
LeRobot 数据时发现 action modality 不一致：原 SFT `meta/modality.json` 的 target
标记为 `dtype=float32, absolute=false`，增量数据缺少这两个字段。
这不是 GPU OOM，也没有产生可恢复的 DAgger step checkpoint。

增量打包现从 SFT normalization source 复制完整 modality 合同；两份已打包的
round-01 数据按各自主机的 SFT modality 修正，旧 metadata 和失败日志保留。
两台机器分别用 StarVLA 的真实混合数据加载器验证 metadata merge 成功，随后从
原 SFT step-5000 模型权重重新启动同一正式 run，旧不完整 `training/dagger`
目录单独归档。OFT 与 PI 已开始 optimizer 更新，W&B run 正常。
队列 `jobs.json` 的 PID 已更新为恢复后的 controller；GR00T 仍等待资源。
随后第一台 GPU 2/3 空出，调度器启动 GR00T 正式 DAgger；三路现均在训练。

三路 naive 正式完成并通过全部评测后，由现有 15 分钟监控依次调用
`scripts/tasks/mikasa_intercept_grab_fast/publish_h8_checkpoints.py`
上传 DAgger 选优推理 bundle 至 `latency-sensitive-bench/benchmark-models`。
上传结果写入各 run 的 `publication_hf_dagger.json` 并回读 Hub；原 SFT H8
bundle 已在模型仓库中。

## 三路正式完成与发布

三路均完成十轮共 5,000 DAgger optimizer steps、11 次各 100 局开发评测、
10 份各不少于 16,000 个有效 suffix head 的增量数据、完整双 rank step-5,000
trainer state，以及 Zero、Profile、Fixed-2 各 200 局的 baseline/best/latest 独立
评测。heldout seeds 均为 `4242424542–4242424741`，执行审计和物理 reset seed
核对通过。开发集选中的 best 分别为 OFT step 5000、PI v3 step 4000、GR00T
step 3000。独立 200 局 best 在 Zero 条件下分别为 `196/200`、`199/200`、
`192/200`；完整九组成功率与 return 见 [PROGRESS](../PROGRESS.md)。

| 模型 | 首请求 suffix MAE（n） | 后续请求 suffix MAE（n） | DAgger 发布 revision |
|---|---:|---:|---|
| OFT | 0.04806（645） | 0.11757（21,055） | `af125bd372c22c5b00e326d9af1af64ea3751b56` |
| PI v3 | 0.05647（648） | 0.12140（21,368） | `5b533937a72e9fa923ecf28b1c1be5a16bad488d` |
| GR00T | 0.05423（650） | 0.12354（20,857） | `3fb1d4a7c2645715f0554f315d85032868bf15bb` |

suffix MAE 是十轮 `collection.json` 中首请求/后续请求 mean 按请求数加权。
三个 DAgger 选优推理 bundle 均发布在原 zero-latency H8 模型目录下的新
`h8-naive-dagger-20260923` run ID；PI v3 和 GR00T 目录还保存 step-5,000
`checkpoints/latest_model.pt`。Hugging Face API 对 `main` 的三目录逐文件
比对本地 stage 路径及大小，均通过。原 SFT H8 bundle 未重复上传。
