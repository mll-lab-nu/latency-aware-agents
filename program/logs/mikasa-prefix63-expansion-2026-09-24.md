# MIKASA prefix63：新增 Fixed-1、3、4

用户在看到新 Fixed-2 SFT 开发评测后批准启动其余三个条件，配方沿用 L2。
仅扩展训练条件、GPU 对及端口；原 L2 控制脚本继续运行。

## 固定配方

63D 对齐 prefix，输入动作 min/max 固定 -1/+1；K=训练 latency，跨五档评测不变。
Qwen3-VL-4B-Instruct 固定 revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`，
每路全新 SFT 5000 + DAgger 10×500，seed 0、每卡 batch64/global128/accumulation1、
gradient checkpointing false、不允许 OOM 缩 batch，每500次保存权重与完整状态。
DAgger roll-in 和选优使用各自 matched latency；step0及每500步L0–L4各100局，
seeds `4242424442–4242424541`；同一个 best 的五档终评各200局，
seeds `4242424242–4242424441`。沿用首轮50/50、之后40/40/20及每轮16000有效suffix预算。

## 数据与部署

旧 Fixed-1/3/4 各自的数据已按自身 K 和延迟采集完成，故复用自身原始图像/标签与划分，
只重打包为63D；不使用旧模型权重，不改名复用其他条件的数据。

| 条件 | train rows / valid heads | validation rows / valid heads | host | GPU 对 | controller PID |
|---|---:|---:|---|---|---:|
| Fixed-1 | 10079 / 64455 | 1107 / 7004 | 第一台 `34.131.223.74` | 0,1 | 1434130 |
| Fixed-3 | 13661 / 64298 | 1470 / 6910 | 第二台 `34.131.91.104` | 2,3 | 358223 |
| Fixed-4 | 16864 / 64305 | 1802 / 6894 | 第二台 `34.131.91.104` | 6,7 | 358303 |

两台独立目录均为 `/home/ubuntu/codex/mikasa-prefix63-expansion-20260924`；
run IDs 为 `fixed1-prefix63-20260924`、`fixed3-prefix63-20260924`、`fixed4-prefix63-20260924`。
Fixed-1 原始数据复制到第一台 `mikasa-h8/outputs/starvla/datasets/mikasa_intercept_grab_fast/prefix63-expansion-sources/fixed-1`；
Fixed-3/4 读取第二台原 `mikasa-fixed-grid-20260924` 中各自数据。
SFT/DAgger 端口分别为 29916/29917、29936/29937、29946/29947。

每路仅暴露自己的两张物理卡，逻辑0负责模拟、逻辑1独立推理；resident阶段使用两rank，
模拟器与逻辑0共卡，沿用已验证的缓存释放与锁。锁名对应模拟器物理 GPU。

## 验证与启动记录

本地128项通过；两台远端初跑各141项通过、4项报告测试暴露缺少paper runtime文件，
补齐本地 `paper_colors.py` 与论文样式后4项全部通过。也为L2目录补齐这两个报告依赖，
未修改其训练代码或重启进程。未运行GPU短流程。
三路正式控制器已启动，先重打包、全量审计和核对原七维/输出动作统计，再自动接SFT、DAgger、终评与报告。

入口改为 `scripts/tasks/mikasa_intercept_grab_fast/run_fixed_prefix63.sh`。
报告从每路 run.json 读取训练条件，只按该条件检查K；无旧列的模型直接输出五档终评，
Fixed-2仍可显式传旧列生成对照图。代码先本地修改及暂存，部署以冻结L2快照加本轮覆盖为准。

[部署证据](../../outputs/tasks/mikasa_intercept_grab_fast/prefix63-expansion-20260924-launch/launch.json)
及同目录 `source.json`、runtime包和CPU测试日志；未执行 commit。

启动后复查：Fixed-1 已完成重打包、全量审计和统计一致性检查并进入 SFT；
生成配置确认 state_dim63、gradient checkpointing false、microbatch64、accumulation1、
max_train_steps5000。Fixed-3/4 重打包仍在推进，控制器存活，未见报错。
每路独立结果同步进程在完成或失败后收集报告，运行中安静检查，不重启或停止任务。
