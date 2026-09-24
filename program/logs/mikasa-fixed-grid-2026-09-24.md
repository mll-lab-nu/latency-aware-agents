# MIKASA Fixed-1/3/4 与 5×5 grid 执行记录

## 启动决定与代码

用户已批准同步并启动，随后明确要求跳过 GPU 短流程，直接使用正式预算。
本轮没有运行 smoke，也没有 commit 或发布 Hugging Face 模型。

将本地 Git index 中的运行源码同步到两台 H200 的独立 worktree：
`/home/ubuntu/codex/mikasa-fixed-grid-20260924`。保留原
`/home/ubuntu/codex/mikasa-h8` 的代码和其他任务进程，复用它的 StarVLA/MIKASA 环境、
模型缓存和 teacher。运行源码来自本地 `3f76b6d33334bf073bc5c51a6bf50b3bf61fbcb9`
加已暂存的 H8 改动；远端 worktree 基线为 `75b4097d`，实际源码与同步包一起保存。
[本地启动材料与 source manifest](../../outputs/tasks/mikasa_intercept_grab_fast/fixed_grid_launch_20260924/)。

## 正式流水线

| 条件 | 主机 | 双训练卡 | DAgger simulator | DDP port |
|---|---|---|---|---|
| Fixed-1 | `34.131.91.104` | 0,1 | 6（锁串行） | 29931 |
| Fixed-3 | `34.131.91.104` | 2,3 | 6（锁串行） | 29933 |
| Fixed-4 | `34.131.91.104` | 4,5 | 6（锁串行） | 29935 |
| Fixed-2 已有模型评测 | `34.131.223.74` | simulator 2 / inference 3 | — | — |

正式数据采集各使用本路训练卡中的第一张，128 个 vector slots，train/validation 至少
64,288 / 6,884 个有效 suffix heads。三路已产出第一批样本；记录中的实际 head 分别为
Fixed-1 的 1、Fixed-3 的 3–5、Fixed-4 的 4–7，与 matched-latency 协议一致。

每个后台 controller 顺序执行：正式采集与打包/审计 → SFT 5k 与 matched eval →
DAgger 10×500 与开发选优/独立终评 → 选中模型在 L0–L4 的五格正式评测。
SFT/DAgger 均为双 H200、每卡64、global128、accumulation1，gradient checkpointing关闭；
完整模型/trainer state每500步保存，OOM不自动缩batch。没有把采集或训练启动当成完成。

远端每路状态目录：
`outputs/tasks/mikasa_intercept_grab_fast/fixed-grid-launch-20260924/fixed-N/`，包含
`controller.pid`、`controller.log` 和 `phase`。当前进程状态应以实时核查为准。
完整命令保存在独立 worktree 根部 `run_formal_arm.sh`、`run_fixed_two_eval.sh`；
本地启动材料目录保留相同脚本。

数据：`outputs/starvla/datasets/mikasa_intercept_grab_fast/fixed-grid-20260924/`。
SFT：`outputs/starvla/runs/mikasa_intercept_grab_fast/fixed-grid-sft-20260924/`。
DAgger：`outputs/starvla/runs/mikasa_intercept_grab_fast/fixed-grid-dagger-20260924/`。
矩阵：`outputs/tasks/mikasa_intercept_grab_fast/fixed-grid-20260924/`。

## 部署修复

Fixed-2 首次 L0 评测在导入 evaluator 时发现运行包遗漏根目录 `training/`，尚未执行 episode。
已从本地 index 补同步该目录到两台独立 worktree，保留失败日志和目录
`train_2/eval_0-missing-training-runtime`，重新启动同一正式评测。随后已加载固定 revision
的模型和 backbone；两帧 prefix 保持不变。三路数据采集未受此修复影响。

后续训练终点、矩阵结果和成功率图均按
[冻结协议](../MIKASA_FIXED_LATENCY_GRID.md) 核验；不复用不同 seeds 的旧 Fixed-2 终评。

## 启动后的核验

Fixed-1/3 已进入正式 SFT，观察到连续 optimizer updates 与 loss 日志；实际 bootstrap 配置为
per-GPU batch64、accumulation1、5000 steps、H8、state71、gradient checkpointing=false。
Fixed-4 已完成正式 raw 数据采集并进入打包；controller 已串好相同训练和后续评测步骤。

第一台 H200 的 0/1 在模型加载期间被另一个任务启动的 trainer 占用，因此仅停止本轮
Fixed-2 评测进程树，保留 `eval_0-gpu-contention`，将评测移到空闲的 2/3 并从 L0 重跑。
第二台曾在 SSH 密钥协商阶段超时，客户端指定
`-o KexAlgorithms=curve25519-sha256` 后恢复连接；uptime 确认机器没有重启，三路继续运行。
未修改服务器 SSH 配置。

Fixed-2 的 L0/L1 已各完成200局：均为1/200，mean return `3.3988588637`，population SD
`5.8904802582`，逐局 return 向量完全一致；L2 为177/200、mean return `32.4183442223`、
population SD `8.7692328349`。三项执行审计均通过，均使用矩阵公共 seeds
`4242424242–4242424441`，prefix K 始终为2。低成功率照实保留，剩余设置继续评测；
其后 L3/L4 也完成，各为0/200。全部五格原始记录已回传本地，并从回传副本重新计算，
与远端 summary 完全一致；正式结果已写入 [RESULTS](../RESULTS.md)。

回传时修正了矩阵汇总的证据路径：从保存的 `eval_config.yaml` 读取原始 output_dir，
避免在另一台机器复核时仅因本地路径改变而误判 summary 不一致。修改先在本地完成，
相关64项测试及五格真实记录复核通过后同步；不改变模型推理、训练或评测协议。
