# Balance Simple / GR00T 工程状态

## Goal and scope

让 GR00T VLA 在 `g1_sonic_balance_simple` 上直接拟合 teacher，把 closed-loop return 稳定提高到 500 以上（teacher 约 900–1000）。当前保持最简单的 direct VLA；不同时引入 SONIC、residual、分层控制或新 action head。

复现入口见 [`docs/humanoidbench/cookbook.md`](docs/humanoidbench/cookbook.md)；正式公开的 [dataset repo](https://huggingface.co/datasets/latency-sensitive-bench/humanoidbench) 和 [model repo](https://huggingface.co/latency-sensitive-bench/humanoidbench) 均为 `latency-sensitive-bench/humanoidbench`。

## Current status

- 新一轮 fresh 10k 正式训练已经完成，W&B 状态为 `finished`，当前没有训练进程，GPU 已释放。
- 10-seed 最佳/最终结果来自 checkpoint-10000：return `353.532 ± 133.571`，mean length `371.0`；相比历史 5k 最佳 `211.844` 明显提高，但仍未达到目标 500。
- checkpoint-9500 和 checkpoint-10000 是当前仅保留的权重；10k cosine schedule 已耗尽，末次 LR 为 `2.5956e-12`，不要直接 resume 这个 optimizer/scheduler。
- 20 个 periodic eval 均完成；每个 eval 10 episodes、一个原始 head-camera clip，invalid/dropped/underflow actions 全为 0。
- 20 个 eval 的代表 episode 已用保存的 action trace 重渲染为 `cam_default` 第三人称视频；没有重新运行模型推理，原始视频均保留。

## Current formal setup

- 模型语义：`198D state → H40 repeat-current → 43D absolute action`。
- 保留 pretrained H40 token/head 结构，标签为 `[a_t, …, a_t]`；每个 50 Hz 控制帧重新规划，只执行 chunk index 0。
- fixed Gaussian seed 0；train/inference 使用同一 latent；one-step flow（train `t=0`，inference 一次 Euler）；action-head dropout 0。
- action mask 为 `[B,H,D]`，屏蔽 43:132 padding；同一 observation 的推理完全确定。
- 数据：90 train episodes、86,048 frames、90 个不同 seeds（10–879），平均 956 frames/episode；val 为 10 episodes、9,736 frames。`episode_sampling_rate=0.1` 只是把 timestep 分到不同 shard，没有丢掉 90% 数据。
- 下一轮 fresh recipe：2 GPUs，global batch 128，peak LR `1e-4`，250 warmup steps（ratio `1/60`），覆盖 15,000 optimizer steps 的 cosine decay；约 22.31 nominal epochs。
- 每 500 steps 保存 → validation → seeds 0–9 closed loop → 写入同一 W&B run → 恢复 optimizer/scheduler。

跨机器下载、训练和模型自动上传入口：`scripts/official_gr00t/train_balance_simple_15k.sh`；周期训练与评估核心：`scripts/official_gr00t/train_balance_simple_periodic_eval.py`。

10k 正式产物：

- W&B：https://wandb.ai/saberrr-zju/finetune-gr00t-n1d7/runs/fp10k1e4
- Output：`outputs/official_gr00t/g1_balance_simple_obs198_h40_repeat_fixedz0_onestep_dropout0_frompretrained_bs128_lr1e4_10k_periodic500`
- Checkpoints：上述目录中的 `checkpoint-9500`、`checkpoint-10000`。
- 主日志：`outputs/official_gr00t/logs/g1_balance_simple_obs198_h40_repeat_fixedz0_onestep_dropout0_frompretrained_bs128_lr1e4_10k_periodic500.log`
- 首次鉴权失败日志：`outputs/official_gr00t/logs/g1_balance_simple_obs198_h40_repeat_fixedz0_onestep_dropout0_frompretrained_bs128_lr1e4_10k_periodic500.attempt1.log`；这次失败发生在 step 0 前，没有可用 checkpoint。

## Formal 10k result

| step | val body MAE | val all MAE | 10-seed return | mean length |
|---:|---:|---:|---:|---:|
| 500 | 0.069924 | 0.051008 | 63.424 ± 8.282 | 86.5 |
| 1000 | 0.065812 | 0.047276 | 70.279 ± 17.065 | 77.0 |
| 1500 | 0.065539 | 0.046783 | 60.646 ± 2.626 | 76.4 |
| 2000 | 0.060883 | 0.043833 | 81.971 ± 12.902 | 99.0 |
| 2500 | 0.061060 | 0.043437 | 76.132 ± 14.820 | 87.4 |
| 3000 | 0.057481 | 0.040758 | 107.390 ± 14.417 | 124.2 |
| 3500 | 0.058176 | 0.040995 | 104.039 ± 22.941 | 116.2 |
| 4000 | 0.057258 | 0.040426 | 107.622 ± 10.590 | 128.1 |
| 4500 | 0.056004 | 0.039697 | 202.449 ± 51.634 | 216.2 |
| 5000 | 0.056805 | 0.040131 | 129.333 ± 14.520 | 142.5 |
| 5500 | 0.053792 | 0.037977 | 194.826 ± 55.875 | 212.6 |
| 6000 | 0.054135 | 0.038262 | 135.982 ± 19.053 | 150.4 |
| 6500 | 0.052961 | 0.037226 | 205.294 ± 105.629 | 222.5 |
| 7000 | 0.051700 | 0.036425 | 199.866 ± 65.180 | 212.2 |
| 7500 | 0.051586 | 0.036284 | 193.143 ± 74.423 | 202.5 |
| 8000 | 0.050746 | 0.035756 | 294.753 ± 104.671 | 312.0 |
| 8500 | 0.050516 | 0.035533 | 243.261 ± 100.844 | 259.8 |
| 9000 | 0.049983 | 0.035177 | 332.190 ± 130.518 | 350.7 |
| 9500 | 0.049987 | 0.035184 | 330.300 ± 155.922 | 348.0 |
| 10000 | **0.049929** | **0.035149** | **353.532 ± 133.571** | **371.0** |

Checkpoint-10000 的逐 seed return 为：

`[275.822, 382.193, 454.332, 608.010, 143.887, 232.278, 454.882, 439.554, 331.636, 212.722]`

对应 episode lengths：

`[292, 395, 479, 626, 161, 249, 473, 459, 347, 229]`

一个 seed 已达到 return `608.010` / length `626`，但低端 seeds 仍只有约 `144–232`。所有 20 次 eval 的 `repeat_std_max=0`；最终 checkpoint metadata 已核对为 state 198D、H40 repeat delta 全 0、fixed noise seed 0、one-step、action-head dropout 0。

## Historical 5k reference

历史 run 使用相同 peak LR 和 250 warmup steps，但 cosine horizon 只有 5k；因此不能只按相同步数与新 run 横向比较。

- W&B：https://wandb.ai/saberrr-zju/finetune-gr00t-n1d7/runs/fp5k1e4
- Output：`outputs/official_gr00t/archive/experiments/g1_balance_simple_obs198_h40_repeat_fixedz0_onestep_dropout0_frompretrained_bs128_lr1e4_5k_periodic500`
- 最佳 checkpoint-4500：return `211.844 ± 49.587`，body MAE `0.053877`，mean length `232.5`。
- checkpoint-5000：return `206.709 ± 58.818`，body MAE `0.053731`，mean length `221.8`；末次 LR `1.09e-11`。

## New insights after 10k

1. **更长的有效 schedule 明显有效，但尚未解决任务。** 10k 最佳 return `353.532`，比 5k 最佳 `211.844` 高约 67%；最终 body MAE 从 `0.053731` 降到 `0.049929`，约下降 7.1%。这支持继续保留 direct H40 setup，但不能外推为“再翻倍 steps 一定过 500”。
2. **step 数必须连同 LR path 一起解释。** 新 run 在 step5000 只有 `129.333`，低于旧 run 的 `206.709`；两条 run 的 decay horizon 和数据顺序复现条件不同。相同步数不代表相同优化状态。
3. **主要提升出现在后段，closed loop 曲线仍很抖。** 8k–10k 的 return 为 `294.8, 243.3, 332.2, 330.3, 353.5`。10-seed std 已到 `105–156`，附近 checkpoint 的均值差不足以可靠排序。
4. **teacher-distribution MAE 已近平台，但鲁棒性仍不足。** 9k/9.5k/10k body MAE 几乎不变（约 `0.04998/0.04999/0.04993`），同时有的 seed 超过 500、有的很早失败。模型已经能产生成功轨迹，当前瓶颈越来越像 lower-tail robustness / off-manifold covariate shift，而不是所有 seed 上统一不会平衡；不过现有 val probe 太窄，尚不能据此直接决定加 recovery data。
5. **offline MAE 只能看趋势，closed loop 才是 gate。** return 在相邻 eval 间的变化远大于 MAE 变化；不能用更低 loss 代替多 seed 闭环评估。
6. **已排除当前执行链中的明显随机/队列异常。** fixed latent 后推理 deterministic；所有正式 eval 的 invalid、dropped、underflow actions 均为 0；repeat-current 语义和 checkpoint metadata 一致。
7. **更多同分布 teacher 帧不等于 recovery coverage。** 当前已有 90 条长 episode/不同 seed；普通 teacher rollout 不包含 student 偏离 teacher manifold 后的状态。若后续证据指向 covariate shift，数据增量应增加恢复状态覆盖，而不是复制同分布长轨迹。
8. **此前结论仍成立。** H40 repeat-current 在短预算下显著优于原始 H40 future-action；tiny set 能拟合到 body MAE `0.0156`，说明 head/objective 有拟合能力；可见误差主要在关键 body/腿部而不是 hand。

## Next decision

下一轮已确定为 fresh 15k；其余判断按以下顺序收紧：

1. 用更多 seeds（建议 50）重新评估仅存的 checkpoint-9500 和 checkpoint-10000，报告 mean、std、lower tail 和成功率，再选择 checkpoint。两者当前均值只差约 23，而 std 为 `134–156`，10-seed 结果不足以稳定排序。
2. 把 validation probe 扩展为有代表性的跨 episode aggregate。当前 periodic validation 只取 val episode 0 前 128 samples，不能可靠判断 teacher-distribution 误差是否真的平台。
3. 下一轮保持 direct H40 和 batch 128，运行 fresh 15k schedule；warmup 仍为 250 steps。10k scheduler 已归零，不能直接 resume。
4. 若 offline error 已低且更大 seed eval 仍显示明显 lower-tail failure，再加入 off-manifold/recovery state coverage。不要同时改变数据、batch、模型和 schedule，否则无法归因。

由于只保留 checkpoint-9500/10000，其他 periodic eval 目录只能用于分析轨迹，不能恢复对应 step 的权重。

## Third-person rerenders

- 每个 step 的输出目录模式：`outputs/official_gr00t/g1_balance_simple_obs198_h40_repeat_fixedz0_onestep_dropout0_frompretrained_bs128_lr1e4_10k_periodic500/evals/eval_step{STEP}/videos/`。
- checkpoint-10000 示例：`outputs/official_gr00t/g1_balance_simple_obs198_h40_repeat_fixedz0_onestep_dropout0_frompretrained_bs128_lr1e4_10k_periodic500/evals/eval_step10000/videos/bin_00_episode_000007.mp4`。
- 每个目录包含一个 MP4 和 manifest；`steps.jsonl` 通过 symlink 指向原 eval trace，避免复制大文件。
- 20 个 replay 的 reward/done 与原轨迹一致，20 个 MP4 均通过 ffprobe。
- `cam_default` 是 HumanoidBench 的标准外部第三人称视角：观察角固定但 `trackcom` 会跟随机器人 COM，并非严格 world-fixed。`cam_kitchen` 才是 world-fixed，但对该任务取景较差。

## Ruled out or deferred

- 不把 model horizon/token sequence 改成 1；继续用 H40 repeat-current。
- 不使用 all-zero latent；train/inference 固定同一 Gaussian 并 mask padding。
- 不再试 min/max normalization、全局关闭 clipping、startup50 mix 或 body-only 29D。
- 旧 LR `1e-5` warm-start 是混杂实验，不能据此断言 `1e-5` 普遍无效。
- SONIC/residual、分层控制和 regression-head replacement 不属于近期下一步。
- recovery data 不再无条件延后，但必须由更大 seed eval 和代表性跨 episode validation 的结果触发。
- eval 每次保留一个 clip 即可用于人工 debug；不需要为所有 episode 生成视频。

## Implementation and verification

- Root runner 已配置为下一轮 fresh 15k schedule，并支持 `--start-step`、周期 stop/eval/resume、W&B teardown 和最终模型上传。
- `third_party/Isaac-GR00T` 已实现 repeat-current、fixed latent、one-step flow、dropout override、3D action mask 和 `PeriodicStopCallback`。
- 验证结果：GR00T 聚焦回归 `44 passed`；video tests `4 passed`；runner Ruff、syntax 和 diff check 通过。
- 当前 Transformers tokenizer 初始化仍会访问 Hugging Face Hub 的 gated metadata；仅有完整本地 cache 仍不够。后续训练进程必须继承可访问 `nvidia/Cosmos-Reason2-2B` 的有效 Hugging Face credential。不要把 token 写入脚本、日志或本文档。
- 已知限制：periodic validation 只取 val episode 0 前 128 samples；resume sampler 的严格数据顺序复现仍未封口。
- root worktree 还有用户的无关改动；后续不要清理、覆盖或提交不属于本任务的文件。

## Suggested skills

- 修改 runner、eval 或训练代码时使用 `$ponytail`，保持最小变更并复用现有路径。
- 在落地下一项实验变更或 staging 前使用 `$occam-review`，重点检查是否同时引入了多个不可归因变量。
- 下一次需要交接长任务状态时使用 `$handoff`，原位更新本文档，不再另建重复状态文件。
