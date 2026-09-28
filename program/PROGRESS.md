# Benchmark 核心进展

最后核查：**2026-09-20**。本文只记录四个主任务的已核验结果、关键决策、当前缺口和下一步；清理前的完整时间线见
[历史快照](logs/PROGRESS-pre-cleanup-2026-09-17.md)，运行手册和专题记录见 [program 索引](README.md)。

固定公共版本：

- zero-latency 模型：[benchmark-models@e05ca26](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/e05ca26afde4f509b0f838742c522f3a068596a8)
- RTX 3090 profiles：[profiles@e8be1e5](https://huggingface.co/datasets/latency-sensitive-bench/profiles/tree/e8be1e5dfd309f85ab7ff704dc02bd4898b7ceb4/profiles)
- Parkour latency small：[benchmark-models@6348f44](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/6348f44ce65c1414dbc4f2ec46e0d5eb525ab770/latency-aware/extreme-parkour-go2/small-policy)
- MIKASA WanOFT latency-aware VLA：[benchmark-models@ad40d57](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/ad40d57c5747761705770aa557db0f287888f727/latency-aware/mikasa-intercept-grab-fast/vla/starvla-wanoft-h1/mikasa_wanoft_latency_3090_20260916T094307Z)
- MIKASA H1 corrected latency-aware VLAs：[benchmark-models@ae81bdb3](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/ae81bdb3b6c6605926255e91b499b4badf93a947/latency-aware/mikasa-intercept-grab-fast/vla)

## 阅读口径

- “训练完成”“评测完成”“达标”和“已发布”是四个独立状态。
- VLA return 使用 `episode_return_env`，长度使用 `survival_steps`；原生 PPO 保留源指标名。
- `±` 是 episode return 的总体标准差；跨 training seeds 的标准差会单独注明。
- 不同频率、latency profile、seed 集和 episode 数的结果不可直接混用。
- smoke 只证明流水线可运行，不作为策略质量结果；旧 running 消息不代表当前进程状态。
- 缺失指标写“未记录”，不从其他指标或模型推断。

## 四任务总览

| 任务 | Zero VLA | RTX 3090 profile | Latency small | Latency VLA | 当前结论 |
|---|---|---|---|---|---|
| Balance Simple | 4/4 完成 | 4/4 完成 | H8/H16 专题实验完成多组，尚无统一正式产物 | 正式四模型终点未完成 | 延迟控制可由 oracle/BC 部分恢复，但现有 RL/PPO 未超过 BC；新 DAgger 矩阵待终点结果 |
| MIKASA InterceptGrabFast | 4/4 完成；H8 zero SFT 与三路 naive DAgger 已发布 | 4/4 完成；H8 六路 conditioned DAgger 终评与发布完成 | 4/4 达标并发布 | H1 三路纠正评测完成；H8 九路 DAgger 均完成训练、终评与发布 | Figure 2 的 H1（Hub `56b2c095`）与 zero-latency H8 DAgger（Hub `5b533937`）完成 L0–L4 各 200 局：H1 三路在 L2 均为 0/200，H8 三路在 L2 为 115/123/101/200。[结果与证据](RESULTS.md) |
| Extreme Parkour Go2 | 3/4 完成 | 3/4 完成 | 3/4 训练、评测、发布完成 | 未找到正式产物 | 三个 small policy 在 profile 下均完成正式评测；未预设 acceptance threshold |
| HAIC PullCart | 1/4 完成 | 1/4 完成 | 正式终点未核验 | 未找到正式产物 | 当前只有 QwenOFT 正式 VLA 结果，任务覆盖仍不完整 |

## 公共 zero-latency small policy

| 任务 | checkpoint / 训练进度 | 已核验结果 | 状态与证据 |
|---|---|---|---|
| Balance | `checkpoint.pth`；25M 预算，选中 20,864,000 步 | 独立 L0 return/SR/n 未记录 | 已发布；checkpoint 名中的 702.976 不是独立 eval；[BALANCE](BALANCE.md) |
| MIKASA | `checkpoint.pt`；12,288,000 步 | strict success 100%；canonical 100/100 | 已发布；[MIKASA](MIKASA.md) |
| Parkour | `model_14999.pt`；15k iterations | reward 23.14±6.02；length 907.95±296.15；progress 1.00±0.01 | n/seed 未记录；[EXTREME_PARKOUR](EXTREME_PARKOUR.md) |
| HAIC | `checkpoint_final.pt`；499,908,608 frames | native success 93.8965%；源 `episode_cnt=2308` | 未达到 native 100% gate；[HAIC](HAIC.md) |

## 正式 zero-latency VLA 与 profile

### Balance Simple

50/50Hz；L0 各 100 episodes。profile 为 1×RTX 3090 temporal observation-to-action latency。

| 模型 | 训练 | L0 return ± SD | mean length | profile mean |
|---|---:|---:|---:|---:|
| QwenOFT | SFT 15k | 98.4357±15.4165 | 111.10 | 37.881ms |
| QwenGR00T | SFT 15k | 27.4931±2.4959 | 39.12 | 60.013ms |
| QwenPI v3 | SFT 15k | 41.1143±7.7280 | 54.44 | 99.176ms |
| WanOFT | SFT 15k | 90.6691±13.8048 | 100.97 | 300.024ms |

模型、评测和 profile 均在上方固定公共版本中；这些 L0 文件没有 task success 指标。

### MIKASA InterceptGrabFast

QwenOFT/QwenGR00T/QwenPI v3 为 20/20Hz，WanOFT zero VLA 主表也按 20/20Hz。
L0 使用 50 个共同 seeds `4242424242–4242424291`。本节依次列出 H1 zero baseline、
H1 single-action profile training，以及 H8 action-chunk 两组实验；不同协议不可混用。

| 模型 | 训练 | L0 return ± SD | task success | profile mean |
|---|---:|---:|---:|---:|
| QwenOFT | SFT 5k | 21.7417±16.4361 | 56% | 99.050ms |
| QwenGR00T | SFT 5k | 21.2928±16.6083 | 56% | 99.352ms |
| QwenPI v3 | SFT 5k | 21.9671±17.2095 | 54% | 148.689ms |
| WanOFT | SFT 5k | 18.9461±16.4028 | 42% | 300.539ms |

本表是共同 50-seed 补评测；中央模型中的其他 100-episode 文件使用不同协议，不与本表合并。

#### H1 single-action profile training

这批模型明确为 `action_horizon=1`，每次推理只输出一个 7D action，不是后面的 H8
action chunk。四个 profile-trained VLA 均完成 5,000-step SFT。QwenOFT、QwenPI v3、
QwenGR00T 的 2026-09-22 checkpoint 使用 200 episodes、reset seeds
`4242424242–4242424441`、latency seed `271828` 和 deterministic inference 完成
纠正后的 matched-profile 评测；WanOFT 保留既有 100-episode 结果。WanOFT 为 10Hz，
其余为 20Hz。表内为 `task success；mean return±SD`；旧 zero 比较项仅作协议内参考，
不与新 200-episode 样本合并。

| 模型 | zero→zero | zero→profile | profile→profile | 训练 / 发布 |
|---|---:|---:|---:|---|
| QwenOFT | 57%；22.0986 | 0%；0.4274 | 101/200；17.77506±15.43550 | SFT 5k、纠正评测与新 run ID 发布完成 |
| QwenPI v3 | 60%；23.0983 | 0%；0.2862 | 90/200；13.55261±13.75740 | SFT 5k、纠正评测与新 run ID 发布完成 |
| QwenGR00T | 63%；24.1050 | 0%；0.4136 | 100/200；17.51561±14.29526 | SFT 5k、纠正评测与新 run ID 发布完成 |
| WanOFT | 41%；16.8500 | 0%；0.3368 | 21%；6.8902 | SFT 5k 完成；已发布 |

三个新结果分别来自 `mikasa_qwenoft_latency_3090_2xh200_20260922`、
`mikasa_qwenpi_v3_latency_3090_2xh200_20260922`、
`mikasa_qwengr00t_latency_3090_2xh200_20260922` 的 step-5,000 checkpoint；mean latency
为 `99.091 / 148.740 / 99.365ms`。三路均完成 12,000 环境步，其中非 noop 步数为
`11,561 / 11,346 / 11,554`，逐局 return 向量两两不同。此前三路 0/200、
`0.189403±0.031062` 的结果因实际 12,000 步全部执行 noop 而失效；训练数据与 checkpoint
不受影响。旧 Hub 权重保留，但其附带的旧 eval metadata 不得引用；纠正后的 bundle 已以
新 `*_evalfix_20260923` run ID 发布，并在最终 revision
[`ae81bdb3`](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/ae81bdb3b6c6605926255e91b499b4badf93a947/latency-aware/mikasa-intercept-grab-fast/vla)
核验三条路径各含 7 个必需文件且 checkpoint 大小一致。
[三路纠正评测证据](../outputs/tasks/mikasa_intercept_grab_fast/h1_hold_last_reval_20260923/REPORT.md)、
[WanOFT 证据](../outputs/tasks/mikasa_intercept_grab_fast/wanoft_latency_20260916T094307Z/eval_matrix/results.json)；
[H1 合同与复核记录](LATENCY_RL_HANDOFF.zh-CN.md)。

#### H8 action-chunk zero-trained baseline

以下结果全部是 **H8 action chunk**。正式结果与代码均按正确 action-chunk 语义收尾；teacher
baseline 的三个 step-5,000 模型均完成 50-seed Zero 评测。Profile 与 Fixed-2 在原共同
reset seeds `4242424242–4242424291` 的 50 episodes 上，补跑不重叠 seeds
`4242424292–4242424441` 的 150 episodes，并合并为 200-episode 统计。共同协议为
20/20Hz、单 inference worker、每局 60 个实际 environment transitions；Profile 使用
latency seed 0，Fixed-2 严格为 100ms / 2 frames。环境从 reset 立即开始；首动作到达前
执行 noop 并消耗 budget，worker 忙时丢弃请求并继续旧 chunk 的
observation-aligned 后续 head。

| 模型 | Zero（50） | RTX 3090 Profile（200） | Fixed-2（200） |
|---|---:|---:|---:|
| QwenOFT | 25/50，50% | 29/200，14.5% | 33/200，16.5% |
| QwenPI v3 | 34/50，68% | 26/200，13.0% | 51/200，25.5% |
| QwenGR00T | 31/50，62% | 36/200，18.0% | 41/200，20.5% |

Zero / Profile / Fixed-2 return 分别为 QwenOFT
`19.42654±16.93815 / 7.45255±11.82559 / 8.84059±12.99718`、QwenPI v3
`26.42206±13.34111 / 5.12064±10.51648 / 10.79818±14.93077`、QwenGR00T
`22.92212±16.49738 / 8.32811±12.75548 / 8.56929±12.49691`。

九组 mean length 均为 60，原 50 局与新增 150 局的 execution audit 均通过，200 个
seed 已核验唯一。三个 Zero setting 达到原预设 `>=25/50` gate；按相同比例折算为
`>=100/200` 时，三个 Profile setting 均未达到。训练与三套 H8 bundle 发布均已完成。
Hub API 已在最终 revision
[`d546ea62d9be8f741f9b52c52a7a2122db1a1b42`](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/d546ea62d9be8f741f9b52c52a7a2122db1a1b42/zero-latency/mikasa-intercept-grab-fast/vla)
核验 OFT/PI/GR00T 三个目录各包含 8 个正式文件，checkpoint 分别为
`9,785,101,729 / 10,922,637,469 / 9,976,844,067` bytes。冻结首动作的旧 Profile
结果和 H200 wall-clock measured 结果已被此协议取代，不作为正式证据。
[200-episode 扩展结果](../outputs/tasks/mikasa_intercept_grab_fast/h8_zero_extended_eval_20260922/RESULTS.md)；
[原 50-episode 主证据](../outputs/tasks/mikasa_intercept_grab_fast/h8_teacher_baseline_20260920/REPORT.md)；
[运行记录](logs/mikasa-h8-zero-latency-2026-09-20.md)。

#### H8 action-chunk conditioned prefix training

conditioned-trained 六个 step-5,000 终点均完成训练；第一台三项使用标量单槽 reset
完成有效的 200 episodes 正式评测，第二台三项旧评测因物理 seed 错配失效；表内
zero-trained→Profile / Fixed-2 也是 200 episodes。共同协议为 20/20Hz、每局 60 transitions、capacity 1、
无首动作冻结；Profile 使用 RTX 3090 revision
`66414463f2a609316da63935941a79abf0a73ce9` 和 latency seed 0，Fixed-2 严格为
100ms / 2 frames。表格单元格为 `success_once；return mean±SD`，zero→zero 是两张表
共享的 zero-trained 参考。

Profile：

| 模型 | zero→zero | zero→profile | profile→profile |
|---|---:|---:|---:|
| QwenOFT | 25/50；19.42654±16.93815 | 29/200；7.45255±11.82559 | 15/200；3.92213±9.23980 |
| QwenPI v3 | 34/50；26.42206±13.34111 | 26/200；5.12064±10.51648 | —（旧 reset 失效） |
| QwenGR00T | 31/50；22.92212±16.49738 | 36/200；8.32811±12.75548 | —（旧 reset 失效） |

Fixed-2：

| 模型 | zero→zero | zero→fixed-2 | fixed-2→fixed-2 |
|---|---:|---:|---:|
| QwenOFT | 25/50；19.42654±16.93815 | 33/200；8.84059±12.99718 | 12/200；3.47094±8.22190 |
| QwenPI v3 | 34/50；26.42206±13.34111 | 51/200；10.79818±14.93077 | —（旧 reset 失效） |
| QwenGR00T | 31/50；22.92212±16.49738 | 41/200；8.56929±12.49691 | 13/200；4.07645±9.08028 |

第二台旧三项虽记录唯一 seeds `4242424242–4242424441` 且执行了非零动作，单槽 reset
实际仍传入整张 seed 列表，物理状态与记录 seed 不符。独立标量 reset 诊断重评三项均完成
200 episodes 且 execution audit 通过：Profile-GR00T `11/200`、Profile-PI `6/200`、
Fixed-2-PI `5/200`；诊断输出不覆盖旧正式 JSON，正式 replacement 尚待归档。
六路旧 SFT 端点均低于 `>=100/200` gate，未作性能验收；其训练权重后来作为下述
DAgger 起点归档发布。五路首请求 train/validation 泛化差距和三路 student 状态重标误差
递增促成了 student roll-in 对照；Profile-PI 全训练首请求 MAE `.0755`。
[完整诊断](../outputs/tasks/mikasa_intercept_grab_fast/h8_endpoint_diagnosis_20260923/REPORT.md)；
[运行记录](logs/mikasa-h8-conditioned-2026-09-21.md)。

#### H8 student-rollout DAgger（2026-09-23）

六路 conditioned 和三路 zero-latency naive 均从对应 SFT step-5,000 **训练权重**
初始化新优化器，连续完成 10×500 DAgger updates；每路有 11 次 100 局开发评测、
10 份各不少于 16,000 有效 suffix head 的 student rollout/PPO 标签，以及完整双 rank
step-5,000 trainer state。每轮复用常驻 trainer 的当前模型。共同协议为 H8×7、
20/20Hz、capacity 1；开发 seeds `4242424442–4242424541`，独立终评 seeds
`4242424542–4242424741`。Profile 使用 RTX 3090 profile，Fixed-2 为严格 100ms /
2 frames。最佳模型由开发集 100 局成功数选取，同分比较 mean return；以下所有终评
单元格为 **成功数/200；mean return**，执行审计均通过。

| Conditioned 模型 | SFT baseline | DAgger best | DAgger latest | 选中 step / 发布证据 |
|---|---:|---:|---:|---|
| Profile OFT | 12；3.348 | 138；25.752 | 同 best | 5000；[HF](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/af8d7bfacbb03b59b5b5eabbed331577dd5fa690/latency-aware/mikasa-intercept-grab-fast/vla/starvla-qwenoft-h8/h8-conditioned-dagger-20260923-profile) |
| Profile PI v3 | 8；2.176 | 89；17.558 | 同 best | 5000；[HF](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/af8d7bfacbb03b59b5b5eabbed331577dd5fa690/latency-aware/mikasa-intercept-grab-fast/vla/starvla-qwenpi_v3-h8/h8-conditioned-dagger-20260923-profile) |
| Profile GR00T | 13；3.744 | 134；25.235 | 125；23.500 | 4000；[HF](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/af8d7bfacbb03b59b5b5eabbed331577dd5fa690/latency-aware/mikasa-intercept-grab-fast/vla/starvla-qwengr00t-h8/h8-conditioned-dagger-20260923-profile) |
| Fixed-2 OFT | 10；2.962 | 175；31.680 | 同 best | 5000；[HF](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/af8d7bfacbb03b59b5b5eabbed331577dd5fa690/latency-aware/mikasa-intercept-grab-fast/vla/starvla-qwenoft-h8/h8-conditioned-dagger-20260923-fixed-2) |
| Fixed-2 PI v3 | 9；3.149 | 159；29.303 | 同 best | 5000；[HF](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/af8d7bfacbb03b59b5b5eabbed331577dd5fa690/latency-aware/mikasa-intercept-grab-fast/vla/starvla-qwenpi_v3-h8/h8-conditioned-dagger-20260923-fixed-2) |
| Fixed-2 GR00T | 12；3.509 | 164；29.866 | 同 best | 5000；[HF](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/af8d7bfacbb03b59b5b5eabbed331577dd5fa690/latency-aware/mikasa-intercept-grab-fast/vla/starvla-qwengr00t-h8/h8-conditioned-dagger-20260923-fixed-2) |

六份 conditioned SFT bundle 和六份 DAgger 选优 bundle 均已发布；Profile GR00T
另发布 step-5,000 `latest_model.pt`。按原 `>=100/200` gate，六路 conditioned 中
五路达到，Profile PI v3 的 `89/200` 尚未达到。SFT 与 DAgger 的完整发布 revision
见 [conditioned 记录](logs/mikasa-h8-dagger-2026-09-23.md)。12 份 bundle 已在
[`af8d7bfa`](https://huggingface.co/latency-sensitive-bench/benchmark-models/commit/af8d7bfacbb03b59b5b5eabbed331577dd5fa690)
归入 `latency-aware`，训练条件保留在 run ID 和 provenance 中。

| Naive 模型 / 终评条件 | SFT baseline | DAgger best | DAgger latest | 选中 step / 发布证据 |
|---|---:|---:|---:|---|
| OFT / Zero | 95；18.730 | 196；36.108 | 同 best | 5000；[HF](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/af125bd372c22c5b00e326d9af1af64ea3751b56/zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenoft-h8/h8-naive-dagger-20260923) |
| OFT / Profile | 13；4.599 | 101；18.596 | 同 best | 同上 |
| OFT / Fixed-2 | 20；5.679 | 109；20.955 | 同 best | 同上 |
| PI v3 / Zero | 118；21.933 | 199；36.785 | 196；35.968 | 4000；[HF](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/5b533937a72e9fa923ecf28b1c1be5a16bad488d/zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenpi_v3-h8/h8-naive-dagger-20260923) |
| PI v3 / Profile | 13；3.301 | 40；8.082 | 42；8.461 | 同上 |
| PI v3 / Fixed-2 | 42；7.938 | 115；21.473 | 117；22.180 | 同上 |
| GR00T / Zero | 132；24.611 | 192；33.514 | 194；33.756 | 3000；[HF](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/3fb1d4a7c2645715f0554f315d85032868bf15bb/zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwengr00t-h8/h8-naive-dagger-20260923) |
| GR00T / Profile | 15；4.542 | 73；14.314 | 83；15.941 | 同上 |
| GR00T / Fixed-2 | 25；6.229 | 95；17.838 | 93；18.814 | 同上 |

三路 naive DAgger 选优 bundle 均已发布；PI v3 与 GR00T 另存 step-5,000 latest。
六路 conditioned 的十轮 student/PPO suffix MAE 首请求范围为 `0.052–0.080`，
后续请求为 `0.176–0.252`；三路 naive 分别为 `0.048–0.056` 与 `0.118–0.124`
（按各轮请求数加权）。逐路值与训练/发布记录见
[conditioned](logs/mikasa-h8-dagger-2026-09-23.md) 和
[naive](logs/mikasa-h8-naive-dagger-2026-09-23.md)。

#### H8 Fixed-2 输入合同修复

旧 Fixed-2 DAgger step-5000（Hub `795468e53947d26314e891cbc375775c6e0bd491`）在
L0–L4 各 200 局、20/20Hz、60 transitions、seeds `4242424242–4242424441` 下为
**1、1、177、0、0 / 200**；[原始结果](../outputs/tasks/mikasa_intercept_grab_fast/fixed-grid-20260924/fixed2_column.json)。
调度审计通过，但旧 71D 输入中的常量维归一化会丢失跨延迟 prefix 信息，因而这些结果
不能单独归因于延迟泛化。保留旧数据、checkpoint 和结果作为修复对照。

已批准方案 A：63D 当前时刻对齐 prefix，动作输入固定 [-1,1]，只重训 Fixed-2，
SFT 5k + DAgger 5k；DAgger step 0 及每 500 steps 五档各 100 局，仍只按 L2 选 best，
最终 best 五档各 200 局共同旧 seeds。此前 Fixed-1/3/4 由用户停止，完整 5×5 尚未完成。
新 Fixed-2（`fixed2-prefix63-20260924`）SFT5k+DAgger5k及best5000五档终评完成；
共同seeds `4242424242–4242424441`、20/20Hz、60 transitions下，L0–L4为
**186、186、177、121、44 / 200**。[结果归档](../outputs/tasks/mikasa_intercept_grab_fast/prefix63-20260924-launch/final-evidence.tar.gz)。
新Fixed-1/3/4在首个完整DAgger500边界之前失败，原始trainer异常待取证，完整5×5未完成。
第一台原训练数据盘已不可访问：32块Local SSD均无RAID元数据，分布抽样全零；
新权重未备份到本地，第二台产物尚待核验。[故障证据](../outputs/tasks/mikasa_intercept_grab_fast/failure-audit-20260925/DISK_FINDINGS.md)。


最新报告口径：用户决定采用四路63D Fixed-1/2/3/4的共同 **DAgger step-3500** W&B快照填充论文grid；各列matched成功率为 **94/100、86/100、65/100、57/100**。每格100局development，seeds `4242424442–4242424541`、20/20Hz、60 transitions；Zero列仍为已发布step-5000的200局held-out结果，不视为同预算、同评测集的终评。附录保留完整SFT5k loss及DAgger至3500的loss/五档成功率曲线。[冻结数据与run来源](../paper/figures/fig20_mikasa_fixed_latency_learning_curves/README.md)。
W&B记录四路SFT完成；Fixed-1/2/3 DAgger记录至5000，Fixed-4训练loss至4000、最后完整五档评测为3500。新终评及模型发布状态本轮未核验，不由开发快照推断完成。

### Extreme Parkour Go2

50Hz control、10Hz decision；L0 各 100 episodes。

| 模型 | 训练 | L0 return ± SD | waypoint progress | profile mean |
|---|---:|---:|---:|---:|
| QwenOFT | SFT 5k + DAgger 5k | 23.3283±7.7106 | 0.9075 | 103.426ms |
| QwenGR00T | SFT 5k + DAgger 5k | 22.3789±6.7888 | 0.9163 | 157.357ms |
| QwenPI v3 | SFT 5k + DAgger 5k | 21.8744±7.9515 | 0.8500 | 238.729ms |
| WanOFT | 未完成 | — | — | — |

### HAIC PullCart

50/50Hz。QwenOFT 完成 SFT 5k + DAgger 5k；100-episode L0 return
13.3497±8.2155、mean length 187.09、task success 56%，RTX 3090 profile mean 126.288ms。
其余三模型未找到正式 zero VLA、profile 或 latency-aware 产物。

## Latency-aware small policy

### MIKASA

单 worker、temporal profile、deterministic mean action；环境 seed `20260910`、latency seed `271828`，各 256 episodes。
既有 gate 为 `success_at_end >= 90%`。WanOFT 为 10Hz，其余为 20Hz，不能把四行当作同频率公平比较。

| Profile | checkpoint step | success_at_end | success_once | mean return | 状态 |
|---|---:|---:|---:|---:|---|
| QwenOFT 20Hz | 13,824,000 | 98.83% | 99.61% | 33.6446 | 达标、已发布 |
| QwenGR00T 20Hz | 20,951,040 | 98.05% | 98.05% | 31.7504 | 达标、已发布 |
| QwenPI v3 20Hz | 15,360,000 | 96.88% | 98.83% | 28.6463 | 达标、已发布 |
| WanOFT 10Hz | 23,040,000 | 93.75% | 98.44% | 30.8664 | 达标、已发布 |

完整训练/eval 配置和 profile 固定版本见各发布 bundle；旧 WanOFT 20Hz 结果不替代当前 10Hz 版本。

### Parkour

三个模型均训练 3,000 PPO iterations，并在各自 RTX 3090 profile 与 fixed 0ms 下各评测 100 episodes；
协议为 50Hz control、10Hz decision、seed 1、1500 raw-frame cap。

| 模型 | Profile return ± SD / progress | Zero return ± SD / progress | 状态 |
|---|---|---|---|
| QwenOFT | 16.2985±5.8936 / 0.8314 | 16.7771±6.1831 / 0.8329 | 已发布 |
| QwenPI v3 | 17.1554±4.3398 / 0.9214 | 11.8676±4.5410 / 0.6757 | 已发布 |
| QwenGR00T | 18.6098±3.5573 / 0.9814 | 12.7761±4.3001 / 0.7900 | 已发布 |

原始结果与 checkpoint 审计见 [completion audit](../outputs/latency_rl/parkour/h200_20260916_completion_audit.md)。
该项目没有预设 acceptance threshold，因此这里只声明训练、评测和发布完成。

## Balance 关键实验与决策

### Action chunk 与监督合同

- H8 profile 合同修复后，6M-step 配对结果从 mean length 72.88 提升到 172.14，`>=100` 从 3% 提升到 93%；旧结果不能代表修复后实现。[CHUNK](CHUNK.md)
- 第一版 4-seed 消融中，full mean length 304.73±232.32，删除 `ABCHJ` 后为 417.24±88.05 且 `>=100` 为 100%。这支持删除对应机制，但 D/E 组合与单独消融仍未完成。[CHUNK_RL_ABLATION](CHUNK_RL_ABLATION.md)
- 延迟标签必须对应实际 application state 和同一 request 真正执行的 heads；“整段 teacher chunk”与“executed heads”是需要正式对照的两个不同合同。

### Fixed100 诊断

- startup no-op 曾把绝对 SONIC 默认姿态当作 normalized MuJoCo action；修复前依赖该启动行为的 delayed 结果不再作为当前证据。
- 修复后 H16 QwenPI 50k optimizer-step 对照仍失败：temporal profile mean length 61.75，fixed100 mean length 69.625，各 n=8，均低于 100 rejection gate。[pair record](../outputs/tasks/humanoidbench_balance_simple/h16_profile_fixed100_corrected_noop_50k/20260915T174801Z/launch.json)
- native fixed100 oracle 在 32 seeds 上达到 mean length 673.1875、`>=100` 32/32、`>=500` 23/32，说明该延迟合同本身可解；问题在学习和状态覆盖，而不是单纯“延迟必然不可控”。[oracle](../outputs/tasks/humanoidbench_balance_simple/h16_fixed100_delayed_imitation/20260916T061955Z/oracle.json)

### BC、RL 与 PPO

所有下表 fixed100 结果均使用 H16、50/50Hz、capacity 1；不同表行的 seed 集按证据文件保留。

| 路线 | 正式结果 | 结论 |
|---|---|---|
| Execution-aligned BC | mean length 387.906，`>=100` 31/32，`>=500` 8/32 | 明显优于 mapped-teacher baseline 71.0，但只有 oracle 的 57.6% |
| Student-state correction | mean length 464.188，`>=100` 30/32，`>=500` 13/32 | 比等预算旧数据继续训练好；仍只有一个 training seed |
| Joint TD3 updates | 464.188 → 24.313，32/32 paired seeds 退化 | 现有在线 actor update 会摧毁监督策略 |
| Lower LR + frozen-BC anchor | 四个 endpoint mean length 29.69–65.47 | 只能减缓漂移，不能修复 |
| Return-guided bounded regression | 463.156 vs BC 464.188，95% interval 跨 0 | 保留点性能但没有证明增益 |
| SF MC PPO | candidate latest 514.563 vs BC 527.406，n=64 | 训练/评测完成；stability gate 与 policy acceptance 均失败 |

主要证据：[BC/监督修正](../outputs/tasks/humanoidbench_balance_simple/h16_fixed100_delayed_imitation/20260916T061955Z/followup_20260916T070429Z/REPORT.md)、
[anchored RL](../outputs/tasks/humanoidbench_balance_simple/h16_fixed100_delayed_imitation/20260916T061955Z/anchored_rl_20260916T074351Z/REPORT.md)、
[return-guided](../outputs/tasks/humanoidbench_balance_simple/h16_fixed100_delayed_imitation/20260916T061955Z/return_guided_rl_20260916T085703Z/REPORT.md)、
[MC PPO](BALANCE_MC_PPO.md)。

### Profile BC/DAgger 与当前状态

- Profile BC 在 64 fresh paired streams 上将 mean length 从 69.125 提升到 111.156，paired gain +42.031、95% interval `[32.296, 52.391]`；但仍有 24/64 在 100 frames 前结束，`>=500` 为 0。[report](../outputs/tasks/humanoidbench_balance_simple/profile_bc/formal_s1_20260916T1330Z/REPORT.md)
- 旧 QwenOFT profile DAgger 每轮错误地从长期不变的 `best` model 重启，不能作为连续 5k DAgger 结果；修正版要求 `latest` 驱动采集并连续恢复 optimizer/scheduler/RNG。
- **QwenOFT H8 训练与发布完成**：`zero_full` 与 `profile_conditioned` 均完成 5k bootstrap + 5k DAgger optimizer updates，best/latest 均为各自 `bundles/dagger_10_5000`。`zero_full` 在 zero-latency、共同 held-out seeds 820001--820064 下为 mean length `297.8125`、mean return `275.56409`、`>=100` `64/64`、`>=500` `4/64`。发布 revision 分别为 `24cc874d59bec339bcf40046764e52fa26334dc8` 和 `04bf16cfef9a469354256ed179db4fe548be03d3`。
- **QwenPI H16 训练与发布完成**：`zero_full` 与 `profile_conditioned` 均完成 5k bootstrap + 5k DAgger optimizer updates，best/latest 都是各自 `bundles/dagger_10_5000`。`zero_full` 在 zero-latency、共同 held-out seeds 820001--820064 下为 mean length `351.59375`、mean return `328.60200`、`>=100` `64/64`、`>=500` `7/64`。发布 revision 分别为 `72747b40b4289766a624b7cf004ab93cf48e5668` 和 `01bf53c4d69a845ef76bcd595b606fc66a4c6398`。
- **QwenGR00T H16 训练与发布完成**：`zero_full` 与 `profile_conditioned` 均完成 5k bootstrap + 5k DAgger optimizer updates，best/latest 均为各自 `bundles/dagger_10_5000`。`zero_full` 在 zero-latency、共同 held-out seeds 820001--820064 下为 mean length `287.609375`、mean return `267.52374`、`>=100` `64/64`、`>=500` `3/64`。发布 revision 分别为 `19f71fe13865a7d8a8bae95e3d4939822d6ad1c0` 和 `891220398157a663aaf3233e42145c077a327acc`。

### 固定均值 prefix 的因果评测（2026-09-20）

三模型两臂均复用上面的 `bundles/dagger_10_5000` 与既有发布 revision，未重新训练。
**六组正式评测及协议核验完成**，各 64 episodes；50/50Hz、capacity 1、1000-frame cap，
各自 RTX 3090 temporal profile（instance `3a0d42681a03715c`）、profile seed 271828、
inference seed 941731、共同 held-out seeds 820001–820064。Baseline 不传显式 prefix；
conditioned 使用 `K=ceil(profile mean_ms/20)`，OFT/PI/GR00T 为 2/5/4，
不读取本次未来延迟。QwenOFT 不消费显式 prefix，其原有 action-context 输入保持不变。

| 模型 | Baseline length ± SD | Conditioned length ± SD | Baseline / conditioned return | >=100（baseline / conditioned） | Paired length gain [95% CI] |
|---|---:|---:|---:|---:|---:|
| QwenOFT H8 | 63.29688 ± 12.23555 | 133.35938 ± 34.77039 | 50.74563 / 114.62513 | 1/64 / 51/64 | +70.06250 [61.20312, 79.01562] |
| QwenPI H16 | 57.03125 ± 7.63006 | 86.21875 ± 30.93515 | 46.84542 / 72.31230 | 0/64 / 24/64 | +29.18750 [22.03125, 36.54727] |
| QwenGR00T H16 | 56.81250 ± 13.75667 | 104.03125 ± 31.15935 | 45.34041 / 90.42080 | 0/64 / 37/64 | +47.21875 [39.32812, 54.98438] |

六组 `>=500` 均为 0/64，chunk underflow 均为 0。10,000 次 seed-paired bootstrap
（seed 42）下三组 length/return gain 区间均为正；这仅是这些 checkpoint 的相对优势，
不代表跨 training seeds 结论或新增性能达标判定。本轮不重新发布模型。
旧 oracle-delay / 强制 prefix baseline 比较已被本协议替代，原始结果保留用于审计。
GR00T 取消 baseline 强制 prefix 后 mean length 提升 12.21875，95% CI
[8.546875, 16.078125]；两个 flow conditioned 改用固定 K 后的 length 变化区间均跨 0。
QwenOFT 两臂的逐 seed return/length 与旧结果完全一致。
[主证据与完整指标](../outputs/tasks/humanoidbench_balance_simple/causal_profile_eval_20260920/REPORT.md)；
[评测合同与复现入口](BALANCE_CAUSAL_EVAL.md)。

## 下一决策

- **Balance**：固定均值 prefix / no-prefix 六组因果评测完成，以新协议比较为准；已决定新训练数据与所有学生推理统一使用 profile 均值固定 K；公共合同迁移及相关测试已验证（[证据](../outputs/tasks/humanoidbench_balance_simple/fixed_k_unification_20260920/README.md)），尚未重训；下一步决定是否按新标签合同重训。D/E 消融仍是独立待办。
- **MIKASA**：论文采用63D四路共同DAgger3500开发快照与既有Zero列；终评、发布与开发快照分别记录。后续若恢复实验，再核验剩余预算、共同held-out终评及产物发布，不以当前grid替代。[快照协议](../paper/figures/fig6_latency_robustness/mikasa/README.md)。
- **Parkour**：补 WanOFT H1/profile/latency-small；在新的质量声明前先定义 acceptance threshold。
- **HAIC**：核验其余三模型、RTX 3090 profiles、latency-small 终点和正式 latency-aware VLA；不以 smoke 代替质量结果。
