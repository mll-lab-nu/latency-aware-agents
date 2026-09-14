# Benchmark 进度与评测结果

最后核查：**2026-09-14 13:22 UTC**。文档入口：[program 索引](README.md)。
本表记录已确认的产物和评测；各行证据采用固定版本，后续工作确认结果后同步更新。

- 模型仓库：[e05ca26afde4f509b0f838742c522f3a068596a8](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/e05ca26afde4f509b0f838742c522f3a068596a8)，最后修改 2026-09-14T01:41:19.000Z。
- Profile 仓库：[e8be1e5dfd309f85ab7ff704dc02bd4898b7ceb4](https://huggingface.co/datasets/latency-sensitive-bench/profiles/tree/e8be1e5dfd309f85ab7ff704dc02bd4898b7ceb4/profiles)，最后修改 2026-09-14T12:59:01.000Z。
- 四模型：`qwenoft`、`qwengr00t`、`qwenpi_v3`（用户所称 qwenpi）、`wanoft`；主表为 StarVLA H1，official GR00T 和其他实验另列。

## 状态与口径

“已完成”只表示相应训练/评测完成；“达标”按既有 task gate 判断；“已发布”表示远端产物存在。
“未核验”表示没有最终证据；“未找到正式产物”仅描述本次仓库/本地产物核查，不断言从未运行。
两步或少量 episode 的 smoke 单独记录。旧 running/远端训练消息不作为当前进程状态。

VLA return 使用 `episode_return_env`，长度使用 `survival_steps`，单位为该配置的控制步；
原生 PPO 保留源 `metrics.return` 与 `metrics.episode_len`。
± 是 episode return 的标准差，跨 training seeds 的 SD 会另行注明。
成功率保留源指标名：`success_at_end`、`success_once`、task `success` 分别记录。
源文件没有的指标/时间/seed 写“未记录”，不从 return、长度或其他模型推断。

## 四任务总览

七阶段依次为：zero-latency small model、zero-latency VLA training、zero-latency VLA eval、
latency profile、latency-aware small model、latency-aware VLA training、latency-aware VLA eval。

| 任务 | 0 small | 0 VLA train | 0 VLA eval | RTX 3090 profile | Latency small | Latency VLA train | Latency VLA eval / 下一步 |
|---|---|---|---|---|---|---|---|
| Balance Simple | FastTD3 teacher 已发布 | 4/4；各 SFT 15k | 4/4；QwenOFT return98.4357±15.4165，n=100 | 4/4 | H1/H8 实验已完成，合同不同；H8 消融见下 | 四模型正式产物未找到；旧 GR00T 两步 smoke | 正式结果未找到；D/E 组合与单独消融待验证 |
| MIKASA InterceptGrabFast | PPO teacher；100% | 4/4；各 SFT 5k | 4/4 本地补评测；QwenOFT success56%，return21.7417±16.4361，n=50；两类HF标准文件缺失 | 4/4 | 4/4 已发布；end SR 96.88–98.83%（20Hz），Wan 93.75%（10Hz） | QwenOFT SFT 5k 本地完成；其他未找到正式产物 | QwenOFT temporal success 57%，return 19.4161±15.0048；补齐发布/评测元数据 |
| Extreme Parkour Go2 | PPO teacher；reward 23.14±6.02 | 3/4；各 SFT 5k + DAgger 5k | 3/4；QwenOFT return23.3283±7.7106，n=100 | 3/4 | 历史训练暂停/退化；本轮正式最终结果未核验 | 四模型正式产物未找到；旧 GR00T 两步 smoke | 正式结果未找到；Wan H1 与 latency small 最终状态待核验 |
| HAIC PullCart | PPO teacher；native success 93.8965%，未达 native 100% gate | 1/4；QwenOFT SFT 5k + DAgger 5k | 1/4；QwenOFT success56%，return13.3497±8.2155，n=100 | 1/4 | 历史训练/smoke 存在，正式最终结果未核验 | 四模型正式产物未找到；旧 GR00T 两步 smoke | 正式结果未找到；其余三模型与 latency small 最终状态待核验 |

## 公共 zero-latency 小模型

| 任务 | checkpoint / 训练进度 | 已记录评测 | 证据与限制 |
|---|---|---|---|
| Balance | `checkpoint.pth`；训练总预算25M，选中20,864,000步模型 | 中央 `wandb-summary.json` 是训练摘要，独立 L0 return/SR/n 未记录 | [B-teacher]；选择分数702.976来自训练checkpoint名，不标作eval；[runbook](BALANCE.md) |
| MIKASA | `checkpoint.pt`；tuned zero teacher 12,288,000步 | strict `success_rate=100%`；文档 canonical 为100/100 | [M-teacher]；该JSON未记return/n/seed，100局口径见[MIKASA](MIKASA.md#8-实际运行记录) |
| Parkour | `model_14999.pt`；fixed LR，15k iterations | reward 23.14±6.02；length 907.95±296.15；waypoint progress 1.00±0.01；n/seed未记录 | [任务运行记录](EXTREME_PARKOUR.md#9-实际运行记录)；数值来自文档，独立原始eval尚未回读 |
| HAIC | `checkpoint_final.pt`；499,908,608 frames，3814 iterations | native `success=0.93896484375`；2048 env，源 `episode_cnt=2308`；return未记录 | [任务运行记录](HAIC.md#5-实际运行记录)；成功率的分母语义按原生metrics保留；VLA标签源已接受，native student gate未达到 |

## Balance Simple：四模型明细

控制/观测频率 **50/50 Hz**。zero small 使用上方公共 teacher。

L0 数值来自本次固定中央模型revision的 `eval_l0.json`，各100 episodes；
JSON未记录评测时间，episode seed记录保留在该文件。训练seed见provenance。

| 模型 | 0 VLA training | 0 VLA eval：return ± SD；长度；成功率；n | Latency profile | Latency small | Latency VLA training | Latency VLA eval |
|---|---|---|---|---|---|---|
| qwenoft | 已完成SFT 15,000；[B-qwenoft-source] | 98.4357±15.4165；111.10步；SR未记录；n=100 [B-qwenoft-eval] | 已发布；均值 37.88ms [B-qwenoft-profile] | H1/H8实验完成；profile合同见下方实验表 | 未找到正式产物 | 未找到正式评测 |
| qwengr00t | 已完成SFT 15,000；[B-qwengr00t-source] | 27.4931±2.4959；39.12步；SR未记录；n=100 [B-qwengr00t-eval] | 已发布；均值 60.01ms [B-qwengr00t-profile] | 未找到该profile对应正式最终产物 | 未找到正式产物 | 未找到正式评测 |
| qwenpi_v3 | 已完成SFT 15,000；[B-qwenpi_v3-source] | 41.1143±7.7280；54.44步；SR未记录；n=100 [B-qwenpi_v3-eval] | 已发布；均值 99.18ms [B-qwenpi_v3-profile] | 未找到该profile对应正式最终产物 | 未找到正式产物 | 未找到正式评测 |
| wanoft | 已完成SFT 15,000；[B-wanoft-source] | 90.6691±13.8048；100.97步；SR未记录；n=100 [B-wanoft-eval] | 已发布；均值 300.02ms [B-wanoft-profile] | 未找到该profile对应正式最终产物 | 未找到正式产物 | 未找到正式评测 |

## MIKASA InterceptGrabFast：四模型明细

控制/观测频率 **20/20 Hz**。zero small 使用上方公共 teacher。

L0 主对比采用本地 **50个共同seed，4242424242–4242424291** 的补评测（2026-09-08）；
下载模型revision `aed114b6c4746917e84fabc3eaf0999be25aadb9`。100局复核/远端评测另列，保留各自版本。

| 模型 | 0 VLA training | 0 VLA eval：return ± SD；长度；成功率；n | Latency profile | Latency small | Latency VLA training | Latency VLA eval |
|---|---|---|---|---|---|---|
| qwenoft | 已完成SFT 5,000；[M-qwenoft-source] | 21.7417±16.4361；60.00步；task success 56%；n=50 [M-qwenoft-eval] | 已发布；均值 99.05ms [M-qwenoft-profile] | 已发布；strict评测见下方小模型表 | 本地SFT 5,000；未发布 [M-latency-run] | temporal task success 57%；19.4161±15.0048；n=100 [M-latency-eval] |
| qwengr00t | 已完成SFT 5,000；[M-qwengr00t-source] | 21.2928±16.6083；60.00步；task success 56%；n=50 [M-qwengr00t-eval] | 已发布；均值 99.35ms [M-qwengr00t-profile] | 已发布；strict评测见下方小模型表 | 未找到正式产物 | 未找到正式评测 |
| qwenpi_v3 | 已完成SFT 5,000；[M-qwenpi_v3-source] | 21.9671±17.2095；60.00步；task success 54%；n=50 [M-qwenpi_v3-eval] | 已发布；均值 148.69ms [M-qwenpi_v3-profile] | 已发布；strict评测见下方小模型表 | 未找到正式产物 | 未找到正式评测 |
| wanoft | 已完成SFT 5,000；[M-wanoft-source] | 18.9461±16.4028；60.00步；task success 42%；n=50 [M-wanoft-eval] | 已发布；均值 300.54ms [M-wanoft-profile] | 已发布；strict评测见下方小模型表；10Hz | 未找到正式产物 | 未找到正式评测 |

## Extreme Parkour Go2：四模型明细

控制/观测频率 **50/10 Hz**。zero small 使用上方公共 teacher。

L0 数值来自本次固定中央模型revision的 `eval_l0.json`，各100 episodes；
JSON未记录评测时间，episode seed记录保留在该文件。训练seed见provenance。

| 模型 | 0 VLA training | 0 VLA eval：return ± SD；长度；成功率；n | Latency profile | Latency small | Latency VLA training | Latency VLA eval |
|---|---|---|---|---|---|---|
| qwenoft | 已完成SFT 5,000 + DAgger 5,000；[P-qwenoft-source] | 23.3283±7.7106；927.49步；progress 0.9075；SR未记录；n=100 [P-qwenoft-eval] | 已发布；均值 103.43ms [P-qwenoft-profile] | 历史训练/暂停记录；最终状态未核验 | 未找到正式产物 | 未找到正式评测 |
| qwengr00t | 已完成SFT 5,000 + DAgger 5,000；[P-qwengr00t-source] | 22.3789±6.7888；887.38步；progress 0.9163；SR未记录；n=100 [P-qwengr00t-eval] | 已发布；均值 157.36ms [P-qwengr00t-profile] | 未找到该profile对应正式最终产物 | 未找到正式产物 | 未找到正式评测 |
| qwenpi_v3 | 已完成SFT 5,000 + DAgger 5,000；[P-qwenpi_v3-source] | 21.8744±7.9515；938.61步；progress 0.8500；SR未记录；n=100 [P-qwenpi_v3-eval] | 已发布；均值 238.73ms [P-qwenpi_v3-profile] | 未找到该profile对应正式最终产物 | 未找到正式产物 | 未找到正式评测 |
| wanoft | 未找到正式产物；已有两步smoke | 正式评测未找到 | 未找到正式RTX3090 profile | 未找到该profile对应正式最终产物 | 未找到正式产物 | 未找到正式评测 |

## HAIC PullCart：四模型明细

控制/观测频率 **50/50 Hz**。zero small 使用上方公共 teacher。

L0 数值来自本次固定中央模型revision的 `eval_l0.json`，各100 episodes；
JSON未记录评测时间，episode seed记录保留在该文件。训练seed见provenance。

| 模型 | 0 VLA training | 0 VLA eval：return ± SD；长度；成功率；n | Latency profile | Latency small | Latency VLA training | Latency VLA eval |
|---|---|---|---|---|---|---|
| qwenoft | 已完成SFT 5,000 + DAgger 5,000；[H-qwenoft-source] | 13.3497±8.2155；187.09步；task success 56%；n=100 [H-qwenoft-eval] | 已发布；均值 126.29ms [H-qwenoft-profile] | 历史训练/暂停记录；最终状态未核验 | 未找到正式产物 | 未找到正式评测 |
| qwengr00t | 未找到正式产物；已有两步smoke | 正式评测未找到 | 未找到正式RTX3090 profile | 未找到该profile对应正式最终产物 | 未找到正式产物 | 未找到正式评测 |
| qwenpi_v3 | 未找到正式产物；已有两步smoke | 正式评测未找到 | 未找到正式RTX3090 profile | 未找到该profile对应正式最终产物 | 未找到正式产物 | 未找到正式评测 |
| wanoft | 未找到正式产物；已有两步smoke | 正式评测未找到 | 未找到正式RTX3090 profile | 未找到该profile对应正式最终产物 | 未找到正式产物 | 未找到正式评测 |

## 逐局task指标与MIKASA远端100局结果

下表由固定中央版本 `eval_l0.json` 的 `episodes[].metadata.task_metrics` 聚合；
各组n=100。Progress/score取episode等权平均，±为episode总体SD；success按源0/1指标计数。
MIKASA远端100局与主表本地50局补评测分别保留，不能混用其成功率或return。

| Task / model | task指标 | mean return ± SD | mean length | 证据 |
|---|---|---:|---:|---|
| extreme-parkour-go2 / qwenoft | waypoint progress=0.9075±0.1901 | 23.3283±7.7106 | 927.49步 | [原始eval](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/extreme-parkour-go2/vla/starvla-qwenoft-h1/20260912T072315-nocache32v2/eval_l0.json) |
| extreme-parkour-go2 / qwengr00t | waypoint progress=0.9163±0.1871 | 22.3789±6.7888 | 887.38步 | [原始eval](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/extreme-parkour-go2/vla/starvla-qwengr00t-h1/20260912T180303/eval_l0.json) |
| extreme-parkour-go2 / qwenpi_v3 | waypoint progress=0.8500±0.2417 | 21.8744±7.9515 | 938.61步 | [原始eval](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/extreme-parkour-go2/vla/starvla-qwenpi_v3-h1/20260912T175510/eval_l0.json) |
| haic-pull-cart / qwenoft | task success=56%（56/100）；cart_progress=0.3563；pullcart_score=58.0530 | 13.3497±8.2155 | 187.09步 | [原始eval](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/haic-pull-cart/vla/starvla-qwenoft-h1/haic-qwenoft-50hz-fresh-20260913T035850Z/eval_l0.json) |
| mikasa-intercept-grab-fast / qwenpi_v3 | task success=51%（51.0/100） | 20.1415±15.8556 | 60.00步 | [原始eval](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenpi_v3-h1/mikasa_intercept_grab_fast_qwenpi_v3_formal_latest_r8gpu8_20260907T082154Z/eval_l0.json) |
| mikasa-intercept-grab-fast / wanoft | task success=31%（31.0/100） | 13.6323±14.2860 | 60.00步 | [原始eval](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/mikasa-intercept-grab-fast/vla/starvla-wanoft-h1/mikasa_intercept_grab_fast_wanoft_formal_latest_r8gpu7_20260907T075003Z/eval_l0.json) |

## MIKASA latency-aware 小模型：独立严格评测

四类原生PPO均从scratch训练，单worker、temporal profile、deterministic mean actions；
environment seed `20260910`、latency seed `271828`，各256 episodes。
acceptance 为既有 `success_at_end >= 90%`。JSON未记录episode return SD。
WanOFT最新bundle使用10Hz，其他三类使用20Hz；下表分别报告，不作为同频率公平比较。

| Profile | checkpoint训练步数 / W&B | success_at_end | success_once | mean return | mean length | admission/drop | 状态 / 证据 |
|---|---|---|---|---:|---:|---|---|
| qwenoft（20Hz） | 13,824,000；[ks6d837t](https://wandb.ai/saberrr-zju/MIKASA-Robo-dataset-collectors/runs/ks6d837t) | 98.83%（253/256） | 99.61%（255/256） | 33.6446 | 60步 | 7606/7754 | 已发布/达标；[MS-qwenoft] [MS-qwenoft-source] |
| qwengr00t（20Hz） | 20,951,040；[2papyzq7](https://wandb.ai/saberrr-zju/MIKASA-Robo-dataset-collectors/runs/2papyzq7) | 98.05%（251/256） | 98.05%（251/256） | 31.7504 | 60步 | 7582/7778 | 已发布/达标；[MS-qwengr00t] [MS-qwengr00t-source] |
| qwenpi_v3（20Hz） | 15,360,000；[u46f6pzq](https://wandb.ai/saberrr-zju/MIKASA-Robo-dataset-collectors/runs/u46f6pzq) | 96.88%（248/256） | 98.83%（253/256） | 28.6463 | 60步 | 5099/10261 | 已发布/达标；[MS-qwenpi_v3] [MS-qwenpi_v3-source] |
| wanoft（10Hz） | 23,040,000；[s463nqus](https://wandb.ai/saberrr-zju/MIKASA-Robo-dataset-collectors/runs/s463nqus) | 93.75%（240/256） | 98.44%（252/256） | 30.8664 | 60步 | 5119/10241 | 已发布/达标；[MS-wanoft] [MS-wanoft-source] |

严格评测配置为各bundle的 `eval.yaml`，训练配置为 `train.yaml`，checkpoint为 `checkpoint.pt`；具体路径：

- qwenoft：[训练配置][MS-qwenoft-train]、[评测配置][MS-qwenoft-config]、[训练使用的固定profile][MS-qwenoft-profile]。
- qwengr00t：[训练配置][MS-qwengr00t-train]、[评测配置][MS-qwengr00t-config]、[训练使用的固定profile][MS-qwengr00t-profile]。
- qwenpi_v3：[训练配置][MS-qwenpi_v3-train]、[评测配置][MS-qwenpi_v3-config]、[训练使用的固定profile][MS-qwenpi_v3-profile]。
- wanoft：[训练配置][MS-wanoft-train]、[评测配置][MS-wanoft-config]、[训练使用的固定profile][MS-wanoft-profile]。

WanOFT旧20Hz run `sw3b981e`：69,120,000步，end success 122/256（47.66%），
mean return 22.5840。该历史配置与clock10当前发布版本不同；旧记录见
[原本地评测](../outputs/latency_rl/mikasa/wanoft_rtx3090_recipe_25m/20260910T093825Z/strict_eval_best_150m/evaluation.json)。

## MIKASA QwenOFT VLA：100 episode复核

2026-09-11，20/20Hz，H1持续hold上一动作，env seeds 4242424242–4242424341。
旧baseline下载revision `c72713d3eeba1f33d08c9c909d7caa868b62d717`；新模型是本地SFT5k，
`no_publish=true`。具体比较和输出目录见 [M-latency-eval]，训练身份见 [M-latency-run]。
`success` 为源比较文件记录的 task success；各组平均长度60步，n=100。

| 模型 | Eval latency | success_rate | mean return ± SD | 结果 |
|---|---|---:|---:|---|
| Baseline QwenOFT | zero | 57% | 22.0986±15.8734 | L0复核完成 |
| Baseline QwenOFT | temporal | 0% | 0.4274±0.1224 | 延迟退化 |
| Latency-aware QwenOFT | zero | 0% | 0.5868±0.1957 | zero下退化，不能用temporal结果覆盖 |
| Latency-aware QwenOFT | temporal | 57% | 19.4161±15.0048 | 本地复核完成，temporal恢复；未发布 |

Temporal各组applied/dropped/unfinished为2857/10/100。未把它们解释为success计数。
Latency seed、完整逐局数据与执行配置以该组输出中的原始文件为准；汇总没有的字段不补造。

## 已发布 RTX 3090 profile 数值

12组profile，均记录session IDs 0–4，使用temporal observation-to-action latency。
单位ms，五场是否能用于某一后续实验仍按具体模型、频率及运行合同判断。
模型来源由各profile的 `provenance.model_artifact` 记录；该来源未记resolved revision时明确保留缺失，
不能把本次模型仓库revision当成其测量时revision。

| Task / model | 控制频率 / physical instance | mean ms | p95 ms | p99 ms | samples；admission/drop | source run / 证据 |
|---|---|---:|---:|---:|---|---|
| Balance Simple / qwenoft | 50Hz；`instance_3a0d42681a03715c` | 37.881 | 39.449 | 41.419 | 10,840；10,840/10,869 | `20260909T050028237618Z`；[B-qwenoft-profile] |
| Balance Simple / qwengr00t | 50Hz；`instance_3a0d42681a03715c` | 60.013 | 63.154 | 64.689 | 5,577；5,577/11,834 | `20260909T042707692652Z`；[B-qwengr00t-profile] |
| Balance Simple / qwenpi_v3 | 50Hz；`instance_3a0d42681a03715c` | 99.176 | 103.613 | 107.123 | 2,847；2,847/11,326 | `20260909T035342113762Z`；[B-qwenpi_v3-profile] |
| Balance Simple / wanoft | 50Hz；`instance_3a0d42681a03715c` | 300.024 | 307.783 | 310.533 | 1,134；1,134/14,432 | `20260909T031304517915Z`；[B-wanoft-profile] |
| MIKASA InterceptGrabFast / qwenoft | 20Hz；`instance_3a0d42681a03715c` | 99.050 | 100.327 | 100.644 | 7,505；7,505/7,495 | `20260909T044501695676Z`；[M-qwenoft-profile] |
| MIKASA InterceptGrabFast / qwengr00t | 20Hz；`instance_3a0d42681a03715c` | 99.352 | 100.351 | 100.632 | 7,500；7,500/7,500 | `20260909T041122796788Z`；[M-qwengr00t-profile] |
| MIKASA InterceptGrabFast / qwenpi_v3 | 20Hz；`instance_3a0d42681a03715c` | 148.689 | 150.353 | 150.654 | 5,000；5,000/10,000 | `20260909T033716462687Z`；[M-qwenpi_v3-profile] |
| MIKASA InterceptGrabFast / wanoft | 20Hz；`instance_3a0d42681a03715c` | 300.539 | 300.201 | 349.782 | 2,498；2,498/12,502 | `20260909T025113875647Z`；[M-wanoft-profile] |
| Extreme Parkour Go2 / qwenoft | 50Hz；`instance_6e332477bffea7da` | 103.426 | 146.078 | 150.313 | 52,094；52,094/0 | `20260913T140346673977Z`；[P-qwenoft-profile] |
| Extreme Parkour Go2 / qwengr00t | 50Hz；`instance_6e332477bffea7da` | 157.357 | 163.036 | 166.160 | 52,952；52,952/0 | `20260914T033146019182Z`；[P-qwengr00t-profile] |
| Extreme Parkour Go2 / qwenpi_v3 | 50Hz；`instance_b19d7af6f1bdd95d` | 238.729 | 246.982 | 250.157 | 53,087；53,087/0 | `20260913T140435541240Z`；[P-qwenpi_v3-profile] |
| HAIC PullCart / qwenoft | 50Hz；`instance_b19d7af6f1bdd95d` | 126.288 | 130.240 | 132.731 | 9,791；9,791/0 | `20260914T033618205895Z`；[H-qwenoft-profile] |

早期MIKASA QwenOFT另有 `1x-rpbzzz6` profile：mean 99.563ms，p95 168.197ms、
p99 219.746ms，16,473 samples，admission/drop 16,473/13,527；与RTX3090分开保留。
证据见 [PROFILE](PROFILE.md) 及
[固定profile](https://huggingface.co/datasets/latency-sensitive-bench/profiles/blob/e8be1e5dfd309f85ab7ff704dc02bd4898b7ceb4/profiles/qwenoft/1x-rpbzzz6/mikasa_intercept_grab_fast/instance_e906b84a82f9e086/profile.json)。

## H1 训练产物身份

下列run-id与中央模型snapshot共同标识主表产物。训练配置、teacher/dataset/backbone版本、
training seed与代码状态以各 `provenance.json` 为准；配置为同bundle `config.yaml`。
MIKASA两类checkpoint-only来源未完整记录training seed，不从其他模型补填。

| Task / model | run-id | checkpoint / 发布完整性 | provenance |
|---|---|---|---|
| Balance Simple / qwenoft | `humanoidbench_balance_simple_qwenoft_formal_latest` | `checkpoints/model.pt`；inference/eval已发布；质量gate未据此推断 | [B-qwenoft-source] |
| Balance Simple / qwengr00t | `humanoidbench_balance_simple_qwengr00t_formal_latest` | `checkpoints/model.pt`；inference/eval已发布；质量gate未据此推断 | [B-qwengr00t-source] |
| Balance Simple / qwenpi_v3 | `humanoidbench_balance_simple_qwenpi_v3_formal_latest` | `checkpoints/model.pt`；inference/eval已发布；质量gate未据此推断 | [B-qwenpi_v3-source] |
| Balance Simple / wanoft | `humanoidbench_balance_simple_wanoft_formal_latest` | `checkpoints/model.pt`；inference/eval已发布；质量gate未据此推断 | [B-wanoft-source] |
| MIKASA InterceptGrabFast / qwenoft | `mikasa_intercept_grab_fast_qwenoft_formal_latest_r8gpu8_20260907T082154Z` | `checkpoints/model.pt`；checkpoint-only；HF eval_l0缺失 | [M-qwenoft-source] |
| MIKASA InterceptGrabFast / qwengr00t | `mikasa_intercept_grab_fast_qwengr00t_formal_latest_r8gpu7_20260907T075003Z` | `checkpoints/model.pt`；checkpoint-only；HF eval_l0缺失 | [M-qwengr00t-source] |
| MIKASA InterceptGrabFast / qwenpi_v3 | `mikasa_intercept_grab_fast_qwenpi_v3_formal_latest_r8gpu8_20260907T082154Z` | `checkpoints/model.pt`；inference/eval已发布 | [M-qwenpi_v3-source] |
| MIKASA InterceptGrabFast / wanoft | `mikasa_intercept_grab_fast_wanoft_formal_latest_r8gpu7_20260907T075003Z` | `checkpoints/model.pt`；inference/eval已发布 | [M-wanoft-source] |
| Extreme Parkour Go2 / qwenoft | `20260912T072315-nocache32v2` | `checkpoints/model.pt`；inference/eval已发布；质量gate未据此推断 | [P-qwenoft-source] |
| Extreme Parkour Go2 / qwengr00t | `20260912T180303` | `checkpoints/model.pt`；inference/eval已发布；质量gate未据此推断 | [P-qwengr00t-source] |
| Extreme Parkour Go2 / qwenpi_v3 | `20260912T175510` | `checkpoints/model.pt`；inference/eval已发布；质量gate未据此推断 | [P-qwenpi_v3-source] |
| HAIC PullCart / qwenoft | `haic-qwenoft-50hz-fresh-20260913T035850Z` | `checkpoints/model.pt`；inference/eval已发布；质量gate未据此推断 | [H-qwenoft-source] |

## 历史 official GR00T、Balance实验与smoke

这些结果采用不同checkpoint、输入、频率或协议，分别保留；不替换上方中央H1基线。

| 路线 | training / evaluation协议 | 已记录数值 | 状态 / 证据 |
|---|---|---|---|
| Balance official GR00T | fresh10k；checkpoint-10000；10个seed；zero | return353.5317±133.5709；length371.0 | 已完成/历史已发布，未达目标return500；[固定eval](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/humanoidbench-balance-simple/vla/gr00t-n1.7-10k-v1/zero-latency-eval.json)；fresh15k未核验 |
| MIKASA早期QwenOFT-v1 | SFT5k；canonical50episode zero | success27/50（54%）；return22.326 | 已完成/历史发布；[runbook](MIKASA.md)；不是主表formal_latest checkpoint |
| Balance修正标签DAgger | 500updates；100episode zero；seed0–99 | return68.0205±3.0410；length79.73；前50为68.3031±3.1028 | 已完成/未发布/低于原始模型；[短交接](BALANCE_DAGGER.md) |
| Balance十轮DAgger scratch/checkpoint | 各5000updates；共同50episode final；2026-09-11记录 | return144.532006/377.179249；length157.30/397.54；SD未记录 | 训练和逐轮eval文档记载完成；终点确认/全流程验收待核验；[待办](BALANCE_DAGGER.md#ten-round-dagger-latest-recorded-result-and-pending-verification) |
| Balance fixed1 F2 | 6M raw frames；固定1帧延迟 | mean survival525.44帧 | fixed1已恢复；[历史诊断](archive/LATENCY_DEBUG_PLAN.md)；不能外推到fixed2/profile |
| Balance H8 profile修复前/后 | 5,000,064raw steps；25env seeds×4latency seeds | length72.88→172.14；return59.45→150.37；≥100帧3%→93% | 修复配对完成，SD以原始表为准；[CHUNK](CHUNK.md) |
| Balance H8 fixed40 | 50Hz；6M；25场景；保留/移除自定义限幅 | length83.40/244.80；≥100帧20%/100% | 已完成历史对照；与当前D/E生产版分开；[CHUNK](CHUNK.md) |
| Balance H8第一版消融full | 4training seeds；各10M；每seed100评测场景 | length304.73±232.32；mean return269.956；≥100比例0.742 | 20run全部训练/final eval完成；±为training-seed间样本SD；[消融](CHUNK_RL_ABLATION.md) |
| Balance H8第一版删除ABCHJ | 同上四seed协议 | length417.24±88.05；mean return378.106；≥100比例1.000 | 通过既有相对full筛选；D/E-only组合尚未直接验证；[消融](CHUNK_RL_ABLATION.md) |
| Balance旧H1 temporal small | 25,000,192步；2026-09-07log已退出 | log FPS4188.2；该run最终strict eval未找到 | 训练完成，评测/质量未核验；[log](../outputs/latency_rl/balance/h1/train.log) |
| Parkour旧latency small | 文档记录model_600、iteration610附近collapse/暂停 | 最终同种子zero/profile数值未核验 | 历史消息不代表当前running；[中文交接](LATENCY_RL_HANDOFF.zh-CN.md) |
| HAIC旧latency small | h1 log最后记录911/15258iterations；acceptance曾32+16iterations | 正式最终eval未找到；不以finite/resume检查代替成功率 | 最终运行状态未核验；[log](../outputs/latency_rl/haic/h1/train.log)、[中文交接](LATENCY_RL_HANDOFF.zh-CN.md) |
| 四任务×四模型acceptance-v1 | 各2updates、保存/重载和短L0 | 16组smoke完成 | pipeline功能验证，非正式策略质量；[SCALE_MODELS](SCALE_MODELS.md#smoke-acceptance--acceptance-v1) |
| Balance旧GR00T latency smoke | checkpoint-2；zero/profile各2episode | new return7.5175±0.0136 / 11.0231±0.4348；length20.5/23.5 | smoke；[zero](../outputs/latency_rl/balance/final_revalidated/new_zero/queue_eval_latency_0.json)、[profile](../outputs/latency_rl/balance/final_revalidated/new_profile/queue_eval_latency_profile_sample.json) |
| Parkour旧GR00T latency smoke | checkpoint-2；zero/profile各1episode | new return3.6618 / 2.4672；length1500/494 | 旧local-backbone wiring；[zero](../outputs/latency_rl/parkour/real_chain_remote/evals/new_zero/queue_eval_latency_0.json)、[profile](../outputs/latency_rl/parkour/real_chain_remote/evals/new_iid/queue_eval_latency_profile_sample.json) |
| HAIC旧GR00T latency smoke | checkpoint-2；zero/profile各1episode | new return2.4688 / 2.0471；length45/37；task success0 | acceptance3 wiring；[zero](../outputs/latency_rl/haic/acceptance3/evals/eval_new_zero/queue_eval_latency_0.json)、[profile](../outputs/latency_rl/haic/acceptance3/evals/eval_new_profile/queue_eval_latency_profile_sample.json) |

## 待推进与更新约定

- Balance：既有D/E-only生产组合、单独D/E及zero/fixed对照未验证；十轮DAgger终点确认和完整回传待核验。官方GR00T fresh15k与StarVLA SFT15k分别记录。
- MIKASA：补齐qwenoft/qwengr00t远端canonical eval文件；latency-aware QwenOFT保留zero退化与temporal恢复，发布及其余模型进展待核验。Wan10Hz和旧20Hz分开维护。
- Parkour：Wan H1正式产物、latency small终点状态及正式latency VLA缺口待核验。
- HAIC：其余三类H1模型及profile、latency small终点状态及正式latency VLA缺口待核验。50Hz H1与旧20Hz latent/H40合同分别记录。

后续训练、评测、profile、发布、暂停或失败确认后，同步更新受影响的总览、明细、数值、
checkpoint/配置/版本、核查日期和下一步。保留可比较的旧结果；更新日志只写实际确认的变化。

| 核查日期（UTC） | 更新 | 证据范围 |
|---|---|---|
| 2026-09-14 | 初始化四任务总览、16模型明细、12组RTX3090 profile、4组MIKASA严格小模型eval与QwenOFT VLA复核；迁移历史实验待办 | 上方固定HF版本、本地补评测/复核JSON及注明为文档记载的历史结果 |

## 证据链接

[B-teacher]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/humanoidbench-balance-simple/small-policy/fasttd3-v1/wandb-summary.json
[M-teacher]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/mikasa-intercept-grab-fast/small-policy/ppo-v1/strict-eval.json
[B-qwenoft-source]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/humanoidbench-balance-simple/vla/starvla-qwenoft-h1/humanoidbench_balance_simple_qwenoft_formal_latest/provenance.json
[B-qwenoft-eval]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/humanoidbench-balance-simple/vla/starvla-qwenoft-h1/humanoidbench_balance_simple_qwenoft_formal_latest/eval_l0.json
[B-qwenoft-profile]: https://huggingface.co/datasets/latency-sensitive-bench/profiles/blob/e8be1e5dfd309f85ab7ff704dc02bd4898b7ceb4/profiles/qwenoft/1x-rtx3090/g1_sonic_balance_simple/instance_3a0d42681a03715c/profile.json
[B-qwengr00t-source]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/humanoidbench-balance-simple/vla/starvla-qwengr00t-h1/humanoidbench_balance_simple_qwengr00t_formal_latest/provenance.json
[B-qwengr00t-eval]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/humanoidbench-balance-simple/vla/starvla-qwengr00t-h1/humanoidbench_balance_simple_qwengr00t_formal_latest/eval_l0.json
[B-qwengr00t-profile]: https://huggingface.co/datasets/latency-sensitive-bench/profiles/blob/e8be1e5dfd309f85ab7ff704dc02bd4898b7ceb4/profiles/qwengr00t/1x-rtx3090/g1_sonic_balance_simple/instance_3a0d42681a03715c/profile.json
[B-qwenpi_v3-source]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/humanoidbench-balance-simple/vla/starvla-qwenpi_v3-h1/humanoidbench_balance_simple_qwenpi_v3_formal_latest/provenance.json
[B-qwenpi_v3-eval]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/humanoidbench-balance-simple/vla/starvla-qwenpi_v3-h1/humanoidbench_balance_simple_qwenpi_v3_formal_latest/eval_l0.json
[B-qwenpi_v3-profile]: https://huggingface.co/datasets/latency-sensitive-bench/profiles/blob/e8be1e5dfd309f85ab7ff704dc02bd4898b7ceb4/profiles/qwenpi_v3/1x-rtx3090/g1_sonic_balance_simple/instance_3a0d42681a03715c/profile.json
[B-wanoft-source]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/humanoidbench-balance-simple/vla/starvla-wanoft-h1/humanoidbench_balance_simple_wanoft_formal_latest/provenance.json
[B-wanoft-eval]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/humanoidbench-balance-simple/vla/starvla-wanoft-h1/humanoidbench_balance_simple_wanoft_formal_latest/eval_l0.json
[B-wanoft-profile]: https://huggingface.co/datasets/latency-sensitive-bench/profiles/blob/e8be1e5dfd309f85ab7ff704dc02bd4898b7ceb4/profiles/wanoft/1x-rtx3090/g1_sonic_balance_simple/instance_3a0d42681a03715c/profile.json
[M-qwenoft-source]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenoft-h1/mikasa_intercept_grab_fast_qwenoft_formal_latest_r8gpu8_20260907T082154Z/provenance.json
[M-qwenoft-eval]: ../outputs/starvla/eval_backfill/results/mikasa/qwenoft/job.json
[M-qwenoft-profile]: https://huggingface.co/datasets/latency-sensitive-bench/profiles/blob/e8be1e5dfd309f85ab7ff704dc02bd4898b7ceb4/profiles/qwenoft/1x-rtx3090/mikasa_intercept_grab_fast/instance_3a0d42681a03715c/profile.json
[M-latency-run]: ../outputs/latency_rl/mikasa/qwenoft_rtx3090_h1_v1/vla/run.json
[M-latency-eval]: ../outputs/latency_rl/mikasa/qwenoft_rtx3090_h1_v1/direct_eval_revalidation_20260911/comparison.json
[M-qwengr00t-source]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwengr00t-h1/mikasa_intercept_grab_fast_qwengr00t_formal_latest_r8gpu7_20260907T075003Z/provenance.json
[M-qwengr00t-eval]: ../outputs/starvla/eval_backfill/results/mikasa/qwengr00t/job.json
[M-qwengr00t-profile]: https://huggingface.co/datasets/latency-sensitive-bench/profiles/blob/e8be1e5dfd309f85ab7ff704dc02bd4898b7ceb4/profiles/qwengr00t/1x-rtx3090/mikasa_intercept_grab_fast/instance_3a0d42681a03715c/profile.json
[M-qwenpi_v3-source]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenpi_v3-h1/mikasa_intercept_grab_fast_qwenpi_v3_formal_latest_r8gpu8_20260907T082154Z/provenance.json
[M-qwenpi_v3-eval]: ../outputs/starvla/eval_backfill/results/mikasa/qwenpi_v3/job.json
[M-qwenpi_v3-profile]: https://huggingface.co/datasets/latency-sensitive-bench/profiles/blob/e8be1e5dfd309f85ab7ff704dc02bd4898b7ceb4/profiles/qwenpi_v3/1x-rtx3090/mikasa_intercept_grab_fast/instance_3a0d42681a03715c/profile.json
[M-wanoft-source]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/mikasa-intercept-grab-fast/vla/starvla-wanoft-h1/mikasa_intercept_grab_fast_wanoft_formal_latest_r8gpu7_20260907T075003Z/provenance.json
[M-wanoft-eval]: ../outputs/starvla/eval_backfill/results/mikasa/wanoft/job.json
[M-wanoft-profile]: https://huggingface.co/datasets/latency-sensitive-bench/profiles/blob/e8be1e5dfd309f85ab7ff704dc02bd4898b7ceb4/profiles/wanoft/1x-rtx3090/mikasa_intercept_grab_fast/instance_3a0d42681a03715c/profile.json
[P-qwenoft-source]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/extreme-parkour-go2/vla/starvla-qwenoft-h1/20260912T072315-nocache32v2/provenance.json
[P-qwenoft-eval]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/extreme-parkour-go2/vla/starvla-qwenoft-h1/20260912T072315-nocache32v2/eval_l0.json
[P-qwenoft-profile]: https://huggingface.co/datasets/latency-sensitive-bench/profiles/blob/e8be1e5dfd309f85ab7ff704dc02bd4898b7ceb4/profiles/qwenoft/1x-rtx3090/extreme_parkour_go2/instance_6e332477bffea7da/profile.json
[P-qwengr00t-source]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/extreme-parkour-go2/vla/starvla-qwengr00t-h1/20260912T180303/provenance.json
[P-qwengr00t-eval]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/extreme-parkour-go2/vla/starvla-qwengr00t-h1/20260912T180303/eval_l0.json
[P-qwengr00t-profile]: https://huggingface.co/datasets/latency-sensitive-bench/profiles/blob/e8be1e5dfd309f85ab7ff704dc02bd4898b7ceb4/profiles/qwengr00t/1x-rtx3090/extreme_parkour_go2/instance_6e332477bffea7da/profile.json
[P-qwenpi_v3-source]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/extreme-parkour-go2/vla/starvla-qwenpi_v3-h1/20260912T175510/provenance.json
[P-qwenpi_v3-eval]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/extreme-parkour-go2/vla/starvla-qwenpi_v3-h1/20260912T175510/eval_l0.json
[P-qwenpi_v3-profile]: https://huggingface.co/datasets/latency-sensitive-bench/profiles/blob/e8be1e5dfd309f85ab7ff704dc02bd4898b7ceb4/profiles/qwenpi_v3/1x-rtx3090/extreme_parkour_go2/instance_b19d7af6f1bdd95d/profile.json
[H-qwenoft-source]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/haic-pull-cart/vla/starvla-qwenoft-h1/haic-qwenoft-50hz-fresh-20260913T035850Z/provenance.json
[H-qwenoft-eval]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/zero-latency/haic-pull-cart/vla/starvla-qwenoft-h1/haic-qwenoft-50hz-fresh-20260913T035850Z/eval_l0.json
[H-qwenoft-profile]: https://huggingface.co/datasets/latency-sensitive-bench/profiles/blob/e8be1e5dfd309f85ab7ff704dc02bd4898b7ceb4/profiles/qwenoft/1x-rtx3090/haic_pull_cart/instance_b19d7af6f1bdd95d/profile.json
[MS-qwenoft]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwenoft-rtx3090-v1/strict-eval.json
[MS-qwenoft-source]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwenoft-rtx3090-v1/provenance.json
[MS-qwenoft-train]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwenoft-rtx3090-v1/train.yaml
[MS-qwenoft-config]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwenoft-rtx3090-v1/eval.yaml
[MS-qwenoft-profile]: https://huggingface.co/datasets/latency-sensitive-bench/profiles/blob/66414463f2a609316da63935941a79abf0a73ce9/profiles/qwenoft/1x-rtx3090/mikasa_intercept_grab_fast/instance_3a0d42681a03715c/profile.json
[MS-qwengr00t]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwengr00t-rtx3090-v1/strict-eval.json
[MS-qwengr00t-source]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwengr00t-rtx3090-v1/provenance.json
[MS-qwengr00t-train]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwengr00t-rtx3090-v1/train.yaml
[MS-qwengr00t-config]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwengr00t-rtx3090-v1/eval.yaml
[MS-qwengr00t-profile]: https://huggingface.co/datasets/latency-sensitive-bench/profiles/blob/66414463f2a609316da63935941a79abf0a73ce9/profiles/qwengr00t/1x-rtx3090/mikasa_intercept_grab_fast/instance_3a0d42681a03715c/profile.json
[MS-qwenpi_v3]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwenpi_v3-rtx3090-v1/strict-eval.json
[MS-qwenpi_v3-source]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwenpi_v3-rtx3090-v1/provenance.json
[MS-qwenpi_v3-train]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwenpi_v3-rtx3090-v1/train.yaml
[MS-qwenpi_v3-config]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwenpi_v3-rtx3090-v1/eval.yaml
[MS-qwenpi_v3-profile]: https://huggingface.co/datasets/latency-sensitive-bench/profiles/blob/66414463f2a609316da63935941a79abf0a73ce9/profiles/qwenpi_v3/1x-rtx3090/mikasa_intercept_grab_fast/instance_3a0d42681a03715c/profile.json
[MS-wanoft]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-wanoft-rtx3090-v1/strict-eval.json
[MS-wanoft-source]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-wanoft-rtx3090-v1/provenance.json
[MS-wanoft-train]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-wanoft-rtx3090-v1/train.yaml
[MS-wanoft-config]: https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/e05ca26afde4f509b0f838742c522f3a068596a8/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-wanoft-rtx3090-v1/eval.yaml
[MS-wanoft-profile]: https://huggingface.co/datasets/latency-sensitive-bench/profiles/blob/66414463f2a609316da63935941a79abf0a73ce9/profiles/wanoft/1x-rtx3090/mikasa_intercept_grab_fast/instance_3a0d42681a03715c/profile.json
