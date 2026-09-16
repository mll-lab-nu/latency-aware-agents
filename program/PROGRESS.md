# Benchmark 进度与评测结果

最后核查：**2026-09-15 08:58 UTC**。文档入口：[program 索引](README.md)。
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
| MIKASA InterceptGrabFast | PPO teacher；100% | 4/4；各 SFT 5k | 4/4 本地补评测；QwenOFT success56%，return21.7417±16.4361，n=50；两类HF标准文件缺失 | 4/4 | 4/4 已发布；end SR 96.88–98.83%（20Hz），Wan 93.75%（10Hz） | QwenOFT SFT 5k 本地完成；远端QwenPI SFT 5k完成，12线程、micro128＋CPU offload；未发布 | QwenOFT temporal success 57%，return 19.4161±15.0048；QwenPI三项100局矩阵已完成：no_latency_model_no_latency return23.0983, n=100；no_latency_model_profile_latency return0.2862, n=100；latency_aware_model_profile_latency return14.9605, n=100 |
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
| qwengr00t | 已完成SFT 5,000；[M-qwengr00t-source] | 21.2928±16.6083；60.00步；task success 56%；n=50 [M-qwengr00t-eval] | 已发布；均值 99.35ms [M-qwengr00t-profile] | 已发布；strict评测见下方小模型表 | 远端SFT5,000运行中；12线程、micro128＋CPU offload；未发布 | 正式评测待训练完成 |
| qwenpi_v3 | 已完成SFT 5,000；[M-qwenpi_v3-source] | 21.9671±17.2095；60.00步；task success 54%；n=50 [M-qwenpi_v3-eval] | 已发布；均值 148.69ms [M-qwenpi_v3-profile] | 已发布；strict评测见下方小模型表 | 远端SFT5,000完成；12线程、micro128＋CPU offload；未发布 | 三项100局矩阵完成：no_latency_model_no_latency return23.0983, n=100；no_latency_model_profile_latency return0.2862, n=100；latency_aware_model_profile_latency return14.9605, n=100；详见下方证据 |
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

## MIKASA QwenPI：2026-09-14 远端训练与批量测试

服务器 `xinyuan@81.85.2.28`，独立目录 `/home/xinyuan/mikasa-qwenpi-latency`，
单张 RTX PRO 6000 Blackwell Server Edition（nvidia-smi 总显存97,887MiB）。
训练代码已包含 performance 提交 `978359fc65e34915c5971d815a700de729bf823f`；
StarVLA revision `7a4ee2014ddd389404b5c1d7d8053144619f00ea`。
旧实验/缓存全部保留；本地两卡训练未启动。

本轮额外重训了一个 QwenPI-profile small：PPO seed123，1024env，60-step rollout，
lr1e-4，8epochs；25M预算实际24,944,640步，选中15,360,000步的 `run/small/best_policy.pt`。
256局独立评测（20Hz，env seed20260910，latency seed271828）end success247/256（96.484375%），
once254/256（99.21875%），return29.688919，length60，admission/drop5099/10261，达标。
该新增checkpoint未发布HF；上方四个已发布small的状态不变。
证据：[独立评测](../outputs/tasks/mikasa_intercept_grab_fast/qwenpi_latency_20260914T122034Z/small_acceptance.json)、
[small W&B](https://wandb.ai/saberrr-zju/starvla_mikasa/runs/mikasa-qwenpi-small-20260914T122034Z)。
使用固定profile revision `66414463f2a609316da63935941a79abf0a73ce9`，
`profiles/qwenpi_v3/1x-rtx3090/mikasa_intercept_grab_fast/instance_3a0d42681a03715c`。

采集275成功episodes，train250/val25，admitted issued-action样本3,691/385；
H1，state/action7，两路RGB224×224输入。QwenPI从Qwen3-VL-4B-Instruct backbone开始全参数训练，
5,072.110M trainable，bf16，ZeRO2，Qwen3 gradient checkpointing启用（non-reentrant，use_cache=false）。
初始正式配置单GPU micro8、累积16、global128，CPU optimizer offload启用，计划5,000更新；
base/action lr3e-5/1e-4，warmup100，cosine min1e-6，seed0。
2026-09-14 14:12 UTC在第100更新保存 `run/vla/training/bootstrap/checkpoints/steps_100_state`
及 `steps_100_pytorch_model.pt`；14:13 UTC进程退出，显存归零，正式训练暂停。
正式训练/正式评测/发布均未完成；两步CPU-offload smoke训练及2局rollout已完成。
[正式 VLA W&B](https://wandb.ai/zihanwang-ai-northwestern-university/starvla_tasks/runs/mikasa-qwenpi_v3-mikasa_qwenpi_latency_3090_20260914T122034Z-bootstrap)。

已依次测试micro16无offload、micro32有offload、micro32无offload，累积8/4/4，global128。
用户追加后测试micro128有offload（累积1）和micro8无offload（累积16），global仍为128。
均保留上述模型、数据、checkpointing，启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，
显存caching保持启用。各组12更新，排除前2更新预热；计时包含整次梯度累积，记录Torch峰值显存及NVML采样。
首次计时入口缺少multiprocessing main入口隔离，造成数据worker重复启动训练，已修正并通过CPU子进程导入检查；
该无效尝试不作为OOM或速度结果。

| Microbatch / optimizer offload | 完整更新数 / 结果 | 稳态秒/更新 | 稳态样本/秒 | Torch allocated/reserved峰值（GiB） | NVML used峰值（GiB） | W&B |
|---|---|---:|---:|---:|---:|---|
| 8 / 无 | 1；第2次优化器更新FP32梯度转换OOM，需16.88GiB，剩余9.27GiB | 未形成稳态 | 未形成稳态 | 85.29 / 87.65 | 89.24 | [8无offload](https://wandb.ai/zihanwang-ai-northwestern-university/starvla_tasks/runs/mikasa-qwenpi-bench-20260914-v2-mb8_no_offload) |
| 16 / 无 | 1；第2次优化器更新FP32梯度转换OOM，需16.88GiB，剩余8.95GiB | 未形成稳态 | 未形成稳态 | 85.68 / 88.39 | 89.97 | [16无offload](https://wandb.ai/zihanwang-ai-northwestern-university/starvla_tasks/runs/mikasa-qwenpi-bench-20260914-v2-mb16_no_offload) |
| 32 / CPU | 12；通过，排除前2更新，统计10更新 | 8.419945 | 15.202000 | 28.36 / 31.81 | 33.57 | [32有offload](https://wandb.ai/zihanwang-ai-northwestern-university/starvla_tasks/runs/mikasa-qwenpi-bench-20260914-v2-mb32_offload) |
| 32 / 无 | 1；第2次更新action-head前向attention key投影OOM，需38MiB，剩余27.56MiB | 未形成稳态 | 未形成稳态 | 93.10 / 93.36 | 94.94 | [32无offload](https://wandb.ai/zihanwang-ai-northwestern-university/starvla_tasks/runs/mikasa-qwenpi-bench-20260914-v2-mb32_no_offload) |
| 128 / CPU | 12；通过，排除前2更新，统计10更新 | 7.582636 | 16.880672 | 76.93 / 77.04 | 78.77 | [128有offload](https://wandb.ai/zihanwang-ai-northwestern-university/starvla_tasks/runs/mikasa-qwenpi-bench-20260914-v2-mb128_offload) |

128＋offload相对32＋offload吞吐提高11.04%，每更新耗时降低9.94%。
此前正式micro8＋offload的稳态tqdm约13.7秒/完整更新，累积16、global128；
这是历史正式训练的近似参考，非本轮12-update计时。
128＋offload相对该参考吞吐约提高80%；micro8无offload只完成1更新，不能拿含冷启动的26.37秒当稳态速度。
micro8参考证据：[原正式配置](../outputs/tasks/mikasa_intercept_grab_fast/qwenpi_latency_20260914T122034Z/vla/bootstrap_micro8_cpu_offload.yaml)、
[原正式日志](../outputs/tasks/mikasa_intercept_grab_fast/qwenpi_latency_20260914T122034Z/vla/bootstrap_micro8_cpu_offload.log)。

原始日志/配置/逐更新统计已回传至 [批量测试目录](../outputs/tasks/mikasa_intercept_grab_fast/qwenpi_latency_20260914T122034Z/batch_benchmark/)，
[聚合结果](../outputs/tasks/mikasa_intercept_grab_fast/qwenpi_latency_20260914T122034Z/batch_benchmark/results.json)；
各W&B run的summary也已记录最终结果、offload、allocator、峰值显存和完整更新吞吐。
14:27 UTC启动正式恢复，micro32、累积4、global128，CPU optimizer offload；
第100步完整state及LR scheduler恢复成功，实际推进到110更新。
按用户补充，14:32 UTC在110停止更新；14:37 UTC已保存完整 `steps_110_state` 和
`steps_110_pytorch_model.pt`，进程退出且显存归零。
micro128＋offload与micro8无offload实测均已完成；micro64暂不运行。
[110步暂停/恢复证据](../outputs/tasks/mikasa_intercept_grab_fast/qwenpi_latency_20260914T122034Z/vla/bootstrap_micro32_cpu_offload_until110.log)。
14:47 UTC重新启动正式恢复，micro128、累积1、global128、CPU optimizer offload，
保留bf16、checkpointing、expandable_segments及显存caching，恢复第110步完整state/scheduler。
使用 [恢复脚本](../outputs/tasks/mikasa_intercept_grab_fast/qwenpi_latency_20260914T122034Z/resume_after_benchmark_128.sh)，
tmux session `mikasa-qwenpi-latency`。14:53 UTC已确认完整state恢复成功并实际推进至122/5,000更新，
DeepSpeed运行配置micro128、累积1、global128；第112–118更新约7.5–7.9秒，显存80,663MiB，GPU利用率100%。
[当前正式配置](../outputs/tasks/mikasa_intercept_grab_fast/qwenpi_latency_20260914T122034Z/vla/bootstrap.yaml)、
[恢复后正式日志](../outputs/tasks/mikasa_intercept_grab_fast/qwenpi_latency_20260914T122034Z/vla/bootstrap.log)已回传。
按7.58秒/更新估算完整5k纯训练约10小时32分；checkpoint保存及最终评测另计，预计总计11–12小时。
正式训练仍在运行，下一步完成5k、L0/temporal各100局评测。

按用户追加的CPUAdam线程优化测试，14:57 UTC在156/5,000更新请求安全停止，
15:02 UTC已保存完整 `steps_156_state`、LR scheduler及模型checkpoint，worker退出、显存归零。
服务器12个物理核心、24个逻辑线程、单NUMA节点；各组16完整更新，排除前2更新，统计14稳态更新，
保持micro128、累积1、global128及模型/数据/checkpointing/allocator一致。
通过CPUAdam原生扩展的 `omp_get_max_threads()` 核对实际线程数；CPUAdam计时累计全部参数组。

| 实际线程数 | 完整更新秒/步 | CPUAdam秒/步 | 相对8线程吞吐 | W&B |
|---|---:|---:|---:|---|
| 8 | 7.579100 | 0.901108 | 基准 | [8线程](https://wandb.ai/zihanwang-ai-northwestern-university/starvla_tasks/runs/mikasa-qwenpi-cputhreads-20260914-t8-v1) |
| 12 | 7.475353 | 0.768198 | +1.38785% | [12线程](https://wandb.ai/zihanwang-ai-northwestern-university/starvla_tasks/runs/mikasa-qwenpi-cputhreads-20260914-t12-v1) |
| 16 | 7.578428 | 0.906631 | +0.00887% | [16线程](https://wandb.ai/zihanwang-ai-northwestern-university/starvla_tasks/runs/mikasa-qwenpi-cputhreads-20260914-t16-v2) |

12线程比8线程每步省0.103747秒，5k纯训练约省8.65分钟。各组Torch峰值allocated/reserved均76.93/77.04GiB。
[线程测试聚合](../outputs/tasks/mikasa_intercept_grab_fast/qwenpi_latency_20260914T122034Z/thread_benchmark/results.json)、
[日志/配置/逐步记录](../outputs/tasks/mikasa_intercept_grab_fast/qwenpi_latency_20260914T122034Z/thread_benchmark/)已回传，W&B summary已更新。
首次16线程尝试环境变量为16、PyTorch及CPUAdam实际12，已停止并保留 `t16_invalid_default12`，不计入比较；
重测通过 `torch.set_num_threads()` 显式设置，原生CPUAdam实际16已确认。32线程比较按用户要求取消，不参与配置选择。
15:26 UTC按用户选择启动12线程正式恢复，micro128、累积1、global128、CPU offload、bf16、checkpointing、
expandable_segments及显存caching保持启用。使用 [12线程恢复脚本](../outputs/tasks/mikasa_intercept_grab_fast/qwenpi_latency_20260914T122034Z/resume_threads12.sh)，
继续原正式W&B run `mikasa-qwenpi_v3-mikasa_qwenpi_latency_3090_20260914T122034Z-bootstrap`；已确认从156步完整state恢复并实际推进至182/5,000更新，最近7.4–7.5秒/更新；
原W&B显示Resuming run，worker环境确认OMP_NUM_THREADS=12、WANDB_RESUME=allow及原WANDB_RUN_ID。

训练完成后追加以下三项100局评测：

1. 已发布no-latency QwenPI，在no latency下。
2. 同一个no-latency QwenPI，在固定profile latency下。
3. 本轮latency-aware QwenPI的5,000更新checkpoint，在同一profile latency下。

通用训练入口内置的latency-aware L0评测保留；上方三项为本次追加对照矩阵。
基线固定HF模型revision `e05ca26afde4f509b0f838742c522f3a068596a8`，
`zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenpi_v3-h1/mikasa_intercept_grab_fast_qwenpi_v3_formal_latest_r8gpu8_20260907T082154Z`；
config、dataset_statistics、task_contract、provenance已在远端核验，使用基线自己的归一化统计。
三项均20/20Hz、H1、100局，环境reset seeds4242424242–4242424341、单env/单模拟worker、inference batch1；
两个temporal项共用上述固定RTX3090 profile revision及latency seed271828。
[评测脚本](../outputs/tasks/mikasa_intercept_grab_fast/qwenpi_latency_20260914T122034Z/post_eval_matrix.py)已接入恢复流水线，
结果写入远端 `run/eval_matrix` 和同一正式W&B run的 `eval_matrix/<model_latency_condition>/...` summary指标。
本地tmux `mikasa-qwenpi-eval-progress`运行 [回传/进度更新脚本](../outputs/tasks/mikasa_intercept_grab_fast/qwenpi_latency_20260914T122034Z/sync_eval_progress.py)，
等待远端流水线退出后回传原始结果/配置/日志，并将已确认的结果更新至本地本文件。
正式5k训练及三项100局矩阵评测已完成，原始结果见下方。

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
- MIKASA：远端QwenPI用12线程、micro128＋offload从156步继续至5k，完成no-latency模型zero/profile及latency-aware模型profile三项各100局评测；补齐qwenoft/qwengr00t远端canonical eval文件。latency-aware QwenOFT保留zero退化与temporal恢复，发布及其他模型进展待核验；Wan10Hz和旧20Hz分开维护。
- Parkour：Wan H1正式产物、latency small终点状态及正式latency VLA缺口待核验。
- HAIC：其余三类H1模型及profile、latency small终点状态及正式latency VLA缺口待核验。50Hz H1与旧20Hz latent/H40合同分别记录。

后续训练、评测、profile、发布、暂停或失败确认后，同步更新受影响的总览、明细、数值、
checkpoint/配置/版本、核查日期和下一步。保留可比较的旧结果；更新日志只写实际确认的变化。

| 核查日期（UTC） | 更新 | 证据范围 |
|---|---|---|
| 2026-09-14 | 初始化四任务总览、16模型明细、12组RTX3090 profile、4组MIKASA严格小模型eval与QwenOFT VLA复核；迁移历史实验待办 | 上方固定HF版本、本地补评测/复核JSON及注明为文档记载的历史结果 |
| 2026-09-14 15:33 | 12线程正式恢复已核查170/5k，最近7.4–7.5秒/更新；原W&B已续上。远端后置三项100局评测与本地tmux结果回传均已接入 | 正式worker环境、当前日志、评测/回传脚本 |
| 2026-09-14 15:30 | CPUAdam线程8/12/16实测完成，12线程7.475秒/更新、吞吐+1.39%；选择12从156完整state恢复，继续原W&B。三项100局矩阵评测及自动回传/PROGRESS更新已安排 | 线程聚合、恢复/评测/回传脚本、原正式W&B run |
| 2026-09-14 14:53 | MIKASA QwenPI已确认从110完整state恢复，micro128＋CPU offload、累积1、global128，实际122/5k，最近7.5–7.9秒/更新；正式评测/发布未完成 | 当前正式配置、恢复后正式日志、同一正式W&B run |
| 2026-09-14 14:48 | MIKASA额外small达标、未发布；275成功episodes；QwenPI完整state保存于110步。5组批量测试：8/16/32无offload均OOM，32/128有offload通过；选择128恢复正式训练 | 本轮任务目录、逐更新统计/峰值显存、原正式日志与各W&B summary |

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

### 2026-09-15 07:29 UTC — Balance QwenPI H16 诊断

`balance-qwenpi_v3-h16-replay6553600-25m-20260915T061441Z-profile-rl`
在 4,976,384 raw frames 被人工中断，保留
`checkpoint_000077618_4976384.pth`；配置为 H16、50Hz observation、QwenPI v3
RTX 3090 temporal profile、capacity 1、replay 6,553,600、64 raw frames/update。
固定 8 reset seeds × 4 latency seeds 的正式评测 n=32：return
27.892326±2.722401，mean survival 35.46875 raw frames，≥100/≥500 比例均为 0。
[训练 W&B](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/balance-qwenpi_v3-h16-replay6553600-25m-20260915T061441Z-profile-rl)；
[评测 W&B](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/balance-qwenpi_v3-h16-replay6553600-25m-20260915T061441Z-profile-rl-diagnostic-eval)；
[诊断目录](../outputs/tasks/humanoidbench_balance_simple/h16_causal_diagnostics/20260915T071310Z)。
该 profile 训练未完成，未验收、未发布。后续两个旧语义因果 arm 已于 08:42 UTC 完成，
均为 H16、zero latency、obs_fps=10、capacity=1、约 5M raw frames、n=8：
`update64` checkpoint `checkpoint_000078040_5000972.pth` 的 return
39.335382±4.280865、mean survival 55.75；`update160` checkpoint
`checkpoint_000031217_5001189.pth` 的 return 60.552981±6.150873、mean survival 78.375；
两者 ≥100/≥500 比例均为 0。它们按 raw frames 给 optimizer 累积 credit，而 chunk replay
只存完成的 execution segment，因此只作为修复前历史结果保留，不能用于新的等 optimizer-step 比较。

08:55 UTC 已启动修复后的 QwenPI H16 zero10 诊断：update credit 改为新完成的 replay
transitions，raw frames 只作环境吞吐计数；训练停止、TensorBoard/W&B `global_step` 和周期评测横轴
统一为 critic optimizer step，目标精确为 500,000，`64 transitions/update`，即 warmup 后
32,000,000 个 credited replay transitions。run 已验证到 optimizer step 458 / raw frames 137,157，
replay 29,981，首个 checkpoint 为 `checkpoint_000000102_35697.pth`。
[训练 W&B](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/balance-qwenpi_v3-h16-zero10-replay6553600-500kopt-20260915T084909Z-rl)；
[配置与启动记录](../outputs/tasks/humanoidbench_balance_simple/h16_optimizer500k/20260915T084909Z/launch.json)。
源码基线 revision `a81af1e4cec246d4b8401a41f19d39db18dbd0f1`、Sample Factory
`d74dd462919c1583601079a06d0d9055eeef501e`，实际补丁哈希保存在 launch 记录中。
当前仅确认训练正常 running；评测、验收和发布均未完成。每 25,000 optimizer steps 保留周期
checkpoint，训练结束后串行评测；约 75k 的 checkpoint 用于和旧 77,618-update 结果比较。若恢复
学习，再运行其余机制消融及新的 QwenOFT H8 zero/profile 500k optimizer-step baseline。

W&B history 复核确认，`episode/return` 是 Sample Factory 对最近 `stats_avg=100` 个 episode
求均值后写出的训练曲线，并与 optimizer `global_step` 同行记录；它可直接用于当前 run 的学习趋势。
逐局原始值为 `episode_train/latest_return`，其原生横轴仍是
`episode_train/completed_episodes`。后续 run 已显式将 `episode/*` 的 step metric 声明为
`global_step`；当前 run 无需停止或重训。

### 2026-09-15T10:00:06.202003+00:00 — Balance H8 VLA training checkpoint saved

latency-aware bootstrap, optimizer step 5000, formal; H8, seed=0, microbatch=64, accumulation=1, global batch=128, GPUs=2. [Configuration](/pfs/users/lxy/latency-sensitive-bench/outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/vla/sft5k_only/profile/bootstrap.yaml), [log](/pfs/users/lxy/latency-sensitive-bench/outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/vla/sft5k_only/profile/bootstrap.log), [checkpoint](/pfs/users/lxy/latency-sensitive-bench/outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/vla/sft5k_only/profile/training/bootstrap/checkpoints/steps_5000_pytorch_model.pt), [W&B](https://wandb.ai/zihanwang-ai-northwestern-university/starvla_tasks/runs/balance-qwenoft-balance-qwenoft-h8-25m-sft5k-20260915T040718Z-profile-bootstrap). Training checkpoint saved; next: evaluation. Acceptance/publication remain separate.

Code revision `6ab79709c2ac8b40609a878ebc36c1cd8cd43e1a` (dirty workspace; run directory retains configuration and diff).

### 2026-09-15T10:02:20.109074+00:00 — Balance H8 evaluation complete

Checkpoint `/pfs/users/lxy/latency-sensitive-bench/outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/vla/sft5k_only/profile/inference/checkpoints/model.pt`, step 5000; H8, 50 Hz, `temporal`, formal, n=32. Return 106.463683 ± 21.284356; mean length 121.281250; fraction ≥100 0.875000, ≥500 0.000000. [Exact seed protocol and metrics](/pfs/users/lxy/latency-sensitive-bench/outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/vla/sft5k_only/profile/evals/final/summary.json); [W&B](https://wandb.ai/zihanwang-ai-northwestern-university/starvla_tasks/runs/balance-qwenoft-balance-qwenoft-h8-25m-sft5k-20260915T040718Z-profile-bootstrap-eval-5000). Evaluation completion does not establish acceptance or publication. Next: continue the scheduled training/publication stage.

Code revision `6ab79709c2ac8b40609a878ebc36c1cd8cd43e1a` (dirty workspace; run directory retains configuration and diff).

### 2026-09-15T10:04:23.614225+00:00 — Balance H8 evaluation complete

Checkpoint `/pfs/users/lxy/latency-sensitive-bench/outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/vla/sft5k_only/profile/inference/checkpoints/model.pt`, step 5000; H8, 50 Hz, `zero`, formal, n=25. Return 121.023290 ± 28.059735; mean length 137.760000; fraction ≥100 0.920000, ≥500 0.000000. [Exact seed protocol and metrics](/pfs/users/lxy/latency-sensitive-bench/outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/vla/sft5k_only/profile/evals/final_matrix/zero/summary.json); [W&B](https://wandb.ai/zihanwang-ai-northwestern-university/starvla_tasks/runs/balance-qwenoft-h8-25m-sft5k-20260915T040718Z-profile-final-zero). Evaluation completion does not establish acceptance or publication. Next: continue the scheduled training/publication stage.

Code revision `6ab79709c2ac8b40609a878ebc36c1cd8cd43e1a` (dirty workspace; run directory retains configuration and diff).

### 2026-09-15T10:10:17.891953+00:00 — Balance H8 evaluation complete

Checkpoint `/pfs/users/lxy/latency-sensitive-bench/outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/vla/sft5k_only/profile/inference/checkpoints/model.pt`, step 5000; H8, 50 Hz, `temporal`, formal, n=100. Return 102.433854 ± 15.147160; mean length 117.900000; fraction ≥100 0.880000, ≥500 0.000000. [Exact seed protocol and metrics](/pfs/users/lxy/latency-sensitive-bench/outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/vla/sft5k_only/profile/evals/final_matrix/temporal/summary.json); [W&B](https://wandb.ai/zihanwang-ai-northwestern-university/starvla_tasks/runs/balance-qwenoft-h8-25m-sft5k-20260915T040718Z-profile-final-temporal). Evaluation completion does not establish acceptance or publication. Next: continue the scheduled training/publication stage.

Code revision `6ab79709c2ac8b40609a878ebc36c1cd8cd43e1a` (dirty workspace; run directory retains configuration and diff).

### 2026-09-15T10:11:39.415694+00:00 — Balance H8 publication complete

profile QwenOFT H8 pure-SFT model, optimizer step 5000, DAgger updates 0; published to `latency-sensitive-bench/benchmark-models/latency-aware/humanoidbench-balance-simple/vla/starvla-qwenoft-h8/balance-qwenoft-h8-25m-sft5k-20260915T040718Z-profile` at revision `4ada4d52f711167a3bb40829b8617e4caafbac2f`. [Publication record](/pfs/users/lxy/latency-sensitive-bench/outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/vla/sft5k_only/profile/publication.json). Training and final matrix evaluation completed before publication; acceptance remains separate.

### 2026-09-15T12:11:30+00:00 — MIKASA QwenGr00T resumed after disk-full interruption

The single-GPU RTX PRO 6000 latency-aware SFT reached optimizer step 1500 before the root filesystem filled while writing the separate model file. The DeepSpeed `steps_1500_state` had already completed; superseded QwenGr00T step-500 and step-1000 state/model pairs were removed, leaving 137 GiB free. Training resumed from the complete step-1500 state at 12:08 UTC and was verified at step 1512/5000, with recent updates taking 7.40–7.49 seconds and about 90.3 GiB GPU memory in use. The original W&B run is `running`: [MIKASA QwenGr00T SFT](https://wandb.ai/zihanwang-ai-northwestern-university/starvla_tasks/runs/mikasa-qwengr00t-mikasa_qwengr00t_latency_3090_20260915T035000Z-bootstrap).

The resumed run keeps micro/global batch 128, accumulation 1, CPU optimizer offload, 12 CPUAdam threads, bf16, gradient checkpointing, CUDA expandable segments, and GPU caching. Intermediate checkpoint writes are disabled until step 5000 for this recovery, and the source setting is restored automatically when the process exits. At the observed rate, pure training has about 7 hours 13 minutes remaining; final checkpoint export and the queued three-condition 100-episode evaluation matrix add roughly 25–35 minutes plus save/load time. [Recovery script and task record](../outputs/tasks/mikasa_intercept_grab_fast/qwengr00t_latency_20260915T035000Z/).

### 2026-09-15 14:27 UTC — Balance FastTD3 correctness and throughput verification

The final H16 FastTD3 path was verified on the idle private NVIDIA RTX PRO 6000 host with
seed 1, replay capacity 6,553,600, batch 32,768, 64 credited replay transitions per
optimizer update, and 2,000 optimizer updates. The zero-latency protocol used 50 Hz env,
10 Hz observations, 64 workers × 2 envs × 2 splits. Three final runs reached exactly step
2,000 with median 13.662786 optimizer updates/s, 4,064.565 raw frames/s, 181.89 s wall
time, and replay/raw 0.214302. The old environment plus unordered replay baseline was
11.993094 updates/s and 202.19 s, so the final correct path improves optimizer throughput
13.92%, reduces wall time 10.04%, and changes the 500k-step projection from 11.58 h to
10.17 h. [Aggregate results and exact run links](../outputs/tasks/humanoidbench_balance_simple/final_layout_benchmark/summary.json).

The replay failure was traced to startup decorrelation: with 64 × 2 × 2, the second split
advanced each environment before its first trace, consuming frames 0–4 while ordered
replay waited for frame 0. Disabling that FastTD3 decorrelation restored the zero run from
1,203,707 raw frames and replay/raw 0.106338 to the final median 0.214302. Replay now
buffers cross-batch events per environment and drains them in episode/frame order. Latency
and reset streams are keyed by global env id; all 128 historical 128 × 1 seeds are
preserved. Accelerator batching now counts worker splits: min request target 42 versus 21
improved the paired three-run median from 13.222920 to 13.662786 updates/s (+3.33%).

The topology check used the same target 42 and seed mapping. The 128 workers × 1 env × 1
split zero run reached step 2,000 at 13.211374 updates/s and 202.39 s; 64 × 2 × 2 remained
3.42% faster by stable optimizer throughput and reduced wall time 10.13%. The QwenPI v3
RTX 3090 temporal-profile run used 50 Hz observations and 64 × 2 × 2; it reached step
2,000 at 6.743590 updates/s, 2,468.717 raw frames/s, 748,416 raw frames, and replay/raw
0.171028 (about 1/5.85), with no missing-frame wait or expanding backlog. Exact configs,
logs, exit codes, source snapshots, and per-run JSON are under the linked evidence directory.

The optimized G1 wrapper was also checked against the prior implementation for zero,
fixed 40 ms, and temporal latency over 50 nonconstant H16 decisions each; state, reward,
terminal flags, replay trace, and scheduler metadata matched exactly. This entry records
correctness and performance only. No policy-quality evaluation, acceptance, or publication
was performed. The local source baselines were repository revision
`558af6c2573ecb00041a4413cb1b1065084f4570` and Sample Factory revision
`f6050eeefa3739968c494b68060cbd50c09017f9`, both dirty. The benchmark host was a file
sync rather than a git checkout, so its exact hot-path sources are retained in the evidence
directory. Next: use 64 × 2 × 2 for the next long zero/profile learning run, then run the
documented seeded evaluation protocol separately.

### 2026-09-15 15:20 UTC — Balance QwenPI H16 target-actor / stale-state treatments running

The zero10 `w128e1s1` control completed 100,000 critic optimizer steps at 31,400,509
raw frames with final checkpoint `checkpoint_000100000_31400509.pth`; no new evaluation,
acceptance, or publication has been performed. Its target-actor treatment is running on GPU 0
with the same seed 1, H16, zero latency at 10 Hz observations, replay 6,553,600, batch
32,768, 64 replay transitions/update, and 100,000-step budget. The only algorithm flag enabled
for this arm is `fasttd3_target_actor`; it was verified running through optimizer step 2,088 at
628,846 raw frames. [Target-actor W&B](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/balance-qwenpi_v3-h16-zero10-w128e1s1-target_actor-replay6553600-100kopt-20260915T151542Z-s1-rl).

The slow profile control was intentionally stopped at optimizer step 48,346 / 17,344,000 raw
frames (`checkpoint_000048346_17344000.pth`, exit 2). Its stale-state treatment is running on
GPU 1 with the same seed 1, H16, QwenPI v3 RTX 3090 temporal profile, 50 Hz observations,
capacity 1, replay/update settings, and 100,000-step budget. This arm leaves target actor off and
enables observation-boundary result release plus the causal scheduled chunk remainder/head index;
it was verified through optimizer step 283 at 116,992 raw frames. [Chunk-context W&B](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/balance-qwenpi_v3-h16-profile-w128e1s1-chunk_context-replay6553600-100kopt-20260915T151542Z-s1-rl).
[Exact configs, isolated runtime hashes, scripts, and logs](../outputs/tasks/humanoidbench_balance_simple/h16_target_actor_chunk_context_100k/20260915T151542Z/launch.json).
The isolated runtime preserves the controls' worker/vector seed mapping; evaluation, acceptance,
and publication remain pending. Next: compare training return and periodic n=8 evaluation at equal
25k/50k/75k/100k optimizer steps.

### 2026-09-15 18:03 UTC — Balance H16 treatments rejected; G1 no-op and periodic eval corrected

The target-actor treatment completed 100,000 optimizer steps / 31,302,628 raw frames at
`checkpoint_000100000_31302628.pth`. Deterministic equal-checkpoint evaluation used 8 episodes
per policy. At 25k/50k/75k/100k, the zero-control mean lengths were
239.750/294.625/714.250/875.000; the treatment's deployed live actor was
76.125/231.500/90.500/871.750, and its stored EMA actor was
100.375/188.875/93.250/834.875. Thus the EMA bootstrap slowed learning, did not smooth deployed
behavior, and gave no final gain. The standard evaluator/deployment loads checkpoint `model`, not
the separate experimental `target_actor`; target/live relative parameter L2 was only about
0.48%-0.57% with tau 0.1. This treatment is rejected, not accepted or published.

The chunk-context treatment was intentionally terminated after its exact 50k checkpoint was
loaded and verified (`checkpoint_000050000_17866880.pth`, 60,044,671 bytes, 48 model/critic
tensors all finite). Shutdown occurred at 2026-09-15 17:36:01 UTC; the later automatic
`checkpoint_000053067_18937600.pth` is retained, but comparisons remain anchored at 50k. At 25k
under the corrected no-op runtime, its n=4 mean length was 35.500 versus 36.500 for the profile
control, so the added scheduled-chunk remainder/head input did not improve the blocking behavior.
The target-actor, chunk-context, observation-boundary release, config flags, checkpoint state, and
their tests were deleted rather than retained as inactive compatibility surface.

The deeper fault was at the G1 action boundary: the latency scheduler received the configured
normalized no-op after the native environment had already switched to `direct_absolute_position`,
while ordinary policy actions were converted by the outer raw-request wrapper. Wrong-versus-raw
maximum observation errors over the first six frames were
18.21/15.11/13.68/13.09/11.60/9.66; converting the no-op through the same G1 transform restores
raw/direct trajectories to about 1e-6. The old profile 25k checkpoint scored mean length 50.375
with the wrong training no-op but only 32.500 in corrected-runtime n=8 evaluation, so it cannot
stand in for a retrained corrected-profile arm. A second bug caused no-suite periodic jobs to
write `{"summaries": []}` without running episodes; they now evaluate the current condition and
the terminal periodic step is not repeated as a final eval.

Implementation revisions are root `3586443134acc6f68739f76f575e7b4aac0f1871` and Sample Factory
`636e92744a2f47e0f9f2c34c795acee13022e6fc`. Targeted tests passed 75 with 6 skips; the broader
related selection passed 175 tests, with 13 unrelated Flappy imports unavailable in this
environment. Isolated 2k smokes both exited 0: corrected profile at
`checkpoint_000002000_759680.pth` and fixed100/capacity1 at
`checkpoint_000002000_741632.pth`; both had finite model/critic tensors and no target-actor state.
[Diagnosis and episode metrics](../outputs/tasks/humanoidbench_balance_simple/h16_profile_fixed100_corrected_noop_50k/20260915T174801Z/diagnosis.json) and
[paired configs, hashes, smoke records, and decision gates](../outputs/tasks/humanoidbench_balance_simple/h16_profile_fixed100_corrected_noop_50k/20260915T174801Z/launch.json).
Training completion and evaluation completion for the formal pair remain false. The owner-confirmed
private host is one RTX PRO 6000 Blackwell, so the two arms use the same GPU sequentially. The
corrected-profile arm launched at 2026-09-15 18:17:27 UTC in tmux
`balance_noop_pair_174801`; it crossed warmup and a checkpoint snapshot was verified at optimizer
step 389 / 165,504 raw frames with finite model/critic tensors and no target-actor state.
Fixed100/capacity1 is queued behind its successful training and deferred 25k/50k evaluations.
[Profile W&B](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/balance-qwenpi_v3-h16-profile-corrected_noop-w128e1s1-replay6553600-50kopt-20260915T174801Z-s1-rl) and
[fixed100 W&B](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/balance-qwenpi_v3-h16-fixed100-cap1-corrected_noop-w128e1s1-replay6553600-50kopt-20260915T174801Z-s1-rl).
Acceptance and publication remain false. Next: verify warmup/optimizer checkpoints, let both arms
reach 50k, then apply the documented mean-length gates; resume both to 100k only if either result is
in the 100-300 grey zone.

### 2026-09-15T18:15:27.313160+00:00 — Balance H8 DAgger failure

zero pipeline exited with code 1. [Pipeline log](/pfs/users/lxy/latency-sensitive-bench/outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/vla/dagger10_20260915T180849Z/zero.log), [status](/pfs/users/lxy/latency-sensitive-bench/outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/vla/dagger10_20260915T180849Z/status.json). Completed rounds and checkpoints are retained; next: diagnose and resume this exact run.

### 2026-09-15T18:18:38.519283+00:00 — Balance H8 evaluation complete

Checkpoint `/pfs/users/lxy/latency-sensitive-bench/outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/vla/sft5k_only/zero/inference/checkpoints/model.pt`, step 0; H8, 50 Hz, `zero`, formal, n=8. Return 71.686991 ± 16.825506; mean length 87.625000; fraction ≥100 0.250000, ≥500 0.000000. [Exact seed protocol and metrics](/pfs/users/lxy/latency-sensitive-bench/outputs/tasks/humanoidbench_balance_simple/local_h8/closed_loop_h8_replay6553600/vla/dagger10_20260915T180849Z/zero/evals/sft_0/summary.json); [W&B](https://wandb.ai/zihanwang-ai-northwestern-university/starvla_tasks/runs/balance-qwenoft-balance-qwenoft-h8-25m-sft5k-dagger10-20260915T180849Z-zero-dagger). Evaluation completion does not establish acceptance or publication. Next: continue the scheduled training/publication stage.

Code revision `6ab79709c2ac8b40609a878ebc36c1cd8cd43e1a` (dirty workspace; run directory retains configuration and diff).

### 2026-09-16 11:10 UTC — Native MC PPO implemented; smoke and resume verified, formal budget not yet started

Public `training.train` now dispatches `backend.algo: PPO` to complete-episode,
synchronous Monte Carlo PPO; existing FAST_TD3/APPO paths remain. Root revision
`f0929454423a5f170eb3c226aec42103ea351248` plus the uncommitted PPO implementation;
SF revision `636e92744a2f47e0f9f2c34c795acee13022e6fc` unchanged. Initial weights are
`h16_fixed100_delayed_imitation/20260916T061955Z/followup_20260916T070429Z/student_correction_2000.pth`.
Native H16/fixed100/capacity1/50-50Hz/heads5:10, normalized reset-pose noop,
frozen BC normalization, gamma1, survival-frame MC targets /1000, independent
128/128 V with remaining time; no Q gradient or BC-relative coordinate tether.
Gymnasium CPU vector workers feed centralized GPU inference at fixed batch shape.

The proposed actor LR1e-5 overshot the joint Gaussian KL target0.02 in one step:
11.39 in smoke,15.15 on a new128-episode calibration batch. LR1e-6 gave0.151;
LR3e-7 gave0.024 after3 actor steps, so the formal recipe uses3e-7. Other planned
budgets are unchanged:128 complete episodes/round, minibatch1024,2 epochs,
200 rounds; V LR3e-4 and128-episode/10-epoch warmup, fixed pre-tanh std0.02,
PPO clip0.2, joint KL early-stop for actor only. Calibration is not policy
acceptance. [Calibration](../outputs/tasks/humanoidbench_balance_simple/ppo_fixed100/lr_calibration.json).

Serial BC audit exactly reproduced464.1875 length /421.429847 return /30 of32
>=100 /13 of32 >=500 on seeds104730–104761, cap1000. Batched FP32 inference
changes closed-loop trajectories, so each run re-evaluates its own BC baseline;
the old scalar baseline is not substituted into batched comparisons.
[Serial audit](../outputs/tasks/humanoidbench_balance_simple/ppo_fixed100/bc_serial_audit.json).

Final implementation smoke used4 workers,8 episodes/round,2 rounds, then resumed
to round3 on the same online W&B run. Latest checkpoint is
`balance_h16_fixed100_mcppo_smoke_final_20260916T1107Z/checkpoint_p0/checkpoint_latest.pth`:
6 actor /16 V optimizer calls,14,370 training+warmup raw frames,32 episodes.
Round3 development32:487.34375 length /442.997474 return /32 of32 >=100;
its matched BC was491.0 /445.265735 /30 of32. Selected best remains round0 BC;
the8 smoke test seeds650001–650008 give paired gain0, interval[0,0], not accepted.
First/final smoke KL0.018626/0.009496. Training and evaluation completion here
refer only to smoke, not the formal200-round budget. No publication.
[Smoke result](../outputs/tasks/humanoidbench_balance_simple/ppo_fixed100/balance_h16_fixed100_mcppo_smoke_final_20260916T1107Z/result.json),
[online W&B](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/lvsuko7t).

Verification:128 native wrapper/scheduler/executor/PPO/checkpoint/W&B tests and
58 FAST_TD3 tests pass. GPU tests require explicit `CUDA_VISIBLE_DEVICES`;
the first invocation omitted it, then all58 passed with it set. Broader Flappy
tests are unavailable because `flappy_bird_gymnasium` is not installed.
[128-test log](../outputs/tasks/humanoidbench_balance_simple/ppo_fixed100/tests_final.log),
[58-test log](../outputs/tasks/humanoidbench_balance_simple/ppo_fixed100/fasttd3_tests_final.log).
Occam review retained one dedicated episode trainer and existing installed
vector-env/actor implementations; no new dependency or parallel framework.
Next: finish the public eval smoke, benchmark16/32/64 workers on128 BC episodes
three times, then complete seed1's200 rounds and64 heldout episodes per policy.
Development seeds104730–104761 select best every10 rounds; heldout seeds640001–640064
are not used for selection. [Recipe](../configs/examples/gymnasium/g1_sonic_balance_simple/ppo_train.yaml).

### 2026-09-16 11:15 UTC — Formal Balance MC PPO seed1 running, round11/200 verified

Online run `balance_h16_fixed100_mcppo_s1_20260916T1110Z`, PID1334277, GPU0,
uses the recipe and revisions above. No performance-based stop was added.
Three128-episode BC timing repeats per setting gave median raw frames/s:
16 workers3316.75,32 workers5998.17,64 workers9315.48. The smallest setting
within95% of peak is64 workers (2.81x the16-worker median); GPU inference batch
is fixed64, CPU simulation remains the dominant collection cost. The cgroup
CPU quota is43 cores. The profiling episodes are separate from the training
budget. [Timing evidence](../outputs/tasks/humanoidbench_balance_simple/ppo_fixed100/balance_h16_fixed100_mcppo_s1_20260916T1110Z/throughput.json).

Verified immutable `checkpoint_p0/checkpoint_0011.pth`: round11,131 actor updates,
402 V updates including130 warmup updates,727,092 raw training+warmup frames,
1,536 complete episodes including128 warmup episodes. Round11 collection128
episodes/63,943 frames took7.139s; learner0.048s, KL0.02309,
value explained variance0.63191. Round10 development32 seeds104730–104761:
469.875 mean length /427.424475 mean return /30 of32 >=100 /15 of32 >=500.
Matched fixed-batch64 frozen BC is525.1875 /477.932544 /30 of32 /19 of32.
This interim policy is not an improvement; selected best remains BC at round0.
The200-round run continues, and no heldout640001–640064 evaluations have run yet.
Formal training completion, final evaluation completion, acceptance and
publication are all false at this audit.
[Round11](../outputs/tasks/humanoidbench_balance_simple/ppo_fixed100/balance_h16_fixed100_mcppo_s1_20260916T1110Z/round_0011.json),
[round10 evaluation](../outputs/tasks/humanoidbench_balance_simple/ppo_fixed100/balance_h16_fixed100_mcppo_s1_20260916T1110Z/round_0010.json),
[online W&B](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/z3y85xez),
[launch log](../outputs/tasks/humanoidbench_balance_simple/ppo_fixed100/formal_s1_20260916T1110Z.log).

Public `--mode eval --checkpoint-kind best` also completed on the smoke run,
loading round0 and reproducing paired gain0 on its8 test seeds without changing
the training run's files or W&B history. [Eval result](../outputs/tasks/humanoidbench_balance_simple/ppo_fixed100/balance_h16_fixed100_mcppo_smoke_final_20260916T1107Z/eval_best_1789556969818619221/result.json).
Resume this formal run, only if its process has stopped, with:
`CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m training.train --config configs/examples/gymnasium/g1_sonic_balance_simple/ppo_train.yaml --run-name balance_h16_fixed100_mcppo_s1_20260916T1110Z --restart-behavior resume`.
Next: complete200 rounds; automatic best-checkpoint/BC evaluation uses64 heldout
seeds each and paired95% intervals. Only a verified gain with no >=100-count
regression warrants the planned seed2/3 repetitions. No success claim from smoke
or development-set checkpoint selection.
