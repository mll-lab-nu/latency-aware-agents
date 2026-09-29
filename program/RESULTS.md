# Results

## Gymnasium New Environments

Source: user-provided `Latency -- New Envs.pdf`. Return and profile-latency
cells below are arithmetic means; FPS values are configured rates. The source
PDF's standard deviations and latency P95 values are intentionally omitted.
`terminated` means the run produced no reported endpoint; it is not a zero score.
The PDF does not state episode counts or define the reported dispersion.
`Env/obs FPS (F)` is the configured environment/observation clock for the
profile-trained VLA real-time condition; a `terminated` cell has no completed
F evaluation. Frequencies follow the user-provided task/model FPS table.

### OpenVLA (Qwen OFT)

| Environment | Env/obs FPS (F) | Profile latency (ms) | Small model L0 → L0 | VLA L0 → L0 | VLA L0 → real-time | Small model profile → profile | VLA profile → real-time |
|---|---:|---:|---:|---:|---:|---:|---:|
| InvertedPendulum | 10/10 | 67.09 | 1,000.00 | 1,000.00 | 19.80 | 791.90 | 586.85 |
| AirRaid | 50/12.5 | 58.41 | 2,557.50 | 2,000.00 | 878.75 | 3,242.50 | 2,353.75 |
| Ant | 10/10 | 90.56 | 5,011.32 | 5,565.29 | 194.33 | 2,301.03 | 1,853.33 |
| Hopper | 10/10 | 71.63 | 2,682.09 | 1,419.93 | 67.88 | 3,657.61 | 424.58 |
| HalfCheetah | 10/10 | 71.82 | 5,414.26 | 3,675.25 | 357.85 | 5,217.47 | 288.53 |
| Walker2D | 10/10 | 70.50 | 3,743.51 | 983.92 | 103.65 | 4,102.87 | 857.44 |
| Humanoid | 10/10 | 286.64 | 6,331.47 | 272.06 | 226.92 | 1,036.93 | terminated |

### GR00T (QwenGR00T)

| Environment | Env/obs FPS (F) | Profile latency (ms) | Small model L0 → L0 | VLA L0 → L0 | VLA L0 → real-time | Small model profile → profile | VLA profile → real-time |
|---|---:|---:|---:|---:|---:|---:|---:|
| InvertedPendulum | 5/5 | 108.09 | 1,000.00 | 1,000.00 | 15.55 | 1,000.00 | 1,000.00 |
| AirRaid | 50/12.5 | 81.49 | 2,557.50 | 2,101.25 | 733.75 | 2,572.50 | 1,586.25 |
| Ant | 5/5 | 111.64 | 5,011.32 | 6,222.60 | 165.37 | 3,199.56 | 1,955.97 |
| Hopper | 5/5 | 114.62 | 2,682.09 | 872.83 | 8.55 | 3,814.84 | 242.55 |
| HalfCheetah | 5/5 | 110.88 | 5,414.26 | 5,012.47 | -262.47 | 1,847.22 | terminated |
| Walker2D | 5/5 | 111.29 | 3,743.51 | 3,527.99 | 63.24 | 4,228.54 | 852.87 |
| Humanoid | 5/5 | 118.38 | 6,331.47 | 6,225.77 | 1,158.63 | 2,883.10 | terminated |

### Pi0.5 (QwenPI)

| Environment | Env/obs FPS (F) | Profile latency (ms) | Small model L0 → L0 | VLA L0 → L0 | VLA L0 → real-time | Small model profile → profile | VLA profile → real-time |
|---|---:|---:|---:|---:|---:|---:|---:|
| InvertedPendulum | 5/5 | 141.77 | 1,000.00 | 1,000.00 | 20.15 | 1,000.00 | 830.40 |
| AirRaid | 10/2.5 | 134.55 | 2,557.50 | 1,943.75 | 1,386.25 | 3,785.00 | 3,227.50 |
| Ant | 5/5 | 145.42 | 5,011.32 | 5,782.50 | 131.60 | 4,497.52 | 1,774.71 |
| Hopper | 5/5 | 199.10 | 2,682.09 | 3,482.49 | 8.40 | 2,616.17 | 235.17 |
| HalfCheetah | 5/5 | 145.71 | 5,414.26 | 5,357.98 | -347.44 | terminated | terminated |
| Walker2D | 5/5 | 188.57 | 3,743.51 | 3,631.42 | 80.95 | 3,305.17 | 371.09 |
| Humanoid | 5/5 | 182.02 | 6,331.47 | 6,667.56 | 891.67 | terminated | terminated |

## Flappy Bird, Demon Attack, and Deadly Corridor — H1 VLA with H8 decision history

Each VLA cell is the mean return over 100 episodes on the same evaluation seeds
`1,000,000–1,000,099`. All 18 zero- and profile-trained checkpoints completed
5,000 optimizer steps. Evaluation uses 32 parallel environments, worker
capacity 1, and a raw-frame cap of 3,600 / 7,200 / 3,600 for Flappy Bird /
Demon Attack / Deadly Corridor. Their environment/observation rates are
10/10, 60/15, and 35/8.75 FPS, respectively. Each architecture uses its own
RTX 3090 temporal profile; scores across architectures do not compare equal
latency. Deadly Corridor uses the corrected training-matched rendering
protocol (`sf-render-v4`).

| Game | Model | Small model L0 → L0 † | VLA L0 → L0 | VLA L0 → Profile | Small model profile → profile † | VLA profile → Profile |
|---|---|---:|---:|---:|---:|---:|
| Flappy Bird | QwenOFT | 444.60 | 439.13 | 19.19 | 1,233.51 | 373.78 |
| Flappy Bird | QwenGR00T | 444.60 | 428.48 | 6.62 | 1,995.24 | 414.52 |
| Flappy Bird | QwenPI v3 | 444.60 | 408.72 | 5.58 | 1,194.79 | 344.48 |
| Demon Attack | QwenOFT | 8,727.15 | 2362.15 | 584.35 | 5,605.65 | 1452.80 |
| Demon Attack | QwenGR00T | 8,727.15 | 2370.90 | 231.90 | 4,035.40 | 1203.10 |
| Demon Attack | QwenPI v3 | 8,727.15 | 2350.95 | 112.30 | 2,919.80 | 907.30 |
| Deadly Corridor | QwenOFT | 2,116.79 | 2098.67 | 742.35 | 1,933.82 | 2091.23 |
| Deadly Corridor | QwenGR00T | 2,116.79 | 2100.46 | 341.76 | 2,155.71 | 2090.82 |
| Deadly Corridor | QwenPI v3 | 2,116.79 | 2104.09 | 140.94 | 1,397.40 | 1450.83 |

† Small-model cells are independently evaluated teacher checkpoints, not training
rewards or VLA scores. The shared zero teachers use 100 episodes, actual seeds
`104771–104870`, and caps of 3,600 / 14,400 / 30,000 for Flappy / Demon /
Deadly. Profile teachers use the training-selected **best** checkpoint from each
25M-step, seed-42 run, 100 episodes on `104771–104870`, env/obs rates
10/10, 60/15, 35/8.75 FPS, and caps of 18,000 / 14,400 / 3,600.
Demon profile scores use the corrected 2026-09-29 full-game, raw-reward,
16-way CuLE evaluation. The former GPU values stopped at the first life loss;
all 300 old score/length pairs were reproduced by that counterfactual and are
superseded. The invalid Deadly 60/15 FPS runs remain excluded. These teacher
columns have different seed/cap/backend protocols from the VLA columns and
must not be treated as a matched teacher–VLA comparison or a paired L0–profile
comparison. In particular, Flappy's larger profile return has a longer cap.

The teacher–VLA comparison now uses a second evaluation of the same selected
weights on the student cohort: 100 seeds `1000000–1000099`, latency seed
`271828`, 32 environments, original temporal profiles, and matched caps of
7,200 / 3,600 raw frames for Demon / Flappy. Student evaluations already
continued across life losses and retain their archived scores.

| Game | Model | Matched teacher | Student | Student / teacher |
|---|---|---:|---:|---:|
| Demon Attack | QwenOFT | 1603.60 | 1452.80 | 90.6% |
| Demon Attack | QwenGR00T | 1305.00 | 1203.10 | 92.2% |
| Demon Attack | QwenPI v3 | 980.05 | 907.30 | 92.6% |
| Flappy Bird | QwenOFT | 392.00 | 373.78 | 95.4% |
| Flappy Bird | QwenGR00T | 437.90 | 414.52 | 94.7% |
| Flappy Bird | QwenPI v3 | 385.60 | 344.48 | 89.3% |

Ratios use unrounded means. The [audit report](../outputs/demon_cule_audit_20260929/REPORT.md)
records the raw-reward fix, first-life counterfactual, residual grayscale
rounding effects and CPU/CuLE comparisons. All checkpoints and training curves
are retained; no model was retrained or reselected.

Zero-teacher evidence: [Flappy](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/03984454d1b715d49e27750b17788dacfc39a2a1/zero-latency/flappy/small-policy/sample-factory-v1/eval_l0.json) · [Demon](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/03984454d1b715d49e27750b17788dacfc39a2a1/zero-latency/demon-attack/small-policy/sample-factory-v1/eval_l0.json) · [Deadly](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/03984454d1b715d49e27750b17788dacfc39a2a1/zero-latency/deadly-corridor/small-policy/sample-factory-v1/eval_l0.json).
[Profile-teacher best-checkpoint scores, run links and episode records](../outputs/tasks/decision_history_nine_profile_20260923/nine_profile_results.json).

The zero-trained VLA policy loses return under its profile in all nine cases;
profile training improves the return under that profile in all nine cases.
[Verified evaluation summary, checkpoint identities, and per-run audit links](../outputs/tasks/decision_history_vla_hist8_v2/evaluation_summary.md).

## Atlantis and Asterix — H1 VLA with H8 decision history

All twelve zero- and profile-trained VLA checkpoints completed 5,000 optimizer
steps. Each VLA cell is mean raw return over 100 episodes on the same actual seeds
`1,000,000–1,000,099`, using a GPU simulator, 32 parallel environments, worker
capacity 1, 60/15 environment/observation Hz, and a 7,200-raw-frame cap.
The input is current RGB plus eight binary decision-history windows; Atlantis
uses 4 actions / 56D history and Asterix uses native 9 actions / 136D history.
Each model's two profile-environment evaluations use the same fixed RTX 3090
temporal profile. Profiling uses CPU simulation/rendering and one visible GPU;
the scores here come from separate GPU evaluations. These are single-training-seed
comparisons, not estimates of training-seed variability.

| Game | Model | Small model L0 → L0 † | VLA L0 → L0 | VLA L0 → Profile | Small model profile → profile † | VLA profile → Profile |
|---|---|---:|---:|---:|---:|---:|
| Atlantis | QwenOFT | 88,613.00 | 39,560.00 | 18,588.00 | 38,411.00 | 39,229.00 |
| Atlantis | QwenGR00T | 88,613.00 | 39,662.00 | 8,167.00 | 29,064.00 | 28,488.00 |
| Atlantis | QwenPI v3 | 88,613.00 | 39,615.00 | 4,369.00 | 27,780.00 | 26,447.00 |
| Asterix | QwenOFT | 6,041.00 | 5,803.50 | 374.50 | 3,941.00 | 3,653.50 |
| Asterix | QwenGR00T | 6,041.00 | 5,714.00 | 182.00 | 1,852.50 | 1,440.00 |
| Asterix | QwenPI v3 | 6,041.00 | 5,758.50 | 143.50 | 1,141.00 | 1,015.00 |

† The shared zero teacher for each game is `sample-factory-hist8-cpu15m-s0-v1`;
its archived L0 evaluation uses 100 episodes, seeds `104771–104870`, 60/15 Hz,
and a **14,400**-frame cap. It is not matched to the VLA's 7,200-frame cap.
The six profile-teacher cells use the training-selected best checkpoints from
15M-step, seed-0 runs and the corrected 100-episode GPU evaluations: seeds
`1,000,000–1,000,099`, 60/15 Hz, parallel 32, capacity 1, 7,200 frames and each
model's RTX 3090 temporal profile. Zero and profile teacher columns therefore
must not be interpreted as a paired latency-effect estimate.

Zero-teacher evidence: [Atlantis](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/03984454d1b715d49e27750b17788dacfc39a2a1/zero-latency/atlantis/small-policy/sample-factory-hist8-cpu15m-s0-v1/eval_l0.json) · [Asterix](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/03984454d1b715d49e27750b17788dacfc39a2a1/zero-latency/asterix/small-policy/sample-factory-hist8-cpu15m-s0-v1/eval_l0.json).
[Six verified profile-teacher means and pinned checkpoints](../outputs/tasks/atari_hist8_v1/teachers.complete.json).

All 18 VLA evaluations (1,800 episodes) are complete and published. Episode records,
action audits, protocol/checkpoint/profile identities and W&B summaries passed
verification. All six profile-trained models improve over their zero-trained
counterparts under the same profile, while only Atlantis OFT approaches its
zero-latency baseline. The twelve VLA bundles, six profile-aware teachers and
six profiles are published and readback-verified; teacher evaluations add 600
independent episodes. Table 1 and Figure 1 use all 18 VLA means.

[Local complete comparison and checkpoint/profile identities](../outputs/tasks/atari_hist8_v1/evaluation_progress.md) ·
[Local per-episode results and audits](../outputs/tasks/atari_hist8_v1/evaluation_evidence/) ·
[Published report and model index](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/ea2ad5a90c28de1297793956a93dd3a7ed605c4b/reports/atlantis-asterix-h8-v1/README.md) ·
[Post-shutdown Hub checkpoint verification](../outputs/tasks/atari_hist8_v1/paper_update_20260925/hub_checkpoint_verification.json).

## HumanoidBench Balance Simple

Cells report mean return for zero-trained → zero, zero-trained → profiled, and
profile-conditioned → profiled evaluation. QwenOFT uses H8 action chunks;
QwenPI v3 and QwenGR00T use H16. The zero-latency denominators and the two
profiled evaluations use their verified matched 64-seed cohorts. The profiled
arms use env/obs **50/50 Hz**, capacity 1, a 1,000-frame cap, and each model's
RTX 3090 temporal profile.

| Model | Zero-trained → Zero | Zero-trained → Profile | Profile-trained → Profile |
|---|---:|---:|---:|
| QwenOFT | 275.56409 | 50.745633 | 114.625128 |
| QwenPI v3 | 328.60200 | 46.845423 | 72.312299 |
| QwenGR00T | 267.52374 | 45.340414 | 90.420796 |

[Formal evaluation report and protocol](../outputs/tasks/humanoidbench_balance_simple/causal_profile_eval_20260920/REPORT.md) ·
[raw comparison](../outputs/tasks/humanoidbench_balance_simple/causal_profile_eval_20260920/comparison.json).

## MIKASA InterceptGrabFast

Cells report `task success; mean return`. Corrected profile-trained H1 results
include `mean±SD`; the H8 DAgger tables report mean return. QwenOFT, QwenPI v3,
and QwenGR00T use env/obs 20/20 Hz in these tables. The archived WanOFT H1
row uses 10/10 Hz, as recorded in its [profile evaluation configuration](../outputs/tasks/mikasa_intercept_grab_fast/complete_200_eval_20260922/legacy_906fc688_7a4ee201/wanoft/profile-trained_profile/extra100/eval_config.yaml).

### 1. Single Action (No Action Chunk), RTX 3090 Profile

| Model | Small model L0 → L0 † | VLA L0 → L0 | VLA L0 → Profile | Small model profile → profile † | VLA Profile-trained → Profile |
|---|---:|---:|---:|---:|---:|
| QwenOFT | 100/100 (100%); 36.81948 | 114/200 (57.0%); 21.9662 | 0/200 (0.0%); 0.4615 | 255/256 (99.61%); 33.64463 | 101/200 (50.5%); 17.77506±15.43550 |
| QwenPI v3 | 100/100 (100%); 36.81948 | 106/200 (53.0%); 21.0783 | 0/200 (0.0%); 0.2845 | 253/256 (98.83%); 28.64633 | 90/200 (45.0%); 13.55261±13.75740 |
| QwenGR00T | 100/100 (100%); 36.81948 | 133/200 (66.5%); 24.7617 | 0/200 (0.0%); 0.4353 | 251/256 (98.05%); 31.75039 | 100/200 (50.0%); 17.51561±14.29526 |
| WanOFT | 100/100 (100%); 36.81948 | 92/200 (46.0%); 19.2724 | 0/200 (0.0%); 0.3314 | 252/256 (98.44%); 30.86635 | 63/200 (31.5%); 10.3123 |

† The shared zero PPO teacher's archived strict evaluation reports 100/100
successes and mean return 36.81948, computed from its 100 raw returns on seeds
`4242424242–4242424341`; it is not a 200-episode VLA evaluation.
[Zero-teacher raw evaluation](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/03984454d1b715d49e27750b17788dacfc39a2a1/zero-latency/mikasa-intercept-grab-fast/small-policy/ppo-v1/strict-eval.json).
H1 profile-teacher cells use separate native 256-episode evaluations with
`environment_seed=20260910`, `latency_seed=271828` and 60 transitions.
Teacher success in this table is **success_once**, matching the VLA task-success
metric; the source's top-level `success_rate` instead denotes `success_at_end`.
These native teacher evaluations are not paired with the VLA's 200 seeds.

Profile-teacher evidence: [OFT](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/03984454d1b715d49e27750b17788dacfc39a2a1/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwenoft-rtx3090-v1/strict-eval.json) · [Pi0.5](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/03984454d1b715d49e27750b17788dacfc39a2a1/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwenpi_v3-rtx3090-v1/strict-eval.json) · [GR00T](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/03984454d1b715d49e27750b17788dacfc39a2a1/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-qwengr00t-rtx3090-v1/strict-eval.json) · [WanOFT](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/03984454d1b715d49e27750b17788dacfc39a2a1/latency-aware/mikasa-intercept-grab-fast/small-policy/ppo-wanoft-rtx3090-v1/strict-eval.json).

The corrected QwenOFT/QwenPI v3/QwenGR00T cells use unique seeds
`4242424242–4242424441`, latency seed `271828`, and mean profile latencies
`99.091 / 148.740 / 99.365 ms`. Their execution audits found
`11,561 / 11,346 / 11,554` non-noop steps out of 12,000. The superseded
0/200, `0.189403±0.031062` evaluations were invalid because all environment
steps received noop actions. [Correction evidence](../outputs/tasks/mikasa_intercept_grab_fast/h1_hold_last_reval_20260923/REPORT.md).
The three corrected bundles are published under new `*_evalfix_20260923` run
IDs and were jointly verified at
[`benchmark-models@ae81bdb3`](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/ae81bdb3b6c6605926255e91b499b4badf93a947/latency-aware/mikasa-intercept-grab-fast/vla).

### 2. H8 Action Chunk DAgger Best, RTX 3090 Profile

| Model | Small model L0 → L0 † | VLA L0 → L0 | VLA L0 → Profile | Small model profile → profile † | VLA Profile-trained → Profile |
|---|---:|---:|---:|---:|---:|
| QwenOFT | 100/100 (100%); 36.81948 | 196/200 (98.0%); 36.108 | 101/200 (50.5%); 18.596 | N/A (action chunk) | 138/200 (69.0%); 25.752 |
| QwenPI v3 | 100/100 (100%); 36.81948 | 199/200 (99.5%); 36.785 | 40/200 (20.0%); 8.082 | N/A (action chunk) | 89/200 (44.5%); 17.558 |
| QwenGR00T | 100/100 (100%); 36.81948 | 192/200 (96.0%); 33.514 | 73/200 (36.5%); 14.314 | N/A (action chunk) | 134/200 (67.0%); 25.235 |

### 3. H8 Action Chunk DAgger Best, Fixed-2 (100 ms / 2 Frames)

| Model | Small model L0 → L0 † | VLA L0 → L0 | VLA L0 → Fixed-2 | Small model fixed-2 → fixed-2 † | VLA Fixed-2-trained → Fixed-2 |
|---|---:|---:|---:|---:|---:|
| QwenOFT | 100/100 (100%); 36.81948 | 196/200 (98.0%); 36.108 | 109/200 (54.5%); 20.955 | N/A (action chunk) | 175/200 (87.5%); 31.680 |
| QwenPI v3 | 100/100 (100%); 36.81948 | 199/200 (99.5%); 36.785 | 115/200 (57.5%); 21.473 | N/A (action chunk) | 159/200 (79.5%); 29.303 |
| QwenGR00T | 100/100 (100%); 36.81948 | 192/200 (96.0%); 33.514 | 95/200 (47.5%); 17.838 | N/A (action chunk) | 164/200 (82.0%); 29.866 |

`N/A (action chunk)` marks the absence of a separate profile/fixed-2 teacher
for these H8 DAgger comparisons. The repeated L0 teacher cells refer to the
same 100-episode PPO evaluation above, not additional H8 teacher runs.

Each H8 run completed 5,000 DAgger optimizer updates; the tables use its best
checkpoint selected with 100-episode development evaluations. `Zero-trained`
denotes naive zero-latency DAgger; Profile- and Fixed-2-trained columns use
their respective conditioned DAgger runs. Independent final evaluations use
the same 200 seeds `4242424542–4242424741`, H8×7 actions, 20/20 Hz, and
worker capacity 1.
Execution audits passed, and all nine best bundles were published. Best steps
for QwenOFT/QwenPI v3/QwenGR00T are 5,000/4,000/3,000 for naive,
5,000/5,000/4,000 for Profile, and 5,000/5,000/5,000 for Fixed-2.
The Profile QwenPI v3 result is 89/200, below the conditioned-run 100/200 gate;
the other five conditioned runs meet it. These held-out seeds differ from the
Figure 2 sweep below. Full results and pinned revisions are in the
[naive DAgger record](logs/mikasa-h8-naive-dagger-2026-09-23.md) and
[conditioned DAgger record](logs/mikasa-h8-dagger-2026-09-23.md).

## H1 / H8 DAgger Fixed Latency 0–4 — MIKASA InterceptGrabFast

Three zero-latency-trained H1 checkpoints from
[benchmark-models@56b2c095](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/56b2c095d76ea5b10dd304e6e03c2ca81801630b/zero-latency/mikasa-intercept-grab-fast/vla)
and three zero-latency naive H8 DAgger best checkpoints from
[benchmark-models@5b533937](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/5b533937a72e9fa923ecf28b1c1be5a16bad488d/zero-latency/mikasa-intercept-grab-fast/vla)
were evaluated at fixed delays of 0–4 environment frames (0, 50, 100, 150,
200 ms at 20 Hz). Every point contains 200 episodes with the same ordered seeds
`4242424242–4242424441`. The 15 previously verified H1 conditions are unchanged;
all 15 H8 DAgger conditions were newly evaluated. Success is `success_once`.
DAgger best was selected at step 5000 for QwenOFT, 4000 for QwenPI v3, and
3000 for QwenGR00T; profile- and fixed-trained H8 checkpoints are not part of
this sweep.

QwenPI v3 H1 uses the original Hub checkpoint weights with an isolated inference
config setting `interleave_self_attention: false`. The Hub config says `true`,
but that checkpoint was trained under the older effective non-interleaved
attention behavior. On the same first 20 seeds, current inference with the Hub
config scored 0/20, the corrected config 11/20, and the older inference code
10/20. The [corrected config](../outputs/tasks/mikasa_intercept_grab_fast/figure2_h1_h8_naive_dagger_fixed_latency_20260923/qwenpi_h1_model_config_override.yaml)
and per-condition requested configs preserve this distinction; the original Hub
bundle was not modified. [Current-code A/B traces](../outputs/tasks/mikasa_intercept_grab_fast/figure2_pi_interleave_false_probe_20260923/) and
[old-code probe](../outputs/tasks/mikasa_intercept_grab_fast/figure2_pi_h1_legacy_probe_20_20260923/qwenpi_v3/h1/latency_0/summary.json)
retain the 20-episode checks.

### Success rate

| Model / horizon | L0 | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|---:|
| QwenOFT H1 | 110/200 (55.0%) | 19/200 (9.5%) | 0/200 | 0/200 | 0/200 |
| QwenOFT H8 DAgger | 195/200 (97.5%) | 113/200 (56.5%) | 115/200 (57.5%) | 44/200 (22.0%) | 15/200 (7.5%) |
| QwenPI v3 H1 | 100/200 (50.0%) | 11/200 (5.5%) | 0/200 | 0/200 | 0/200 |
| QwenPI v3 H8 DAgger | 198/200 (99.0%) | 154/200 (77.0%) | 123/200 (61.5%) | 56/200 (28.0%) | 12/200 (6.0%) |
| QwenGR00T H1 | 127/200 (63.5%) | 22/200 (11.0%) | 0/200 | 0/200 | 0/200 |
| QwenGR00T H8 DAgger | 191/200 (95.5%) | 123/200 (61.5%) | 101/200 (50.5%) | 31/200 (15.5%) | 15/200 (7.5%) |

### Episode return, mean ± population SD

| Model / horizon | L0 | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|---:|
| QwenOFT H1 | 21.44710±16.60358 | 4.08698±8.37014 | 0.48012±0.12433 | 0.33467±0.10682 | 0.26848±0.07918 |
| QwenOFT H8 DAgger | 35.68453±6.73871 | 22.03428±13.63211 | 21.77157±15.11639 | 8.79956±12.96139 | 3.63666±8.38446 |
| QwenPI v3 H1 | 20.56688±16.49483 | 3.57036±7.31282 | 0.45217±0.16058 | 0.31099±0.11818 | 0.25571±0.14577 |
| QwenPI v3 H8 DAgger | 36.93959±5.38621 | 28.57693±12.49792 | 23.09833±14.97713 | 10.50514±14.10796 | 3.01435±7.38387 |
| QwenGR00T H1 | 23.25781±15.64883 | 5.06853±9.32082 | 0.47333±0.14359 | 0.30635±0.06614 | 0.25650±0.12395 |
| QwenGR00T H8 DAgger | 33.96167±8.02164 | 23.74414±12.54277 | 18.13737±14.74691 | 6.81241±11.37256 | 3.02715±7.68752 |

All episode lengths are 60 transitions (mean 60, population SD 0, min/max
60/60). Evaluation used deterministic inference, 10 parallel environments,
batch size 8, simulated worker capacity 1, `latest_ready + hold +
hold_last_chunk_action`, and an immediate environment start with noop until the
first action arrives. H1 emits a 7D action; H8 executes observation-aligned
`[8,7]` chunks. All 30 conditions passed fixed-latency and step-trace audits;
the 12 delayed H8 conditions also executed later chunk heads. This is a
controlled simulated-latency sweep, not a hardware wall-clock latency measure.
QwenOFT H8 DAgger L1/L2 were independently rerun with the same 200 seeds;
success counts, per-episode success/return vectors and execution audits matched
the formal `113/200` and `115/200` results exactly.

[Full aggregate, checkpoint paths, min/max statistics and execution audits](../outputs/tasks/mikasa_intercept_grab_fast/figure2_h1_h8_naive_dagger_fixed_latency_20260923/combined_results.json) ·
[artifact manifest](../outputs/tasks/mikasa_intercept_grab_fast/figure2_h1_h8_naive_dagger_fixed_latency_20260923/artifact_manifest.json) ·
[complete raw outputs](../outputs/tasks/mikasa_intercept_grab_fast/figure2_h1_h8_naive_dagger_fixed_latency_20260923/) ·
[QwenOFT L1/L2 identical-seed replication](../outputs/tasks/mikasa_intercept_grab_fast/figure2_h8_dagger_oft_l1_l2_replication_20260923/) ·
[Figure 2 data and panel](../paper/figures/2_2_performance_degradation/README.md).

## QwenOFT H8 Fixed-2 → Fixed Latency 0–4 — MIKASA InterceptGrabFast

The published Fixed-2 DAgger step-5000 checkpoint at
[benchmark-models@795468e5](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/795468e53947d26314e891cbc375775c6e0bd491/fixed-2/mikasa-intercept-grab-fast/vla/starvla-qwenoft-h8/h8-conditioned-dagger-20260923)
was evaluated with its trained **K=2 hard prefix retained at every latency**, including L0.
Each setting uses 200 episodes, common seeds `4242424242–4242424441`, 20/20Hz,
60 transitions, deterministic inference, capacity 1, and observation-aligned H8×7 chunks.
The 71D conditioned input is retained; the existing naive Zero model uses 7D without a prefix.

| Metric | L0 | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|---:|
| Success | 1/200 (0.5%) | 1/200 (0.5%) | 177/200 (88.5%) | 0/200 (0.0%) | 0/200 (0.0%) |
| Return mean ± population SD | 3.39886±5.89048 | 3.39886±5.89048 | 32.41834±8.76923 | 0.94342±0.74828 | 0.83236±1.71474 |

All five execution audits passed. L0 and L1 have identical per-episode return vectors.
No off-diagonal result was discarded or used to reselect the model. The remaining
Fixed-1/3/4 training columns are pending; this is not a completed 5×5 grid.
[Column summary](../outputs/tasks/mikasa_intercept_grab_fast/fixed-grid-20260924/fixed2_column.json) ·
[Raw records](../outputs/tasks/mikasa_intercept_grab_fast/fixed-grid-20260924/train_2/) ·
[Execution protocol](MIKASA_FIXED_LATENCY_GRID.md).
