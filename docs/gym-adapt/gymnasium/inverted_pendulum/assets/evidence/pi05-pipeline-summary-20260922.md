# Pi0.5 pipeline results — 22 September 2026

Mean ± population standard deviation; blank means unavailable or the corresponding stage was not run.

| Environment | Profiling Mean / P95 (ms) | Small Model L0 → L0¹ | VLA L0 → L0 | VLA L0 → Real-time | Small Model Profile → Profile² | VLA Profile → Real-time |
| --- | --- | --- | --- | --- | --- | --- |
| InvertedPendulum | 141.77 / 148.17 |  | 1,000.00 ± 0.00 | 20.15 ± 6.48 | 1,000.00 ± 0.00 | 830.40 ± 339.20 |
| AirRaid | 134.55 / 139.89 | 2,557.50 ± 491.81 | 1,943.75 ± 776.85 | 1,386.25 ± 434.99 | 3,785.00 ± 384.09 | 3,227.50 ± 1,095.58 |
| Ant | 145.42 / 151.63 | 5,011.32 ± 1,880.32 | 5,782.50 ± 963.07 | 131.60 ± 93.72 | 4,497.52 ± 71.51 | 1,774.71 ± 824.61 |
| HalfCheetah | 145.71 / 151.53 | 5,414.26 ± 740.00 | 5,357.98 ± 736.99 | −347.44 ± 82.66 |  |  |
| Hopper | 199.10 / 264.06 | 2,682.09 ± 900.29 | 3,482.49 ± 290.80 | 8.40 ± 0.66 | 2,616.17 ± 629.87 | 235.17 ± 28.38 |
| Walker2d | 188.57 / 209.44 | 3,743.51 ± 996.07 | 3,631.42 ± 775.19 | 80.95 ± 10.98 | 3,305.17 ± 1,032.90 | 371.09 ± 123.54 |
| Humanoid | 182.02 / 198.61 | 6,331.47 ± 407.20 | 6,667.56 ± 46.18 | 891.67 ± 623.80 |  |  |

1. L0 small-model figures are formal historical evaluations of the shared source teachers at their original clocks; they are not new evaluations at the current VLA clock. InvertedPendulum has no independent formal L0 teacher summary; accepted rollout scores are not substituted. AirRaid VLA uses env/obs 10/2.5 FPS; the other VLA runs use 5/5 FPS.
2. Profile small-model figures consistently use E10, which repeats the first ten selection seeds and is not an independent validation set. HalfCheetah and Humanoid stopped at selection (0/20 above their respective gates), so E10 and downstream profile VLA/F are blank. Their selection20 results are 1889.53 ± 44.32 and 1538.29 ± 686.65 respectively.
3. VLA evaluations use 20 episodes. HalfCheetah B is checked against completed status and 20 episode metrics, not yet full-trace audited. Ant B preserves 18 episodes plus 2 after server interruption, with the session/RNG reset disclosed.
4. Hopper collection was explicitly amended to strict return >2500 with no attempt cap. Profiling warmup drift warnings for Hopper session1 and Humanoid sessions0/2/3 remain in source audits. Different clocks, measured sessions and co-resident loads do not establish controlled architecture or equal-latency causal comparisons.
5. The example IP values 108.09/114.50, B15.55 ±4.79 and F1000 belong to GR00T, not this Pi0.5 H1 run.

## Evidence

- Per-task current runs under `outputs/pi05`: `h1_5fps_20260918` (InvertedPendulum), `h1_10env_2p5obs_g128_20260920` (AirRaid), `h1_5fps_g128_20260920` (Ant), and `h1_5fps_g128_20260921` (remaining environments).
- P: `P-verification.json` or `profile-payload/validation/P-verification.json`; L0: `l0-audit.json` (Humanoid report embeds the exact audit).
- B/F: root or `profile_latency/{B,F}-verification.json`; Ant B: `B-recovered-audit.json`; Half B: `B-summary-for-table.json`.
- E10: IP/Air/Ant `teacher-verification.json`; Walker `profile_latency/teacher-verification.json`; Hopper `profile-data-gt2500-review/source-gt2500.json` → `teacher.E10`.
- Original L0 teachers: `outputs/examples/gymnasium_air_raid/eval_random_noop/queue_eval_latency_0.json`, `outputs/examples/gymnasium_ant/eval/queue_eval_latency_0.json`, `outputs/examples/gymnasium_hopper_rgb_state/eval/queue_eval_latency_0.json`; Half/Walker formal teacher tables in corresponding `docs/gym-adapt/gymnasium/*/index.html`; Humanoid `results-summary.json` → `historical_L0_teacher`.
