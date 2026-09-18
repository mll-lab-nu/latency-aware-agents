# Hopper — GR00T and OpenVLA results

## Historical L0 teacher result — corrected 2026-09-18

**2682.09 ± 900.29**, 10 episodes; population SD (`ddof=0`). This is the original APPO teacher evaluation, not a new 5 FPS evaluation. Recomputed from the original per-episode returns; the HTML report and `results-summary.json` now include the standard deviation.


[Open the experiment report](index.html#gr00t-results). Updated 18 September 2026 from completed, pinned evidence.

GR00T H1 at 5 FPS: measured return **8.55 ± 0.09 → 242.55 ± 127.57**, mean episode length **9 → 100.1**, mean paired gain **+234.00**. All 20 matched seeds improve. These are population standard deviations over episodes from one training run. F still ends well before the 1,000-step cap on average.

| Measured metric | B | F |
|---|---:|---:|
| Episodes / seeds | 20 / 42–61 | 20 / 42–61 |
| Return | 8.55 ± 0.09 | 242.55 ± 127.57 |
| Mean length | 9.0 | 100.1 |
| Mean of episode latency means (ms) | 111.10969 | 110.79241 |
| Observation / action drops / invalid | 0 / 0 / 0 | 0 / 0 / 0 |

B/F share measured-only latency, 5 FPS, H1 and cap 1000. Only model paths, prompt key 0 → 1 and output change. F uses no added IID delay. Across 2,002 raw steps, within-episode frame intervals average 200.00776 ms (198.96645–201.19602). Similar means do not imply identical delays or equal-latency causality.

## Separate training and simulation evidence

- Native Hopper-v4 physics, state 11, native torque action 3, one RGB224 frame and H1. Train-only state min-max is applied once.
- L0 VLA: reused original 100 APPO H1 episodes, 88,212 / 9,917 rows; prompt and nominal clock metadata retimed 125 → 5 FPS. Fresh 5,000 updates. Separate L0 simulation: **872.83 ± 479.65**, length 261.75 (20 episodes).
- Actual RTX 3090 P: 5 × 10 episodes / cap 900 / no calibration; 450 observations, mean 114.61719 ms and P95 122.60570 ms.
- Fresh profile APPO teacher: original 10M-step recipe, seed 3333; deterministic Final/best 20 each, best selected at **3815.46 ± 7.63**. E10: **3814.84 ± 8.80**, reusing the first 10 selection seeds.
- Strict teacher replay return >3000 yields **100 accepted / 101 attempts**, 90,000 / 10,000 rows. The teacher gate is not a VLA acceptance criterion.
- Profile VLA: fresh 5,000 updates, seed 42, global 64 = 2 × micro 8 × accumulation 4; DiT-B, 4 inference timesteps / 8 diffusion repeats, LR VLM 1e-5 / head 1e-4, warmup 100, held-out 200 batches / save every 500.

## Published evidence

[Dataset and final compact evidence](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/f572d99f0e14d858c8a21a8f3fd182ecb34f4795/hopper/profile_latency/GR00T/h1_5fps_20260917) · [Final model](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/51e9093cb9e8da5e0ab2a5dcc1e52ea0e69f7d13/hopper/profile_latency/GR00T) · [Selected teacher](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/ee6177b9772236e765fb95f9900a2ddd610afd5d/hopper/profile_latency/small_model/GR00T) · [L0 model](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/50e51e28c1fc7f55159eba4c611857b2df56539b/hopper/zero_latency/GR00T) · [Results JSON](results-summary.json) · [Publication receipt](gr00t-publication.json) · [Paired chart](gr00t-paired-returns.svg).

Evaluated F model revision `d33af26b7f87357c1b7a2b75bae4923bd90b6cc1`; SHA256 `fe4fb89a88d9aab7f266a69921b32cfc00a0e2964664be3953988a8f4062d3ca`. Documentation revision `51e9093cb9e8da5e0ab2a5dcc1e52ea0e69f7d13` preserves weights; dataset revision `f572d99f0e14d858c8a21a8f3fd182ecb34f4795`.

Full step/action/latency traces were audited server-side and remain on the evaluation server. Published compact evidence includes episode metrics/configurations/exit receipts/summaries; no claim that all raw traces are uploaded.

## Preserved OpenVLA history

Historical OpenVLA at 10 FPS: **67.88 ± 150.09 → 424.58 ± 348.01**. Different clocks and recipes preclude a controlled cross-model ranking. All original anchors, media and conditions remain in the [HTML record](index.html#profile-pipeline-20260911).

### Original Hopper RGB+state adaptation

Latest experiment: [September 2026 profile-latency results](index.html#profile-pipeline-20260911). Earlier L0/L2 results below refer to historical experiments with their own evaluation conditions.

Open [`index.html`](index.html) for the canonical Hopper-v4 adaptation record.
The current path uses an independent task-owned wrapper, the native three-dimensional
continuous torque space, an ordered 11-dimensional state contract, L0 Sample Factory
training, L0-L4 latency-bench evaluation, 100 accepted L0 teacher episodes,
native RGB+11D-state LeRobot conversion, and a 5k-step QwenOFT run. The final
text-bin VLA checkpoint failed closed-loop evaluation (L0 mean return 173.136;
L2 mean return 9.513). A controlled 5k-step continuous-state projector rerun
raises those means to 1,419.935 and 315.890, respectively, while remaining well
below the same-seed L0 teacher. The report preserves the original failure
diagnosis and separates representation gains from the remaining distribution gap.

Fresh profiling-only run: [20260911T023128Z-p-only4](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/hopper/profile_latency/provenance/profiling/status.md), env/obs FPS 10 / 10. Profiling and publication are complete; the subsequent full pipeline is recorded below.

- [September 11 realtime baseline](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/hopper/profile_latency/provenance/baseline/status.md): completed 20 episodes at 10 FPS; return 67.88 ± 150.09.

- [Hopper pipeline continuation](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/hopper/profile_latency/provenance/pipeline/status.md): reuses P/B; H100 completed the 10M-frame teacher recipe with the new profile at 10 FPS; simulated teacher evaluation completed at 3,657.61 ± 13.88, with all 10 episodes exceeding 3000. R/C produced 98,817 verified rows with all 100 episodes above the original gate; 123 data files are uploaded and verified; VLA completed 5000 steps with exit 0 and W&B finished (held-out L1 0.01595611); all 22 final bundle files are published and verified; matched F evaluation completed all 20 seeds: return 424.58 ± 348.01 versus B 67.88 ± 150.09 (6.26×; 20/20 paired gains), mean length 139.45, no episode reaches 1000 steps. Hopper retains strict data return >3000 and the continuous-projector VLA.

## Standard Pipeline archive

[Complete experiment archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/hopper) · [Verification receipt](assets/standard_pipeline.json). This local directory retains report display media; original records are linked at a fixed revision. Only raw demonstrations are distributed, with conversion performed on the training server.


[OpenVLA per-seed real-time comparison](openvla-paired-returns.svg) · [Plot data and fixed-revision sources](openvla-paired-results.json). 20 wins, 0 ties, 0 losses; mean paired gain +356.70.


The HTML tables follow the AirRaid schema: B/F mean gain and percentage; separate L0/B/E/F evaluation rows with episode counts and mean raw steps; measured runtime audit; pipeline stages; publication index. [Normalized table values and provenance](report-tables.json). Unrun stages and unrecorded fields are explicitly distinct.
