# Walker2D — GR00T and OpenVLA results

[Open the experiment report](index.html#gr00t-results). Updated 18 September 2026 from completed, pinned evidence.

GR00T H1 at 5 FPS: measured return **63.24 ± 6.58 → 852.87 ± 357.10**, mean episode length **61.15 → 250.6**, mean paired gain **+789.63**. All 20 matched seeds improve. These are population standard deviations over episodes from one training run. F still ends well before the 1,000-step cap on average.

| Measured metric | B | F |
|---|---:|---:|
| Episodes / seeds | 20 / 42–61 | 20 / 42–61 |
| Return | 63.24 ± 6.58 | 852.87 ± 357.10 |
| Mean length | 61.15 | 250.6 |
| Mean of episode latency means (ms) | 107.04821 | 107.06146 |
| Observation / action drops / invalid | 0 / 0 / 0 | 0 / 0 / 0 |

B/F share measured-only latency, 5 FPS, H1 and cap 1000. Only model paths, prompt key 0 → 1 and output change. F uses no added IID delay. Across 5,012 raw steps, within-episode frame intervals average 200.00296 ms (198.95315–201.21660). Similar means do not imply identical delays or equal-latency causality.

## Separate training and simulation evidence

- Native Walker2d-v4 physics, state 17, native torque action 6, one RGB224 frame and H1. Train-only state min-max is applied once.
- L0 VLA: reused original 100 APPO H1 episodes, 85,160 / 9,664 rows; prompt and nominal clock metadata retimed 125 → 5 FPS. Fresh 5,000 updates. Separate L0 simulation: **3527.99 ± 945.27**, length 901.45 (20 episodes).
- Actual RTX 3090 P: 5 × 10 episodes / cap 900 / no calibration; 3,095 observations, mean 111.28701 ms and P95 118.31860 ms.
- Fresh profile APPO teacher: original sf_official_10m recipe, seed 0; deterministic Final/best 20 each, best selected at **4223.32 ± 18.18**. E10: **4228.54 ± 14.59**, reusing the first 10 selection seeds.
- Strict teacher replay return >3000 yields **100 accepted / 100 attempts**, 90,000 / 10,000 rows. The teacher gate is not a VLA acceptance criterion.
- Profile VLA: fresh 5,000 updates, seed 42, global 64 = 2 × micro 32 × accumulation 1; DiT-B, 4 inference timesteps / 8 diffusion repeats, LR VLM 1e-5 / head 1e-4, warmup 100, held-out 200 batches / save every 500.

## Published evidence

[Dataset and final compact evidence](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/b59922566cb475e61455342b1111b9e9771c09de/walker2d/profile_latency/GR00T/h1_5fps_20260917) · [Final model](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/54172536f1a4819babf336e599ea38ef4e88809a/walker2d/profile_latency/GR00T) · [Selected teacher](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/3ceab6778f0e7aef377fdfd7e4faa9685ede7a78/walker2d/profile_latency/small_model/GR00T) · [L0 model](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/9ea50f42a1e93a9035c8df6a392e5e9f141493de/walker2d/zero_latency/GR00T) · [Results JSON](results-summary.json) · [Publication receipt](gr00t-publication.json) · [Paired chart](gr00t-paired-returns.svg).

Evaluated F model revision `af4f38060b0374c5684c43f22cd504257cf2bffd`; SHA256 `1859cec5319db37b97a62d8190f081771f38978fa2ade39cfb82b155ac56b83b`. Documentation revision `54172536f1a4819babf336e599ea38ef4e88809a` preserves weights; dataset revision `b59922566cb475e61455342b1111b9e9771c09de`.

Full step/action/latency traces were audited server-side and remain on the evaluation server. Published compact evidence includes episode metrics/configurations/exit receipts/summaries; no claim that all raw traces are uploaded.

## Preserved OpenVLA history

Historical OpenVLA at 10 FPS: **103.65 ± 4.90 → 857.44 ± 971.04**. Different clocks and recipes preclude a controlled cross-model ranking. All original anchors, media and conditions remain in the [HTML record](index.html#profile-pipeline-20260912).

### Original Walker2D continuous-action adaptation

Latest experiment: [September 2026 profile-latency results](index.html#profile-pipeline-20260912). Earlier L0/L2 results below refer to historical experiments with their own evaluation conditions.

Open [`index.html`](index.html) for the Walker2D adaptation record. The task
uses the generic `env.name: gymnasium` path and preserves the native
six-dimensional continuous torque space.

The zero-latency APPO run using the vendored Sample Factory MuJoCo recipe
completed 10,006,528 frames on H200. Its best checkpoint reached a training
statistic of 4,212.219 and was evaluated for ten episodes at each of latency 0
through 4. L0 achieved 3,743.507 mean return, while one or more frames of
latency caused the zero-latency policy to fail quickly. The six native torque
values remain unchanged through OpenVLA training and evaluation. With online
observations aligned to the model's saved image size, the 5k continuous-projector
QwenVLA checkpoint reaches 983.915 mean return and 469.5 mean steps at L0; 4 of
20 episodes reach the full 1,000-step horizon. The report uses this post-fix L0
result directly and does not retain the superseded evaluation comparison.

## September 12 full pipeline

Completed at 10 / 10 FPS. Matched 20-seed realtime return improves from **103.650 ± 4.904** to **857.437 ± 971.038** (8.27×; 17/20 seeds improve). Final mean length is 400.25 steps, with no full 1,000-step episodes. Simulated IID teacher evaluation is separate: 4102.871 ± 593.643 across 10 episodes.

All required data/model uploads are hash-verified. The 100-episode dataset has 96,730 rows; one tiny probe/replay return difference is explicitly documented. See the [report](index.html#profile-pipeline-20260912), [run ledger](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/walker2d/profile_latency/provenance/pipeline/status.md), and [paired results](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/walker2d/profile_latency/provenance/pipeline/paired_B_F.json). One training seed; no commit or push performed.

## Standard Pipeline archive

[Complete experiment archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/walker2d) · [Verification receipt](assets/standard_pipeline.json). This local directory retains report display media; original records are linked at a fixed revision. Only raw demonstrations are distributed, with conversion performed on the training server.


[OpenVLA per-seed real-time comparison](openvla-paired-returns.svg) · [Plot data and fixed-revision sources](openvla-paired-results.json). 17 wins, 0 ties, 3 losses; mean paired gain +753.79.


The HTML tables follow the AirRaid schema: B/F mean gain and percentage; separate L0/B/E/F evaluation rows with episode counts and mean raw steps; measured runtime audit; pipeline stages; publication index. [Normalized table values and provenance](report-tables.json). Unrun stages and unrecorded fields are explicitly distinct.
