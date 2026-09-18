# Humanoid — GR00T H1 collection-budget stop

## Historical L0 teacher result — corrected 2026-09-18

**6331.47 ± 407.20**, 20 episodes; population SD (`ddof=0`). This is the original APPO teacher evaluation, not a new 5 FPS evaluation. The [formal September 13 evaluation](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/humanoid/zero_latency/small_model/evaluation/small_model_l0_eval_20260913/summary.json) supersedes the earlier **5571.30** threshold-probe mean in summary tables. The probe remains a separate historical cohort in the HTML report.


[Open the report](index.html#gr00t-results). Audited 18 September 2026.

The profile branch stopped at **16 accepted specs / 120 total attempts** (104 rejected), short of the required 100. The ceiling includes the original 12-attempt probe; strict teacher return **>5000** was unchanged. Exporter exit 1 at **2026-09-18 00:27:37 UTC** records the expected insufficient-data stop.

**Replay never started.** The 16 specs are admission/seed records, not replayed demonstrations. The 104 rejected attempts have counter evidence; their individual returns were not persisted or reconstructed. No profile raw parquet, profile VLA or F was created. There is no current B/F comparison.

| Stage | Count | Return mean ± population SD | Result |
|---|---:|---:|---|
| L0 VLA, zero simulation | 20 | 6225.77 ± 1077.26 | Length 920.4; completed |
| B, RTX 3090 measured | 20 | 1158.63 ± 646.19 | Length 220.65; completed |
| Profile teacher Final | 20 | 2784.74 ± 1596.81 | 3/20 >5000 |
| Profile teacher best, selected | 20 | 3036.04 ± 1802.44 | 4/20 >5000 |
| Teacher E10 | 10 | 2883.10 ± 1804.11 | 2/10 >5000; repeats best's first ten seeds |
| Initial probe | 12 attempts | — | 3 accepted specs |
| Full admission budget | 120 total attempts | — | 16 accepted specs / 104 rejected |
| Replay / profile VLA / F | Not run | — | Collection target unmet |

The >5000 gate filters teacher demonstrations, not VLA acceptance. Episode statistics from one training run are not independent training-seed results.

## Protocol and evidence

- Native Humanoid-v4 physics; 376D state, 17 native torques ±0.4, RGB224, H1, env/obs 5 FPS. L0 fresh QwenGR00T / DiT-B: 5000 updates, seed 42, micro 16 × accumulation 4 = global 64; 4 inference timesteps, 8 diffusion repeats, train-only state min-max once.
- P: 5 × 10 episodes / cap 900 / no FPS calibration; 9660 admitted observations, zero capacity drops; observation-to-action mean 118.37880 ms, P95 125.04214 ms.
- B: seeds 42–61 / cap 1000 / measured-only; mean of episode latency means 117.25731 ms; zero observation drops, action drops or invalid actions. Across 4,413 raw steps, 4,393 within-episode intervals average 200.00292 ms (198.94464–201.12287). Commands in logs may precede native torque clipping.
- Fresh original APPO profile teacher: seed 2222, 10M target (Final 10,002,432 frames / step 19,536); selected best 9,875,456 frames / step 19,288. Selection uses 20 seeds 104771–104790 per checkpoint, higher mean/tieFinal. E10 reuses 104771–104780 exactly.
- The continuation preserved the initial probe's accepted specs, reset seeds and counters, adjusting only split labels to the 100-episode plan. It did not increase the 120-attempt budget or lower the strict gate.

[Final dataset evidence](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/a260732e0bce9771ca62a3c336bb446ecd0ea5df/humanoid/profile_latency/GR00T/h1_5fps_20260917) · [Teacher checkpoints](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/9379b6c0cd6c2c95dd8a176130c12b4a608f4d92/humanoid/profile_latency/small_model/GR00T) · [L0 model / simulation](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/7aa5afb6f20d52509849e8d8156b5d8583ea288f/humanoid/zero_latency/GR00T) · [Teacher plot, full size](gr00t-teacher-returns.svg) · [Results JSON](results-summary.json) · [Publication index](gr00t-publication.json).

Teacher model revision `9379b6c0cd6c2c95dd8a176130c12b4a608f4d92`; final small-evidence revision `a260732e0bce9771ca62a3c336bb446ecd0ea5df`. These publications contain evidence of the stop, not a new replayed dataset. Full B traces remain server-side; compact B episode/config/audit evidence is published.

## Historical OpenVLA and terminated H8 record

The original OpenVLA record is preserved in [the report](index.html#humanoid-profile-20260913). Its 10 FPS profile branch stopped after teacher quality assessment. The old GR00T H8 / 15 FPS experiment is **TERMINATED / SUPERSEDED**; the following dated notes preserve earlier measurements and launch-time statuses only. They are not the current H1 state and do not support a controlled architecture ranking.

<details><summary>Preserved historical notes (including terminated H8)</summary>

### Original Humanoid adaptation

Latest completed profile-latency experiment: [September 2026 results](index.html#humanoid-profile-20260913). Earlier L0/L2 results below refer to historical experiments with their own evaluation conditions.

## September 17 GR00T native H8 pipeline

This historical, terminated experiment used native Humanoid-v4, 376D state, 17D continuous
actions, H8 chunks, RGB224, and 15/15 benchmark FPS. Native physics, rewards,
termination, and the 1000-step limit are unchanged. The profile arm is IID;
the results and pinned publication receipts below preserve this terminated arm.
They do not describe current launches.

L0 FastTD3 training exited successfully at 10,000,128 frames. Final and
training-best checkpoints were evaluated separately on seeds 42–61:

| Candidate | Episodes | Mean return ± population SD | Episodes strictly above 5000 |
| --- | --- | --- | --- |
| Final, selected | 20 | 5348.253 ± 95.395 | 20/20 |
| Training-best | 20 | 3978.039 ± 2079.640 | 11/20 |

The selected final checkpoint is `checkpoint_000156147_10000128.pth`, SHA256
`8fe80d4e7b0e44c88ed7c6dd236cf33b9052152be23b93b2279eed23192f7b75`.
Its 12-episode exporter probe passed 12/12, with returns 5147.57–5456.45
and length 1000 throughout. Formal collection accepted 100 episodes in 102
attempts, within the 120-attempt budget and unchanged strict >5000 gate.
Accepted returns were 5347.342 ± 93.578 (population SD), range
5116.418–5492.126, with length 1000 throughout. Seeds 33 and 76 were rejected
(returns 313.685 and 684.677; lengths 56 and 123). The 90/10 episode split
contains 89,370 training and 9,930 validation H8 windows.

The first conversion failed while reading the 2.54 GB nested image column.
The shared reader now uses [Parquet batches](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html#pyarrow.parquet.ParquetFile.iter_batches),
with 36 local tests passing and 5 optional-dependency tests skipped. The v8
server retry and actual train/validation loaders passed: action `[8,17]`,
state `[1,376]`, RGB224, all-true masks, and identical train-derived
normalization (SHA256 `b85b959389b3e679b4d307f92b1f1e490bf9feb06ca4692f03de77d8f4810beb`).
The raw data remains unchanged. An 8-H200 smoke failed with a second process
occupying 10.75 GiB; its ownership is unknown. On the newly available single
H200, micro16 × accumulation4 completed 20 updates and strict save/reload.
Spawned validation workers failed separately. The actual trainer path then
passed 200 validation batches/3200 samples with explicit `eval_num_workers=0`;
the underlying IPC unlink cause remains unproven.

The v9 reader preserves numeric vectors directly while retaining nested H8
action handling. Six server semantic tests passed; selected real train/val
samples matched bitwise, including dtypes and images. Sixteen sequential
sample reads took 8.3581 seconds before and 1.1760 seconds after; this measures
item access, not total training throughput.

Fresh v9 formal training started on September17 at00:52:34UTC: 5000 updates,
seed42, global64, original learning rates/warmup, trainworkers4, and explicit
evalworkers0 with validation every500 updates for200 batches. The
[W&B run](https://wandb.ai/dongqianyu99-zhejiang-university/latency-sensitive-bench/runs/humanoid-zero-gr00t-h8-v9-host75-20260917)
and [published preflight/launch evidence](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/38219a6da33c2873e77ddb1bc06226bd0bb388a3/humanoid/zero_latency/GR00T/native_h8_15fps_20260917/vla)
record the exact configuration and retained failures. Training completion,
closed-loop acceptance, and P/B/E/F remain pending.

The [selected teacher bundle](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/a63adee9284c1295df8e2be002f55e06a8fd5a2c/humanoid/zero_latency/small_model/GR00T)
and [training, evaluation, and probe evidence](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/4eafdd783eacb44b798371a2176e9518c97a7a63/humanoid/zero_latency/GR00T/native_h8_15fps_20260917)
are published at fixed revisions. All 21 files matched remote sizes and
Git blob SHA1/LFS SHA256. These revisions contain no formal demonstrations;
the [raw dataset](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/90f36e651dc6a021de01d783618660388821c7f8/humanoid/zero_latency/GR00T/native_h8_15fps_20260917/demonstrations/raw)
is published separately: 105 files, 3,974,064,100 bytes, with all remote sizes
and blob/LFS hashes verified. The [latest experiment and conversion evidence](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/2b5da32db2904e7539a560fb1f109bb6e4870894/humanoid/zero_latency/GR00T/native_h8_15fps_20260917)
records completed conversion and loader validation; converted dataset bulk
stays on the training server.

Open [`index.html`](index.html) for the completed Humanoid-v4 native 17D continuous-action and 376D RGB+state adaptation. The report covers the 10M-frame APPO teacher, 100 gated rollout episodes, 5k QwenOFT run, and the post-fix final-checkpoint L0 evaluation. With the saved 224×224 image contract restored, the VLA still terminates within 66 steps in all 20 episodes.

## September 13 profile-latency pipeline

[Final run ledger](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/humanoid/profile_latency/provenance/pipeline/status.md): **terminated after teacher quality assessment**, at 10 / 10 FPS. The 10M-frame teacher completed normally. Fixed-seed evaluation averaged 1,036.93 return and 196.9 steps; the bounded exporter probe averaged 1,275.39 return and 244.42 steps. All episodes ended early and reward was dominated by the alive bonus. No lower gate was selected; R/C/V/F were not started. Profiling and the complete small model are published and hash-verified; diagnostic evidence and sampling limitations are retained in the report. The monitor was deleted.

## Standard Pipeline archive

[Complete experiment archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/humanoid) · [Verification receipt](assets/standard_pipeline.json). This local directory retains report display media; original records are linked at a fixed revision. Only raw demonstrations are distributed, with conversion performed on the training server.

</details>


[OpenVLA per-seed real-time comparison](openvla-paired-returns.svg) · [Plot data and fixed-revision sources](openvla-paired-results.json). Only B is available. Profile teacher quality blocked F; no paired difference can be computed.


The HTML tables follow the AirRaid schema: B/F mean gain and percentage; separate L0/B/E/F evaluation rows with episode counts and mean raw steps; measured runtime audit; pipeline stages; publication index. [Normalized table values and provenance](report-tables.json). Unrun stages and unrecorded fields are explicitly distinct.
