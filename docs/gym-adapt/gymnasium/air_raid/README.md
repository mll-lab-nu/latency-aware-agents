# AirRaid — OpenVLA and GR00T

[Open the experiment report](index.html). Updated 2026-09-17 from completed evaluation receipts and fixed Hugging Face revisions.

## Real-time results

Both zero-latency-trained (B) and profile-latency-trained (F) checkpoints were evaluated on RTX 3090 with measured inference latency, 20 episodes each, seeds 42–61, target env/obs 50/12.5 FPS, RGB224, H1, noop30 + FireReset, and a 3,600-raw-frame cap. AirRaid does not use the embodied tasks’ 5 FPS setting.

| Model / run | B: zero-trained | F: profile-trained | Mean gain |
|---|---:|---:|---:|
| OpenVLA / QwenOFT, Sep 11 | 878.75 ± 301.69 | 2353.75 ± 425.39 | +1475.00 (+167.85%) |
| GR00T / QwenGR00T, Sep 17 | 733.75 ± 268.71 | 1586.25 ± 564.83 | +852.50 (+116.18%) |

Values are mean ± population SD. These are within-model comparisons from one training run, not a controlled ranking of architectures: training recipes and measured latency distributions differ. OpenVLA is this project's archive label for QwenOFT with Qwen3-VL-4B-Instruct.

## GR00T

- L0 training: 5000 updates, seed42, global batch64 (micro16 × accumulation4), completed 00:52:39 UTC. Simulated zero-latency evaluation20: **2101.25 ± 667.01**, mean length3265.25; separate from real-time B.
- P: 5×10 episodes, 25491 canonical samples, 500 warmups excluded; mean81.49ms, P9597.82ms.
- APPO profile teacher: 10,002,432 frames; selected training-best step38,304 / 9,805,824 frames. Simulated IID E10: **2572.50 ± 259.43**, 7/10 above2500.
- Demonstrations: 100 accepted /164 attempts, 90/10 episodes, 81000/9000 rows. Canonical first12 probe:7 accepted/5 rejected;10 overrun attempts excluded. Collection had no enforced attempt cap; archive retains these deviations.
- Fresh profile VLA: 5000 updates, 03:51:49–08:08:37 UTC, exit0; strict reload and finite real-sample forward `[1,1,6]` passed. QwenGR00T DiT-B, 4 inference timesteps, 8 diffusion repeats, global64, warmup100, VLM/action LR1e-5/1e-4. Model training completion is distinct from evaluation.
- F20 completed 09:40:26 UTC, exit0; paired **18 wins /0 ties /2 losses**. No additional VLA acceptance threshold is imposed.
- B/F action-weighted latency:82.19/78.40ms; P95:95.14/85.81ms. Mean of episode latency means:83.29/78.45ms. Observation drops:3023/14131 (21.39%) vs775/14096 (5.50%). Invalid/dropped actions:0/0 in both. Actual env FPS episode means48.29/48.27, distinct from the target50 FPS.

[Zero checkpoint](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/87bb2e4d18b793293f8948c646d2186ed53d5a31/air_raid/zero_latency/GR00T) · [Profile checkpoint](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/06a0d90722e3b1cfd2117a3577e34ed01e4aed74/air_raid/profile_latency/GR00T) · [Complete profile experiment](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/6dfb05ebfc2dd01fbd64076b1b2f8d78898cc3e8/air_raid/profile_latency/GR00T/gr00t_h1_iid_20260917) · [Per-seed B/F comparison](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/6dfb05ebfc2dd01fbd64076b1b2f8d78898cc3e8/air_raid/profile_latency/GR00T/gr00t_h1_iid_20260917/comparison.json).

L0 reuses [100 original H1 episodes](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/483301cd09aabec98e06657a135c5f041515b73e/air_raid/zero_latency/shared/native_100ep_v1/demonstrations/raw). [L0 training archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/0086d5b61dd8ca0f1fdabe563a3884de2eac2c42/air_raid/zero_latency/GR00T/gr00t_h1_5000_20260917) and [L0 simulated evaluation](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/8962cc2e9bf6f3629aae92af829e8cd545835d4b/air_raid/zero_latency/GR00T/gr00t_h1_5000_20260917/vla/evaluation/l0-simulated) are separately pinned.

## OpenVLA

[Canonical September11 archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/571cb5801720488ed6458b9cfe3f4c595b069cea/air_raid/profile_latency/OpenVLA/native_20260911) preserves the completed P/B/S/E/R/C/V/F pipeline. Profile VLA:5000 updates, global16; real-time F20 **2353.75 ±425.39**, all20 at the3600-frame cap. Teacher E10 is simulated IID: **3242.50 ±300.01**. Historical simulated L0/L2 scores (**2000.00 ±706.13 /681.25 ±277.92**) remain in the HTML detailed record and are not the real-time B/F cohort.

## Publication layout

Model weights remain in the model repository. The dataset repository contains training configs/completion receipts, P/B/E/F scientific traces and summaries, warmup records, raw demonstrations/validation and paired comparisons. Converted data are generated on the training server; optimizer states, duplicate inference traces and raw operational logs are omitted.

GR00T final evidence was published at `6dfb05ebfc2dd01fbd64076b1b2f8d78898cc3e8`: **46 files /82,329,005 bytes**, all remote sizes and content hashes verified. [Publication receipt](gr00t-publication.json), [compact results](results-summary.json), [source/output manifest](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/6dfb05ebfc2dd01fbd64076b1b2f8d78898cc3e8/air_raid/profile_latency/GR00T/gr00t_h1_iid_20260917/curation/finalization-manifest.json) and [curation notes](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/6dfb05ebfc2dd01fbd64076b1b2f8d78898cc3e8/air_raid/profile_latency/GR00T/gr00t_h1_iid_20260917/FINALIZATION.md). `SHA256SUMS.finalization` covers this finalization only; earlier manifests describe their historical publication subsets.

The report separates models and evaluation conditions. Original OpenVLA evidence links and [historical archive receipt](assets/standard_pipeline.json) remain available. Only report/display media and compact indices are kept here; bulk original evidence is linked on HF.


[OpenVLA per-seed real-time comparison](openvla-paired-returns.svg) · [Plot data and fixed-revision sources](openvla-paired-results.json). 20 wins, 0 ties, 0 losses; mean paired gain +1,475.00.
