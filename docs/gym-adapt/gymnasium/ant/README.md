# Ant

[HTML report](index.html) · [All environments](../index.html) · [Results JSON](results.json)

| Model | Profile mean / P95 (ms) | L0 teacher → L0 | L0 VLA → L0 | L0 VLA → real-time (B) | Profile teacher → profile (E10) | Profile VLA → real-time (F) |
| --- | --- | --- | --- | --- | --- | --- |
| GR00T · Completed | 111.64 / 118.45 ms | 5,011.32 ± 1,880.32 | 6,222.60 ± 133.92 | 165.37 ± 224.74 | 3,199.56 ± 317.95 | 1,955.97 ± 780.51 |
| OpenVLA · Completed | 90.56 / 92.13 ms | 5,011.32 ± 1,880.32 | 5,565.29 ± 1,351.05 | 194.33 ± 259.05 | 2,301.03 ± 37.60 | 1,853.33 ± 508.69 |
| Pi0.5 · Completed | 145.42 / 151.63 ms | 5,011.32 ± 1,880.32 | 5,782.50 ± 963.07 | 131.60 ± 93.72 | 4,497.52 ± 71.51 | 1,774.71 ± 824.61 |

Returns are mean ± population standard deviation across episodes. The profile column reports latency mean / P95 in milliseconds. L0 is simulated zero latency; B/F use measured real-time inference latency. E10 is the ten-episode simulated profile-teacher evaluation, not selection20; repeated selection seeds are disclosed per model. Source L0 teachers retain their stated clocks. A dash means unavailable or not run; the model status explains which. One training run and separately realized delays do not establish controlled architecture rankings or equal-latency causal effects. Teacher data gates are not VLA acceptance thresholds.

## GR00T

Completed

### Protocol

Ant-v4, state 27/native action 8, RGB224, H1; env/obs5/5 FPS; L0 simulated zero, B/F RTX3090 measured-only, n 20 each, seeds 42–61, cap 1000, one worker, hold/FIFO, prompt 0/1, no IID. P 5×10 episodes, cap 900, 2655 samples. E10 simulated IID repeats seeds 104771–104780.

### Training

Fresh QwenGR00T L0/profile 5000 updates, seed 42, global 64; profile 2×micro 8×accum 4, DiT-B, 4 inference timesteps/8 diffusion repeats, LR 1e-5/1e-4, warmup 100. APPO seed 2222, target 10M/actual 10,002,432; Final 20 3025.65 ±758.47 beats best 2982.28 ±814.55.

### Data

L0 shared APPO 100 episodes, 88,667/10,000 rows; clock/prompt metadata20→5 FPS. Profile100 accepted/109 attempts including probe12, ceiling120, strict>3000; 90/10 episodes, 90,000/10,000 rows, train-only min-max once. E10 nine of ten above3000.

### Evaluation scope

- One training run per model; episode population SD is not uncertainty across training seeds. Different clocks, training recipes and measured latency distributions preclude controlled architecture ranking or equal-delay causal claims.
- Teacher E10 repeats the first ten selection seeds (104771–104780); it is not independent validation.
- L0 teacher is a shared historical 10-episode evaluation, configured seed 2222, at the original 20/20 FPS clock; it is not a new model-specific teacher evaluation.
- Full B/F traces audited on the evaluation server; published evidence is compact. E10 mean episode length is not recorded and is not inferred.

### Models, data and evidence

- [Profile data and final compact evidence](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/f23f353a9c6b3efaa4407d32134fda8fa30fe78e/ant/profile_latency/GR00T/h1_5fps_20260917)

- [Final model](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/43f38eff5728642c0bb4dcaf844d15781fb47e68/ant/profile_latency/GR00T)

- [Historical shared L0 teacher](assets/evidence/queue_eval_latency_0.json)

- [Detailed experiment report](index.html)

- [L0 VLA checkpoint](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/1a249d62d641428a6bd8b784d8f025655ceb5fb6/ant/zero_latency/GR00T)

## OpenVLA

Completed

### Protocol

QwenOFT; native state 27/action 8, RGB224, H1. Historical L0 original 20/20 FPS, n 20; B/F env/obs10/10 FPS, RTX3090 measured-only, n 20, seeds 42–61; native 1000-step TimeLimit (outer cap 3600). E10 simulated IID, one-worker occupancy. P 5×10 episodes, 7976 admitted/8270 attempted observations. P sampling cap: 60 seconds, raw-step cap 600.

### Training

QwenOFT L0/profile 5000 updates; original APPO teacher seed 2222, 10M-frame recipe. Profile teacher lineage preserved after explicit gate amendment; final VLA bundle22 files verified.

### Data

L0 strict>3000, 100 accepted/108 attempts, 90/10 episodes, 88,667/10,000 rows. Profile E0/10 and probe0/12 passed original>3000 gate; user explicitly amended collection to>2500, then100 accepted demonstrations/100,000 rows passed actual replay audit.

### Evaluation scope

- One training run per model; episode population SD is not uncertainty across training seeds. Different clocks, training recipes and measured latency distributions preclude controlled architecture ranking or equal-delay causal claims.
- L0 teacher is a shared historical 10-episode evaluation, configured seed 2222, at the original 20/20 FPS clock; it is not a new model-specific teacher evaluation.
- Historical L0 uses20 FPS while B/F use10 FPS. Historical E10 follows training-best selection; paired selection20 seed overlap is not established.
- Profile data uses the authorized>2500 gate rather than original>3000; E10 and initial probe remain failed at the original gate.
- F dropped20/18,137 observations; no invalid/dropped actions.

### Models, data and evidence

- [Complete fixed experiment archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/ant)

- [Completed pipeline and gate amendment](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/ant/profile_latency/provenance/pipeline/status.md)

- [Historical shared L0 teacher](assets/evidence/queue_eval_latency_0.json)

- [Detailed experiment report](index.html)

- [L0 VLA checkpoint](https://huggingface.co/latency-sensitive-bench/extra-envs-checkpoints/tree/98647be31ed0eac5743d6348eba25e85c4978f06/ant/zero_latency/OpenVLA)

- [Profile-trained VLA checkpoint](https://huggingface.co/latency-sensitive-bench/extra-envs-checkpoints/tree/98647be31ed0eac5743d6348eba25e85c4978f06/ant/profile_latency/OpenVLA)

## Pi0.5

Completed

### Protocol

Native state 27/action 8, RGB224, H1; env/obs5/5 FPS; L0 simulated zero, B/F RTX3090 measured-only sync_cuda, n 20 each, seeds 42–61, cap 1000, one worker, hold/FIFO, prompt 0/1, no IID. P 5×10 episodes. Teacher E10 simulated IID repeats first 10 selection seeds. P raw-step cap: 900.

### Training

Fresh QwenPI_v3 L0/profile 5000 updates, seed 42, global 128=2×micro 32×accum 2, GC disabled, ZeRO-2; 4 inference timesteps/2 diffusion repeats, no Bridge initialization. APPO seed 2222, target 10M/actual 10,002,432; final/best 20 tie4474.69 ±71.06 selects Final.

### Data

Profile probe12/12,100 accepted/101 attempts, strict>3000, 90/10 episodes, 89,808/9,757 rows; continuous projector, train-only min-max once; dataset revisionbc8880117681c51eae3a38747b1976dba0a89fe2.

### Evaluation scope

- One training run per model; episode population SD is not uncertainty across training seeds. Different clocks, training recipes and measured latency distributions preclude controlled architecture ranking or equal-delay causal claims.
- Teacher E10 repeats the first ten selection seeds (104771–104780); it is not independent validation.
- L0 teacher is a shared historical 10-episode evaluation, configured seed 2222, at the original 20/20 FPS clock; it is not a new model-specific teacher evaluation.
- B combines18 episodes plus2 after a server interruption; session/RNG reset remains disclosed.
- Model publication used CPU state validation and does not establish a saved-model GPU forward. F 20 full-step audit passed with zero invalid/dropped actions or observations.

### Models, data and evidence

- [Reconciled metrics and evidence map](assets/evidence/pi05-pipeline-summary-20260922.md)

- [Recovered B full audit](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/f7107e354eca862df087362307f0d574c2490349/ant/profile_latency/Pi05/h1_5fps_g128_20260920/baseline/validation/B-recovered-audit.json)

- [F20 full audit](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/591db41843567bf4a1946ddfe66d2f51aaf8d7c6/ant/profile_latency/Pi05/h1_5fps_g128_20260920/vla/evaluation/validation/F-verification.json)

- [Teacher verification](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/ff90f1b6c46a6eaae55bbf341762f03a2be6e013/ant/profile_latency/Pi05/h1_5fps_g128_20260920/small_model/evaluation/verification.json)

- [Profile checkpoint](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/20f7452df3e281b6aea0f5a66a1602eb70297490/ant/profile_latency/Pi05)

- [Historical shared L0 teacher](assets/evidence/queue_eval_latency_0.json)

- [Detailed experiment report](index.html)

- [L0 VLA checkpoint](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/95dedf8583692024daa890fe3499b5540167eb77/ant/zero_latency/Pi05)
