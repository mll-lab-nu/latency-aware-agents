# AirRaid

[HTML report](index.html) · [All environments](../index.html) · [Results JSON](results.json)

| Model | Profile mean / P95 (ms) | L0 teacher → L0 | L0 VLA → L0 | L0 VLA → real-time (B) | Profile teacher → profile (E10) | Profile VLA → real-time (F) |
| --- | --- | --- | --- | --- | --- | --- |
| GR00T · Completed | 81.49 / 97.82 ms | 2,557.50 ± 491.81 | 2,101.25 ± 667.01 | 733.75 ± 268.71 | 2,572.50 ± 259.43 | 1,586.25 ± 564.83 |
| OpenVLA · Completed | 58.41 / 59.03 ms | 2,557.50 ± 491.81 | 2,000.00 ± 706.13 | 878.75 ± 301.69 | 3,242.50 ± 300.01 | 2,353.75 ± 425.39 |
| Pi0.5 · Completed | 134.55 / 139.89 ms | 2,557.50 ± 491.81 | 1,943.75 ± 776.85 | 1,386.25 ± 434.99 | 3,785.00 ± 384.09 | 3,227.50 ± 1,095.58 |

Returns are mean ± population standard deviation across episodes. The profile column reports latency mean / P95 in milliseconds. L0 is simulated zero latency; B/F use measured real-time inference latency. E10 is the ten-episode simulated profile-teacher evaluation, not selection20; repeated selection seeds are disclosed per model. Source L0 teachers retain their stated clocks. A dash means unavailable or not run; the model status explains which. One training run and separately realized delays do not establish controlled architecture rankings or equal-latency causal effects. Teacher data gates are not VLA acceptance thresholds.

## GR00T

Completed

### Protocol

RGB224, no state, six actions, H1; env/obs50/12.5 FPS, noop30+FireReset, cap 3600 raw frames. L0 simulated zero; B/F RTX3090 measured-only, 20 episodes each, seeds 42–61, one worker, hold/FIFO. P 5×10 episodes, 25,491 canonical samples, 500 warmups excluded. E10 simulated IID; selection-seed overlap not established in compact evidence.

### Training

Fresh QwenGR00T L0/profile, 5000 updates each, seed 42, global 64=micro 16×accum 4; Qwen3-VL-4B-Instruct+DiT-B, 4 inference timesteps/8 diffusion repeats, LR 1e-5/1e-4, warmup 100. APPO seed 0, 10,002,432 steps; selected training-best step 38,304/9,805,824 frames. Saved profile reload and finite [1, 1, 6] forward passed.

### Data

L0 reuses100 original H1 episodes. Profile strict>2500:100 accepted/164 attempts, 90/10 episodes, 81,000/9,000 rows. Canonical first 12 probe7 accepted/5 rejected; 10 overrun attempts excluded. E10 seven of ten above2500.

### Evaluation scope

- One training run per model; episode population SD is not uncertainty across training seeds. Different clocks, training recipes and measured latency distributions preclude controlled architecture ranking or equal-delay causal claims.
- L0 teacher is a shared historical 10-episode evaluation, configured seed 0, random-noop + FireReset at the original 50/12.5 FPS clock; it is not a new model-specific teacher evaluation.
- Profile collection had no enforced attempt cap; canonical probe overrun is retained as a protocol deviation.
- Observation drops B 3023/14131 (21.39%) and F 775/14096 (5.50%); invalid/dropped actions 0.

### Models, data and evidence

- [Complete profile experiment](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/12528865f7815edb7a351a6bf8c1c317467e5919/experiments/benchmark-training/archive/standard-pipeline/air_raid/profile_latency/GR00T/gr00t_h1_iid_20260917)

- [Profile checkpoint](https://huggingface.co/MLL-Lab/LAGEN-models/tree/e3fea81529fd98b5d5782db38803fe4062215dda/latency-aware/air-raid/vla/starvla-qwengr00t-h1/standard-pipeline-2208875f92b2)

- [L0 checkpoint](https://huggingface.co/MLL-Lab/LAGEN-models/tree/e3fea81529fd98b5d5782db38803fe4062215dda/zero-latency/air-raid/vla/starvla-qwengr00t-h1/air_raid_zero_latency_gr00t_h1)

- [Historical shared L0 teacher](assets/evidence/queue_eval_latency_0.json)

- [Detailed experiment report](index.html)

## OpenVLA

Completed

### Protocol

QwenOFT archive label; RGB224/no state/six actions/H1; env/obs50/12.5 FPS, noop30+FireReset, cap 3600 raw frames. L0 simulated zero, n 20; B/F RTX3090 measured-only, n 20, seeds 42–61; E10 simulated IID/one-worker occupancy. P 5×10 episodes, 35,908 admitted samples, zero capacity drops. P sampling cap: 60 seconds, raw-step cap 3000.

### Training

QwenOFT with Qwen3-VL-4B-Instruct; 5000 updates, global 16. Profile APPO 10,006,528 frames, training-best selected. Profile VLA fresh training completed 5000 updates and saved model publication verified.

### Data

Profile100 accepted episodes, strict>2500, 90/10 episode split, 81,000/9,000 rows; actual minimum2575, mean3186.50. E10 all10 above2500. Reset preserves random-noop+FireReset.

### Evaluation scope

- One training run per model; episode population SD is not uncertainty across training seeds. Different clocks, training recipes and measured latency distributions preclude controlled architecture ranking or equal-delay causal claims.
- L0 teacher is a shared historical 10-episode evaluation, configured seed 0, random-noop + FireReset at the original 50/12.5 FPS clock; it is not a new model-specific teacher evaluation.
- Historical E10 follows training-best checkpoint selection; overlap with a paired selection20 cohort is not established.
- B and F each dropped one observation; no invalid/dropped actions.

### Models, data and evidence

- [Canonical complete experiment](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/12528865f7815edb7a351a6bf8c1c317467e5919/experiments/benchmark-training/archive/standard-pipeline/air_raid/profile_latency/OpenVLA/native_20260911)

- [Profile checkpoint](https://huggingface.co/MLL-Lab/LAGEN-models/tree/e3fea81529fd98b5d5782db38803fe4062215dda/latency-aware/air-raid/vla/starvla-qwenoft-h1/air_raid_profile_20260911T054920Z_openvla_native_sft_5k)

- [L0 checkpoint](https://huggingface.co/MLL-Lab/LAGEN-models/tree/e3fea81529fd98b5d5782db38803fe4062215dda/zero-latency/air-raid/vla/starvla-qwenoft-h1/air_raid_l0_random_noop_fire_reset_return_gt2500_100ep_openvla_native_sft_5k)

- [Historical shared L0 teacher](assets/evidence/queue_eval_latency_0.json)

- [Detailed experiment report](index.html)

## Pi0.5

Completed

### Protocol

RGB224/no state/six actions/H1; env/obs10/2.5 FPS, noop30+FireReset, cap 3600 raw frames. L0 simulated zero; B/F RTX3090 measured-only sync_cuda, n 20 each, seeds 42–61, one worker, hold/FIFO, prompt 0/2, no added IID. P 5×10 episodes, 30,865 samples, 500 warmups excluded. E10 repeats first 10 selection seeds. P raw-step cap: 3000.

### Training

Fresh QwenPI_v3 L0/profile 5000 updates, seed 42, global 128=2×micro 64×accum 1, GC disabled, ZeRO-2. APPO seed 0, target 10M counted steps (~40M raw frames at stride 4), actual 10,002,432. Final 20 3895.00 ±316.68 selected over best 3716.25 ±503.50.

### Data

Profile probe12/12; 100 accepted episodes, strict>2500, 90/10 split, 81,000/9,000 rows; prompt 2 and actual replay verified. Teacher gate is not a VLA acceptance threshold.

### Evaluation scope

- One training run per model; episode population SD is not uncertainty across training seeds. Different clocks, training recipes and measured latency distributions preclude controlled architecture ranking or equal-delay causal claims.
- L0 teacher is a shared historical 10-episode evaluation, configured seed 0, random-noop + FireReset at the original 50/12.5 FPS clock; it is not a new model-specific teacher evaluation.
- Teacher E10 repeats the first ten selection seeds (104771–104780); it is not independent validation.
- Original profiling predates the later60-second P policy and was retained unchanged. B/F full audits cover64,470/67,159 steps with zero invalid/dropped actions or observations.

### Models, data and evidence

- [Reconciled metrics and evidence map](assets/evidence/pi05-pipeline-summary-20260922.md)

- [Final B/F comparison](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/blob/12528865f7815edb7a351a6bf8c1c317467e5919/experiments/benchmark-training/archive/standard-pipeline/air_raid/profile_latency/Pi05/h1_10env_2p5obs_g128_20260920/vla/evaluation/comparison.json)

- [Profile model metadata](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/blob/12528865f7815edb7a351a6bf8c1c317467e5919/reports/source-repository-retirement-20261008/model-metadata/Standard-Pipeline/air_raid/profile_latency/Pi05/README.md)

- [Teacher E10 verification](assets/evidence/teacher-verification.json)

- [Historical shared L0 teacher](assets/evidence/queue_eval_latency_0.json)

- [Detailed experiment report](index.html)
