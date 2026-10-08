# Walker2D

[HTML report](index.html) · [All environments](../index.html) · [Results JSON](results.json)

| Model | Profile mean / P95 (ms) | L0 teacher → L0 | L0 VLA → L0 | L0 VLA → real-time (B) | Profile teacher → profile (E10) | Profile VLA → real-time (F) |
| --- | --- | --- | --- | --- | --- | --- |
| GR00T · Completed | 111.29 / 118.32 ms | 3,743.51 ± 996.07 | 3,527.99 ± 945.27 | 63.24 ± 6.58 | 4,228.54 ± 14.59 | 852.87 ± 357.10 |
| OpenVLA · Completed | 70.50 / 72.07 ms | 3,743.51 ± 996.07 | 983.91 ± 373.09 | 103.65 ± 4.90 | 4,102.87 ± 593.64 | 857.44 ± 971.04 |
| Pi0.5 · Completed | 188.57 / 209.44 ms | 3,743.51 ± 996.07 | 3,631.42 ± 775.19 | 80.95 ± 10.98 | 3,305.17 ± 1,032.90 | 371.09 ± 123.54 |

Returns are mean ± population standard deviation across episodes. The profile column reports latency mean / P95 in milliseconds. L0 is simulated zero latency; B/F use measured real-time inference latency. E10 is the ten-episode simulated profile-teacher evaluation, not selection20; repeated selection seeds are disclosed per model. Source L0 teachers retain their stated clocks. A dash means unavailable or not run; the model status explains which. One training run and separately realized delays do not establish controlled architecture rankings or equal-latency causal effects. Teacher data gates are not VLA acceptance thresholds.

## GR00T

Completed

### Protocol

VLA 5/5 FPS, H1, RGB224, native state/action 17 / 6; L0/B/F 20 episodes, seeds 42–61, cap 1000. B/F RTX 3090 measured-only, one worker. P 5×10 episodes. E10 simulated IID; seeds 104771–104780 repeat first ten selection seeds. Historical L0 teacher 125/125 FPS, n=10. P raw-step cap: 900.

### Training

Fresh VLA 5,000 updates, seed 42, global 64; DiT-B, 4 inference timesteps / 8 diffusion repeats. Profile micro 32 × accumulation 1 × two GPUs. Fresh APPO 10M, seed 0; training-best selected at step 14536 / 7442432 frames.

### Data

L0 reused 100 source episodes, 90/10 split, 85,160/9,664 rows; clock metadata 125→5 only. Profile strict >3000:100 accepted /100 attempts; 90,000/10,000 rows.

### Evaluation scope

- L0/P/B/E/F audited; compact evidence published.
- Returns are mean ± population SD across episodes from one training run; no controlled architecture ranking or equal-latency causal claim.
- L0 teacher is the shared source teacher at its original clock, not reevaluated at the current VLA clock.
- Teacher return gates apply to demonstrations, not VLA acceptance. Native clipping is preserved; logged actions can be preclip commands.
- E10 repeats selection seeds; it is not independent validation.
- Full runtime traces were audited server-side; compact publication does not contain every raw trace.

### Models, data and evidence

- [Pinned GR00T evidence](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/12528865f7815edb7a351a6bf8c1c317467e5919/experiments/benchmark-training/archive/standard-pipeline/walker2d/profile_latency/GR00T/h1_5fps_20260917)

- [teacher archive](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/12528865f7815edb7a351a6bf8c1c317467e5919/experiments/benchmark-training/archive/standard-pipeline/versions/7d088338043f96511df41c74bc540e62d27aa417/walker2d/zero_latency/small_model)

- [Profile-trained VLA checkpoint](https://huggingface.co/MLL-Lab/LAGEN-models/tree/e3fea81529fd98b5d5782db38803fe4062215dda/latency-aware/walker2d/vla/starvla-qwengr00t-h1/walker2d_profile_gr00t_h1_5fps_2h200_20260917)

- [L0 VLA checkpoint](https://huggingface.co/MLL-Lab/LAGEN-models/tree/e3fea81529fd98b5d5782db38803fe4062215dda/zero-latency/walker2d/vla/starvla-qwengr00t-h1/walker2d_l0_gr00t_h1_5fps_2h200_20260917)

## OpenVLA

Completed

### Protocol

B/E/F 10/10 FPS; B/F 20 episodes, seeds 42–61, cap 1000, measured-only RTX 3090; E10 simulated IID, one worker. H1, RGB224, state/action 17 / 6. Historical L0 teacher 125/125 FPS, n=10; historical L0 VLA 20 episodes at 125/125 FPS.

### Training

QwenOFT / Qwen3-VL-4B, continuous state projector, native action regression, L1, 5,000 updates, seed 42. L0 and profile globalbatch 16; fresh profile APPO 10M completed 10006528 frames.

### Data

L0 reused 100 episodes. Profile strict>3000 accepted 100; 96,730 rows, 87,052/9,678 split.

### Evaluation scope

- Historical L0/P/B/E/F complete and published.
- Returns are mean ± population SD across episodes from one training run; no controlled architecture ranking or equal-latency causal claim.
- L0 teacher is the shared source teacher at its original clock, not reevaluated at the current VLA clock.
- Teacher return gates apply to demonstrations, not VLA acceptance. Native clipping is preserved; logged actions can be preclip commands.
- OpenVLA is this project’s QwenOFT implementation, not an architecture-matched comparison with GR00T/Pi0.5.
- Profile replay episode 99 had a small documented return mismatch; seeds, split and lengths matched the probe. The pinned archive retains the audit.
- Historical E10 is the simulated profile-teacher evaluation; overlap with selection seeds is not established in local evidence.

### Models, data and evidence

- [Pinned complete historical archive](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/12528865f7815edb7a351a6bf8c1c317467e5919/experiments/benchmark-training/archive/standard-pipeline/versions/7d088338043f96511df41c74bc540e62d27aa417/walker2d)

- [Profile teacher E10](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/blob/12528865f7815edb7a351a6bf8c1c317467e5919/experiments/benchmark-training/archive/standard-pipeline/versions/7d088338043f96511df41c74bc540e62d27aa417/walker2d/profile_latency/small_model/evaluation/summary.json)

- [L0 VLA checkpoint](https://huggingface.co/latency-sensitive-bench/extra-envs-checkpoints/tree/98647be31ed0eac5743d6348eba25e85c4978f06/walker2d/zero_latency/OpenVLA)

- [Profile-trained VLA checkpoint](https://huggingface.co/latency-sensitive-bench/extra-envs-checkpoints/tree/98647be31ed0eac5743d6348eba25e85c4978f06/walker2d/profile_latency/OpenVLA)

## Pi0.5

Completed

### Protocol

5/5 FPS, H1, RGB224, native state/action 17 / 6; L0/B/F 20 episodes, seeds 42–61, cap 1000. B/F measured-only CUDA-synchronized latency, one worker, no added IID. P 5×10 episodes, 100 warmups/session, cap 900 and 60-second sampling cap. Historical L0 teacher 125/125 FPS, n=10.

### Training

Fresh QwenPI_v3, 5,000 updates, seed 42, global 128, GC off, ZeRO-2, 4 inference timesteps/2 diffusion repeats. L0 micro 64 × accumulation 1 × two GPUs; profile approved micro 32 × accumulation 2 × two GPUs after micro 64 OOM. APPO 10M; training-best step 18928 /9691136 selected.

### Data

L0 source 90/10 episodes, 85,160/9,664 rows; prompt/clock 125→5 only. Profile >3000:100/118 accepted, 90/10 episodes, 88,708/10,000 rows.

### Evaluation scope

- L0/P/B/E/F audited; F model/evidence published; B locally audited, not uploaded.
- Returns are mean ± population SD across episodes from one training run; no controlled architecture ranking or equal-latency causal claim.
- L0 teacher is the shared source teacher at its original clock, not reevaluated at the current VLA clock.
- Teacher return gates apply to demonstrations, not VLA acceptance. Native clipping is preserved; logged actions can be preclip commands.
- E10 seeds 104771–104780 repeat first ten selection seeds 104771–104790; not independent validation.
- B/F gain+290.14; 19 wins, 0 ties, 1 loss. Separate sessions and changed co-residency; action-weighted latency183.79→196.25 ms.
- B evidence is local/server-only in this update; publication is separate from audit.
- State provenance is verified through manifest/loader/saved reload; no per-step state vectors were logged.
- P uses a 60-second sampling cap per episode, excluding initialization/reset/warmup; this does not change B/F or teacher caps.
- P: all five warmup sessions stable; 500 warmup samples excluded, 2,977 admitted samples and 310 capacity drops.

### Models, data and evidence

- [Current B full-step audit](assets/evidence/B-verification.json)

- [Pi0.5 stage summary](assets/evidence/pi05-pipeline-summary-20260922.md)

- [L0 pinned model](https://huggingface.co/MLL-Lab/LAGEN-models/tree/e3fea81529fd98b5d5782db38803fe4062215dda/zero-latency/walker2d/vla/starvla-qwenpi_v3-h1/walker2d_pi05_l0_h1_5fps_g128_20260921)

- [teacher archive](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/12528865f7815edb7a351a6bf8c1c317467e5919/experiments/benchmark-training/archive/standard-pipeline/versions/7d088338043f96511df41c74bc540e62d27aa417/walker2d/zero_latency/small_model)

- [E10 selected teacher](assets/evidence/E10-summary.json)

- [F full-step audit](assets/evidence/F-verification.json)

- [Pinned F model](https://huggingface.co/MLL-Lab/LAGEN-models/tree/e3fea81529fd98b5d5782db38803fe4062215dda/latency-aware/walker2d/vla/starvla-qwenpi_v3-h1/walker2d_pi05_profile_h1_5fps_g128_20260922_m32)

- [Published F evidence](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/12528865f7815edb7a351a6bf8c1c317467e5919/experiments/benchmark-training/archive/standard-pipeline/walker2d/profile_latency/Pi05/h1_5fps_g128_20260921)

- [Approved microbatch32 configuration](assets/evidence/vla-m32-preparation.json)
