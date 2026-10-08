# Hopper

[HTML report](index.html) · [All environments](../index.html) · [Results JSON](results.json)

| Model | Profile mean / P95 (ms) | L0 teacher → L0 | L0 VLA → L0 | L0 VLA → real-time (B) | Profile teacher → profile (E10) | Profile VLA → real-time (F) |
| --- | --- | --- | --- | --- | --- | --- |
| GR00T · Completed | 114.62 / 122.61 ms | 2,682.09 ± 900.29 | 872.83 ± 479.65 | 8.55 ± 0.09 | 3,814.84 ± 8.80 | 242.55 ± 127.57 |
| OpenVLA · Completed | 71.63 / 74.11 ms | 2,682.09 ± 900.29 | 1,419.93 ± 552.54 | 67.88 ± 150.09 | 3,657.61 ± 13.88 | 424.58 ± 348.01 |
| Pi0.5 · Completed | 199.10 / 264.06 ms | 2,682.09 ± 900.29 | 3,482.49 ± 290.80 | 8.40 ± 0.66 | 2,616.17 ± 629.87 | 235.17 ± 28.38 |

Returns are mean ± population standard deviation across episodes. The profile column reports latency mean / P95 in milliseconds. L0 is simulated zero latency; B/F use measured real-time inference latency. E10 is the ten-episode simulated profile-teacher evaluation, not selection20; repeated selection seeds are disclosed per model. Source L0 teachers retain their stated clocks. A dash means unavailable or not run; the model status explains which. One training run and separately realized delays do not establish controlled architecture rankings or equal-latency causal effects. Teacher data gates are not VLA acceptance thresholds.

## GR00T

Completed

### Protocol

H1; RGB224 + native continuous state/actions; env/obs 5/5 FPS; L0 simulated zero latency and B/F RTX 3090 measured-only CUDA-synchronized latency, one worker; n=20, seeds 42–61, cap 1000. P: 5×10 episodes, seeds 42–51 each session, cap 900, no FPS calibration. State 11/action 3. E10 n=10, seeds 104771–104780, repeats first 10 selection seeds. Historical L0 teacher n=10, original 125 FPS, seed 3333.

### Training

QwenGR00T, 5000 updates, seed 42, global batch 64; DiT-B, 4 inference timesteps, 8 diffusion repeats, VLM/action LR 1e-5/1e-4, warmup 100, held-out 200 batches and saves every 500. Profile batch 2×micro 8×acc 4. Fresh APPO profile teacher: original 10M recipe, seed 3333; selected training-best selection 20 3815.46 ± 7.63.

### Data

L0: original 100 APPO H1 episodes, 90/10 split, 88212/9917 rows; metadata125→5 FPS. Profile strict return>3000:100 accepted/101 attempts; 90000/10000 rows, 90/10 episodes.

### Evaluation scope

- Completed: L0/P/B/profile teacher/data/VLA/F; final model and compact evidence published.
- One training run; population SD describes episode variation. Different clocks, recipes and realized latency distributions do not establish controlled architecture rankings or equal-latency causal effects.
- E10 repeats selection seeds and is not independent validation.
- B→F +234.00; 20/20 seeds improve, but F mean episode length100.1 remains far below cap 1000.
- Full traces audited on evaluation server; published compact evidence does not imply all raw traces uploaded.
- Historical L0 teacher is the original 125 FPS evaluation, not a new5 FPS evaluation.

### Models, data and evidence

- [Final compact evidence](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/12528865f7815edb7a351a6bf8c1c317467e5919/experiments/benchmark-training/archive/standard-pipeline/hopper/profile_latency/GR00T/h1_5fps_20260917)

- [Historical L0 teacher](assets/evidence/queue_eval_latency_0.json)

- [L0 model](https://huggingface.co/MLL-Lab/LAGEN-models/tree/e3fea81529fd98b5d5782db38803fe4062215dda/zero-latency/hopper/vla/starvla-qwengr00t-h1/hopper_l0_gr00t_h1_5fps_2h100_20260917)

- [Profile checkpoint](https://huggingface.co/MLL-Lab/LAGEN-models/tree/e3fea81529fd98b5d5782db38803fe4062215dda/latency-aware/hopper/vla/starvla-qwengr00t-h1/hopper_profile_gr00t_h1_5fps_2h100_20260917)

## OpenVLA

Completed

### Protocol

Historical P/B/E/F at env/obs 10/10 FPS, one worker; B/F measured RTX 3090, n=20, seeds 42–61, native cap 1000 (configured evaluation cap 3600). P: 5×10 episodes, 60 s per episode. E10: simulated IID, n=10. Historical L0 VLA n=20 at original clock. State 11/action 3, continuous-state projector. Historical teacher n=10 at 125 FPS, seed 3333; L0 VLA original 125 FPS.

### Training

QwenOFT continuous-state projector, 5000 steps, seed 42. Profile APPO original 10M recipe, seed 3333, actual 10006528 frames. Profile VLA 5000 steps, held-out L1 0.01595611.

### Data

L0 100 accepted episodes, 90/10 split, 88212/9917 rows. Profile100 episodes, 98817 rows, all strict return>3000; 123 data files and22 inference files verified.

### Evaluation scope

- Completed historical pipeline; 20/20 paired seeds improve.
- One training run; population SD describes episode variation. Different clocks, recipes and realized latency distributions do not establish controlled architecture rankings or equal-latency causal effects.
- E10 is the historical simulated teacher evaluation; selection-seed overlap is not established by local evidence.
- B→F +356.70; 20/20 seeds improve; F mean length139.45 and no episode reaches1000.
- Continuous-projector L0 result is used; earlier text-bin L0 failure173.14 is a separate ablation.
- Historical L0 teacher/VLA at 125 FPS are separate from 10 FPS P/B/E/F.

### Models, data and evidence

- [Complete fixed-revision archive](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/12528865f7815edb7a351a6bf8c1c317467e5919/experiments/benchmark-training/archive/standard-pipeline/versions/7d088338043f96511df41c74bc540e62d27aa417/hopper)

- [Historical L0 teacher](assets/evidence/queue_eval_latency_0.json)

- [L0 VLA checkpoint](https://huggingface.co/latency-sensitive-bench/extra-envs-checkpoints/tree/98647be31ed0eac5743d6348eba25e85c4978f06/hopper/zero_latency/OpenVLA)

- [Profile-trained VLA checkpoint](https://huggingface.co/latency-sensitive-bench/extra-envs-checkpoints/tree/98647be31ed0eac5743d6348eba25e85c4978f06/hopper/profile_latency/OpenVLA)

## Pi0.5

Completed

### Protocol

H1; RGB224 + native continuous state/actions; env/obs 5/5 FPS; L0 simulated zero latency and B/F RTX 3090 measured-only CUDA-synchronized latency, one worker; n=20, seeds 42–61, cap 1000. P: 5×10 episodes, seeds 42–51 each session, cap 900, no FPS calibration. State 11/action 3. E10 n=10, seeds 104771–104780, repeats first 10 selection seeds 104771–104790. Historical L0 teacher n=10 at 125 FPS, seed 3333. P additionally uses a 60 s sampling wall-time cap excluding reset/init/warmup; 100 warmup samples per session, 500 total, excluded from canonical latency samples.

### Training

QwenPI_v3, 5000 updates, seed 42, global batch 128 = micro 64 × accumulation 1 × 2 GPUs; GC off, ZeRO-2; train-only state min-max applied once. L0 NCCL timeout after 3342: restored full model/optimizer step 3000, repaired LR/data cursor, recomputed 3001–3342, completed 5000; main RNG restored after iterator reconstruction. Profile APPO seed 3333, actual 10002432 frames / step 19536; Final selected (selection 20 2473.70 ± 983.60).

### Data

L0 100 source episodes, 90/10 split, 88212/9917 rows, metadata125→5 FPS. Profile original strict>3000 accepted 53/120. Explicit user amendment: strict>2500 with unlimited attempts; 47 further accepted/98 attempts. No old rejection reclassified. Final 100/218 accepted, 90/10 episodes, 87404/9776 rows.

### Evaluation scope

- Completed and audited: L0/P/B/profile teacher/data/VLA/F; F model/evidence published; B audit local.
- One training run; population SD describes episode variation. Different clocks, recipes and realized latency distributions do not establish controlled architecture rankings or equal-latency causal effects.
- E10 repeats first 10 selection seeds and is not independent validation.
- P session 1 warmup drift warning retained; 364 admitted samples, 84 capacity drops, 500 warmup samples excluded.
- Recovery does not claim worker-RNG or future-CUDA bitwise identity.
- B/F full-step audits passed; B audit is local and not uploaded in this update. F mean length108.25 remains below1000; 20/20 seeds improve, mean paired gain226.78.
- State contract verified via manifest/loader/reload; no per-step state vectors. Historical teacher result remains at 125 FPS.

### Models, data and evidence

- [P audit](assets/evidence/P-verification.json)

- [L0 audit](assets/evidence/l0-audit.json)

- [B audit](assets/evidence/B-verification.json)

- [F audit](assets/evidence/F-verification.json)

- [Revised collection and E10](assets/evidence/source-gt2500.json)

- [Recovery lineage](assets/evidence/recovery-completion-verification.json)

- [Published F evidence](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/12528865f7815edb7a351a6bf8c1c317467e5919/experiments/benchmark-training/archive/standard-pipeline/hopper/profile_latency/Pi05/h1_5fps_g128_20260921)

- [Historical L0 teacher](assets/evidence/queue_eval_latency_0.json)

- [L0 checkpoint](https://huggingface.co/MLL-Lab/LAGEN-models/tree/e3fea81529fd98b5d5782db38803fe4062215dda/zero-latency/hopper/vla/starvla-qwenpi_v3-h1/hopper_pi05_l0_h1_5fps_g128_20260921)

- [Profile checkpoint](https://huggingface.co/MLL-Lab/LAGEN-models/tree/e3fea81529fd98b5d5782db38803fe4062215dda/latency-aware/hopper/vla/starvla-qwenpi_v3-h1/hopper_pi05_profile_gt2500_h1_5fps_g128_20260922)
