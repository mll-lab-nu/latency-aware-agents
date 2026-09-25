# Ant — completed GR00T measured B/F

[Experiment report](index.html#gr00t-results) · [Full-size paired SVG](gr00t-paired-returns.svg) · [Results JSON](results-summary.json).

GR00T H1 / 5 FPS measured return **165.37 ± 224.74 → 1955.97 ± 780.51**; mean episode length **117.95 → 864.15 raw steps**. Mean paired gain is **+1790.59**, with **19 wins, 0 ties and 1 loss**. Seed 52 declines by 211.50. F returns range from 158.86 to 2579.21. Population SD describes episodes from one training run.

| Measured metric | B | F |
|---|---:|---:|
| Episodes / seeds | 20 / 42–61 | 20 / 42–61 |
| Return ± population SD | 165.37 ± 224.74 | 1955.97 ± 780.51 |
| Mean length (raw steps) | 117.95 | 864.15 |
| Mean of episode latency means (ms) | 109.68601 | 109.09412 |
| Observation / action drops / invalid | 0 / 0 / 0 | 0 / 0 / 0 |
| Audited raw steps | 2,359 | 17,283 |

F has **17,263 within-episode intervals**, averaging **200.000775 ms** (198.892736–201.134116 ms), distinct from its 17,283 raw steps. B/F share measured-only latency, 5 FPS, seeds 42–61, cap 1000, one worker and hold/FIFO scheduling. Only four policy paths, prompt key 0 → 1 and the output directory differ. F has no added IID delay. Similar mean latencies do not establish equal-latency causality.

## Separate training and simulation evidence

- Native Ant-v4: state 27, native action 8, RGB224 and H1; physics and torque clipping unchanged. DiT-B, 4 inference steps / 8 diffusion repeats; train-only state min-max applied once.
- L0 data: original APPO H1, 100 episodes, 88,667 / 10,000 rows. Only prompt and nominal clock metadata changed 20 → 5 FPS. Fresh VLA: 5000 updates, seed 42, global batch 64. Separate L0 simulation: **6222.60 ± 133.92**, mean length 1000 (20 episodes).
- Measured P: 5 × 10 episodes, cap 900, no calibration; 2655 samples. Observation-to-action latency mean 111.64496 ms, P95 118.44892 ms, maximum 137.37081 ms.
- Profile APPO teacher: original seed 2222, 10M target / 10,002,432 actual steps. Final20 **3025.65 ± 758.47** exceeds best20 **2982.28 ± 814.55**; Final selected. E10 **3199.56 ± 317.95**, 9/10 >3000, repeats selection seeds 104771–104780.
- Profile demonstrations: **100 accepted / 109 attempts**, within the 120-attempt ceiling including probe12. Split 90/10 episodes and 90,000 / 10,000 rows; all actual replay returns >3000. This teacher filter is not a VLA acceptance criterion.
- Profile VLA: fresh 5000 updates, seed 42, global batch 64 = 2 × microbatch 8 × accumulation 4. VLM/head LR 1e-5 / 1e-4, warmup 100, held-out 200 batches and save every 500. Saved reload and measured F20 completed.

## Published evidence

[Dataset and final compact evidence](https://huggingface.co/datasets/placeholder/Standard-Pipeline/tree/f23f353a9c6b3efaa4407d32134fda8fa30fe78e/ant/profile_latency/GR00T/h1_5fps_20260917) · [Final model](https://huggingface.co/placeholder/Standard-Pipeline/tree/43f38eff5728642c0bb4dcaf844d15781fb47e68/ant/profile_latency/GR00T) · [Publication receipt](gr00t-publication.json).

Evaluated F revision `bfab5cbb9abbadb03eb39593d3f403471b835882`, checkpoint SHA256 `77dfb363a28a363a3acec7f53892e80c773cd8e2e5850ecff980abf6dbfff18b`. Documentation revision `43f38eff5728642c0bb4dcaf844d15781fb47e68` preserves these weights. Full step/action/latency traces were audited on the evaluation server and remain there; only compact evidence is published.

## Preserved OpenVLA history

Historical OpenVLA / 10 FPS: **194.33 ± 259.05 → 1853.33 ± 508.69**. Different clocks and recipes preclude a controlled architecture ranking. Original HTML sections, anchors, media and the original README below are preserved verbatim.

---

# Ant adaptation

Latest experiment: [September 2026 profile-latency results](index.html#profile-pipeline-20260911). Earlier L0/L2 results below refer to historical experiments with their own evaluation conditions.

Open [`index.html`](index.html) for the verified Ant-v4 native 8D continuous-action and 27D RGB+state adaptation. The report includes the EGL probe, 10M-frame L0 APPO run, best-checkpoint L0-L4 evaluation, 100 gated teacher episodes, native LeRobot conversion, 5k QwenOFT training, and final-checkpoint L0/L2 VLA evaluation.

Fresh profiling-only run: [20260911T023128Z-p-only4](https://huggingface.co/datasets/placeholder/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/ant/profile_latency/provenance/profiling/status.md), env/obs FPS 10 / 10. Profiling is complete; the subsequent pipeline is tracked below.

- [September 11 realtime baseline](https://huggingface.co/datasets/placeholder/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/ant/profile_latency/provenance/baseline/status.md): completed 20 episodes at 10 FPS; return 194.33 ± 259.05.

- [Ant pipeline continuation](https://huggingface.co/datasets/placeholder/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/ant/profile_latency/provenance/pipeline/status.md): S/E completed at 10/10 FPS; E return 2301.03 ± 37.60. E 0/10 and bounded exporter probe 0/12 passed the original >3000 gate; The user subsequently authorized a >2500 data gate; R/C completed 100,000 rows, all 100 actual episode returns pass >2500, and VLA training completed 5,000 steps with exit code 0 using the same teacher lineage. The final model bundle is uploaded and all 22 files verified; the 20-episode measured evaluation completed with exit 0: return 1,853.33 ± 508.69 versus baseline 194.33 ± 259.05, with all 20 paired seeds improving and 16/20 reaching 1,000 steps.

## Standard Pipeline archive

[Complete experiment archive](https://huggingface.co/datasets/placeholder/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/ant) · [Verification receipt](assets/standard_pipeline.json). This local directory retains report display media; original records are linked at a fixed revision. Only raw demonstrations are distributed, with conversion performed on the training server.


[OpenVLA per-seed real-time comparison](openvla-paired-returns.svg) · [Plot data and fixed-revision sources](openvla-paired-results.json). 20 wins, 0 ties, 0 losses; mean paired gain +1,659.00.


The HTML tables follow the AirRaid schema: B/F mean gain and percentage; separate L0/B/E/F evaluation rows with episode counts and mean raw steps; measured runtime audit; pipeline stages; publication index. [Normalized table values and provenance](report-tables.json). Unrun stages and unrecorded fields are explicitly distinct.
