# InvertedPendulum — GR00T and OpenVLA

[Open the experiment report](index.html). Updated 18 September 2026 from fixed artifact revisions and completed evaluation receipts.

## Measured real-time results

Both B (zero-trained) and F (profile-trained) use RTX3090 measured-only inference latency,20 episodes each, seeds 42–61 and a 1000-step cap. Values are mean ± population SD.

| Model / experiment | Env / obs FPS | B | F | Within-run mean gain |
|---|---: |---: |---: |---: |
| GR00T / QwenGR00T H1,17 September |5 / 5|15.55 ± 4.79|1000.00 ± 0.00|+984.45|
| Historical OpenVLA / QwenOFT |10 / 10|19.80 ± 6.21|586.85 ± 457.53|+567.05|

These are within-model comparisons from one training run. Different clocks, recipes and measured latency distributions prevent a controlled architecture ranking. The historical OpenVLA detailed record, original adaptation and media remain in the HTML report.

## GR00T H1 /5 FPS pipeline — complete

- Contract: native InvertedPendulum-v 4 physics unchanged; discrete forces[-3,-1.5,0,1.5,3], state 4, single RGB224, H1, no action chunks; 5/5 FPS, hold/FIFO and one inference worker.
- L0: reused original APPO H1 data,90/10 episodes and 90,000/10,000 rows. Only prompt/FPS metadata changed 25→5; no L0 teacher retraining. VLA 2000 updates; separate zero-latency L0 simulation (20 episodes) scored 1000 ± 0.
- P: 5×10 episodes,900-step cap with natural termination; 800 canonical samples. Mean 108.09 ms, P95 114.50 ms, max 134.29 ms.
- Teacher: fresh APPO seed 0, 1M requested frames,1,007,616 actual at the batch boundary. Final and training-best each 20 scored 1000 ± 0; tie selectsFinal step 1968. E10 scored 1000 ± 0. Selection uses 104771–104790; E10 repeats its first 10 seeds.
- Data: 12/12 bounded probe; 100 accepted/100 attempts, 0 rejected,90/10 split,90,000/10,000 rows. Every actual replay return 1000, strict gate>900. Train-only minmax state normalization is applied once.
- Profile VLA: fresh 2000 updates, seed 42, global 64=micro 16×accum 4. Qwen3-VL-4B-Instruct +DiT-B,4 inference timesteps,8 diffusion repeats, warmup 100, VLM/action LR1e-5/1e-4,200 held-out batches every 500 updates. Saved reload and real held-out forward[1,1,5] finite passed.
- F20: all 20 reached 1000 steps; 20 wins/0 ties/0 losses against B. B/F mean episode latency 106.07135/106.65249 ms, observation drops 0/311 versus 0/20,000, zero invalid or dropped actions. Trace step intervals average 200.04377/200.00061 ms. No new teacher threshold is used as a VLA acceptance rule.

Prompt key 1 means one raw tick(200 ms at 5 FPS); this is a condition label, not an added 200 ms delay. F uses measured-only latency. Source checkpoint/FPS history remains provenance rather than being rewritten.

[Zero model](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/6358f39350b2ce864479637c4d9cbbf4137343a0/inverted_pendulum/zero_latency/GR00T) · [Profile model / final result](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/df76ab5e19a7b90d49c2a27a8f4133701d6d559f/inverted_pendulum/profile_latency/GR00T) · [APPO teacher](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/964ee0be2ba7348652fff3a5addbf9119e0316c6/inverted_pendulum/profile_latency/small_model/GR00T) · [L0 data](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/a3702d961080ffad9d34f5797d0ccd4cde64f4fb/inverted_pendulum/zero_latency/GR00T/h1_5fps_20260917) · [Profile data and evidence](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/b561e4d1e15ee32dd133046896d4ded3087898a9/inverted_pendulum/profile_latency/GR00T/h1_5fps_20260917) · [Paired comparison](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/b561e4d1e15ee32dd133046896d4ded3087898a9/inverted_pendulum/profile_latency/GR00T/h1_5fps_20260917/vla/evaluation/comparison.json).

The evaluated F weights are revision`f6183233368798a9d98cf345e219410065bcae8a`, SHA256`9ec338c86369f59593c802c5c5b29e1c2956ee46a38eecac47e1788acf7fcdb8`. Final model documentation revision`df76ab5e19a7b90d49c2a27a8f4133701d6d559f` changes no weights. Dataset final evidence is pinned at`b561e4d1e15ee32dd133046896d4ded3087898a9`.

## Evidence scope

[Compact results](results-summary.json) · [Publication index](gr00t-publication.json) · [Per-seed chart](gr00t-paired-returns.svg). Only display media and compact indices are local. Public artifacts include model bundles, raw demonstrations, QA, profiles, episode metrics and summaries. The audited B/F step, action, latency and observation-attempt records, including the 20,000-step F evaluation and both warmups, are now published: 18 trace files (89,646,252 bytes) plus four manifest/checksum files. All 22 files passed pinned size/hash verification; bulk traces remain off the local workspace.

[Baseline traces](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/b561e4d1e15ee32dd133046896d4ded3087898a9/inverted_pendulum/profile_latency/GR00T/h1_5fps_20260917/baseline/evaluation/traces) · [Final traces](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/b561e4d1e15ee32dd133046896d4ded3087898a9/inverted_pendulum/profile_latency/GR00T/h1_5fps_20260917/vla/evaluation/traces).

## Historical references

[OpenVLA 10 FPS detailed report](index.html#openvla) · [Original pinned archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/inverted_pendulum). Historical 25 FPS zero/fixed-L1 experiments remain separately labeled.

<details><summary>Terminated native H8 /15 FPS GR00T experiment</summary>


This terminated, superseded experiment used native InvertedPendulum-v 4, 4D state, 1D continuous
action chunks of eight, RGB224, and 15/15 benchmark FPS. Physics, rewards,
termination, and the native 1000-step limit are unchanged. Its profile arm is
IID. The results and pinned evidence below preserve that terminated arm as historical stage outcomes.

The L0 FastTD3 teacher completed 1,000,064 frames and 15,592 updates. Final and
training-best checkpoints each returned 1000 in all 20 evaluation episodes
(seeds 42–61); the final checkpoint won the tie-break. The 12-episode exporter
probe passed, followed by 100 accepted episodes, each return 1000, against the
unchanged strict >900 gate. The 90/10 episode split yielded 89,370 training and
9,930 validation H8 windows. Loader checks verified action `[8,1]`, state
`[1,4]`, full chunk masks, and train-only normalization shared by validation.

The [teacher bundle](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/cc850b1770b0ab4fec7b61583886fc718a486cb7/inverted_pendulum/zero_latency/small_model/GR00T)
and [raw data and evidence](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/408539fbe0d543600520afff4e802ddaf21f5490/inverted_pendulum/zero_latency/GR00T/native_h8_15fps_20260917)
are published at fixed revisions with file hashes verified. VLA training completed
2000 updates with exit 0 at 2026-09-17 02: 39: 26 UTC; final checkpoint 9,976,816,419B.
Portable packaging and H200 zero-latency 20-episode evaluation completed: mean
return **4.85 ± 2.33** (populationSD), range 3–11, zero invalid/dropped actions.
The controller has poor L0 performance. Checked I/O/normalization/head-loading
contracts showed no mismatch; the cause is unresolved, and no extra training
budget was added. The [fixed 2000-step inference model](https://huggingface.co/latency-sensitive-bench/Standard-Pipeline/tree/a10cfff82d2d4c084f473c25b13a0978f3babab0/inverted_pendulum/zero_latency/GR00T)
and [training/evaluation evidence](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/894ba80a64140299e2df1a5268b9950bf3073374/inverted_pendulum/zero_latency/GR00T/native_h8_15fps_20260917/vla)
are hash-verified. Publication was not control-quality acceptance. This arm was terminated before P/B/E/F; it is not scheduled to resume.


</details>

See the [Gymnasium report index](../README.md) and [adaptation guide](../../../gymnasium_adapter.md).


[OpenVLA per-seed real-time comparison](openvla-paired-returns.svg) · [Plot data and fixed-revision sources](openvla-paired-results.json). 19 wins, 1 ties, 0 losses; mean paired gain +567.05.


The HTML tables follow the AirRaid schema: B/F mean gain and percentage; separate L0/B/E/F evaluation rows with episode counts and mean raw steps; measured runtime audit; pipeline stages; publication index. [Normalized table values and provenance](report-tables.json). Unrun stages and unrecorded fields are explicitly distinct.
