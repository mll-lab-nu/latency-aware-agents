# AirRaid agent-assisted adaptation

Latest experiment: [September 2026 profile-latency results](index.html#profile-pipeline-20260911). Earlier L0/L2 results below refer to historical experiments with their own evaluation conditions.

Open [`index.html`](index.html) for the complete AirRaid release record. It
documents the PAL raw-frame contract, native six-action catalog, four-frame CHW
policy observation, zero-latency 10M-frame Atari APPO training, seeded
random-noop + FireReset evaluation, 100 gated teacher episodes, native six-action
LeRobot conversion, 5k-step QwenOFT training, and final L0/L2 latency-bench results.

The current rerun contract is encoded in
[`scripts/bash_scripts/rollout_data/air_raid.sh`](../../../../scripts/bash_scripts/rollout_data/air_raid.sh):
teacher rollout uses random-noop 30 plus FireReset, 3,600 raw frames, and
`context_window=0`. The small-model and VLA evaluation configs also use a
3,600-raw-frame episode budget. The HTML report and synchronized evaluation
assets now use this contract.

The final VLA checkpoint reaches mean return 2,000.0 at L0 and 681.25 at L2
over 20 episodes per latency. The corresponding teacher dataset and W&B run are
linked from the HTML report.

Fresh profiling-only run: [20260911T023128Z-p-only4](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/air_raid/provenance/report_sources/profiling_ledger.md), env/obs FPS 50 / 12.5. Profiling and publication are complete; the subsequent full pipeline is recorded below.

Pipeline continuation: [20260911T054920Z-airraid](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/blob/7d088338043f96511df41c74bc540e62d27aa417/air_raid/provenance/report_sources/profile_latency_ledger.md), reusing the measured profile above. P/B/S/E/R/C/V/F are complete with all uploads verified. Matched 20-episode realtime return improved from **878.75 ± 301.69** to **2,353.75 ± 425.39** (+167.85%), at 50/12.5 FPS.

## Standard Pipeline archive

[Complete AirRaid archive](https://huggingface.co/datasets/latency-sensitive-bench/Standard-Pipeline/tree/7d088338043f96511df41c74bc540e62d27aa417/air_raid) contains adaptation, zero-latency and profile-latency evidence, including raw demonstrations and B/E/F traces. Converted datasets are generated on the training server and are not included. [Verification receipt](assets/standard_pipeline.json). Model weights remain in the model repository with fixed references inside the archive.

Local files contain only the report, display media and archive index. Original experiment files are linked to a fixed Hugging Face revision; viewing those files requires a network connection. Raw demonstrations are converted on the training server.
