# Demon Attack teacher evaluation audit — 2026-09-29

The former GPU teacher scores measured the first life, capped at 14,400 raw
frames. Reverting only the life-loss continuation behavior reproduces all
300 archived best-checkpoint episode scores and lengths exactly:
OFT 1525.00, GR00T 663.25, π₀.₅ 447.30. The Python fix had already existed in
`b67c4231`; the old H100 runtime revision was not archived, so the precise stale
checkout or missed deployment step remains unproven.

The same training-selected best weights now have accepted full-game CuLE
returns **5605.65 / 4035.40 / 2919.80** (OFT/GR00T/π₀.₅): 100 episodes,
seeds 104771–104870, latency seed 42, 60/15 Hz, H8, four observation frames,
capacity 1, hold/latest_ready, original RTX 3090 temporal profiles, 14,400-frame
cap and 16 environments. CPU counterparts are 5544.70/4114.70/3056.70.
Gray-image rounding can flip close argmax decisions; persistent per-replica
temporal state then also changes later episode streams. These residual
differences are quantified separately from the first-life scoring error.

GPU raw-reward handling now honors `evaluation.eval_raw_reward`; the existing
False default is resolved once at the configuration boundary. Former GR00T/Pi
final returns 32.90/27.25 were clipped; their raw first-life game scores were
571.40/426.75. They do not establish a collapse of training performance.

Matched 100-episode teacher evaluations use the archived student cohort
1000000–1000099, latency seed 271828, 32 environments and caps 7200/3600 for
Demon/Flappy. Student/teacher percentages are **90.6/92.2/92.6%** for Demon and
**95.4/94.7/89.3%** for Flappy (OFT/GR00T/π₀.₅). Student scores remain the
verified archived full-episode evaluations. No weights were retrained or reselected.

Code verification includes 104 configuration/environment/executor regressions,
14 GPU alignment cases, a frozen-checkpoint CPU/CuLE full-executor regression,
and an explicit 16-slot peer-reset check. Formal evaluation and result acceptance
are complete. The paper's teacher table and matched comparison were regenerated,
visually inspected, compiled with the 44-page manuscript and published to Overleaf
as `cbd01f8`.

The OFT original final weights were not recovered from local archives, fixed HF
revision, W&B files or logged artifacts. This does not claim an exact rerun of
the old 5631.6/1022.45 final-checkpoint pair. The causal evidence uses the three
recoverable published best weights and their complete old 300-episode record.

[Full report and primary evidence](../../outputs/demon_cule_audit_20260929/REPORT.md)
· [Reusable evaluation contract](../DEMON_ATTACK_EVAL_AUDIT.md).
