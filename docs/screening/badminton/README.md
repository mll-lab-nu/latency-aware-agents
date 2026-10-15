# Badminton screening scripts (docs/screening/badminton.md)

Model-free runs of step 4 and the B9 re-run; nothing here is imported by the sport. Each script runs in the sport's
image with the repository read-only at `/repo` and this folder at `/work` (`run.sh NAME REPO SCRIPT VAR=...`);
`queue.py` runs `jobs.jsonl` with a cap on concurrent containers (file `MAXC`, 2 when the host's load is above 40).

- `scr.py`: match-wallclock, the reference perception-free (true shuttle + 3 mm, study.PerfectRef of
  `docs/benchmarks/badminton-minicourt-20261008/`), policies `ref center reflex fixedracket fixed random serveonly
  tossonly nop`, ablations `a1_2 a1_5 a2_k05 a2_k0 a4_half a4_full b2_rand b2_fixed`; `OPP=file` patches `bm.LEVELS`
  (the calibration grid); writes `out/<TAG>.jsonl` (summary, final fields, the opponent's plan inputs).
- `oppsim.py`: the opponent's decision offline on logged shots under other opponent rules (the calibration's first pass).
- `b9.py`: B9 with self-collision (the self-collision study's script); `phys.py`: contacts on, pairs, E12/E13 and the new
  tests; `runtest.py`: named tests of `tests/test_badminton.py`.
- `summ.py`, `paired.py`, `tables.py`: the tables of the report (`tables.py` = section 6); `boundary.py`: the low
  level's boundary bound; `det.py`: the reference's camera detection error; `capwatch.sh`: the container cap.
- Policies added during the screening: `shortalt` (short and wide, alternating corners: the tactic a model found in
  step 5), `refcam` (the reference with its real camera perception), `CAM=1` (any policy with the camera).
- Revision wallclock-11 (2026-10-12): policies `shortrand` (short to a random front corner), `ref2` (the reference
  of round 12 with `MIX` = the share to the open side), `flatserve` (the reference with a flat fast short serve to
  the receiver's front corner, E14); env `ZKILL`, `KBONUS` (the reference's kill window and bonus); the per-seed
  results of rounds 13-15 were summarised with `qsum.py` (wins, points, share, matches at the hit cap, hits per point,
  kill attempts) and `tables_final.py` (section 6's tables, ablations paired by seed); `msum.py` summarises model
  trials (harbor `result.json`).
- `run_model.sh`: one single-player model match (Codex 0.158.0; astra, or sol on its own Azure route and a lock).
- Revision wallclock-12 (2026-10-13, the joint-velocity rule, report section 6b): `jvrec.py` traces the arm's physical
  joint velocity per physics step while replaying an episode (golden) in the image, `jvsum.py` summarises the traces
  (over-speed share, episodes, joints, torque-rate limiting, base acceleration); `jvhook.py` keeps the same statistics
  online inside a match (`jvhook.install(bm)`; `scr.py`-style runners add `jvhook.stats(match.s)` to their output) and
  `jvq.py` tabulates them with the reflex counts and B9; `fairness_200.json`: tools/check stage 4's duel repeated
  200 times (seeds 1-100, both first servers; the camera reference on both seats) on revision 12.
