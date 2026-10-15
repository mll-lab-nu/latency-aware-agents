# Screening report: badminton

Owner: Chi Gui. Screened 2026-10-10..12 from `main` 3473d9e7 (revision `badminton-wallclock-9+duel-11+core-0.2`) to
revision **`badminton-wallclock-12+duel-14+core-0.2`** (this pull request; main's `wallclock-10 / duel-12` is the
JerkLimiter servo of #94, merged here; 12/14 adds the joint-velocity rule, 2026-10-13, section 6b). Process: [../construction_zh.md](../construction_zh.md). Scripts:
[`badminton/`](badminton/README.md). A Chinese walkthrough of every round is in the review folder
(`olympiad_review/screening/badminton_walkthrough_zh.md`). Model-free matches of rounds 13-15 ran on a
compute cluster in the same image (equivalence: 3 matches bit-identical to the local docker runs in every field).

## 1. Conclusion

| Check | Result |
|---|---|
| Step 1 define | pass: one game to 11 in both formats (user, 2026-10-10/11); B2 kept secondary (section 6) |
| Step 2 robot and venue | pass: base study (user decision 2026-10-11): a slower real base, the Festo EGC-185-BS-KF-40P ball-screw gantry (2 m/s, 15 m/s^2 per axis); B9 88/88 legal serves returned on it (RB-Y1 40/88, dropped); with the joint-velocity rule (setpoints at 0.9 of the velocity limits, the real robot's 1.05 x velocity reflex; user, 2026-10-13) still 88/88, physical over-speed 2.27 % -> 0.019 % of arm time |
| Step 3 interface | pass: no automatic return of the base after a stroke; `move` and an optional `base` in `act` (P2); contract `skill_api.py` (P6) |
| Step 4 model-free | pass (revision 12, section 6b) under the user's rules BASEPERC (2026-10-11) and BMRAND (2026-10-12): after level 2's reaction 0.30 -> 0.25 s the truth reference wins L1 20/20 and falls with the level (point share 0.82 / 0.51 / 0.04); every truth baseline loses L3 (the gate); at L1 and L2 the truth short-wide baselines win, their camera versions lose (1/20 and 2/20 at L1, 0/20 at L2): L1 and L2 mainly test perception and execution. The camera reference wins L1 17/20. A1, A2 and B2 (being predictable) sensitive; A4 small at x20 |
| Step 5 model trials | pass: revision 11: Opus 5.5 won 11:8 and 11:6, Astra lost 1:11 and 4:11; revision 12 (L1, one each, user decision 2026-10-13): Opus 5.5 **won 11:7** (s101), Astra lost 2:11 (s111); both valid, 0 API errors; no loophole |
| Step 6 freeze | frozen: hidden official seeds registered (20 per cell), goldens re-recorded; tools/check stages 1/2/3/5 pass on revision 12 (section 8) |

**Configuration**: Franka Panda (FCI limits, JerkLimiter servo) on a Festo EGC-185-BS gantry, racket gain 4; BWF
court x 0.4, net 1.524 m; high level `act` (+ optional `base`), `toss`, `move`; match-wallclock against three scripted
levels, duel-wallclock; both one game to 11; hit cap 400 per match; slowdown x20; score = win and point share.

## 2. Candidates and pruning

| Candidate | Evidence | Verdict |
|---|---|---|
| Panda + gantry, BWF court, racket gain 20, arm acceleration unlimited (Season 1) | gain 20 ~5x a real racket | dropped: not real |
| Panda (real limits), BWF court, gain 4 | no clear reaches the back court | dropped |
| **Panda (real limits), court x 0.4, gain 4** | B9 98.9 % (EGC-TB), 100 % (EGC-BS) | **kept** |
| Base Festo EGC-185-TB (5 m/s, 50 m/s^2) | where the robot waits never mattered (0 "could not reach" in 1866 acts) | replaced (user, 2026-10-11) |
| **Base Festo EGC-185-BS-KF-40P (2 m/s, 15 m/s^2, 3000 N, 3 m stroke)** | B9 88/88; waiting position matters once the opponent's guarantee is relaxed | **kept** (user, 2026-10-11) |
| Table-tennis gantry (2 m/s, 10 m/s^2 estimated; arXiv 2309.03315) | B9 88/88; acceleration not sourced | alternative |
| Rainbow RB-Y1 base (1.5 m/s, 2 m/s^2) | B9 40/88 | dropped (fails B9); R1 Pro, Ridgeback slower: dropped |
| Opponent with `min_t` (main) | every point "too quick"; "always to the middle" won 10/10 | dropped |
| Opponent v10 (time model, slow recovery, mean-landing anticipation) on the new base | truth short-wide 20/20 at L1, reference 0/10 at L2 | replaced (rounds 10-15) |
| **Opponent v11: habit reading with early commitment, block, lob over a net player, kills, overhead attacks** | section 6 | **kept** (user, BMOPP 2026-10-12) |

## 3. Step 1: define the sport

Core challenge: predict a high-drag shuttle from two cameras, commit a stroke about 0.9 s ahead, turn the shot you
want into the racket's state at contact through the published contact law, place it away from the opponent and not
in a pattern, decide where the base waits, and serve by the service laws. Rule book: BWF Laws (Statutes 4.1 v5.0);
deviations D1-D9 in `sport.toml` (D1: one game to 11, both formats; match limit 400 hits or 40 points).
Capabilities: a1 2, a2 2, a3 2, a4 2, b1 1, b2 1, b3 0. Public seeds: duel 1-3, match L1 101/111/121, L2 202/212/222,
L3 303/313/323; hidden seeds from the core (user, 2026-10-11). Score: win and point share.

## 4. Step 2: robot and venue

Real limits through `bodies/limits.py` (Franka FCI, JerkLimiter) and the gantry (a `traj` beyond 2 m/s or 15 m/s^2
per axis is refused, E12). **Base study** (user request of 2026-10-11): candidates and sources in section 2; B9
(forced serves x upper-bound receiver, 0 reaction): EGC-TB 87/88, **EGC-BS 88/88**, table-tennis gantry 88/88, RB-Y1
40/88. The reference on level 1 of that time: EGC-TB 0.69, TT gantry 0.63, RB-Y1 0.41, R1 Pro 0.37, Ridgeback 0.37.
Physical consistency as before (contacts on, explicit self-collision pairs, E13). **Joint-velocity rule (revision 12, user decision 2026-10-13: fix now)**: the setpoints never exceed the limits, but
the physical joints overshot them (PD tracking, the torque-rate limit, and the base's acceleration dragging joints 1
and 4 through the arm-base coupling): on revision 11, 2.27 % of arm time in 60 truth reference matches (1.50 % in its
goldens; the 10.4 % / 3.95x reported earlier was main's wallclock-10 goldens on the 50 m/s^2 base). As table tennis
(revision 15): the setpoint generator and the stroke controller at 0.9 of the velocity limits, and the real robot's
velocity reflex: a physical arm joint above 1.05 x its limit stops the robot for 1 s (the arm brakes to the trigger
pose, the gantry to rest within its own limits), then it resumes from rest. Result in section 6b: over-speed
0.019 % of arm time, 17 reflexes in 60 matches, B9 88/88.
**Kills** (user request of 2026-10-12): the reference's downward shots from high contacts are delivered 65 % of the
time up to 2.2 m and 13-31 % above 2.3 m ("head speed limited by the joint speed limits": the arm near full extension
cannot swing fast), so its kill window ends at 2.2 m; 1-3 kills per match.

## 5. Step 3: interface

`[modes.high]`: `act` (t, pos, normal, vel, optional base), `toss`, `move` (base); contract `skill_api.py`
(`game spec` -> `skill_contract`). P2 change: the base no longer returns home after a stroke; where it waits is the
player's (`move`), a stroke's stance can be set (`act` base); the stance formula is published. Tests:
`test_base_moves_only_when_told`, `test_skill_contract_is_the_simulator`. P1, P3-P6 as in the earlier round (the
aim-based legacy stroke removed). Low level: a separate line (PR #98), unchanged arm/racket contract.

## 6. Step 4: model-free validation

Setup: match-wallclock, x20, the reference's perception replaced by the true shuttle + 3 mm ("truth") unless marked
camera; paired seeds 101-291 (20 per cell); one core per match. Gates (user decisions): the truth reference wins the
lowest level and falls with the level, plus evidence that cameras can win (BASEPERC); every simple baseline loses the
majority at the highest level with truth perception, and at lower levels either with truth or with cameras (BASEPERC,
BMRAND); capability sensitivity: paired point-share difference pooled over the levels, 95 % bootstrap interval below 0.

### Rounds

| Round | Change | Finding |
|---|---|---|
| 0-7 | (wallclock-10, earlier report) opponent rebuilt as a robot time model; game to 11; short-wide alternating as a baseline | final then: reference 18/20, 9/20, 0/20 |
| 8 | new base (EGC-BS), no automatic return | old levels too weak: every policy won L1 (reference 0.88, short-wide 0.88) |
| 9 | reach / swing grid on the new base, full suite | short-wide alternating beat the reference at every grid point; L1 20/20 truth (camera 0/10), L2 reference 0/10 |
| 10 | read alternation (0.4 m lag reader) | short-wide still 5/5: the robot meets the shuttle at the net (median contact y -0.34 m, z 1.93 m) and a push reaches a front corner in 0.6 s < reaction 0.3 + swing 0.55 s |
| 11 | clears over a net player (above 2.8 m), lob, block 0.25-0.45 s; **bug fixed** (the interception point could be one it misses while a later one was makeable, since v10) | block 0.35 + reach 0.9: a wall (300-hit cap, 28 hits per point) |
| 12 | reference v2 (net play), v3 (60/40 side mixing); side-transition reader with full-speed commitment | L1 reach 0.7: reference 10/10, short-wide 5/6 |
| 13 | L1 reach 0.75 / 0.85 | 0.85: reference 10/10, short-wide alternating 0/10 (user: L1 = 0.85, BMOPP approved) |
| 13b | random short-wide (`shortrand`) | wins L1 20/20 and the then-L2 11/18 (user: BMRAND; kills; cap 400) |
| 14 | merge of main (JerkLimiter); reference kills (window 2.2 m); high opponent lifts; kills aim at the hardest shot (also from normal returns); overhead attacks; reader of one spot with a prepared stroke | below |
| 15 | final levels | below |

### 6b. Revision 12 (2026-10-13): the joint-velocity rule, re-calibration (final)

Measured with a per-step recorder of the arm's physical joint velocity (`d.qvel`) in the servo, in 60 truth reference
matches (L1-L3 x 20 seeds, the same seeds and opponent before and after) and B9:

| | revision 11 | revision 12 (0.9 margin + 1.05 x reflex) |
|---|---|---|
| physical over-speed, share of arm time | 2.27 % | 0.019 % |
| above 1.05 x | 0.30 % | 0.007 % |
| max ratio | 1.88 | 1.58 (the step that triggers a reflex) |
| over-speed episode median / p90 | 16 / 71 ms | 7 / 39 ms |
| reflex stops | (no rule) | 17 in 60 matches (0.28 per match) |
| B9 returned / legal | 88/88 | 88/88 (returns with any over-speed 14/88 -> 1/88, no reflex) |

Base coupling (most remaining over-speed is joint 1 while the base accelerates, the arm's torque rate unable to follow):
two base changes on top of the rule made it worse and were not taken: the go-to generator's jerk 375 -> 150 m/s^3:
60 reflexes in 60 matches; an acceleration feed-forward in the gantry's drive: 148.

The rule made the old level 2 easy (reference 17/20, point share 0.63, was 6/20, 0.45). One-parameter grid at L2
(20 seeds, truth reference): kill 0.7 12/20 (0.56); reach 0.875 15/20 (0.59); reach 0.9 17/20 (0.63); **reaction 0.25 s
10/20 (0.51), taken**; reaction 0.25 s + kill 0.7 1/20 (0.38). Full suite on `16f1cad1` (20 paired seeds per cell,
truth unless marked; 1320 + 140 camera matches on the cluster):

<!--S50-->
| policy | L1 wins | L1 points | L1 share [95 %] | L1 cap, hits/pt | L2 wins | L2 points | L2 share [95 %] | L2 cap, hits/pt | L3 wins | L3 points | L3 share [95 %] | L3 cap, hits/pt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 20/20 | 220:55 | 0.82 [0.76, 0.88] | 0/20, 10.0 | 10/20 | 199:197 | 0.51 [0.46, 0.55] | 0/20, 9.7 | 0/20 | 11:220 | 0.04 [0.02, 0.07] | 0/20, 6.4 |
| short-wide, random corner | 20/20 | 220:48 | 0.83 [0.79, 0.87] | 0/20, 5.7 | 17/20 | 215:168 | 0.56 [0.52, 0.60] | 0/20, 4.8 | 1/20 | 127:218 | 0.36 [0.32, 0.40] | 0/20, 5.4 |
| short-wide, alternating | 20/20 | 220:66 | 0.78 [0.74, 0.82] | 0/20, 9.3 | 4/20 | 167:215 | 0.43 [0.39, 0.47] | 0/20, 6.9 | 0/20 | 55:220 | 0.19 [0.14, 0.23] | 0/20, 5.8 |
| always to the middle | 0/20 | 0:1 | 0.00 [0.00, 0.00] | 20/20, 14298.0 | 0/20 | 0:7 | 0.00 [0.00, 0.00] | 20/20, 2094.0 | 0/20 | 0:153 | 0.00 [0.00, 0.00] | 18/20, 62.3 |
| return to sender | 0/20 | 2:207 | 0.01 [0.00, 0.02] | 11/20, 38.0 | 0/20 | 3:220 | 0.01 [0.00, 0.03] | 0/20, 10.2 | 0/20 | 2:220 | 0.01 [0.00, 0.02] | 0/20, 5.2 |
| fixed racket | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 8.2 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 4.6 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 3.0 |
| reference + flat short serves | 20/20 | 220:34 | 0.88 [0.84, 0.91] | 0/20, 8.5 | 17/20 | 208:147 | 0.59 [0.52, 0.66] | 0/20, 10.0 | 0/20 | 4:220 | 0.02 [0.00, 0.03] | 0/20, 6.1 |
| reference, no waiting move | 20/20 | 220:59 | 0.80 [0.75, 0.85] | 0/20, 9.6 | 6/20 | 168:200 | 0.45 [0.40, 0.51] | 0/20, 9.9 | 0/20 | 14:220 | 0.05 [0.03, 0.09] | 0/20, 6.2 |
| fixed act | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 |
| random act | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 |
| serve only | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 |
| toss only | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.0 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.0 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.0 |
| nothing | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.0 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.0 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.0 |

| ablation (paired with the reference, same seeds) | L1 share (diff) | L2 share (diff) | L3 share (diff) | pooled paired diff [95 %] |
|---|---|---|---|---|
| A1 noise 2 cm | 0.30 (-0.52, n 20) | 0.15 (-0.35, n 20) | 0.05 (+0.00, n 20) | -0.289 [-0.361, -0.217] |
| A1 noise 5 cm | 0.01 (-0.81, n 20) | 0.01 (-0.49, n 20) | 0.00 (-0.04, n 20) | -0.449 [-0.532, -0.364] |
| A2 drag x0.5 | 0.00 (-0.82, n 20) | 0.00 (-0.51, n 20) | 0.00 (-0.04, n 20) | -0.457 [-0.541, -0.372] |
| A2 no drag | 0.00 (-0.82, n 20) | 0.00 (-0.51, n 20) | 0.00 (-0.04, n 20) | -0.457 [-0.541, -0.372] |
| A4 half the lead | 0.79 (-0.03, n 20) | 0.39 (-0.12, n 20) | 0.06 (+0.01, n 20) | -0.045 [-0.082, -0.010] |
| A4 no lead | 0.75 (-0.07, n 20) | 0.44 (-0.07, n 20) | 0.11 (+0.07, n 20) | -0.022 [-0.057, +0.013] |
| B2 random side | 0.82 (+0.00, n 20) | 0.45 (-0.06, n 20) | 0.05 (+0.00, n 20) | -0.017 [-0.055, +0.019] |
| B2 one fixed corner | 0.01 (-0.81, n 20) | 0.00 (-0.51, n 20) | 0.00 (-0.04, n 20) | -0.453 [-0.536, -0.369] |
| waits at (-0.5, -2.1) | 0.82 (-0.01, n 20) | 0.39 (-0.12, n 20) | 0.02 (-0.02, n 20) | -0.049 [-0.084, -0.017] |
| no waiting move | 0.80 (-0.02, n 20) | 0.45 (-0.05, n 20) | 0.05 (+0.01, n 20) | -0.020 [-0.059, +0.016] |

<!--/S50-->

Camera reruns (BASEPERC / BMRAND; 20 seeds): reference L1 17/20 (0.68 [0.60, 0.74]), L2 2/20 (0.29), L3 0/20 (0.05);
random short-wide L1 1/20 (0.35), L2 0/20 (0.15); alternating short-wide L1 2/20 (0.36), L2 0/20 (0.17). Legal returns
of the camera reference 92 % / 81 % / 61 % (truth 96 % / 90 % / 74 %). Reflexes: truth reference 21 in 60 matches,
camera reference 11 in 60. Reading as in revision 11: the reference falls with the level; every truth baseline loses
L3; L1 and L2 mainly test perception and execution; A1, A2, B2 (one fixed corner) sensitive; "always to the middle"
still runs every L1/L2 match to the 400-hit cap at 0:0 (open).

### Final configuration of revision 11 (superseded by 6b; 20 paired seeds per cell, truth unless marked; code 4d2ff098)

Levels (`bm.LEVELS`, all: swing 0.55 s, window 0.3 s, block 0.45 s, read 0.9, lob 0.7, open_side 0.7, anticipate
0.7, follow 0.35 s): **L1** reaction 0.30 s, reach 0.85 m, kill 0.2, reach margin 0.45 s, mild attacks (15 m/s,
70 % width), recover 0.6 m/s; **L2** as L1 with level-2 attacks (17 m/s, full width), kill 0.5, margin 0.25 s;
**L3** reaction 0.20 s, reach 0.9 m, kill 0.9, margin 0.25 s, level-2 attacks, recover 2.0 m/s.

| policy | L1 wins | L1 points | L1 share [95 %] | L1 cap, hits/pt | L2 wins | L2 points | L2 share [95 %] | L2 cap, hits/pt | L3 wins | L3 points | L3 share [95 %] | L3 cap, hits/pt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 20/20 | 220:63 | 0.80 [0.74, 0.85] | 0/20, 8.3 | 6/20 | 173:206 | 0.45 [0.40, 0.50] | 0/20, 7.5 | 0/20 | 16:220 | 0.06 [0.04, 0.09] | 0/20, 5.8 |
| short-wide, random corner | 20/20 | 220:43 | 0.85 [0.81, 0.89] | 0/20, 5.8 | 18/20 | 221:123 | 0.67 [0.61, 0.73] | 0/20, 4.8 | 2/20 | 151:226 | 0.39 [0.35, 0.43] | 0/20, 5.4 |
| short-wide, alternating | 20/20 | 220:64 | 0.79 [0.74, 0.84] | 0/20, 9.8 | 13/20 | 192:162 | 0.55 [0.49, 0.61] | 0/20, 6.2 | 0/20 | 59:220 | 0.20 [0.17, 0.24] | 0/20, 6.3 |
| always to the middle | 0/20 | 0:1 | 0.00 [0.00, 0.00] | 20/20, 14316.0 | 0/20 | 0:6 | 0.00 [0.00, 0.00] | 20/20, 2392.8 | 0/20 | 0:147 | 0.00 [0.00, 0.00] | 17/20, 61.4 |
| return to sender | 0/19 | 3:168 | 0.03 [0.00, 0.08] | 12/19, 44.1 | 0/20 | 1:220 | 0.00 [0.00, 0.01] | 0/20, 14.0 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 4.9 |
| fixed racket | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 7.9 | 0/20 | 1:220 | 0.00 [0.00, 0.01] | 0/20, 5.0 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 2.9 |
| reference + flat short serves | 20/20 | 220:70 | 0.78 [0.73, 0.83] | 0/20, 8.0 | 14/20 | 212:165 | 0.58 [0.52, 0.63] | 0/20, 7.8 | 0/20 | 6:220 | 0.02 [0.01, 0.04] | 0/20, 5.7 |
| reference, no waiting move | 19/20 | 219:75 | 0.77 [0.71, 0.82] | 0/20, 9.3 | 14/20 | 203:156 | 0.58 [0.52, 0.65] | 0/20, 8.8 | 0/20 | 17:220 | 0.07 [0.04, 0.10] | 0/20, 7.2 |
| fixed act | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 |
| random act | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 |
| serve only | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.2 |
| toss only | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.0 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.0 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.0 |
| nothing | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.0 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.0 | 0/20 | 0:220 | 0.00 [0.00, 0.00] | 0/20, 0.0 |

| ablation (paired with the reference, same seeds) | L1 share (diff) | L2 share (diff) | L3 share (diff) | pooled paired diff [95 %] |
|---|---|---|---|---|
| A1 noise 2 cm | 0.44 (-0.36, n 20) | 0.20 (-0.25, n 20) | 0.02 (-0.04, n 20) | -0.214 [-0.270, -0.161] |
| A1 noise 5 cm | 0.01 (-0.78, n 20) | 0.02 (-0.43, n 20) | 0.00 (-0.06, n 20) | -0.426 [-0.501, -0.343] |
| A2 drag x0.5 | 0.00 (-0.80, n 20) | 0.00 (-0.45, n 20) | 0.00 (-0.06, n 20) | -0.437 [-0.513, -0.353] |
| A2 no drag | 0.00 (-0.80, n 20) | 0.00 (-0.45, n 20) | 0.00 (-0.06, n 20) | -0.437 [-0.513, -0.353] |
| A4 half the lead | 0.71 (-0.09, n 20) | 0.55 (+0.10, n 20) | 0.02 (-0.04, n 20) | -0.010 [-0.046, +0.027] |
| A4 no lead | 0.74 (-0.06, n 20) | 0.62 (+0.17, n 20) | 0.02 (-0.04, n 20) | +0.026 [-0.015, +0.067] |
| B2 random side | 0.77 (-0.03, n 20) | 0.49 (+0.04, n 20) | 0.03 (-0.03, n 20) | -0.007 [-0.046, +0.034] |
| B2 one fixed corner | 0.01 (-0.78, n 20) | 0.02 (-0.43, n 20) | 0.00 (-0.06, n 20) | -0.426 [-0.503, -0.342] |
| waits at (-0.5, -2.1) | 0.79 (-0.00, n 20) | 0.53 (+0.08, n 20) | 0.05 (-0.02, n 20) | +0.021 [-0.011, +0.055] |
| no waiting move | 0.77 (-0.03, n 20) | 0.58 (+0.13, n 20) | 0.07 (+0.00, n 20) | +0.035 [-0.004, +0.077] |

Camera reruns (BASEPERC / BMRAND; 20 seeds):

| policy (cameras) | L1 wins | L1 share | L2 wins | L2 share | L3 wins | L3 share |
|---|---|---|---|---|---|---|
| reference | 15/20 | 0.58 [0.49, 0.67] | 0/20 | 0.30 [0.25, 0.34] | 0/20 | 0.03 [0.01, 0.05] |
| short-wide, random corner | 6/20 | 0.42 [0.32, 0.52] | 0/20 | 0.23 [0.18, 0.28] | - | - |
| short-wide, alternating | 2/20 | 0.27 [0.20, 0.34] | 0/20 | 0.22 [0.18, 0.26] | - | - |


Reading:
- Reference: L1 20/20 (0.80), L2 6/20 (0.45), L3 0/20 (0.06): falls with the level; about 6-8 shots per rally, no
  match at the cap. L3 is beyond the reference (a ceiling level, as before).
- Gate: at L3 every truth baseline loses (random short-wide 2/20, alternating 0/20, middle 0/20, sender 0/20, fixed
  racket 0/20). At L1/L2 the truth short-wide baselines win (random 20/20 and 18/20, alternating 20/20 and 13/20); with
  cameras they lose (random 6/20 and 0/20, alternating 2/20 and 0/20): those levels mainly test perception and execution (BMRAND (c)).
- Skill necessity: serve only, toss only, nothing, fixed and random acts: 0 points at every level.
- Capability sensitivity: A1 (-0.21 / -0.43) and A2 (-0.44 / -0.44) drop at both strengths; B2 "one fixed corner"
  -0.43 [-0.50, -0.34] (the opponent reads a repeated spot), "random side" -0.01 (n.s.): being predictable is
  punished, reading the opponent's position is not rewarded further (B2 stays secondary); A4 not measurable at x20
  (half the lead -0.01, no lead +0.02: at x20 the lead is 1-2 control steps; test A4 at x1-x5). Waiting position: waiting at (-0.5, -2.1) +0.02, no waiting move +0.04 (both n.s.).
- Flat short serve (a model's 10-point streak on the old opponent, E14): 0 aces in 430 flat serves at L1-L3; it
  pressures the receiver at L2 (65 % of the serving points vs 45 % with the reference's serve), not a sure point.
- Open: "always to the middle" ends every L1/L2 match at the 400-hit cap at 0:0 (a loss, but 400 hits): its shots
  arrive straight at the opponent, which is always mildly pressed and its attacks are retrieved (decision for the user).

**Reference check (BASEPERC)**: opponent calibration by the truth reference (above); camera evidence: the camera reference itself wins L1 15/20 (point share 0.58), so cameras can win;
end to end (camera reference): legal returns 863/998 (86 %) at L1, 652/872 (75 %) at L2, 292/512 (57 %) at L3; the truth reference 1043/1104 (94 %), 1225/1430 (86 %), 583/803 (73 %).

### Engineering, fairness, safety

`tools/check badminton` on the final code (after merging main's #97 and declaring `[score]`): stages 1-3
**97/97 passed** (contract incl. the continuous score point_share; generic and sport tests incl. E14; goldens 3/3
identical: duel-s1-A 11:4, duel-s3-B 11:4, match-L1-s1 11:6). Stage 4 (duel fairness, camera reference on both seats, N = 40,
seeds 1-20 per order) FAILED once: first server -22.5 +- 13.8 pp. A 200-duel re-run of exactly that check on a
compute cluster (seeds 1-100 per order, seat-swapped) gives **first server -6.0 +- 6.9 pp, seat A +5.0 +- 6.9 pp**
(the server wins 2311/3615 points, 64 %): no advantage beyond +-5 pp is established, so the N = 40 FAIL was the draw of
seeds 1-20 (with truth perception, 60 duels: first server 33/60). Under the rule this is a WARN (interval wider than
+-5 pp) at N = 200; the local check keeps N = 40 (an open decision: raise `matches_per_order`, about a day per local
check). Stage 5 with the new `[reference] match` hook (`reference_check.py`, the step-4 loop): **pass**: truth
reference L1 20/20 (point share 0.78), L2 6/20 (0.46), L3 0/20 (0.07), legal returns 85 %; camera reference L1 4/5
(point share 0.57, legal returns 86 %) and Opus's 11:8 registered as camera evidence. Stage 6: no low level here. `ci_checks/run_all.sh` and the GitHub CI of this
pull request pass.

## 7. Step 5: model trial runs

| Round | Configuration | SOL | Astra | Opus 5.5 | Outcome |
|---|---|---|---|---|---|
| 1 | game to 5, old opponent | 0:5, 0:5 | 0:5, 0:5 | - | all zero; stance formula published, game to 11 |
| 2 | game to 11 | 1:11, 1:11 | 4:11, **11:8** | - | Astra's short-wide alternation -> baseline |
| 3 | wallclock-10 final | 0:11 | 1:11, 4:11 | **14:12**, 1:11 | discriminating |
| 4 | new base, round-9 opponent | - | 8:11, 6:11 | 4:11, 7:11 | all lost but all scored; `move` used 22-42 times, `act` base 6-9 |
| 5 | final (wallclock-11 / duel-13) | - | 1:11, 4:11 | **11:8**, **11:6** | discriminating: Opus won both, Astra lost; no loophole |
| 6 | revision 12 (the joint-velocity rule, L2 recalibrated) | - | 2:11 | **11:7** | discriminating; no loophole |

Round 5 details: Opus s101 returned 59/67 rally shots (170 `move`, 121 `act`, 18 reviews, 50 k output tokens, 66 min),
its winners mostly to the front corners; Opus s111 53/59 (192 `act`, 118 `move`, 16 reviews, 54 k tokens, 65 min), its
winners short and deep on both sides (no fixed pattern). Astra s101 19/30, lost mainly to its own net errors (11
reviews, 33 min). Astra s111's first run hit the Azure rate limit (6 unrecovered API errors, an invalid trial) and was
re-run as a technical interruption: 4:11 (23/34, 14 reviews, 36 min, no API error). Commands of every match scanned: only the match API was used.
Round 6 (revision 12): Opus s101 returned 126/133 rally shots (424 `act`, 17 reviews, 76 k output tokens, about 2 h;
3 velocity reflexes); Astra s111 22/33 (12 reviews, 25 k tokens, 41 min; 1 reflex). Opus's first try was a technical
failure (the task directory was regenerated during the match) and was re-run.

## 8. Freeze

- Version `badminton-wallclock-12+duel-14+core-0.2`, image `ghcr.io/mll-lab-nu/eai-olympiad:0.1.0`; goldens re-recorded
  on it: duel-s1-A 11:8, duel-s3-B 7:11, match-L1-s1 11:7 (the camera reference against level 1); their physical
  over-speed 0.01 % of arm time (revision 11's goldens 1.50 %), one reflex.
- **Frozen**: hidden official seeds registered (`tools/seeds.py`, 20 per cell: duel-wallclock, match-wallclock L1-L3;
  `[official.hidden_seeds]`, `seed_commitments.json`; the private set only in the official store on mll-4090-1).
  Public seeds (duel 1-3, match L1 101/111/121, L2 202/212/222, L3 303/313/323) for practice only.
- Levels L1 (default), L2, L3 (ceiling); First Exam: L1 and L2. Scoring: win and point share.
- `tools/check badminton` on revision 12: stages 1-3 **99/99** (one WARN: no declared reaction delay - the screening
  used the hooks' delay, so none is declared); after merging main (core 0.2.19) stage 1 57/57 and stage 3 4/4 again.
  Stage 5 **pass**: truth reference L1 20/20 (point share 0.80), L2 10/20 (0.50), L3 0/20 (0.05), legal returns 89 %,
  21 velocity reflexes in 4099 rally shots; camera reference L1 5/5 (0.63), legal returns 91 %; camera wins registered:
  Opus 11:7 (revision 12) and 11:8 (revision 11). Badminton's own code did not change in the main merges after
  stage 5, so its result stands. Stage 4 (user decision 2026-10-13): `matches_per_order` stays 20; evidence
  `docs/screening/badminton/fairness_200.json`: 200 seat-swapped camera duels on revision 12, first server
  0.0 +- 6.9 pp, seat A +1.0 +- 6.9 pp, the server wins 61 % of points (revision 11: -6.0 +- 6.9, +5.0 +- 6.9).
  Stage 6: the low level (#98). GitHub CI green.

## 9. Decisions and process issues

Decisions taken by the user (2026-10-10..12): game to 11 (both formats); B9 only for physical tasks; 20-seed baseline
gate; point share; BASEPERC; the EGC-BS base and the 0.45 s reach margin; no automatic return, `move`, `act` base;
BMOPP (habit-reading opponent, reference net play and mixing, L1 reach 0.85); BMRAND (camera reruns, else only L3
gates truth baselines; `kill`; cap 400; kills at every level, a reference kill, realistic rallies).
Open for the user: the passive "always to the middle" wall at L1/L2 (options: accept, a rally-length rule, or an
opponent that attacks body shots; accepted for now); the JOINTVEL rule for badminton (left to the next revision for
now). Decisions taken for the user tonight: L2/L3 parameters, the reference's kill window (2.2 m), the opponent's
spot reading with a prepared stroke and its overhead attacks.

| Issue | What we did | Suggestion |
|---|---|---|
| Simple baselines are an open list; the strongest ones (short-wide, alternating then random) came from a model and from probing | added each to step 4 | write: a model's winning tactic becomes a baseline; probe randomised variants of every pattern baseline |
| The reference itself played a weaker tactic than a simple baseline | net play and mixing added to the reference | check the reference whenever a baseline beats its point share |
| A one-line bug in the opponent (earliest near-optimal interception) decided whole rounds | fixed, unit-tested | unit tests for the opponent's decisions |
| A passive retriever never loses a point against a guaranteed-reachable opponent | kills, overhead attacks, a spot reader | put "the opponent can finish points" into step 1's opponent design |
| Kill feasibility is limited by the arm near full extension | kill window 2.2 m | B9-style tests for every stroke type a reference uses, not only the serve receive |
| Rounds were expensive locally (4-8 CPU h each) | moved to a compute cluster (3 matches bit-identical) | run step 4 on a cluster by default |
| Host load and container caps changed several times | | |
