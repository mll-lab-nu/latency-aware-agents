# Badminton with real physics on a small court (2026-10-08)

- **Branch:** `badminton-minicourt` (rebased on origin/main `da5500f8`, after #32 badminton-delay-fault, #26 robot library, #33 rng-duels, #38); revision **badminton-wallclock-8 + duel-10**.
- **Decision (user, 2026-10-08, option B):** keep real physics and shrink the court. Input: the serve-receivability study
  (branch `badminton-receive-study`, `docs/benchmarks/badminton-serve-receivability-20261008.md`): with the physically
  justified racket gain 4, the Franka Panda's FCI limits and a Festo EGC gantry the Panda can only serve short on the
  BWF court, cannot reach the back court, and reference rallies never reach 3 shots.
- **Scripts and raw results:** `docs/benchmarks/badminton-minicourt-20261008/` (`run.sh <script.py>` in the
  eai-olympiad image, one CPU, no network; at most 3 containers at a time). No model API calls.
- **Related branches:** `badminton-delay-fault` (#32, merged: each undue delay a fault; revision wallclock-6 / duel-8)
  and `rng-duels` (#33, merged: each robot its own execution-scatter stream; revision wallclock-7 / duel-9) are
  included by the rebase; this branch takes **wallclock-8 / duel-10**. The measurements in §1-§6 were run before the
  rebase onto #33 (one scatter sequence for both robots); #32 changes only what follows an undue-delay fault (none in
  the measured matches). After the rebase onto #33 the golden cases were re-recorded and the fairness check re-run
  (§5, "after the rebase onto #33").

## 0. Summary

| | BWF court, today's rules (gain 20, invented base) | BWF court, real physics (study) | **small court, real physics (this branch)** |
|---|---|---|---|
| court (singles) | 13.40 x 5.18 m, net 1.524 m | same | **5.36 x 2.072 m (x 0.4), net 1.524 m** |
| racket gain / arm / base | 20 / velocity only / 6 m/s per axis | 4 / FCI limits / EGC | 4 / FCI limits / EGC (+ coupling feed-forward) |
| legal serves the server produces (grid of 105) | 80 (short 15, long 28 + 14) | 17 (16 short, no long) | **100 (short 41, mid 29, long-flat 13, long-high 17)** |
| upper-bound receiver, reaction 0 | 80/80 | 17/17 | **97/100 (97 %)** |
| upper-bound receiver, reaction 0.2 s | 78/80 | 12/17 (71 %) | **88/100 (40 strokes searched); 92/100 (100)** |
| basic shots exist (upper bound) | yes | no clear | **clear, drop, drive, downward ("smash") all exist** |
| upper-bound rallies (informed players) | — | — | **4.7 hits per point (19 points; 12 with >= 4)** |
| reference vs reference | 4.2 hits per point | 0.45 | **2.71 hits per point (N = 40, 320 points)** |

## 1. The court

**Official reduced formats** (research 2026-10-08; sources in §8). None is short enough:

| Format | Court per side (live depth) | Net | Note |
|---|---|---|---|
| BWF Laws | 6.70 m (singles 13.40 x 5.18) | 1.524 m centre / 1.55 m posts | reference |
| BWF AirBadminton | 8 m (16 x 5 singles), 2 m dead zone | posts 1.55 (sand 1.45 centre) | longer, heavier shuttle |
| Spain FESBA minibádminton (U9/U11) | about 5.94 m (back line = doubles long service line), 4.20 m wide | 1.37 centre / 1.40 posts | |
| Argentina FEBARA mini (U9/U11) | about 5.94 m, half width | 1.30 (U11) / 1.10 (U9) | |
| Swiss Kids, France/Belgium mini-bad | about 5.94 m | 1.40 / 1.372 | |
| BWF para, wheelchair singles | live 1.98-6.70 m (4.72 m), half width | 1.524 | the front zone is out |

The robot's range sets the scale. Measured with the official stroke controller under the Panda's limits (probe
`probe_deliver.py`): from its ready pose the arm delivers a stroke reliably about 0.9-1 s ahead (25 of 27 contacts at
1.0 s, 13 of 27 at 0.7 s) with a head speed along a forward face of 1.5-2.5 m/s (median 1.9). With gain 4 the shuttle
leaves at 6-11 m/s; the real drag (terminal 6.8 m/s, kept) then carries it at most 4-5 m (8 m/s: 4.0 m from 1 m high,
10 m/s: 4.9 m; a lofted 55 deg shot of 9 m/s: 3.9 m in 1.4 s). The study's 7.5 m (3.5-4 m per half) assumed 18 m/s,
the Panda's best head speed in its best direction; forward strokes in play reach half of that.

**Chosen: every horizontal line of the BWF court x 0.4, the net at its BWF height** (D8):
singles side lines |x| = 1.036, back lines |y| = 2.68, short service lines 0.792 m from the net, posts at 1.22.
- A clear from the robot's home contact (about 1.0 m behind the net) to 0.85 of the opponent's half is 3.3 m: within
  the robot's range; from its own back line it is 4.4 m, at the edge (a player pushed back can only lift to the
  middle, as in real badminton).
- The robot is human-sized (racket 0.4-2.5 m high, reach about 1 m from the base): the court's reach-to-width ratio
  stays about the human one (robot 0.95 m / 1.04 m half width; human about 1.6 m / 2.59 m). So the net is not scaled
  (junior nets of 1.30-1.40 m would only make low flat shots easier).
- Tried first: x 0.5 (half 3.35 m). By the same numbers a clear from home would be 4.1 m and from the back line
  5.6 m: beyond the forward head speeds measured in play (the first duels on it, with an earlier controller, had
  almost no return of serve delivered in time); x 0.4 was then chosen and everything below is measured on it.
- The singles back line is also the long service line (a BWF singles-only court), so the service court is 0.792-2.68 m.
- Serve spot (D4) x 0.4: x = +-0.12, y = -1.6. The serve's contact is then about 1.07 m behind the net and below
  1.15 m: a serve must leave at 40-75 deg to rise over the net (all serves are "lofted"; flight 1.04-1.74 s).

## 2. The shuttle

Kept: the point mass with the real feather shuttle's drag (K = 0.212 /m, terminal 6.8 m/s, no spin, no bounce). The
drag is what makes the slow game possible: every lofted shot takes 1.0-1.5 s, which the slow arm needs. A lighter or
lower-drag shuttle (e.g. AirShuttle) would fly further for the same speed but faster too, and would need a new
justified model; not needed at x 0.4.

## 3. The physics as implemented (sport code)

- `RACKET_GAIN = 4` (D9). `BASE_VMAX, BASE_AMAX, BASE_JMAX, BASE_FORCE = 5, 50, 1250, 2500` (Festo EGC-185-TB-KF, part
  556817; jerk 25 x acceleration as in the audit); the scene's base motors 2500 N; the go-to generator limits the jerk
  and plans at 40 m/s^2 (the 2500 N move the carriage, pedestal and arm, about 58 kg, at up to 43 m/s^2); a low-level
  traj is refused beyond 5 m/s or 50 m/s^2 (`bm.base_rows_problem`).
- The arm: `bodies/franka_panda` limits through `bodies/limits.JerkLimiter` (velocity, acceleration, jerk, the jerk also
  capped by the torque rate at the pose) + inertia feed-forward + PD + bias compensation + `TorqueRateLimiter` (87 / 12
  N m, 1000 N m/s), at 1 kHz, in the live world, the twin rehearsals and the duel (`bm.Phys.servo`); the base's
  commanded acceleration (second difference of its targets) is fed forward into the arm torques through the inertia
  coupling (without it a gantry at 40 m/s^2 shoves the arm: strokes with a moving base were delivered 3 of 27 times at
  1.0 s, with it 25 of 27). sport.toml `[arm] body = "franka_panda"`.
- Stroke controller (legacy, adapted): ready pose with the racket head 1.15 m high (searched: the quickest to the slow
  game's contacts under the acceleration limits; the legacy ready pose, head 1.78 m with the elbow straight, needs 1.5
  rad of the elbow to reach down); IK solutions ranked by rest-to-rest time under the acceleration limits; the swing
  from rest as short as 97 % of the speed and 90 % of the acceleration limits allow, with a slower head at contact when
  the time left does not allow the commanded one (the reply says "head speed limited"); the follow-through to rest
  within the acceleration limits; the base counted arrived within 1.5 cm (about 0.15 s earlier than the exact settle);
  the arm back to its ready pose at 90 % of its speed limits (was 60 %); the rehearsal score weighs the head velocity
  3 x more (a head 0.5 m/s off sends the shuttle 2 m/s off). `state.command` adds the rehearsal's head velocity
  (`vel_rehearsed`) and position error (`pos_error_rehearsed_m`).
- Cameras closer for the small court (behind: 7.3 m behind the net, 2.9 m high; side: 7 m from the centre line); the
  reference's detection error at a serve: about 2 cm.

## 4. B9 feasibility

### 4a. The server's basic serves (`study.py`, the reference's serve forced to a grid)

Grid: the receivability study's 5 x 7 targets over the diagonal service court, scaled x 0.4 (x = 0.08-0.98, depth
0.82-2.64 m) x 3 launch-angle families. The study's families (2-16, 20-36, 44-60 deg) cannot clear the net from 1.07 m
behind it; the small court's: flat 38-54, mid 54-66, high 66-78 deg. The server tries head-speed caps 1.8, 2.4, 3.0,
4.2 m/s; the first legal serve counts. Types by the actual flight (the study's depth bounds x 0.4): short (< 1.4 m from
the net), mid, long (>= 2.0 m): long-high if the apex is >= 2.52 m.

| | attempted | legal | short | mid | long-flat | long-high | speeds |
|---|---|---|---|---|---|---|---|
| flat 38-54 deg | 35 | 30 | 10 | 10 | 10 | 0 | 5.5-8.2 m/s |
| mid 54-66 deg | 35 | 35 | 15 | 10 | 3 | 7 | 5.5-9.5 m/s |
| high 66-78 deg | 35 | 35 | 16 | 9 | 0 | 10 | 6.1-12.6 m/s |
| **all** | 105 | **100** | **41** | **29** | **13** | **17** | 5.5-12.6 m/s |

Both short and long serves exist (5 flat serves to the front corners cannot be produced: they hit the net). The
forced server clears the net by 0.3 m (the reference's own serve: 0.4 m).

The reference as receiver (perception-free) returned 57 of these 100 serves into the server's court (short 17/41,
mid 16/29, long-flat 8/13, long-high 16/17); the full court with real physics: 0 of 17 (study).

### 4b. The upper-bound receiver (`oracle.py`: the true state REACT steps after the serve's hit, a search of up to 40
strokes through the official planner with its scatter, on copies of the world)

| reaction | returned / legal | short | mid | long-flat | long-high |
|---|---|---|---|---|---|
| 0 | **97/100 (97 %)** | 41/41 | 27/29 | 12/13 | 17/17 |
| 0.2 s | **88/100 (88 %)** | 39/41 | 22/29 | 10/13 | 17/17 |
| 0.2 s, search of up to 100 strokes for the 12 misses | **92/100 (92 %)** | 39/41 | 24/29 | 12/13 | 17/17 |

The unreturned serves are flat serves (38-54 deg, 6-7.6 m/s) landing 1.7-2.3 m deep after 1.07-1.13 s of flight: the
receiver at y = -1.6 must go back and meet them low within about 0.85 s (0.2 s reaction), and every candidate stroke
misses or goes out. A wider search (up to 100 strokes instead of 40) returns 4 of the 12: **92 %**, above the 90 %
bar; the 8 left are 6 flat mid/long serves and the two front-corner short serves (x = -0.98, 0.82 m deep), which the
base cannot reach in time. For comparison, the BWF court with real physics: 71 % at 0.2 s (study).

### 4c. Basic shots exist (`shots.py`: upper bound, incoming lofted shots, B at home at rest)

Nine incoming flights (lofted 50 / 65 deg to depths 1.2-2.3 m, a high lift near the net); a shot type exists if some
stroke through the official planner lands as the type says.

| type | exists for | e.g. |
|---|---|---|
| clear (lands >= 0.75 L deep) | 8 of 9 | contact 1.86 m, 0.51 s ahead, 40 deg, lands 2.45 m deep |
| drop (lands <= short service line + 0.3 m) | 7 of 9 | contact 2.36 m, 20 deg, lands 0.65 m from the net |
| drive (launch <= 15 deg, in) | 8 of 9 | contact 2.49 m, 10 deg |
| downward "smash" (launch < 0, in) | 6 of 9 | from contacts 2.26-2.57 m, -2 to -5 deg (the net is 1.524 m: only a high contact hits down) |

### 4d. Rallies

- **Upper bound** (`rally_oracle.py`: the reference serves; every shot is returned by an informed player: the true
  state, strokes searched through the official planner, lofted rally-keeping aims first): 19 points, **4.74 hits per
  point**, 12 of 19 with >= 4, longest 10.
- **Reference vs reference** (perception-free, N = 40): §5.

## 5. The reference player and fairness

(N = 40: seeds 1-20 x both first servers, the reference on both seats, perception-free: the true shuttle position at
the frames' times + 3 mm noise; the camera check of the detection: §3.)

| | value |
|---|---|
| first server wins | 16 of 40: **-10.0 +- 15.2 pp** |
| seat A wins | 20 of 40: **0.0 +- 15.5 pp** |
| verdict (tools/check stage 4 rule) | **pass**: no interval lies beyond +-5 pp (WARN: wider than +-5 pp at N = 40) |
| points won by the server | 144 of 320 (45 %) |
| hits per point | **2.71** (median 2, longest 9); 46 % of points >= 3 hits, 17 % >= 4 |
| distribution (hits: points) | 1: 26, 2: 147, 3: 91, 4: 30, 5: 13, 6: 6, 7: 3, 8: 3, 9: 1 |
| how points end | unreturned shot 128, own side 92, net 64, unreturned serve 26, out 10 |
| shot speed (median) | 6.85 m/s |

(The two orders of a seed are not mirror images here: the perception-free noise and the reference's random placement
are seeded by the seat. With the images they are, as in tools/check stage 4.)

**After the rebase onto #33** (revision wallclock-7 / duel-9 underneath: each robot its own execution-scatter stream,
`bm.scatter_rng`; re-run on the rebased branch, same scripts, results `fair-r34`, `b9f-r34`):

| | before the rebase (one scatter sequence) | after the rebase (one stream per robot) |
|---|---|---|
| first server wins (N = 40) | 16: -10.0 +- 15.2 pp | 17: **-7.5 +- 15.3 pp** |
| seat A wins | 20: 0.0 +- 15.5 pp | 21: **+2.5 +- 15.5 pp** |
| verdict | pass | **pass** (WARN: wider than +-5 pp at N = 40) |
| points won by the server | 144 of 320 (45 %) | 142 of 345 (41 %) |
| hits per point | 2.71 (median 2, longest 9; >= 3: 46 %, >= 4: 17 %) | 2.80 (median 2, longest 11; >= 3: 47 %, >= 4: 19 %) |
| how points end | unreturned 128, own side 92, net 64, unreturned serve 26, out 10 | unreturned 142, net 95, own side 76, unreturned serve 17, out 15 |
| B9 smoke: legal grid serves (of 105) | 100 | 89 |
| B9 smoke: upper-bound receiver, reaction 0 | 97 of 100 | **88 of 89 (99 %)** |

The 11 grid serves legal before and out now are all at the grid's shortest depth (0.82 m from the net, 2.8 cm beyond
the short service line, x from -0.98 to -0.08): every grid serve is the server's first act on the same seed, so all
of them share one scatter draw, and the new stream's draw sends these 2.8 cm-margin serves short. It is one draw,
not a change of the physics; the reference's own short serves aim deeper (no service fault in either fairness run).

**Target not reached: >= 4 hits per point.** The upper bound plays 4.7, so rallies exist under the physics; the
reference's limit is the arm's time. Measured over its acts in duels: strokes planned >= 0.9 s ahead were delivered 13
of 13 times, under 0.85 s 4 of 19; most rally shots leave the receiver 0.5-0.9 s after its reaction (0.13 s: four
camera frames), and a shot landing short or deep in a corner then cannot be returned. What the reference does
(builtin_players.py): contacts scored by the time left (0.9 s), the head-speed margin and the placement (lofted deep
shots first), the net cleared by 0.4 m, and the controller's rehearsal read back from `state.command`
(`vel_rehearsed`: the command scaled when the head speed is off; a pose not delivered: a later contact). Each of these
moved it from 1.7 (first version) to 2.7 hits per point. Tried and dropped: contacts up to 2.3 m (2.5 hits per point),
a cubic replaced by a bang-bang swing (fewer deliveries: the torque-rate jerk cap), a stance with the contact beside the
base (fewer deliveries).

## 6. The scripted opponent (match-wallclock)

Depths and widths x 0.4, shots <= 18 m/s, its shots leave your robot >= 1.05 s and 0.95 s of swing (`REACH_*`), and it
needs `min_t` after your hit (level 1: 1.15 s, level 2: 1.0 s). Reference (perception-free): level 1 seeds 101, 111,
121: won 3 of 3 (6-4, 5-0, 5-0); level 2 seeds 202, 212, 222: lost 3 of 3 (0-5): level 2 stays beyond the reference,
as before.

## 7. Deviations from the BWF Laws (sport.toml)

D1 (short game), D2 (the toss), D3 (point mass), D4 (serve spots +-0.12 / -1.6; the base a real gantry), D5 (last
resort), D6 (undue delay), D7 (no toss for the first server), **D8 (the court x 0.4, the net at its BWF height)**,
**D9 (the robots: the Panda's limits, the gantry, gain 4)**.

## 8. Sources

- BWF Laws of Badminton (Section 4.1), Diagrams A, D-F; BWF "Variations in court and equipment" (singles-only court).
- AirBadminton: Badminton Oceania, "AirBadminton - Getting started" (16 x 6 / 5 m, 2 m dead zone).
- FESBA, "Reglas del Juego del Minibádminton" (2016); FEBARA, "Reglamento Mini-Bádminton" (2021); BASPO mobilesport,
  "Badminton: Regeln" (Swiss Kids); FFBad technical regulation (2010) and the Walloon facility sheet (2019).
- Festo EGC-185-TB-KF data sheet (part 556817); Franka FCI limits (`bodies/franka_panda/body.toml`).
- Shuttle and racket: the receivability study §2, §8.
