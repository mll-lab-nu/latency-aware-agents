# Robot baseball: program play (batting)

You play robot baseball alone. Robot baseball in real time (slowed down): a kinematic swing rig with a 0.84 m wooden bat against pitches with drag and Magnus lift, a bat-ball impulse model and fixed fielders; the score is the bases a batter collects (walk or single 1, double 2, home run 4); format batting: you bat 5 plate appearances against a scripted computer pitcher (levels 1-4, `game spec` match.opponent_level) and need at least 4 bases; format duel: two players alternate as batter and pitcher, 5 plate appearances each, more bases wins.
This task is generated from `sports/baseball/sport.toml` (task `baseball-batting-wallclock-program-standard`).

## Time
The world runs in real time at 1/S of real speed (S = `timing.slowdown` in `game spec`); your thinking costs game time. At the end of every unit the world freezes for the reviews.

Whenever the world freezes for a review (after every unit: a point, a shot, a heat; see `game spec`), send `game ready "WHY" "PLAN"` (a short review) before the next unit starts.
The match is graded when your session ends: stay until it is over. **If your session ends before the match is over, you lose the match** (a valid result, success 0), also if you never sent `game start`.

## Write a program
You may write a program that observes, measures in the images, computes your next action and sends it, without a
new model response for each action. You stay in the loop: watch its log, and stop, fix and restart it whenever
needed. Only you send `game start` and `game ready`.

- Use only the camera images, the service's replies and this document; no simulator internals or hidden state.
- One action at a time: an action sent before the previous one has returned is refused (HTTP 429).
- Your container: Python 3.12 with numpy, SciPy and Pillow; no internet during the match. Keep your scripts, logs
  and notes in `/app/output/`.

## Interface
Use the `game` command, or the same calls as HTTP with JSON at `http://sim:8000/<SEAT>/` (your seat is `$SEAT`):

| `game` command | HTTP | |
| --- | --- | --- |
| `game spec` | `GET /spec` | the actions (types, fields, ranges), the cameras with their intrinsics, the rule constants |
| `game status` | `GET /status` | the current state, no image |
| `game start` | `POST /start {}` | you are ready; the match begins when every seat has started |
| `game wait DT` | `POST /wait {"dt": DT}` | let DT game seconds pass (0 < DT <= 1) |
| `game observe` | `POST /observe {}` | the state and the camera images (one PNG of tiles: `observation.tiles`) |
| `game act JSON` | `POST /act JSON` | one action |
| `game swing T PLANE HEIGHT SPEED REACH` | `POST /act {"type":"swing", ...}` | one action (`game spec` lists the fields and ranges) |
| `game none ` | `POST /act {"type":"none", ...}` | one action (`game spec` lists the fields and ranges) |
| `game ready "WHY" "PLAN"` | `POST /ready {"why":"...","plan":"..."}` | your review after each of your actions |

Errors: 400 = an invalid action (nothing happens), 409 = not now (not your turn, or a review is due), 429 = one
action in flight, or more than 20 requests per second or 4 open requests from your seat (empty body: retry more slowly).

## Rules
- robot baseball in real time (slowed down): a kinematic swing rig with a 0.84 m wooden bat against pitches with drag and Magnus lift, a bat-ball impulse model and fixed fielders; the score is the bases a batter collects (walk or single 1, double 2, home run 4); format batting: you bat 5 plate appearances against a scripted computer pitcher (levels 1-4, `game spec` match.opponent_level) and need at least 4 bases; format duel: two players alternate as batter and pitcher, 5 plate appearances each, more bases wins.
- `plate_appearance`: a plate appearance ends with a strikeout (3 strikes), a walk (4 balls), a ball in play (an out or a hit) or, at 15 pitches, an out; a foul is a strike unless there are already 2 strikes; a ball in play is a single, double or home run (over the fence) unless a fielder catches it in the air or throws the batter out at first base
- `pitch`: the pitcher winds up for 0.60-1.20 s with the ball in its hand at the release point (-0.50, 16.80, 1.80) m, then releases; the pitch is over 0.05 s after the contact or after the ball reaches the plate, then the world freezes for the reviews
- `swing`: {'type':'swing','t':T,'plane':PLANE_DEG,'height':HEIGHT,'speed':SPEED,'reach':REACH}: the bat square (along +x over the plate) T s after the control step at which the command takes effect (0 < T <= 2.5); the swing starts 3.9741 / SPEED s before that (at once if that is already past) and, once started, cannot be changed or stopped; a swing command before the release is refused; a later command replaces a planned swing that has not started; {'type':'none'} cancels it. Ranges: -15 <= PLANE_DEG <= 35, 0.30 <= HEIGHT <= 1.40 m, 15 <= SPEED <= 34 m/s, -0.20 <= REACH <= 0.20 m. The executed swing scatters by about 3 ms, 1 degree, 8 mm, 1.5 % and 8 mm
- `duel_pitch`: format duel, the pitcher: {'type':'pitch','pitch':TYPE,'speed':m/s,'x':m,'z':m} during the windup (a later one replaces an earlier one; at the release the last one is thrown, with a scatter of 3 cm on the target and 0.3 m/s on the speed); types fastball 34-42, slider 33-37.5, changeup 29-35.5, curveball 28-33.5 m/s, each with its fixed spin (`game spec` match.pitching); target -0.6 <= x <= 0.6, 0.2 <= z <= 1.6 m at the plate; no pitch by the release is an automatic ball
- `duel`: format duel: plate appearances alternate (match.first_player bats in plate appearances 1, 3, 5, ...); each player bats 5 times and pitches to the other; each player's execution scatter is independent of the opponent's within a match: the windups and the pitch and swing scatters of your plate appearances (as batter) are one seeded stream, those of the opponent's another; the two matches of a duel pair (the same seed, the first batter swapped) swap the scatter sequences, so over the pair both players meet the same draws; level after 5 each: one more plate appearance each until the bases differ, at most 5 times; then the winner of the last pitch wins
- `ball`: mass 0.145 kg, radius 0.0365 m, air density 1.19 kg/m^3, C_D 0.35, Magnus lift C_L = 1.5 S (S < 0.1) else 0.09 + 0.6 S with S = R|w|/|v|, spin decaying with a 30 s time constant; the bat-ball impact has a coefficient of restitution max(0.10, 0.50 - 4.0 (s - 0.69)^2) along the bat (s from the knob) and friction 0.5
- `observation`: a PNG of 2 rows (catcher camera behind home plate, side camera near the pitcher) x 4 columns (frames 30 ms apart, oldest left, each stamped with its time since release; view_times); no ball coordinates; `game spec` gives both cameras' poses and intrinsics (pinhole, tile 480 x 360). The state gives the count, the bases, the time since release, what you heard (release, bat, mitt), your command and the last pitch's result
- You play alone: success = a valid, complete match with a score of at least 4.
- Deviation D1 (rulebook 5.05, 5.06, 7.01): no base runners, runs or innings: each plate appearance scores the bases the batter reaches (walk or single 1, double 2, home run 4; no triples); format duel: plate appearances alternate between the players, 5 each
- Deviation D2 (rulebook 5.05(b)(2)): the batter has no body, so there is no hit by pitch
- Deviation D3 (rulebook 5.09(a)(1), Foul Tip): fouls are never caught and there are no foul tips
- Deviation D4 (rulebook Fair Ball): fair or foul is judged at the first landing point, also for short balls
- Deviation D5 (rulebook Strike Zone): the strike zone is judged in one plane (y = 0) at fixed heights (0.46-1.07 m, |x| <= 0.216 m, grown by the ball radius)
- Deviation D6 (rulebook 2.01): a uniform fence 100 m from home plate, 3 m high (MLB: 325 ft at the lines, 400 ft to centre); fixed fielders without errors
- Deviation D7 (rulebook -): a plate appearance that reaches 15 pitches without ending is an out (MLB has no limit; two-strike fouls cannot stall the game)
- Deviation D8 (rulebook 5.07(c)): format duel: the pitcher only chooses the pitch (type, speed, target at the plate; a fixed windup and release point, a seeded execution scatter); no pitch commanded by the release is an automatic ball, like a pitch-clock violation
- Deviation D9 (rulebook 7.01(b)): format duel: level bases after the regulation plate appearances: one more each, up to 5 extra pairs (MLB plays extra innings until decided); still level, the last resort is not from the rulebook: the winner of the last pitch (the batter if it ended the plate appearance on base, else the pitcher)

## Geometry and physics
Everything you need is in `game spec`: the world frame, the release point, the cameras (pose, field of view, size,
`focal_px`, `principal_point_px`; pinhole: u = cx + f * x_right / depth, v = cy - f * y_up / depth) and the
execution scatter. The physics engine is mujoco 3.3.7 (scene, rig, rendering) + a published ball model (RK4 in pure Python, bb.py).
