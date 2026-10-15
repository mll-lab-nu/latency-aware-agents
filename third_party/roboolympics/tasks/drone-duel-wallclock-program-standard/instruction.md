# Drone racing: program play (duel)

You play drone racing against an opponent model in the same world (seat `A` or `B`). Fly an original simulated quadrotor through a seeded course of numbered gates by its front camera and its own telemetry, commanding world-frame velocity and yaw rate; format racing: three qualifying rounds alone (level 1-3), format duel: heats against another player who flies the same level-2 course at the same time in a world of its own, the first to two heat wins wins.
This task is generated from `sports/drone/sport.toml` (task `drone-duel-wallclock-program-standard`).

## Time
The world runs in real time at 1/S of real speed (S = `timing.slowdown` in `game spec`); your thinking costs game time. At the end of every unit the world freezes for the reviews.

Whenever the world freezes for a review (after every unit: a point, a shot, a heat; see `game spec`), send `game ready "WHY" "PLAN"` (a short review) before the next unit starts.
The match is graded when your session ends: stay until it is over. **If you never send `game start`, your side loses the match** (unless no side started), also when the sessions end before the start window closes.

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
| `game fly VX VY VZ YAW_RATE DURATION` | `POST /act {"type":"fly", ...}` | one action (`game spec` lists the fields and ranges) |
| `game ready "WHY" "PLAN"` | `POST /ready {"why":"...","plan":"..."}` | your review after each of your actions |

Errors: 400 = an invalid action (nothing happens), 409 = not now (not your turn, or a review is due), 429 = one
action in flight, or more than 20 requests per second or 4 open requests from your seat (empty body: retry more slowly).

## Rules
- fly an original simulated quadrotor through a seeded course of numbered gates by its front camera and its own telemetry, commanding world-frame velocity and yaw rate; format racing: three qualifying rounds alone (level 1-3), format duel: heats against another player who flies the same level-2 course at the same time in a world of its own, the first to two heat wins wins.
- `real_time`: this sport is real time (where the generic text above says otherwise, this line holds): `game wait DT` (POST /wait {"dt": DT}, 0 < DT <= 1) lets the world run DT game seconds; `game ready "WHY" "PLAN"` is due only after each round or heat, not after each action; acts are never out of turn, one in flight at a time; in the duel the opponent flies in a world of its own (not the same world); there is no release point and no execution scatter: your command goes to the drone's attitude controller
- `course`: numbered orange square gates spaced 3.5-4.3 m along +x, alternating left and right with changes of height and yaw; level 1: 6 gates with 1.4 m openings; level 2: 8 gates, 1.15 m, wind and two moving magenta shutters (gates 3 and 6); level 3: 10 gates, 0.85 m, stronger wind, four faster shutters (gates 2, 4, 7, 9) and two blue floor-to-ceiling panels to fly round; the course is fixed by the seed, wind and shutter phases change every round or heat
- `gate_tag`: above each frame a black plate carries the gate's number in eight cells read left to right from the front: white = 1, black = 0; the first four cells are the number (most significant bit first), the last four its complement
- `flight`: pass every gate in order from its front (the side facing the start) to its back; a gate counts when the drone's centre crosses the gate's plane forwards at least 0.18 m inside all four edges; a gate passed out of order, backwards or too close to an edge does not count (you may turn and fly it again); the flight's time is the moment the centre crosses the last gate, measured from the start of the round
- `flight_end`: the flight ends when the drone finishes, touches anything (gate, panel, shutter, floor, wall), leaves the arena (-2 < x < 52, |y| < 5.8, 0.15 < z < 4.8 m) or reaches the round's time limit (race_timing.max_race_s)
- `placing`: a finished flight beats an unfinished one; two finished flights by time; unfinished flights by the gates passed, then by the earlier time at the last gate passed; a flight without a gate comes last
- `racing_format`: racing: three qualifying rounds on one course, each one flight from the start, a review after rounds 1 and 2; ranked by the best round (finish time), then the second and third best, then the gates passed; success = at least one finished round in a valid, complete event
- `duel_format`: duel: heats on one level-2 course; both drones fly each heat at the same time, each in a world of its own (no contact, the other drone is not visible); a heat goes to the better placed drone and ends as soon as its placing is decided; a tied heat counts for nobody; the first to two heat wins wins; both players review after every heat
- `action`: fly: a world-frame target velocity (vx, vy, vz) in m/s, a yaw rate in rad/s and a duration in game seconds; horizontal speed <= 4, |vz| <= 2, |yaw_rate| <= 1.5, 0.05 <= duration <= 1; it takes effect at the next 10 ms control step and replaces the one under way; after its duration the target is zero velocity and the heading is held (the drone brakes, it does not hold a position); the onboard controller tilts the drone and sets the rotor thrusts, it does not find gates or avoid obstacles
- `drone`: mass 0.35 kg; four rotors at (+-0.105, +-0.105) m, each at most 2.5 N with a 25 ms lag; linear drag 0.06 N per m/s of air speed; gravity 9.81 m/s^2; starts each round at (0, 0, 1.45) hovering and facing +x; physics 500 Hz, control 100 Hz
- `camera`: one FPV camera 0.12 m ahead of and 0.035 m above the drone's centre, looking along the body x axis (it tilts with the drone), vertical field of view 70 degrees, 320 x 240; /spec camera gives the mount, /spec cameras its world pose when /spec was read
- `observation`: your own telemetry (robot: position, body-to-world quaternion w x y z, world velocity, body angular velocity; command), the referee's calls (gates passed and their times in this unit, your results, standing, last heat with the opponent's result in the duel); no gate, panel or shutter coordinates and no seed
- `world`: {"+x": "forward along the course", "+y": "left", "+z": "up", "units": "SI"}
- Winner: the first player to win two heats wins (a tied heat counts for nobody and is re-flown); after five heats, more heat wins; then the better best heat result, the better second best, ...; then the winner of the last heat that had one; last resort (not from the rulebook): a lot drawn from the seed before the race. There are no draws.
- Deviation D1 (rulebook C.2, C.6): one flight through an open course (the C.2 'open loop to be flown once') per round or heat, not three laps of a closed circuit in 3 minutes; the time limit is 35 / 60 / 120 s for levels 1 / 2 / 3
- Deviation D2 (rulebook C.6.3): format racing: exactly 3 qualifying rounds, Option 1 ranking (the best result; the 2nd and 3rd results split ties) with one lap; a round's result is its finish time, or the gates passed and the time at the last gate when it did not finish (gates as the sectors C.2 allows)
- Deviation D3 (rulebook C.6.4, C.6.5): format duel: a race of two pilots instead of four, flown as successive final races (C.6.5.3): the first to win two heats wins; a tied heat is re-flown, at most 5 heats; then more heat wins, the C.6.3 a order (best, second best, ... heat result), the winner of the last heat that had one, and a seeded lot
- Deviation D5 (rulebook C.2, C.6.2): both drones start at the same point, each in a world of its own (no mid-air collision, no restarts for collisions, C.8.2), and neither sees the other
- Deviation D7 (rulebook C.7.2): a gate counts only when the drone's centre crosses its plane forwards at least 0.18 m inside every edge (the whole drone through the opening); a missed gate may be flown again
- Deviation D8 (rulebook C.7.3, C.9.1): any contact (gate, panel, shutter, floor, wall) ends the flight; leaving the arena (the safety line) ends it as in C.9.1
- Deviation D9 (rulebook C.2): added: moving shutters (levels 2, 3), wind (levels 2, 3) and floor-to-ceiling panels to fly round (level 3); the course is fixed by the seed, the wind and the shutter phases change every round

## Geometry and physics
Everything you need is in `game spec`: the world frame, the release point, the cameras (pose, field of view, size,
`focal_px`, `principal_point_px`; pinhole: u = cx + f * x_right / depth, v = cy - f * y_up / depth) and the
execution scatter. The physics engine is mujoco (CPU, 500 Hz rigid-body quadrotor with lagged rotors, linear drag in a seeded wind, kinematic shutters).
