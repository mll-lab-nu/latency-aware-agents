# Ten-pin bowling: program play (single)

You play ten-pin bowling alone. Bowl a 10-frame game of ten-pin bowling on a regulation MuJoCo lane (oiled front end, dry back end, 10 rigid pins, a 7 kg ball), each ball a release (board, launch angle, speed, rev rate, axis rotation) chosen from the camera images.
This task is generated from `sports/bowling/sport.toml` (task `bowling-program-standard`).

## Time
Nothing moves while you think: the world waits for the player whose turn it is. Each player has a chess clock of 60 minutes of real time, running only on its own turn (thinking, tool calls, your programs and your reviews all count). **If your clock runs out, you lose.**

After each of your actions (except one that ends the match), send `game ready "WHY" "PLAN"` (a short review) before the game goes on.
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
| `game wait` | `POST /wait {"timeout_s": 60}` | block until it is your turn (or the match is over) |
| `game observe` | `POST /observe {}` | the state and the camera images (one PNG of tiles: `observation.tiles`) |
| `game act JSON` | `POST /act JSON` | one action |
| `game roll BOARD ANGLE_DEG SPEED RPM AXIS_DEG` | `POST /act {"type":"roll", ...}` | one action (`game spec` lists the fields and ranges) |
| `game ready "WHY" "PLAN"` | `POST /ready {"why":"...","plan":"..."}` | your review after each of your actions |

Errors: 400 = an invalid action (nothing happens), 409 = not now (not your turn, or a review is due), 429 = one
action in flight, or more than 20 requests per second or 4 open requests from your seat (empty body: retry more slowly).

## Rules
- bowl a 10-frame game of ten-pin bowling on a regulation MuJoCo lane (oiled front end, dry back end, 10 rigid pins, a 7 kg ball), each ball a release (board, launch angle, speed, rev rate, axis rotation) chosen from the camera images.
- `game`: one game of 10 frames per player; standard ten-pin scoring: a strike scores 10 plus your next two balls, a spare 10 plus your next ball; a strike or spare in the 10th frame gives bonus balls (three balls at most); 300 is perfect
- `deck`: each frame's first ball (and in the 10th frame each ball after a strike or spare) gets a fresh rack; after the first ball of a frame the fallen pins are swept away and the pins still standing stay where they are, even if they were moved
- `pin_down`: a pin counts as down if it tilts more than 20 degrees or leaves the lane surface (gutter, pit, kickbacks); a pin standing anywhere on the lane counts as standing (USBC 7b)
- `gutter_ball`: a ball that leaves the lane before it reaches the pins scores no pins, and the pins it moved are respotted (USBC 6b.1)
- `roll`: {'type':'roll','board':B,'angle_deg':A,'speed':V,'rpm':R,'axis_deg':X}: board = where the ball's centre is laid down, 35 cm past the foul line, in boards from the RIGHT gutter (39 boards of 27.03 mm; 1 = rightmost, 20 = centre, 39 = leftmost; fractions allowed; 1 to 39); angle_deg = launch direction from the lane's axis, + = towards the left (-10 to 10); speed = ball speed at release, m/s (4.47 to 10.73, the E.A.R.L. bowling robot's range, D5); rpm = rev rate, revolutions per minute (0 to 700); axis_deg = axis rotation: the spin axis is horizontal, 0 = end over end (no hook), 90 = all side roll; + hooks left, - hooks right (-90 to 90)
- `execution_scatter_1sigma`: {"board": 1.0, "angle_deg": 0.24, "speed_rel": 0.015, "rpm_rel": 0.04, "axis_deg": 2.0}
- `lane`: the front of the lane is oiled (the ball skids), the back end is dry (a spinning ball hooks, then rolls); the oil runs further down the middle boards than the outside boards; the pattern and the friction are not given and change with the seed
- `world`: {"origin": "the foul line at the centre of the lane, on the lane surface", "+x": "down the lane towards the pins (head pin at 18.288 m)", "+y": "across the lane, to the left", "+z": "up"}
- `image_layout`: one PNG of four views (their rectangles are `image.tiles` in /spec and `observation.tiles`): approach (the bowler's eye behind the foul line), pindeck (the pins standing now), pindeck_after (the deck right after your previous ball settled, before the sweep; black before your first ball) and lane (the lane from straight above in four stacked 5 m panels, image right = towards the pins, image up = left; the path of your previous ball as yellow dots, one every 1/30 s); no ball or pin coordinates are given
- `hearing`: after each ball you are told how many pins fell (not which ones)
- You play alone: success = a valid, complete match with a score of at least 200.
- Deviation D1 (rulebook 6a): a pin counts as down if it tilts more than 20 degrees or leaves the lane surface (gutter, pit, kickbacks); a pin standing anywhere on the lane counts as standing (USBC 7b)
- Deviation D5 (rulebook Chapter 2 (the bowler delivers the ball)): the ball is delivered by a declared release machine, like the United States Bowling Congress's E.A.R.L. bowling robot (Enhanced Automated Robotic Launcher, USBC equipment specifications, https://bowl.com/equipment-specifications2024/e-a-r-l-the-robot), not by a bowler: you set its release and it rolls the ball with a stated execution scatter. Release speed 4.47 to 10.73 m/s = E.A.R.L.'s 10 to 24 mph (until bowling-3: 3 to 11 m/s, beyond the robot's range at both ends); rev rate 0 to 700 rpm (E.A.R.L.: 50 to 900 rpm; rates below 50 rpm, a ball with practically no revs, are kept, an allowance beyond the robot); the caps hold for the commanded release, the executed one scatters around it; the execution scatter is an estimate, not a number of E.A.R.L.'s

## Geometry and physics
Everything you need is in `game spec`: the world frame, the release point, the cameras (pose, field of view, size,
`focal_px`, `principal_point_px`; pinhole: u = cx + f * x_right / depth, v = cy - f * y_up / depth) and the
execution scatter. The physics engine is mujoco.
