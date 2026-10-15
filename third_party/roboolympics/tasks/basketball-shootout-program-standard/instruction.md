# Robot basketball shootout: program play (single)

You play robot basketball shootout alone. A declared shooting machine (a launcher like Toyota's CUE robots) on a 1 m pedestal shoots from five spots, mid-range to beyond the three-point line, past a static defender, finding the hoop in the camera images; every make counts 1.
This task is generated from `sports/basketball/sport.toml` (task `basketball-shootout-program-standard`).

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
| `game shoot ANGLE_DEG SPEED SPIN_HZ YAW_DEG` | `POST /act {"type":"shoot", ...}` | one action (`game spec` lists the fields and ranges) |
| `game ready "WHY" "PLAN"` | `POST /ready {"why":"...","plan":"..."}` | your review after each of your actions |

The reply to an action also carries images of its result, in the format of `game observe` (`observation.image_png_b64` and `observation.tiles`; `game` saves them in `/app/output/views/`): the reply to each shot shows a picture strip: 2 rows x 4 columns of 400 x 300 tiles, 100 ms apart around the ball's closest approach to the hoop (each tile's `label`: ms from it); row 1: the rim camera (a side view of the hoop), row 2: your shooter_zoom camera.

Errors: 400 = an invalid action (nothing happens), 409 = not now (not your turn, or a review is due), 429 = one
action in flight, or more than 20 requests per second or 4 open requests from your seat (empty body: retry more slowly).

## Rules
- a declared shooting machine (a launcher like Toyota's CUE robots) on a 1 m pedestal shoots from five spots, mid-range to beyond the three-point line, past a static defender, finding the hoop in the camera images; every make counts 1.
- `shootout`: 10 shots, two from each of five spots in order (spot 1 twice, then spot 2 twice, ... spot 5 twice), from mid-range to beyond the three-point line; every make counts 1
- `shot`: shoot ANGLE SPEED SPIN YAW: angle_deg = release elevation above horizontal (20 to 80); speed = release speed in m/s (3 to 14); spin_hz = revolutions per second, + = backspin, - = topspin (-5 to 5); yaw_deg = release direction relative to the shooter's current heading, + = to the left, counter-clockwise from above (-35 to 35). The shooter is parked facing roughly, not exactly, towards the hoop (a few degrees off, different at every spot)
- `limits`: {"angle_deg": [20.0, 80.0], "speed": [3.0, 14.0], "spin_hz": [-5.0, 5.0], "yaw_deg": [-35.0, 35.0]}
- `execution_scatter_1sigma`: {"speed_rel": 0.005, "angle_deg": 0.4, "yaw_deg": 0.4, "spin_hz": 0.15}
- `execution_scatter_source`: an estimate bounded by a real shooting robot: Toyota's CUE3 made 2,020 free throws in a row (Guinness World Records, May 2019), so its make rate from the free-throw line is at least 99.85 % (one-sided 95 % bound); with this scatter the machine, aimed at the centre of the make region, makes 99.87 % from the free-throw line (6000 seeded draws), so the declared scatter is about the largest one consistent with that record; it is drawn per shot from a seeded generator
- `shooter`: a shooting machine (D7) on a 1.0 m pedestal: a turret turns its launch tube (the yaw), the tube sets the elevation, and the machine releases the ball from the tube's muzzle with your release parameters plus its execution scatter; nothing of the machine touches the ball after the release; its frame: origin on the floor at the pedestal axis, +x = its heading, +y = its left, z up
- `release_point_in_shooter_frame_m`: [0.5, 0.0, 2.17]
- `release`: the release point is the muzzle; its envelope: 0.50 m from the pedestal axis, 2.17 m above the floor, turned by the yaw (the tube pivots about the muzzle, so the elevation does not move it): shooter_pos + 0.50 (cos(h + yaw), sin(h + yaw)) at height 2.17, h = the heading; the ball leaves 0.45 s after the shot command (the machine aims first)
- `throw_to_release_s`: 0.45
- `speed_cap`: 14 m/s, the machine's declared maximum release speed: within a real shooting robot's reach (Toyota's CUE6 made a 24.55 m shot, Guinness World Records 2024-09-26, which needs at least about 20 m/s from this release height with this sport's drag); the farthest spot needs at most about 9.5 m/s
- `ball`: size 7: radius 0.1195 m, mass 0.62 kg (thin shell)
- `ball_radius_m`: 0.1195
- `ball_mass_kg`: 0.62
- `flight`: gravity 9.81 m/s^2 (down); air drag F = -drag_coef |v| v (N); Magnus F = magnus_coef (omega x v) (N), omega in rad/s; backspin: omega = 2 pi spin_hz (d x z), d = the horizontal release direction
- `gravity_m_s2`: 9.81
- `drag_coef_n_s2_m2`: 0.0145355
- `magnus_coef`: 0.00418165
- `contacts`: the floor, the rim, the bracket and the backboard are hard and bouncy (restitution about 0.8 floor, 0.6 rim, 0.65 backboard, 0.3 defender); the net is visual only
- `make`: a make: the ball drops through the rim from above (FIBA Art. 16.1.1); it counts 1, from any spot
- `defender`: a static defender stands on the line from you to the hoop, min(3.0 m, 0.55 x your distance to the hoop) from you (axis to axis), its hands up (top of the hands at 2.545 m); it never moves; a ball whose first contact is the defender is blocked and does not count
- `court`: NBA court markings and equipment sizes: the hoop's centre at (0, 0, 3.048); the floor z = 0; the court is y < 0, the baseline y = +1.60 behind the backboard; rim inner diameter 0.4572 m (orange, 16 mm tube); backboard 1.83 x 1.07 m, its front face 0.381 m behind the rim centre, bottom edge 2.90 m, red border and a red 0.61 x 0.45 m shooter's square above the rim; white 5 cm lines: lane 4.88 m wide, free-throw line y = -4.19 with its 1.8 m circle, three-point arc 7.24 m from the hoop centre (straight 6.71 m from the centre line in the corners), sidelines x = +-7.62
- `hoop_centre_m`: [0.0, 0.0, 3.048]
- `rim_inner_diameter_m`: 0.4572
- `world`: {"origin": "the floor below the hoop's centre", "+y": "from the court towards the baseline (the shooter's half is y < 0)", "+z": "up"}
- `images`: one PNG of four 640 x 480 camera tiles (observation.tiles): shooter_wide and shooter_zoom (both on your shooter, behind and above the arm, looking along its heading, pitched up 8 deg; vertical field of view 55 and 16 deg), broadcast and baseline (fixed, overlooking the court); no hoop, ball or shooter coordinates; `cameras` in /spec gives every camera's pose (the two shooter cameras in the shooter's frame) and intrinsics (pinholes, square pixels)
- `act_reply`: after each shot: its result (made, rim, backboard, blocked, airball), how it went in, what the ball touched, the makes and shots left, and a picture strip (observation.image_png_b64 with observation.tiles, as observe; `game` saves it as observation.image_file): 2 rows x 4 columns of 400 x 300 tiles, 100 ms apart around the ball's closest approach to the hoop; row 1: the rim camera (a side view of the hoop), row 2: your shooter_zoom camera
- `not_given`: the shooter's position and heading and the spots' distances: judge them from the images
- You play alone: success = a valid, complete match with a score of at least 6.
- Deviation D1 (rulebook FIBA 3x3 Shoot-out Contest): a shooting contest instead of a game: 10 shots, two from each of five spots
- Deviation D2 (rulebook Art. 16.2.1): every make counts 1, from any spot, instead of 2 or 3 points by the release area
- Deviation D4 (rulebook Art. 33 (contact)): a static defender with its hands up stands on the line from the shooter to the hoop; a ball that touches it first is blocked and does not count (instead of live defence)
- Deviation D5 (rulebook Art. 29 (shot clock)): time is a chess clock (60 minutes, running while the shootout is on) instead of a shot clock
- Deviation D7 (rulebook Art. 4 (teams: the players)): the shooter is a declared shooting machine, like Toyota's CUE basketball robots, not a player: you set its release (yaw, elevation, speed up to 14 m/s, spin) and it releases the ball from its muzzle with a stated execution scatter; 14 m/s is within what a real shooting robot reaches (CUE6's 24.55 m record shot needs about 20 m/s with this sport's drag), while a robot arm's hand (a Franka Panda: about 2 m/s) could not reach the hoop from any spot

## Geometry and physics
Everything you need is in `game spec`: the world frame, the release point, the cameras (pose, field of view, size,
`focal_px`, `principal_point_px`; pinhole: u = cx + f * x_right / depth, v = cy - f * y_up / depth) and the
execution scatter. The physics engine is MuJoCo 3.3.7 (ball with quadratic drag and Magnus force; rim, backboard, floor and defender contacts).
