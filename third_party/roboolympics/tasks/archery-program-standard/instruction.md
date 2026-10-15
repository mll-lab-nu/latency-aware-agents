# Target archery in the wind: program play (single)

You play target archery in the wind alone. Shoot a round of 4 ends x 3 arrows at a World Archery 122 cm 10-ring face (50, 70, 70, 90 m) in a seeded, gusty wind, reading the wind from windsocks in the camera images.
This task is generated from `sports/archery/sport.toml` (task `archery-program-standard`).

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
| `game shoot ELEV_DEG AZIM_DEG DRAW` | `POST /act {"type":"shoot", ...}` | one action (`game spec` lists the fields and ranges) |
| `game ready "WHY" "PLAN"` | `POST /ready {"why":"...","plan":"..."}` | your review after each of your actions |

The reply to an action also carries images of its result, in the format of `game observe` (`observation.image_png_b64` and `observation.tiles`; `game` saves them in `/app/output/views/`): the reply to each arrow shows the target close-up (600 x 600, the `target` camera) of the face with this end's arrows, the one just shot included (also the third arrow of an end, before the arrows are pulled).

Errors: 400 = an invalid action (nothing happens), 409 = not now (not your turn, or a review is due), 429 = one
action in flight, or more than 20 requests per second or 4 open requests from your seat (empty body: retry more slowly).

## Rules
- shoot a round of 4 ends x 3 arrows at a World Archery 122 cm 10-ring face (50, 70, 70, 90 m) in a seeded, gusty wind, reading the wind from windsocks in the camera images.
- `round`: 4 ends x 3 arrows at a World Archery 122 cm 10-ring face: end 1 at 50 m, ends 2 and 3 at 70 m, end 4 at 90 m; 120 points at most
- `distances_m`: [50.0, 70.0, 70.0, 90.0]
- `arrows_per_end`: 3
- `scoring`: each arrow scores the zone it lands in: 10 (gold centre) down to 1 (outer white); X = the inner 10, counts 10; an arrow touching a dividing line scores the higher zone; off the face = M (0)
- `face`: zones 6.1 cm wide: 1-2 white, 3-4 black, 5-6 blue, 7-8 red, 9-10 gold; the gold centre at (D, 0, 1.30); on a 1.30 m square straw boss; white lines cross the range every 10 m
- `face_centre_z_m`: 1.3
- `world`: {"origin": "the ground at the middle of the shooting line", "+x": "down the range (the target face is the vertical plane x = D)", "+y": "to your left", "+z": "up"}
- `launch_point_m`: [0.0, 0.0, 1.5]
- `shot`: shoot ELEV AZIM DRAW: elev_deg = the arrow's launch direction above horizontal (-10 to 30); azim_deg = its heading, 0 = straight down the range, + = to your left (-20 to 20); draw = the draw fraction (0.5 to 1.0): launch speed = 68 m/s x draw
- `full_draw_speed_m_s`: 68.0
- `execution_scatter_1sigma`: {"elev_deg": 0.03, "azim_deg": 0.03, "speed_rel": 0.003}
- `arrow_flight`: a point mass (20 g, shaft diameter 5.5 mm) under gravity (9.81 m/s^2) and quadratic air drag on its velocity relative to the air: a = -g z - K |v - w| (v - w), K = rho Cd A / (2 m) (rho = 1.2 kg/m^3, Cd = 1.5 on the shaft cross-section A = pi d^2 / 4), w = the wind
- `drag_K_per_m`: 0.0010691
- `wind`: horizontal, uniform over the range and constant during a flight; it changes from end to end and gusts from arrow to arrow; the windsocks always show the wind of the arrow you are about to shoot
- `windsocks`: a windsock points downwind; its angle below horizontal is 90 deg x (1 - speed / 10 m/s), horizontal at 10 m/s or more; 1.2 m long, 5 stripes, orange at the mouth; three on 4 m poles at y = +3.5 and x = D/4, D/2, 3D/4, a 0.6-size one on a 3.1 m pole at (D, 1.1); `windsocks_by_distance` gives their mouths
- `windsock_full_speed_m_s`: 10.0
- `images`: one PNG of four camera tiles (observation.tiles): archer (1000 x 600, from just behind the bow down the range, with the windsocks), scope (600 x 600, from the arrow rest centred on the gold), side (800 x 500, from 12 m right of the range at half the distance, looking across at the middle windsock; image right = towards the target), target (600 x 600, a close-up of the face with your arrows of this end: nock and fletching magenta = arrow 1, lime = 2, cyan = 3; arrows are pulled after each end); no coordinates and no wind numbers; `cameras_by_distance` gives every camera's pose and intrinsics (pinholes, square pixels)
- `act_reply`: after each arrow: its score (X, 10 ... 1, M), the spotter's call (the clock position of an arrow on the face, e.g. '9 at 2 o'clock', or where a miss went), the target close-up of the face with this end's arrows (observation.image_png_b64, as observe; the third arrow included, before the arrows are pulled) and, after the third arrow, the end's scorecard
- You play alone: success = a valid, complete match with a score of at least 110.
- Deviation D1 (rulebook Book 3, standard rounds): a short custom round (4 ends x 3 arrows at 50, 70, 70 and 90 m) instead of a standard round; no practice ends
- Deviation D2 (rulebook 11.2): the arrows are shot by a release primitive with a stated scatter; time is a chess clock (60 minutes, running while the round is on) instead of end and arrow time limits

## Geometry and physics
Everything you need is in `game spec`: the world frame, the release point, the cameras (pose, field of view, size,
`focal_px`, `principal_point_px`; pinhole: u = cx + f * x_right / depth, v = cy - f * y_up / depth) and the
execution scatter. The physics engine is deterministic point-mass arrow ballistics (RK4, 1 ms) with quadratic drag in the wind + MuJoCo scene.
