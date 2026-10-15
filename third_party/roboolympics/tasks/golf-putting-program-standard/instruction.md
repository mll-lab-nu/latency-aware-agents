# Mini-golf putting: program play (single)

You play mini-golf putting alone. Play a round of 4 mini-golf putting holes (par 2, 2, 2, 3; par 9) on sloped greens enclosed by walls, one hole with an obstacle block; each stroke is an aim and a speed chosen from the camera images, and the fewest strokes is best.
This task is generated from `sports/golf/sport.toml` (task `golf-putting-program-standard`).

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
| `game putt AIM_DEG SPEED` | `POST /act {"type":"putt", ...}` | one action (`game spec` lists the fields and ranges) |
| `game ready "WHY" "PLAN"` | `POST /ready {"why":"...","plan":"..."}` | your review after each of your actions |

Errors: 400 = an invalid action (nothing happens), 409 = not now (not your turn, or a review is due), 429 = one
action in flight, or more than 20 requests per second or 4 open requests from your seat (empty body: retry more slowly).

## Rules
- play a round of 4 mini-golf putting holes (par 2, 2, 2, 3; par 9) on sloped greens enclosed by walls, one hole with an obstacle block; each stroke is an aim and a speed chosen from the camera images, and the fewest strokes is best.
- `round`: 4 holes, par 2, 2, 2, 3 (par 9); each hole starts from its own fixed spot and the ball is played from where it stopped
- `scoring`: the score is the total number of strokes, fewer is better (WMF 9.1); a hole not holed after 6 strokes is picked up and scores 7
- `success_single`: success = the round complete in 9 strokes or fewer (par); fewer strokes is better
- `putt`: {'type':'putt','aim_deg':A,'speed':V}: aim_deg = where the putter face points, i.e. the ball's start direction, in the green's frame: 0 = +x (down the green, away from the end the ball starts at), + = counter-clockwise seen from above (90 = +y), -180 to 180, not relative to the cup; walls can be used for bank shots; speed = the ball's initial rolling speed, m/s, 0.05 to 6.0
- `execution_scatter_1sigma`: {"aim_deg": 0.3, "speed_rel": 0.015}
- `green`: each green has its own frame: x in [0, L] along the green, y in [-W/2, W/2] across it, z up (sizes in match.holes); the surface is smooth but not flat: tilts, bumps, hollows and ridges of up to a few centimetres to a decimetre; the slopes are not given and change with the seed
- `ball_physics`: the ball (radius 21.35 mm) rolls without slipping: slopes accelerate it downhill ((5/7) g sin(slope), g = 9.81 m/s^2); rolling resistance slows it at a constant rate (stimpmeter 10 ft: on a flat green a ball starting at 1.83 m/s stops after 3.05 m); it comes to rest when it is slow and the slope cannot move it
- `walls`: walls and blocks: the ball rebounds, keeping about 70 % of its speed into the wall and about 92 % along it
- `cup`: the cup (radius 54 mm): a ball whose centre passes over the cup drops if it is slow enough (straight at the centre up to about 1.7 m/s, less when it only catches the edge); faster, it lips out or rolls over the edge
- `image_layout`: one PNG of 2000 x 1000 (the rectangles are image.tiles in /spec and observation.tiles): behind (800 x 500, top left): 2.0 m behind the ball along -x, 1.1 m above the green there, looking along +x, pitched 14 deg down; side (800 x 500, bottom left): fixed per hole beyond the green's left wall at mid-green (match.holes[].side_camera_pos_m), looking along -y, pitched 12 deg down, image right = +x; overhead (1200 x 600, right): straight down over the whole green from (L/2, 0, match.holes[].overhead_camera_height_m), image right = +x, image up = +y, with the path of your previous stroke on this hole as yellow dots (one every 0.1 s). White grid lines every 0.5 m follow the surface; mowing stripes are 1 m wide; the cup is a black disc with a white rim and a yellow flagstick with a red flag; obstacle blocks are red. No ball or cup coordinates and no slope numbers are given
- `hearing`: after each stroke you are told its result (holed, stopped, lipped out, rolled over the edge of the cup), the wall or block hits and the roll time; your scorecard is in every observation
- You play alone: success = a valid, complete match with a score of at most 9.
- Deviation D2 (rulebook 11.3): no time limit per stroke: the player has a chess clock instead (clock out: the round ends there, the unfinished hole and the holes not played score 7 each)

## Geometry and physics
Everything you need is in `game spec`: the world frame, the release point, the cameras (pose, field of view, size,
`focal_px`, `principal_point_px`; pinhole: u = cx + f * x_right / depth, v = cy - f * y_up / depth) and the
execution scatter. The physics engine is a deterministic rolling-ball model on a height field (pure Python, fixed step; MuJoCo draws the scene).
