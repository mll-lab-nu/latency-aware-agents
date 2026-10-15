# Darts (501): program play (501)

You play darts (501) against a scripted opponent (you are seat `A`). Best of 3 legs of 501 double-out at a regulation board, each dart a release command (yaw, pitch, speed) with seeded scatter under gravity and quadratic drag; format 501: you against a built-in scripted thrower (opponent level 1-4, `game spec` opponent), format duel: two players at one board.
This task is generated from `sports/darts/sport.toml` (task `darts-501-program-standard`).

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
| `game throw YAW PITCH SPEED` | `POST /act {"type":"throw", ...}` | one action (`game spec` lists the fields and ranges) |
| `game ready "WHY" "PLAN"` | `POST /ready {"why":"...","plan":"..."}` | your review after each of your actions |

The reply to an action also carries images of its result, in the format of `game observe` (`observation.image_png_b64` and `observation.tiles`; `game` saves them in `/app/output/views/`): the reply to each dart shows the board close-up (the `board` camera, 900 x 900) with the darts of your current visit, the one just thrown included.

Errors: 400 = an invalid action (nothing happens), 409 = not now (not your turn, or a review is due), 429 = one
action in flight, or more than 20 requests per second or 4 open requests from your seat (empty body: retry more slowly).

## Rules
- best of 3 legs of 501 double-out at a regulation board, each dart a release command (yaw, pitch, speed) with seeded scatter under gravity and quadratic drag; format 501: you against a built-in scripted thrower (opponent level 1-4, `game spec` opponent), format duel: two players at one board.
- `match`: best of 3 legs of 501 double-out; visits of 3 darts; the players take turns, one visit each; the first player throws first in legs 1 and 3, the other player in leg 2
- `rules_501`: each player starts each leg on 501 and each dart subtracts its score; the dart that brings the score to exactly 0 must land in a double or the bull (the bull counts as a double); the first player to check out wins the leg; a dart that leaves you on 1, below 0, or on 0 without a double is a bust: the visit ends at once and the score goes back to its value at the start of the visit
- `visit_cap`: 30 visits each per leg; then the lower remaining score wins the leg; if level, the player with more chess-clock time left; if level too, the player who did not start the leg
- `release_point_m`: [0.4, 0.0, 1.68]
- `throw`: {"yaw_deg": "horizontal release direction: 0 = along +x (perpendicular to the board), + = to your left", "pitch_deg": "elevation above horizontal, + = up", "speed_m_s": "release speed"}
- `limits`: {"yaw_deg": [-20.0, 20.0], "pitch_deg": [-10.0, 40.0], "speed_m_s": [5.0, 7.0]}
- `execution_scatter_1sigma`: {"yaw_deg": 0.35, "pitch_deg": 0.3, "speed_rel": 0.008}
- `scatter_draws`: each player's execution scatter is independent of the opponent's within a match (its own seeded stream: your deviations cannot be read off the opponent's darts); the two matches of a duel pair (the same seed, the first player swapped) swap the scatter sequences, so over the pair both players meet the same draws
- `gravity_m_s2`: 9.81
- `drag`: a = -k |v| v, k fixed for the match by the seed in [0.02, 0.05] 1/m (not given)
- `sticks`: where the dart's tip reaches the board plane; darts do not deflect off wires or other darts
- `board_face_plane_x_m`: 2.37
- `bull_centre_m`: [2.37, 0.0, 1.73]
- `radii_mm`: {"bull": 6.35, "outer_bull": 15.9, "treble_ring": [99.0, 107.0], "double_ring": [162.0, 170.0], "board_edge": 225.0}
- `scoring`: bull 50 (a double), outer bull 25, treble ring 3x, double ring 2x, the other beds 1x the segment number; outside the double ring: no score; beyond the board edge: missed the board
- `segments_clockwise_from_top`: [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
- `segment_width_deg`: 18
- `world`: {"origin": "floor, centre of the throw line", "+x": "towards the board", "+y": "to your left", "+z": "up"}
- `board_coordinates`: u = to your right (= -y), v = up (= z - 1.73), metres from the bull centre
- `images`: one PNG of three tiles (observation.tiles: x, y, width, height): oche (your view from the throw line, looking at the bull), board (straight at the board face from 0.9 m; image right = your right, image up = up) and side (from your right, y = -3.4 m; image right = towards the board)
- `board_darts`: the board shows the darts of the latest visit, yours or your opponent's, with yellow (dart 1), cyan (dart 2) and magenta (dart 3) flights; they are pulled out when the next visit begins; a dart that missed the board is not shown
- `dart_model`: a stuck dart's tip is 6 mm inside the board; the dart lies along its arrival velocity; its flight (3 cm cross, 3.8 cm long) is centred 13.5 cm behind the tip
- `act_reply`: after each dart: the caller's call, the leg, visit and dart number, and the board close-up with the darts of your current visit, the one just thrown included (observation.image_png_b64, as observe); the board of observe shows the latest visit, which after your third dart may soon be your opponent's
- `caller`: the state lists the caller's calls of the darts thrown since your previous dart (darts_since_your_last) and of your last dart (your_last_dart, with the command you sent)
- Winner: the first player to win 2 legs (best of 3) wins the match; within a leg the first player to check out wins it; a leg nobody has checked out after 30 visits each goes to the lower remaining score, then (last resort, not from the rulebook) to the player with more chess-clock time left, then to the player who did not start the leg. There are no draws.
- Deviation D1 (rulebook WDF 1.02, 13.01): the darts are thrown by a release mechanism with a stated scatter, not by hand, and there are no practice darts
- Deviation D2 (rulebook WDF 1.05-1.06, 3.02-3.03): darts always stick where the point reaches the board: no bounce-outs, fall-outs or wire deflections
- Deviation D3 (rulebook WDF 12.01-12.03): the throw order is fixed instead of a throw for the bull: in format 501 the opponent throws first in legs 1 and 3; in the duel the configured first player (`game spec` first_player) throws first in legs 1 and 3, and both orders are played
- Deviation D4 (rulebook WDF 12 (play continues until a player checks out)): a cost limit: a leg nobody has checked out after 30 visits each goes to the lower remaining score; if level, to the player with more chess-clock time left (the scripted opponent uses no clock time, so it wins a level leg); if the clocks are level too, to the player who did not start the leg
- Deviation D5 (rulebook WDF 1.02 (the darts are thrown by hand)): the release mechanism (D1) is declared at a human elite throw, not a machine's: release speed 5 to 7 m/s, an estimate (no dart-throwing robot publishes its release speed; a skilled thrower releases at about 5.1 to 5.5 m/s, e.g. arXiv:2604.01130, https://arxiv.org/abs/2604.01130, after Venkadesan et al.); until darts-5: 4 to 9 m/s, where 9 m/s is far beyond a human throw and below about 5 m/s no dart reaches the top of the board at all. The cap holds for the commanded speed, the executed one scatters around it; the execution scatter is an estimate too

## Geometry and physics
Everything you need is in `game spec`: the world frame, the release point, the cameras (pose, field of view, size,
`focal_px`, `principal_point_px`; pinhole: u = cx + f * x_right / depth, v = cy - f * y_up / depth) and the
execution scatter. The physics engine is analytic dart flight (RK4, gravity + quadratic drag, darts.py) with a MuJoCo scene for the cameras.
