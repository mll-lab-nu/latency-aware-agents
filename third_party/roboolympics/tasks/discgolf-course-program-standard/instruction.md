# Disc golf: program play (single)

You play disc golf alone. Play a 6-hole disc golf round (par 20) through woods, around two doglegs, over OB water and past a mandatory in a seeded wind; a disc-launcher robot throws, each throw a release (disc, speed, launch angle, nose angle, bank, spin, aim) chosen from the camera images, and the disc flies with a deterministic aerodynamic model (lift, drag and pitching moment of Hummel 2003, gyroscopic precession, spin-down).
This task is generated from `sports/discgolf/sport.toml` (task `discgolf-course-program-standard`).

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
| `game throw DISC SPEED LAUNCH_DEG NOSE_DEG BANK_DEG SPIN_RPS AIM_DEG` | `POST /act {"type":"throw", ...}` | one action (`game spec` lists the fields and ranges) |
| `game ready "WHY" "PLAN"` | `POST /ready {"why":"...","plan":"..."}` | your review after each of your actions |

Errors: 400 = an invalid action (nothing happens), 409 = not now (not your turn, or a review is due), 429 = one
action in flight, or more than 20 requests per second or 4 open requests from your seat (empty body: retry more slowly).

## Rules
- play a 6-hole disc golf round (par 20) through woods, around two doglegs, over OB water and past a mandatory in a seeded wind; a disc-launcher robot throws, each throw a release (disc, speed, launch angle, nose angle, bank, spin, aim) chosen from the camera images, and the disc flies with a deterministic aerodynamic model (lift, drag and pitching moment of Hummel 2003, gyroscopic precession, spin-down).
- `round`: 6 holes, par 3, 3, 4, 3, 3, 4 (par 20); stroke play (PDGA 808): the score is the number of throws plus penalty throws, fewer is better; every hole's tee sign (the basket, the course limits, the OB areas, the mandatory, the ground, the overview's centre and scale) is /spec match.holes
- `tee_and_lie`: each hole starts at its tee, (0, 0) (802.04); the next throw is made from where the disc came to rest, below it if it rests above the ground (802.05, 805.01)
- `out_of_bounds`: a disc that comes to rest outside the course limits or in an OB area (the water) is out of bounds: one penalty throw, and the next lie is the point where it was last in bounds (806.02.B)
- `mandatory`: on a hole with a mandatory (match.holes[].mandatory), until your disc has passed it on the correct side, a throw that crosses the restricted line misses it: one penalty throw, and the next lie is the drop zone (804.01)
- `holing_out`: the disc is holed when its centre comes within catch_radius_m of the pole between 0.70 and 1.40 m above the basket's ground (or drops into the tray from above) at no more than max_catch_speed_m_s (/spec contacts.basket); faster, the chains spit it out and it drops at the basket (807)
- `cap`: a hole not holed out after par + 3 throws (penalty throws included) ends there and scores par + 4
- `single_success`: the single round (no opponent) succeeds when it is complete in 21 throws or fewer (par 20 + 1, penalty throws included): lower is better, so the success line is a score of AT MOST 21
- `throw`: {'type':'throw','disc':D,'speed':V,'launch_deg':A,'nose_deg':N,'bank_deg':B,'spin_rps':S,'aim_deg':H}: disc = putter, midrange or driver (coefficients in /spec discs); speed = release speed, m/s (5 to 30); launch_deg = the velocity's angle above horizontal (-5 to 35); nose_deg = the disc's pitch above the velocity, its angle of attack at release in still air (-10 to 15); bank_deg = the disc's roll about the flight direction, + = its left edge down seen from behind (hyzer for a clockwise spin) (-60 to 60); spin_rps = revolutions per second, + = clockwise seen from above (a right-hand backhand), at least 4 in magnitude (-20 to 20); aim_deg = the heading of the release in the hole's frame, 0 = +x, counter-clockwise positive (-180 to 180), not relative to the basket; the launcher releases the disc 1.0 m above the ground at your lie
- `execution_scatter_1sigma`: {"aim_deg": 1.5, "launch_deg": 1.5, "nose_deg": 1.5, "bank_deg": 3.0, "speed_frac": 0.04, "spin_frac": 0.05}
- `flight`: the disc (mass, diameter, Izz in /spec discs) flies with the model of Hummel (2003; as in FrisPy): with u = v - w the velocity relative to the air, n the disc's unit normal, q = rho |u|^2 / 2 and the angle of attack alpha = atan2(-u.n, |u - (u.n) n|): lift q A (CL0 + CLa alpha) perpendicular to u towards n; drag q A (CD0 + CDa (alpha - ALPHA0)^2) along -u; pitching moment q A d (CM0 + CMa alpha) (+ = nose up) about -(n x e), e = the unit vector of u's component in the disc's plane; spin-down torque q A d CS (w_s d / (2 |u|)) about n; the normal precesses as dn/dt = tau / (Izz w_s); RK4, dt 0.01 s. A clockwise disc with a nose-up moment (slow) rolls left and fades left; fast, the moment is nose-down and it turns right
- `wind`: uniform and horizontal over the hole, constant during a flight; each hole has its own wind and it gusts from throw to throw; the windsock (/spec cameras.wind) always shows the wind of the throw you are about to make: it points downwind and hangs at 90 deg x (1 - speed / 10 m/s) below horizontal; no wind numbers are given
- `trees_basket_ground`: trees are vertical cylinders, 12 to 22 m tall: a trunk (radius 0.30 m) from the ground to the top, the canopy (the green disc seen from above) from 2.5 m to the top; a trunk bounces the disc, a canopy catches it and it falls; the ground is a tilt plus Gaussian hills (match.holes[].ground); on landing the disc skids (v_h - mu_impact v_z)^2 / (2 mu_kinetic g); constants in /spec contacts and air
- `world`: each hole has its own frame: the tee at (0, 0), x down the fairway, y to the left, z up (metres)
- `image_layout`: one PNG (its rectangles are `image.tiles` in /spec and `observation.tiles`): overview (top, 1400 x 700, orthographic straight down on the whole hole: trees as green discs, the basket yellow in the white 10 m circle, OB lines white, OB water blue, a mandatory's line red and its drop zone a yellow square, your launcher orange at your lie, the flight of your previous throw on this hole as yellow dots), tee (bottom left, 1000 x 600, a pinhole camera behind your launcher looking at the basket) and wind (bottom right, 400 x 400, orthographic straight down on the windsock); poses and scales in /spec cameras; no tree coordinates or wind numbers are given
- `decision_point`: referee information in every state: the hole, its par, the throw you are about to make (penalty throws counted), your lie and its ground height, the basket and its distance and heading, whether the mandatory is cleared, and your previous throw's result on this hole
- You play alone: success = a valid, complete match with a score of at most 21.
- Deviation D2 (rulebook 806.02): out of bounds always uses option 806.02.B (the next lie is the point where the disc was last in bounds), without the 1 m relief
- Deviation D3 (rulebook 808): a hole not holed out after par + 3 throws (penalty throws included) ends there and scores par + 4 (not in the rules; it bounds the round's length)
- Deviation D5 (rulebook 802.03): no time limit per throw: the player has a chess clock instead; when it runs out the round ends there, and the unfinished hole and the holes not played score par + 4 each

## Geometry and physics
Everything you need is in `game spec`: the world frame, the release point, the cameras (pose, field of view, size,
`focal_px`, `principal_point_px`; pinhole: u = cx + f * x_right / depth, v = cy - f * y_up / depth) and the
execution scatter. The physics engine is disc-flight model (Hummel 2003, RK4, dt 0.01 s) with tree, basket and ground contact; MuJoCo scene for the views.
