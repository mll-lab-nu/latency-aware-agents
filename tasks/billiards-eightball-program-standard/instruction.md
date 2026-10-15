# Billiards: program play (eightball)

You play billiards against a scripted opponent (you are seat `A`). One rack of American 8-ball on a 7-ft pool table: break, then call every shot (a ball and a pocket, or safety), clear your group (solids or stripes) and pot the 8 into the pocket you call; you shoot by cue direction, cue speed and tip offset, and place the cue ball when you have ball in hand; in the `eightball` format the opponent is a scripted computer player, in the `duel` format another player.
This task is generated from `sports/billiards/sport.toml` (task `billiards-eightball-program-standard`).

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
| `game ready "WHY" "PLAN"` | `POST /ready {"why":"...","plan":"..."}` | your review after each of your actions |

Errors: 400 = an invalid action (nothing happens), 409 = not now (not your turn, or a review is due), 429 = one
action in flight, or more than 20 requests per second or 4 open requests from your seat (empty body: retry more slowly).

## Rules
- one rack of American 8-ball on a 7-ft pool table: break, then call every shot (a ball and a pocket, or safety), clear your group (solids or stripes) and pot the 8 into the pocket you call; you shoot by cue direction, cue speed and tip offset, and place the cue ball when you have ball in hand; in the `eightball` format the opponent is a scripted computer player, in the `duel` format another player.
- `world`: metres; the playing surface (inside the cushion noses) is 0 <= x <= W = 0.9906, 0 <= y <= L = 1.9812 with the cloth at z = 0; z up; balls have radius R = 0.028575, all the same mass
- `pockets`: six pockets (centres and radii in `game spec` table.pockets): lb (0, 0), rb (W, 0), lc (0, L/2), rc (W, L/2), lt (0, L), rt (W, L) (l/r: x = 0 / W; b/c/t: y = 0 (head end) / L/2 / L (foot end); the letters are not image directions); the openings are about 11.7 cm wide at the corners and 13.5 cm at the sides; side pockets accept only balls that come in fairly straight (within about 45 degrees of the cushion normal)
- `spots`: head spot (W/2, L/4), foot spot (W/2, 3L/4); the head string is the line y = L/4 and `above the head string` means y < L/4 (the line itself excluded); diamonds on the rails every L/8 along the long rails and every W/4 along the short rails
- `shot`: game act '{"type": "shoot", "aim_deg": 87.5, "speed": 2.0, "spin_x": 0.0, "spin_y": -0.2, "call": "3:rc"}' (HTTP POST /act with that body): the cue stick's state at impact, not a target. aim_deg: the direction the stick points and strikes, degrees counter-clockwise from +x (0 = +x, 90 = +y; taken modulo 360); with a centre hit the cue ball leaves in exactly this direction. speed: 0.3-7 m/s; a centre hit sends the cue ball off at about 1.54 x speed; on an empty table a centre-hit cue ball travels (cushions included) about 0.6 m at 0.3, 1.6 m at 0.5, 3.3 m at 1.0 and 5.6 m at 2.0. spin_x, spin_y (default 0): where the tip hits, in ball radii from the centre: spin_x > 0 right english, spin_y > 0 follow, spin_y < 0 draw; sqrt(spin_x^2 + spin_y^2) <= 0.5; sidespin turns the start direction a little (squirt: about 1.2 degrees clockwise seen from above for spin_x = 0.3)
- `ball_in_hand`: cue_x, cue_y (both or neither, only with ball in hand): place the cue ball first. It must lie on the table (R <= x <= W - R, R <= y <= L - R), not touch a ball, not lie in a pocket opening, and with the ball in hand above the head string lie at y < L/4. Without them the cue ball stays where it is; a potted cue ball goes on the head spot (or the nearest free point on the long centre line), or, in hand above the head string, on (W/2, L/4 - 0.05) (or the nearest free point there)
- `call`: required on every shot except the break, refused on the break: BALL:POCKET with a ball you may hit first (e.g. 3:rc, 8:lt), or safety. A refused action (HTTP 400) changes nothing
- `execution_noise_1sigma`: {"aim_deg": 0.06, "speed_rel": 0.02, "spin": 0.01, "note": "every shot of both players; the n-th shot of the game gets the same random draw whoever shoots it"}
- `physics_model`: sliding friction 0.2, rolling resistance 0.01, ball-ball restitution 0.95, cushion restitution 0.85; a ball struck without spin slides first and then rolls; after a full-ball hit a stun cue ball stops, a follow ball keeps rolling forward and a draw ball comes back; in a cut shot the object ball leaves along the line of centres at contact (ghost-ball aim: send the cue ball to the point 2R behind the object ball on the line from the pocket through the object ball), slightly thrown by friction
- `rack`: the 15 object balls in a triangle with the front ball on the foot spot, the 8 in the middle of the third row, one solid and one stripe in the back corners; the order of the others and a small jitter depend on the seed
- `break_rule`: the breaker has ball in hand above the head string; the break is not called; it is legal if a ball is potted or at least four object balls reach a cushion. After a legal break that pots a ball (the 8 included: it is spotted again) the breaker shoots again; the table is open after the break. After an illegal break the balls are racked again in the same order and the other player breaks. A scratch, or an object ball driven off the table, on an otherwise legal break is a foul: the other player has ball in hand above the head string. A spotted ball goes on the long centre line as close as possible to the foot spot, towards the foot rail; if that is blocked, as close as possible above the foot spot
- `called_shots`: you stay at the table only if the called ball is pocketed in the called pocket on a legal shot; other balls pocketed on a legal shot stay down and count for whoever's group they belong to; a safety ends your turn
- `groups`: on an open table you may hit any object ball except the 8 first, unless a group has been completely pocketed: then you may claim that group and shoot (and call) the 8. The groups are decided when a player legally pockets its called ball: that ball's group (solids 1-7 or stripes 9-15) becomes the shooter's, the other group the opponent's
- `fouls`: after the break: the cue ball is potted; an object ball is driven off the table; no ball is hit; the first ball hit is not one you may hit first; after the first contact no ball is potted and no ball reaches a cushion; with ball in hand above the head string, the first ball contacted lies above the head string and the cue ball had not crossed the head string before that contact. Balls potted on a foul stay down; after a foul the other player has ball in hand anywhere
- `eight_ball`: once your group is cleared you are on the 8 and call it into a pocket: you win by pocketing it in the called pocket on a legal shot; you lose if you pocket it before your group is cleared, in a pocket you did not call (or on a safety), on a foul, or drive it off the table
- `shot_limit`: 60 shots each; a player who has used all 60 passes (ball in hand stays); when both have, the player with fewer balls left in its group wins (an open table counts as 7), then the player with more chess-clock time left
- `eightball_format`: one player (seat A) against a scripted computer opponent of level RCB_LEVEL 1-2 (`game spec` match.opponent_level): it calls and plays the easiest direct pot of its group with an aiming error (level 1: 0.5 degrees, level 2: 0.25 degrees and safer shot choice) and plays safe when it has none; at level 1 you break, at level 2 it breaks; its chess clock never runs, so a tie at the shot limit goes to it. Success = winning the rack
- `duel_format`: seat A (player 0) against seat B (player 1); a coin toss drawn from the seed decides who breaks (FIRST_PLAYER if it falls 0, else the other player; `game spec` match.breaker and match.you_break say who)
- `observation`: the state: your group and the opponent's (open, solids, stripes; `(on the 8)` once cleared), the balls left in each group, the shots taken, the balls on the table by number, who broke (the coin toss in the duel), the shots since your last one (who, the call and whether it was made, the first ball hit, the balls potted and their pockets, fouls, the groups assigned; no coordinates) and, on your turn, break_shot, ball_in_hand (none, above the head string, anywhere), on_the_8, call_required. One PNG of two camera tiles (observation.tiles): top (1120 x 640, fixed overhead 2.6 m above the table centre looking straight down; image right = +y, image down = +x) and sight (960 x 540, an eye 0.75 m behind the cue ball and 0.32 m above the cloth looking at the cloth 1.0 m ahead along the sighting direction; game observe '{"aim": DEG}' chooses it, default towards the table centre; the cue stick is drawn along it in both views; with the cue ball off the table it looks from the head spot). Both are ideal pinhole cameras with OpenCV axes; their poses and intrinsics are in the observation's `cameras`: p_cam = R_world_to_camera @ (p_world - position_world), u = fx x / z + cx, v = fy y / z + cy; a ball's image centre is the projection of its centre at z = R. Solids are coloured all over, the 8 black, stripes white with a coloured band; each object ball shows its number on a white disc; the cue ball is plain white
- Winner: the rack is won by potting the 8 legally in the called pocket after clearing your group and lost by an illegal 8 (before your group is cleared, in a pocket not called, on a foul, off the table); at the 60-shot limit the player with fewer balls left in its group wins (an open table counts as 7); if level, the player with more chess-clock time left (eightball: the scripted opponent's clock never runs, so a tie goes to it); if the clocks are equal too, (last resort, not from the rulebook) the player who did not break. There are no draws.
- Deviation D1 (rulebook 4.3(e), 4.3(f)): the 8 pocketed on the break is always spotted and the breaker continues if the break was legal (no re-break option)
- Deviation D2 (rulebook 4.3(d)): after an illegal break the other player always re-racks the same rack and breaks (option (2); the other options do not exist)
- Deviation D3 (rulebook 4.3(g), 4.3(h)): a foul on the break gives the other player ball in hand above the head string (no option to accept the table)
- Deviation D4 (rulebook 1.6): with ball in hand above the head string you cannot ask for an object ball above it to be spotted
- Deviation D5 (rulebook 1.7, 4.6): every shot after the break is called explicitly (a ball and a pocket, or safety): no obvious shots
- Deviation D8 (rulebook 4.11): a 60-shot limit per player replaces the stalemate rule: a player out of shots passes; when both have used all 60 the player with fewer balls left in its group wins (an open table counts as 7); if level, the player with more chess-clock time left (eightball: the scripted opponent's clock never runs, so a tie goes to it); if the clocks are equal too, (last resort, not from the rulebook) the player who did not break
- Deviation D9 (rulebook 1.2, 4.1): no lag for the break (the rulebook: the players lag and the winner of the lag chooses who breaks; neither the Rules of Play nor the Playing Regulations provide a coin toss): duel: a coin toss drawn from the seed decides who breaks (FIRST_PLAYER if it falls 0, else the other player; the toss decides, nobody chooses), and every pairing plays both first orders; eightball: the configuration decides (level 1 you, level 2 the scripted opponent)
- Deviation D10 (rulebook none (simulation)): a shot the physics engine cannot resolve (a rare internal failure of pooltool) is simulated again from the same position with the aim nudged by 0.001 degree steps (+, -, up to 0.01) until it resolves; the shot report says so (physics_retry)

## Geometry and physics
Everything you need is in `game spec`: the world frame, the release point, the cameras (pose, field of view, size,
`focal_px`, `principal_point_px`; pinhole: u = cx + f * x_right / depth, v = cy - f * y_up / depth) and the
execution scatter. The physics engine is pooltool 0.6.0 (event-based billiards physics with spin, cushions and pockets, float64, CPU); the sport's own PIL pinhole renderer draws the cameras.
