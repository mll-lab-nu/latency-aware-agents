# Curling: program play (duel)

You play curling against an opponent model in the same world (seat `A` or `B`). Two teams slide granite stones down a sheet of ice towards the house, 4 ends of 5 stones per team; you skip one team and throw every one of its stones (release speed, direction and spin); in the `match` format the other team is a scripted computer team, in the `duel` format another player.
This task is generated from `sports/curling/sport.toml` (task `curling-duel-program-standard`).

## Time
Nothing moves while you think: the world waits for the player whose turn it is. Each player has a chess clock of 60 minutes of real time, running only on its own turn (thinking, tool calls, your programs and your reviews all count). **If your clock runs out, you lose.**

After each of your actions (except one that ends the match), send `game ready "WHY" "PLAN"` (a short review) before the game goes on.
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
| `game wait` | `POST /wait {"timeout_s": 60}` | block until it is your turn (or the match is over) |
| `game observe` | `POST /observe {}` | the state and the camera images (one PNG of tiles: `observation.tiles`) |
| `game act JSON` | `POST /act JSON` | one action |
| `game ready "WHY" "PLAN"` | `POST /ready {"why":"...","plan":"..."}` | your review after each of your actions |

Errors: 400 = an invalid action (nothing happens), 409 = not now (not your turn, or a review is due), 429 = one
action in flight, or more than 20 requests per second or 4 open requests from your seat (empty body: retry more slowly).

## Rules
- two teams slide granite stones down a sheet of ice towards the house, 4 ends of 5 stones per team; you skip one team and throw every one of its stones (release speed, direction and spin); in the `match` format the other team is a scripted computer team, in the `duel` format another player.
- `world`: metres; the release point (the hack) is the origin, +y runs down the sheet towards the far house, +x is to the right as seen from the hack, z up
- `sheet`: far hog line y = 32.004, tee line y = 38.405 (the tee, the centre of the house, is at (0, 38.405)), back line y = 40.234; the house has radius 1.829 with rings at 1.219 and 0.610 and the button at 0.152; the sheet spans |x| <= 2.375; stones have radius 0.145 and mass 19.96 kg (top of a stone at z = 0.112, handle top at 0.146)
- `throw`: game act '{"type": "throw", "speed": 2.39, "angle": 3.1, "spin": "ccw"}' (HTTP POST /act with that body): the stone's state at release, not a target. speed: release speed at the hack, 0.5-4.0 m/s; angle: release direction in degrees from the centre line, + = to the right (+x), |angle| <= 5; spin: ccw (counter-clockwise seen from above) curls the stone to the left (-x), cw to the right (+x)
- `execution_noise_1sigma`: {"speed_m_s": 0.004, "angle_deg": 0.03, "note": "about +-14 cm in length for a draw and +-2 cm sideways at the house; each player's execution noise is independent of the opponent's within a match (its own seeded stream, its size the thrower's own); the two matches of a duel pair (the same seed, the first player swapped) swap the noise sequences, so over the pair both players meet the same draws"}
- `ice`: the speed v decreases at a(v) = (0.00200985 / (v + 0.06385782) + 0.00626286) x 9.80665 m/s^2; a rotating stone curls: the direction of its velocity turns at 0.00820 x v^-0.8 rad/s, towards -x for ccw and +x for cw; every stone is released with |omega| = pi/2 rad/s, the spin decays at 0.025 / v rad/s^2 and this never reverses the curl. The curl is strongest when the stone is slow; a draw to the tee (no other stones) needs about 2.39 m/s and curls about 2.1 m sideways: aim about 3.1 degrees to the right with ccw, or to the left with cw. Takeouts use about 2.7-3.5 m/s and curl much less
- `collisions`: elastic (restitution 1, equal masses) with friction 0.2 at the contact; a full hit on a stationary stone roughly stops the thrower and sends the struck stone on at the thrower's speed; glancing hits split the motion
- `in_play`: a delivered stone that does not come to rest completely beyond the far hog line (stone centre y < 32.149) is removed from play unless it struck another stone; a stone that touches the side line (|x| + 0.145 >= 2.375) or completely crosses the back line (y - 0.145 > 40.234) is removed
- `free_guard_zone`: the area between the far hog line and the tee line, excluding the house; before the 6th stone of an end, if a delivered stone moves an opposition stone from the zone out of play (directly or indirectly), the delivered stone is removed and every moved stone is put back
- `scoring`: after all 10 stones of an end, the team with the stone closest to the tee (among stones in the house: centre distance < 1.829 + 0.145) scores one point for each of its stones in the house closer than the other team's closest stone; no stone in the house: a blank end
- `hammer`: the team without the hammer (the last stone of the end) throws first and the teams alternate; after an end the team that scored throws first in the next end; a blank end keeps the hammer
- `match_format`: one player (seat A, yellow) against a scripted computer team (red) of level RCB_LEVEL 1-2: it draws to the button, takes out your stones and, at level 2, also plays off-centre and firm takeouts and guards; its execution error is its own size; it has the hammer in end 1 and its chess clock never runs. Success = winning the match
- `duel_format`: seat A (yellow) against seat B (red); a coin toss drawn from the seed decides who throws the first stone of end 1 (FIRST_PLAYER when it falls 0, else the other player; `game spec` match.throws_first_in_end_1 says who), and the other player has the hammer in end 1
- `observation`: the score, the end, the throw in the end, the hammer, the stones left to throw and in play, the ends so far and the throws since your last one (what was heard: collisions, stones removed and why; no stone coordinates), and one PNG of two camera tiles: house (straight above the far house, image up = +y, image right = +x) and hack (behind the hack looking down the sheet). Stones are grey granite with a coloured top and handle
- Winner: the higher total after 4 ends wins; if level, extra ends are played until a team scores, at most 3; if still level, the player with more chess-clock time left wins (match format: the scripted team's clock never runs, so a tie goes to it); if the clocks are equal too, (last resort, not from the rulebook) the player who had the hammer in end 1. There are no draws.
- Deviation D1 (rulebook C5, R2(b)): 4 ends of 5 stones per team (official: 8 or 10 ends of 8 stones)
- Deviation D2 (rulebook R12(a)): at most 3 extra ends while tied; then the player with more chess-clock time left wins (official: extra ends until a team scores)
- Deviation D3 (rulebook R8): no sweeping
- Deviation D5 (rulebook R7): no no-tick rule
- Deviation D6 (rulebook R5(a)): match format: the scripted team has the hammer in end 1 (a fixed handicap; official: coin toss or last stone draw); duel: a coin toss drawn from the seed decides who throws the first stone of end 1 (FIRST_PLAYER if it falls 0, else the other player; it falls 0 for the official seeds 1-3); the other player has the hammer in end 1

## Geometry and physics
Everything you need is in `game spec`: the world frame, the release point, the cameras (pose, field of view, size,
`focal_px`, `principal_point_px`; pinhole: u = cx + f * x_right / depth, v = cy - f * y_up / depth) and the
execution scatter. The physics engine is curling ice model (DigitalCurling FCV1, pure Python, 1 ms steps) with impulse collisions; MuJoCo renders the cameras.
