# Fencing (epee): the rules check

Every article and value cited by the fencing tasks (README, prompt, code), checked against the two PDFs of the FIE rules, edition of August 2026:

- Technical Rules: https://static.fie.org/uploads/40/204126-Technical%20rules%20August%202026%20ang.pdf
- Material Rules: https://static.fie.org/uploads/40/204156-book%20material%20August%202026%20ang.pdf


| Article | The rulebook says | Implemented | |
|---|---|---|---|
| t.18 | combat area 1.50 m wide; piste 14 m, each fencer 2 m from the centre line with 5 m to retreat | 14 m piste drawn 1.5 m wide; footwork on a rail along its centre | match (width: deviation, rail) |
| t.19 | centre line; on-guard lines 2 m each side; rear limits 7 m from the centre; last 2 m before them in another colour | the same lines and shading | match |
| t.22.1 | the fencer called first stands on the referee's right | `FIRST_PLAYER` names the seats (the piste is symmetric) | match |
| t.22.5-9 | on guard at a distance where points in line cannot meet; after a valid hit in the middle of the piste; no hit: where they were; extra minute in the middle; nobody put behind the rear line who was in front of it | carts 4 m apart (points in line 1.2 m apart); on-guard lines after a hit and before the extra minute; otherwise where they were, each moved back equally; t.22.9 respected | match |
| t.22.11, t.23 | 'Play!' starts, 'Halt!' stops; only movements before 'Halt!' count | the world runs from `start`/`ready` until a halt | match |
| t.24 | fencing at close quarters allowed while the weapons can be wielded | arms and blades touching: play goes on | match |
| t.25.1, t.25.3 | corps a corps (the two fencers in contact) stops the bout; bringing it about by advancing is no fault at epee | halt without a hit when a trunk (cart, pedestal, shoulder) touches the other robot | deviation: arm-on-arm contact is not a corps a corps |
| t.31 / t.32 | after 'Halt!' ground gained is held, both retire an equal distance; after a corps a corps the fencer who sustained it keeps his place | equal retreat to 4 m in every no-hit halt | deviation for t.32 |
| t.33, t.35 | lateral boundaries | none (rail on the centre line) | deviation |
| t.34 | crossing the rear limit with both feet: a hit against | cart entirely behind the rear limit: a hit against | match |
| t.37 | duration = effective fencing time between 'Play!' and 'Halt!' | the clock runs only while the world runs | match |
| t.38.1 | pool bout ends at 5 hits; at epee at 4-4 a double hit is not counted and the fencers stay where they are | the same | match |
| t.38.2 | 3 minutes of fencing time; more hits wins; equal: one minute for a deciding hit, the winner if still equal drawn by lot before it | the same (lot seeded by the seed); a double in the extra minute scores for both and the minute goes on | match |
| t.44.1 | at the end of time the apparatus is cut off; hits registered before count | a hit registered before the end of time counts, its lockout window closes the phrase | match |
| t.90.1 | epee: thrusting weapon, hits only with the point | only the point registers | match |
| t.91 | target: the whole body, clothing and equipment | the whole robot (cart, pedestal, arm links) | match (the robot is the body) |
| t.92 | both hit and both registered: a double hit, one hit against each | the same | match |
| t.93 / t.94.1, Annexe B (d) | hits on the ground, between points or on the guard are not registered / annulled | point on the floor, rail, blade or guard does not register | match |
| t.124, t.170 (0.5) | one minute of fencing without a hit: 'Halt!'; direct elimination / teams: 1st time P-red to both, 2nd time P-black to the lower score (equal: lower seeding, lots if unranked); no P-card at 14-14; the minute restarts after each hit, annulled hit, penalty hit; not at equal scores at the end of regulation time | applied to the pool bout: 60 s of fencing time without a touch, 1st P-red to both, 2nd P-black (equal: seeded lot), no card at 4-4, restarts after touches / doubles / annulled doubles / rear-limit touches / P-red, not in the extra minute | deviation D9 (the FIE has no pool-bout clause) |
| m.16.3 | blade at most 90 cm | 0.90 m | match |
| m.17.1 | guard passes a 13.5 cm gauge, depth 3-5.5 cm | 13.5 cm diameter, 4 cm deep | match |
| m.19.2 | the point registers under a pressure of more than 750 g | more than 7.35 N (750 g x 9.81 m/s^2) on the point along the blade | match |
| Annexe B, B. Epee (b) | under 40 ms between hits: a double; over 50 ms: one hit; tolerance between | 45 ms lockout | match |
| Annexe B, B. Epee (c) | hits of 2-10 ms registered; nothing under 2 ms | at least 2 ms (3 consecutive 1 ms samples) | match |

Deviations that are not in the rulebook's articles (robot limits): the arm is a Franka Panda held to the
manufacturer's joint limits (fencing-3, user decision 2026-10-07; fencing-2 used 1.5 x its joint speed and acceleration
limits, accepted 2026-10-05, see `feasibility_results.json`); footwork on a rail, a Festo EGC-TB-185 toothed-belt axis
run inside its catalogue limits (fencing-4: 2 m/s, 6 m/s^2, jerk 150 m/s^3 (estimate), 2500 N; D8); no cards or penalties
(t.158-t.170) except the P-cards of t.124; one pool bout instead of a competition. The 2026 edition starts fencing with 'Play!' (t.22.11, t.23.1)
and puts the rear-line rule for placements at t.22.9 (the code cites these). In the extra minute (t.38.2.b) the 2026
text asks only for a deciding hit: a double there scores for both like any double (t.92) and the scores stay equal; only
at 4-4 is a double not counted (t.38.1.b).
