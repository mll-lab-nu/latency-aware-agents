# seasons/

A season is a set of matches frozen at one version (docs/framework_zh.md §6, §9 tool rule 4): `seasons/<name>.toml`
names the tasks, the players (ids of `players/players.toml` with their `config_version`), the seeds and first orders,
and the versions every result must carry (protocol, core, each sport's revision, player config). Opening a season
changes only this directory; results are ranked only with results of the same versions.

Status (v0.2 stage 0): the format below is a proposal; Season 1 ran on the legacy runner
(`scripts/olympiad/run_round.py`, `competition.py`) and is not described here. A version-freeze check comes with the
first v0.2 season.

Before a season opens, work through the pre-season checklist `seasons/PREP_zh.md` (seed plan, formats, time limits,
version freeze).

```toml
[season]
name = "example"
core = "0.2.0"                      # core/__init__.py VERSION
protocols = ["eai-standard-turn/1.0", "eai-standard-realtime/1.0"]

[[sport]]
name = "hello"                      # sports/<name>/sport.toml
revision = "hello-1"
tasks = ["hello-duel-program-standard"]
seeds = [1, 2, 3]
first = ["A", "B"]                  # both first orders
renderer = "mujoco"                 # the players' image renderer (results are compared per renderer)

[[player]]
id = "opus"                         # players/players.toml
config_version = "1"
```
