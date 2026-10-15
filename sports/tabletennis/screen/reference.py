"""The [reference] match hook of tools/check stage 5 (sport.toml [reference]): the step-4 screening loop
(tools/screen/play.py) with this directory's hooks: camera = the reference unchanged, truth = the same player with the
ball's true position (augment_view)."""
from reference_hook import hook

match = hook("hooks.py", camera="reference", truth="reference_truth")
