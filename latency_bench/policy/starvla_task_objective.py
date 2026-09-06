"""H1 supervision through the benchmark's frozen downstream controllers."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
import runpy

# Load the shared pure-Torch loss without importing GR00T's full model registry.
ActionObjective = runpy.run_path(str(
    Path(__file__).resolve().parents[2] / 'third_party/Isaac-GR00T/gr00t/model/action_objectives.py'
))['ActionObjective']


class TaskActionObjective:
    """Apply the existing actor objective to a single predicted task command."""

    def __init__(self, config) -> None:
        self.name = config["name"]
        self.objective = ActionObjective(
            self.name,
            config["actor_checkpoint"],
            torch.tensor(config["output_min"]),
            torch.tensor(config["output_max"]),
        )

    def __call__(self, prediction: torch.Tensor, examples: list[dict]) -> torch.Tensor:
        def tensor(key: str) -> torch.Tensor:
            return torch.as_tensor(
                np.stack([example[key] for example in examples]),
                device=prediction.device,
                dtype=prediction.dtype,
            )

        if self.name == "parkour":
            # A 10 Hz H1 command is consumed with five fresh 50 Hz actor observations.
            prediction = prediction.expand(-1, 5, -1)
            inputs = {
                "actor_observation": tensor("actor_observation"),
                "action_target": tensor("action_target").expand(-1, 5, -1),
                "termination": tensor("termination"),
            }
        else:
            inputs = {
                "actor_input": tensor("actor_input").reshape(-1, 1, 605),
                "teacher_action": tensor("teacher_action").reshape(-1, 1, 23),
                "termination": tensor("termination").reshape(-1, 1),
            }
        return self.objective({"action_prediction": prediction}, inputs)["task_loss"]
