"""Native MIKASA teacher inference."""
from pathlib import Path
import torch


class MikasaPpoPolicy:
    """Run the native MIKASA state-only PPO teacher."""
    def __init__(self, env, checkpoint: Path):
        from baselines.ppo.ppo_memtasks import AgentStateOnly

        self.device = torch.device(env.unwrapped.device)
        self.agent = AgentStateOnly(env).to(self.device)
        self.agent.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.agent.eval()

    def forward(self, observation):
        with torch.no_grad():
            return self.agent.get_action(
                {key: value.to(self.device) for key, value in observation.items()},
                deterministic=True,
            )
