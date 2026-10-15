"""Register frozen RoboOlympics matches and their visual reference players."""

from latency_bench.core.types import Action, PolicyOutput
from latency_bench.envs.factory import register_env_adapter
from latency_bench.policy.base import PolicyRunner


@register_env_adapter("roboolympics_tabletennis", integration_name="roboolympics")
@register_env_adapter("roboolympics_badminton", integration_name="roboolympics")
@register_env_adapter("roboolympics_soccer", integration_name="roboolympics")
@register_env_adapter("roboolympics_soccer_ssl", integration_name="roboolympics")
def make_env(config, noop_action):
    """Construct the native match for single-environment simulated execution."""
    from latency_bench.envs.adapters.roboolympics import RoboOlympicsAdapter

    executor = config["executor"]
    if executor["mode"] != "simulated" or executor["parallel_envs"] != 1 or executor["inference_pool"]:
        raise ValueError("RoboOlympics requires single-environment simulated execution without an inference pool")
    if config["scheduler"]["hold_policy"] != "one_frame_then_noop":
        raise ValueError("RoboOlympics commands require scheduler.hold_policy=one_frame_then_noop")
    env = config["env"]
    adapter = RoboOlympicsAdapter(
        sport=env["name"].removeprefix("roboolympics_"),
        level=env["level"], seed=config["experiment"]["seed"],
        max_env_steps=config["evaluation"]["max_env_steps"], noop_action=noop_action,
    )
    if env["env_fps"] != adapter.env_fps:
        raise ValueError(f"RoboOlympics {adapter.sport.name} requires env.env_fps={adapter.env_fps}")
    return adapter


class RoboOlympicsReferencePolicy(PolicyRunner):
    """Run the upstream camera-only player and retain its state between points."""

    def __init__(self, sport):
        # The registration module must load without the optional simulator.
        from latency_bench.envs.adapters.roboolympics import load_sport

        self._reference = load_sport(sport).players()["reference"]
        self._player = None
        self._unit = None

    def predict(self, observation):
        if self._player is None or observation.env_step == 0:
            self._player = self._reference(
                observation.metadata["spec"], "A", observation.metadata["episode_seed"],
            )
        elif observation.metadata["unit"] != self._unit:
            self._player.review(observation.data["state"])
        self._unit = observation.metadata["unit"]
        command = self._player.act(observation.data)
        return PolicyOutput(
            action=Action(value=command, name="wait" if command is None else command["type"], is_noop=command is None),
            raw_output=command,
        )


def make_reference_policy(config, action_resolver):
    """Construct the reference player through the public custom-policy factory."""
    return RoboOlympicsReferencePolicy(config["env"]["name"].removeprefix("roboolympics_"))
