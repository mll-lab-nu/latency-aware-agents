"""RL-games data registry for StarVLA LeRobot datasets."""

from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class FlappyDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.image"]
    state_keys = ["state.game_state"]
    action_keys = ["action.button"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = [0]

    def modality_config(self):
        from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig

        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
        from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor

        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionToTensor(apply_to=self.action_keys),
        ])


class DemonAttackDataConfig(FlappyDataConfig):
    pass


class DefendTheLineDataConfig(FlappyDataConfig):
    pass


class DeadlyCorridorDataConfig(FlappyDataConfig):
    pass


class AsterixDataConfig(FlappyDataConfig):
    pass


class AtlantisDataConfig(FlappyDataConfig):
    pass


class GymnasiumDataConfig(FlappyDataConfig):
    """Discrete Gymnasium data: normalize state, keep action one-hot intact."""

    def transform(self):
        from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
        from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
            StateActionToTensor,
            StateActionTransform,
        )

        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=[*self.state_keys, *self.action_keys]),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={
                        "state.game_state": "min_max",
                    },
                ),
            ]
        )


class GymnasiumNativeDataConfig(FlappyDataConfig):
    """Configure native Gymnasium state and action vectors."""

    state_keys = ["state.native"]
    action_keys = ["action.native"]

    def transform(self):
        from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
        from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
            StateActionToTensor,
            StateActionTransform,
        )

        keys = [*self.state_keys, *self.action_keys]
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=keys),
                StateActionTransform(
                    apply_to=keys,
                    normalization_modes={
                        "state.native": "q99",
                        "action.native": "q99",
                    },
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "rl_games_flappy": FlappyDataConfig(),
    "rl_games_demon_attack": DemonAttackDataConfig(),
    "rl_games_defend_the_line": DefendTheLineDataConfig(),
    "rl_games_deadly_corridor": DeadlyCorridorDataConfig(),
    "rl_games_asterix": AsterixDataConfig(),
    "rl_games_atlantis": AtlantisDataConfig(),
    "rl_games_gymnasium": GymnasiumDataConfig(),
    "rl_games_gymnasium_discrete": GymnasiumDataConfig(),
    "rl_games_gymnasium_native": GymnasiumNativeDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "rl_games_flappy": EmbodimentTag.NEW_EMBODIMENT,
    "rl_games_demon_attack": EmbodimentTag.NEW_EMBODIMENT,
    "rl_games_defend_the_line": EmbodimentTag.NEW_EMBODIMENT,
    "rl_games_deadly_corridor": EmbodimentTag.NEW_EMBODIMENT,
    "rl_games_asterix": EmbodimentTag.NEW_EMBODIMENT,
    "rl_games_atlantis": EmbodimentTag.NEW_EMBODIMENT,
    "rl_games_gymnasium": EmbodimentTag.NEW_EMBODIMENT,
    "rl_games_gymnasium_discrete": EmbodimentTag.NEW_EMBODIMENT,
    "rl_games_gymnasium_native": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES = {
    "flappy_train": [("flappy_train", 1.0, "rl_games_flappy")],
    "flappy_train__bridge": [("flappy_train__bridge", 1.0, "rl_games_flappy")],
    "flappy_mixed_latency_train": [("flappy_mixed_latency_train", 1.0, "rl_games_flappy")],
    "flappy_mixed_latency_train__bridge": [
        ("flappy_mixed_latency_train__bridge", 1.0, "rl_games_flappy")
    ],
    "demon_attack_train": [("demon_attack_train", 1.0, "rl_games_demon_attack")],
    "demon_attack_train__bridge": [("demon_attack_train__bridge", 1.0, "rl_games_demon_attack")],
    "demon_attack_mixed_latency_train": [("demon_attack_mixed_latency_train", 1.0, "rl_games_demon_attack")],
    "demon_attack_mixed_latency_train__bridge": [
        ("demon_attack_mixed_latency_train__bridge", 1.0, "rl_games_demon_attack")
    ],
    "defend_the_line_train": [("defend_the_line_train", 1.0, "rl_games_defend_the_line")],
    "defend_the_line_train__bridge": [
        ("defend_the_line_train__bridge", 1.0, "rl_games_defend_the_line")
    ],
    "defend_the_line_mixed_latency_train": [
        ("defend_the_line_mixed_latency_train", 1.0, "rl_games_defend_the_line")
    ],
    "defend_the_line_mixed_latency_train__bridge": [
        ("defend_the_line_mixed_latency_train__bridge", 1.0, "rl_games_defend_the_line")
    ],
    "deadly_corridor_train": [("deadly_corridor_train", 1.0, "rl_games_deadly_corridor")],
    "deadly_corridor_train__bridge": [("deadly_corridor_train__bridge", 1.0, "rl_games_deadly_corridor")],
    "deadly_corridor_mixed_latency_train": [
        ("deadly_corridor_mixed_latency_train", 1.0, "rl_games_deadly_corridor")
    ],
    "deadly_corridor_mixed_latency_train__bridge": [
        ("deadly_corridor_mixed_latency_train__bridge", 1.0, "rl_games_deadly_corridor")
    ],
    "asterix_train": [("asterix_train", 1.0, "rl_games_asterix")],
    "asterix_train__bridge": [("asterix_train__bridge", 1.0, "rl_games_asterix")],
    "asterix_mixed_latency_train": [("asterix_mixed_latency_train", 1.0, "rl_games_asterix")],
    "asterix_mixed_latency_train__bridge": [
        ("asterix_mixed_latency_train__bridge", 1.0, "rl_games_asterix")
    ],
    "atlantis_train": [("atlantis_train", 1.0, "rl_games_atlantis")],
    "atlantis_train__bridge": [("atlantis_train__bridge", 1.0, "rl_games_atlantis")],
    "atlantis_mixed_latency_train": [("atlantis_mixed_latency_train", 1.0, "rl_games_atlantis")],
    "atlantis_mixed_latency_train__bridge": [
        ("atlantis_mixed_latency_train__bridge", 1.0, "rl_games_atlantis")
    ],
    "inverted_pendulum_mixed_latency_train": [
        ("inverted_pendulum_mixed_latency_train", 1.0, "rl_games_gymnasium_discrete")
    ],
    "inverted_pendulum_mixed_latency_train__bridge": [
        ("inverted_pendulum_mixed_latency_train__bridge", 1.0, "rl_games_gymnasium_discrete")
    ],
    "inverted_pendulum_mixed_latency_train_no_latency_prompt": [
        ("inverted_pendulum_mixed_latency_train_no_latency_prompt", 1.0, "rl_games_gymnasium_discrete")
    ],
    "inverted_pendulum_mixed_latency_train_no_latency_prompt__bridge": [
        ("inverted_pendulum_mixed_latency_train_no_latency_prompt__bridge", 1.0, "rl_games_gymnasium_discrete")
    ],
    "h1hand_balance_hard": [("h1hand_balance_hard", 1.0, "rl_games_gymnasium_native")],
}
