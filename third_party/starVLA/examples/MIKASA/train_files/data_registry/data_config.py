from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class MikasaFrankaH1DataConfig:
    """Define the modalities and transforms for MIKASA Franka H1 episodes."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.top", "video.wrist"]
    state_keys = ["state.proprio"]
    action_keys = ["action.eef"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        keys = [*self.state_keys, *self.action_keys]
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=keys),
                StateActionTransform(
                    apply_to=keys,
                    normalization_modes={key: "min_max" for key in keys},
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {"mikasa_franka_h1": MikasaFrankaH1DataConfig()}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {"mikasa_franka_h1": EmbodimentTag.NEW_EMBODIMENT}
DATASET_NAMED_MIXTURES = {
    "local/intercept_grab_fast_vla_v0_h1_train": [
        ("local/intercept_grab_fast_vla_v0_h1_train", 1.0, "mikasa_franka_h1")
    ]
}
