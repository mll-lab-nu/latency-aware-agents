"""Isaac-GR00T modality for the EXTREME_PARKOUR Go2 latent policy."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


PARKOUR_GR00T_MODALITY_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["state"],
    ),
    "action": ModalityConfig(
        delta_indices=[0],
        modality_keys=["action"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    PARKOUR_GR00T_MODALITY_CONFIG,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)


__all__ = ["PARKOUR_GR00T_MODALITY_CONFIG"]
