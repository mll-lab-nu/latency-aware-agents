"""Isaac-GR00T modality for HAIC issued command chunks."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from latency_bench.data.haic_dagger import HAIC_ACTION_HORIZON
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


HAIC_ISSUED_GR00T_MODALITY_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["state"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(HAIC_ACTION_HORIZON)),
        modality_keys=["issued_command_chunk"],
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
    HAIC_ISSUED_GR00T_MODALITY_CONFIG,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)


__all__ = ["HAIC_ISSUED_GR00T_MODALITY_CONFIG"]
