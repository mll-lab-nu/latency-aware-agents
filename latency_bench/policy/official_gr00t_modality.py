"""Isaac-GR00T NEW_EMBODIMENT modality registration.

Run this module with the Isaac-GR00T Python environment, for example through
``scripts/official_gr00t/finetune.sh``.  Keeping the import here (instead of
in the benchmark package initializer) prevents the official Python 3.12
dependencies from becoming dependencies of the benchmark's Python 3.10
environment.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


OFFICIAL_GR00T_MODALITY_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["state"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
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
    OFFICIAL_GR00T_MODALITY_CONFIG,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
