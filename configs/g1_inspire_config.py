"""Stored targets remain absolute; the processor makes arm targets relative."""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig,
)

KEYS = ["left_arm", "right_arm", "left_hand", "right_hand"]
g1_inspire_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=KEYS),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=KEYS,
        action_configs=[
            ActionConfig(
                rep=representation,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for representation in [
                ActionRepresentation.RELATIVE,
                ActionRepresentation.RELATIVE,
                ActionRepresentation.ABSOLUTE,
                ActionRepresentation.ABSOLUTE,
            ]
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.task_description"]
    ),
}
register_modality_config(
    g1_inspire_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
