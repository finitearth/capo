from dataclasses import dataclass
from enum import Enum

from ..common.base_classes import UniversalBaseClass


# Set of Prompt Management Techniques supported by Vellm co-pilot
# Hyperparameters defined in promptopt_config.yaml
class SupportedPromptOpt(Enum):
    CRITIQUE_N_REFINE = "critique_n_refine"

    @classmethod
    def all_values(cls):
        return ",".join([member.value for member in SupportedPromptOpt])

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


@dataclass
class PromptOptimizationLiterals:
    PROMPT_TECHNIQUE_NAME = "prompt_technique_name"


@dataclass
class PromptOptimizationParams(UniversalBaseClass):
    """
    Parent class for all Prompt Optimization classes.
    """
    prompt_technique_name: str


@dataclass
class PromptPool(UniversalBaseClass):
    """
    Parent class for all classes that handle prompt strings for each techniques.
    """
    system_prompt: str
    final_prompt: str
    eval_prompt: str
