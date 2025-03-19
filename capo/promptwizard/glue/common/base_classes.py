from dataclasses import dataclass
from enum import Enum
from inspect import getmembers, ismethod
from typing import List, Optional

# This file has class definitions for config yaml files

# TODO: add comments  for class definition and variable definition


class UniversalBaseClass:
    def __str__(self) -> str:
        attributes_string = []
        for member in getmembers(self):

            # remove private and protected attributes
            if not member[0].startswith('_'):

                # remove methods that does not start with underscore
                if not ismethod(member[1]):
                    attributes_string.append(member)
        return str(attributes_string)

######################################################################################
# Classes related to llm_config.yaml


@dataclass
class LLMModel(UniversalBaseClass):
    unique_model_id: str
    model_type: str
    track_tokens: str
    req_per_min: int
    tokens_per_min: int
    error_backoff_in_seconds: int

@dataclass
class UserLimits(UniversalBaseClass):
    max_num_requests_in_time_window: int
    time_window_length_in_seconds: int


@dataclass
class LLMQueueSchedulerLimits(UniversalBaseClass):
    ttl_in_seconds: int
    max_queue_size: int


@dataclass
class AzureAOIModels(LLMModel, UniversalBaseClass):
    model_name_in_azure: str
    deployment_name_in_azure: str


@dataclass
class AzureAOILM(UniversalBaseClass):
    api_key: str
    api_version: str
    api_type: str
    azure_endpoint: str
    azure_oai_models: List[AzureAOIModels]

    def __post_init__(self):
        azure_oai_models_obj = []
        if self.azure_oai_models:
            for azure_oai_model in self.azure_oai_models:
                azure_oai_models_obj.append(AzureAOIModels(**azure_oai_model))
        self.azure_oai_models = azure_oai_models_obj


@dataclass
class CustomLLM(LLMModel):
    path_to_py_file: str
    class_name: str


@dataclass
class LLMConfig(UniversalBaseClass):
    azure_open_ai: AzureAOILM
    user_limits: UserLimits
    scheduler_limits: LLMQueueSchedulerLimits
    custom_models: List[CustomLLM]

    def __post_init__(self):
        self.azure_open_ai = AzureAOILM(**self.azure_open_ai)
        custom_model_obj = []
        if self.custom_models:
            for custom_model in self.custom_models:
                custom_model_obj.append(CustomLLM(**custom_model))
        self.custom_models = custom_model_obj

######################################################################################
# Classes related to setup_config.yaml


@dataclass
class AssistantLLM(UniversalBaseClass):
    prompt_opt: str


@dataclass
class Dir(UniversalBaseClass):
    base_dir: str
    log_dir_name: str


class OperationMode(Enum):
    ONLINE = "online"
    OFFLINE = "offline"


@dataclass
class SetupConfig(UniversalBaseClass):
    assistant_llm: AssistantLLM
    dir_info: Dir
    experiment_name: str
    mode: OperationMode
    description: str

    def __post_init__(self):
        if self.dir_info:
            self.dir_info = Dir(**self.dir_info)
        if self.assistant_llm:
            self.assistant_llm = AssistantLLM(**self.assistant_llm)

######################################################################################
# Classes related to prompt_library_config.yaml

@dataclass
class TaskConfig:
    name: str
    prompt_template: str
    llm_request_type: str
    prepend_system_prompts: Optional[bool] = True
    prepend_system_guidelines: Optional[bool] = True
    emb_model_id: Optional[str] = None
    llm_model_id: Optional[str] = None

@dataclass
class Mode:
    chat: List[TaskConfig]
    generation: List[TaskConfig]

    def __post_init__(self):
        chat_obj = []
        if self.chat:
            for chat_config in self.chat:
                chat_obj.append(TaskConfig(**chat_config))
        self.chat = chat_obj

        gen_obj = []
        if self.generation:
            for gen_config in self.generation:
                gen_obj.append(TaskConfig(**gen_config))
        self.generation = gen_obj


@dataclass
class PromptLibraryConfig:
    mode: Mode
    system_prompts: Optional[str] = None
    system_guidelines: Optional[str] = None

    def __post_init__(self):
        if self.mode:
            self.mode = Mode(**self.mode)
