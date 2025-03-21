from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal

DatasetType = Literal["sst-5", "agnews", "subj", "gsm8k", "copa"]
OptimizerType = Literal["EvoPromptGA", "CAPO"]
ModelType = Literal[
    "shuyuej/Llama-3.3-70B-Instruct-GPTQ",
    "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
    "ConfidentialMind/Mistral-Small-24B-Instruct-2501_GPTQ_G128_W4A16_MSE",
]


@dataclass
class OptimizerConfig:
    name: str
    optimizer: OptimizerType
    optimizer_params: Dict


@dataclass
class ModelConfig:
    model: ModelType
    alias: str
    max_model_len: int
    batch_size: int
    model_storage_path: Path
    revision: str


@dataclass
class ExperimentConfig:
    name: str
    datasets: List[DatasetType]
    models: List[ModelConfig]
    optimizers: List[OptimizerConfig]
    random_seeds: List[int]
    budget_per_run: int
    output_dir: Path
