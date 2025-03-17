"""Dataset configuration for all datasets."""

from dataclasses import dataclass, field
from typing import Callable, List

from capo.configs.initial_prompts import INITIAL_PROMPTS
from capo.configs.task_descriptions import TASK_DESCRIPTIONS


@dataclass
class SplitConfig:
    train: str
    test: str


@dataclass
class DatasetConfig:
    name: str
    alias: str
    revision: str
    input: str | Callable
    target: str | Callable
    names: SplitConfig = field(default_factory=lambda: SplitConfig(train=None, test=None))
    splits: SplitConfig = field(default_factory=lambda: SplitConfig(train="train", test="test"))
    initial_prompts: List[str] = field(default_factory=list)
    task_description: str = None


_SST5_CONFIG = DatasetConfig(
    name="SetFit/sst5",
    alias="sst-5",
    revision="e51bdcd8cd3a30da231967c1a249ba59361279a3",
    input="text",
    target=lambda df: df["label_text"],
    initial_prompts=INITIAL_PROMPTS["sst-5"],
    task_description=TASK_DESCRIPTIONS["sst-5"],
)

_AGNEWS_CONFIG = DatasetConfig(
    name="SetFit/ag_news",
    alias="agnews",
    revision="ca5ba619eb034211db5f70932b6702efd21e7c73",
    input="text",
    target="label_text",
    initial_prompts=INITIAL_PROMPTS["agnews"],
    task_description=TASK_DESCRIPTIONS["agnews"],
)

_SUBJ_CONFIG = DatasetConfig(
    name="SetFit/subj",
    alias="subj",
    revision="f3c1162e678417f664d76b21864fdb87b0615fcf",
    input="text",
    target="label_text",
    initial_prompts=INITIAL_PROMPTS["subj"],
    task_description=TASK_DESCRIPTIONS["subj"],
)

_RTE_CONFIG = DatasetConfig(
    name="SetFit/rte",
    alias="rte",
    revision="23f2a468b9bc13030f5595a2e5f9307cb165280c",
    input=lambda df: "Text 1:\n" + df["text1"] + "\n Text 2:\n" + df["text2"],
    target=lambda df: df["label"].map({1: "No Entailment", 0: "Entailment"}),
    splits=SplitConfig(train="train", test="validation"),
    initial_prompts=INITIAL_PROMPTS["rte"],
    task_description=TASK_DESCRIPTIONS["rte"],
)

_GSM8K_CONFIG = DatasetConfig(
    name="openai/gsm8k",
    alias="gsm8k",
    revision="e53f048856ff4f594e959d75785d2c2d37b678ee",
    input="question",
    target=lambda df: df["answer"].str.extract(r"#### (.*)"),
    names=SplitConfig(train="main", test="main"),
    initial_prompts=INITIAL_PROMPTS["gsm8k"],
    task_description=TASK_DESCRIPTIONS["gsm8k"],
)

ALL_DATASETS = {
    "sst-5": _SST5_CONFIG,
    "agnews": _AGNEWS_CONFIG,
    "subj": _SUBJ_CONFIG,
    "rte": _RTE_CONFIG,
    "gsm8k": _GSM8K_CONFIG,
}
