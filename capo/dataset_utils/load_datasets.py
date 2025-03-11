from typing import List, Tuple

import pandas as pd
from datasets import load_dataset
from promptolution.tasks import ClassificationTask

from capo.dataset_utils.tasks import INITIAL_PROMPTS, TASK_DESCRIPTIONS
from capo.task import CAPOClassificationTask

DATASET_INFO = {
    "sst-5": {
        "name": "SetFit/sst5",
        "revision": "e51bdcd8cd3a30da231967c1a249ba59361279a3",
        "splits": {"train": "train", "test": "test"},
        "columns": {
            "input": "text",
            "target": lambda df: df["label_text"].map(
                {
                    "very negative": "veryNegative",
                    "negative": "negative",
                    "neutral": "neutral",
                    "positive": "positive",
                    "very positive": "veryPositive",
                }
            ),
        },
    },
    "agnews": {
        "name": "SetFit/ag_news",
        "revision": "ca5ba619eb034211db5f70932b6702efd21e7c73",
        "splits": {"train": "train", "test": "test"},
        "columns": {"input": "text", "target": "label_text"},
    },
    "subj": {
        "name": "SetFit/subj",
        "revision": "f3c1162e678417f664d76b21864fdb87b0615fcf",
        "splits": {"train": "train", "test": "test"},
        "columns": {"input": "text", "target": "label_text"},
    },
    "rte": {
        "name": "SetFit/rte",
        "revision": "23f2a468b9bc13030f5595a2e5f9307cb165280cmain",
        "splits": {"train": "train", "test": "test"},
        "columns": {
            "input": lambda df: df["text1"] + "\n" + df["text2"],
            "target": lambda df: df["label"].map({1: "NoEntailment", 0: "Entailment"}),
        },
    },
    "gsm8k": {
        "name": "openai/gsm8k",
        "revision": "e53f048856ff4f594e959d75785d2c2d37b678ee",
        "splits": {"train": "train", "test": "test"},
        "columns": {
            "input": "question",
            "target": lambda df: df["answer"].str.extract(r"#### (.*)"),
        },
    },
}


def get_tasks(
    dataset_name: str,
    optimizer_name: str,
    fs_size: int = 200,
    val_size: int = 300,
    test_size: int = 500,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and process a dataset, returning three splits (validation, few-shot, test).

    Args:
        dataset_name: Name of the dataset (must be defined in DATASET_CONFIG)
        fs_size: Size of the few-shot split
        val_size: Size of the validation split
        test_size: Size of the test split
        seed: Random seed for reproducibility

    Returns:
        Tuple of (val_df, fs_df, test_df)
    """
    if dataset_name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    config = DATASET_INFO[dataset_name]

    train_df = load_dataset(
        config["name"], split=config["splits"]["train"], revision=config["revision"]
    ).to_pandas()

    test_df = load_dataset(
        config["name"], split=config["splits"]["test"], revision=config["revision"]
    ).to_pandas()

    if len(test_df) > test_size:
        test_df = test_df.sample(test_size, random_state=seed)
    else:
        raise ValueError("Not enough data in test split")

    # Sample and split training data
    if len(train_df) > (val_size + fs_size):
        train_sample = train_df.sample(val_size + fs_size, random_state=seed)
        val_df = train_sample.iloc[:val_size]
        fs_df = train_sample.iloc[val_size:]
    else:
        raise ValueError(
            "Not enough data in training split to create validation and few-shot splits"
        )

    for df in [val_df, fs_df, test_df]:
        for target_col, source in config["columns"].items():
            if callable(source):
                df[target_col] = source(df)
            else:
                df[target_col] = df[source]

    # create a task from each dataset
    if optimizer_name == "capo":
        val_task = CAPOClassificationTask.from_dataframe(
            val_df,
            description=TASK_DESCRIPTIONS[dataset_name],
            x_column="input",
            y_column="target",
        )
        fs_task = CAPOClassificationTask.from_dataframe(
            fs_df,
            description=TASK_DESCRIPTIONS[dataset_name],
            x_column="input",
            y_column="target",
        )
        test_task = CAPOClassificationTask.from_dataframe(
            test_df,
            description=TASK_DESCRIPTIONS[dataset_name],
            x_column="input",
            y_column="target",
        )
    else:
        val_task = ClassificationTask.from_dataframe(
            val_df,
            description=TASK_DESCRIPTIONS[dataset_name],
            x_column="input",
            y_column="target",
        )
        fs_task = ClassificationTask.from_dataframe(
            fs_df,
            description=TASK_DESCRIPTIONS[dataset_name],
            x_column="input",
            y_column="target",
        )
        test_task = ClassificationTask.from_dataframe(
            test_df,
            description=TASK_DESCRIPTIONS[dataset_name],
            x_column="input",
            y_column="target",
        )

    return val_task, fs_task, test_task


def get_initial_prompts(dataset_name: str) -> List:
    return INITIAL_PROMPTS[dataset_name]
