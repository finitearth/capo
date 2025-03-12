from typing import Tuple

import pandas as pd
from datasets import load_dataset
from promptolution.tasks import ClassificationTask

from capo.configs.base_config import OptimizerType
from capo.configs.dataset_config import ALL_DATASETS
from capo.task import CAPOClassificationTask


def get_tasks(
    dataset_name: str,
    optimizer_name: OptimizerType,
    block_size: int,
    dev_size: int = 300,
    fs_size: int = 200,
    test_size: int = 500,
    seed: int = 42,
) -> Tuple[ClassificationTask, pd.DataFrame, ClassificationTask]:
    """
    Load and process a dataset, returning three splits (dev, few-shot, test).

    Args:
        dataset_name: Name of the dataset (must be defined in ALL_DATASETS)
        dev_size: Size of the validation split
        fs_size: Size of the few-shot split
        test_size: Size of the test split
        seed: Random seed for reproducibility

    Returns:
        Tuple of (dev_task, fs_df, test_task)
    """
    if dataset_name not in ALL_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    config = ALL_DATASETS[dataset_name]

    train_df = load_dataset(
        config.name,
        name=config.names.train,
        split=config.splits.train,
        revision=config.revision,
    ).to_pandas()

    test_df = load_dataset(
        config.name,
        name=config.names.test,
        split=config.splits.test,
        revision=config.revision,
    ).to_pandas()

    if len(test_df) >= test_size:
        test_df = test_df.sample(test_size, random_state=seed, replace=False)
    else:
        raise ValueError("Not enough data in test split")

    # Sample and split training data
    if len(train_df) >= (dev_size + fs_size):
        train_sample = train_df.sample(dev_size + fs_size, random_state=seed, replace=False)
        dev_df = train_sample.iloc[:dev_size]
        fs_df = train_sample.iloc[dev_size:]
    else:
        raise ValueError("Not enough data in training split to create dev and few-shot splits")

    for df in [dev_df, fs_df, test_df]:
        if callable(config.input):
            df.loc[:, "input"] = config.input(df)
        else:
            df.loc[:, "input"] = df[config.input]

        if callable(config.target):
            df.loc[:, "target"] = config.target(df)
        else:
            df.loc[:, "target"] = df[config.target]

        print(df.head())

    # create a task from each dataset
    if optimizer_name == "CAPO":
        dev_task = CAPOClassificationTask(
            df=dev_df,
            description=config.task_description,
            initial_prompts=config.initial_prompts,
            x_column="input",
            y_column="target",
            block_size=block_size,
        )
        test_task = CAPOClassificationTask(
            df=test_df,
            description=config.task_description,
            initial_prompts=config.initial_prompts,
            x_column="input",
            y_column="target",
            block_size=block_size,
        )
    else:
        dev_task = ClassificationTask(
            df=dev_df,
            description=config.task_description,
            initial_prompts=config.initial_prompts,
            x_column="input",
            y_column="target",
        )
        test_task = ClassificationTask(
            df=test_df,
            description=config.task_description,
            initial_prompts=config.initial_prompts,
            x_column="input",
            y_column="target",
        )

    # exception
    if dataset_name == "gsm8k":
        dev_task.classes = None
        test_task.classes = None

    return dev_task, fs_df, test_task
