"""
Script for evaluating the performance of initial prompts on unseen test data.
"""

import os
from logging import getLogger

import pandas as pd
from promptolution.llms import get_llm
from promptolution.predictors import MarkerBasedClassificator

from capo.configs.experiment_configs import llama, mistral, qwen
from capo.configs.initial_prompts import UNINFORMATIVE_INIT_PROMPTS
from capo.load_datasets import get_tasks
from capo.utils import seed_everything

logger = getLogger(__name__)

if __name__ == "__main__":
    datasets = ["agnews", "gsm8k", "subj", "copa", "sst-5"]
    llms = [
        (mistral.model, mistral.revision, mistral.alias),
        (qwen.model, qwen.revision, qwen.alias),
        (llama.model, llama.revision, llama.alias),
    ]
    for llm_name, revision, alias in llms:
        llm = get_llm(
            model_id=llm_name,
            max_model_len=2048,
            batch_size=None,
            model_storage_path="../models/",
            revision=revision,
            seed=42,
        )
        for dataset in datasets:
            path = f"init_results/{dataset}/{alias}/"
            if os.path.exists(path):
                logger.critical(f"Skipping {dataset} with {alias} as it already exists")
                continue
            else:
                os.makedirs(path, exist_ok=True)

            seed_everything(42)

            _, _, test_task = get_tasks(
                dataset_name=dataset, optimizer_name="initial", seed=42, block_size=30
            )

            predictor = MarkerBasedClassificator(llm=llm, classes=test_task.classes)

            prompts = test_task.initial_prompts + UNINFORMATIVE_INIT_PROMPTS

            logger.critical(f"Evaluating {len(prompts)} unique prompts on {dataset} with {alias}")
            scores = test_task.evaluate(prompts, predictor)

            df = pd.DataFrame({"prompt": prompts, "test_score": scores, "llm": llm_name})

            df.to_csv(path + "eval.csv")
