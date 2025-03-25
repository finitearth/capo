import json
import os
from glob import glob
from logging import getLogger

import pandas as pd
from promptolution.llms import get_llm
from promptolution.predictors import MarkerBasedClassificator
from capo.utils import seed_everything


from capo.load_datasets import get_tasks

logger = getLogger(__name__)

if __name__ == "__main__":
    datasets = ["agnews", "gsm8k", "subj", "copa", "sst-5"]
    llms = ["qwen", "llama", "mistral"]
    for revision, llm in llms:       
        llm = get_llm(
            model_id=llm,
            max_model_len=2048,
            batch_size=None,
            model_storage_path="../models/",
            revision=revision,
            seed=42,
        )
        for dataset in datasets:
            seed_everything(42)

            predictor = MarkerBasedClassificator(llm=llm, classes=test_task.classes)
            
            _, _, test_task = get_tasks(
                dataset_name=dataset,
                optimizer_name="initial",
                seed=42
            )

            prompts = test_task.initial_prompts + [
                "Let's think step by step.",
                "Let's work this out in a step by step way to be sure we have the right answer.",
                ""
            ]

            logger.critical(f"Evaluating {len(prompts)} unique prompts on {dataset} with {llm}")
            scores = test_task.evaluate(prompts, predictor)

            df = pd.DataFrame({
                "prompt": prompts,
                "score": scores
            })

            df.to_csv(f"init_results/{dataset}/{llm}/")
