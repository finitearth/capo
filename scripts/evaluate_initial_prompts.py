import os
from logging import getLogger

import pandas as pd
from promptolution.llms import get_llm
from promptolution.predictors import MarkerBasedClassificator

from capo.configs.experiment_configs import llama, mistral, qwen
from capo.load_datasets import get_tasks
from capo.utils import seed_everything

logger = getLogger(__name__)

if __name__ == "__main__":
    datasets = ["agnews", "gsm8k", "subj", "copa", "sst-5"]
    llms = [
        (llama.model, llama.revision),
        (qwen.model, qwen.revision),
        (mistral.model, mistral.revision),
    ]
    for llm, revision in llms:
        llm = get_llm(
            model_id=llm,
            max_model_len=2048,
            batch_size=None,
            model_storage_path="../models/",
            revision=revision,
            seed=42,
        )
        for dataset in datasets:
            path = f"init_results/{dataset}/{llm}"
            if os.path.exists(path):
                logger.critical(f"Skipping {dataset} with {llm} as it already exists")
                continue
            else:
                os.makedirs(path, exist_ok=True)
            seed_everything(42)

            _, _, test_task = get_tasks(
                dataset_name=dataset, optimizer_name="initial", seed=42, block_size=30
            )

            predictor = MarkerBasedClassificator(llm=llm, classes=test_task.classes)

            prompts = test_task.initial_prompts + [
                "Let's think step by step.",
                "Let's work this out in a step by step way to be sure we have the right answer.",
                "",
            ]

            logger.critical(f"Evaluating {len(prompts)} unique prompts on {dataset} with {llm}")
            scores = test_task.evaluate(prompts, predictor)

            df = pd.DataFrame({"prompt": prompts, "score": scores})

            df.to_csv(path + "eval.csv")
