from logging import getLogger

import pandas as pd
from promptolution.callbacks import LoggerCallback
from promptolution.llms import get_llm
from promptolution.predictors.classificator import Classificator
from promptolution.utils.prompt_creation import create_prompts_from_samples

from capo.capo import CAPOptimizer
from capo.statistical_tests import paired_t_test
from capo.task import CAPOClassificationTask

BLOCK_SIZE = 20
FS_SPLIT = 0.2
BATCH_SIZE = 512

try:
    token = open("deepinfratoken.txt", "r").read()
except FileNotFoundError:
    token = None
model_name = "vllm-shuyuej/Llama-3.3-70B-Instruct-GPTQ"

if "vllm" in model_name:
    llm = get_llm(
        model_name,
        batch_size=BATCH_SIZE,
        model_storage_path="../models/",
    )
else:
    llm = get_llm(model_name, token)

downstream_llm = llm
meta_llm = llm

df = pd.read_json("hf://datasets/SetFit/sst5/train.jsonl", lines=True).sample(500)

task = CAPOClassificationTask.from_dataframe(
    df,
    description="The dataset consists of movie reviews with five levels of sentiment labels: terrible, bad, neutral, okay, good, and great. The task is to classify each movie review into one of these five sentiment categories. The class mentioned first in the response of the LLM will be the prediction.",
    x_column="text",
    y_column="label",
)

task = CAPOClassificationTask.from_task(task, block_size=BLOCK_SIZE, few_shot_split_size=FS_SPLIT)
initial_prompts = [create_prompts_from_samples(task, downstream_llm) for _ in range(5)]

predictor = Classificator(downstream_llm, task.classes)
test_statistic = lambda x, y: paired_t_test(x, y, alpha=0.2)

logger = getLogger(__name__)
callback = LoggerCallback(logger)

optimizer = CAPOptimizer(
    initial_prompts=initial_prompts,
    task=task,
    meta_llm=meta_llm,
    downstream_llm=downstream_llm,
    length_penalty=5e-4,
    block_size=BLOCK_SIZE,
    crossovers_per_iter=4,
    upper_shots=3,
    max_n_blocks_eval=10,
    p_few_shot_reasoning=0.5,
    few_shot_split_size=FS_SPLIT,
    test_statistic=test_statistic,
    predictor=predictor,
    callbacks=[callback],
    shuffle_blocks_per_iter=False,
)
best_prompts = optimizer.optimize(n_steps=5)
print(f"Best instructions:\n\n {best_prompts}")
