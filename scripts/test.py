from logging import getLogger

import pandas as pd
from promptolution.callbacks import LoggerCallback
from promptolution.llms.api_llm import APILLM
from promptolution.predictors.classificator import Classificator
from promptolution.tasks import ClassificationTask
from promptolution.utils.prompt_creation import create_prompts_from_samples

from capo.capo import CAPOptimizer
from capo.statistical_tests import paired_t_test
from capo.task import CAPOTask

BLOCK_SIZE = 20
FS_SPLIT = 0.2

token = open("deepinfratoken.txt", "r").read()

meta_llm = APILLM("meta-llama/Meta-Llama-3-8B-Instruct", token)
downstream_llm = meta_llm

df = pd.read_parquet(
    "hf://datasets/stanfordnlp/imdb/plain_text/train-00000-of-00001.parquet"
).sample(1_000)
task = ClassificationTask.from_dataframe(
    df,
    description="Is the Movie review positive =1 or negative =0",
    x_column="text",
    y_column="label",
)

task = CAPOTask.from_task(task, block_size=BLOCK_SIZE, few_shot_split_size=FS_SPLIT)
initial_prompts = [create_prompts_from_samples(task, downstream_llm) for _ in range(10)]

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
    crossovers_per_iter=10,
    upper_shots=3,
    max_n_blocks_eval=10,
    few_shot_split_size=FS_SPLIT,
    test_statistic=test_statistic,
    predictor=predictor,
    callbacks=[callback],
    shuffle_blocks_per_iter=False,
)
best_prompts = optimizer.optimize(n_steps=20)
print(f"Best instructions:\n\n {best_prompts}")
