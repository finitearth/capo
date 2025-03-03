from logging import getLogger

import pandas as pd
from promptolution.callbacks import LoggerCallback
from promptolution.llms.api_llm import APILLM
from promptolution.predictors.classificator import Classificator
from promptolution.utils.prompt_creation import create_prompts_from_samples

from capo.capo import CAPOptimizer
from capo.statistical_tests import paired_t_test
from capo.task import CAPOClassificationTask

DEFAULT_POP_SIZE = 12
DEFAULT_LENGTH_PENALTY = 1e-3
DEFAULT_N_CROSSOVERS = 8

BLOCK_SIZE = 30
FS_SPLIT = 0.2

def evaluate_config(
    fit_task,
    eval_task,
    pop_size=DEFAULT_POP_SIZE,
    length_penalty=DEFAULT_LENGTH_PENALTY,
    n_crossovers=DEFAULT_N_CROSSOVERS,
):
    initial_prompts = [create_prompts_from_samples(fit_task, downstream_llm) for _ in range(pop_size)]

    optim = CAPOptimizer(
        initial_prompts=initial_prompts,
        task=fit_task,
        meta_llm=meta_llm,
        downstream_llm=downstream_llm,
        length_penalty=length_penalty,
        block_size=BLOCK_SIZE,
        crossovers_per_iter=n_crossovers,
        upper_shots=3,
        max_n_blocks_eval=10,
        p_few_shot_reasoning=0.5,
        few_shot_split_size=FS_SPLIT,
        test_statistic=test_statistic,
        predictor=predictor,
        callbacks=[callback],
        shuffle_blocks_per_iter=False,
    )

    prompts = optim.optimize(n_steps=5)
    scores = eval_task.evaluate(prompts, predictor)

    return prompts, scores

    
    

token = open("deepinfratoken.txt", "r").read()

meta_llm = APILLM("meta-llama/Meta-Llama-3-8B-Instruct", token)
downstream_llm = meta_llm

df_fit = pd.read_parquet(
    "hf://datasets/stanfordnlp/imdb/plain_text/train-00000-of-00001.parquet"
).sample(1_000)
task_fit = CAPOClassificationTask.from_dataframe(
    df_fit,
    description="Is the Movie review positive =1 or negative =0",
    x_column="text",
    y_column="label",
)
task_fit = CAPOClassificationTask.from_task(task_fit, block_size=BLOCK_SIZE, few_shot_split_size=FS_SPLIT)

df_eval = pd.read_parquet(
    "hf://datasets/stanfordnlp/imdb/plain_text/test-00000-of-00001.parquet"
).sample(1_000)

task_eval = CAPOClassificationTask.from_dataframe(
    df_eval,
    description="Is the Movie review positive =1 or negative =0",
    x_column="text",
    y_column="label",
)
task_eval = CAPOClassificationTask.from_task(task_eval, block_size=BLOCK_SIZE, few_shot_split_size=FS_SPLIT)

predictor = Classificator(downstream_llm, task_fit.classes)


test_statistic = lambda x, y: paired_t_test(x, y, alpha=0.2)

logger = getLogger(__name__)
callback = LoggerCallback(logger)

df_scores = dict(prompt=[], score=[], task=[])

for pop_size in [7, 12, 17, 22]:
    prompts, scores = evaluate_config(task_fit, task_eval, pop_size=pop_size)

