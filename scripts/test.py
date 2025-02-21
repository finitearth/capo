import pandas as pd
from promptolution.llms.api_llm import APILLM
from promptolution.predictors.classificator import Classificator
from promptolution.tasks import ClassificationTask
from promptolution.callbacks import LoggerCallback

from promptolution.utils.prompt_creation import create_prompts_from_samples
from capo.capo import CAPOptimizer
from capo.statistical_tests import hoeffdings_inequality_test_diff, paired_t_test
from capo.task import CAPOTask
from logging import getLogger
import pandas as pd



token = open("deepinfratoken.txt", "r").read()

meta_llm = APILLM("meta-llama/Meta-Llama-3-8B-Instruct", token)
downstream_llm = meta_llm

df = pd.read_parquet(
    "hf://datasets/stanfordnlp/imdb/plain_text/train-00000-of-00001.parquet"
).sample(300)
task = ClassificationTask.from_dataframe(
    df, description="Is the Movie review positive =1 or negative =0",
    x_column="text", y_column="label"
)

task = CAPOTask.from_task(task, block_size=30, few_shot_split_size=0.2)
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
    length_penalty=0.01,
    block_size=30,
    crossovers_per_iter=2,
    upper_shots=5,
    max_n_blocks_eval=5,
    few_shot_split_size=0.2,
    test_statistic=test_statistic,
    predictor=predictor,
    callbacks=[callback],
    shuffle_blocks_per_iter=False,
)
best_prompts = optimizer.optimize(n_steps=12)
print(f"Best instructions:\n\n {best_prompts}")
