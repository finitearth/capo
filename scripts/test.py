from promptolution.predictors.classificator import Classificator
from promptolution.utils.prompt_creation import create_prompts_from_samples
from promptolution.llms.api_llm import APILLM
from promptolution.tasks import ClassificationTask

from capo.capo import CAPOptimizer
from capo.statistical_tests import hoeffdings_inequality_test_diff

import pandas as pd

token = open("deepinfratoken.txt", "r").read()

meta_llm = APILLM("meta-llama/Meta-Llama-3-8B-Instruct", token)
downstream_llm = meta_llm
# Create the task – note that the task already loads its dataset.
df = pd.read_csv(
    "hf://datasets/nakamoto-yama/dt-mappings/yama_dt_mappings.csv"
).rename({"Degree Type": "x", "Mapping": "y"}, axis=1)
task = ClassificationTask.from_dataframe(df, description="test test")
# Ensure task.xs and task.ys are set (here, converting DataFrame columns to numpy arrays)
task.xs = df["x"].to_numpy()
task.ys = df["y"].to_numpy()

initial_prompts = [
    create_prompts_from_samples(task, downstream_llm) for _ in range(10)
]

predictor = Classificator(downstream_llm, df["y"].unique())
test_statistic = lambda x, y, n: hoeffdings_inequality_test_diff(x, y, n, delta=0.5)

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
    test_statistic=test_statistic,
    predictor=predictor,
)
best_prompts = optimizer.optimize(n_steps=3)
print(f"Best instructions:\n\n{[p.construct_prompt() for p in best_prompts]}")