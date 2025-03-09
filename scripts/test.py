import time
from logging import getLogger

import pandas as pd
from promptolution.callbacks import CSVCallback, LoggerCallback
from promptolution.llms import get_llm
from promptolution.predictors.classificator import Classificator
from promptolution.utils.prompt_creation import create_prompts_from_samples

from capo.capo import CAPOptimizer
from capo.statistical_tests import paired_t_test
from capo.task import CAPOClassificationTask

BLOCK_SIZE = 30
FS_SPLIT = 0.1
BATCH_SIZE = 256

# start time
start = time.time()

try:
    token = open("deepinfratoken.txt", "r").read()
except FileNotFoundError:
    token = None
# model_name = "vllm-Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

if "vllm" in model_name:
    llm = get_llm(
        model_name,
        max_model_len=512,
        batch_size=BATCH_SIZE,
        model_storage_path="../temp/",
    )
else:
    llm = get_llm(model_name, token)

downstream_llm = llm
meta_llm = llm

df = pd.read_json("hf://datasets/SetFit/sst5/train.jsonl", lines=True).sample(1000)

# map very negative to veryNegative, very positive to veryPositive
df["label_text"] = df["label_text"].map(
    {
        "very negative": "veryNegative",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
        "very positive": "veryPositive",
    }
)

task = CAPOClassificationTask.from_dataframe(
    df,
    description="The dataset consists of movie reviews with five levels of sentiment labels: veryNegative, negative, neutral, positive, and veryPositive. The task is to classify each movie review into one of these five sentiment categories. The class mentioned first in the response of the LLM will be the prediction.",
    x_column="text",
    y_column="label_text",
)


task = CAPOClassificationTask.from_task(task, block_size=BLOCK_SIZE, few_shot_split_size=FS_SPLIT)
task.classes = [str(c) for c in task.classes]
predictor = Classificator(downstream_llm, task.classes)
test_statistic = lambda x, y: paired_t_test(x, y, alpha=0.2)
initial_prompts = [
    create_prompts_from_samples(task, downstream_llm, n_samples=5) for _ in range(10)
]

logger = getLogger(__name__)
logger.setLevel("INFO")
callbacks = [LoggerCallback(logger), CSVCallback("temp/results/test.csv")]
logger.warning(f"Initial prompts: {initial_prompts}")
logger.warning("hier gehts los")

optimizer = CAPOptimizer(
    initial_prompts=initial_prompts,
    task=task,
    meta_llm=meta_llm,
    downstream_llm=downstream_llm,
    length_penalty=1e-5,
    block_size=BLOCK_SIZE,
    crossovers_per_iter=5,
    upper_shots=6,
    max_n_blocks_eval=30,
    p_few_shot_reasoning=0.5,
    few_shot_split_size=FS_SPLIT,
    test_statistic=test_statistic,
    predictor=predictor,
    callbacks=callbacks,
    shuffle_blocks_per_iter=False,
    verbosity=1,
)
best_prompts = optimizer.optimize(n_steps=10)
end = time.time()

print(f"Best instructions:\n\n {best_prompts}")
print(f"Time taken: {end - start} seconds")
