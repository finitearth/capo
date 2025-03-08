"""Script for running CAPO optimization with multiple models and datasets."""
import argparse
import time
from logging import getLogger

import pandas as pd
from promptolution.callbacks import LoggerCallback
from promptolution.llms import get_llm
from promptolution.predictors.classificator import Classificator
from promptolution.utils.prompt_creation import create_prompts_from_samples

from capo.capo import CAPOptimizer
from capo.statistical_tests import paired_t_test
from capo.task import CAPOClassificationTask

parser = argparse.ArgumentParser()
parser.add_argument("--models", required=True)
parser.add_argument("--dataset", default="SetFit/sst5")
parser.add_argument("--output", default="outputs/performance_tests.csv")
parser.add_argument("--token", default=None)
parser.add_argument("--batch-size", default=None)
parser.add_argument("--revision", default="main")
parser.add_argument("--max-model-len", type=int, default=2000)
parser.add_argument("--model-storage-path", default="../models/")
parser.add_argument("--block-size", type=int, default=30)
parser.add_argument("--fs-split", type=float, default=0.1)
parser.add_argument("--n-steps", type=int, default=10)
parser.add_argument("--n-initial-prompts", type=int, default=10)
parser.add_argument("--crossovers-per-iter", type=int, default=4)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--length-penalty", type=float, default=1e-5)
parser.add_argument("--upper-shots", type=int, default=3)
parser.add_argument("--max-n-blocks-eval", type=int, default=30)
parser.add_argument("--p-few-shot-reasoning", type=float, default=0.5)
args = parser.parse_args()

# TODO: investigate if we have to clear the cache after each model run
# TODO: implement revision for loaded datasets

# Create or load the results file
try:
    results_df = pd.read_csv(args.output)
except FileNotFoundError:
    results_df = pd.DataFrame(
        columns=[
            "model",
            "dataset",
            "block_size",
            "fs_split",
            "n_steps",
            "crossovers_per_iter",
            "alpha",
            "length_penalty",
            "upper_shots",
            "max_n_blocks_eval",
            "p_few_shot_reasoning",
            "time_seconds",
            "token_count",
            "best_prompt",
        ]
    )

# Load dataset
dataset_path = f"hf://datasets/{args.dataset}/train.jsonl"
df = pd.read_json(dataset_path, lines=True)

# Run optimization for each model
for model_name in args.models.strip("[]").split(","):
    print(f"\n\n{'=' * 40}\nRunning optimization for model: {model_name}\n{'=' * 40}\n")

    start_time = time.time()

    # Set up CAPO task
    task = CAPOClassificationTask.from_dataframe(
        df,
        description="The dataset consists of text samples with sentiment labels. The task is to classify each text into the correct sentiment category. The class mentioned first in the response of the LLM will be the prediction.",
        x_column="text",
        y_column="label",
    )
    task = CAPOClassificationTask.from_task(
        task, block_size=args.block_size, few_shot_split_size=args.fs_split
    )

    # Set up LLM
    if "vllm" in model_name:
        llm = get_llm(
            model_name,
            max_model_len=args.max_model_len,
            batch_size=args.batch_size,
            model_storage_path=args.model_storage_path,
            # revision=args.revision,
        )
    else:
        llm = get_llm(model_name, args.token)

    downstream_llm = llm
    meta_llm = llm

    # Create initial prompts
    initial_prompts = [
        create_prompts_from_samples(task, downstream_llm) for _ in range(args.n_initial_prompts)
    ]

    # Set up predictor and callbacks
    predictor = Classificator(downstream_llm, task.classes)
    test_statistic = lambda x, y: paired_t_test(x, y, alpha=args.alpha)
    logger = getLogger(__name__)
    callback = LoggerCallback(logger)

    # Initialize optimizer
    optimizer = CAPOptimizer(
        initial_prompts=initial_prompts,
        task=task,
        meta_llm=meta_llm,
        downstream_llm=downstream_llm,
        length_penalty=args.length_penalty,
        block_size=args.block_size,
        crossovers_per_iter=args.crossovers_per_iter,
        upper_shots=args.upper_shots,
        max_n_blocks_eval=args.max_n_blocks_eval,
        p_few_shot_reasoning=args.p_few_shot_reasoning,
        few_shot_split_size=args.fs_split,
        test_statistic=test_statistic,
        predictor=predictor,
        callbacks=[callback],
        shuffle_blocks_per_iter=False,
    )

    # Run optimization
    best_prompts = optimizer.optimize(n_steps=args.n_steps)
    best_prompt = best_prompts[0] if best_prompts else ""

    end_time = time.time()
    total_time = end_time - start_time

    # Print results
    print(f"\nBest instructions for {model_name}:\n\n{best_prompt}")
    print(f"Time taken: {total_time:.2f} seconds")
    # input_tokens = llm.get_token_count()["input_tokens"]
    # output_tokens = llm.get_token_count()["output_tokens"]

    # Create new row
    new_result = pd.DataFrame(
        {
            "model": [model_name],
            "dataset": [args.dataset],
            "time_seconds": [total_time],
            # "input_tokens": [input_tokens],
            # "output_tokens": [output_tokens],
            "block_size": [args.block_size],
            "fs_split": [args.fs_split],
            "n_steps": [args.n_steps],
            "crossovers_per_iter": [args.crossovers_per_iter],
            "alpha": [args.alpha],
            "length_penalty": [args.length_penalty],
            "upper_shots": [args.upper_shots],
            "max_n_blocks_eval": [args.max_n_blocks_eval],
            "p_few_shot_reasoning": [args.p_few_shot_reasoning],
            "best_prompt": [best_prompt],
        }
    )

    # Append to CSV
    new_result.to_csv(args.output, mode="a", header=False, index=False)
    print(f"Results for {model_name} saved to {args.output}")

print("\nAll optimization runs completed successfully!")
