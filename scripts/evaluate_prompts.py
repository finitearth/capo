"""
Script for evaluating optimized prompts resulting from experiments on unseen test data.
Measures the generalization performance of prompts produced by different optimization methods.
example usage: python scripts/evaluate_prompts.py --experiment-path results/ --validation-size 500 --max-tokens 10000000
"""

import argparse
import json
import os
from glob import glob
from logging import getLogger

import pandas as pd
from promptolution.llms import get_llm
from promptolution.predictors import MarkerBasedClassificator

from capo.load_datasets import get_tasks

logger = getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--experiment-path", type=str, default="results/")
parser.add_argument("--find-unevaluated", action="store_true")
parser.add_argument("--n-val-samples", type=int, default=500)
parser.add_argument("--max-tokens", type=int, default=5000000)
parser.add_argument("--only-best", action="store_true")
parser.add_argument("--reverse", action="store_true")
args = parser.parse_args()


def run_evaluation(experiment_path: str):
    # read experiment args
    with open(f"{experiment_path}args.json", "r") as f:
        experiment_args = json.load(f)
    logger.critical(f"Running evaluation with args: {experiment_args}")
    # read experiment results by using the best prompt per step from the step_results.csv
    df = pd.read_parquet(f"{experiment_path}step_results.parquet")

    # take best per step
    if args.only_best:
        df = df.groupby("step").apply(lambda x: x.nlargest(1, "score")).reset_index(drop=True)

    prompts = df["prompt"].unique().tolist()

    sys_prompt = (
        df["system_prompt"].unique().tolist()[0] if "system_prompt" in df.columns else None
    )  # for prompt wizard
    logger.critical(f"Found {len(prompts)} unique prompts")
    _, _, test_task = get_tasks(
        dataset_name=experiment_args["dataset"],
        optimizer_name=experiment_args["optimizer"],
        seed=42,
        block_size=experiment_args["block_size"],
        test_size=args.n_val_samples,
    )

    llm = get_llm(
        model_id=experiment_args["model"],
        max_model_len=experiment_args["max_model_len"],
        batch_size=experiment_args["batch_size"],
        model_storage_path=experiment_args["model_storage_path"],
        revision=experiment_args["model_revision"],
        seed=42,
    )
    predictor = MarkerBasedClassificator(llm=llm, classes=test_task.classes)

    scores = test_task.evaluate(prompts, predictor, system_prompts=sys_prompt)

    df_results = pd.DataFrame({"prompt": prompts, "test_score": scores})

    # save results to the step_results as extra column by joining on the prompt
    df = df.merge(df_results, on="prompt", how="left")
    # delete the empty file to allow other evaluations
    df.to_csv(f"{experiment_path}step_results_eval.csv", index=False)

    logger.critical(f"Finished evaluation of {experiment_path}")


if __name__ == "__main__":
    if args.find_unevaluated:
        experiments = glob(f"{args.experiment_path}**/step_results.parquet", recursive=True)
        if args.reverse:
            experiments = experiments[::-1]
        logger.critical(f"Found {len(experiments)} experiments")
        for experiment in experiments:
            if not os.path.exists(
                experiment.replace("step_results.parquet", "step_results_eval.parquet")
            ):
                experiment_path = experiment.replace("step_results.parquet", "")
                run_evaluation(experiment_path)
            else:
                logger.critical(f"Skipping {experiment} as it was already evaluated")
    else:
        run_evaluation(args.experiment_path)
