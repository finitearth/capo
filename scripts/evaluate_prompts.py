import argparse
import json

import pandas as pd
from promptolution.llms import get_llm
from promptolution.predictors import MarkerBasedClassificator

from capo.load_datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--experiment-path", required=True)
parser.add_argument("--max-tokens", type=int, default=5000000)
args = parser.parse_args()

if __name__ == "__main__":
    # read experiment args
    with open(f"{args.experiment_path}args.json", "r") as f:
        experiment_args = json.load(f)

    # read experiment results by using the best prompt per step from the step_results.csv
    df = pd.read_csv(f"{args.experiment_path}step_results.csv")

    # take best per step
    df_results = df.groupby("step").apply(lambda x: x.nlargest(1, "score")).reset_index(drop=True)
    prompts = df_results["prompt"].unique().tolist()

    _, _, test_task = load_dataset(args.dataset, "", val_size=args.n_eval_samples)
    llm = get_llm(
        model=experiment_args["model"],
        max_model_len=experiment_args["max_model_len"],
        batch_size=experiment_args["batch_size"],
        model_storage_path=experiment_args["model_storage_path"],
        revision=experiment_args["model_revision"],
    )

    predictor = MarkerBasedClassificator(llm, df, test_task.classes)

    scores = test_task.evaluate(prompts, predictor)

    df_results["test_score"] = scores

    # save results to the step_results as extra column by joining on the prompt
    df = df.join(df_results.set_index("prompt")["test_score"], on="prompt")
    df.to_csv(f"{args.experiment_path}step_results.csv", index=False)
