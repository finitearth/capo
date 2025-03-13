import argparse

import pandas as pd
from promptolution.llms import get_llm
from promptolution.predictors import Classificator

from capo.load_datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--csv-path", required=True)
parser.add_argument("--output-path", required=True)
parser.add_argument("--dataset", default="SetFit/rte")
parser.add_argument("--output-dir", default="results/")
parser.add_argument("--token", default=None)
parser.add_argument("--batch-size", type=int, default=None)
parser.add_argument("--revision", default="main")
parser.add_argument("--max-model-len", type=int, default=2000)
parser.add_argument("--model-storage-path", default="../models/")
parser.add_argument("--block-size", type=int, default=30)
parser.add_argument("--fs-split", type=float, default=0.1)
parser.add_argument("--n-steps", type=int, default=999)
parser.add_argument("--n-initial-prompts", type=int, default=10)
parser.add_argument("--n-eval-samples", type=int, default=300)
parser.add_argument("--max-tokens", type=int, default=5000000)
args = parser.parse_args()

if __name__ == "__main__":
    df = pd.read_csv(args.csv_path)
    # take best per step
    df = df.groupby("step").apply(lambda x: x.nlargest(1, "score")).reset_index(drop=True)
    prompts = df["prompt"].tolist()
    _, _, test_task = load_dataset(args.dataset, "", val_size=args.n_eval_samples)
    llm = get_llm(
        args.model,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        model_storage_path=args.model_storage_path,
        revision=args.revision,
        token=args.token,
    )

    predictor = Classificator(llm, df, test_task.classes)

    scores = test_task.evaluate(prompts, predictor)

    df["test_score"] = scores
    df.to_csv(args.output_path, index=False)

# do by handing over the path get the configuration
