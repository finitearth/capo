from argparse import ArgumentParser
from glob import glob

import pandas as pd

parser = ArgumentParser()
parser.add_argument("--path", type=str, default="results/")

if __name__ == "__main__":
    args = parser.parse_args()
    complete_path = glob(args.path + "**/*.csv", recursive=True)
    results = pd.DataFrame()
    c = 0
    for path in complete_path:
        if "step_results_eval.csv" not in path:
            continue
        df_eval = pd.read_csv(path)
        df_dev = pd.read_parquet(path.replace("step_results_eval.csv", "step_results.parquet"))
        if len(df_eval) != len(df_dev):
            print(f"{path}: length mismatch")
        c += 1

    print(f"Total number of evaluated jobs: {c}")
