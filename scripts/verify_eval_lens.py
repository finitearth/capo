from argparse import ArgumentParser
from glob import glob

import pandas as pd

parser = ArgumentParser()
parser.add_argument("--path", type=str, default="results/")

if __name__ == "__main__":
    args = parser.parse_args()
    complete_path = glob(args.path + "**/*.parquet", recursive=True)
    results = pd.DataFrame()
    c = 0
    for path in complete_path:
        if "step_results_eval.parquet" not in path:
            continue
        df_eval = pd.read_csv(path)
        df_dev = pd.read_csv(path.replace("step_results_eval.parquet", "step_results_dev.parquet"))

        if len(df_eval) != len(df_dev):
            print(f"{path}: length mismatch")
        c += 1

    print(f"Total number of evaluated jobs: {c}")
