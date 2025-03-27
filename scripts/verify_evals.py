from glob import glob

import pandas as pd

if __name__ == "__main__":
    complete_path = glob("results/**/*.csv", recursive=True)
    # complete_path += glob("ablation_results/**/*.csv", recursive=True)
    # complete_path += glob("hp_results/**/*.csv", recursive=True)
    results = pd.DataFrame()
    c = 0
    for path in complete_path:
        if "step_results_eval.csv" not in path or "PromptWizard" in path:
            continue

        df_eval = pd.read_csv(path)
        df_dev = pd.read_parquet(path.replace("step_results_eval.csv", "step_results.parquet"))

        if len(df_eval) != len(df_dev):
            print(f"{path}: length mismatch")
            continue

        df_dev["input_tokens"] = (
            df_dev["input_tokens_meta_llm"] + df_dev["input_tokens_downstream_llm"]
        )

        tokens = df_dev.groupby("step").first()["input_tokens"]
        cum_token_map = tokens.cumsum().to_dict()
        df_dev["cum_token"] = df_dev.apply(lambda x: cum_token_map[x.step], axis=1)
        max_cum_token = df_dev["cum_token"].max()

        max_step = df_dev["step"].max()
        if max_step < 4:
            print(f"{path[:80]}: max step mismatch: {max_step}")
            continue

        if not 1_000_000 < max_cum_token:
            print(f"{path[:80]}: token count mismatch: {max_cum_token}")
            continue

        c += 1

    print(f"Total number of evaluated jobs: {c}")
