from glob import glob

import numpy as np
import pandas as pd


def get_results(dataset, model, optim):
    """Get the evaluated step results for a given combination."""
    files = glob(
        f"../results/{dataset}/{model}/{optim}/*/*/*/step_results_eval.csv", recursive=True
    )
    seeds = [int(f.replace("sst-5", "sst5").split("\\")[-4].split("seed")[-1]) for f in files]
    df = pd.concat([pd.read_csv(p).assign(seed=seed) for seed, p in zip(seeds, files)], axis=0)

    # Add a score column if it doesn't exist (e.g. for PromptWizard)
    if "score" not in df.columns:
        df["score"] = 0

    df["input_tokens_sum"] = df["input_tokens_meta_llm"] + df["input_tokens_downstream_llm"]
    df["output_tokens_sum"] = df["output_tokens_meta_llm"] + df["output_tokens_downstream_llm"]

    # calculate the cumulative sum of tokens
    tokens_df = (
        df.groupby(["seed", "step"])
        .first()[["input_tokens_sum", "output_tokens_sum"]]
        .reset_index()
    )
    # calculate the cumulative sum of tokens for each seed
    tokens_df["input_tokens_cum"] = tokens_df.groupby("seed")["input_tokens_sum"].cumsum()
    tokens_df["output_tokens_cum"] = tokens_df.groupby("seed")["output_tokens_sum"].cumsum()

    # merge the cumulative sum of tokens
    df = df.merge(
        tokens_df[["input_tokens_cum", "output_tokens_cum", "seed", "step"]], on=["seed", "step"]
    )

    # caluclate prompt lengths
    df["instr_len"] = df["prompt"].str.split("Input:").apply(lambda x: x[0]).str.split().apply(len)
    df["prompt_len"] = df["prompt"].str.split().apply(len)

    return df


def aggregate_results(df: pd.DataFrame, how="mean", ffill_col="step"):
    """Aggregate the results for each step."""
    if how == "mean":
        df = df.groupby([ffill_col, "seed"], as_index=False).mean(numeric_only=True)
    elif how == "median":
        df = df.groupby([ffill_col, "seed"], as_index=False).median(numeric_only=True)
    elif how == "best_test":
        df = df.groupby([ffill_col, "seed"], as_index=False).apply(
            lambda x: x.loc[x["test_score"].idxmax()]
        )
    elif how == "best_train":
        df = df.groupby([ffill_col, "seed"], as_index=False).apply(
            lambda x: x.loc[x["score"].idxmax()]
        )
    else:
        raise ValueError(f"Unknown aggregation method: {how}")

    if "tokens" in ffill_col:
        unique_token_counts = df[ffill_col].unique()
        seeds = df["seed"].unique()
        pseudo_steps = pd.DataFrame(
            {
                ffill_col: np.repeat(unique_token_counts, len(seeds)),
                "seed": np.tile(seeds, len(unique_token_counts)),
            }
        )

        # merge the pseudo steps with the original dataframe
        df = pseudo_steps.merge(df, on=[ffill_col, "seed"], how="left")

        # group the dataframe by seed and sort by ffill column then call ffill for each group
        df = df.sort_values(by=["seed", ffill_col])

        df = df.groupby("seed").apply(lambda x: x.ffill()).reset_index(drop=True)

    return df
