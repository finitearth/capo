from glob import glob
from typing import Literal

import numpy as np
import pandas as pd

UNINFORMATIVE_INIT_PROMPTS = [
    "Let's think step by step.",
    "",
    "Let's work this out in a step by step way to be sure we have the right answer.",
]


def get_results(dataset, model, optim, path_prefix=".."):
    """Get the evaluated step results for a given combination."""
    is_initial = "init" in optim.lower() and "generic_init" not in optim.lower()

    paths = [
        f"{path_prefix}/results/{dataset}/{model}/{optim}/*/*/*/step_results_eval.csv",
        f"{path_prefix}/ablation_results/{dataset}/{model}/{optim}/*/*/*/step_results_eval.csv",
        f"{path_prefix}/hp_results/{dataset}/{model}/{optim}/*/*/*/step_results_eval.csv",
    ]

    if is_initial:
        paths += [f"{path_prefix}/init_results/{dataset}/{model}/eval.csv"]

    files = []
    for path in paths:
        files.extend(glob(path))
    seeds = []
    for f in files:
        if not is_initial:
            seed = int(f.replace("sst-5", "sst5").split("\\")[-4].split("seed")[-1])
            seeds.append(seed)
        else:
            seeds.append(0)

    try:
        df = pd.concat([pd.read_csv(p).assign(seed=seed) for seed, p in zip(seeds, files)], axis=0)
    except Exception as e:
        print(f"Failed to load {dataset} for {optim}: {e}")
        return pd.DataFrame(
            columns=[
                "prompt",
                "score",
                "step",
                "seed",
                "test_score",
                "prompt_len",
                "instr_len",
                "system_prompt",
                "input_tokens_cum",
                "output_tokens_cum",
            ]
        )

    # Add a score column if it doesn't exist (e.g. for PromptWizard)
    if "score" not in df.columns:
        df["score"] = 0

    if is_initial:
        df["step"] = 0
        df["prompt"] = df["prompt"].fillna("")  # is actually empty string
        # drop uninformative prompts
        # print(df["prompt"])
        df = df[~df["prompt"].isin(UNINFORMATIVE_INIT_PROMPTS)]
        # treat each prompt as its own seed
        df["seed"] = df["prompt"]

    df = df.dropna(subset=["prompt"])

    df["input_tokens_sum"] = (
        df["input_tokens_meta_llm"] + df["input_tokens_downstream_llm"] if not is_initial else 0
    )
    df["output_tokens_sum"] = (
        df["output_tokens_meta_llm"] + df["output_tokens_downstream_llm"] if not is_initial else 0
    )

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
    if "system_prompt" not in df.columns:
        df["system_prompt"] = "You are a helpful assistant."

    try:
        df["few_shots"] = df["prompt"].str.split(r"\[Question\]|Input:").apply(lambda x: x[1:])
    except Exception as e:
        print(e)
        df["few_shots"] = [[]] * len(df)
    df["instr_len"] = (
        (df["system_prompt"] + " " + df["prompt"])
        .str.split("Input:")
        .apply(lambda x: x[0])
        .str.split()
        .apply(len)
    )
    df["prompt_len"] = (df["system_prompt"] + " " + df["prompt"]).str.split().apply(len)

    df["is_new"] = ~df.groupby(["seed", "prompt"]).cumcount().astype(bool)
    df["is_last_occ"] = (~df.groupby(["seed", "prompt"]).cumcount(ascending=False).astype(bool)) & (
        df["step"] != df["step"].max()
    )
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df


def aggregate_results(
    df: pd.DataFrame,
    how: Literal["mean", "median", "best_test", "best_train"] = "mean",
    ffill_col="step",
):
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
        # fill score col
        df["score"] = df["score"].fillna(0)
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

        # drop rows with NaN values
        df = df.dropna(subset=["score"])

    return df


def get_prompt_scores(dataset, model, optim, path_prefix="../results/"):
    """Get the scores for each prompt and block."""
    files = glob(
        f"{path_prefix}{dataset}/{model}/{optim}/*/*/*/prompt_scores.parquet", recursive=True
    )

    if not files:
        return pd.DataFrame()

    seeds = [int(f.replace("sst-5", "sst5").split("\\")[-4].split("seed")[-1]) for f in files]
    df = pd.concat([pd.read_parquet(p).assign(seed=seed) for seed, p in zip(seeds, files)], axis=0)
    return df


def generate_comparison_table(
    datasets=["sst-5", "agnews", "subj", "gsm8k", "copa"],
    optims=["Initial", "OPRO", "PromptWizard", "EvoPromptGA", "CAPO"],
    model: Literal["llama", "mistral", "qwen"] = "llama",
    cutoff_tokens: int = 5_000_000,
    score_col: str = "test_score",
    path_prefix="../results/",
):
    """Generate a comparison table for the given datasets and optimizers."""
    results = {"optimizer": [], "dataset": [], "mean": [], "std": []}
    for optim in optims:
        for dataset in datasets:
            df = get_results(dataset, model, optim, path_prefix)
            if len(df) == 0:
                print(f"No results found for {dataset} and {optim}")
                continue
            df = aggregate_results(df, how="best_train", ffill_col="step")
            steps_data = []
            for seed in df.seed.unique():
                df_seed = df[df.seed == seed]
                last_step = df_seed.loc[df_seed["input_tokens_cum"] < cutoff_tokens, "step"].max()

                step_df = df_seed[df_seed["step"] == last_step]
                steps_data.append(step_df)

            combined_df = pd.concat(steps_data)
            results["optimizer"].append(optim)
            results["dataset"].append(dataset)
            results["mean"].append(combined_df[score_col].mean())
            results["std"].append(combined_df[score_col].std())

    df = pd.DataFrame(results)
    df["optimizer"] = pd.Categorical(df["optimizer"], categories=optims, ordered=True)
    df["dataset"] = pd.Categorical(df["dataset"], categories=datasets, ordered=True)
    df = df.set_index("optimizer")
    df = df.pivot(columns="dataset")
    df["avg"] = df["mean"].mean(axis=1)
    if "score" in score_col:
        df["avg"] = df["avg"].mul(100)
        df["mean"] = df["mean"].mul(100)
        df["std"] = df["std"].mul(100)

    # For the avg column
    df["avg"] = df["avg"].round(2)
    df["avg"] = df["avg"].map(lambda x: f"{x:.2f}")  # Use map instead of apply

    # For the mean and std columns
    df["mean"] = df["mean"].round(2)
    df["std"] = df["std"].round(2)
    df["mean"] = df["mean"].map(lambda x: f"{x:.2f}") + "±" + df["std"].map(lambda x: f"{x:.2f}")
    df = df.drop(columns=["std"])
    df.columns = [col[1] if col[0] == "mean" else col[0] for col in df.columns]
    df.index.name = None
    df = df.style.highlight_max(axis=0, props="font-weight: bold;")

    return df
