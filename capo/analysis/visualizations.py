"""
Implementation of plotting functions for all paper visualizations.
Contains specialized chart generators for performance comparisons, ablation studies, and hyperparameter analysis.
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from capo.analysis.style import set_style
from capo.analysis.utils import aggregate_results, get_results

set_style()


def plot_population_scores(
    dataset: str,
    model: str,
    optim: str,
    agg: str = "mean",
    plot_seeds: bool = False,
    plot_stddev: bool = False,
    score_col: str = "test_score",
    x_col: str = "step",
    seed_linestyle: str = "--",
    mean_linestyle: str = "-",
    ax=None,
    color: list[str] = None,
    add_title: bool = False,
    add_legend: bool = False,
    label_suffix: str = "",
    fillstyle: str = None,
    marker: list["str"] = None,
    label: list["str"] = None,
    n_seeds_to_plot_std: int = 1,
):
    if ax is None:
        fig, ax = plt.subplots()

    df = get_results(dataset, model, optim)
    if len(df) == 0:
        return ax

    if optim not in ["Initial", "PromptWizard"]:
        df = aggregate_results(df, how=agg, ffill_col=x_col)

    # Plot individual seeds if requested
    seeds_count = df["seed"].nunique()
    if plot_seeds:
        for seed in df["seed"].unique():
            df_seed = df[df["seed"] == seed]
            if len(df_seed) > seeds_count:
                sns.lineplot(
                    data=df_seed,
                    x=x_col,
                    y=score_col,
                    linestyle=seed_linestyle,
                    # label=f"{optim} - Seed {seed}",
                    drawstyle="steps-post",
                    ax=ax,
                    color=color,
                    alpha=0.5,
                )

    # Calculate and plot the mean across seeds (but only if all seeds are available at the given x_col)
    grouped = df.groupby(x_col)
    filtered_df = grouped.filter(lambda x: len(x) >= n_seeds_to_plot_std)
    mean_df = filtered_df.groupby(x_col)[score_col].agg("mean").reset_index()
    std_df = filtered_df.groupby(x_col)[score_col].agg(lambda x: x.std(ddof=0)).reset_index()

    if "tokens" in x_col:
        ax.set_xlim(0, 5_000_000)

    if len(filtered_df.step.unique()) == 1:
        x_value = filtered_df[x_col].mean()
        y_value = mean_df[score_col].mean()
        ax.axhline(y=y_value, color=color, linewidth=1.5, linestyle=seed_linestyle)

        ax.plot(
            mean_df[x_col][0],
            y_value,
            marker=None if optim == "Initial" else "*",
            linestyle=seed_linestyle,
            markersize=10,
            color=color,
            fillstyle="full" if fillstyle is None else fillstyle,
            label=label if label is not None else f"{optim}{label_suffix}",
        )

        if plot_stddev and optim != "Initial":
            ax.errorbar(
                x_value,
                y_value,
                yerr=mean_df[score_col].std(ddof=0),
                color=color,
                linestyle=seed_linestyle,
                capsize=5,
                alpha=0.5,
            )
    else:
        ax.plot(
            mean_df[x_col],
            mean_df[score_col],
            markersize=3,
            marker=marker,
            drawstyle="steps-post",
            linestyle=mean_linestyle,
            label=label if label is not None else f"{optim}{label_suffix}",
            color=color,
        )

        # if the max x_col is less than 5_000_00 (if tokens) then plot a horizontal line at the last y value (from the last x value up to 5_000_000)
        if "tokens" in x_col and mean_df[x_col].max() < 5_000_000:
            ax.plot(
                [mean_df[x_col].max(), 5_000_000],
                [mean_df[score_col].iloc[-1], mean_df[score_col].iloc[-1]],
                linestyle=seed_linestyle,
                color=color,
            )

        # Add standard deviation shading if requested
        if plot_stddev:
            std_df = filtered_df.groupby(x_col)[score_col].std(ddof=0).reset_index()
            ax.fill_between(
                mean_df[x_col],
                mean_df[score_col] - std_df[score_col],
                mean_df[score_col] + std_df[score_col],
                alpha=0.3,
                color=color,
                edgecolor="w",
                step="post",
            )

            if "tokens" in x_col and mean_df[x_col].max() < 5_000_000:
                ax.fill_between(
                    [mean_df[x_col].max(), 5_000_000],
                    mean_df[score_col].iloc[-1] - std_df[score_col].iloc[-1],
                    mean_df[score_col].iloc[-1] + std_df[score_col].iloc[-1],
                    alpha=0.1,
                    color=color,
                    edgecolor=color,
                    linewidth=0,
                    hatch_linewidth=2,
                    hatch="///",
                    step="post",
                )

    if add_title:
        ax.set_title(f"{optim} on {dataset} using {model}")
    if add_legend:
        ax.legend(loc="best")

    return ax


def plot_population_scores_comparison(
    dataset: str,
    model: str,
    optims: list[str],
    agg: str = "mean",
    plot_seeds: bool = False,
    plot_stddev: bool = False,
    score_col: str = "test_score",
    x_col: str = "step",
    seed_linestyle: str = "--",
    figsize: tuple[float] = (5.4, 3.6),
    colors: list[str] = None,
    continuous_colors: bool = False,
    markers: bool = False,
    labels: list[str] = None,
    n_seeds_to_plot_std: int = 1,
    ncols: int = 3,
):
    fig, ax = plt.subplots(figsize=figsize)

    if continuous_colors:
        cmap = sns.color_palette("blend:#001843,#006C7C,#1B9E77,#66D874,#DAF9CC", as_cmap=True)
        colors = [cmap(i / len(optims)) for i in range(len(optims) + 1)]
    elif colors is None:
        colors = sns.color_palette("Dark2")

    # Plot each optimizer on the same axes
    for i, optim in enumerate(optims):
        plot_population_scores(
            dataset,
            model,
            optim,
            agg=agg,
            plot_seeds=plot_seeds,
            plot_stddev=plot_stddev,
            score_col=score_col,
            x_col=x_col,
            seed_linestyle=seed_linestyle,
            color=colors[i],
            ax=ax,
            marker=markers[i] if markers else None,
            label=labels[i] if labels else None,
            n_seeds_to_plot_std=n_seeds_to_plot_std,
        )

    x_col = (
        " ".join(x_col.split("_")[:-1]).capitalize()
        if "cum" in x_col
        else x_col.replace("_", " ").capitalize()
    )
    ax.set_xlabel(x_col)
    y_col = (
        score_col.replace("len", "length").replace("_", " ").capitalize()
        if "len" in score_col
        else score_col.replace("_", " ").capitalize()
    )
    ax.set_ylabel(y_col)

    ax.legend(ncols=min(len(optims), ncols), loc="upper center", bbox_to_anchor=(0.5, 1.25))

    return fig


def plot_population_members(
    dataset: str,
    model: str,
    optim: str,
    x_col: str = "step",
    score_col: str = "test_score",
    seeds: list[int] = [42, 43, 44],
    figsize: tuple[float] = (5.4, 3.6),
):
    fig, ax = plt.subplots(figsize=figsize)
    df = get_results(dataset, model, optim)

    # Filter the dataframe to only include the specified seeds
    df = df[df["seed"].isin(seeds)]
    df["seed"] = df["seed"].astype(str)

    df_last_occ = df[df["is_last_occ"]].copy()
    for seed in seeds:
        max_step = df[df["seed"] == str(seed)]["step"].max()
        df_last_occ = df_last_occ[~df_last_occ["step"].isin([max_step])]
        df_last_occ["category"] = "killed"

    df_old_prompt = df[(~df["is_new"]) & (~df.index.isin(df_last_occ.index))].copy()
    df_old_prompt["category"] = "survived"

    df_new_prompt = df[(df["is_new"]) & (~df.index.isin(df_last_occ.index))].copy()
    df_new_prompt["category"] = "new"
    # Plot each category
    sns.scatterplot(
        data=df_old_prompt,
        x=x_col,
        y=score_col,
        ax=ax,
        s=20,
        alpha=0.3,
        color=sns.color_palette("Dark2")[0],
        hue="category",
    )

    sns.scatterplot(
        data=df_new_prompt,
        x=x_col,
        y=score_col,
        ax=ax,
        s=90,
        style="category",
        markers=[(8, 1, 0)],
        color=sns.color_palette("Dark2")[0],
    )

    sns.scatterplot(
        data=df_last_occ,
        x=x_col,
        y=score_col,
        ax=ax,
        s=70,
        style="category",
        markers=["X"],
        color=sns.color_palette("Dark2")[0],
    )
    df_max = df.groupby(["seed", x_col])[score_col].max().reset_index()
    sns.lineplot(
        data=df_max,
        x=x_col,
        y=score_col,
        ax=ax,
        style="seed",
        dashes=False,
        legend=False,
        alpha=0.7,
    )

    x_col = (
        " ".join(x_col.split("_")[:-1]).capitalize()
        if "cum" in x_col
        else x_col.replace("_", " ").capitalize()
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(score_col.replace("_", " ").capitalize())

    ax.legend(ncols=min(len(optim), 3), loc="upper center", bbox_to_anchor=(0.5, 1.15))

    return fig


def plot_length_score(
    dataset: str,
    model: str,
    optims: list[str],
    x_col: Literal["prompt_len", "instr_len"],
    score_col: Literal["score", "test_score"],
    log_scale: bool = True,
    colors: list[str] = None,
    labels: list[str] = None,
    figsize: tuple[float] = (5.4, 3.6),
    ncols: int = 3,
):
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = sns.color_palette("Dark2")

    for i, optim in enumerate(optims):
        df = get_results(dataset, model, optim)
        df = df.sort_values(by=["step", "score"])

        df_best_last_step = df.groupby(["seed"]).last()
        df = df[~df.index.isin(df_best_last_step.index)]

        df = df.drop_duplicates(subset=["prompt"], keep="last")

        color = colors[i]
        sns.scatterplot(
            data=df,
            x=x_col,
            y=score_col,
            ax=ax,
            label=labels[i] if labels else optim,
            color=color,
            s=70,
            alpha=0.5,
        )

        sns.scatterplot(
            data=df_best_last_step,
            x=x_col,
            y=score_col,
            ax=ax,
            color=color,
            marker="*",
            s=300,
            linewidth=0.5,
            edgecolor="black",
        )

    if log_scale:
        ax.set_xscale("log")

    ax.scatter([], [], marker="*", s=300, edgecolor="black", facecolor="none", label="Best")
    ax.set_xlabel("Prompt length")
    ax.set_ylabel(score_col.replace("_", " ").capitalize())
    ax.legend(ncols=min(len(optims), ncols), loc="upper center", bbox_to_anchor=(0.5, 1.25))

    return fig


def plot_performance_profile_curve(
    datasets: list[str] = ["sst-5", "agnews", "copa", "subj", "gsm8k"],
    models: list[str] = ["llama", "qwen", "mistral"],
    optims: list[str] = ["CAPO", "OPRO", "EvoPromptGA", "PromptWizard"],
    markers: list[str] = ["o", "s", "p", "d"],
    x_max: float = 0.5,
):
    # get all results
    dfs = []
    for dataset in datasets:
        for model in models:
            for optim in optims:
                df = get_results(dataset, model, optim)
                df = aggregate_results(df, how="best_train", ffill_col="step")
                # only take last step per seed
                df = df.groupby("seed").last().reset_index()
                df = df.assign(dataset=dataset, model=model, optim=optim)
                dfs.append(df)

    df = pd.concat(dfs)
    # avg across seeds
    df = df.groupby(["dataset", "model", "optim"], as_index=False).mean(numeric_only=True)

    # calculate the difference to the best score
    df["diff"] = df.groupby(["dataset", "model"])["test_score"].transform(lambda x: x.max() - x)

    taus = np.linspace(0, 1, 100)
    performance_profiles = []
    for optim in optims:
        for tau in taus:
            performance_profile = (
                df.loc[df["optim"] == optim, "diff"] <= tau
            ).mean()  # fraction of datasets where the difference is less than tau
            performance_profiles.append(
                dict(optim=optim, tau=tau, performance_profile=performance_profile)
            )

    df = pd.DataFrame(performance_profiles)

    # drop rows where the performance profile does not change compared to the previous one
    df = df[df["performance_profile"].diff().ne(0)]
    # add a row for each optim with tau=1 and performance_profile=1
    for optim in optims:
        df = pd.concat(
            [df, pd.DataFrame({"optim": [optim], "tau": [1], "performance_profile": [1]})],
            ignore_index=True,
        )

    fig, ax = plt.subplots()
    for i, optim in enumerate(optims):
        optim_data = df[df["optim"] == optim]
        ax.step(
            optim_data["tau"],
            optim_data["performance_profile"],
            where="post",
            color=sns.color_palette("Dark2")[i],
            marker=markers[i],
            markersize=8,
            alpha=0.6,
            label=optim,
            linewidth=3,
        )
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\rho(\tau)$")

    ax.legend(ncols=min(len(optims), 2), loc="upper center", bbox_to_anchor=(0.5, 1.25))

    # zoom into x-axis: 0 to 0.3
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.03)

    return fig


def plot_train_test_comparison(
    dataset: str,
    model: str,
    optims: list[str],
    agg: str = "mean",
    plot_seeds: bool = False,
    plot_stddev: bool = False,
    x_col: str = "step",
    seed_linestyle: str = "--",
    add_title: bool = True,
):
    fig, ax = plt.subplots()

    # Plot each optimizer on the same axes
    for i, optim in enumerate(optims):
        plot_population_scores(
            dataset,
            model,
            optim,
            agg=agg,
            plot_seeds=plot_seeds,
            plot_stddev=plot_stddev,
            score_col="score",
            x_col=x_col,
            seed_linestyle=seed_linestyle,
            color=sns.color_palette("Dark2")[i],
            ax=ax,
            mean_linestyle="--",
            label_suffix=" (Train)",
            fillstyle="none",
        )
        plot_population_scores(
            dataset,
            model,
            optim,
            agg=agg,
            plot_seeds=plot_seeds,
            plot_stddev=plot_stddev,
            score_col="test_score",
            x_col=x_col,
            seed_linestyle=seed_linestyle,
            color=sns.color_palette("Dark2")[i],
            ax=ax,
            label_suffix=" (Test)",
        )
    if add_title:
        ax.set_title(f"Train/Test Score Comparison ({agg}) on {dataset} using {model}", y=1.25)
    x_col = (
        " ".join(x_col.split("_")[:-1]).capitalize()
        if "cum" in x_col
        else x_col.replace("_", " ").capitalize()
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel("Score")

    ax.legend(ncols=min(len(optims), 3), loc="upper center", bbox_to_anchor=(0.5, -0.25))

    return fig


def plot_few_shot_boxplots(
    dataset: str, model: str, optim: str, score_column: str = "test_score", top_k: int = 3
):
    df = get_results(dataset, model, optim)

    # Calculate mean test score for each feature when it's True
    feature_scores = {}
    number_occ = {}

    unique_few_shots = df["few_shots"].explode().dropna().unique()

    few_shots = []
    for x in unique_few_shots:
        x = x.split("Output:")[0].strip()
        if x != "":
            few_shots.append(x)

    for element_id, element in enumerate(few_shots):
        df[f"has_{element_id}"] = df["few_shots"].apply(lambda x: any([element in y for y in x]))

    df["has_none"] = df["few_shots"].apply(lambda x: len(x) == 0)

    for feature_id, _ in enumerate(few_shots):
        feature_name = f"has_{feature_id}"
        # Get only True values and calculate mean score
        scores = df.loc[df[feature_name], score_column]
        if len(scores) >= 3:  # Only consider features with enough data
            feature_scores[feature_id] = scores.mean()
            number_occ[feature_id] = len(scores)

    # Sort features by mean score and get top k
    top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_feature_ids = [feature_id for feature_id, _ in top_features]

    # Create a long-format dataframe for seaborn with only top features
    plot_data = []

    # Add "ALL" population data first
    for score in df[score_column]:
        plot_data.append({"few_shot": "ALL", "Score": score, "Occurences": pd.NA})

    # Add "NONE" population data
    for score in df.loc[df["has_none"], score_column]:
        plot_data.append({"few_shot": "NONE", "Score": score, "Occurence": pd.NA})

    # Then add top features
    for feature_id in top_feature_ids:
        feature_name = f"has_{feature_id}"
        scores = df.loc[df[feature_name], score_column]

        # Add each score with its feature label
        for score in scores:
            plot_data.append(
                {"few_shot": feature_id, "Score": score, "Occurences": number_occ[feature_id]}
            )

    plot_df = pd.DataFrame(plot_data)

    # Create the single plot with ALL population and top features
    plt.figure(figsize=(12, 6))

    # Plot with custom palette
    sns.boxplot(x="few_shot", y="Score", data=plot_df, hue="few_shot", legend=False)

    plt.tight_layout()
    plt.show()
