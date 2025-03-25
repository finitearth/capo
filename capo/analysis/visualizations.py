from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns

from capo.analysis.style import set_style
from capo.analysis.utils import aggregate_results, get_results

set_style()


def plot_population_scores(
    dataset,
    model,
    optim,
    agg="mean",
    plot_seeds=False,
    plot_stddev=False,
    score_col="test_score",
    x_col="step",
    seed_linestyle="--",
    ax=None,
    color=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    df = get_results(dataset, model, optim)
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
                    drawstyle="steps-pre",
                    ax=ax,
                    color=color,
                    alpha=0.5,
                )

    # Calculate and plot the mean across seeds (but only if all seeds are available at the given x_col)
    grouped = df.groupby(x_col)
    filtered_df = grouped.filter(lambda x: len(x) == seeds_count)
    mean_df = filtered_df.groupby(x_col)[score_col].agg("mean").reset_index()

    if "tokens" in x_col:
        ax.set_xlim(0, 5_000_000)

    if len(mean_df) == 1:
        y_value = mean_df[score_col].iloc[0]
        ax.axhline(y=y_value, color=color, linewidth=1.5, linestyle=seed_linestyle)
        # add a marker at the single point
        ax.plot(
            mean_df[x_col],
            mean_df[score_col],
            marker="*",
            linestyle=seed_linestyle,
            markersize=10,
            color=color,
            label=f"{optim}",
        )
    else:
        # Plot the mean line as before
        ax.plot(
            mean_df[x_col],
            mean_df[score_col],
            linewidth=2.5,
            markersize=4,
            drawstyle="steps-pre",
            label=f"{optim}",
            color=color,
        )

        # Add standard deviation shading if requested
        if plot_stddev:
            std_df = filtered_df.groupby(x_col)[score_col].agg("std").reset_index()
            ax.fill_between(
                mean_df[x_col],
                mean_df[score_col] - std_df[score_col],
                mean_df[score_col] + std_df[score_col],
                alpha=0.3,
                color=color,
                step="pre",
            )

    return ax


def plot_population_scores_comparison(
    dataset,
    model,
    optims,
    agg="mean",
    plot_seeds=False,
    plot_stddev=False,
    score_col="test_score",
    x_col="step",
    seed_linestyle="--",
):
    fig, ax = plt.subplots(figsize=(6, 5))

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
            color=sns.color_palette("Dark2")[i],
            ax=ax,
        )

    # Set title and layout for the comparison plot
    # ax.set_title(f"Score Comparison ({agg}) on {dataset} using {model}", y=1.25)
    ax.set_xlabel(x_col)
    ax.set_ylabel(score_col)

    # Improve legend placement and formatting
    ax.legend(ncols=min(len(optims), 2), loc="upper center", bbox_to_anchor=(0.5, -0.25))

    plt.tight_layout()
    return fig


def plot_population_members(
    dataset, model, optim, x_col="step", score_col="test_score", seeds=[42, 43, 44]
):
    fig, ax = plt.subplots()
    df = get_results(dataset, model, optim)

    # Filter the dataframe to only include the specified seeds
    df = df[df["seed"].isin(seeds)]
    df["seed"] = df["seed"].astype(str)

    df_old_prompt = df[(~df["is_new"]) & (~df["is_last_occ"])].copy()
    df_old_prompt["category"] = "survived"

    df_new_prompt = df[(df["is_new"]) & (~df["is_last_occ"])].copy()
    df_new_prompt["category"] = "new"

    df_last_occ = df[df["is_last_occ"]].copy()
    df_last_occ["category"] = "killed"

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

    # Customize the plot
    ax.set_xlabel(x_col)
    ax.set_ylabel(score_col)
    ax.set_title(f"Score of {optim} on {dataset} using {model}")

    # Adjust legend
    ax.legend(ncols=min(len(optim), 3), loc="upper center", bbox_to_anchor=(0.5, -0.25))

    plt.tight_layout()
    return fig


def plot_length_score(
    dataset,
    model,
    optims,
    x_col: Literal["prompt_len", "instr_len"],
    score_col: Literal["score", "test_score"],
    log_scale=True,
):
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = sns.color_palette("Dark2")

    for i, optim in enumerate(optims):
        df = get_results(dataset, model, optim)
        df = df.sort_values(by=["step", score_col])

        df_last_step = df.groupby(["seed"]).last()
        df = df[~df.index.isin(df_last_step.index)]

        color = colors[i]
        sns.scatterplot(data=df, x=x_col, y=score_col, ax=ax, label=optim, color=color, alpha=0.5)

        sns.scatterplot(
            data=df_last_step,
            x=x_col,
            y=score_col,
            ax=ax,
            color=color,
            marker="*",
            s=300,
            edgecolor="black",
            label=f"Best per Seed - {optim}",
        )

    if log_scale:
        ax.set_xscale("log")

    ax.set_xlabel("Prompt Length")
    ax.set_ylabel("Score")
    ax.set_title(f"Score vs. Number of Tokens on {dataset} using {model}")
    ax.legend(ncols=min(len(optims), 2), loc="upper center", bbox_to_anchor=(0.5, -0.25))

    plt.tight_layout()
    return fig
