import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
        ax.axhline(y=y_value, color=color, linewidth=1.5, label=f"{optim}")
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
    ax.legend(ncols=min(len(optims), 2), loc=(0, 1))

    plt.tight_layout()
    return fig


def plot_population_members(dataset, model, optim, x_col="step", score_col="test_score"):
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots()
    df = get_results(dataset, model, optim)

    # Plot individual seeds and population members as scatter plot
    sns.scatterplot(data=df, x=x_col, y=score_col, hue="seed", ax=ax)

    # Customize the plot
    ax.set_xlabel(x_col)
    ax.set_ylabel(score_col)
    ax.set_title(f"Score of {optim} on {dataset} using {model}")

    # Adjust legend
    ax.legend(frameon=True, loc="best", fontsize=10, facecolor="white", edgecolor="gray")

    plt.tight_layout()
    return fig

def plot_performance_profile_curve(
        datasets=["sst-5", "agnews", "copa", "subj", "gsm8k"],
        models=["llama", "qwen", "mistral"],
        optims=["CAPO", "EvoPromptGA", "OPRO", "PromptWizard"]
    ):
    # get all results
    dfs = []
    for dataset in datasets:
        for model in models:
            for optim in optims:
                try:
                    df = get_results(dataset, model, optim)
                except:
                    print(f"Failed to load {dataset} {model} {optim}")
                    continue
                df = aggregate_results(df, how="best_train", ffill_col="step")
                df = df.assign(dataset=dataset, model=model, optim=optim)
                dfs.append(df)

    df = pd.concat(dfs)
    # avg across seeds
    df = df.groupby(["dataset", "model", "optim"], as_index=False).mean(numeric_only=True)

    # calculate the difference to the best score
    df["diff"] = df.groupby(["dataset", "model"])["test_score"].transform(lambda x: x.max() - x)
    taus = np.linspace(0, 0.3, 100)
    performance_profiles = []
    for optim in optims:
        for tau in taus:
            performance_profile = (df.loc[df["optim"] == optim, "diff"] <= tau).mean()
            performance_profiles.append(dict(optim=optim, tau=tau, performance_profile=performance_profile))

    df = pd.DataFrame(performance_profiles)
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="tau", y="performance_profile", hue="optim", ax=ax, drawstyle="steps-post")
    ax.legend(loc="best")

    # zoom into x-axis: 0 to 0.3
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 1)

    plt.show()