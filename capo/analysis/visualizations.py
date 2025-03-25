from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

from capo.utils import aggregate_results, get_results

from capo.analysis.style import set_style

set_style()

def plot_population_scores(
    dataset,
    model,
    optim,
    agg="mean",
    plot_seeds=False,
    score_col="test_score",
    x_col="step",
    ffill_col="step",
    seed_linestyle="--",
):
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots()
    df = get_results(dataset, model, optim)
    df = aggregate_results(df, how=agg, ffill_col=ffill_col)

    # Plot individual seeds if requested
    if plot_seeds:
        for seed in df["seed"].unique():
            df_seed = df[df["seed"] == seed]
            sns.lineplot(
                data=df_seed,
                x=x_col,
                y=score_col,
                linestyle=seed_linestyle,
                label=f"Seed {seed}",
                drawstyle="steps-pre",
                ax=ax,
            )

    # Calculate and plot the mean across seeds
    mean_df = df.groupby(x_col, as_index=False)[score_col].mean()

    ax.plot(
        mean_df[x_col],
        mean_df[score_col],
        linewidth=2.5,
        marker="o",
        markersize=4,
        drawstyle="steps-pre",
        label=f"{agg} across seeds",
    )

    # Customize the plot
    ax.set_xlabel(x_col)
    ax.set_ylabel(score_col)
    ax.set_title(f"Score ({agg}) of {optim} on {dataset} using {model}")

    # Adjust legend
    ax.legend(frameon=True, loc="best", fontsize=10, facecolor="white", edgecolor="gray")

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


def plot_population_scores_comparison(
    dataset,
    model,
    optims: List[str],
    agg="mean",
    plot_seeds=False,
    score_col="test_score",
    x_col="step",
    seed_linestyle="--",
    seed_marker="",
):
    fig, ax = plt.subplots()

    for optim in optims:
        df = get_results(dataset, model, optim)
        df = aggregate_results(df, how=agg, ffill_col=x_col)

        # Plot individual seeds if requested
        if plot_seeds:
            for seed in df["seed"].unique():
                df_seed = df[df["seed"] == seed]
                sns.lineplot(
                    data=df_seed,
                    x=x_col,
                    y=score_col,
                    alpha=0.5,
                    linestyle=seed_linestyle,
                    label=f"{optim} Seed {seed}",
                    ax=ax,
                )
                # Add markers at data points
                ax.plot(
                    df_seed[x_col],
                    df_seed[score_col],
                    marker=seed_marker,
                    linestyle="",
                    markersize=8,
                    alpha=0.7,
                    color=ax.lines[-1].get_color(),  # Match color with corresponding line
                )

        # Calculate and plot the mean across seeds
        mean_df = df.groupby(x_col, as_index=False)[score_col].mean()
        ax.plot(
            mean_df[x_col],
            mean_df[score_col],
            linewidth=2.5,
            marker="o",
            markersize=4,
            label=f"{agg} {optim}",
        )
