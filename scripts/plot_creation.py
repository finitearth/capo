"""
Functions for generating and saving plots from experimental results.
This allows for the regeneration of the graphics provided in the paper
"""

import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns

from capo.analysis.visualizations import (
    plot_length_score,
    plot_performance_profile_curve,
    plot_population_members,
    plot_population_scores_comparison,
)

OPTIMS = ["CAPO", "OPRO", "EvoPromptGA", "PromptWizard", "Initial"]
DATASETS = ["sst-5", "agnews", "copa", "gsm8k", "subj"]
MODELS = ["llama", "mistral", "qwen"]

parser = argparse.ArgumentParser()

# general parameters
parser.add_argument("--all", action="store_true")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--hp", action="store_true")
parser.add_argument("--ablation", action="store_true")

args = parser.parse_args()

assert any([args.all, args.benchmark, args.hp, args.ablation])


if __name__ == "__main__":
    os.makedirs("./results/plots", exist_ok=True)

    if args.benchmark or args.all:
        for model in MODELS:
            for dataset in DATASETS:
                fig = plot_population_scores_comparison(
                    dataset,
                    model,
                    OPTIMS,
                    agg="mean",
                    plot_seeds=False,
                    plot_stddev=True,
                    x_col="input_tokens_cum",
                    figsize=(5.4, 3),
                )
                fig.savefig(
                    f"./results/plots/{dataset}_{model}_population_scores.png", bbox_inches="tight"
                )
                plt.close()
        fig = plot_population_scores_comparison(
            "gsm8k",
            "mistral",
            optims=OPTIMS,
            agg="mean",
            plot_seeds=False,
            plot_stddev=True,
            x_col="input_tokens_cum",
        )
        fig.savefig("./results/plots/gsm8k_mistral_population_scores_main.png", bbox_inches="tight")
        plt.close()
        fig = plot_population_scores_comparison(
            "subj",
            "qwen",
            optims=OPTIMS,
            agg="mean",
            plot_seeds=False,
            plot_stddev=True,
            x_col="input_tokens_cum",
        )
        fig.savefig("./results/plots/subj_qwen_population_scores_main.png", bbox_inches="tight")
        plt.close()
        fig = plot_performance_profile_curve()
        fig.savefig("./results/plots/performance_profile_curve.png", bbox_inches="tight")
        plt.close()

        fig = plot_length_score(
            "gsm8k",
            "mistral",
            ["CAPO", "OPRO", "EvoPromptGA", "PromptWizard"],
            x_col="prompt_len",
            score_col="test_score",
            log_scale=False,
        )
        fig.savefig("./results/plots/gsm8k_mistral_prompt_length_score.png", bbox_inches="tight")
        plt.close()

        fig = plot_population_members(
            "sst-5",
            "mistral",
            "CAPO",
            x_col="step",
            score_col="test_score",
            seeds=[42],
            figsize=(5.4, 3),
        )
        fig.savefig("./results/plots/sst-5_mistral_42_population_members.png", bbox_inches="tight")
        plt.close()

        fig = plot_population_members(
            "subj",
            "qwen",
            "CAPO",
            x_col="step",
            score_col="test_score",
            seeds=[42],
            figsize=(5.4, 3),
        )
        fig.savefig("./results/plots/subj_qwen_42_population_members.png", bbox_inches="tight")
        plt.close()

    if args.hp or args.all:
        hp_runs = [
            "CAPO_no_lp",
            "CAPO_gamma_0.01",
            "CAPO_gamma_0.02",
            "CAPO_gamma_0.05",
            "dummy",
            "dummy",
            "CAPO_gamma_0.1",
        ]
        markers = ["8", "s", "d", "o", None, None, "p"]
        labels = [
            r"$\gamma=0$",
            r"$\gamma=0.01$",
            r"$\gamma=0.02$",
            r"$\gamma=0.05$ (CAPO)",
            "Dummy",
            "Dummy",
            r"$\gamma=0.1$",
        ]

        for dataset in ["agnews", "gsm8k"]:
            for score_col in ["test_score", "prompt_len"]:
                fig = plot_population_scores_comparison(
                    dataset,
                    "llama",
                    hp_runs,
                    "mean",
                    plot_seeds=False,
                    plot_stddev=True,
                    x_col="step",
                    score_col=score_col,
                    continuous_colors=True,
                    markers=markers,
                    labels=labels,
                    figsize=(5.4, 3),
                )
                fig.savefig(
                    f"./results/plots/hyperparameter_tuning_lp_{dataset}_{score_col}.png",
                    bbox_inches="tight",
                )
                plt.close()

        # Population size
        hp_runs = ["CAPO_pop_6", "CAPO_pop_8", "CAPO_pop_10", "Dummy", "CAPO_pop_12"]
        markers = ["8", "s", "o", None, "p"]
        labels = [r"$\mu=6$", r"$\mu=8$", r"$\mu=10$ (CAPO)", "Dummy", r"$\mu=12$"]

        for dataset in ["agnews", "gsm8k"]:
            fig = plot_population_scores_comparison(
                dataset,
                "llama",
                hp_runs,
                "mean",
                plot_seeds=False,
                plot_stddev=True,
                x_col="step",
                score_col="test_score",
                continuous_colors=True,
                markers=markers,
                labels=labels,
                ncols=2,
                figsize=(5.4, 3),
            )
            fig.savefig(
                f"./results/plots/hyperparameter_tuning_pop_{dataset}.png",
                bbox_inches="tight",
            )
            plt.close()

        # Number of crossovers
        hp_runs = [
            "Dummy",
            "Dummy",
            "CAPO_ncrossovers_4",
            "CAPO_ncrossovers_7",
            "CAPO_ncrossovers_10",
        ]
        markers = [None, None, "o", "p", "d"]
        labels = ["Dummy", "Dummy", r"$c=4$ (CAPO)", r"$c=7$", r"$c=10$"]

        for dataset in ["agnews", "gsm8k"]:
            fig = plot_population_scores_comparison(
                dataset,
                "llama",
                hp_runs,
                "mean",
                plot_seeds=False,
                plot_stddev=True,
                x_col="step",
                score_col="test_score",
                continuous_colors=True,
                markers=markers,
                labels=labels,
                figsize=(5.4, 3),
            )
            fig.savefig(
                f"./results/plots/hyperparameter_tuning_cross_{dataset}.png",
                bbox_inches="tight",
            )

            plt.close()

        #  Shuffling
        for dataset in ["agnews", "gsm8k"]:
            fig = plot_population_scores_comparison(
                dataset,
                "llama",
                ["CAPO", "nan", "CAPO_shuffling"],
                plot_stddev=True,
                plot_seeds=False,
                x_col="step",
                colors=["#1b9e77", "#7570b3", "#66D874"],
                markers=["o", "o", "d"],
                labels=["CAPO", "", "CAPO w/ shuffling"],
                ncols=2,
                figsize=(5.4, 3),
            )

            fig.savefig(
                f"./results/plots/hyperparameter_tuning_shuffling_{dataset}.png",
                bbox_inches="tight",
            )

            plt.close()

        # alpha
        hp_runs = [
            "CAPO_alpha_0.05",
            "CAPO_alpha_0.1",
            "CAPO_alpha_0.2",
            "CAPO_alpha_0.5",
        ]

        markers = ["8", "s", "o", "d"]
        labels = [
            r"$\alpha=0.05$",
            r"$\alpha=0.1$",
            r"$\alpha=0.2$ (CAPO)",
            r"$\alpha=0.5$ ",
        ]

        for dataset in ["agnews", "gsm8k"]:
            fig = plot_population_scores_comparison(
                dataset,
                "llama",
                hp_runs,
                "mean",
                plot_seeds=False,
                plot_stddev=True,
                x_col="step",
                score_col="test_score",
                continuous_colors=True,
                markers=markers,
                labels=labels,
                figsize=(5.4, 3),
            )
            fig.savefig(
                f"./results/plots/hyperparameter_tuning_alpha_{dataset}.png",
                bbox_inches="tight",
            )
            plt.close()

    if args.ablation or args.all:
        DATASETS = ["agnews", "gsm8k"]
        colors = ["#1b9e77", "#7570b3", "#66D874", "#9570b2"]
        markers = ["o", "o", "d", "d"]

        fig = plot_population_scores_comparison(
            "agnews",
            "llama",
            ["CAPO", "CAPO_no_racing", "EvoPromptGA"],
            colors=[sns.color_palette("Dark2")[0], "#66D874", sns.color_palette("Dark2")[2]],
            labels=["CAPO", "CAPO w/o Racing", "EvoPromptGA"],
            agg="mean",
            plot_seeds=False,
            plot_stddev=True,
            x_col="step",
            score_col="input_tokens_sum",
            figsize=(5.4, 2.8),
        )

        fig.savefig(
            "./results/plots/ablation_racing_tokens_per_step.png",
            bbox_inches="tight",
        )
        plt.close()

        markers = ["o", "d", "s", "p"]
        for dataset in DATASETS:
            fig = plot_population_scores_comparison(
                dataset,
                "llama",
                ["CAPO", "", "CAPO_no_racing"],
                labels=["CAPO", "", "CAPO w/o racing"],
                plot_stddev=True,
                plot_seeds=False,
                x_col="input_tokens_cum",
                colors=colors,
                markers=markers,
                ncols=2,
                figsize=(5.4, 2.8),
            )
            fig.savefig(
                f"./results/plots/ablation_racing_{dataset}.png",
                bbox_inches="tight",
            )
            plt.close()
            fig = plot_population_scores_comparison(
                dataset,
                "llama",
                ["CAPO", "EvoPromptGA", "CAPO_zero_shot"],
                labels=["CAPO", "EvoPromptGA", "CAPO zero shot"],
                plot_stddev=True,
                plot_seeds=False,
                x_col="input_tokens_cum",
                colors=colors,
                markers=markers,
                ncols=2,
                figsize=(5.4, 2.8),
            )

            fig.savefig(
                f"./results/plots/ablation_zero_shot_{dataset}.png",
                bbox_inches="tight",
            )
            plt.close()
            fig = plot_length_score(
                dataset,
                "llama",
                ["CAPO", "EvoPromptGA", "CAPO_zero_shot"],
                labels=["CAPO", "EvoPromptGA", "CAPO zero shot"],
                colors=colors,
                x_col="prompt_len",
                score_col="test_score",
                log_scale=False,
                ncols=2,
                figsize=(5.4, 2.8),
            )

            fig.savefig(
                f"./results/plots/ablation_zero_shot_{dataset}_len_obj.png",
                bbox_inches="tight",
            )
            plt.close()
            fig = plot_population_scores_comparison(
                dataset,
                "llama",
                ["nan", "EvoPromptGA", "nan", "EvoPromptGA_TD"],
                labels=["nan", "EvoPromptGA", "nan", "EvoPromptGA simplified"],
                plot_stddev=True,
                plot_seeds=False,
                x_col="input_tokens_cum",
                colors=colors,
                markers=markers,
                ncols=2,
                figsize=(5.4, 2.8),
            )

            fig.savefig(
                f"./results/plots/ablation_evo_td_{dataset}.png",
                bbox_inches="tight",
            )
            plt.close()
            fig = plot_population_scores_comparison(
                dataset,
                "llama",
                ["CAPO", "EvoPromptGA", "CAPO_generic_init", "EvoPromptGA_generic_init"],
                labels=[
                    "CAPO",
                    "EvoPromptGA",
                    "CAPO w/ generic init",
                    "EvoPromptGA generic init",
                ],
                plot_stddev=True,
                plot_seeds=False,
                x_col="input_tokens_cum",
                colors=colors,
                markers=markers,
                ncols=2,
                figsize=(5.4, 2.8),
            )

            fig.savefig(
                f"./results/plots/ablation_generic_init_{dataset}.png",
                bbox_inches="tight",
            )

            plt.close()

        fig = plot_population_scores_comparison(
            "agnews",
            "llama",
            ["CAPO", "EvoPromptGA", "CAPO_generic_init", "EvoPromptGA_generic_init"],
            labels=[
                "CAPO",
                "EvoPromptGA",
                "CAPO w/ generic init",
                "EvoPromptGA generic init",
            ],
            plot_stddev=True,
            plot_seeds=False,
            x_col="input_tokens_cum",
            colors=colors,
            markers=markers,
            ncols=2,
            figsize=(5.4, 3.6),
        )

        fig.savefig(
            "./results/plots/ablation_generic_init_agnews_main_pre.png",
            bbox_inches="tight",
        )

        plt.close()

        fig = plot_population_scores_comparison(
            "gsm8k",
            "llama",
            ["CAPO", "", "CAPO_no_racing_gamma_0_no_fs"],
            labels=[
                "CAPO",
                "",
                "CAPO w/o anything",
            ],
            plot_stddev=True,
            plot_seeds=False,
            x_col="input_tokens_cum",
            colors=colors,
            markers=markers,
            ncols=2,
            figsize=(5.4, 3.6),
        )

        fig.savefig(
            "./results/plots/ablation_capo_wo_anything_gsm8k.png",
            bbox_inches="tight",
        )

        plt.close()

        fig = plot_population_scores_comparison(
            "agnews",
            "llama",
            ["CAPO", "", "CAPO_no_racing_gamma_0_no_fs"],
            labels=[
                "CAPO",
                "",
                "CAPO w/o anything",
            ],
            plot_stddev=True,
            plot_seeds=False,
            x_col="input_tokens_cum",
            colors=colors,
            markers=markers,
            ncols=2,
            figsize=(5.4, 3.6),
        )
        fig.savefig(
            "./results/plots/ablation_capo_wo_anything_agnews.png",
            bbox_inches="tight",
        )
