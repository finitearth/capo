import os

from capo.analysis.visualizations import (
    plot_length_score,
    plot_performance_profile_curve,
    plot_population_scores_comparison,
)

OPTIMS = ["CAPO", "OPRO", "EvoPromptGA", "PromptWizard", "Initial"]
DATASETS = ["sst-5", "agnews", "copa", "gsm8k", "subj"]
MODELS = ["llama", "mistral", "qwen"]

if __name__ == "__main__":
    os.makedirs("./results/plots", exist_ok=True)

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
                path_prefix=".",
                figsize=(5.4, 3),
            )
            fig.savefig(
                f"./results/plots/{dataset}_{model}_population_scores.png", bbox_inches="tight"
            )

    fig = plot_population_scores_comparison(
        "gsm8k",
        "mistral",
        optims=OPTIMS,
        agg="mean",
        plot_seeds=False,
        plot_stddev=True,
        x_col="input_tokens_cum",
        path_prefix=".",
    )
    fig.savefig("./results/plots/gsm8k_mistral_population_scores_main.png", bbox_inches="tight")

    fig = plot_population_scores_comparison(
        "subj",
        "qwen",
        optims=OPTIMS,
        agg="mean",
        plot_seeds=False,
        plot_stddev=True,
        x_col="input_tokens_cum",
        path_prefix=".",
    )
    fig.savefig("./results/plots/subj_qwen_population_scores_main.png", bbox_inches="tight")

    fig = plot_performance_profile_curve(path_prefix=".")
    fig.savefig("./results/plots/performance_profile_curve.png", bbox_inches="tight")

    fig = plot_length_score(
        "gsm8k",
        "mistral",
        ["CAPO", "OPRO", "EvoPromptGA", "PromptWizard"],
        x_col="prompt_len",
        score_col="test_score",
        log_scale=False,
        path_prefix=".",
    )
    fig.savefig("./results/plots/gsm8k_mistral_prompt_length_score.png", bbox_inches="tight")

    # HYPERPARAMETER TUNING PLOTS
    # Length penalty
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
                plot_stddev=False,
                x_col="step",
                path_prefix=".",
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

    # Population size
    hp_runs = ["CAPO_pop_6", "CAPO_pop_8", "CAPO_pop_10", None, "CAPO_pop_12"]
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
            path_prefix=".",
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

    # Number of crossovers
    hp_runs = ["Dummy", "Dummy", "CAPO_ncrossovers_4", "CAPO_ncrossovers_7", "CAPO_ncrossovers_10"]
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
            path_prefix=".",
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
