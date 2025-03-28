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
