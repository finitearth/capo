import argparse
import os

from capo.analysis.visualizations import plot_population_scores_comparison

OPTIMS = ["CAPO", "OPRO", "EvoPromptGA", "PromptWizard", "Initial"]
DATASETS = ["sst-5", "agnews", "copa", "gsm8k", "subj"]
MODELS = ["llama", "mistral", "qwen"]

parser = argparse.ArgumentParser()
parser.add_argument("--population-score-comp", action="store_true")

if __name__ == "__main__":
    os.makedirs("./results/plots", exist_ok=True)

    if parser.parse_args().population_score_comp:
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
                )
                fig.savefig(
                    f"./results/plots/{dataset}_{model}_population_scores.png", bbox_inches="tight"
                )
