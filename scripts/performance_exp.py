"""Script for running CAPO optimization with multiple models and datasets."""
import argparse
from logging import getLogger

from datasets import load_dataset
from promptolution.callbacks import CSVCallback, LoggerCallback
from promptolution.llms import get_llm
from promptolution.predictors.classificator import MarkerBasedClassificator
from promptolution.utils.prompt_creation import create_prompts_from_samples

from capo.capo import CAPOptimizer
from capo.statistical_tests import paired_t_test
from capo.task import CAPOClassificationTask

parser = argparse.ArgumentParser()
parser.add_argument("--models", required=True)
parser.add_argument("--dataset", default="SetFit/sst5")
parser.add_argument("--output-dir", default="results/")
parser.add_argument("--token", default=None)
parser.add_argument("--batch-size", type=int, default=None)
parser.add_argument("--revision", default="main")
parser.add_argument("--max-model-len", type=int, default=2000)
parser.add_argument("--model-storage-path", default="../models/")
parser.add_argument("--block-size", type=int, default=30)
parser.add_argument("--fs-split", type=float, default=0.1)
parser.add_argument("--n-steps", type=int, default=10)
parser.add_argument("--n-initial-prompts", type=int, default=10)
parser.add_argument("--crossovers-per-iter", type=int, default=4)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--length-penalty", type=float, default=0.05)
parser.add_argument("--upper-shots", type=int, default=3)
parser.add_argument("--max-n-blocks-eval", type=int, default=10)
parser.add_argument("--p-few-shot-reasoning", type=float, default=0.5)
args = parser.parse_args()

# Load dataset
ds = load_dataset(
    # "openai/gsm8k", "main", revision="e53f048856ff4f594e959d75785d2c2d37b678ee", split="train"
    "SetFit/rte",
    split="train",
)
df = ds.to_pandas()
df["target"] = df["label"].map({0: "NoEntailment", 1: "Entailment"})

# df["target"] = df["answer"].str.split("#### ").apply(lambda x: x[-1]).str.strip()

# Run optimization for each model
for model_name in args.models.strip("[]").split(","):
    print(f"\n\n{'=' * 40}\nRunning optimization for model: {model_name}\n{'=' * 40}\n")

    # Set up CAPO task
    task = CAPOClassificationTask.from_dataframe(
        df,
        description="",
        # The dataset contains linguistically diverse grade school math word problems that require multi-step reasoning. The answer is the final number and will be extracted after the <answer> tag.
        x_column="question",
        y_column="target",
    )
    task = CAPOClassificationTask.from_task(
        task, block_size=args.block_size, few_shot_split_size=args.fs_split
    )

    # Set up LLM
    llm = get_llm(
        model_name,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        model_storage_path=args.model_storage_path,
        revision=args.revision,
    )

    downstream_llm = llm
    meta_llm = llm

    # Set up predictor and callbacks
    predictor = MarkerBasedClassificator(downstream_llm, task.classes)
    test_statistic = lambda x, y: paired_t_test(x, y, alpha=args.alpha)
    logger = getLogger(__name__)
    callbacks = [LoggerCallback(logger), CSVCallback(args.output_dir + model_name + "/")]

    meta_prompt = """You are asked to give the corresponding prompt that gives the following outputs given these inputs for the following task:
The dataset contains linguistically diverse grade school math word problems that require multi-step reasoning. The answer is the final number and will be extracted after the <answer> tag.
Return it starting with <prompt> and ending with </prompt> tags.

<input_output_pairs>

The instruction was"""
    # TODO: should use samples only from the few-shot dataset here
    initial_prompts = create_prompts_from_samples(
        task,
        downstream_llm,
        meta_prompt=meta_prompt,
        task_description="",
        n_prompts=args.n_initial_prompts,
        n_samples=3,
    )

    # initial_prompts = [
    #     "Given the following inputs, determine the final number by performing the necessary calculations.",
    #     "Given the following inputs, determine the final number based on the multi-step reasoning required for each problem.",
    #     "Given the following input, determine the final number after performing the necessary calculations. The answer should be extracted after the <answer> tag.",
    #     "Given the following inputs, determine the final number by extracting it after the <answer> tag.",
    #     "Given the following math word problem, determine the final number.",
    #     "Given the following inputs, provide the corresponding prompt that gives the outputs.",
    #     "Given the following inputs, determine the final number by solving the problem step by step. The answer will be extracted after the <answer> tag.",
    #     "Given the following math word problems, determine the final number by following the steps outlined in the problem. The answer should be extracted after the <answer> tag.",
    #     "Given the following input, determine the final number after performing the multi-step reasoning. The answer should be extracted after the <answer> tag.",
    #     "Given the following inputs, provide the corresponding prompt that gives the outputs for the task. The dataset contains linguistically diverse grade school math word problems that require multi-step reasoning. The answer is the final number and will be extracted after the <answer> tag.",
    # ]

    # Initialize optimizer
    optimizer = CAPOptimizer(
        initial_prompts=initial_prompts,
        task=task,
        meta_llm=meta_llm,
        downstream_llm=downstream_llm,
        length_penalty=args.length_penalty,
        block_size=args.block_size,
        crossovers_per_iter=args.crossovers_per_iter,
        upper_shots=args.upper_shots,
        max_n_blocks_eval=args.max_n_blocks_eval,
        p_few_shot_reasoning=args.p_few_shot_reasoning,
        few_shot_split_size=args.fs_split,
        test_statistic=test_statistic,
        predictor=predictor,
        callbacks=callbacks,
        shuffle_blocks_per_iter=False,
        n_trials_generation_reasoning=5,
        verbosity=1,
    )

    # Run optimization
    best_prompts = optimizer.optimize(n_steps=args.n_steps)

print("\nAll optimization runs completed successfully!")
