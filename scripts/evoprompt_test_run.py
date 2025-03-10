"""Script for running CAPO optimization with multiple models and datasets."""
import argparse
from logging import getLogger

from datasets import load_dataset
from promptolution.callbacks import CSVCallback, LoggerCallback, TokenCountCallback
from promptolution.llms import get_llm

# from promptolution.optimizers import EvoPromptGA
from promptolution.predictors.classificator import FirstOccurrenceClassificator
from promptolution.tasks import ClassificationTask

# from promptolution.templates import EVOPROMPT_GA_TEMPLATE_TD
from promptolution.utils.prompt_creation import create_prompts_from_samples

parser = argparse.ArgumentParser()
parser.add_argument("--models", required=True)
parser.add_argument("--dataset", default="SetFit/rte")
parser.add_argument("--output-dir", default="results/")
parser.add_argument("--token", default=None)
parser.add_argument("--batch-size", type=int, default=None)
parser.add_argument("--revision", default="main")
parser.add_argument("--max-model-len", type=int, default=2000)
parser.add_argument("--model-storage-path", default="../models/")
parser.add_argument("--block-size", type=int, default=30)
parser.add_argument("--fs-split", type=float, default=0.1)
parser.add_argument("--n-steps", type=int, default=999)
parser.add_argument("--n-initial-prompts", type=int, default=10)
parser.add_argument("--n-eval-samples", type=int, default=300)
parser.add_argument("--max-tokens", type=int, default=1000)
args = parser.parse_args()

logger = getLogger(__name__)

# Load dataset
ds = load_dataset(
    "SetFit/rte",
    split="train",
)
df = ds.to_pandas()
df["input"] = df["text1"] + "\n" + df["text2"]
df["target"] = df["label"].map({1: "NoEntailment", 0: "Entailment"})

task_description = """Given is a Textual Entailment Recognition (RTE) task of determining a inferential relationship between natural language hypothesis and premise. The task is to classify each text into the correct category "NoEntailment" or "Entailment". The class mentioned first in the response of the LLM will be the prediction."""
# TODO: no task description provided after promptolution update
meta_prompt = f"""You are asked to give the corresponding prompt that gives the following outputs given these inputs for the following task:
{task_description}
Return it starting with <prompt> and ending with </prompt> tags.
Include the name of the output classes in the prompt.

<input_output_pairs>

The instruction was"""

# Run optimization for each model
for model_name in args.models.strip("[]").split(","):
    print(f"\n\n{'=' * 40}\nRunning optimization for model: {model_name}\n{'=' * 40}\n")
    callbacks = [
        LoggerCallback(logger),
        CSVCallback(args.output_dir + model_name + "/"),
        TokenCountCallback(args.max_tokens, "input_tokens"),
    ]

    # Set up LLM
    llm = get_llm(
        model_name,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        model_storage_path=args.model_storage_path,
        revision=args.revision,
        token=args.token,
    )

    downstream_llm = llm
    meta_llm = llm

    task = ClassificationTask.from_dataframe(
        df,
        description=task_description,
        x_column="input",
        y_column="target",
    )

    # TODO: should use samples only from the few-shot dataset here (adapt this in CAPO Task)
    initial_prompts = create_prompts_from_samples(
        task,
        downstream_llm,
        meta_prompt=meta_prompt,
        task_description="",
        n_prompts=args.n_initial_prompts,
        n_samples=1,
    )

    logger.warning(initial_prompts)

    # Set up predictor and callbacks
    predictor = FirstOccurrenceClassificator(downstream_llm, task.classes)

    score, seq = task.evaluate(
        initial_prompts, predictor, n_samples=args.n_eval_samples, return_seq=True
    )
    logger.warning(f"Initial score: {score}")
    logger.warning(f"Initial sequences: {seq}")

    print(f"Initial score: {score}")
    print(f"Initial sequences: {seq}")

    # # Initialize optimizer
    # optimizer = EvoPromptGA(
    #     task=task,
    #     prompt_template=EVOPROMPT_GA_TEMPLATE_TD.replace("<task_desc>", task_description),
    #     predictor=predictor,
    #     meta_llm=meta_llm,
    #     initial_prompts=initial_prompts,
    #     callbacks=callbacks,
    #     n_eval_samples=args.n_eval_samples,
    # )

    # # Run optimization
    # best_prompts = optimizer.optimize(n_steps=args.n_steps)

    # # clear cache by deleting the used llm
    # llm.__del__()

print("\nAll optimization runs completed successfully!")
