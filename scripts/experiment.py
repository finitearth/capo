"""Main script to run all experiments."""

import argparse
import random
from logging import getLogger

from promptolution.callbacks import CSVCallback, LoggerCallback, TokenCountCallback
from promptolution.llms import get_llm
from promptolution.optimizers import EvoPromptGA
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.predictors import FirstOccurrenceClassificator, MarkerBasedClassificator
from promptolution.templates import EVOPROMPT_GA_TEMPLATE

from capo.capo import CAPOptimizer
from capo.statistical_tests import paired_t_test
from capo.utils import generate_random_hash

parser = argparse.ArgumentParser()

# general parameters
parser.add_argument("--experiment-name", required=True)
parser.add_argument("--random-seed", type=int, required=True)
parser.add_argument("--budget-per-run", type=int, required=True)
parser.add_argument("--output-dir", default="results/")

# dataset parameters
parser.add_argument("--dataset", required=True)

# model parameters
parser.add_argument("--model", required=True)
parser.add_argument("--model-revision", required=True)
parser.add_argument("--max-model-len", type=int, required=True)
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--model-storage-path", default="../models/")

# optimizer parameters
parser.add_argument("--optimizer", required=True)
parser.add_argument("--n-steps", type=int, default=999)

# optimizer-specific parameters: all evolutionary optimizers
parser.add_argument("--population-size", type=int)

# optimizer-specific parameters: EvoPromptGA
parser.add_argument("--n-eval-samples", type=int)

# optimizer-specific parameters: CAPO
parser.add_argument("--block-size", type=int)
parser.add_argument("--length-penalty", type=float)
parser.add_argument("--crossovers-per-iter", type=int)
parser.add_argument("--upper-shots", type=int)
parser.add_argument("--max-n-blocks-eval", type=int)
parser.add_argument("--population-size", type=int)
parser.add_argument("--alpha", type=float)
parser.add_argument("--shuffle-blocks-per-iter", type=bool)

args = parser.parse_args()

logger = getLogger(__name__)

if __name__ == "__main__":
    # set-up callbacks
    callbacks = [
        LoggerCallback(logger),
        CSVCallback(args.output_dir + args.experiment_name + "/" + generate_random_hash() + "/"),
        TokenCountCallback(args.budget_per_run, "input_tokens"),
    ]

    # Set up LLM
    llm = get_llm(
        model_id=args.model,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        model_storage_path=args.model_storage_path,
        revision=args.revision,
    )

    downstream_llm = llm
    meta_llm = llm

    # set-up task (including task description and initial prompts)
    fewshot_task, dev_task = ...  # from args.dataset

    # set-up predictor
    if args.dataset in ...:
        predictor = FirstOccurrenceClassificator(downstream_llm, dev_task.classes)  # TODO
    elif args.dataset in ...:
        predictor = MarkerBasedClassificator(downstream_llm, dev_task.classes)  # TODO
    else:
        raise ValueError(f"Task {args.dataset} not supported.")

    # initialize population
    initial_prompts = random.sample(dev_task.initial_prompts, args.population_size)

    # initialize optimizer
    optimizer: BaseOptimizer
    if args.optimizer == "EvoPromptGA":
        optimizer = EvoPromptGA(
            task=dev_task,
            prompt_template=EVOPROMPT_GA_TEMPLATE,
            predictor=predictor,
            meta_llm=meta_llm,
            initial_prompts=initial_prompts,
            callbacks=callbacks,
            n_eval_samples=args.n_eval_samples,
        )
    elif args.optimizer == "CAPO":
        optimizer = CAPOptimizer(
            task=dev_task,
            predictor=predictor,
            meta_llm=meta_llm,
            initial_prompts=initial_prompts,
            callbacks=callbacks,
            block_size=args.block_size,
            length_penalty=args.length_penalty,
            crossovers_per_iter=args.crossovers_per_iter,
            upper_shots=args.upper_shots,
            max_n_blocks_eval=args.max_n_blocks_eval,
            p_few_shot_reasoning=0.5,
            n_trials_generation_reasoning=5,
            test_statistic=lambda x, y: paired_t_test(x, y, alpha=args.alpha),
            shuffle_blocks_per_iter=args.shuffle_blocks_per_iter,
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")

    # run optimization
    best_prompts = optimizer.optimize(n_steps=args.n_steps)

    # clear cache by deleting the used llm
    llm.__del__()
