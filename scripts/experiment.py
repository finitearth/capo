"""Main script to run all experiments.

for minimal example run:
python scripts/experiments.py --experiment-name test --random-seed 42 \
    --budget-per-run 1000000 --output-dir results/ --dataset fewrel \
    --model vllm-Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4\
    --model-revision c83e67dfb2664f5039fd4cd99e206799e27dd800 \
    --max-model-len 1024 --optimizer CAPO --n-steps 10 --population-size 10 \
    --n-eval-samples 10 --block-size 30 --length-penalty 0.01 \
    --crossovers-per-iter 4 --upper-shots 10 --max-n-blocks-eval 10 --alpha 0.2
"""

import argparse
import json
import os
import random
from logging import getLogger

from promptolution.callbacks import LoggerCallback, TokenCountCallback
from promptolution.llms import get_llm
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.predictors import MarkerBasedClassificator
from promptolution.templates import EVOPROMPT_GA_TEMPLATE

from capo.callbacks import CSVCallback, PickleCallback, PromptScoreCallback
from capo.capo import CAPOptimizer
from capo.evopromptga import EvoPromptGAPickable
from capo.load_datasets import get_tasks
from capo.statistical_tests import paired_t_test
from capo.templates import EVOPROMPT_GA_SIMPLIFIED_TEMPLATE
from capo.utils import copy_llm, generate_random_hash, seed_everything

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
parser.add_argument("--batch-size", type=int, default=None)
parser.add_argument("--model-storage-path", default="../models/")

# optimizer parameters
parser.add_argument("--optimizer", required=True)
parser.add_argument("--n-steps", type=int, default=999)

# optimizer-specific parameters: all evolutionary optimizers
parser.add_argument("--population-size", type=int)

# optimizer-specific parameters: EvoPromptGA
parser.add_argument("--n-eval-samples", type=int)
parser.add_argument("--evoprompt-ga-template", default="standard")

# optimizer-specific parameters: CAPO
parser.add_argument("--block-size", type=int)
parser.add_argument("--length-penalty", type=float)
parser.add_argument("--crossovers-per-iter", type=int)
parser.add_argument("--upper-shots", type=int)
parser.add_argument("--max-n-blocks-eval", type=int)
parser.add_argument("--alpha", type=float)
parser.add_argument("--shuffle-blocks-per-iter", action="store_true", default=False)

args = parser.parse_args()

logger = getLogger(__name__)

if __name__ == "__main__":
    seed_everything(args.random_seed)

    # set-up callbacks
    logging_dir = args.output_dir + args.experiment_name + "/" + generate_random_hash() + "/"
    os.makedirs(logging_dir, exist_ok=True)

    # write arguments as json in output directory
    with open(logging_dir + "/args.json", "w") as f:
        json.dump(vars(args), f)

    callbacks = [
        LoggerCallback(logger),
        CSVCallback(logging_dir),
        TokenCountCallback(args.budget_per_run, "input_tokens"),
        PickleCallback(logging_dir),
        PromptScoreCallback(logging_dir),
    ]

    # Set up LLM
    llm = get_llm(
        model_id=args.model,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        model_storage_path=args.model_storage_path,
        revision=args.model_revision,
        seed=args.random_seed,
    )

    downstream_llm = llm
    meta_llm = copy_llm(llm)

    # set-up task (including task description and initial prompts)
    dev_task, df_fewshots, test_task = get_tasks(
        args.dataset, args.optimizer, seed=args.random_seed, block_size=args.block_size
    )

    # set-up predictor
    predictor = MarkerBasedClassificator(downstream_llm, dev_task.classes)

    print(dev_task.initial_prompts)

    # initialize population
    initial_prompts = random.sample(dev_task.initial_prompts, args.population_size)

    # set-up EvoPromptGA template
    if args.evoprompt_ga_template == "standard":
        evoprompt_template = EVOPROMPT_GA_TEMPLATE
    elif args.evoprompt_ga_template == "simplified":
        evoprompt_template = EVOPROMPT_GA_SIMPLIFIED_TEMPLATE.replace(
            "<task_desc>", dev_task.description
        )
    else:
        raise ValueError(f"Template {args.evoprompt_ga_template} not supported.")

    # initialize optimizer
    optimizer: BaseOptimizer
    if args.optimizer == "EvoPromptGA":
        optimizer = EvoPromptGAPickable(
            task=dev_task,
            prompt_template=evoprompt_template,
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
            df_few_shots=df_fewshots,
            downstream_llm=downstream_llm,
            meta_llm=meta_llm,
            initial_prompts=initial_prompts,
            callbacks=callbacks,
            length_penalty=args.length_penalty,
            crossovers_per_iter=args.crossovers_per_iter,
            upper_shots=args.upper_shots,
            max_n_blocks_eval=args.max_n_blocks_eval,
            test_statistic=lambda x, y: paired_t_test(x, y, alpha=args.alpha),
            shuffle_blocks_per_iter=args.shuffle_blocks_per_iter,
            verbosity=1,
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")

    # run optimization
    best_prompts = optimizer.optimize(n_steps=args.n_steps)

    # clear cache by deleting the used llm
    llm.__del__()
