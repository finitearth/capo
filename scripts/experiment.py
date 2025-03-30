"""
Main script for running all benchmark experiments, ablation studies, and hyperparameter analyses.
Coordinates evaluation of CAPO, OPRO, and EvoPromptGA across multiple datasets and configurations.

for minimal example run:
python scripts/experiment.py --experiment-name test --random-seed 42 \
    --budget-per-run 1000000 --output-dir results/ --dataset subj \
    --model vllm-Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4\
    --model-revision c83e67dfb2664f5039fd4cd99e206799e27dd800 \
    --max-model-len 2048 --optimizer CAPO --n-steps 10 --population-size 10 \
    --max-n-blocks-eval 10 --block-size 30 --length-penalty 0.05 \
    --crossovers-per-iter 4 --upper-shots 5 --alpha 0.2
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

from capo.callbacks import ParquetCallback, PickleCallback, PromptScoreCallback
from capo.capo import CAPOptimizer
from capo.configs.initial_prompts import INITIAL_PROMPTS
from capo.evopromptga import EvoPromptGAPickable
from capo.load_datasets import get_tasks
from capo.opro import OproPickable
from capo.statistical_tests import paired_t_test
from capo.templates import EVOPROMPT_GA_SIMPLIFIED_TEMPLATE
from capo.utils import copy_llm, generate_random_hash, seed_everything

parser = argparse.ArgumentParser()

# general parameters
parser.add_argument("--experiment-name", required=True)
parser.add_argument("--random-seed", type=int, required=True)
parser.add_argument("--budget-per-run", type=int, required=True)
parser.add_argument("--output-dir", default="results/main_results/")
parser.add_argument("--generic-init-prompts", action="store_true")

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

# optimizer-specific parameters: OPRO
parser.add_argument("--max-num-instructions", type=int, default=20)
parser.add_argument("--max-instructions-per-step", type=int, default=8)
parser.add_argument("--num_few_shots", type=int, default=3)


args = parser.parse_args()

logger = getLogger(__name__)

if __name__ == "__main__":
    # set-up callbacks
    logging_dir = args.output_dir + args.experiment_name + "/" + generate_random_hash() + "/"
    os.makedirs(logging_dir, exist_ok=True)

    seed_everything(args.random_seed)

    # write arguments as json in output directory
    with open(logging_dir + "/args.json", "w") as f:
        json.dump(vars(args), f)

    callbacks = [
        LoggerCallback(logger),
        ParquetCallback(logging_dir),
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
    dev_task, df_fewshots, _ = get_tasks(
        args.dataset, args.optimizer, seed=args.random_seed, block_size=args.block_size
    )

    # set-up predictor
    predictor = MarkerBasedClassificator(downstream_llm, dev_task.classes)

    # initialize population
    initial_prompts_pool = (
        dev_task.initial_prompts if not args.generic_init_prompts else INITIAL_PROMPTS["generic"]
    )
    initial_prompts = random.sample(initial_prompts_pool, args.population_size)

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
    elif args.optimizer == "OPRO":
        optimizer = OproPickable(
            meta_llm=meta_llm,
            initial_prompts=initial_prompts,
            callbacks=callbacks,
            predictor=predictor,
            task=dev_task,
            n_steps=args.n_steps,
            max_num_instructions=args.max_num_instructions,
            max_instructions_per_step=args.max_instructions_per_step,
            num_few_shots=args.num_few_shots,
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")

    # run optimization
    best_prompts = optimizer.optimize(n_steps=args.n_steps)

    # clear cache by deleting the used llm
    llm.__del__()
