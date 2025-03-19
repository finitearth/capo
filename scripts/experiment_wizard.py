### This code tries to implement the demos/gsm8k.ipynb notebook in a script format.
"""
example call:
python scripts/experiment_wizard.py --experiment-name prompt_wizard --dataset agnews --max-model-len 4096 --random-seed 42 --optimizer promptwizard --model vllm-ConfidentialMind/Mistral-Small-24B-Instruct-2501_GPTQ_G128_W4A16_MSE --model-revision main --output-dir results/ --n-steps 999 --budget-per-run 1000
"""
from argparse import ArgumentParser

# has to come before imports, as we can only specify model via env variables
parser = ArgumentParser()
parser.add_argument("--experiment-name", required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--model-revision", type=str, default="main")
parser.add_argument("--output-dir", default="results/")
parser.add_argument("--max-model-len", type=int, required=True)
parser.add_argument("--random-seed", type=int, required=True)
parser.add_argument("--optimizer", required=True)

# ignored arguments
parser.add_argument("--n-steps", type=int, default=999)
parser.add_argument("--budget-per-run", type=int, required=True)
parser.add_argument("--population-size", type=int)
parser.add_argument("--n-eval-samples", type=int)
parser.add_argument("--evoprompt-ga-template", default="standard")
parser.add_argument("--block-size", type=int)
parser.add_argument("--length-penalty", type=float)
parser.add_argument("--crossovers-per-iter", type=int)
parser.add_argument("--upper-shots", type=int)
parser.add_argument("--max-n-blocks-eval", type=int)
parser.add_argument("--alpha", type=float)
parser.add_argument("--shuffle-blocks-per-iter", action="store_true", default=False)

args = parser.parse_args()

assert args.optimizer == "promptwizard"

import os

os.environ["MODEL"] = args.model
os.environ["MODEL_REVISION"] = args.model_revision
os.environ["MAX_MODEL_LEN"] = str(args.max_model_len)
os.environ["SEED"] = str(args.random_seed)

import random
import pandas as pd
import yaml

from capo.promptwizard.glue.promptopt.instantiate import GluePromptOpt
from capo.promptwizard.glue.promptopt.techniques.common_logic import DatasetSpecificProcessing

from capo.utils import generate_random_hash, seed_everything
from capo.load_datasets import get_tasks


class Processor(DatasetSpecificProcessing):
    def extract_final_answer(self, answer: str):
        return answer.split("</final_answer>")[0].split("<final_answer>")[-1].strip()
    def dataset_to_jsonl(self, *args, **kwargs):
        pass

if __name__ == "__main__":
    logging_dir = args.output_dir + args.experiment_name + "/" + generate_random_hash() + "/"
    seed_everything(args.random_seed)

    train_file_name = "temp/promptwizard/data.jsonl"
    # make dir if not exist
    os.makedirs(os.path.dirname(train_file_name), exist_ok=True)
    dev_task, _, _ = get_tasks(
        args.dataset, args.optimizer, block_size=args.block_size, seed=args.random_seed
    )
    pd.DataFrame({"question": dev_task.xs, "final_answer": dev_task.ys}).to_json(train_file_name)

    with open("promptwizard_config/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)        
    config["base_instruction"] = random.sample(dev_task.initial_prompts, 1)[0]
    config["task_description"] = dev_task.description
    
    with open("promptwizard_config/temp_config.yaml", "w") as f:
        yaml.dump(config, f)

    path_to_config = "promptwizard_config"
    promptopt_config_path = os.path.join(path_to_config, "temp_config.yaml")
    setup_config_path = os.path.join(path_to_config, "setup_config.yaml")

    gp = GluePromptOpt(promptopt_config_path, setup_config_path, train_file_name, Processor())

    best_prompt, expert_profile = gp.get_best_prompt(
        use_examples=True, run_without_train_examples=False, generate_synthetic_examples=False
    )

    print(f"Best prompt: {best_prompt} \nExpert profile: {expert_profile}")

    pd.DataFrame(
        {"step": [1], "prompt": [best_prompt], "system_prompt": [expert_profile]}
    ).to_parquet(logging_dir + "step_results.parquet")
