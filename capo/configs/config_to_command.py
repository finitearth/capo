"""Convert an experiment configuration to command line strings."""

import itertools
from typing import List

from capo.configs.base_config import ExperimentConfig


def generate_individual_configs(config: ExperimentConfig) -> List[ExperimentConfig]:
    """Generate individual experiment configs from the cross product of parameters."""
    individual_configs = []

    # Get all combinations of datasets, models, optimizers, and seeds
    combinations = itertools.product(
        config.datasets, config.models, config.optimizers, config.random_seeds
    )

    for dataset, model, optimizer, seed in combinations:
        # Create the individual experiment config
        individual_config = ExperimentConfig(
            name=f"{config.name}_{dataset}_{model.alias}_{optimizer.name}_{seed}",
            datasets=[dataset],
            models=[model],
            optimizers=[optimizer],
            random_seeds=[seed],
            budget_per_run=config.budget_per_run,
            output_dir=f"{config.output_dir}/{dataset}/{model.alias}/{optimizer.name}/seed{seed}/",
        )

        individual_configs.append(individual_config)

    return individual_configs


def generate_command(
    config: ExperimentConfig,
    evaluate: bool = False,
    partition: str = None,
    ntasks: int = 1,
    gres: str = None,
    time: str = None,
    qos: str = "mcml",
) -> str:
    command = "sbatch"

    if partition is not None:
        command += f" --partition={partition}"
    if ntasks is not None:
        command += f" --ntasks={ntasks}"
    if gres is not None:
        command += f" --gres={gres}"
    if time is not None:
        command += f" --time={time}"
    if qos is not None:
        command += f" --qos={qos}"
    command += f" --job-name={config.name}"
    command += " --output=logs/%x-%j.out"
    command += " --error=logs/%x-%j.err"

    # Add all the parameters
    def add_param_if_exists(command, param_name, param_value):
        if param_value is not None:
            return command + f" --{param_name} {param_value}"
        return command

    if evaluate:
        command += f' --wrap "poetry run python scripts/evaluate_prompts.py --experiment-path {config.output_dir}"'
    else:
        if config.optimizers[0].name == "PromptWizard":
            command += ' --wrap "poetry run python scripts/experiment_wizard.py'
        else:
            command += ' --wrap "poetry run python scripts/experiment.py'
        command = add_param_if_exists(command, "optimizer", config.optimizers[0].optimizer)
        command = add_param_if_exists(
            command, "n-steps", config.optimizers[0].optimizer_params["n_steps"]
        )

        command = add_param_if_exists(
            command, "population-size", config.optimizers[0].optimizer_params.get("population_size")
        )

        command = add_param_if_exists(
            command, "n-eval-samples", config.optimizers[0].optimizer_params.get("n_eval_samples")
        )
        command = add_param_if_exists(
            command,
            "evoprompt-ga-template",
            config.optimizers[0].optimizer_params.get("evoprompt_ga_template"),
        )

        command = add_param_if_exists(
            command, "block-size", config.optimizers[0].optimizer_params.get("block_size")
        )
        command = add_param_if_exists(
            command, "length-penalty", config.optimizers[0].optimizer_params.get("length_penalty")
        )
        command = add_param_if_exists(
            command,
            "crossovers-per-iter",
            config.optimizers[0].optimizer_params.get("crossovers_per_iter"),
        )
        command = add_param_if_exists(
            command, "upper-shots", config.optimizers[0].optimizer_params.get("upper_shots")
        )
        command = add_param_if_exists(
            command,
            "max-n-blocks-eval",
            config.optimizers[0].optimizer_params.get("max_n_blocks_eval"),
        )
        command = add_param_if_exists(
            command, "alpha", config.optimizers[0].optimizer_params.get("alpha")
        )

        command = add_param_if_exists(command, "experiment-name", config.name)
        command = add_param_if_exists(command, "random-seed", config.random_seeds[0])
        command = add_param_if_exists(command, "budget-per-run", config.budget_per_run)
        command = add_param_if_exists(command, "output-dir", config.output_dir)

        command = add_param_if_exists(command, "dataset", config.datasets[0])

        command = add_param_if_exists(command, "model", config.models[0].model)
        command = add_param_if_exists(command, "model-revision", config.models[0].revision)
        command = add_param_if_exists(command, "max-model-len", config.models[0].max_model_len)
        command = add_param_if_exists(command, "batch-size", config.models[0].batch_size)
        command = add_param_if_exists(
            command, "model-storage-path", config.models[0].model_storage_path
        )

        if config.optimizers[0].optimizer_params.get("shuffle_blocks_per_iter"):
            command += " --shuffle-blocks-per-iter"

    command += '"'

    return command
