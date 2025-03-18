"""Experiment configurations for all experiments in the paper."""

from capo.configs.base_config import ExperimentConfig, ModelConfig, OptimizerConfig

llama = ModelConfig(
    model="vllm-shuyuej/Llama-3.3-70B-Instruct-GPTQ",
    alias="llama",
    max_model_len=1024,
    batch_size=None,
    model_storage_path="../models/",
    revision="3a7f7f7d46e362291821aaefb0a38b632f1190a8",
)

qwen = ModelConfig(
    model="vllm-Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
    alias="qwen",
    max_model_len=1024,
    batch_size=None,
    model_storage_path="../models/",
    revision="c83e67dfb2664f5039fd4cd99e206799e27dd800",
)

mistral = ModelConfig(
    model="vllm-ConfidentialMind/Mistral-Small-24B-Instruct-2501_GPTQ_G128_W4A16_MSE",
    alias="mistral",
    max_model_len=1024,
    batch_size=None,
    model_storage_path="../models/",
    revision="803393813b8fc4046fb663af2e3c56339a5b520b",
)

BENCHMARK_CONFIG = ExperimentConfig(
    name="benchmark_experiment",
    datasets=["sst-5", "agnews", "subj", "rte", "gsm8k"],
    models=[llama, qwen, mistral],
    optimizers=[
        OptimizerConfig(
            name="EvoPromptGA",
            optimizer="EvoPromptGA",
            optimizer_params={
                "n_steps": 999,
                "population_size": 10,
                "n_eval_samples": 300,
                "evoprompt_ga_template": "standard",
            },
        ),
        OptimizerConfig(
            name="CAPO",
            optimizer="CAPO",
            optimizer_params={
                "n_steps": 999,
                "population_size": 10,
                "block_size": 30,
                "length_penalty": 0.05,
                "crossovers_per_iter": 4,
                "upper_shots": 5,
                "max_n_blocks_eval": 10,
                "alpha": 0.2,
                "shuffle_blocks_per_iter": False,
            },
        ),
    ],
    random_seeds=[42, 43, 44],
    budget_per_run=10_000_000,
    output_dir="results/",
)


ABLATION_CONFIG = ExperimentConfig(
    name="ablation_experiment",
    datasets=["agnews", "rte"],
    models=[llama],
    optimizers=[
        OptimizerConfig(
            name="CAPO_zero_shot",
            optimizer="CAPO",
            optimizer_params={
                "n_steps": 999,
                "population_size": 10,
                "block_size": 30,
                "length_penalty": 0.05,
                "crossovers_per_iter": 4,
                "upper_shots": 0,  # TODO: to be checked
                "max_n_blocks_eval": 10,
                "alpha": 0.2,
                "shuffle_blocks_per_iter": False,
            },
        ),
        OptimizerConfig(
            name="CAPO_no_length_penalty",
            optimizer="CAPO",
            optimizer_params={
                "n_steps": 999,
                "population_size": 10,
                "block_size": 30,
                "length_penalty": 0.0,  # TODO: to be checked
                "crossovers_per_iter": 4,
                "upper_shots": 5,
                "max_n_blocks_eval": 10,
                "alpha": 0.2,
                "shuffle_blocks_per_iter": False,
            },
        ),
        OptimizerConfig(
            name="CAPO_no_racing",
            optimizer="CAPO",
            optimizer_params={
                "n_steps": 999,
                "population_size": 10,
                "block_size": 300,  # TODO: to be checked
                "length_penalty": 0.05,
                "crossovers_per_iter": 4,
                "upper_shots": 3,
                "max_n_blocks_eval": 1,  # TODO: to be checked
                "alpha": 0.2,
                "shuffle_blocks_per_iter": False,
            },
        ),
        OptimizerConfig(
            name="EvoPromptGA_simplified_with_TD",
            optimizer="EvoPromptGA",
            optimizer_params={
                "n_steps": 999,
                "population_size": 10,
                "n_eval_samples": 300,
                "evoprompt_ga_template": "simplified",
            },
        ),
    ],
    random_seeds=[42, 43, 44],
    budget_per_run=10_000_000,
    output_dir="results/",
)

length_penalty_grid = [0.01, 0.02, 0.05, 0.1]
population_size_grid = [6, 8, 10, 12]
ncrossovers_grid = [4, 7, 10]

HYPERPARAMETER_CONFIG = ExperimentConfig(
    name="hyperparameter_experiment",
    datasets=["agnews", "rte"],
    models=[llama],
    optimizers=[
        OptimizerConfig(
            name=f"CAPO_gamma_{length_penalty}",
            optimizer="CAPO",
            optimizer_params={
                "n_steps": 999,
                "population_size": 10,
                "block_size": 30,
                "length_penalty": length_penalty,
                "crossovers_per_iter": 4,
                "upper_shots": 3,
                "max_n_blocks_eval": 10,
                "alpha": 0.2,
                "shuffle_blocks_per_iter": False,
            },
        )
        for length_penalty in length_penalty_grid
    ]
    + [
        OptimizerConfig(
            name=f"CAPO_pop_{population_size}",
            optimizer="CAPO",
            optimizer_params={
                "n_steps": 999,
                "population_size": population_size,
                "block_size": 30,
                "length_penalty": 0.05,
                "crossovers_per_iter": 4,
                "upper_shots": 3,
                "max_n_blocks_eval": 10,
                "alpha": 0.2,
                "shuffle_blocks_per_iter": False,
            },
        )
        for population_size in population_size_grid
    ]
    + [
        OptimizerConfig(
            name=f"CAPO_ncrossovers_{ncrossovers}",
            optimizer="CAPO",
            optimizer_params={
                "n_steps": 999,
                "population_size": 10,
                "block_size": 30,
                "length_penalty": 0.05,
                "crossovers_per_iter": ncrossovers,
                "upper_shots": 3,
                "max_n_blocks_eval": 10,
                "alpha": 0.2,
                "shuffle_blocks_per_iter": False,
            },
        )
        for ncrossovers in ncrossovers_grid
    ],
    random_seeds=[42, 43, 44],
    budget_per_run=10_000_000,
    output_dir="results/",
)
