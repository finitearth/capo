from capo.configs.base_config import ExperimentConfig, ModelConfig, OptimizerConfig

llama = ModelConfig(
    model="shuyuej/Llama-3.3-70B-Instruct-GPTQ",
    max_model_len=1024,
    batch_size=None,
    model_storage_path="../models/",
    revision="3a7f7f7d46e362291821aaefb0a38b632f1190a8",
)

qwen = ModelConfig(
    model="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
    max_model_len=1024,
    batch_size=None,
    model_storage_path="../models/",
    revision="c83e67dfb2664f5039fd4cd99e206799e27dd800",
)

mistral = ModelConfig(
    model="ConfidentialMind/Mistral-Small-24B-Instruct-2501_GPTQ_G128_W4A16_MSE",
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
                "upper_shots": 3,
                "max_n_blocks_eval": 10,
                "alpha": 0.2,
                "shuffle_blocks_per_iter": True,
            },
        ),
    ],
    random_seeds=[42, 43, 44],
    budget_per_run=10_000_000,
    output_dir="results/",
)
