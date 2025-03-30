# CAPO - Cost-Aware Prompt Optimization

## Installation

To run experiments or contribute to this project, it is recommended to follow these steps:

### Set-up poetry and install dependencies

1.) Clone or fork this repository

2.) Install pipx:
```
pip install --user pipx
```

3.) Install Poetry:
```
pipx install poetry
```

*3a.) Optionally (if you want the env-folder to be created in your project)*:
```
poetry config virtualenvs.in-project true
```

4.) Install this project:
```
poetry install
```

*4a.) If the specified Python version (3.11 for this project) is not found:*

If the required Python version is not installed on your device, install Python from the official [python.org](https://www.python.org/downloads) website.

Then run
```
poetry env use <path-to-your-python-version>
```
and install the project:
```
poetry install
```

## Run Experiments

All experiments can be executed via the `scripts/experiment.py` script (`scripts/experiment_wizard.py` for PromptWizard).

Experiments can be parametrized directly per command line arguments.

Example experiment (CAPO with Qwen2.5-32B on Subj, 10 steps / 1M input token budget with default parametrization):
```
poetry run python scripts/experiment.py --experiment-name test --random-seed 42 \
    --budget-per-run 1000000 --output-dir results/ --dataset subj \
    --model vllm-Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4\
    --model-revision c83e67dfb2664f5039fd4cd99e206799e27dd800 \
    --max-model-len 2048 --optimizer CAPO --n-steps 10 --population-size 10 \
    --max-n-blocks-eval 10 --block-size 30 --length-penalty 0.05 \
    --crossovers-per-iter 4 --upper-shots 5 --alpha 0.2
```

## Repository Structure

Overview of relevant files and directories:

```python
├── capo/  # package with all functions and classes, including the CAPO algorithm, and configs and utilities for the experiments and analysis
│   ├── analysis/  # function for analysis in visualization
│   │   ├── style.py  # styling of plots
│   │   ├── utils.py  # utilities for plotting: preparation and aggregation of the raw results
│   │   └── visualizations.py  # plot functions for all visualizations in the paper
│   ├── configs/  # experiment configuration
│   │   ├── promptwizard_config/  # configuration file for PromptWizard
│   │   ├── base_config.py  # base dataclasses for configs
│   │   ├── config_to_command.py  # functions to split a large experiment config into separate configs and generate commands to execute the experiments
│   │   ├── dataset_config.py  # dataset configurations, defining huggingface ID, revision, columns, splits, etc.
│   │   ├── experiment_configs.py  # configs of all our experiments in the paper, including benchmark experiments, ablation studies, and hyperparam analysis
│   │   ├── initial_prompts.py  # initial instructions for each dataset
│   │   └── task_descriptions.py  # task descriptions for each dataset
│   ├── callbacks.py  # modified callbacks to track experiment results
│   ├── capo.py  # implementation of the CAPO algorithm
│   ├── evopromptga.py  # wrapper for EvoPromptGA
│   ├── load_datasets.py  # function to load a dataset based on a config
│   ├── opro.py  # wrapper for OPRO
│   ├── prompt.py  # prompt class
│   ├── statistical_tests.py  # implementation of statistical tests for racing
│   ├── task.py  # task class specifically for CAPO
│   ├── templates.py  # prompt templates
│   └── utils.py  # utility files for seeding, hashing, copying, etc.
├── notebooks/  # notebooks to display and analyze all plots and tables
├── results/  # raw results obtained from the experiments and evaluations
└── scripts/  # scripts to execute experiments and evaluation
    ├── evaluate_initial_prompts.py  # script to evaluate initial prompts (on unseen test data)
    ├── evaluate_prompts.py  # script to evaluate prompts resulting from the experiments (on unseen test data)
    ├── experiment_wizard.py  # script to run the experiments with PromptWizard
    ├── experiment.py  # script to run all experiments (benchmark, ablation, hyperparam) for CAPO, OPRO, and EvoPrompt
    ├── plot_creation.py  # script to create the graphics provided in the paper
    └── job_creation.py  # script to automatically generate commands to execute jobs (note: based on our infrastructure)

```

## Dependencies

Exact dependencies are documented in the poetry.lock. Executing `poetry install`` will produce an environment with exactly the dependencies used for our experiments.
