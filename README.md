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
