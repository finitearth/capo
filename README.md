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

### Set-up pre-commit

```
poetry run pre-commit install
```

To run pre-commit:
```
poetry run pre-commit [--all-files]
```
