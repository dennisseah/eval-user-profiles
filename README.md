# eval-user-profiles

## Setup

1. Clone the repository
2. `cd eval-user-profiles` (root directory of this git repository)
3. `python -m venv .venv`
4. `poetry install` (install the dependencies)
5. code . (open the project in vscode)
6. install the recommended extensions (cmd + shift + p -> `Extensions: Show Recommended Extensions`)
7. `pre-commit install` (install the pre-commit hooks)
8. copy the `.env.example` file to `.env` and fill in the values

## Unit Test Coverage

```sh
python -m pytest -p no:warnings --cov-report term-missing --cov=eval_user_profiles tests
```

## Simple Experiment

```sh
python -m experiments.test
```