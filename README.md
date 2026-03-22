# InstaOptima

InstaOptima is an instruction optimization framework for text classification and Aspect-Based Sentiment Analysis (ABSA).
It combines evolutionary search with multi-objective selection to discover high-quality instructions.

## Highlights

- Uses OpenAI models to generate offspring instructions through mutation and crossover operators.
- Uses `google/flan-t5-base` as the task model to evaluate each candidate instruction.
- Optimizes multiple objectives simultaneously (performance objective, instruction length, and perplexity).
- Applies NSGA-II style survivor selection on the loop `P -> Q -> P U Q -> next P`.
- Stores full per-run artifacts, including Pareto front and final population.

## Repository Structure

- `main.py`: CLI entrypoint.
- `instaoptima/config.py`: experiment configuration schema and YAML loading.
- `instaoptima/runner.py`: end-to-end experiment lifecycle and artifact writing.
- `instaoptima/data_loader.py`: Hugging Face and local dataset loaders.
- `instaoptima/flan_t5_evaluator.py`: task-model training/inference for instruction evaluation.
- `instaoptima/evaluator.py`: metrics, objective computation, and evaluation cache.
- `instaoptima/operators.py`: instruction mutation and crossover operators.
- `instaoptima/pareto.py`: non-dominated sorting and crowding-distance based selection.
- `instaoptima/perplexity.py`: perplexity objective scorer.
- `instaoptima/population.py`: initial population construction.

## Requirements

- Python 3.10+
- OpenAI API key
- Enough compute for repeated Flan-T5 fine-tuning (GPU strongly recommended)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

## Quick Start

Run with the default configuration:

```bash
python3 main.py --config config.yaml
```

Run ABSA debug preset (small and fast for sanity checks):

```bash
python3 main.py --config config_absa_debug.yaml
```

## Command-Line Interface

```bash
python3 main.py --config <path-to-yaml> [--runs-per-launch N] [--artifact-root <dir>]
```

- `--config`: YAML configuration file (default: `config.yaml`).
- `--runs-per-launch`: number of runs to execute in the current launch.
- `--artifact-root`: existing artifact directory to continue appending runs.

This makes it easy to split a long multi-run experiment into several launches.

Example:

```bash
python3 main.py --config config_absa_full.yaml --runs-per-launch 1
python3 main.py --config config_absa_full.yaml \
  --artifact-root artifacts/laptop14_absa_YYYYMMDD_HHMMSS \
  --runs-per-launch 2
```

## Experiment Flow

For each run:

1. Initialize a population of size `M`.
2. Evaluate all initial candidates with Flan-T5.
3. For each generation, produce `M` offspring via mutation/crossover.
4. Evaluate offspring and merge populations (`P U Q`).
5. Select next generation using non-dominated sorting + crowding distance.
6. Save final population and Pareto front.

## Configuration Guide

Key configuration groups in YAML:

- Search budget: `population_size`, `generations`, `num_runs`
- Operator behavior: `model`, `operator_model`, `temperature`, `operator_temperature`, `max_generation_tokens`
- Data: `dataset_source`, split names, sample sizes, local file paths
- Task model: `task_model_*` (epochs, LR, batch sizes, lengths, device)
- Objectives: `minimization_objectives`, `performance_metric_names`, `perplexity_model_name`

Available presets in this repository:

- `config.yaml`: general default config.
- `config_absa.yaml`: compact ABSA setup.
- `config_absa_full.yaml`: full ABSA experiment.
- `config_absa_debug.yaml`: lightweight ABSA debug run.

## Dataset Modes

### Hugging Face mode

Set:

- `dataset_source: huggingface`
- `dataset_name`, `dataset_subset`, and split names

### Local mode (recommended for ABSA)

Set:

- `dataset_source: local`
- `local_train_path`, `local_validation_path`, `local_test_path`
- ABSA-related fields such as `task_type: absa`, `aspect_field`, and `label_space`

If local files do not exist and `auto_download_local_dataset: true`, the loader can auto-materialize supported ABSA datasets into your configured local paths.

## Artifacts

Each launch writes to a timestamped directory under `artifacts/`, for example:

```text
artifacts/<dataset>_<task>_<timestamp>/
  config.json
  summary.json
  run_1/
    initial_population.json
    final_population.json
    pareto_front.json
    metrics.json
    best_instruction.txt
```

`summary.json` is updated from all completed runs under the artifact root.

## Notes on Runtime Cost

- The first run will download task/perplexity models (for example, Flan-T5 and RoBERTa).
- Evaluation is compute-intensive because each candidate instruction is trained/evaluated by the task model.
- For quick validation, start with `config_absa_debug.yaml` before scaling up.
