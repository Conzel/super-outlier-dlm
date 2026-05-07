# Scripts

## `run.py`

Main evaluation pipeline. Uses Hydra to load configs from `configs/`, applies optional pruning, evaluates with lm-eval harness, and saves JSON results to `out/`.

```bash
# Run with default config (wanda pruning on GSM8K)
python scripts/run.py

# Override pruning config
python scripts/run.py pruning=magnitude_alpha pruning.sparsity=0.3

# Disable pruning (baseline)
python scripts/run.py pruning=none

# Override multiple settings
python scripts/run.py pruning.sparsity=0.5 evaluation.limit=200 use_cache=false
```

Results are saved as JSON files to `out/` (configurable via `output_dir`) and cached in `out/.cache/`.

## `plot.py`

Aggregates JSON result files and creates comparison plots. Supports flexible axis selection, multi-curve grouping, and dot-notation filtering by any config parameter.

```bash
# Accuracy vs sparsity, one curve per pruning strategy
python scripts/plot.py -x pruning.sparsity -c pruning.strategy \
    --evaluation.task gsm8k -t "GSM8K Accuracy vs Sparsity"

# Plot a specific metric from additional_metrics
python scripts/plot.py -x pruning.sparsity -c pruning.strategy \
    -y metrics.exact_match,flexible-extract \
    --evaluation.task gsm8k -t "GSM8K Flexible Accuracy"

# Group curves by two parameters (color + linestyle)
python scripts/plot.py -x pruning.sparsity \
    -c pruning.strategy -c pruning.sparsity_strategy \
    --evaluation.task gsm8k -t "Strategy Comparison"

# Max over a hyperparameter at each x-value
python scripts/plot.py -x pruning.sparsity -c pruning.strategy \
    -m pruning.alpha_epsilon --evaluation.task gsm8k -t "Best Epsilon"

# Paper-quality PDF
python scripts/plot.py -x pruning.sparsity -c pruning.strategy \
    --evaluation.task gsm8k -t "GSM8K Results" --style paper -f pdf
```

Plots are saved to `plots/` with the title as filename. Filters use `--<config.key> <value>` syntax (e.g. `--evaluation.task gsm8k --model.model_type llada-8b`).

## `best_hyperparams.py`

Finds the best hyperparameter configuration at each sparsity level. Prints a ranked table per sparsity with the top configs and a consistency summary across levels.

```bash
# Find best configs for GSM8K
python scripts/best_hyperparams.py \
    --evaluation.task gsm8k --model.model_type llada-8b

# With additional filters
python scripts/best_hyperparams.py \
    --evaluation.task gsm8k --evaluation.limit 200 --evaluation.num_fewshot 8
```

Uses the same dot-notation filter syntax as `plot.py`.

## `pruning_statistics.py`

Visualizes pruning properties without running evaluation. Two subcommands:

### `pruning_statistics.py alpha`

Computes and plots per-layer metrics (alpha exponent, spectral norm, stable rank) for a model. One line per sublayer type (q_proj, k_proj, etc.).

```bash
# Default: alpha_peak for LLaDA-8B
python scripts/pruning_statistics.py alpha --model-type llada-8b

# Different metric
python scripts/pruning_statistics.py alpha --model-type llada-8b --metric stable_rank

# Paper style, custom output
python scripts/pruning_statistics.py alpha --model-type llada-8b \
    --metric spectral_norm --style paper -o plots/spectral_llada.pdf
```

Requires loading the model (needs GPU).

### `pruning_statistics.py sparsity`

Bar plot showing what per-layer sparsity each strategy would assign. Computes sparsities directly from strategy functions without actually pruning.

```bash
# Compare non-alpha strategies (no model needed)
python scripts/pruning_statistics.py sparsity --num-layers 32 \
    --strategies uniform deeper-is-sparser earlier-is-sparser

# Different target sparsity and epsilon
python scripts/pruning_statistics.py sparsity --num-layers 32 \
    --sparsity 0.7 --alpha-epsilon 0.5

# Include alpha-pruning (needs model for weight metrics)
python scripts/pruning_statistics.py sparsity --model-type llada-8b \
    --strategies uniform alpha-pruning --sparsity 0.5
```

Non-alpha strategies only need `--num-layers`. Alpha-pruning requires loading the model to compute layer metrics.
