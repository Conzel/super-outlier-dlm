# Diffusion-Prune

Systematic evaluation of pruning and quantization for diffusion language models
(LLaDA-8B, DREAM-7B) compared against autoregressive baselines (Llama 3.1 8B,
Qwen 2.5 7B).

This repository accompanies the paper **"Layer Collapse in Diffusion Language
Models"** by Alexander Conzelmann, Albert Catalan-Tatjer, and Shiwei Liu
(Tübingen AI Center / MPI for Intelligent Systems / ELLIS Institute Tübingen).
arXiv: TODO (link pending). See [`CITATION.cff`](CITATION.cff) for citation
metadata.

## Installation

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

For development:
```bash
uv pip install -e ".[dev]"
pre-commit install
```

Run the test suite:
```bash
pytest tests/
```

## Quick Start

Single run (Hydra config, override from CLI):
```bash
python scripts/run.py model=llada_8b pruning=wanda pruning.sparsity=0.5 evaluation=commonsense
```

HTCondor batch submission:
```bash
python scripts/submit.py model=llada_8b pruning=wanda evaluation=commonsense \
    --multirun pruning.sparsity=0.2,0.3,0.4,0.5,0.6,0.7
```

A SLURM launcher is also available; both are configured via
`configs/condor.yaml` / `configs/slurm.yaml` and overridable per-cluster via
`configs/local/{condor,slurm}.yaml` (see the `*.example` templates).

## Reproducing the paper

The repo's `out/` directory (eval result JSONs, ~3 GB) is gitignored. To
regenerate paper figures you have two options:

**Option A — re-run all experiments end-to-end.** Requires GPU compute
(H100-class, ~hundreds of GPU-hours total).

1. Set environment variables:
   ```bash
   export REPO_DIR=$PWD
   export WORK_DIR=/path/to/scratch
   export MODELS=/path/to/model/cache       # HF model snapshots land here
   export HF_HOME=/path/to/hf/cache         # datasets cache root
   ```
2. Pre-download models, C4 calibration data, and eval datasets:
   ```bash
   python scripts/download_artifacts.py
   ```
3. Submit each surviving experiment. Each `experiments/AXX_*/run.sh` is
   self-contained and writes results into the flat `out/` directory:
   ```bash
   bash experiments/A11_owl_scores/run.sh
   bash experiments/A23_pruning_statistics/run.sh
   bash experiments/A24_pythia160m/run.sh
   bash experiments/A25_activation_histograms/run.sh
   bash experiments/A26_strategy_gap_fill/run.sh
   bash experiments/A27_channel_magnitude_per_step/run.sh
   ```
4. Render every paper figure:
   ```bash
   bash scripts/replot_paper_figures.sh
   ```
   Figures land under `plots/experiments/AXX_*/`.

**Option B — figures only, from cached results.** Download the precomputed
`out/` snapshot from <TODO: Zenodo / HF Hub URL>, extract it into the repo
root, then run step 4 above. This skips all GPU compute.

### Mapping experiments to paper figures

| Experiment | Produces |
|---|---|
| `A11_owl_scores` | OWL outlier-score analyses |
| `A23_pruning_statistics` | Pruning statistics across models |
| `A24_pythia160m` | 160M-scale ablations |
| `A25_activation_histograms` | Per-layer activation heatmaps |
| `A26_strategy_gap_fill` | Sparsity-allocation strategy comparison |
| `A27_channel_magnitude_per_step` | Per-step channel-magnitude sweep |

## Project Structure

- `src/diffusion_prune/`: Main package
  - `model/`: Model loading (AR + DLM)
  - `pruning/`: WANDA, DWANDA (diffusion-aware), magnitude, SparseGPT, OWL /
    alpha sparsity allocation
  - `quantization/`: GPTQ, RTN, plus virtual variants for DLMs
  - `evaluation/`: lm-eval-harness integration with result caching
  - `diffusion_masking.py`: Random-timestep masking for DLM calibration
- `configs/`: Hydra configs (`model/`, `pruning/`, `quantization/`,
  `evaluation/`, plus cluster launchers `condor.yaml` / `slurm.yaml`)
- `scripts/`: Entry points (`run.py`, `submit.py`), figure / table generation
  (`plot.py`, `_tables.py`, `summary_table.py`, `baseline_table.py`,
  `best_hyperparams.py`, `replot_paper_figures.sh`, `pruning_statistics.py`,
  per-stat modules under `stats/`), data download (`download_*.py`)
- `experiments/`: One folder per paper experiment (`AXX_<short_desc>`), each
  with `run.sh` and `plot.sh`
- `out/`: Flat layout of evaluation result JSONs (gitignored)
- `plots/`: Generated figures (gitignored)
- `tests/`: pytest suite

## Models

Base and instruct variants of:
- LLaDA-8B (DLM)
- DREAM-7B (DLM)
- Llama 3.1 8B (AR)
- Qwen 2.5 7B (AR)

Plus small AR/DLM 160M variants for fast iteration.

## Tasks

- **QnA (base models)**: arc_challenge, hellaswag, piqa, winogrande, boolq,
  openbookqa (`evaluation=commonsense`)
- **Reasoning (instruct models)**: GSM8K (`evaluation=gsm8k`)

## Methods

- **Pruning**: WANDA, DWANDA, magnitude, SparseGPT; with uniform / OWL /
  alpha (deeper-is-sparser, earlier-is-sparser) allocations
- **Quantization**: GPTQ, RTN, plus virtual variants for DLMs

## Configuration

Configs are composed via Hydra. The default entry is `configs/config.yaml`;
override fields from the CLI:

```bash
python scripts/run.py model=dream_7b pruning=wanda pruning.sparsity=0.5 \
    pruning.allocation=earlier evaluation=commonsense
```

Paths are controlled by the `REPO_DIR`, `WORK_DIR`, `MODELS`, `HF_HOME` env
vars (see "Reproducing the paper" above).

## License

Released under the Apache 2.0 license — see [`LICENSE`](LICENSE).

## Citation

If you use this code, please cite:

```bibtex
@article{conzelmann2026layercollapse,
  title  = {Layer Collapse in Diffusion Language Models},
  author = {Conzelmann, Alexander and Catalan-Tatjer, Albert and Liu, Shiwei},
  year   = {2026},
  note   = {TODO: arXiv link},
}
```

GitHub also renders a "Cite this repository" widget from
[`CITATION.cff`](CITATION.cff).
