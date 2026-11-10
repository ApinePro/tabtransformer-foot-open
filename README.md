# A Universal Dense Football Event Representation Based on TabTransformer

This repository is source-available for academic, educational, and non-commercial research use only.

Official research code and reproducibility package for:

> Weiran Yang, Daniel Memmert, and Maximilian Klemp.  
> **A Universal Dense Football Event Representation Based on TabTransformer.**

This cleaned repository contains only the experiments used for the final paper:

1. row-masked self-supervised TabTransformer pretraining;
2. match-level five-fold xG experiments and StatsBomb xG comparison;
3. match-level five-fold VAEP experiments;
4. player-style evaluation and similar-player retrieval.

The large derived embedding caches are deliberately not shipped. They can be regenerated from the included data and pretrained checkpoint.

## Repository layout

| Path | Purpose |
|---|---|
| `main.py` | Row-masked self-supervised pretraining |
| `train_xg_*.py`, `run_xg_kfold.py` | xG models and five-fold runner |
| `compare_statsbomb_xg.py` | Comparison with the StatsBomb xG field |
| `train_vaep_*.py`, `run_vaep_kfold.py` | VAEP baselines, TabTransformer variants, and five-fold runner |
| `eval_player_style.py` | Player-style representation evaluation |
| `find_similar_players.py` | Similar-player retrieval and built-in demo queries |
| `checkpoints/tabtransformer_row_mask_pretrain.pt` | Paper pretraining checkpoint |
| `.vaep_cache/` | Runtime-generated caches (not distributed or tracked) |
| `dataset/`, `dataset-test/` | Local StatsBomb event data (ignored by Git) |

## Data

The local hand-off contains both required data roots:

- `dataset/`: 1,823 training matches, 3,656 parquet files, about 785 MB;
- `dataset-test/`: 147 held-out matches, 417 parquet files, about 183 MB.

The training split covers the 2015/16 seasons of the Premier League, La Liga, Serie A, Bundesliga, and Ligue 1. The held-out split contains UEFA Euro 2020, FIFA World Cup 2022, and Copa America 2024.

The data directories are intentionally ignored by Git because of their size and upstream licensing/distribution considerations. To rebuild them from StatsBomb Open Data, run:

```bash
python download_statsbomb_open_data.py
```

See `EVENT_COLUMNS_DICTIONARY.md` for the local event schema.

## Environment

Python 3.10 is recommended. Create an isolated environment and install:

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for pretraining and embedding extraction. Tree-model commands can run on CPU; add `--gpu` where supported to use GPU training.

## Reproducing the paper experiments

All commands below are run from the repository root.

### 1. Self-supervised pretraining

The paper checkpoint is already included. To train it again from the local training data:

```bash
python main.py
```

The downstream scripts expect:

```text
checkpoints/tabtransformer_row_mask_pretrain.pt
```

### 2. xG five-fold experiment (paper Table 2)

The paper configuration uses validation Brier score for checkpoint selection:

```bash
python run_xg_kfold.py --models mlp,gru_frozen --select-metric brier
```

Outputs:

- `xg_kfold_fold_assignments.csv`
- `xg_kfold_metrics.csv`
- `xg_kfold_summary.csv`
- `kfold_results/`

The runner creates and reuses an event-sequence cache under `.vaep_cache/xg_kfold/`. The first run therefore takes longer; later runs reuse the generated cache. Delete it if the dataset or sequence-building code changes.

To rebuild only the summaries from completed folds:

```bash
python run_xg_kfold.py --aggregate-only
```

### 3. StatsBomb xG comparison

```bash
python compare_statsbomb_xg.py
```

The script generates `statsbomb_xg_comparison_metrics.csv` and `statsbomb_vs_model_xg_predictions.csv`.

### 4. VAEP five-fold experiment (paper Table 3)

No preprocessed VAEP feature tables or embeddings are distributed. The raw-feature and SPADL baselines rebuild their tables from `dataset/` and `dataset-test/` on first use. The PCA-128 and full 911-dimensional TabTransformer embeddings must also be generated locally.

Generate the PCA-128 embeddings:

```bash
python train_vaep_tabtransformer_catboost.py --gpu
```

Generate the full 911-dimensional embeddings required by `tt_mlp`:

```bash
python train_vaep_tabtransformer_catboost_full.py
```

The full-dimensional script needs about 25 GB for persistent embedding memmaps and can temporarily require roughly 42 GB more while building quantized CatBoost input. Allow approximately 70 GB of free disk space. Do not pass `--keep-tsv` unless the temporary TSV files are specifically needed.

Run all seven models reported in the paper:

```bash
python run_vaep_kfold.py --models tt_catboost_pca,catboost_raw,xgboost_raw,mlp_baseline,tt_mlp,spadl_catboost,spadl_xgboost --gpu
```

Outputs:

- `vaep_kfold_fold_assignments.csv`
- `vaep_kfold_metrics.csv`
- `vaep_kfold_summary.csv`
- `vaep_kfold_results/`

To rebuild only the summaries from completed folds:

```bash
python run_vaep_kfold.py --aggregate-only
```

### 5. Player-style evaluation and retrieval (paper Table 4)

```bash
python eval_player_style.py
python find_similar_players.py --demo elite --top-k 10 --gpu --output similar_elite6_top10.csv
```

The first run builds a player-gallery cache under `.vaep_cache/`. It is tied to the current dataset and checkpoint and is rebuilt when invoked with `--no-cache` or when its cache key changes.

## Cache policy

No derived caches are distributed or tracked. The scripts generate the following local artifacts as needed:

- xG event sequences;
- raw StatsBomb VAEP train/validation/test feature tables;
- SPADL VAEP train/validation/test feature tables;
- player-gallery representations;
- PCA-128 and full 911-dimensional TabTransformer embeddings.

These are derived artifacts, not source data or trained checkpoints. They can be regenerated from the included code, local datasets, and pretrained checkpoint using the commands above. Cache filenames contain dataset-dependent keys; after changing the data, remove stale cache files and rebuild them.

## Generated results

No precomputed fold assignments, metrics, predictions, or similar-player result files are distributed. The experiment scripts generate them locally, including:

- xG fold outputs under `kfold_results/` and the `xg_kfold_*.csv` summaries;
- VAEP fold outputs under `vaep_kfold_results/` and the `vaep_kfold_*.csv` summaries;
- StatsBomb comparison metrics and predictions;
- player-style evaluation and similar-player CSV outputs.

Generated CSV files store unrounded values. Re-running GPU models may produce small numerical differences even with the fixed fold assignments and seeds.

## Reproducibility notes

- Both cross-validation runners split at match level with five folds and fold seed 42.
- The held-out `dataset-test/` data does not participate in fold construction.
- The included checkpoint is the only model artifact required to extract paper embeddings.
- Full end-to-end experiments are compute- and disk-intensive. Use `--folds`, `--max-train-matches`, and `--max-test-matches` where available for smoke tests only; those reduced runs are not paper results.

