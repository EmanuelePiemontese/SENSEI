# 5.1 Preparing data
# `prepare_data.py`

Data preparation for the **scratch vs. pretrained** comparison on Next Activity Prediction (NAP).

For each `(log, fraction, seed)` combination the script writes one pickle file containing prefixes, labels, activity mappings and metadata, ready to be consumed by the training script (`train_nap_fractions.py`).

## What it does

1. **Loads the event log** from `<logs_dir>/<dataset_name>.csv` (`;`-separated). Activities are cast to string and timestamps are parsed as UTC datetimes, so the sort order is truly chronological.
2. **Chronological split by case** (default 66/34): cases are ordered by the timestamp of their first event; the earliest 66% form the train pool, the rest form the test set.
3. **Builds the test set once.** It is computed a single time and is **identical for every fraction and seed**.
4. **Subsamples only the training cases.** For each seed, one random permutation of the train cases is drawn; taking its first `n` cases gives **nested subsets** (the 5% subset is contained in the 10% subset, and so on, for a given seed).
5. **Builds the activity mapping on the subsampled train only.** Activities are ranked by event frequency (ties broken alphabetically) and mapped to codes such as `ACT_001`, `ACT_002`, ...
6. **Maps unseen activities to `[UNK]`.** Activities that never appear in the subsampled train become `[UNK]`. This never happens in train, but can happen in test (inputs and labels).
7. **Generates prefixes.** For a trace `a1..an`, every prefix `a1..aj` (`1 <= j < n`) is paired with the next activity `a(j+1)` as label.
8. **Saves the pickle** `<out_dir>/<dataset_name>_f<frac>_s<seed>.pkl`.

## Requirements

- Python 3.8+
- `numpy`, `pandas`
- `pm4py` (imported unconditionally at the top of the script, even if the log is a CSV)

## Usage

```bash

python prepare_nap_fractions.py <log_name> \
    --fracs 0.01 0.05 0.1 0.25 0.5 1.0 \
    --seeds 42
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `dataset_name` | (required) | Log name, without extension. The script reads `<logs_dir>/<dataset_name>.csv`. |
| `--logs_dir` | `/home/vincenzo/.../prepare_data/logs` | Directory containing the logs. |
| `--out_dir` | `/home/vincenzo/.../log_nap_frac/pkl` | Output directory for the pickles. |
| `--fracs` | `1.0` | Fractions of the train cases to keep (one pickle per value). |
| `--seeds` | `42` | Seeds for the case subsampling (one pickle per value). |
| `--min_cases` | `10` | Minimum number of train cases, so that very small fractions are not degenerate. |
| `--train_ratio` | `0.66` | Fraction of cases (chronologically first) used as train pool. |
| `--code_fmt` | `ACT_{:03d}` | Format of the activity codes. **Must match the pretraining format** (e.g. `"ACT_{:03d}"` or `"Activity{}"`). |

## Input format

The CSV must contain these columns (edit the constants at the top of the script otherwise):

| Constant | Column name |
|---|---|
| `CASE` | `Case ID` |
| `ACT` | `Activity` |
| `TS` | `time:timestamp` |

A commented-out line in the script shows the alternative standard XES names (`case:concept:name`, `concept:name`).

## Output

Each pickle is a dictionary with:

| Key | Content |
|---|---|
| `X_train_abs`, `y_train_abs` | Train prefixes / labels as activity codes (`ACT_xxx`). |
| `X_test_abs`, `y_test_abs` | Test prefixes / labels as codes (`[UNK]` for unseen activities). |
| `X_train_txt`, `y_train_txt`, `X_test_txt`, `y_test_txt` | Same data with the original activity names. |
| `activity_map` | Original activity -> code (built on the subsampled train only). |
| `inverse_map` | Code -> original activity. |
| `label_set_txt` | Sorted activities of the whole log (fixed class set for metrics). |
| `len_train`, `len_test` | Prefix length of each example. |
| `groups_train`, `groups_test` | Case ID of each prefix (used for the case-level train/val split). |
| `meta` | `log`, `frac`, `seed`, `n_train_cases`, `n_total_train_cases`, `train_ratio`, `min_cases`. |


# 5.2 Fine-tuning

# `fine_tunining_sensei.py`

Training and evaluation script for the **scratch vs. pretrained** comparison on Next Activity Prediction (NAP) with SENSEI encoder, using training expressed in **epochs** with early stopping.

It consumes the pickles produced by `prepare_nap_fractions.py` and appends one result row per `(log, fraction, seed, mode)` to a CSV file.

## Modes

| Mode | Encoder init | Encoder trainable | `lr_enc` | `lr_head` |
|---|---|---|---|---|
| `scratch` | random (same architecture/config as checkpoint) | yes | 1e-4 | 1e-4 |
| `finetune` | pretrained checkpoint | yes | 1e-5 | 1e-3 |
| `frozen` | pretrained checkpoint | no (only the head is trained) | - | 1e-3 |
| `frozen_random` | random | no (only the head is trained) | - | 1e-3 |

`frozen_random` is a control: it measures how much a linear head can get from random features. In frozen modes the encoder is kept in `eval()` mode (no dropout).

The hyperparameter grids in `MODES` currently contain a single value each; add values to the lists to enable real tuning.

## Training protocol

- **Train/validation split** by **case** (`GroupShuffleSplit`, `--val_ratio`, seeded), so prefixes of the same case never appear in both sets.
- **Optimizer**: AdamW with separate learning rates for encoder and head; no weight decay on biases and LayerNorm weights.
- **Scheduler**: linear warmup (`--warmup_ratio` of total steps) then linear decay.
- **Gradient clipping** at norm 1.0.
- **Epoch-based loop**: up to `--epochs` full passes over the training set, with a validation loss computed at the end of every epoch.
- **Early stopping**: stops after `--patience` consecutive epochs without validation-loss improvement. The best-epoch weights are kept in memory.
- **Test**: evaluated with the best checkpoint, selected **only on validation loss**.
- **Micro-batching** (`--micro_bs`): splits each batch into smaller chunks to save VRAM. The loss is `sum / batch_n`, so the gradient is identical to the full-batch mean.

## Requirements

- Python 3.8+
- `torch`, `transformers`, `scikit-learn`, `numpy`
- A pretrained SENSEI checkpoint
- A tokenizer directory (default `tokenizer`) whose vocabulary is extended with the activity codes of each subset
- Pickles from `prepare_nap_fractions.py`

A GPU is used automatically if available.

## Usage

```bash
# All modes, all pickles in the directory
python train_nap_fractions.py --pickle_dir path/to/pkl
```
