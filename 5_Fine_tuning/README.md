# 5. Fine-tuning

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
