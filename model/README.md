# Model

This directory acts as the central repository for all generated model weights, training checkpoints, and tokenizers throughout the SENSEI pipeline. It is dynamically populated by the scripts executed during the Tokenization, Pre-training, and Fine-tuning phases.

When cloning the repository, this directory will be empty. The files will be automatically created and filled during the execution of the pipeline.

## Directory Structure

*Note: the subdirectories below are generated automatically during the execution of the pipeline.*

```text
model/
├── checkpoints_NAP/
├── checkpoints_OP/
├── SENSEI_fine_tuned_NAP/
├── SENSEI_fine_tuned_OP/
├── SENSEI_pre_trained/
├── tokenizer_abstract_symbol_mapping/
└── README.md


```

---

## Components Overview

### `tokenizer_abstract_symbol_mapping/`

Generated during Phase 3 (Tokenization).

* **content:** `vocab.txt`, `tokenizer.json`, and `tokenizer_config.json`.
* **purpose:** holds the custom abstract symbol tokenizer designed to map process activities without splitting them into sub-words. It is injected into all subsequent models.

### `SENSEI_pre_trained/`

Generated at the end of Phase 4 (Pre-training).

* **content:** `.safetensors` weights, `config.json`, `tokenizer.json` and `tokenizer_config.json`.
* **purpose:** contains the final foundation model resulting from the Masked Language Modeling (MLM) pre-training phase. This is the core "brain" of SENSEI, ready to be adapted for downstream tasks.

### `checkpoints_NAP/` & `checkpoints_OP/`

Generated dynamically during Phase 5 (Fine-tuning).

* **content:** intermediate `.pt` state dictionaries.
* **purpose:** acts as temporary safety checkpoints saved during fine-tuning whenever the validation loss improves, preventing data loss in case of early stopping or crashes.

### `SENSEI_fine_tuned_NAP/` & `SENSEI_fine_tuned_OP/`

Generated at the end of Phase 5 (Fine-tuning).

* **content:** task-specific `.pt` state dictionaries, architectural configurations, and embedded tokenizer files.
* **purpose:** the final, fully optimized and deployable models tuned specifically for Next Activity Prediction (NAP) and Outcome Prediction (OP).