# 3. Tokenization

This directory contains the script responsible for creating a custom vocabulary and configuring a dedicated Hugging Face tokenizer. 

Unlike traditional Natural Language Processing (NLP) tasks that use WordPiece or Byte-Pair Encoding (BPE) to split words into sub-words, SENSEI requires events to be treated as indivisible atomic concepts. This script ensures that our abstract event symbols (e.g., `ACT_001`, `ACT_002`) are processed correctly without being fragmented.

## Directory Structure

```text
3_Tokenization/
├── Abstract_Symbol_Mapping_Tokenization.py
└── README.md

```

---

## Script Overview

### `Abstract_Symbol_Mapping_Tokenization.py`

This script builds the vocabulary from scratch and initializes a `BertTokenizerFast`.

* **features:**
    * defines standard BERT special tokens (`[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`).
    * dynamically generates a fixed-size vocabulary of abstract activity tokens (up to 994 activities, `ACT_001` to `ACT_994`).
    * adds these tokens to the tokenizer as special non-splittable tokens, preventing the tokenizer from breaking them down at the underscore character `_`.
    * includes a built-in integrity test to verify that a sample sequence is correctly tokenized and decoded.


* **outputs:** the generated artifacts are saved in the global `../model/` directory for downstream use in Pre-training and Fine-tuning.
    * `../model/vocab.txt`: the raw text file containing all 999 vocabulary items.
    * `../model/tokenizer_abstract_symbol_mapping/`: the directory containing the serialized Hugging Face tokenizer files (`tokenizer_config.json` and `tokenizer.json`).



---

## How to Run the Script

Ensure your virtual environment (`venv_SENSEI`) is activated. All commands must be executed from the root directory of the project.

### Execute Tokenization Setup

Run the following command to generate the vocabulary and save the tokenizer:

```bash
python "3_Tokenization/Abstract_Symbol_Mapping_Tokenization.py"

```

Upon successful execution, you will see a console output detailing the vocabulary size and the results of the tokenization integrity test. You can then inspect the `model/` directory to verify the saved artifacts.

```