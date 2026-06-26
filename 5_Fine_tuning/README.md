# 5. Fine-tuning

This directory contains the core evaluation and fine-tuning scripts for the SENSEI model. After the model has internalized the syntax and control-flow dynamics of event logs during pre-training, this phase adapts those foundational weights to solve specific, high-value Process Mining predictive tasks. 

The scripts evaluate SENSEI against traditional baselines (e.g., standard Google BERT) across multiple datasets, ensuring rigorous benchmarking.

## Directory Structure

```text
5_Fine_tuning/
├── Test_NAP.py
├── Test_OP.py
└── README.md


```

---

## Components Overview

### `Test_NAP.py`

The script dedicated to the Next Activity Prediction benchmark.

* **objective:** predict the exact next event in an ongoing process trace.
* **mechanism:** formulated as a multi-class classification problem. The script dynamically extracts the full vocabulary of possible next activities from the dataset, fine-tunes the model using Mean Pooling over token embeddings, and calculates Accuracy and F1-Macro scores.

### `Test_OP.py`

The script dedicated to the Outcome Prediction benchmark.

* **objective:** predict the final disposition of a running case (e.g., Deviant vs. Regular behavior).
* **mechanism:** formulated as a binary classification problem. It leverages stratified splitting to handle class imbalances and calculates the robust ROC-AUC score alongside Accuracy and F1-Macro.

---

## Execution Workflow

Ensure your virtual environment is active and all previous phases (Pre-processing, Tokenization, and Pre-training) have been successfully completed. All commands should be executed from the root directory of the project.

### Step 1: Run Next Activity Prediction

Launch the script to fine-tune the model and evaluate it on the NAP task across all processed datasets.

```bash
python "5_Fine_tuning/Test_NAP.py"


```

### Step 2: Run Outcome Prediction

Launch the script to fine-tune the model and evaluate it on the OP task.

```bash
python "5_Fine_tuning/Test_OP.py"


```

---

## Generated Artifacts

Upon successful execution, the scripts will dynamically generate the required output folders and populate the root `model/` and `results/` directories:

* `../model/checkpoints_NAP/` & `../model/checkpoints_OP/`: contains intermediate safety checkpoints saved whenever the validation loss improves.
* `../model/SENSEI_fine_tuned_NAP/` & `../model/SENSEI_fine_tuned_OP/`: the final, complete model directories containing the `.pt` state dictionaries, `config.json`, and Tokenizer files ready for deployment.
* `../results/Fine_tuning_benchmark/`: contains the final `risultati_nap_benchmark.csv` and `risultati_op_benchmark.csv` evaluation reports.