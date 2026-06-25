# 4. Pre-training

This directory contains the core engine for pre-training the SENSEI model from scratch. It leverages a Masked Language Modeling (MLM) objective to teach a lightweight BERT architecture the underlying syntax and sequential dynamics of process mining logs. 

The pipeline is highly optimized for high-performance computing (HPC) environments, featuring memory-mapped datasets to prevent RAM bottlenecks and Distributed Data Parallel (DDP) support for multi-GPU scaling.

## Directory Structure

```text
4_Pre_training/
├── dataset_pre_training/ # Created after running convert_dataset.py
├── SENSEI_config.yaml
├── convert_dataset.py
├── pre_training_from_scratch_multi.py
├── run_mlm.sh
└── README.md

```

---

## Components Overview

### `SENSEI_config.yaml`

The master configuration hub. It centralizes all structural definitions, avoiding hardcoded values in the scripts. It dictates:

* **paths:** targets for datasets, tokenizers, and output directories.
* **architecture:** BERT hyper-parameters (e.g., `hidden_size = 512`, `num_hidden_layers = 4`).
* **training settings:** hardware scaling options, batch sizes, learning rates, and early stopping patience.

### `convert_dataset.py`

A crucial memory-optimization script. It reads the compressed `.jsonl.gz` prefixes generated in the pre-processing phase and converts them into massive, static numpy memory-mapped arrays (`input_ids.npy` and `attention_mask.npy`).

* **output:** saves the binary matrices directly into the local `dataset_pre_training/` folder, allowing the GPU to read gigabytes of data instantly from disk without saturating the system RAM.

### `pre_training_from_scratch_multi.py`

The main training script.

* **features:**
    * initializes a raw `BertForMaskedLM` mapping to our custom abstract tokenizer.
    * uses `mmap_mode = "r"` to stream data dynamically.
    * fully supports PyTorch DDP for multi-GPU/multi-node scaling.
    * implements mixed-precision training (`bfloat16`) and Fused AdamW for maximum computational throughput.


* **outputs:** generates epoch checkpoints and the final model, saving them cleanly to the root `../model/` directory.

### `run_mlm.sh`

A shell script designed to launch the distributed training process. It is typically configured for HPC workload managers (like SLURM) or `torchrun` to allocate multiple GPUs and set up the master/worker environment variables properly.

---

## Execution Workflow

Ensure your virtual environment is active and all paths in `SENSEI_config.yaml` correctly point to your local project structure. All commands should be executed from the root directory of the project.

### Step 1: Dataset Conversion

You must generate the memory-mapped arrays before starting the training. This only needs to be done once per dataset.

```bash
python "4_Pre_training/convert_dataset.py"

```

### Step 2: Start Pre-training

For a standard local execution or single-GPU testing, you can launch the Python script directly. If you are deploying on a multi-GPU cluster, use the bash script.

```bash
# Standard / Single-GPU Execution
python "4_Pre_training/pre_training_from_scratch_multi.py"

# Cluster / Distributed Execution
bash "4_Pre_training/run_mlm.sh"

```

---

## Generated Artifacts

Upon successful execution, the training script will populate the root `model/` directory:

* `../model/checkpoints/`: contains intermediate checkpoints saved whenever the validation loss improves (`best_epoch_XXX`).
* `../model/SENSEI_pre_trained/`: the final, complete model directory containing the `.safetensors` weights, `config.json`, and the embedded custom tokenizer. This folder is perfectly primed for the downstream Fine-Tuning phase.