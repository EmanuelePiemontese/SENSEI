"""
Converts the compressed JSONL dataset into two numpy memmap files:
  - input_ids.npy      (N, MAX_LEN) int32
  - attention_mask.npy (N, MAX_LEN) int8

Execute ONLY ONCE before training:
  python convert_dataset.py

Using int32 and int8 drastically reduces the disk footprint:
  input_ids:      N * MAX_LEN * 4 bytes
  attention_mask: N * MAX_LEN * 1 byte
Ensure you have sufficient storage space on your drive/Lustre.
"""

import json
import os
import gzip
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizerFast

# --- PATH CONFIGURATION ---
# SCRIPT_DIR is SENSEI/4_Pre_training
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "SENSEI_config.yaml")

def load_config(path = DEFAULT_CONFIG_PATH):
    """Load the YAML configuration file."""
    with open(path, "r", encoding = "utf-8") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    p = config["paths"]
    m = config["model"]

    # Resolve paths absolute to the project structure
    root_dir = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    
    # Locate the input files inside results directory
    input_file = os.path.join(root_dir, p["processed_dataset"])
    csv_stats_path = os.path.join(os.path.dirname(input_file), "statistics_pretraining_prefixes.csv")
    
    tokenizer_dir = os.path.join(root_dir, p["tokenizer_dir"])
    
    # Explicitly set the output directory to 4_Pre_training/dataset_pre_training
    out_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "dataset_pre_training"))
    out_ids = os.path.join(out_dir, "input_ids.npy")
    out_mask = os.path.join(out_dir, "attention_mask.npy")
    
    max_len = int(m["max_len"])

    # ---------------------------
    # TOKENIZER
    # ---------------------------
    print("\n[INFO] Loading custom tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir)

    # ---------------------------
    # FAST ROW COUNT
    # ---------------------------
    print("[INFO] Retrieving exact row count from statistics CSV...")
    if not os.path.exists(csv_stats_path):
        raise FileNotFoundError(f"[ERROR] Statistics file not found at: {csv_stats_path}. Please check config.yaml paths.")
        
    df_stats = pd.DataFrame(pd.read_csv(csv_stats_path))
    global_row = df_stats[df_stats["Log_Name"] == "GLOBAL_TOTAL"]
    
    if global_row.empty:
        raise ValueError("[ERROR] Could not find 'GLOBAL_TOTAL' row in the statistics CSV.")
        
    n_rows = int(global_row["Generated_Prefixes"].values[0])

    print(f"[INFO] Valid rows to process:  {n_rows:,}")
    print(f"[INFO] Max sequence length:    {max_len}")

    # ---------------------------
    # ALLOCATE MEMMAP
    # ---------------------------
    # Ensure the target directory exists before creating the memmap files
    os.makedirs(out_dir, exist_ok = True)
    
    print(f"\n[INFO] Allocating memmap: input_ids {n_rows}x{max_len} int32...")
    ids_mm = np.lib.format.open_memmap(out_ids, mode = "w+", dtype = np.int32, shape = (n_rows, max_len))

    print(f"[INFO] Allocating memmap: attention_mask {n_rows}x{max_len} int8...")
    mask_mm = np.lib.format.open_memmap(out_mask, mode = "w+", dtype = np.int8, shape = (n_rows, max_len))

    # ---------------------------
    # TOKENIZE AND WRITE
    # ---------------------------
    print("[INFO] Starting tokenization and writing to disk...", flush = True)
    
    i = 0
    with gzip.open(input_file, "rt", encoding = "utf-8") as f:
        for line in tqdm(f, total = n_rows, desc = "Tokenization"):
            if not line.strip():
                continue
                
            record = json.loads(line)
            words = record["prefix"].split()
            
            enc = tokenizer(
                words,
                is_split_into_words = True,
                truncation = True,
                padding = "max_length",
                max_length = max_len,
            )
            
            ids_mm[i] = enc["input_ids"]
            mask_mm[i] = enc["attention_mask"]
            i += 1

            # Periodic flush to prevent RAM saturation
            if i % 500000 == 0:
                ids_mm.flush()
                mask_mm.flush()

    # Final flush
    ids_mm.flush()
    mask_mm.flush()
    
    # Delete references to safely close the files
    del ids_mm
    del mask_mm

    print(f"\n[SUCCESS] Conversion completed: {i:,} samples written.")
    print(f" -> {out_ids}")
    print(f" -> {out_mask}\n")

if __name__ == "__main__":
    main()