import os
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- PATH CONFIGURATION ---
# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the base path to the raw data directory
BASE_PATH_RAW = os.path.join(SCRIPT_DIR, "..", "data", "Dataset_Pre_training")
# The absolute path to the base directory containing the XES files
BASE_PATH = os.path.abspath(BASE_PATH_RAW)

# Construct the output path to the specific results subdirectory
OUTPUT_DIR_RAW = os.path.join(SCRIPT_DIR, "..", "results", "Pre_processing_pre_training")
# The absolute path to the directory where JSON and CSV will be saved
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR_RAW)

PATH_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "processed_pretraining_dataset.json")
PATH_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "statistics_pretraining.csv")

def get_synthetic_files(base_path):
    """Retrieve all .xes files from the given directory."""
    files_list = []
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.endswith(".xes"):
                files_list.append(os.path.join(root, f))
    return files_list

def process_single_synthetic_log(file_path):
    """
    Parse a single XES file using regex to extract activities and convert them to symbolic tokens.
    ABSTRACT SYMBOL MAPPING: convert each unique activity to a symbolic token based on its frequency of occurrence (most frequent activity gets ACT_001, second most frequent gets ACT_002, etc.).
    """
    base_name = os.path.basename(file_path)
    parent_dir = os.path.basename(os.path.dirname(file_path))
    unique_filename = f"{parent_dir}_{base_name}"
    
    try:
        with open(file_path, "r", encoding = "utf-8", errors = "ignore") as f:
            content = f.read()
            
        traces = []
        trace_blocks = content.split("<trace>")[1:]
        
        for block in trace_blocks:
            end_idx = block.find("</trace>")
            if end_idx != -1:
                block = block[:end_idx]
                
            events = []
            for line in block.split("\n"):
                if "concept:name" in line:
                    match = re.search(r"value=\"([^\"]+)\"", line)
                    if match:
                        events.append(match.group(1))
            if events:
                traces.append(events)
                
        all_events = [e for t in traces for e in t]
        counts = Counter(all_events)
        mapping = {item[0]: f"ACT_{i + 1:03d}" for i, item in enumerate(counts.most_common())}
        
        symbolic_sequences = []
        for t in traces:
            sym_t = [mapping[act] for act in t]
            trace_string = " ".join(sym_t)
            symbolic_sequences.append(trace_string)
            
        return unique_filename, symbolic_sequences, mapping, None
        
    except Exception as e:
        error_name = type(e).__name__
        error_message = str(e)
        return unique_filename, None, None, f"{error_name}: {error_message}"

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # Ensure the target results directory exists before saving anything
    os.makedirs(OUTPUT_DIR, exist_ok = True)
    
    # ==========================================
    # PHASE 1: PARALLEL EXTRACTION
    # ==========================================
    files_to_process = get_synthetic_files(BASE_PATH)

    if not files_to_process:
        print(f"\n[ERROR] No files processed. There are no .xes files in: {BASE_PATH}")
    else:
        print(f"\n[INFO] Found {len(files_to_process)} synthetic logs. Starting massive extraction...")
        
        pretraining_dataset = {}
        error_log = []
        
        with ProcessPoolExecutor(max_workers = 4) as executor:
            futures = {executor.submit(process_single_synthetic_log, path): path for path in files_to_process}
            
            for future in tqdm(as_completed(futures), total = len(files_to_process), desc = "Multi-Core Processing"):
                name, seq, mapping, err = future.result()
                
                if err:
                    error_log.append(f"{name}: {err}")
                elif name and seq:
                    pretraining_dataset[name] = seq

        # Save JSON
        with open(PATH_OUTPUT_JSON, "w", encoding = "utf-8") as f:
            json.dump(pretraining_dataset, f, indent = 4)
            
        print(f"\n[SUCCESS] JSON Dataset saved at: {os.path.abspath(PATH_OUTPUT_JSON)}")

        # ==========================================
        # PHASE 2: CALCULATE STATISTICS AND SAVE CSV
        # ==========================================
        print("\n[INFO] Calculating global statistics...")
        
        stats_list = []
        global_vocab = set()
        total_traces_global = 0
        total_events_global = 0

        # Iterate directly over the memory dictionary without reloading the JSON
        for log_name, traces in pretraining_dataset.items():
            num_traces = len(traces)
            log_vocab = set()
            lengths = []

            for trace in traces:
                events = trace.split(" ")
                num_events = len(events)

                lengths.append(num_events)
                log_vocab.update(events)
                total_events_global += num_events

            total_traces_global += num_traces
            global_vocab.update(log_vocab)

            stats_list.append({
                "Log": log_name,
                "#Traces": num_traces,
                "#Events": sum(lengths),
                "#Activities": len(log_vocab),
                "Min Length": min(lengths) if lengths else 0,
                "Avg Length": round(np.mean(lengths), 2) if lengths else 0,
                "Max Length": max(lengths) if lengths else 0
            })

        # Create DataFrame
        df_stats = pd.DataFrame(stats_list)

        # Add totals row
        global_row = pd.DataFrame([{
            "Log": "GLOBAL_TOTAL",
            "#Traces": total_traces_global,
            "#Events": total_events_global,
            "#Activities": len(global_vocab),
            "Min Length": df_stats["Min Length"].min(),
            "Avg Length": round(total_events_global / total_traces_global, 2) if total_traces_global > 0 else 0,
            "Max Length": df_stats["Max Length"].max()
        }])

        df_final = pd.concat([df_stats, global_row], ignore_index = True)
        
        # Save CSV
        df_final.to_csv(PATH_OUTPUT_CSV, index = False)
        print(f"[SUCCESS] CSV Statistics Report saved at: {os.path.abspath(PATH_OUTPUT_CSV)}")

        # Print global results to screen
        global_row_data = df_final.iloc[-1]
        
        print("\n" + "=" * 80)
        header_text = "GLOBAL STATISTICAL REPORT - PRE-TRAINING"
        print(f"{header_text:^80}")
        print("=" * 80)
        
        # Using variables to avoid single quotes in f-string formatting
        val_traces = int(global_row_data["#Traces"])
        val_events = int(global_row_data["#Events"])
        val_activities = int(global_row_data["#Activities"])
        val_avg_length = global_row_data["Avg Length"]
        
        print(f"Total Logs Processed:      {len(pretraining_dataset):,}")
        print(f"Total Traces:              {val_traces:,}")
        print(f"Total Events:              {val_events:,}")
        print(f"Unique Vocabulary:         {val_activities:,} abstract tokens (ACT_XXX)")
        print(f"Average Trace Length:      {val_avg_length} events/trace")
        print("=" * 80)

        if error_log:
            print(f"\n[WARNING] {len(error_log)} logs generated errors and were skipped.")