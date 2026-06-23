import os
import time
import json
import pandas as pd
import ijson
import gzip
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- PATH CONFIGURATION ---
# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# All files for this step are located in the results/Pre_processing folder from the previous script
PRE_PROCESSING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results", "Pre_processing_pre_training"))

INPUT_PATH = os.path.join(PRE_PROCESSING_DIR, "processed_pretraining_dataset.json")
OUTPUT_PATH = os.path.join(PRE_PROCESSING_DIR, "processed_pretraining_dataset_prefixes.jsonl.gz")
CSV_OUTPUT_PATH = os.path.join(PRE_PROCESSING_DIR, "statistics_pretraining_prefixes.csv")

def generate_prefixes_streaming_fast_and_safe(input_path, output_path, csv_output_path):
    """
    Generate all possible prefixes for each trace in the pre-processed dataset and save them in a compressed JSONL file. Additionally, compute and save statistics about the prefixes in a CSV file.
    """
    start_time = time.time()

    if not os.path.exists(input_path):
        print(f"[ERROR] The file {input_path} does not exist. Please run the previous Pre-processing script first.")
        return

    stats_list = []
    global_vocabulary = set()
    total_traces_global = 0
    total_events_global = 0
    total_prefixes_global = 0

    # Automatic detection of maximum hardware power
    n_cores = os.cpu_count()
    print(f"\n[INFO] Hardware detected: {n_cores} CPU cores available.")

    # Using gzip to compress the output dynamically and save massive amounts of disk space
    with open(input_path, "r", encoding = "utf-8") as f_in, \
         gzip.open(output_path, "wt", encoding = "utf-8", compresslevel = 4) as f_out:

        print(f"[INFO] Starting STREAMING read and COMPRESSED write to: {output_path}\n")
        
        lines_buffer = []
        BUFFER_SIZE = 50000 
        
        # Using ijson to stream the JSON file without overloading the RAM
        parser = ijson.kvitems(f_in, "")
        
        for log_name, traces in tqdm(parser, desc = "Prefix Generation"):
            if not traces:
                continue
                
            log_traces_count = len(traces)
            log_events_count = 0
            log_prefixes_count = 0
            
            min_len = float("inf")
            max_len = 0
            sum_len = 0
            log_vocabulary = set()

            for trace in traces:
                tokens = trace.split()
                n_tokens = len(tokens)
                
                if n_tokens == 0:
                    continue
                
                if n_tokens < min_len: 
                    min_len = n_tokens
                if n_tokens > max_len: 
                    max_len = n_tokens
                sum_len += n_tokens
                
                log_vocabulary.update(tokens)
                log_events_count += n_tokens
                log_prefixes_count += n_tokens
                
                prefix_string = tokens[0]
                
                # Double curly braces {{ }} are used to escape brackets in f-strings to format valid JSON
                lines_buffer.append(f"{{\"log\": \"{log_name}\", \"prefix\": \"{prefix_string}\"}}\n")
                
                for i in range(1, n_tokens):
                    prefix_string += " " + tokens[i]
                    lines_buffer.append(f"{{\"log\": \"{log_name}\", \"prefix\": \"{prefix_string}\"}}\n")
                    
                # Write to disk in chunks to free up memory
                if len(lines_buffer) >= BUFFER_SIZE:
                    f_out.writelines(lines_buffer)
                    lines_buffer.clear()
        
            global_vocabulary.update(log_vocabulary)
            total_traces_global += log_traces_count
            total_events_global += log_events_count
            total_prefixes_global += log_prefixes_count
            
            avg_len = round(sum_len / log_traces_count, 2) if log_traces_count > 0 else 0
            
            stats_list.append({
                "Log_Name": log_name,
                "Original_Traces": log_traces_count,
                "Total_Events": log_events_count,
                "Generated_Prefixes": log_prefixes_count,
                "Min_Length": min_len if min_len != float("inf") else 0,
                "Max_Length": max_len,
                "Avg_Length": avg_len,
                "Unique_Activities": len(log_vocabulary)
            })

        # Write any remaining lines in the buffer
        if lines_buffer:
            f_out.writelines(lines_buffer)

    # --- CSV STATISTICS ---
    df_stats = pd.DataFrame(stats_list)
    df_stats = df_stats.sort_values(by = "Log_Name").reset_index(drop = True)
    
    global_row = pd.DataFrame([{
        "Log_Name": "GLOBAL_TOTAL",
        "Original_Traces": total_traces_global,
        "Total_Events": total_events_global,
        "Generated_Prefixes": total_prefixes_global,
        "Min_Length": df_stats["Min_Length"].min(),
        "Max_Length": df_stats["Max_Length"].max(),
        "Avg_Length": round(total_events_global / total_traces_global, 2) if total_traces_global > 0 else 0,
        "Unique_Activities": len(global_vocabulary)
    }])
    
    df_to_save = pd.concat([df_stats, global_row], ignore_index = True)
    df_to_save.to_csv(csv_output_path, index = False)
    
    print(f"\n[SUCCESS] Operation completed in {(time.time() - start_time) / 60:.2f} minutes. File successfully compressed.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    generate_prefixes_streaming_fast_and_safe(INPUT_PATH, OUTPUT_PATH, CSV_OUTPUT_PATH)