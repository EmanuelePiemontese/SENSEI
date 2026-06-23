import os
import pandas as pd
import pm4py
import multiprocessing
import statistics
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
OUTPUT_DIR_RAW = os.path.join(SCRIPT_DIR, "..", "results", "EDA_Synthetic")
# The absolute path to the directory where CSVs will be saved
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR_RAW)

def get_statistics(file_path):
    """Read a single XES file and calculate its statistics."""
    try:
        log = pm4py.read_xes(file_path, return_legacy_log_object = True)
        
        traces_list = []
        activities_set = set()
        variants_set = set()
        
        for trace in log:
            trace_acts = []
            for event in trace:
                if isinstance(event, str):
                    act = event
                else:
                    act = event.get("concept:name") or event.get("activityNameEN") or event.get("activity") or "UNK"
                
                trace_acts.append(str(act))
                activities_set.add(str(act))
                
            if trace_acts:
                traces_list.append(trace_acts)
                variants_set.add(tuple(trace_acts))
                
        n_traces = len(traces_list)
        if n_traces == 0:
            return file_path, None
            
        lengths = [len(t) for t in traces_list]
        
        stats = {
            "Log_Name": os.path.basename(file_path),
            "Traces": n_traces,
            "Min_Length": min(lengths),
            "Med_Length": statistics.median(lengths),
            "Max_Length": max(lengths),
            "Events": sum(lengths),
            "Activities": len(activities_set),
            "Variants": len(variants_set)
        }
        return file_path, stats
        
    except Exception as e:
        # Error handling: return the file path and the error message
        return file_path, str(e)

def is_pareto(variants, traces):
    """Pareto principle check: if the number of variants is less than 5% of the number of traces, return "Pareto", else "No Pareto"."""
    if traces == 0:
        return "N/A"
    return "Pareto" if (variants / traces) < 0.05 else "No Pareto"


if __name__ == "__main__":
    # --- OUTPUT DIRECTORY CHECK ---
    # Ensure the target results directory exists before saving anything
    os.makedirs(OUTPUT_DIR, exist_ok = True)
    
    # --- FILE SEARCH ---
    total_files = []
    for root, dirs, files in os.walk(BASE_PATH):
        for f in files:
            if f.endswith(".xes"):
                total_files.append(os.path.join(root, f))

    num_cores = multiprocessing.cpu_count()
    print(f"\n[INFO] Found {len(total_files)} XES files. Starting precise extraction (CPU: {num_cores} Cores)...")

    # --- DATA STRUCTURES ---
    results_categories = {
        "log synthetic PLG": [],
        "logs_complex_medium_ws2": [],
        "logs_complex_simple_ws2": [],
        "logs_complex_ws": [],
        "logs_berti": []
    }
    log_errors = []

    # --- MULTI-CORE EXECUTION ---
    with ProcessPoolExecutor(max_workers = num_cores) as executor:
        futures = {executor.submit(get_statistics, path): path for path in total_files}
        
        for future in tqdm(as_completed(futures), total = len(total_files), desc = "Multi-Core Processing"):
            path, result = future.result()
            
            if result is None:
                continue
            if isinstance(result, str):
                log_errors.append(f"{os.path.basename(path)}: {result}")
                continue
                
            file_name = os.path.basename(path)
            
            # Pareto calculation
            result["Pareto"] = is_pareto(result["Variants"], result["Traces"])
            
            # --- EXACT SORTING (STRICT DIRECTORY MATCH) ---
            path_parts = path.split(os.sep)
            
            if "log synthetic PLG" in path_parts:
                results_categories["log synthetic PLG"].append(result)
            elif "logs_complex_medium_ws2" in path_parts:
                results_categories["logs_complex_medium_ws2"].append(result)
            elif "logs_complex_simple_ws2" in path_parts:
                results_categories["logs_complex_simple_ws2"].append(result)
            elif "logs_complex_ws" in path_parts:
                results_categories["logs_complex_ws"].append(result)
            elif "logs_berti" in path_parts:
                results_categories["logs_berti"].append(result)

    # --- DETAILED ERROR PRINTING ---
    if log_errors:
        print(f"\n[WARNING] Errors occurred while reading {len(log_errors)} files.")
        print(f"[EXAMPLE ERROR] {log_errors[0]}")

    # --- EXPORT AND RENDERING ---
    final_columns = ["Log_Name", "Traces", "Min_Length", "Med_Length", "Max_Length", "Events", "Activities", "Variants", "Pareto"]

    csv_file_names = {
        "log synthetic PLG": "statistics_synthetic_PLG.csv",
        "logs_berti": "statistics_berti.csv",
        "logs_complex_medium_ws2": "statistics_complex_medium.csv",
        "logs_complex_simple_ws2": "statistics_complex_simple.csv",
        "logs_complex_ws": "statistics_complex_ws.csv"
    }

    for category, stats_list in results_categories.items():
        print(f"\n--- Category: {category} ---")
        
        if len(stats_list) > 0:
            df_category = pd.DataFrame(stats_list)
            df_category = df_category[final_columns]
            
            # Alphabetical sorting by Log_Name
            df_category = df_category.sort_values(by = "Log_Name")
            
            # Save to CSV in the specified OUTPUT_DIR
            csv_name = csv_file_names[category]
            full_csv_path = os.path.join(OUTPUT_DIR, csv_name)
            df_category.to_csv(full_csv_path, index = False)
            
            print(f"Analyzed {len(df_category)} logs. Fully saved at: {full_csv_path}")
            
            # Print to screen (limited)
            print(df_category.head(15).to_string(index = False))
            
            if len(df_category) > 15:
                print(f"\n... Showing only the first 15 rows out of {len(df_category)}. Open the CSV file to see them all ...")
        else:
            print("No logs analyzed for this category.")

    print(f"\n[SUCCESS] Processing completed! All CSV files are ready in: {OUTPUT_DIR}")