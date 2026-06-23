import os
import pandas as pd

# --- PATH CONFIGURATION ---
# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the base path to the raw data directory
BASE_PATH_RAW = os.path.join(SCRIPT_DIR, "..", "data", "Dataset_Fine_tuning_Test")
# The absolute path to the base directory containing the CSV files
BASE_PATH = os.path.abspath(BASE_PATH_RAW)

# Construct the output path to the specific results subdirectory
OUTPUT_DIR_RAW = os.path.join(SCRIPT_DIR, "..", "results", "EDA_Real_Test")
# The absolute path to the directory where CSVs will be saved
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR_RAW)

# Exact list of files intended for Fine-Tuning (Test)
TARGET_FILES = [
    "BPIC11_f1.csv", "BPIC11_f2.csv", "BPIC11_f3.csv", "BPIC11_f4.csv",
    "BPIC15_1_f2.csv", "BPIC15_2_f2.csv", "BPIC15_3_f2.csv", "BPIC15_4_f2.csv", "BPIC15_5_f2.csv",
    "BPIC17_O_Accepted.csv", "BPIC17_O_Cancelled.csv", "BPIC17_O_Refused.csv",
    "Production.csv",
    "bpic2012_O_ACCEPTED-COMPLETE.csv", "bpic2012_O_CANCELLED-COMPLETE.csv", "bpic2012_O_DECLINED-COMPLETE.csv",
    "hospital_billing_2.csv", "hospital_billing_3.csv",
    "sepsis_cases_1.csv", "sepsis_cases_2.csv", "sepsis_cases_4.csv",
    "traffic_fines_1.csv"
]

def get_complete_statistics(csv_path):
    """Read a single CSV file and calculate its complete statistics."""
    filename = os.path.basename(csv_path)
    
    try:
        # 1. Dynamic separator identification
        sep = ";"
        with open(csv_path, "r", encoding = "utf-8") as f:
            first_line = f.readline()
            if "," in first_line and ";" not in first_line: 
                sep = ","
            
        # 2. Reading and standardizing columns
        df = pd.read_csv(csv_path, sep = sep, low_memory = False)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # 3. Key Columns Identification (Case, Activity, Time)
        act_col = next((c for c in df.columns if c in ["activity", "concept:name", "activity code", "activity_code"]), None)
        case_col = next((c for c in df.columns if c in ["case id", "case_id", "case:concept:name"]), df.columns[0])
        time_col = next((c for c in df.columns if c in ["time:timestamp", "timestamp", "complete timestamp", "time"]), None)
        
        if not act_col or not case_col:
            return {"Log": filename, "Error": "Case ID or Activity columns missing"}
            
        # 4. Null values cleanup and casting
        df = df.dropna(subset = [act_col, case_col])
        df[act_col] = df[act_col].astype(str)
        
        # 5. Temporal sorting (Crucial for calculating variants correctly)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], utc = True, errors = "coerce")
            df = df.sort_values(by = [case_col, time_col])
        else:
            sort_col = "event_nr" if "event_nr" in df.columns else df.index
            df = df.sort_values(by = [case_col, sort_col])

        # 6. Global Metrics Calculation
        num_events = len(df)
        num_activities = df[act_col].nunique()
        
        # 7. Grouping by Case ID for metrics on Traces and Variants
        grouped = df.groupby(case_col)[act_col]
        
        num_traces = grouped.ngroups
        lengths = grouped.size()
        
        min_length = lengths.min()
        max_length = lengths.max()
        avg_length = round(lengths.mean(), 2)
        
        # Variant calculation (transforms sequence into a tuple and counts unique tuples)
        num_variants = grouped.apply(tuple).nunique()
        
        return {
            "Log": filename,
            "#Traces": num_traces,
            "#Activities": num_activities,
            "Min length": min_length,
            "Avg length": avg_length,
            "Max length": max_length,
            "#Variants": num_variants,
            "#Events": num_events,
            "Error": "None"
        }

    except Exception as e:
        return {"Log": filename, "Error": str(e)}

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # Ensure the target results directory exists before saving anything
    os.makedirs(OUTPUT_DIR, exist_ok = True)
    
    print("\n[INFO] Starting scan to calculate advanced log statistics...\n")

    results = []

    for root, _, files in os.walk(BASE_PATH):
        for f in files:
            if f in TARGET_FILES:
                file_path = os.path.join(root, f)
                stats = get_complete_statistics(file_path)
                results.append(stats)

    # --- FINAL REPORT ---
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by = "Log").reset_index(drop = True)
        
        # Separate logs with errors
        df_errors = df_results[df_results["Error"] != "None"]
        df_clean = df_results[df_results["Error"] == "None"].drop(columns = ["Error"])
        
        print("=" * 100)
        header_text = "COMPLETE STATISTICAL REPORT - TEST DATASET"
        print(f"{header_text:^100}")
        print("=" * 100)
        
        if not df_clean.empty:
            print(df_clean.to_string(index = False))
            
            # Save to CSV in the specified OUTPUT_DIR
            csv_name = "statistics_test_dataset.csv"
            full_csv_path = os.path.join(OUTPUT_DIR, csv_name)
            df_clean.to_csv(full_csv_path, index = False)
            
            print("\n" + "=" * 100)
            print(f"[SUCCESS] Analyzed {len(df_clean)} logs. Results successfully saved to: {full_csv_path}")
        else:
            print("\n" + "=" * 100)
        
        # Print Errors
        if not df_errors.empty:
            print("\n[WARNING] The following logs encountered errors:")
            for _, row in df_errors.iterrows():
                # Extracting values to variables to avoid using single quotes inside the f-string
                log_name = row["Log"]
                error_msg = row["Error"]
                print(f"- {log_name}: {error_msg}")

    else:
        print("[ERROR] No files processed. Check the input path and the TARGET_FILES list.")