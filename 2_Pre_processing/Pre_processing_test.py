import os
import pandas as pd
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- PATH CONFIGURATION ---
# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input directory for the Test Datasets
BASE_PATH_RAW = os.path.join(SCRIPT_DIR, "..", "data", "Dataset_Test")
BASE_PATH = os.path.abspath(BASE_PATH_RAW)

# Output directories for NAP and OP
OUTPUT_DIR_NAP = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results", "Pre_processing_test", "NAP"))
OUTPUT_DIR_OP = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results", "Pre_processing_test", "OP"))

# CSV Statistics Output paths
CSV_NAP = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results", "Pre_processing_test", "statistics_nap_prefix.csv"))
CSV_OP = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results", "Pre_processing_test", "statistics_op_prefix.csv"))

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

# ==========================================
# NEXT ACTIVITY PREDICTION (NAP) FUNCTION
# ==========================================
def prepare_nap_data(csv_path, out_nap_dir):
    """Prepare the dataset for Next Activity Prediction (NAP) by extracting prefixes and their corresponding next activities.
    The function reads a CSV file, processes it, and saves the results in a serialized pickle format."""
    filename = os.path.basename(csv_path)
    base_name = filename.replace(".csv", "")
    
    try:
        # 1. Separator detection and loading
        sep = ";"
        with open(csv_path, "r", encoding = "utf-8") as f:
            first_line = f.readline()
            if "," in first_line and ";" not in first_line: 
                sep = ","
            
        df = pd.read_csv(csv_path, sep = sep, low_memory = False)
        df.columns = [c.strip().lower() for c in df.columns]
        
        act_col = next((c for c in df.columns if c in ["activity", "concept:name", "activity code", "activity_code"]), None)
        case_col = next((c for c in df.columns if c in ["case id", "case_id", "case:concept:name"]), df.columns[0])
        time_col = next((c for c in df.columns if c in ["time:timestamp", "timestamp", "complete timestamp", "time"]), None)
        
        if not act_col:
            return filename, "Missing Activity column.", None
            
        # Stable chronological sorting
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], utc = True, errors = "coerce")
            df = df.sort_values(by = [case_col, time_col], kind = "mergesort")
        else:
            sort_col = "event_nr" if "event_nr" in df.columns else df.index
            df = df.sort_values(by = [case_col, sort_col], kind = "mergesort")
            
        df = df.dropna(subset = [act_col])
        df[act_col] = df[act_col].astype(str)
        
        # Abstract symbolic mapping
        counts = df[act_col].value_counts()
        mapping = {act: f"ACT_{i + 1:03d}" for i, act in enumerate(counts.index)}
        
        # 2. CHRONOLOGICAL TEMPORAL SPLIT (66% Train / 34% Test)
        if time_col:
            start_timestamps = df.groupby(case_col)[time_col].min().reset_index()
            start_timestamps = start_timestamps.sort_values(by = time_col, ascending = True, kind = "mergesort")
        else:
            start_timestamps = pd.DataFrame({case_col: df[case_col].unique()})

        split_idx = int(0.66 * len(start_timestamps))
        train_cases = start_timestamps[case_col].iloc[:split_idx].tolist()
        test_cases = start_timestamps[case_col].iloc[split_idx:].tolist()
        
        # 3. EXTRACT PREFIXES AS LISTS (Stop at N-1)
        def extract_nap_prefixes(case_list):
            X_abs, X_txt = [], []
            y_abs, y_txt = [], []
            
            subset_df = df[df[case_col].isin(case_list)]
            
            for _, group in subset_df.groupby(case_col):
                acts_seq = [mapping[act] for act in group[act_col]]
                txt_seq = [str(act) for act in group[act_col]]
                
                N = len(acts_seq)
                # Stop at N-1: the last event has no real "next activity"
                for i in range(1, N):
                    X_abs.append(acts_seq[:i])
                    X_txt.append(txt_seq[:i])
                    
                    y_abs.append(acts_seq[i])
                    y_txt.append(txt_seq[i])
                    
            return X_abs, X_txt, y_abs, y_txt

        X_tr_abs, X_tr_txt, y_tr_abs, y_tr_txt = extract_nap_prefixes(train_cases)
        X_te_abs, X_te_txt, y_te_abs, y_te_txt = extract_nap_prefixes(test_cases)
        
        # 4. Serialized Pickle Saving
        nap_data = {
            "X_train_abs": X_tr_abs, "X_train_txt": X_tr_txt,
            "y_train_abs": y_tr_abs, "y_train_txt": y_tr_txt,
            "X_test_abs": X_te_abs, "X_test_txt": X_te_txt,
            "y_test_abs": y_te_abs, "y_test_txt": y_te_txt,
            "activity_map": mapping
        }
        
        output_file = os.path.join(out_nap_dir, f"{base_name}_NAP.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(nap_data, f, protocol = pickle.HIGHEST_PROTOCOL)
            
        stats = {
            "Log": filename,
            "Total_Cases": len(start_timestamps),
            "Train_Prefixes": len(X_tr_abs),
            "Test_Prefixes": len(X_te_abs),
            "Unique_Activities": len(mapping)
        }
        return filename, None, stats

    except Exception as e:
        return filename, str(e), None

# ==========================================
# OUTCOME PREDICTION (OP) FUNCTION
# ==========================================
def prepare_op_data(csv_path, out_op_dir):
    """Prepare the dataset for Outcome Prediction (OP) by extracting prefixes and their corresponding outcomes."""
    filename = os.path.basename(csv_path)
    base_name = filename.replace(".csv", "")
    
    try:
        # 1. Separator detection and loading
        sep = ";"
        with open(csv_path, "r", encoding = "utf-8") as f:
            first_line = f.readline()
            if "," in first_line and ";" not in first_line: 
                sep = ","
            
        df = pd.read_csv(csv_path, sep = sep, low_memory = False)
        df.columns = [c.strip().lower() for c in df.columns]
        
        act_col = next((c for c in df.columns if c in ["activity", "concept:name", "activity code", "activity_code"]), None)
        case_col = next((c for c in df.columns if c in ["case id", "case_id", "case:concept:name"]), df.columns[0])
        time_col = next((c for c in df.columns if c in ["time:timestamp", "timestamp", "complete timestamp", "time"]), None)
        label_col = next((c for c in df.columns if c in ["label", "outcome", "class", "target"]), None)
        
        if not act_col or not label_col:
            return filename, "Missing structural columns or Label.", None
            
        # Stable chronological sorting
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], utc = True, errors = "coerce")
            df = df.sort_values(by = [case_col, time_col], kind = "mergesort")
        else:
            sort_col = "event_nr" if "event_nr" in df.columns else df.index
            df = df.sort_values(by = [case_col, sort_col], kind = "mergesort")
            
        df = df.dropna(subset = [act_col, label_col])
        df[act_col] = df[act_col].astype(str)
        
        # Symbolic mapping and label mapping
        counts = df[act_col].value_counts()
        mapping = {act: f"ACT_{i + 1:03d}" for i, act in enumerate(counts.index)}
        
        label_map = {
            "regular": 0,
            "deviant": 1
        }
        
        # 2. IDENTICAL CHRONOLOGICAL TEMPORAL SPLIT (66% Train / 34% Test)
        if time_col:
            start_timestamps = df.groupby(case_col)[time_col].min().reset_index()
            start_timestamps = start_timestamps.sort_values(by = time_col, ascending = True, kind = "mergesort")
        else:
            start_timestamps = pd.DataFrame({case_col: df[case_col].unique()})

        split_idx = int(0.66 * len(start_timestamps))
        train_cases = start_timestamps[case_col].iloc[:split_idx].tolist()
        test_cases = start_timestamps[case_col].iloc[split_idx:].tolist()
        
        # 3. EXTRACT PREFIXES ALIGNED TO THE NAP MODEL (Up to N-1)
        def extract_op_prefixes(case_list):
            X_abs, X_txt = [], []
            y_op = []
            
            subset_df = df[df[case_col].isin(case_list)]
            
            for _, group in subset_df.groupby(case_col):
                acts_seq = [mapping[act] for act in group[act_col]]
                txt_seq = [str(act) for act in group[act_col]]
                
                outcome_str = str(group[label_col].iloc[0])
                outcome_idx = label_map.get(outcome_str, 0)
                
                N = len(acts_seq)
                # We use range(1, N) to keep the prefixes identical to those extracted for NAP
                for i in range(1, N):
                    X_abs.append(acts_seq[:i])
                    X_txt.append(txt_seq[:i])
                    
                    y_op.append(outcome_idx)
                    
            return X_abs, X_txt, y_op

        X_tr_abs, X_tr_txt, y_tr_op = extract_op_prefixes(train_cases)
        X_te_abs, X_te_txt, y_te_op = extract_op_prefixes(test_cases)
        
        # 4. Serialized Pickle Saving
        op_data = {
            "X_train_abs": X_tr_abs, "X_train_txt": X_tr_txt,
            "y_train": y_tr_op,
            "X_test_abs": X_te_abs, "X_test_txt": X_te_txt,
            "y_test": y_te_op,
            "activity_map": mapping,
            "outcome_map": label_map
        }
        
        output_file = os.path.join(out_op_dir, f"{base_name}_OP.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(op_data, f, protocol = pickle.HIGHEST_PROTOCOL)
            
        stats = {
            "Log": filename,
            "Total_Cases": len(start_timestamps),
            "Train_Prefixes": len(X_tr_abs),
            "Test_Prefixes": len(X_te_abs),
            "Unique_Outcomes": len(label_map)
        }
        return filename, None, stats

    except Exception as e:
        return filename, str(e), None

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # Ensure all required output directories exist
    os.makedirs(OUTPUT_DIR_NAP, exist_ok = True)
    os.makedirs(OUTPUT_DIR_OP, exist_ok = True)

    # Find all files to process
    files_to_process = []
    for root, _, files in os.walk(BASE_PATH):
        for f in files:
            if f in TARGET_FILES:
                files_to_process.append(os.path.join(root, f))

    if not files_to_process:
        print(f"[ERROR] No target files found in {BASE_PATH}")
    else:
        # --- EXECUTE NAP PHASE ---
        print(f"\n[INFO] Starting parallel NAP processing for {len(files_to_process)} files...\n")
        
        summary_data_nap = []
        with ProcessPoolExecutor(max_workers = 4) as executor:
            futures_nap = {executor.submit(prepare_nap_data, path, OUTPUT_DIR_NAP): path for path in files_to_process}
            for future in tqdm(as_completed(futures_nap), total = len(files_to_process), desc = "NAP Extraction"):
                name, err, stats = future.result()
                if not err:
                    summary_data_nap.append(stats)
                else:
                    print(f"[ERROR] NAP Failed for {name}: {err}")

        df_summary_nap = pd.DataFrame(summary_data_nap).sort_values("Log").reset_index(drop = True)
        df_summary_nap.to_csv(CSV_NAP, index = False)
        
        print("\n" + "=" * 80)
        print(f"{'PRE-PROCESSING REPORT: NEXT ACTIVITY PREDICTION':^80}")
        print("=" * 80)
        print(df_summary_nap.to_string(index = False))
        
        # --- EXECUTE OP PHASE ---
        print(f"\n[INFO] Starting parallel OP processing for {len(files_to_process)} files...\n")
        
        summary_data_op = []
        with ProcessPoolExecutor(max_workers = 4) as executor:
            futures_op = {executor.submit(prepare_op_data, path, OUTPUT_DIR_OP): path for path in files_to_process}
            for future in tqdm(as_completed(futures_op), total = len(files_to_process), desc = "OP Extraction"):
                name, err, stats = future.result()
                if not err:
                    summary_data_op.append(stats)
                else:
                    print(f"[ERROR] OP Failed for {name}: {err}")

        df_summary_op = pd.DataFrame(summary_data_op).sort_values("Log").reset_index(drop = True)
        df_summary_op.to_csv(CSV_OP, index = False)
        
        print("\n" + "=" * 80)
        print(f"{'PRE-PROCESSING REPORT: OUTCOME PREDICTION':^80}")
        print("=" * 80)
        print(df_summary_op.to_string(index = False))
        
        print(f"\n[SUCCESS] NAP and OP preprocessing completed. Data saved in {os.path.dirname(OUTPUT_DIR_NAP)}")