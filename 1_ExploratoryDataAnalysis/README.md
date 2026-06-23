# 1. Exploratory Data Analysis (EDA)

This directory contains the scripts dedicated to the initial Exploratory Data Analysis (EDA) of the process logs. The objective is to extract data statistics, analyze log characteristics, and evaluate metrics across both synthetic pre-training datasets and real test datasets.

## Directory structure

```text
1_ExploratoryDataAnalysis/
├── README.md
├── Real_test_log_EDA.py
└── Synthetic_pretraining_log_EDA.py

```

---

## Scripts overview

### 1. `Synthetic_pretraining_log_EDA.py`

This script focuses on processing raw event logs in `.xes` format intended for the pre-training phase.

* **input path:** reads files from `../data/Dataset_Pre_training`
* **features:** * leverages multi-core processing (`ProcessPoolExecutor`) for parallel parsing of heavy XES structures.
    * **automatically categorizes logs into specific groups (e.g., `log synthetic PLG`, `logs_berti`, `logs_complex_ws`, etc.) based on strict directory matching.
    * **computes standard process mining metrics:** Traces, Activities, Events, Variants, and Min/Median/Max trace lengths.
    * **evaluates the Paretto Principle:** checks if the unique variants count is less than 5% of the total traces.

* **output:** saves comprehensive summary tables for each category as CSV files inside `../results/EDA_Synthetic/`.

### 2. `Real_test_log_EDA.py`

This script handles the exploratory analysis of specific target evaluation logs provided in `.csv` format, which are used during the evaluation or fine-tuning phase.

* **input path:** reads files from `../data/Dataset_Test`
* **features:**
    * dynamically identifies the CSV column separator (detecting commas `,` vs semicolons `;`).
    * automatically discovers and standardizes key process mining structural attributes (Case ID, Activity Name, and Timestamp).
    * sorts events chronologically per case to ensure reliable variant extraction.
    * measures global metrics including total trace counts, unique activities, trace length distributions, and absolute variant counts.


* **output:** displays an aggregated tabular report in the console and saves the consolidated clean metrics to `../results/EDA_Test/statistics_test_dataset.csv`.

---

## How to run the scripts

Ensure you have initialized and activated your virtual environment (`venv_SENSEI`) and installed all the necessary dependencies before execution.

### Step 1: navigate to the project root folder

Always execute the python commands from the main repository directory:

```bash
cd "SENSEI"

```

### Step 2: activate the virtual environment

Activate your environment depending on your operating system:

* **Windows (PowerShell):**
```powershell
venv_SENSEI\Scripts\Activate.ps1

```


* **macOS / Linux:**
```bash
source venv_SENSEI/bin/activate

```



### Step 3: run the scripts

To execute the synthetic pre-training log analysis:

```bash
python "1_ExploratoryDataAnalysis/Synthetic_pretraining_log_EDA.py"

```

To execute the real test log analysis:

```bash
python "1_ExploratoryDataAnalysis/Real_test_log_EDA.py"

```

---

## Outputs and generated artifacts
Upon successful completion, the execution results will be automatically organized within the global `results/` folder:

* **Synthetic Outputs:** look into `results/EDA_Synthetic/` for detailed `statistics_*.csv` files tailored to each specific log category.
* **Test Dataset Outputs:** look into `results/EDA_Test/` for the global `statistics_test_dataset.csv` benchmark file.