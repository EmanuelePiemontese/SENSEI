"""
===============================================================================
Statistical Comparison of Models
===============================================================================

This script performs the statistical comparison of multiple models
evaluated on the same collection of event logs.

The following analyses are performed:

1. Friedman Test
2. Average Rank Computation
3. Best Model per Dataset
4. Nemenyi Post-hoc Test
5. Critical Difference Diagram

===============================================================================
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Any

from scipy.stats import friedmanchisquare, rankdata
import scikit_posthocs as sp
from aeon.visualisation import plot_critical_difference


def print_header(title: str) -> None:
    """Prints a formatted header for output sections."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_dataset(file_path: str, sheet_name: str, models: List[str]) -> pd.DataFrame:
    """Loads the dataset from an Excel file and handles missing files."""
    print_header("Loading Benchmark")

    try:
        df = pd.read_excel(file_path, sheet_name = sheet_name)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error reading the excel file: {e}")
        sys.exit(1)

    print(f"Benchmark          : {sheet_name}")
    print(f"Number of datasets : {len(df)}")
    print(f"Models             : {len(models)}")

    print("\nFirst rows:\n")
    print(df.head())

    return df


def friedman_test(df: pd.DataFrame, models: List[str], alpha: float) -> Tuple[float, float]:
    """Performs the Friedman Test to check for overall statistical differences."""
    print_header("Friedman Test")

    stat, p = friedmanchisquare(*[df[m].values for m in models])

    print(f"Statistic : {stat:.5f}")
    print(f"P-value   : {p:.8f}")

    if p < alpha:
        print("\nResult: Reject H0 (significant differences exist)")
    else:
        print("\nResult: Fail to reject H0")

    return stat, p


def compute_ranks(df: pd.DataFrame, models: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Computes the average rank for each model across all datasets."""
    print_header("Average Ranks")

    # rank per dataset (higher score = better -> rankdata(-x))
    ranks = df[models].apply(
        lambda row: rankdata(-row),
        axis = 1,
        result_type = "expand"
    )

    ranks.columns = models
    avg_ranks = ranks.mean().sort_values()

    print(avg_ranks.to_frame(name = "Average Rank"))

    return ranks, avg_ranks


def best_models(df: pd.DataFrame, models: List[str]) -> None:
    """Identifies and prints the top-performing model for each dataset."""
    print_header("Best Model per Dataset")

    winners = df[models].idxmax(axis = 1)
    
    # Assumption: The first column contains the dataset names
    dataset_names = df.iloc[:, 0]

    for dataset, winner in zip(dataset_names, winners):
        print(f"{dataset:<35} {winner}")


def nemenyi_test(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    """Performs the Nemenyi Post-hoc Test for pairwise comparisons."""
    print_header("Nemenyi Post-hoc Test")

    p_values = sp.posthoc_nemenyi_friedman(df[models])

    print(p_values.round(4))

    return p_values


def plot_cd_diagram(df: pd.DataFrame, models: List[str], sheet_name: str, alpha: float) -> None:
    """Plots and saves the Critical Difference Diagram using aeon."""
    print_header("Critical Difference Diagram")

    data = df[models].values
    labels = models

    try:
        result = plot_critical_difference(
            data,
            labels,
            alpha = alpha,
            test = "nemenyi"
        )

        # FIX: aeon returns (fig, ax)
        if isinstance(result, tuple):
            fig = result[0]
        else:
            fig = result

        output_file = f"CD_{sheet_name}.png"
        fig.savefig(output_file, dpi = 300, bbox_inches = "tight")

        print(f"Saved: {output_file}")
        plt.show()

    except Exception as e:
        print(f"Error generating Critical Difference Diagram: {e}")


def main() -> None:
    # Default Configuration
    default_file = "Unforzen_NAP_OP.xlsx"
    default_sheet = "NAP"
    default_models = [
        "AC_Seq_PMI",
        "Google_BERT",
        "MoE",
        "Sensei"
    ]
    default_alpha = 0.05
    
    # CLI Argument Parsing setup
    parser = argparse.ArgumentParser(description = "Statistical Comparison of Process Mining Language Models")
    parser.add_argument("--file", type = str, default = default_file, help = "Path to the Excel file")
    parser.add_argument("--sheet", type = str, default = default_sheet, help = "Name of the Excel sheet to read")
    parser.add_argument("--alpha", type = float, default = default_alpha, help = "Significance level (alpha)")
    
    args = parser.parse_args()

    # Variables extraction
    file_path = args.file
    sheet_name = args.sheet
    alpha = args.alpha
    models = default_models

    df = load_dataset(file_path, sheet_name, models)

    friedman_test(df, models, alpha)
    ranks, avg_ranks = compute_ranks(df, models)
    best_models(df, models)
    nemenyi_test(df, models)
    plot_cd_diagram(df, models, sheet_name, alpha)


if __name__ == "__main__":
    main()