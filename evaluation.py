"""
Evaluation Suite for the Q3 2024 Agentic System.

Covers:
- Designed test cases
- Metrics collection (accuracy surrogate, efficiency, reliability)
- Behavior analysis over time
"""

import os
import json
from pathlib import Path
import pandas as pd

from main import run_analysis, DATASET_HINT, DATA_DIR, RAW_FILE


EVAL_DIR = Path("outputs/eval")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

BASE_DATASET = Path(RAW_FILE)


# -----------------------------
# TEST CASES
# -----------------------------
def test_case_baseline():
    """Normal baseline execution."""
    return run_analysis(run_tag="baseline")


def test_case_missing_dataset():
    """Simulates missing or deleted dataset."""
    if not BASE_DATASET.exists():
        print("Base dataset missing. Skipping this test.")
        return None

    temp_path = BASE_DATASET.with_suffix(".bak")

    # Temporarily hide dataset
    BASE_DATASET.rename(temp_path)
    try:
        result = run_analysis(run_tag="missing_dataset")
    finally:
        # Restore file
        temp_path.rename(BASE_DATASET)

    return result


def test_case_missing_revenue_column():
    """Creates dataset missing the 'revenue' column."""
    if not BASE_DATASET.exists():
        print("Base dataset missing. Skipping test.")
        return None

    df = pd.read_csv(BASE_DATASET)
    if "revenue" not in df.columns:
        print("Revenue already missing. Skipping test.")
        return None

    # Write modified dataset
    alt_path = Path(DATA_DIR) / f"{DATASET_HINT}_no_revenue.csv"
    df.drop(columns=["revenue"]).to_csv(alt_path, index=False)

    # Override RAW_FILE for this run
    os.environ["EVAL_RAW_FILE_OVERRIDE"] = str(alt_path)

    try:
        result = run_analysis(run_tag="missing_revenue")
    finally:
        del os.environ["EVAL_RAW_FILE_OVERRIDE"]
        alt_path.unlink(missing_ok=True)

    return result


def test_case_large_dataset():
    """Optional test if large dataset exists."""
    large_path = Path(DATA_DIR) / f"{DATASET_HINT}_large.csv"
    if not large_path.exists():
        print("Large dataset missing. Skipping test.")
        return None

    os.environ["EVAL_RAW_FILE_OVERRIDE"] = str(large_path)

    try:
        result = run_analysis(run_tag="large_dataset")
    finally:
        del os.environ["EVAL_RAW_FILE_OVERRIDE"]

    return result


# -----------------------------
# METRICS ANALYSIS
# -----------------------------
def analyze_metrics():
    """Aggregate metrics.jsonl and provide summary statistics."""
    log_path = EVAL_DIR / "metrics.jsonl"
    if not log_path.exists():
        print("No metrics file. Run tests first.")
        return

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("No records found.")
        return

    df = pd.DataFrame(records)
    print("\n====================")
    print(" EVALUATION SUMMARY ")
    print("====================")

    print("\nTotal runs:", len(df))
    print("\nRuns by tag:")
    print(df["run_tag"].value_counts())

    print("\nSuccess Rate by tag:")
    print(df.groupby("run_tag")["success"].mean())

    print("\nAverage Duration by tag (seconds):")
    print(df.groupby("run_tag")["duration_seconds"].mean())

    print("\nRecent runs:")
    print(df.sort_values("start_time").tail(5)[
        ["run_tag", "start_time", "duration_seconds", "success"]
    ])


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("\n=== Running Evaluation Test Cases ===")
    test_case_baseline()
    test_case_missing_dataset()
    test_case_missing_revenue_column()
    test_case_large_dataset()

    print("\n=== Metrics Summary ===")
    analyze_metrics()
