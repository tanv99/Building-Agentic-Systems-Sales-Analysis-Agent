"""
Entry point for the Q3 2024 sales analysis agentic workflow.

Steps:
1. Load environment and verify OpenRouter API key.
2. Create agents and tools.
3. Define tasks for loading, cleaning, analysis, and reporting.
4. Run the Crew and save the final report.
5. Provide run_analysis() so evaluation.py can execute multiple test cases.
6. Use previous evaluation metrics as feedback for the controller agent.
"""

import os
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from crewai import Crew, Process, Task  # noqa: E402
from agents.all_agents import (  # noqa: E402
    create_anomaly_detector_agent,
    create_controller_agent,
    create_data_loader_agent,
    create_data_validator_agent,
    create_exploratory_agent,
    create_report_generator_agent,
    create_statistical_agent,
    create_visualization_agent,
)
from tools.builtin_tools import (  # noqa: E402
    analyze_statistics,
    clean_data,
    create_chart,
    load_dataset,
)
from tools.smart_insight_generator import generate_insights  # noqa: E402


# --------------------------------------------------------------------
# Environment and working directory setup
# --------------------------------------------------------------------

# Set working directory to script location
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Load env variables
load_dotenv(override=True)

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("ERROR. OPENROUTER_API_KEY not set")
    exit(1)

print("✓ API key loaded")


# --------------------------------------------------------------------
# Dataset configuration
# --------------------------------------------------------------------

DATASET_HINT = "ecommerce_q3_2024"
DATA_DIR = "data"
RAW_FILE = f"{DATA_DIR}/{DATASET_HINT}.csv"

# Optional override used by evaluation.py to point to variant datasets
_raw_override = os.getenv("EVAL_RAW_FILE_OVERRIDE")
if _raw_override:
    RAW_FILE = _raw_override


# --------------------------------------------------------------------
# Feedback loop. use past evaluation metrics as context
# --------------------------------------------------------------------

def load_feedback_summary(
    log_path: str = "outputs/eval/metrics.jsonl",
    max_runs: int = 5,
) -> str:
    """
    Read recent evaluation runs and produce a short feedback summary
    that the controller agent can use to improve the workflow.

    This is a simple feedback loop across runs.
    It does not change code automatically. it gives the controller
    information about success rates and typical failures.
    """
    path = Path(log_path)
    if not path.exists():
        return (
            "No prior evaluation runs found. "
            "You are running the workflow for the first time."
        )

    records = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    except Exception:
        return (
            "Evaluation log exists but could not be parsed. "
            "Assume there may have been past failures."
        )

    if not records:
        return "Evaluation log is empty. No feedback available."

    # Keep only the most recent runs
    records = sorted(records, key=lambda r: r.get("start_time", ""))[-max_runs:]

    total = len(records)
    successes = sum(1 for r in records if r.get("success"))
    success_rate = successes / total if total else 0.0

    # Collect a few distinct error messages
    errors = [
        r.get("error_message")
        for r in records
        if not r.get("success") and r.get("error_message")
    ]
    unique_errors = []
    for e_msg in errors:
        if e_msg not in unique_errors:
            unique_errors.append(e_msg)
    unique_errors = unique_errors[:3]

    summary_lines = [
        f"Recent runs analyzed. {total} total, success rate {success_rate:.2f}.",
    ]

    if unique_errors:
        summary_lines.append("Common recent errors.")
        for idx, msg in enumerate(unique_errors, start=1):
            short_msg = msg[:140] + "..." if len(msg) > 140 else msg
            summary_lines.append(f"  {idx}. {short_msg}")
    else:
        summary_lines.append("No recurring runtime errors detected in recent runs.")

    return "\n".join(summary_lines)


# Main business questions
query = f"""
Analyze Q3 2024 sales data from {DATASET_HINT}.csv:
1. Why did revenue drop in September?
2. Which categories underperformed?
3. What can we say about marketing spend vs revenue?
4. Are there any clear anomalies?
5. What should we do in Q4?
"""

# Feedback context passed into the controller task
feedback_context = load_feedback_summary()


# --------------------------------------------------------------------
# Agent creation
# --------------------------------------------------------------------

controller = create_controller_agent(
    tools=[load_dataset, generate_insights, clean_data, analyze_statistics, create_chart]
)
loader = create_data_loader_agent([load_dataset])
validator = create_data_validator_agent([clean_data])
exploratory = create_exploratory_agent([analyze_statistics, create_chart])
statistical = create_statistical_agent([analyze_statistics])
anomaly = create_anomaly_detector_agent([analyze_statistics])
visualizer = create_visualization_agent([create_chart, generate_insights])
reporter = create_report_generator_agent([generate_insights])


# --------------------------------------------------------------------
# Task definitions
# --------------------------------------------------------------------

t0 = Task(
    description=(
        "You are the controller agent for this analysis.\n"
        f"Main business questions.\n{query}\n"
        "You also receive feedback from previous evaluation runs.\n"
        f"{feedback_context}\n\n"
        "Your goals.\n"
        "1. Summarize key objectives for this run.\n"
        "2. Outline the order in which specialized agents should act and why.\n"
        "3. Highlight data quality or tooling risks that other agents should watch for.\n"
        "4. If recent runs had failures, adjust the workflow plan to reduce the same errors.\n"
    ),
    agent=controller,
    expected_output="Short workflow plan that references feedback when relevant.",
)

t1 = Task(
    description=(
        f"Locate the dataset for Q3 2024 ecommerce analysis.\n"
        f"- Search under '{DATA_DIR}' for a CSV containing the name '{DATASET_HINT}'.\n"
        "- Use the CSV Data Loader tool.\n"
        "- If multiple files match, prefer the one with the most rows.\n"
        "- Return JSON with status, path, rows, and columns.\n"
    ),
    agent=loader,
    expected_output="JSON with status, path, rows, and columns.",
)

t2 = Task(
    description=(
        "Clean the dataset.\n"
        "- Use the path returned by the Data Loader task if available. "
        f"Otherwise fall back to '{RAW_FILE}'.\n"
        "- Handle missing values and remove duplicates.\n"
        "- Return JSON with original_rows, clean_rows, and clean_path.\n"
    ),
    agent=validator,
    expected_output="JSON with original_rows, clean_rows, and clean_path.",
)

t3 = Task(
    description=(
        "Run exploratory data analysis on the cleaned dataset.\n"
        "- Focus on revenue by month and category.\n"
        "- Describe high level patterns in plain language.\n"
        "- Point out any obvious shifts across July, August, and September.\n"
    ),
    agent=exploratory,
    expected_output="Narrative summary of key patterns.",
)

t4 = Task(
    description=(
        "Use the Statistical Analyzer tool on the cleaned dataset.\n"
        "- Target column should be revenue.\n"
        "- Report mean, standard deviation, and trend information.\n"
        "- Highlight any strong correlations with other numeric columns.\n"
        "- Return the JSON result for possible downstream use.\n"
    ),
    agent=statistical,
    expected_output="JSON with statistics and correlations.",
)

t5 = Task(
    description=(
        "Identify anomalies using the exploratory context and any statistical output.\n"
        "- Pay special attention to September compared to July and August.\n"
        "- Describe any obvious outliers or strange values.\n"
        "- Suggest possible causes in simple terms.\n"
    ),
    agent=anomaly,
    expected_output="Short anomaly report with likely causes.",
)

t6 = Task(
    description=(
        "Create visualizations and, if possible, convert stats to short insights.\n"
        "- Use the cleaned dataset.\n"
        "- Create at least two charts.\n"
        "  1) Revenue by category over time.\n"
        "  2) Marketing vs revenue or another useful comparison.\n"
        "- Save charts to outputs/visualizations.\n"
        "- If you have JSON stats, you may call the Smart Insight Generator "
        "to produce a short text block of insights.\n"
    ),
    agent=visualizer,
    expected_output="List of chart paths and any generated insight text.",
)

t7 = Task(
    description=(
        "Write the final executive report.\n"
        "Requirements.\n"
        "1) Explain why revenue dropped in September, based on the data.\n"
        "2) List which categories underperformed.\n"
        "3) Summarize what we can say about marketing spend vs revenue.\n"
        "4) Note any clear anomalies.\n"
        "5) Give concrete, business friendly recommendations for Q4.\n"
        "- If JSON stats are available, you may use the Smart Insight Generator.\n"
        "- Do not mention internal tools, agent names, or system errors.\n"
        "- Write for a VP of Sales who wants clear, concise answers.\n"
    ),
    agent=reporter,
    expected_output="Executive style report that answers all five points.",
)


# --------------------------------------------------------------------
# Crew definition
# --------------------------------------------------------------------

crew = Crew(
    agents=[
        controller,
        loader,
        validator,
        exploratory,
        statistical,
        anomaly,
        visualizer,
        reporter,
    ],
    tasks=[t0, t1, t2, t3, t4, t5, t6, t7],
    process=Process.sequential,
    verbose=True,
    memory=False,  # per agent memory stays in the agent factory config
)


# --------------------------------------------------------------------
# Run helper. used by evaluation.py
# --------------------------------------------------------------------

def run_analysis(run_tag: str = "baseline") -> dict:
    """
    Execute full workflow once and return evaluation metrics.
    Appends metrics to outputs/eval/metrics.jsonl.
    """

    print("\n🚀 Starting analysis...")
    start = datetime.now()

    success = True
    error_msg = None
    result_text = ""

    try:
        result = crew.kickoff()
        result_text = str(result)
    except Exception as e:
        success = False
        error_msg = str(e)
        result_text = f"ERROR DURING RUN. {error_msg}"

    end = datetime.now()
    duration = (end - start).total_seconds()

    # Save report
    os.makedirs("outputs/reports", exist_ok=True)
    timestamp = end.strftime("%Y%m%d_%H%M%S")
    report_path = f"outputs/reports/report_{timestamp}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(result_text)

    print(f"\n✓ Completed in {duration:.1f}s")
    print(f"✓ Report saved to {report_path}")

    metrics = {
        "run_tag": run_tag,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "duration_seconds": duration,
        "report_file": report_path,
        "success": success,
        "error_message": error_msg,
    }

    os.makedirs("outputs/eval", exist_ok=True)
    metrics_log = "outputs/eval/metrics.jsonl"
    with open(metrics_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")

    return metrics


# --------------------------------------------------------------------
# Script entry point
# --------------------------------------------------------------------

if __name__ == "__main__":
    run_analysis()
