# tools/builtin_tools.py
"""
Built in tools used by the agents.

Tools:
- CSV Data Loader. locate and validate a dataset.
- Data Cleaner. basic cleaning and row counts.
- Statistical Analyzer. simple stats and correlations.
- Chart Creator. export basic Plotly charts as HTML.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from crewai.tools import tool
from scipy import stats


@tool("CSV Data Loader")
def load_dataset(dataset_name: str, data_dir: str = "data") -> str:
    """
    Locate a CSV file by name inside a data directory.

    Returns a JSON string with status, path, and basic info.
    """
    base = Path(data_dir)
    if not base.exists():
        return json.dumps(
            {
                "status": "error",
                "message": f"Data directory not found: {base.resolve()}",
            }
        )

    candidates = list(base.rglob("*.csv"))
    matched = [p for p in candidates if dataset_name.lower() in p.name.lower()]

    if not matched:
        return json.dumps(
            {
                "status": "error",
                "message": f"No CSV files matching '{dataset_name}' under {base.resolve()}",
                "searched_files": [str(p) for p in candidates],
            }
        )

    path = matched[0]

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to read CSV at {path}: {e}",
            }
        )

    summary = {
        "status": "ok",
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    return json.dumps(summary)


@tool("Data Cleaner")
def clean_data(data_path: str) -> str:
    """
    Clean dataset and save to a new file.

    Fills numeric missing values with medians and drops duplicate rows.
    Returns a JSON string with clean_path and row counts.
    """
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to read CSV at {data_path}: {e}",
            }
        )

    original = len(df)

    # Fill missing numeric values with median, without chained assignment
    for col in df.select_dtypes(include=[np.number]).columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    clean_path = data_path.replace(".csv", "_cleaned.csv")
    try:
        df.to_csv(clean_path, index=False)
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to write cleaned CSV to {clean_path}: {e}",
            }
        )

    return json.dumps(
        {
            "status": "ok",
            "original_rows": int(original),
            "clean_rows": int(len(df)),
            "clean_path": clean_path,
        }
    )


@tool("Statistical Analyzer")
def analyze_statistics(data_path: str, target: str = "revenue") -> str:
    """
    Run simple statistical analysis on a numeric target column.

    Returns a JSON string with mean, std, trend info, and correlations.
    """
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to read CSV at {data_path}: {e}",
            }
        )

    if target not in df.columns:
        return json.dumps(
            {
                "status": "error",
                "message": f"Target column '{target}' not found in dataset.",
                "available_columns": list(df.columns),
            }
        )

    try:
        mean_val = df[target].mean()
        std_val = df[target].std()

        # Simple time index for trend
        x = np.arange(len(df))
        y = df[target].values
        slope, _, r_val, p_val, _ = stats.linregress(x, y)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = {
            col: float(df[target].corr(df[col]))
            for col in numeric_cols
            if col != target
        }

        result = {
            "status": "ok",
            "target": target,
            "mean": float(mean_val),
            "std": float(std_val),
            "trend": {
                "slope": float(slope),
                "r2": float(r_val**2),
                "p_value": float(p_val),
                "direction": (
                    "increasing"
                    if slope > 0
                    else "decreasing"
                    if slope < 0
                    else "stable"
                ),
            },
            "correlations": corr,
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Error during statistical analysis: {e}",
            }
        )


@tool("Chart Creator")
def create_chart(
    data_path: str,
    chart_type: str = "line",
    x: str = "date",
    y: str = "revenue",
) -> str:
    """
    Create a simple chart and save it as an HTML file.

    Returns a JSON string with the saved chart path.
    """
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to read CSV at {data_path}: {e}",
            }
        )

    if x not in df.columns or y not in df.columns:
        return json.dumps(
            {
                "status": "error",
                "message": f"Columns '{x}' or '{y}' not found in dataset.",
                "available_columns": list(df.columns),
            }
        )

    # Build the figure
    try:
        if chart_type == "line":
            fig = px.line(df, x=x, y=y, title=f"{y} over time")
        elif chart_type == "bar":
            fig = px.bar(df, x=x, y=y, title=f"{y} by {x}")
        else:
            fig = px.scatter(df, x=x, y=y, trendline="ols")
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Error while creating chart: {e}",
            }
        )

    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{chart_type}_{y}.html"

    try:
        fig.write_html(str(path))
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Failed to save chart HTML to {path}: {e}",
            }
        )

    return json.dumps(
        {
            "status": "ok",
            "chart_type": chart_type,
            "x": x,
            "y": y,
            "path": str(path),
        }
    )
