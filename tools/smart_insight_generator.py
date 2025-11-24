# tools/smart_insight_generator.py
"""
Custom tool that turns numeric analysis results into short business insights.
"""

import json
from datetime import datetime

from crewai.tools import tool


@tool("Smart Insight Generator")
def generate_insights(analysis_results: str) -> str:
    """
    Convert analysis JSON into a short insight and action summary.

    Accepts a JSON string or dict from Statistical Analyzer or similar tools.
    """
    # Parse input into a dict
    if isinstance(analysis_results, str):
        try:
            data = json.loads(analysis_results)
        except Exception:
            data = {"raw": analysis_results}
    else:
        data = analysis_results

    insights = []
    recommendations = []

    # If upstream reported an error, note it and return early
    if data.get("status") == "error":
        msg = data.get("message", "Unknown analysis error.")
        insights.append(f"Analysis tool reported an error: {msg}")
        actions = [
            "Check dataset path and column names.",
            "Re run the analysis after fixing data issues.",
        ]
        actions_block = "\n".join(
            [f"  {i+1}. {a}" for i, a in enumerate(actions)]
        )

        report = f"""
ANALYSIS INSIGHTS | {datetime.now().strftime('%Y-%m-%d %H:%M')}

FINDINGS:
  - {insights[0]}

ACTIONS:
{actions_block}

Confidence: 40 percent | Quality: Limited (tool error)
"""
        return report

    # Trend insight
    trend = data.get("trend")
    if trend:
        direction = trend.get("direction", "stable")
        slope = trend.get("slope", 0.0)
        insights.append(
            f"Revenue trend is {direction} with slope {slope:.4f} over the period."
        )
        if direction.startswith("decreas"):
            recommendations.append(
                "HIGH: Investigate the reasons for the downward revenue trend."
            )

    # Correlation insight
    correlations = data.get("correlations", {})
    if correlations:
        top = max(correlations.items(), key=lambda x: abs(x[1]))
        col, val = top
        insights.append(
            f"Strongest correlation is between revenue and {col}: {val:.2f}."
        )
        if abs(val) > 0.7:
            recommendations.append(
                "LOW: Use this strong relationship in forecasting and planning."
            )

    # Segment or category insight if provided
    segments = data.get("segments")
    if segments:
        best = max(segments.items(), key=lambda x: x[1])
        worst = min(segments.items(), key=lambda x: x[1])
        gap = ((best[1] - worst[1]) / max(worst[1], 1e-6)) * 100
        insights.append(
            f"{best[0]} segment leads, {worst[0]} lags, with about {gap:.0f} percent gap."
        )
        recommendations.append(
            f"MEDIUM: Build a focused improvement plan for the {worst[0]} segment."
        )

    if not insights:
        insights.append("No structured analysis fields were found in the input.")
        recommendations.append(
            "LOW: Run the full statistical analysis before acting on this data."
        )

    actions_block = "\n".join(
        [f"  {i+1}. {r}" for i, r in enumerate(recommendations)]
    ) or "  1. LOW: No specific actions identified."

    report = f"""
ANALYSIS INSIGHTS | {datetime.now().strftime('%Y-%m-%d %H:%M')}

FINDINGS:
{chr(10).join([f'  - {i}' for i in insights])}

ACTIONS:
{actions_block}

Confidence: 85 percent | Quality: Good
"""
    return report
