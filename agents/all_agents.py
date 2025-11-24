# agents/all_agents.py
"""
Agent factory functions for the data analyst system.
Each function returns a configured CrewAI Agent that shares the same Gemini LLM.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, LLM

load_dotenv()

# Read API key from .env and expose it for CrewAI's Gemini provider
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

# os.environ["GEMINI_API_KEY"] = api_key

# Shared LLM configuration used by all agents
llm = LLM(
    model="openrouter/x-ai/grok-4.1-fast:free",  # can be changed to a different Gemini model if needed
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,
    verbose=True,
)


def create_controller_agent(tools):
    """High level controller that plans the workflow and delegates work."""
    return Agent(
        role="Analysis Controller",
        goal="Plan the workflow, coordinate agents, and ensure the main questions are fully answered.",
        backstory="Senior analytics lead used to coordinating multi step projects.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=True,
        memory=True,
        respect_context_window=True,
        max_iter=2,
    )


def create_data_loader_agent(tools):
    """Agent that locates and validates the dataset."""
    return Agent(
        role="Data Loader",
        goal="Find the correct dataset and make sure it is readable before analysis.",
        backstory="Works with data lakes and local folders to locate files.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=True,
        respect_context_window=True,
        max_iter=2,
    )


def create_data_validator_agent(tools):
    """Agent that cleans data and reports basic quality metrics."""
    return Agent(
        role="Data Quality Engineer",
        goal="Clean the dataset and report basic quality checks.",
        backstory="Experienced in ETL and data quality work.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=True,
        respect_context_window=True,
        max_iter=2,
    )


def create_exploratory_agent(tools):
    """Agent that does exploratory data analysis."""
    return Agent(
        role="Exploratory Analyst",
        goal="Describe high level patterns, trends, and segment behavior.",
        backstory="Comfortable slicing data by time and category.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=True,
        respect_context_window=True,
        max_iter=2,
    )


def create_statistical_agent(tools):
    """Agent that runs core statistical analysis."""
    return Agent(
        role="Statistician",
        goal="Compute trends, correlations, and basic statistical tests.",
        backstory="Statistician who can explain results in simple language.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=True,
        respect_context_window=True,
        max_iter=2,
    )


def create_anomaly_detector_agent(tools):
    """Agent that focuses on anomalies and outliers."""
    return Agent(
        role="Anomaly Specialist",
        goal="Find unusual values and suggest possible causes.",
        backstory="Background in anomaly and fraud detection.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=True,
        respect_context_window=True,
        max_iter=2,
    )


def create_visualization_agent(tools):
    """Agent that creates charts and short visual summaries."""
    return Agent(
        role="Visualization Designer",
        goal="Create simple charts and describe what they show.",
        backstory="Builds dashboards and reports for business users.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=True,
        respect_context_window=True,
        max_iter=2,
    )


def create_report_generator_agent(tools):
    """Agent that writes the final executive report."""
    return Agent(
        role="Business Analyst",
        goal=(
            "Turn all previous findings into a clear executive report "
            "without mentioning internal tools or system errors."
        ),
        backstory="Used to presenting analysis to leadership.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=True,
        respect_context_window=True,
        max_iter=3,
    )
