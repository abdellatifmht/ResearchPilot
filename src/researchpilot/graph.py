from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from researchpilot.agents.analyst import analysis_agent
from researchpilot.agents.researcher import research_agent
from researchpilot.agents.writer import writer_agent


class ResearchPilotState(TypedDict):
    query: str
    language: str
    output_format: Literal["markdown", "md"] | None
    research_output: str
    sources: list[dict[str, Any]]
    analysis_report: dict[str, Any]
    writer_result: dict[str, Any]


def run_researcher(state: ResearchPilotState) -> dict[str, Any]:
    """Run research and preserve its summary and source records."""
    try:
        result = research_agent(state["query"])

        return {
            "research_output": result.get("summary", ""),
            "sources": result.get("sources") or [],
            "language": result.get("language", state.get("language", "en")),
        }
    except Exception as exc:
        raise RuntimeError("Research stage failed.") from exc


def run_analysis(state: ResearchPilotState) -> dict[str, Any]:
    """Analyze the collected sources in the requested language."""
    sources = state.get("sources") or []

    if not sources:
        raise ValueError(
            "Analysis cannot proceed because the researcher returned no sources."
        )

    try:
        result = analysis_agent(
            sources,
            language=state.get("language", "en"),
        )

        return {"analysis_report": result}

    except Exception as exc:
        raise RuntimeError("Analysis stage failed.") from exc


def run_writer(state: ResearchPilotState) -> dict[str, Any]:
    """Generate and validate the final report."""
    sources = state.get("sources") or []
    analysis = state.get("analysis_report") or {}

    if not sources:
        raise ValueError(
            "Report generation cannot proceed because no sources were collected."
        )

    if not analysis:
        raise ValueError(
            "Report generation cannot proceed because the analysis is empty."
        )

    try:
        result = writer_agent(
            analysis=analysis,
            sources=sources,
            language=state.get("language", "en"),
            user_query=state["query"],
            output_format=state.get("output_format", "markdown"),
        )

        return {"writer_result": result}

    except Exception as exc:
        raise RuntimeError("Writing stage failed.") from exc


def build_graph():
    """Build the complete ResearchPilot workflow."""
    builder = StateGraph(ResearchPilotState)

    builder.add_node("researcher", run_researcher)
    builder.add_node("analyst", run_analysis)
    builder.add_node("writer", run_writer)

    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "analyst")
    builder.add_edge("analyst", "writer")
    builder.add_edge("writer", END)

    return builder.compile()


research_pilot_graph = build_graph()


def run_research_pilot(
    query: str,
    language: str = "en",
    output_format: Literal["markdown", "md"] | None = "markdown",
) -> dict[str, Any]:
    """Run research, analysis, and report generation."""
    if not query or not query.strip():
        raise ValueError("Research query cannot be empty.")

    initial_state: ResearchPilotState = {
        "query": query.strip(),
        "language": language,
        "output_format": output_format,
        "research_output": "",
        "sources": [],
        "analysis_report": {},
        "writer_result": {},
    }

    result = research_pilot_graph.invoke(initial_state)

    writer_result = result["writer_result"]

    return {
        "query": result["query"],
        "language": result["language"],
        "research_output": result["research_output"],
        "sources": result["sources"],
        "analysis_report": result["analysis_report"],
        "report": writer_result.get("report", ""),
        "citation_registry": writer_result.get("registry", []),
        "validation": writer_result.get("validation", {}),
        "export_path": writer_result.get("export_path"),
    }
