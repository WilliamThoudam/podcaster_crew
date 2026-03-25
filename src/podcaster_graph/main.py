#!/usr/bin/env python
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

from podcaster_graph.graph import build_graph


def _graph_inputs() -> dict[str, str]:
    """Same kickoff inputs as Crew podcaster.main._crew_inputs."""
    return {
        "topic": os.getenv("TOPIC") or "",
        "current_month": str(datetime.now().month),
        "current_year": str(datetime.now().year),
        "male_host": os.getenv("MALE_HOST", "Jone"),
        "female_host": os.getenv("FEMALE_HOST", "Jane"),
    }


def run() -> dict:
    """
    Run the LangGraph podcaster pipeline (multi-agent, sequential).
    """
    graph = build_graph()
    inputs = _graph_inputs()
    try:
        return graph.invoke(inputs)
    except Exception as e:
        raise RuntimeError(f"An error occurred while running the LangGraph podcaster: {e}") from e


if __name__ == "__main__":
    result = run()
    print(result.get("audio_path", result))
