from langgraph.graph import END, START, StateGraph

from podcaster_graph.nodes import audio_node, reporting_node, research_node, scripting_node
from podcaster_graph.state import PodcasterState


def build_graph():
    """Sequential graph matching CrewAI process=sequential."""
    g = StateGraph(PodcasterState)
    g.add_node("research", research_node)
    g.add_node("reporting", reporting_node)
    g.add_node("scripting", scripting_node)
    g.add_node("audio", audio_node)
    g.add_edge(START, "research")
    g.add_edge("research", "reporting")
    g.add_edge("reporting", "scripting")
    g.add_edge("scripting", "audio")
    g.add_edge("audio", END)
    return g.compile()
