from __future__ import annotations

from fastapi import APIRouter, Response

from app.graph.pulsecast_graph import get_compiled_graph

router = APIRouter(tags=["graph"])

_CONDITION_LABELS: dict[tuple[str, str], str] = {
    ("sub_questions", "__end__"): "paused",
    ("web_search", "__end__"): "paused",
    ("agents:init_agents", "agents:depth_gate"): "needs gate",
    ("agents:init_agents", "agents:analyst"): "continue",
    ("agents:analyst", "agents:host_finalize"): "minimal",
    ("agents:analyst", "agents:marketing"): "direct panel",
    ("agents:forecaster", "agents:web_crawler"): "round 1",
    ("agents:forecaster", "agents:__end__"): "paused",
    ("agents:web_crawler", "agents:challenger"): "continue",
    ("agents:web_crawler", "agents:__end__"): "web paused",
    ("agents:challenger", "agents:host_finalize"): "done",
    ("agents:challenger", "agents:round_gate"): "next round",
    ("agents:challenger", "agents:__end__"): "sql paused",
    ("agents:round_gate", "agents:marketing"): "continue",
    ("agents:round_gate", "agents:__end__"): "paused",
}


def _topology_node_type(node_id: str) -> str:
    if node_id == "__start__":
        return "start"
    if node_id == "__end__" or node_id.endswith(":__end__"):
        return "end"
    nid = node_id
    if nid.endswith(":depth_gate") or nid.endswith(":round_gate"):
        return "gate"
    if nid.startswith("agents:"):
        return "agent"
    return "node"


def _edge_condition_label(source: str, target: str) -> str | None:
    key = (source, target)
    if key in _CONDITION_LABELS:
        return _CONDITION_LABELS[key]
    if source.startswith("agents:") and (target == "__end__" or target.endswith(":__end__")):
        return "paused"
    return "continue"


@router.get("/v1/graph/topology")
async def graph_topology(response: Response):
    response.headers["Cache-Control"] = "public, max-age=3600"

    compiled = get_compiled_graph()
    drawable = compiled.get_graph(xray=True)

    nodes = []
    for node_id in drawable.nodes:
        nodes.append({"id": node_id, "type": _topology_node_type(node_id)})

    edges = []
    for edge in drawable.edges:
        src = edge.source
        tgt = edge.target
        conditional = edge.conditional
        label = _edge_condition_label(src, tgt) if conditional else None
        edges.append({
            "source": src,
            "target": tgt,
            "conditional": conditional,
            "condition_label": label,
        })

    return {"nodes": nodes, "edges": edges}
