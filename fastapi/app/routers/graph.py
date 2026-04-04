from __future__ import annotations

from fastapi import APIRouter, Response

from app.graph.pulsecast_graph import get_compiled_graph

router = APIRouter(tags=["graph"])

_CONDITION_LABELS: dict[str, str] = {
    "end": "paused",
    "__end__": "paused",
}


@router.get("/v1/graph/topology")
async def graph_topology(response: Response):
    response.headers["Cache-Control"] = "public, max-age=3600"

    compiled = get_compiled_graph()
    drawable = compiled.get_graph()

    nodes = []
    for node_id in drawable.nodes:
        if node_id == "__start__":
            ntype = "start"
        elif node_id == "__end__":
            ntype = "end"
        else:
            ntype = "node"
        nodes.append({"id": node_id, "type": ntype})

    edges = []
    for edge in drawable.edges:
        src = edge.source
        tgt = edge.target
        conditional = edge.conditional
        label = _CONDITION_LABELS.get(tgt, "continue") if conditional else None
        edges.append({
            "source": src,
            "target": tgt,
            "conditional": conditional,
            "condition_label": label,
        })

    return {"nodes": nodes, "edges": edges}
