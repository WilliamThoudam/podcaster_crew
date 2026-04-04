from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from app.config import Settings
from app.graph.agents_subgraph import build_agents_subgraph
from app.graph.pulsecast_state import PulsecastState
from app.models.schemas import OpenAIChatCompletionRequest
from app.services.pulsecast_completion_steps import (
    collect_user_queries,
    phase_planning,
    phase_sub_questions,
    phase_web_search_hitl,
)
from app.services.pulsecast_completion_types import (
    CompletionStreamComplete,
    CompletionStreamOutcome,
    CompletionStreamPaused,
)
from app.services.pulsecast_session_store import PulsecastSession, pulsecast_session_store
from app.services.pulsecast_sse_emit import emit_progress

_OUTER_GRAPH_NODES = frozenset({"planning", "sub_questions", "web_search", "agents"})


def _is_tracked_graph_node(name: str) -> bool:
    if name in _OUTER_GRAPH_NODES:
        return True
    if name.startswith("agents:") and not name.endswith(":__end__"):
        return True
    return False


async def _node_planning(state: PulsecastState, config: RunnableConfig) -> dict[str, Any]:
    c = config["configurable"]
    q, hp, ap = await phase_planning(
        settings=c["settings"],
        req=c["req"],
        on_progress=c.get("on_progress"),
    )
    sid = (c.get("session_id") or "").strip()
    persist = c.get("persist_pulsecast_session")
    if sid and persist:
        session = PulsecastSession(
            session_id=sid,
            raw_user_turns=collect_user_queries(c["req"]),
            canonical_question=q,
            host_plan=hp,
            analyst_plan=ap,
            completed_phase="after_planning",
        )
        await persist(session)
    return {"question": q, "host_plan": hp, "analyst_plan": ap}


async def _node_sub_questions(state: PulsecastState, config: RunnableConfig) -> dict[str, Any]:
    c = config["configurable"]
    result = await phase_sub_questions(
        settings=c["settings"],
        req=c["req"],
        question=state["question"],
        host_plan=state["host_plan"],
        analyst_plan=state["analyst_plan"],
        on_progress=c.get("on_progress"),
        persist_session=c.get("persist_pulsecast_session"),
        session_id=c.get("session_id"),
    )
    if isinstance(result, CompletionStreamPaused):
        return {"outcome": result}
    sr, psql, pexe = result
    return {"sub_results": sr, "primary_sql": psql, "primary_exe": pexe}


async def _node_web_search(state: PulsecastState, config: RunnableConfig) -> dict[str, Any]:
    c = config["configurable"]
    result = await phase_web_search_hitl(
        settings=c["settings"],
        req=c["req"],
        question=state["question"],
        host_plan=state["host_plan"],
        analyst_plan=state["analyst_plan"],
        primary_sql=state["primary_sql"],
        primary_exe=state["primary_exe"],
        sub_results=state["sub_results"],
        on_progress=c.get("on_progress"),
    )
    if isinstance(result, CompletionStreamPaused):
        return {"outcome": result}
    completed_web_results, declined = result
    sid = (c.get("session_id") or "").strip()
    persist = c.get("persist_pulsecast_session")
    if sid and persist:
        prev = await pulsecast_session_store.get(sid)
        if prev is not None:
            await persist(
                prev.model_copy(
                    update={
                        "completed_web_results": list(completed_web_results),
                        "user_declined_web_search": bool(declined),
                        "completed_phase": "after_web",
                    }
                )
            )
    return {"completed_web_results": completed_web_results, "user_declined_web_search": declined}


def _route_after_web_search(state: PulsecastState) -> str:
    out = state.get("outcome")
    if isinstance(out, CompletionStreamPaused):
        return "end"
    return "agents"


def _route_after_sub_questions(state: PulsecastState) -> str:
    out = state.get("outcome")
    if isinstance(out, CompletionStreamPaused):
        return "end"
    return "web_search"


def _build_workflow() -> StateGraph:
    agents_compiled = build_agents_subgraph().compile()
    workflow = StateGraph(PulsecastState)
    workflow.add_node("planning", _node_planning)
    workflow.add_node("sub_questions", _node_sub_questions)
    workflow.add_node("web_search", _node_web_search)
    workflow.add_node("agents", agents_compiled)
    workflow.set_entry_point("planning")
    workflow.add_edge("planning", "sub_questions")
    workflow.add_conditional_edges(
        "sub_questions",
        _route_after_sub_questions,
        {"end": END, "web_search": "web_search"},
    )
    workflow.add_conditional_edges(
        "web_search",
        _route_after_web_search,
        {"end": END, "agents": "agents"},
    )
    workflow.add_edge("agents", END)
    return workflow


_compiled_graph: Any = None


def get_compiled_graph() -> Any:
    """Return the compiled graph, creating it on first call.

    If a PostgresSaver checkpointer has been initialised (via ``init_postgres``),
    it is attached to the graph so every run is checkpointed by thread_id.
    """
    global _compiled_graph
    if _compiled_graph is None:
        from app.services.postgres import get_checkpointer

        checkpointer = get_checkpointer()
        workflow = _build_workflow()
        _compiled_graph = workflow.compile(checkpointer=checkpointer)
    return _compiled_graph


def reset_compiled_graph() -> None:
    """Force recompilation (e.g. after the checkpointer is initialised at startup)."""
    global _compiled_graph
    _compiled_graph = None


async def run_pulsecast_completion_graph(
    *,
    settings: Settings,
    req: OpenAIChatCompletionRequest,
    on_progress: Any | None,
) -> CompletionStreamOutcome:
    graph = get_compiled_graph()
    sid = (req.user or "").strip()

    async def persist_pulsecast_session(session: PulsecastSession) -> None:
        await pulsecast_session_store.save(session)

    config: dict[str, Any] = {
        "configurable": {
            "settings": settings,
            "req": req,
            "on_progress": on_progress,
            "session_id": sid if sid else None,
            "persist_pulsecast_session": persist_pulsecast_session if sid else None,
        },
    }
    if sid:
        config["configurable"]["thread_id"] = sid

    final_state: dict[str, Any] = {}
    async for event in graph.astream_events({}, config=config, version="v2"):
        kind = event["event"]
        name = event.get("name", "")

        if kind == "on_chain_start" and _is_tracked_graph_node(name):
            await emit_progress(on_progress, {
                "type": "graph_node_entered",
                "node": name,
            })
        elif kind == "on_chain_end" and _is_tracked_graph_node(name):
            await emit_progress(on_progress, {
                "type": "graph_node_exited",
                "node": name,
            })
            output = event.get("data", {}).get("output")
            if isinstance(output, dict):
                final_state.update(output)

    out = final_state.get("outcome")
    if out is None:
        raise RuntimeError("Pulsecast graph finished without an outcome")

    persist = config["configurable"].get("persist_pulsecast_session")
    if sid and persist and isinstance(out, CompletionStreamComplete):
        prev = await pulsecast_session_store.get(sid)
        if prev is not None:
            await persist(prev.model_copy(update={"completed_phase": "after_agents"}))

    return out
