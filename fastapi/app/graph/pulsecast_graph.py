from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from app.config import Settings
from app.models.schemas import (
    ExecuteSqlResponse,
    OpenAIChatCompletionRequest,
    PlanningAnalystOutput,
    PlanningHostOutput,
    SubResult,
)
from app.services.pulsecast_completion_steps import (
    phase_agents_finalize,
    phase_planning,
    phase_sub_questions,
    phase_web_search_hitl,
)
from app.services.pulsecast_completion_types import CompletionStreamOutcome, CompletionStreamPaused


class PulsecastState(TypedDict, total=False):
    question: str
    host_plan: PlanningHostOutput
    analyst_plan: PlanningAnalystOutput
    sub_results: list[SubResult]
    primary_sql: str
    primary_exe: ExecuteSqlResponse
    completed_web_results: list[dict[str, Any]]
    user_declined_web_search: bool
    outcome: CompletionStreamOutcome


async def _node_planning(state: PulsecastState, config: RunnableConfig) -> dict[str, Any]:
    c = config["configurable"]
    q, hp, ap = await phase_planning(
        settings=c["settings"],
        req=c["req"],
        on_progress=c.get("on_progress"),
    )
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


async def _node_agents(state: PulsecastState, config: RunnableConfig) -> dict[str, Any]:
    c = config["configurable"]
    out = await phase_agents_finalize(
        settings=c["settings"],
        req=c["req"],
        question=state["question"],
        host_plan=state["host_plan"],
        analyst_plan=state["analyst_plan"],
        primary_sql=state["primary_sql"],
        primary_exe=state["primary_exe"],
        sub_results=state["sub_results"],
        completed_web_results=state.get("completed_web_results"),
        user_declined_web_search=bool(state.get("user_declined_web_search", False)),
        on_progress=c.get("on_progress"),
    )
    return {"outcome": out}


_compiled_graph: Any = None


def _get_compiled_graph() -> Any:
    global _compiled_graph
    if _compiled_graph is None:
        workflow = StateGraph(PulsecastState)
        workflow.add_node("planning", _node_planning)
        workflow.add_node("sub_questions", _node_sub_questions)
        workflow.add_node("web_search", _node_web_search)
        workflow.add_node("agents", _node_agents)
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
        _compiled_graph = workflow.compile()
    return _compiled_graph


async def run_pulsecast_completion_graph(
    *,
    settings: Settings,
    req: OpenAIChatCompletionRequest,
    on_progress: Any | None,
) -> CompletionStreamOutcome:
    graph = _get_compiled_graph()
    final = await graph.ainvoke(
        {},
        config={
            "configurable": {
                "settings": settings,
                "req": req,
                "on_progress": on_progress,
            },
        },
    )
    out = final.get("outcome")
    if out is None:
        raise RuntimeError("Pulsecast graph finished without an outcome")
    return out
