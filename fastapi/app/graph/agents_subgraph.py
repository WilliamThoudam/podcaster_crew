from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from app.graph.pulsecast_state import PulsecastState
from app.models.schemas import AgentInsight, AgentPipelineStep
from app.services.pulsecast_llm_agents import (
    DiscussionState,
    DiscussionTurn,
    DiscussantRole,
    LlmAgentsComplete,
    LlmAgentsPaused,
    LlmAgentsPausedDiscussion,
    LlmAgentsPausedWebSearch,
    PulsecastRole,
    _AgentOut,
    _agent_id,
    _analyst_opening_prompt,
    _checkpoint_discussion_to_session,
    _context_blob_compact,
    _discussion_user_prompt,
    _filter_new_web_queries,
    _host_final_answer_with_canonical,
    _host_user_prompt,
    _human_prompt,
    _merge_completed_web_into_ctx,
    _run_host_json,
    _run_panel_agent_json,
    _run_role_call,
    _run_web_crawler_call,
    _serper_configured,
    _sse_challenger_control,
    _sse_discussion_analyst,
    _sse_discussion_turn,
    _web_search_queries_for_hitl,
    discussion_state_from_legacy_prior,
    system_prompt_host_composer,
    system_prompt_host_composer_minimal,
    system_prompt_internal,
)
from app.services.pulsecast_completion_steps import stream_outcome_from_llm_agents_result
from app.services.pulsecast_sse_emit import emit_progress
from app.services.qa_pipeline import build_answer_summary
from app.services.sub_question_tts_guard import is_valid_tts_sub_question


def _prior_models(prior_json: dict[str, dict[str, Any]]) -> dict[str, _AgentOut]:
    return {k: _AgentOut.model_validate(v) for k, v in prior_json.items()}


def _turns_from_json(raw: list[dict[str, Any]] | None) -> list[DiscussionTurn]:
    return [DiscussionTurn.model_validate(x) for x in (raw or [])]


def _pipeline_steps(steps: list[dict[str, Any]] | None) -> list[AgentPipelineStep]:
    return [AgentPipelineStep.model_validate(x) for x in (steps or [])]


async def _node_init_agents(state: PulsecastState, config: RunnableConfig) -> dict[str, Any]:
    c = config["configurable"]
    settings = c["settings"]
    sr_list = list(state["sub_results"])
    hp, ap = state["host_plan"], state["analyst_plan"]
    det = build_answer_summary(state["primary_exe"])
    ctx = _context_blob_compact(
        question=state["question"],
        generated_sql=state["primary_sql"],
        exe=state["primary_exe"],
        deterministic_summary=det,
        sub_results=sr_list,
        host_plan=hp,
        analyst_plan=ap,
        settings=settings,
    )
    cw = state.get("completed_web_results")
    if cw:
        _merge_completed_web_into_ctx(ctx, [dict(x) for x in cw])
    if state.get("user_declined_web_search"):
        ctx["user_declined_web_search"] = True
        ctx["hitl_note"] = (
            "The user declined to run a public web search (or this step). "
            "Answer using warehouse samples and prior panel notes only; "
            "do not imply external web results were retrieved."
        )
    depth: str = hp.discussion_depth if hp else "moderated"
    if depth == "minimal" and ap and ap.web_sub_questions:
        depth = "linear"
    max_rounds = max(1, settings.pulsecast_discussion_max_rounds)
    needs_pre = bool(depth in ("linear", "moderated") and hp is not None and ap is not None)
    return {
        "ctx": ctx,
        "deterministic_summary": det,
        "discussion_depth": depth,  # type: ignore[typeddict-item]
        "max_rounds": max_rounds,
        "agents_needs_pre_discussion_pause": needs_pre,
        "discussion_turns": [],
        "agents_prior": {},
        "agents_pipeline": [],
        "current_round": 1,
    }


def _route_after_init_agents(state: PulsecastState) -> str:
    if state.get("agents_needs_pre_discussion_pause"):
        return "depth_gate"
    return "analyst"


async def _node_depth_gate(state: PulsecastState, config: RunnableConfig) -> dict[str, Any]:
    c = config["configurable"]
    depth = state["discussion_depth"]
    assert depth in ("linear", "moderated")
    llm_out = LlmAgentsPausedDiscussion(
        stage="pre",
        requested_depth=depth,  # type: ignore[arg-type]
        question=state["question"],
        generated_sql=state["primary_sql"],
        primary_exe=state["primary_exe"],
        deterministic_summary=state["deterministic_summary"],
        sub_results=list(state["sub_results"]),
        host_plan=state["host_plan"],
        analyst_plan=state["analyst_plan"],
        max_rounds=int(state["max_rounds"]),
        next_round_index=1,
        focus_for_next_round=None,
        rationale="Start 1-round panel." if depth == "linear" else "Start multi-round panel.",
    )
    outcome = await stream_outcome_from_llm_agents_result(
        agents_out=llm_out,
        req=c["req"],
        on_progress=c.get("on_progress"),
    )
    return {"outcome": outcome}


async def _node_analyst(state: PulsecastState, config: RunnableConfig) -> dict[str, Any]:
    c = config["configurable"]
    settings = c["settings"]
    on_progress = c.get("on_progress")
    ctx = state["ctx"]
    analyst_out = await _run_panel_agent_json(
        settings,
        "ANALYST",
        system_prompt_internal("ANALYST", discussion_aware=False),
        _analyst_opening_prompt(ctx=ctx),
    )
    step = AgentPipelineStep(
        id=_agent_id("ANALYST"),
        status="completed",
        phase=analyst_out.phase,
        detail=analyst_out.detail,
    )
    pipe = _pipeline_steps(state.get("agents_pipeline"))
    pipe.append(step)
    await _sse_discussion_analyst(on_progress, analyst_out)
    ck = (c.get("session_id") or "").strip() or None
    if ck and state.get("discussion_depth") == "moderated":
        await _checkpoint_discussion_to_session(
            ck,
            DiscussionState(analyst=analyst_out, turns=[]),
            pipe,
        )
    prior = dict(state.get("agents_prior") or {})
    prior["ANALYST"] = analyst_out.model_dump(mode="json")
    return {
        "analyst_output": analyst_out.model_dump(mode="json"),
        "agents_pipeline": [p.model_dump(mode="json") for p in pipe],
        "agents_prior": prior,
    }


def _route_after_analyst(state: PulsecastState) -> str:
    if state.get("discussion_depth") == "minimal":
        return "host_finalize"
    return "marketing"


async def _linear_style_role(
    state: PulsecastState,
    config: RunnableConfig,
    role: PulsecastRole,
) -> dict[str, Any]:
    c = config["configurable"]
    settings = c["settings"]
    on_progress = c.get("on_progress")
    ctx = state["ctx"]
    prior = _prior_models(dict(state.get("agents_prior") or {}))
    out = await _run_role_call(
        settings=settings,
        role=role,
        system_prompt=system_prompt_internal(role, discussion_aware=False),
        ctx=ctx,
        prior=prior,
    )
    prior[role] = out
    pipe = _pipeline_steps(state.get("agents_pipeline"))
    pipe.append(
        AgentPipelineStep(
            id=_agent_id(role),
            status="completed",
            phase=out.phase,
            detail=out.detail,
        )
    )
    await _sse_discussion_turn(on_progress, role, 1, out)
    return {
        "agents_prior": {k: v.model_dump(mode="json") for k, v in prior.items()},
        "agents_pipeline": [p.model_dump(mode="json") for p in pipe],
    }


async def _moderated_discussant(
    state: PulsecastState,
    config: RunnableConfig,
    discussant: DiscussantRole,
    round_index: int,
) -> dict[str, Any]:
    """Run one moderated discussant; may set ``outcome`` for SQL HITL on CHALLENGER."""
    c = config["configurable"]
    settings = c["settings"]
    on_progress = c.get("on_progress")
    ctx = state["ctx"]
    analyst_out = _AgentOut.model_validate(state["analyst_output"])
    turns = _turns_from_json(state.get("discussion_turns"))
    focus = None
    user_content = _discussion_user_prompt(
        ctx=ctx,
        analyst=analyst_out,
        turns=turns,
        role=discussant,
        round_index=round_index,
        focus_for_next_round=focus,
    )
    pr: PulsecastRole = discussant
    out = await _run_panel_agent_json(
        settings,
        pr,
        system_prompt_internal(pr, discussion_aware=True),
        user_content,
    )
    turns.append(DiscussionTurn(role=discussant, round_index=round_index, output=out))
    snippet = (out.detail or out.insight or out.phase or "")[:500]
    pipe = _pipeline_steps(state.get("agents_pipeline"))
    pipe.append(
        AgentPipelineStep(
            id=_agent_id(pr),
            status="completed",
            phase=f"round_{round_index}",
            detail=snippet or None,
        )
    )
    await _sse_discussion_turn(on_progress, pr, round_index, out)
    ck = (c.get("session_id") or "").strip() or None
    if ck:
        await _checkpoint_discussion_to_session(
            ck,
            DiscussionState(analyst=analyst_out, turns=list(turns)),
            list(pipe),
        )
    update: dict[str, Any] = {
        "discussion_turns": [t.model_dump(mode="json") for t in turns],
        "agents_pipeline": [p.model_dump(mode="json") for p in pipe],
    }
    if discussant == "CHALLENGER":
        pq = (out.new_question or "").strip()
        if (
            out.needs_more_data
            and pq
            and is_valid_tts_sub_question(pq)
            and state.get("host_plan") is not None
            and state.get("analyst_plan") is not None
        ):
            hp, ap = state["host_plan"], state["analyst_plan"]
            paused = LlmAgentsPaused(
                pipeline=pipe,
                discussion=DiscussionState(analyst=analyst_out, turns=list(turns)),
                question=state["question"],
                generated_sql=state["primary_sql"],
                primary_exe=state["primary_exe"],
                deterministic_summary=state["deterministic_summary"],
                sub_results=list(state["sub_results"]),
                host_plan=hp,
                analyst_plan=ap,
                proposed_sub_question=pq,
                rationale=out.new_question_rationale,
            )
            outcome = await stream_outcome_from_llm_agents_result(
                agents_out=paused,
                req=c["req"],
                on_progress=on_progress,
            )
            update["outcome"] = outcome
            return update
    return update


async def _node_marketing(state: PulsecastState, config: RunnableConfig) -> dict[str, Any]:
    on_progress = config["configurable"].get("on_progress")
    if state["discussion_depth"] == "moderated":
        if on_progress:
            await emit_progress(on_progress, {"type": "discussion_round_started", "round": state["current_round"]})
        extra = await _moderated_discussant(state, config, "MARKETING", state["current_round"])
        if extra.get("outcome") is not None:
            return extra
        return {k: v for k, v in extra.items() if k != "outcome"}
    return await _linear_style_role(state, config, "MARKETING")


async def _node_finance(state: PulsecastState, config: RunnableConfig) -> dict[str, Any]:
    if state.get("outcome") is not None:
        return {}
    if state["discussion_depth"] == "moderated":
        extra = await _moderated_discussant(state, config, "FINANCE", state["current_round"])
        if extra.get("outcome") is not None:
            return extra
        return {k: v for k, v in extra.items() if k != "outcome"}
    return await _linear_style_role(state, config, "FINANCE")


async def _node_forecaster(state: PulsecastState, config: RunnableConfig) -> dict[str, Any]:
    if state.get("outcome") is not None:
        return {}
    if state["discussion_depth"] == "moderated":
        extra = await _moderated_discussant(state, config, "FORECASTER", state["current_round"])
        if extra.get("outcome") is not None:
            return extra
        return {k: v for k, v in extra.items() if k != "outcome"}
    return await _linear_style_role(state, config, "FORECASTER")


def _route_after_forecaster(state: PulsecastState) -> str:
    if state.get("outcome") is not None:
        return "end"
    return "web_crawler"


async def _node_web_crawler(state: PulsecastState, config: RunnableConfig) -> dict[str, Any]:
    if state.get("outcome") is not None:
        return {}
    c = config["configurable"]
    settings = c["settings"]
    on_progress = c.get("on_progress")
    ctx = state["ctx"]
    hp, ap = state.get("host_plan"), state.get("analyst_plan")

    if state["discussion_depth"] == "moderated":
        analyst_out = _AgentOut.model_validate(state["analyst_output"])
        turns = _turns_from_json(state.get("discussion_turns"))
        prior_wc: dict[str, _AgentOut] = {"ANALYST": analyst_out}
        for t in turns:
            prior_wc[t.role] = t.output
        wc_out = await _run_web_crawler_call(settings=settings, ctx=ctx, prior=prior_wc)
        r = state["current_round"]
        turns.append(DiscussionTurn(role="WEB_CRAWLER", round_index=r, output=wc_out))
        wc_snippet = (wc_out.detail or wc_out.insight or wc_out.phase or "")[:500]
        pipe = _pipeline_steps(state.get("agents_pipeline"))
        pipe.append(
            AgentPipelineStep(
                id=_agent_id("WEB_CRAWLER"),
                status="completed",
                phase=f"round_{r}",
                detail=wc_snippet or None,
            )
        )
        await _sse_discussion_turn(on_progress, "WEB_CRAWLER", r, wc_out)
        ck = (c.get("session_id") or "").strip() or None
        if ck:
            await _checkpoint_discussion_to_session(
                ck,
                DiscussionState(analyst=analyst_out, turns=list(turns)),
                list(pipe),
            )
        if _serper_configured(settings) and hp is not None and ap is not None:
            queries = _web_search_queries_for_hitl(wc_out, ap)
            queries = _filter_new_web_queries(ctx, queries)
            queries = queries[:1]
            if wc_out.needs_web_search and queries:
                paused = LlmAgentsPausedWebSearch(
                    pipeline=pipe,
                    discussion=DiscussionState(analyst=analyst_out, turns=list(turns)),
                    question=state["question"],
                    generated_sql=state["primary_sql"],
                    primary_exe=state["primary_exe"],
                    deterministic_summary=state["deterministic_summary"],
                    sub_results=list(state["sub_results"]),
                    host_plan=hp,
                    analyst_plan=ap,
                    proposed_search_query=queries[0],
                    search_queries=queries,
                    pending_search_index=0,
                    completed_web_results=[],
                    rationale=wc_out.web_search_rationale,
                )
                outcome = await stream_outcome_from_llm_agents_result(
                    agents_out=paused,
                    req=c["req"],
                    on_progress=on_progress,
                )
                return {
                    "discussion_turns": [t.model_dump(mode="json") for t in turns],
                    "agents_pipeline": [p.model_dump(mode="json") for p in pipe],
                    "outcome": outcome,
                }
        return {
            "discussion_turns": [t.model_dump(mode="json") for t in turns],
            "agents_pipeline": [p.model_dump(mode="json") for p in pipe],
        }

    prior = _prior_models(dict(state.get("agents_prior") or {}))
    wc_out = await _run_web_crawler_call(settings=settings, ctx=ctx, prior=prior)
    prior["WEB_CRAWLER"] = wc_out
    pipe = _pipeline_steps(state.get("agents_pipeline"))
    pipe.append(
        AgentPipelineStep(
            id=_agent_id("WEB_CRAWLER"),
            status="completed",
            phase=wc_out.phase,
            detail=wc_out.detail,
        )
    )
    await _sse_discussion_turn(on_progress, "WEB_CRAWLER", 1, wc_out)
    if _serper_configured(settings) and hp is not None and ap is not None:
        queries = _web_search_queries_for_hitl(wc_out, ap)
        queries = _filter_new_web_queries(ctx, queries)
        queries = queries[:1]
        if wc_out.needs_web_search and queries:
            ds = discussion_state_from_legacy_prior(prior)
            if ds is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to build discussion state for web search pause",
                )
            paused = LlmAgentsPausedWebSearch(
                pipeline=pipe,
                discussion=ds,
                question=state["question"],
                generated_sql=state["primary_sql"],
                primary_exe=state["primary_exe"],
                deterministic_summary=state["deterministic_summary"],
                sub_results=list(state["sub_results"]),
                host_plan=hp,
                analyst_plan=ap,
                proposed_search_query=queries[0],
                search_queries=queries,
                pending_search_index=0,
                completed_web_results=[],
                rationale=wc_out.web_search_rationale,
            )
            outcome = await stream_outcome_from_llm_agents_result(
                agents_out=paused,
                req=c["req"],
                on_progress=on_progress,
            )
            return {
                "agents_prior": {k: v.model_dump(mode="json") for k, v in prior.items()},
                "agents_pipeline": [p.model_dump(mode="json") for p in pipe],
                "outcome": outcome,
            }
    return {
        "agents_prior": {k: v.model_dump(mode="json") for k, v in prior.items()},
        "agents_pipeline": [p.model_dump(mode="json") for p in pipe],
    }


def _route_after_web_crawler(state: PulsecastState) -> str:
    if state.get("outcome") is not None:
        return "end"
    return "challenger"


async def _node_challenger(state: PulsecastState, config: RunnableConfig) -> dict[str, Any]:
    if state.get("outcome") is not None:
        return {}
    c = config["configurable"]
    settings = c["settings"]
    on_progress = c.get("on_progress")
    ctx = state["ctx"]
    hp, ap = state.get("host_plan"), state.get("analyst_plan")

    if state["discussion_depth"] == "moderated":
        extra = await _moderated_discussant(state, config, "CHALLENGER", state["current_round"])
        if extra.get("outcome") is not None:
            return extra
        return {k: v for k, v in extra.items() if k != "outcome"}

    prior = _prior_models(dict(state.get("agents_prior") or {}))
    out = await _run_role_call(
        settings=settings,
        role="CHALLENGER",
        system_prompt=system_prompt_internal("CHALLENGER", discussion_aware=False),
        ctx=ctx,
        prior=prior,
    )
    prior["CHALLENGER"] = out
    pipe = _pipeline_steps(state.get("agents_pipeline"))
    pipe.append(
        AgentPipelineStep(
            id=_agent_id("CHALLENGER"),
            status="completed",
            phase=out.phase,
            detail=out.detail,
        )
    )
    await _sse_discussion_turn(on_progress, "CHALLENGER", 1, out)
    pq = (out.new_question or "").strip()
    if (
        out.needs_more_data
        and pq
        and is_valid_tts_sub_question(pq)
        and hp is not None
        and ap is not None
    ):
        ds = discussion_state_from_legacy_prior(prior)
        if ds is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to build discussion state for pause",
            )
        paused = LlmAgentsPaused(
            pipeline=pipe,
            discussion=ds,
            question=state["question"],
            generated_sql=state["primary_sql"],
            primary_exe=state["primary_exe"],
            deterministic_summary=state["deterministic_summary"],
            sub_results=list(state["sub_results"]),
            host_plan=hp,
            analyst_plan=ap,
            proposed_sub_question=pq,
            rationale=out.new_question_rationale,
        )
        outcome = await stream_outcome_from_llm_agents_result(
            agents_out=paused,
            req=c["req"],
            on_progress=on_progress,
        )
        return {
            "agents_prior": {k: v.model_dump(mode="json") for k, v in prior.items()},
            "agents_pipeline": [p.model_dump(mode="json") for p in pipe],
            "outcome": outcome,
        }
    return {
        "agents_prior": {k: v.model_dump(mode="json") for k, v in prior.items()},
        "agents_pipeline": [p.model_dump(mode="json") for p in pipe],
    }


def _route_after_challenger(state: PulsecastState) -> str:
    if state.get("outcome") is not None:
        return "end"
    if state["discussion_depth"] == "moderated":
        r = int(state["current_round"])
        mx = int(state["max_rounds"])
        turns = _turns_from_json(state.get("discussion_turns"))
        if not turns:
            return "host_finalize"
        challenger_turn = turns[-1]
        if r >= mx or not challenger_turn.output.continue_discussion:
            return "host_finalize"
        return "round_gate"
    return "host_finalize"


async def _node_round_gate(state: PulsecastState, config: RunnableConfig) -> dict[str, Any]:
    c = config["configurable"]
    hp, ap = state["host_plan"], state["analyst_plan"]
    if hp is None or ap is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Discussion pause requires host_plan and analyst_plan",
        )
    analyst_out = _AgentOut.model_validate(state["analyst_output"])
    turns = _turns_from_json(state.get("discussion_turns"))
    challenger_turn = turns[-1]
    await _sse_challenger_control(c.get("on_progress"), challenger_turn.output)
    r = int(state["current_round"])
    focus = (challenger_turn.output.focus_for_next_round or "").strip() or None
    pipe = _pipeline_steps(state.get("agents_pipeline"))
    llm_out = LlmAgentsPausedDiscussion(
        stage="mid",
        requested_depth="moderated",
        pipeline=list(pipe),
        discussion=DiscussionState(analyst=analyst_out, turns=list(turns)),
        question=state["question"],
        generated_sql=state["primary_sql"],
        primary_exe=state["primary_exe"],
        deterministic_summary=state["deterministic_summary"],
        sub_results=list(state["sub_results"]),
        host_plan=hp,
        analyst_plan=ap,
        max_rounds=int(state["max_rounds"]),
        next_round_index=r + 1,
        focus_for_next_round=focus,
        rationale=challenger_turn.output.stop_reason,
    )
    outcome = await stream_outcome_from_llm_agents_result(
        agents_out=llm_out,
        req=c["req"],
        on_progress=c.get("on_progress"),
    )
    return {"outcome": outcome}


def _route_after_round_gate(state: PulsecastState) -> str:
    if state.get("outcome") is not None:
        return "end"
    return "marketing"


async def _node_host_finalize(state: PulsecastState, config: RunnableConfig) -> dict[str, Any]:
    if state.get("outcome") is not None:
        return {}
    c = config["configurable"]
    settings = c["settings"]
    on_progress = c.get("on_progress")
    ctx = state["ctx"]
    question = state["question"]

    if state["discussion_depth"] == "minimal":
        analyst_out = _AgentOut.model_validate(state["analyst_output"])
        pipe = _pipeline_steps(state.get("agents_pipeline"))
        for role in ("MARKETING", "FINANCE", "FORECASTER", "WEB_CRAWLER", "CHALLENGER"):
            pr: PulsecastRole = role  # type: ignore[assignment]
            pipe.append(
                AgentPipelineStep(
                    id=_agent_id(pr),
                    status="skipped",
                    phase="Skipped (minimal)",
                    detail=None,
                )
            )
        discussion = DiscussionState(analyst=analyst_out, turns=[])
        host_out = await _run_host_json(
            settings,
            "HOST",
            system_prompt_host_composer_minimal(),
            _host_user_prompt(ctx=ctx, discussion=discussion),
        )
        pipe.append(
            AgentPipelineStep(
                id=_agent_id("HOST"),
                status="completed",
                phase=host_out.phase,
                detail=host_out.detail,
            )
        )
        final_text = _host_final_answer_with_canonical(
            canonical_question=str(ctx.get("question") or ""),
            host_body=host_out.text,
        )
        complete = LlmAgentsComplete(
            answer=final_text,
            pipeline=pipe,
            messages=[AgentInsight(role="HOST", text=final_text)],
            discussion=discussion,
        )
    elif state["discussion_depth"] == "moderated":
        analyst_out = _AgentOut.model_validate(state["analyst_output"])
        turns = _turns_from_json(state.get("discussion_turns"))
        discussion = DiscussionState(analyst=analyst_out, turns=turns)
        host_out = await _run_host_json(
            settings,
            "HOST",
            system_prompt_host_composer(),
            _host_user_prompt(ctx=ctx, discussion=discussion),
        )
        pipe = _pipeline_steps(state.get("agents_pipeline"))
        pipe.append(
            AgentPipelineStep(
                id=_agent_id("HOST"),
                status="completed",
                phase=host_out.phase,
                detail=host_out.detail,
            )
        )
        final_text = _host_final_answer_with_canonical(
            canonical_question=question,
            host_body=host_out.text,
        )
        complete = LlmAgentsComplete(
            answer=final_text,
            pipeline=pipe,
            messages=[AgentInsight(role="HOST", text=final_text)],
            discussion=discussion,
        )
    else:
        prior = _prior_models(dict(state.get("agents_prior") or {}))
        discussion = discussion_state_from_legacy_prior(prior)
        if discussion is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to build discussion state after linear agent run",
            )
        host_out = await _run_host_json(
            settings,
            "HOST",
            system_prompt_host_composer(),
            _host_user_prompt(ctx=ctx, discussion=discussion),
        )
        pipe = _pipeline_steps(state.get("agents_pipeline"))
        pipe.append(
            AgentPipelineStep(
                id=_agent_id("HOST"),
                status="completed",
                phase=host_out.phase,
                detail=host_out.detail,
            )
        )
        final_text = _host_final_answer_with_canonical(
            canonical_question=question,
            host_body=host_out.text,
        )
        complete = LlmAgentsComplete(
            answer=final_text,
            pipeline=pipe,
            messages=[AgentInsight(role="HOST", text=final_text)],
            discussion=discussion,
        )

    outcome = await stream_outcome_from_llm_agents_result(
        agents_out=complete,
        req=c["req"],
        on_progress=on_progress,
    )
    return {"outcome": outcome}


def build_agents_subgraph() -> StateGraph:
    wf = StateGraph(PulsecastState)
    wf.add_node("init_agents", _node_init_agents)
    wf.add_node("depth_gate", _node_depth_gate)
    wf.add_node("analyst", _node_analyst)
    wf.add_node("marketing", _node_marketing)
    wf.add_node("finance", _node_finance)
    wf.add_node("forecaster", _node_forecaster)
    wf.add_node("web_crawler", _node_web_crawler)
    wf.add_node("challenger", _node_challenger)
    wf.add_node("round_gate", _node_round_gate)
    wf.add_node("host_finalize", _node_host_finalize)

    wf.set_entry_point("init_agents")
    wf.add_conditional_edges(
        "init_agents",
        _route_after_init_agents,
        {"depth_gate": "depth_gate", "analyst": "analyst"},
    )
    wf.add_edge("depth_gate", END)
    wf.add_conditional_edges(
        "analyst",
        _route_after_analyst,
        {"host_finalize": "host_finalize", "marketing": "marketing"},
    )
    wf.add_edge("marketing", "finance")
    wf.add_edge("finance", "forecaster")
    wf.add_conditional_edges(
        "forecaster",
        _route_after_forecaster,
        {"end": END, "web_crawler": "web_crawler"},
    )
    wf.add_conditional_edges(
        "web_crawler",
        _route_after_web_crawler,
        {"end": END, "challenger": "challenger"},
    )
    wf.add_conditional_edges(
        "challenger",
        _route_after_challenger,
        {"end": END, "round_gate": "round_gate", "host_finalize": "host_finalize"},
    )
    wf.add_conditional_edges(
        "round_gate",
        _route_after_round_gate,
        {"end": END, "marketing": "marketing"},
    )
    wf.add_edge("host_finalize", END)
    return wf
