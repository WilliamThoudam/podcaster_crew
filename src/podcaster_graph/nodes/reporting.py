from langchain_core.messages import HumanMessage, SystemMessage

from podcaster_graph.config import agent_section, context_from_state, task_section
from podcaster_graph.llm import chat_model
from podcaster_graph.paths import new_report_path
from podcaster_graph.state import PodcasterState


def reporting_node(state: PodcasterState) -> dict[str, str]:
    ctx = context_from_state(state)
    role, goal, backstory = agent_section("reporting_analyst", ctx)
    desc, expected, _ = task_section("reporting_task", ctx)
    system = f"You are {role}.\nGoal: {goal}\n\nBackstory:\n{backstory}"
    prior = state.get("research_output") or ""
    human = (
        f"{desc}\n\nResearch notes from the previous step:\n{prior}\n\n"
        f"Produce this output:\n{expected}"
    )
    llm = chat_model()
    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    path = new_report_path()
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"report_output": content, "report_path": path}
