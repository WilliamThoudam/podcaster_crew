from langchain_core.messages import HumanMessage, SystemMessage

from podcaster_graph.config import agent_section, context_from_state, task_section
from podcaster_graph.llm import chat_model
from podcaster_graph.paths import new_script_path
from podcaster_graph.state import PodcasterState


def scripting_node(state: PodcasterState) -> dict[str, str]:
    ctx = context_from_state(state)
    role, goal, backstory = agent_section("scriptwriter", ctx)
    desc, expected, _ = task_section("scripting_task", ctx)
    system = f"You are {role}.\nGoal: {goal}\n\nBackstory:\n{backstory}"
    report = state.get("report_output") or ""
    human = (
        f"{desc}\n\nReport to adapt into a podcast script:\n{report}\n\n"
        f"Produce this output:\n{expected}"
    )
    llm = chat_model()
    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    path = new_script_path()
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"script_output": content, "script_path": path}
