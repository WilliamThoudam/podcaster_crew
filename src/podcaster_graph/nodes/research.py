from langchain_core.messages import HumanMessage, SystemMessage

from podcaster_graph.config import agent_section, context_from_state, task_section
from podcaster_graph.llm import chat_model
from podcaster_graph.state import PodcasterState


def research_node(state: PodcasterState) -> dict[str, str]:
    ctx = context_from_state(state)
    role, goal, backstory = agent_section("researcher", ctx)
    desc, expected, _ = task_section("research_task", ctx)
    system = f"You are {role}.\nGoal: {goal}\n\nBackstory:\n{backstory}"
    human = f"{desc}\n\nProduce this output:\n{expected}"
    llm = chat_model()
    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    return {"research_output": content}
