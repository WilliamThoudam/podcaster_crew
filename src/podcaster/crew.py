from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from crewai.agents.agent_builder.base_agent import BaseAgent
from functools import lru_cache
from typing import List
from .tools import file_read_tool
from .tools.custom_tool import gemini_voice_tool
import os
from datetime import datetime


@lru_cache(maxsize=1)
def _openai_compatible_llm() -> LLM:
    """
    Uses MODEL and OPENAI_API_KEY from the environment.
    Set OPENAI_BASE_URL to any OpenAI-compatible completions API (OpenRouter, LM Studio, vLLM, etc.).
    """
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14")
    base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip() or None
    kwargs = {"model": model}
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    return LLM(**kwargs)

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Podcaster():
    """Podcaster crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            verbose=True,
            llm=_openai_compatible_llm(),
            # tools=[search_tool, file_writer_tool, file_read_tool]
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
            verbose=True,
            llm=_openai_compatible_llm(),
            # tools=[file_writer_tool, file_read_tool]
        )

    @agent
    def scriptwriter(self) -> Agent:
        return Agent(
            config=self.agents_config['scriptwriter'], # type: ignore[index]
            verbose=True,
            llm=_openai_compatible_llm(),
            # No file_writer_tool: agents often save ad-hoc names at repo root (e.g. *_podcast_script.txt).
            # Script is already written by this task's output_file under outputs/.
            tools=[file_read_tool],
        )

    @agent
    def audio_producer(self) -> Agent:
        return Agent(
            config=self.agents_config['audio_producer'],  # type: ignore[index]
            verbose=True,
            llm=_openai_compatible_llm(),
            tools=[gemini_voice_tool],
            allow_delegation=False,
        )

    
    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @before_kickoff
    def _ensure_outputs_dir(self, inputs):
        os.makedirs(os.path.join(os.getcwd(), 'outputs'), exist_ok=True)
        return inputs
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        topic_slug = (os.getenv("TOPIC") or "topic").lower().replace(" ", "-")
        report_path = os.path.join('outputs', f'{topic_slug}-report-{timestamp}.md')
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            output_file=report_path
        )


    @task
    def scripting_task(self) -> Task:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        topic_slug = (os.getenv("TOPIC") or "topic").lower().replace(" ", "-")
        script_path = os.path.join('outputs', f'{topic_slug}-script-{timestamp}.md')
        return Task(
            config=self.tasks_config['scripting_task'], # type: ignore[index]
            output_file=script_path,
        )

    @task
    def audio_task(self) -> Task:
        return Task(
            config=self.tasks_config['audio_task'],  # type: ignore[index]
            context=[self.scripting_task()],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Podcaster crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
