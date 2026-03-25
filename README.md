# Podcaster Crew

Welcome to the Podcaster Crew project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```


Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```

### Setup

1. Create `.env` file at project root and add:

```
MODEL=gpt-4.1-mini-2025-04-14
OPENAI_API_KEY=sk-
# Optional: any OpenAI-compatible Chat Completions API (OpenRouter, LM Studio, vLLM, etc.)
# OPENAI_BASE_URL=https://openrouter.ai/api/v1
GEMINI_API_KEY=
SERPER_API_KEY=
# Podcast host names (used in scripts and Gemini multi-speaker TTS)
MALE_HOST=Jone
FEMALE_HOST=Jane
# Optional: Gemini prebuilt voices (defaults: Puck=male, Kore=female per Gemini catalog)
# GEMINI_VOICE_MALE=Puck
# GEMINI_VOICE_FEMALE=Kore

# Text-to-speech backend: gemini (default) or coqui (local Coqui server)
# TTS_PROVIDER=gemini
# COQUI_TTS_URL=http://localhost:5002
# Use the exact base URL your process can reach (host, Docker service name, etc.); it is not rewritten.
# COQUI_TTS_SPEAKER_MALE=p225
# COQUI_TTS_SPEAKER_FEMALE=p226
# COQUI_TTS_API_KEY=
```

When using a custom `OPENAI_BASE_URL`, set `MODEL` to whatever that provider expects (for example OpenRouter uses names like `openai/gpt-4o-mini` or `meta-llama/llama-3.3-70b-instruct`). `OPENAI_API_KEY` is sent as the Bearer token; use a placeholder if the server does not require a key.

You'll need to add credits for these:
OpenAI API Key: https://platform.openai.com/api-keys
Gemini API Key: https://aistudio.google.com/apikey
Serper API Key: https://serper.dev/


## Running the Project

### CrewAI (default template)

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```

This command initializes the podcaster Crew, assembling the agents and assigning them tasks as defined in your configuration.

### LangGraph (same agents & tasks, sequential graph)

An alternative runner lives under `src/podcaster_graph/`. It reads the same
[`src/podcaster/config/agents.yaml`](src/podcaster/config/agents.yaml) and
[`tasks.yaml`](src/podcaster/config/tasks.yaml) and runs **research → reporting → scripting → audio**
as a LangGraph `StateGraph` (OpenAI-compatible chat via `langchain-openai`, same env vars as the Crew).

After `uv pip install -e .` (or `pip install -e .`) from the repo root on **Python 3.10–3.13**:

```bash
run_graph
```

Or:

```bash
python -m podcaster_graph.main
```

Outputs (`*.md` reports/scripts and `.wav` audio) go to `outputs/` like the Crew path. TTS uses
`podcaster.tools.custom_tool.synthesize_podcast_wav` (Gemini or Coqui per `TTS_PROVIDER`).

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

## Customising
- Modify `src/podcaster/config/agents.yaml` to define your agents
- Modify `src/podcaster/config/tasks.yaml` to define your tasks
- Modify `src/podcaster/crew.py` to add your own logic, tools and specific args
- Modify `src/podcaster/main.py` to add custom inputs for your agents and tasks

## Support

For support, questions, or feedback regarding the Podcaster Crew or crewAI, visit CrewAI [documentation](https://docs.crewai.com).
