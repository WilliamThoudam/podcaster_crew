# Pulsecast API (OpenAI-compatible)

OpenAI-compatible chat API backed by the Pulsecast QA pipeline:
text-to-SQL service + SQL validation + execute_sql + 5 LLM agent reasoning.

## Setup

From this directory:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e .
```

Optional: copy or symlink the repo root `.env` here, or set variables in your environment. Supported settings (see `app/config.py`):

- `TEXT_TO_SQL_BASE_URL` (default `http://localhost:5001`)
- `EXECUTE_SQL_URL` (default `http://localhost:3333/mcp/tools/execute_sql`)
- `OPENAI_BASE_URL` (required; e.g. `https://api.openai.com` or your compatible server)
- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (required)
- `OPENAI_TEMPERATURE` (optional; default `0.4`)
- `OPENAI_TIMEOUT_SECONDS` (optional; default `60`)
- `DEFAULT_USER_ID`, `DEFAULT_USER_DB_ID`, `DEFAULT_DB_TYPE`, `DEFAULT_SCHEMA_NAME`, `DEFAULT_MODEL`, `DEFAULT_MAX_NODES`
- `CORS_ORIGINS` (comma-separated, default includes `http://localhost:5173`)
- `HTTP_TIMEOUT_SECONDS`

## Run

```bash
cd fastapi
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --reload-delay 0.5
```

### Reload noise (`ERROR` / `CancelledError` in the log)

If you see a traceback ending in `asyncio.exceptions.CancelledError` or `KeyboardInterrupt` right after **WatchFiles** reloads, that is the **old worker being stopped** while Starlette’s lifespan loop is still waiting on `receive()`. It is **not** a failure of your routes (the next line is often `Started server process` and requests still return 200).

Per [Uvicorn’s reloading settings](https://www.uvicorn.org/settings/#development):

- Use **`--reload-delay`** (seconds) to **debounce** rapid saves so you do not trigger overlapping reloads (default is `0.25`; `0.5` or `1.0` helps when an editor writes a file twice).
- In **production**, run **without** `--reload`; this path does not apply.

This repo pins **`uvicorn[standard]>=0.41.0`** so you get fixes such as [lifespan shutdown when the reloader exits](https://github.com/Kludex/uvicorn/pull/2812). After upgrading, reinstall: `pip install -e .`

- Health: `GET http://localhost:8000/health`
- Models: `GET http://localhost:8000/v1/models`
- Chat Completions: `POST http://localhost:8000/v1/chat/completions`

Frontend: set `VITE_PULSECAST_API_URL=http://localhost:8000` in `frontend/.env`.

## API contract

`POST /v1/chat/completions` accepts OpenAI-style payloads:
- `model`
- `messages`
- `stream` (`false` or `true`)
- optional `temperature`, `response_format`

When `stream=false`, response is a strict OpenAI `chat.completion` object.
When `stream=true`, response is SSE with OpenAI chunk format and `data: [DONE]`.

The assistant `message.content` is a JSON string containing:
- `answer`
- `generated_sql`
- `execute`
- `pipeline`
- `agent_messages`

## Orchestration and LLM stack

- **LangGraph** runs the main Pulsecast completion flow (`planning` → `sub_questions` → `agents`) in [`app/graph/pulsecast_graph.py`](app/graph/pulsecast_graph.py). Step implementations live in [`app/services/pulsecast_completion_steps.py`](app/services/pulsecast_completion_steps.py); SSE markers are unchanged for the frontend.
- **LangChain** is used as plumbing: **`langchain-openai`** `ChatOpenAI` (OpenAI-compatible `base_url`) and **`langchain-core`** messages in [`app/llm/chat_model.py`](app/llm/chat_model.py), planning, and multi-agent streaming.
- Upgrade **`langgraph`**, **`langchain-core`**, and **`langchain-openai`** together when bumping versions.

## Notes

- **Python 3.14+:** LangChain may log a Pydantic v1 compatibility warning at import time; the app filters it in [`app/__init__.py`](app/__init__.py) (before `main` imports routers). If you still see it in the **reloader** process, set `PYTHONWARNINGS=ignore::UserWarning:langchain_core._api.deprecation` or use **Python 3.12 or 3.13** until LangChain drops internal Pydantic v1 usage.
- `/api/v1/qa` has been removed.
- If required OpenAI-compatible variables are missing or upstream calls fail, API returns OpenAI-style error payloads.
- Set `OPENAI_BASE_URL` to your vendor root including `/v1` when required (e.g. `https://api.openai.com/v1` or your LiteLLM/OpenRouter/vLLM base).
