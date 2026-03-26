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
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

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

## Notes

- `/api/v1/qa` has been removed.
- If required OpenAI-compatible variables are missing or upstream calls fail, API returns OpenAI-style error payloads.
