# Pulsecast API (Fast QA bridge)

Chains the text-to-SQL service and `execute_sql` MCP tool; exposes `POST /api/v1/qa` for the Pulsecast frontend Interaction panel.

Successful responses include BRD-aligned multi-agent metadata:

- `pipeline` — Host → Analyst → Marketing → Finance → Challenger (phase + status) for Agent Status / dashboards
- `agent_messages` — parallel chat turns; the Analyst turn mirrors `answer` (data + SQL summary)

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
- `DEFAULT_USER_ID`, `DEFAULT_USER_DB_ID`, `DEFAULT_DB_TYPE`, `DEFAULT_SCHEMA_NAME`, `DEFAULT_MODEL`, `DEFAULT_MAX_NODES`
- `CORS_ORIGINS` (comma-separated, default includes `http://localhost:5173`)
- `HTTP_TIMEOUT_SECONDS`

## Run

```bash
cd fastapi
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/health`
- QA: `POST http://localhost:8000/api/v1/qa` with JSON body `{"question":"...","session_id":"..."}` (other fields optional; defaults apply).

Frontend: set `VITE_PULSECAST_API_URL=http://localhost:8000` in `frontend/.env`.
