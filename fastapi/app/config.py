from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    text_to_sql_base_url: str = "http://localhost:5001"
    execute_sql_url: str = "http://localhost:3333/mcp/tools/execute_sql"
    merge_bi_query_url: str = "http://15.206.14.199:5333/follow_up_classification"

    # text_to_sql_base_url: str = "https://fastapi.flashdice.dev.sntechlabs.com"
    # execute_sql_url: str = "https://mcp.flashdice.dev.sntechlabs.com/mcp/tools/execute_sql"

    http_timeout_seconds: float = 120.0

    # Mandatory OpenAI-compatible LLM (used for Pulsecast multi-agent reasoning)
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    openai_temperature: float = 0.4
    openai_timeout_seconds: float = 60.0

    # Defaults for missing request fields (override per environment)
    default_user_id: int = 41
    default_user_db_id: int = 1
    default_db_type: str = "snowflake"
    default_schema_name: str = "dev"
    default_model: str = "openai/gpt-4.1-mini-2025-04-14"
    default_max_nodes: str = "15"

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Serper (google.serper.dev) — Web Crawler agent; optional
    serper_api_key: str | None = None
    serper_num_results: int = 6
    web_search_summarize_enabled: bool = True

    pulsecast_discussion_max_rounds: int = 3

    # Aggregation Agent (Query Strategist)
    query_strategy_enabled: bool = True
    query_strategy_row_threshold: int = 10000
    query_strategy_confidence_threshold: float = 0.7

    # Planner caps (Text-to-SQL + Web crawler steps)
    pulsecast_max_plan_sql: int = 4
    pulsecast_max_plan_web: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
