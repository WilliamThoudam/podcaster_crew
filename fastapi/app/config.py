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
    default_model: str = "openai/gpt-3.5-turbo"
    default_max_nodes: str = "15"

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Moderated Marketing/Finance/Challenger multi-round discussion (see pulsecast_llm_agents)
    pulsecast_discussion_enabled: bool = True
    pulsecast_discussion_max_rounds: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
