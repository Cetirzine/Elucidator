from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_KEY", "DEEPSEEK_API_KEY"),
    )
    llm_model: str = "deepseek-v4-flash"
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")

