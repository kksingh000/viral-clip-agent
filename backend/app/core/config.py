"""Application configuration.

All configuration is sourced from environment variables (or a local ``.env``
file). Nothing here carries a real secret default: security-sensitive settings
default to ``None`` and are validated at startup.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ----------------------------------------------------------------- general
    app_name: str = "Viral Clip Agent"
    environment: Literal["local", "development", "staging", "production", "test"] = "local"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_json: bool = True

    @property
    def is_production(self) -> bool:
        return self.environment in ("staging", "production")

    # -------------------------------------------------------------------- auth
    secret_key: str | None = None
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 14
    allow_registration: bool = True

    # ---------------------------------------------------------------- database
    database_url: str = "sqlite+aiosqlite:///./data/viralagent.db"
    database_echo: bool = False
    db_pool_size: int = 10
    db_max_overflow: int = 20

    @property
    def sync_database_url(self) -> str:
        """Celery workers and Alembic use a synchronous driver."""
        url = self.database_url
        if url.startswith("sqlite"):
            return url.replace("+aiosqlite", "")
        return url.replace("+asyncpg", "+psycopg").replace(
            "postgresql://", "postgresql+psycopg://"
        )

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgres")

    # ------------------------------------------------------------------- redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    celery_task_always_eager: bool = False

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url

    # ----------------------------------------------------------------- storage
    storage_provider: Literal["local", "s3"] = "local"
    storage_local_root: Path = REPO_ROOT / "data" / "storage"
    storage_public_base_url: str = "http://localhost:8000/media"
    s3_bucket: str | None = None
    s3_region: str | None = None
    s3_endpoint_url: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    signed_url_ttl_seconds: int = 3600

    # --------------------------------------------------------------- providers
    llm_provider: Literal["anthropic", "openai", "mock"] = "mock"
    llm_model: str = "claude-sonnet-5"
    llm_model_cheap: str = "claude-haiku-4-5-20251001"
    llm_max_output_tokens: int = 8192
    llm_temperature: float = 0.3
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 180
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    transcription_provider: Literal["faster_whisper", "openai", "mock"] = "mock"
    whisper_model_size: str = "base"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    transcription_default_language: str | None = None

    vision_provider: Literal["opencv", "mock"] = "opencv"

    youtube_api_key: str | None = None

    # ------------------------------------------------------------------- media
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None
    workdir: Path = REPO_ROOT / "data" / "work"
    max_upload_bytes: int = 4 * 1024 * 1024 * 1024
    allowed_video_extensions: Annotated[tuple[str, ...], NoDecode] = (
        ".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mpg", ".mpeg", ".wmv",
    )
    allowed_video_mimetypes: Annotated[tuple[str, ...], NoDecode] = (
        "video/mp4", "video/quicktime", "video/x-matroska", "video/webm",
        "video/x-msvideo", "video/mpeg", "video/x-ms-wmv", "application/octet-stream",
    )
    ffmpeg_timeout_seconds: int = 3600
    video_encode_preset: str = "medium"
    video_encode_crf: int = 18

    # ------------------------------------------------------------------ limits
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 300
    rate_limit_window_seconds: int = 60
    # localhost and 127.0.0.1 are different origins to a browser, and both are
    # in everyday use during development, so both are allowed by default.
    # Production deployments should set CORS_ORIGINS explicitly.
    #
    # ``NoDecode`` is essential: pydantic-settings JSON-decodes list-typed
    # fields inside the environment source, *before* any validator runs, so a
    # plain comma-separated value -- the documented format, and what Docker
    # Compose passes -- would abort startup with a parse error.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

    # ---------------------------------------------------------------- pipeline
    max_candidates_per_video: int = 12
    default_clips_per_video: int = 3
    transcript_chunk_chars: int = 14000

    @field_validator(
        "cors_origins",
        "allowed_video_extensions",
        "allowed_video_mimetypes",
        mode="before",
    )
    @classmethod
    def _split_csv(cls, v: Any) -> Any:
        """Accept a comma-separated string for any collection setting.

        Environment variables are flat strings; requiring JSON there is a trap
        that only shows up at deployment time. A JSON array is still accepted,
        because ``NoDecode`` means parsing is now this method's job -- and a
        silently mis-parsed origin is a long debugging session.
        """
        if not isinstance(v, str):
            return v
        text = v.strip()
        if text.startswith("[") and text.endswith("]"):
            import json

            try:
                decoded = json.loads(text)
            except ValueError:
                pass
            else:
                if isinstance(decoded, list):
                    return [str(item).strip() for item in decoded if str(item).strip()]
        return [item.strip() for item in text.split(",") if item.strip()]

    @field_validator("storage_local_root", "workdir", mode="before")
    @classmethod
    def _expand_path(cls, v: Any) -> Any:
        if isinstance(v, str):
            return Path(v).expanduser()
        return v

    @model_validator(mode="after")
    def _validate_security(self) -> "Settings":
        if not self.secret_key:
            if self.is_production:
                raise ValueError("SECRET_KEY must be set in staging/production.")
            object.__setattr__(self, "secret_key", secrets.token_urlsafe(48))
        if self.storage_provider == "s3" and not self.s3_bucket:
            raise ValueError("S3_BUCKET is required when STORAGE_PROVIDER=s3")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
