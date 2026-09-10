"""Configuration parsing and startup guards.

Config bugs only surface at deployment, which is the worst place to find them.
The comma-separated collection formats below are exactly what ``.env.example``
documents and what ``docker-compose.yml`` passes.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings


def build(**env) -> Settings:
    """Construct settings from an explicit environment, ignoring any .env."""
    return Settings(_env_file=None, **env)


@pytest.fixture
def clean_env(monkeypatch):
    """Remove settings the test harness exports.

    ``_env_file=None`` suppresses the .env file but not the process
    environment, so a leaked SECRET_KEY would mask the startup guards.
    """
    for name in ("SECRET_KEY", "ENVIRONMENT", "CORS_ORIGINS", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)


class TestCollectionSettings:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (
                "http://localhost:3000,http://127.0.0.1:3000",
                ["http://localhost:3000", "http://127.0.0.1:3000"],
            ),
            ("https://solo.example.com", ["https://solo.example.com"]),
            (
                "  https://a.example.com , https://b.example.com  ",
                ["https://a.example.com", "https://b.example.com"],
            ),
            ('["https://x.example.com"]', ["https://x.example.com"]),
            (
                "https://a.example.com,,https://b.example.com,",
                ["https://a.example.com", "https://b.example.com"],
            ),
        ],
    )
    def test_cors_origins_accepts_every_documented_format(self, raw, expected):
        """Regression: a list-typed setting was JSON-decoded by
        pydantic-settings *before* validators ran, so the documented
        comma-separated form aborted startup entirely."""
        assert build(secret_key="k", cors_origins=raw).cors_origins == expected

    def test_cors_origins_has_a_working_default(self):
        origins = build(secret_key="k").cors_origins
        assert "http://localhost:3000" in origins
        assert "http://127.0.0.1:3000" in origins

    def test_a_list_value_passes_through(self):
        assert build(secret_key="k", cors_origins=["https://a.example"]).cors_origins == [
            "https://a.example"
        ]

    def test_media_allowlists_also_accept_csv(self):
        settings = build(
            secret_key="k",
            allowed_video_extensions=".mp4,.mov",
            allowed_video_mimetypes="video/mp4,video/quicktime",
        )
        assert list(settings.allowed_video_extensions) == [".mp4", ".mov"]
        assert list(settings.allowed_video_mimetypes) == [
            "video/mp4",
            "video/quicktime",
        ]

    def test_malformed_json_falls_back_to_csv(self):
        assert build(secret_key="k", cors_origins="[not valid json").cors_origins == [
            "https://[not valid json"
        ]


class TestOriginNormalisation:
    """Hosting platforms expose a service address as a bare hostname. An
    origin without a scheme never matches a browser's Origin header, and the
    failure surfaces as an unexplained CORS rejection."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("viralagent-web.onrender.com", ["https://viralagent-web.onrender.com"]),
            ("https://app.example.com", ["https://app.example.com"]),
            ("http://app.example.com", ["http://app.example.com"]),
            ("localhost:3000", ["http://localhost:3000"]),
            ("127.0.0.1:3000", ["http://127.0.0.1:3000"]),
            (
                "a.example.com,https://b.example.com",
                ["https://a.example.com", "https://b.example.com"],
            ),
        ],
    )
    def test_bare_hosts_gain_a_scheme(self, raw, expected):
        assert build(secret_key="k", cors_origins=raw).cors_origins == expected

    def test_trailing_slashes_are_stripped(self):
        """An Origin header never carries a trailing slash, so one here would
        silently fail to match."""
        assert build(
            secret_key="k", cors_origins="https://app.example.com/"
        ).cors_origins == ["https://app.example.com"]

    def test_duplicates_collapse(self):
        assert build(
            secret_key="k",
            cors_origins="app.example.com,https://app.example.com",
        ).cors_origins == ["https://app.example.com"]


class TestStartupGuards:
    def test_production_requires_a_secret_key(self, clean_env):
        with pytest.raises(ValueError, match="SECRET_KEY"):
            build(environment="production")

    def test_staging_requires_a_secret_key(self, clean_env):
        with pytest.raises(ValueError, match="SECRET_KEY"):
            build(environment="staging")

    def test_local_generates_an_ephemeral_key(self, clean_env):
        settings = build(environment="local")
        assert settings.secret_key
        # Ephemeral, so it must not be a stable value baked into the image.
        assert build(environment="local").secret_key != settings.secret_key

    def test_s3_requires_a_bucket(self):
        with pytest.raises(ValueError, match="S3_BUCKET"):
            build(secret_key="k", storage_provider="s3")

    @pytest.mark.parametrize(
        "provider,expected",
        [("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")],
    )
    def test_llm_providers_require_their_key(self, provider, expected):
        with pytest.raises(ValueError, match=expected):
            build(secret_key="k", llm_provider=provider)

    def test_mock_provider_needs_no_credentials(self):
        assert build(secret_key="k", llm_provider="mock").llm_provider == "mock"


class TestDerivedValues:
    def test_async_urls_are_converted_for_sync_consumers(self):
        settings = build(
            secret_key="k",
            database_url="postgresql+asyncpg://u:p@db:5432/app",
        )
        assert settings.sync_database_url == "postgresql+psycopg://u:p@db:5432/app"
        assert settings.is_postgres is True

    def test_bare_postgres_urls_gain_a_sync_driver(self):
        settings = build(secret_key="k", database_url="postgresql://u:p@db:5432/app")
        assert settings.sync_database_url.startswith("postgresql+psycopg://")

    def test_sqlite_drops_the_async_driver(self):
        settings = build(secret_key="k", database_url="sqlite+aiosqlite:///./x.db")
        assert settings.sync_database_url == "sqlite:///./x.db"
        assert settings.is_postgres is False

    def test_broker_falls_back_to_redis_url(self):
        settings = build(secret_key="k", redis_url="redis://cache:6379/2")
        assert settings.broker_url == "redis://cache:6379/2"
        assert settings.result_backend == "redis://cache:6379/2"

    def test_an_explicit_broker_wins(self):
        settings = build(
            secret_key="k",
            redis_url="redis://cache:6379/0",
            celery_broker_url="amqp://rabbit:5672",
        )
        assert settings.broker_url == "amqp://rabbit:5672"

    @pytest.mark.parametrize(
        "environment,expected",
        [("local", False), ("test", False), ("development", False),
         ("staging", True), ("production", True)],
    )
    def test_is_production_covers_staging(self, environment, expected):
        assert build(secret_key="k", environment=environment).is_production is expected
