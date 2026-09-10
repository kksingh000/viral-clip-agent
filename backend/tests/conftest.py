"""Test configuration.

Environment variables are set *before* any application module is imported,
because :data:`app.core.config.settings` is built at import time.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

# --------------------------------------------------------------- environment
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="viralagent-tests-"))
(_TMP_ROOT / "storage").mkdir(parents=True, exist_ok=True)
(_TMP_ROOT / "work").mkdir(parents=True, exist_ok=True)

os.environ.setdefault("ENVIRONMENT", "test")
os.environ["SECRET_KEY"] = "test-secret-key-not-used-outside-tests"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{(_TMP_ROOT / 'test.db').as_posix()}"
os.environ["STORAGE_PROVIDER"] = "local"
os.environ["STORAGE_LOCAL_ROOT"] = str(_TMP_ROOT / "storage")
os.environ["WORKDIR"] = str(_TMP_ROOT / "work")
os.environ["LLM_PROVIDER"] = "mock"
os.environ["TRANSCRIPTION_PROVIDER"] = "mock"
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "true"
os.environ["RATE_LIMIT_ENABLED"] = "false"
os.environ["LOG_JSON"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.database.base import Base  # noqa: E402
from app.database.session import (  # noqa: E402
    get_async_engine,
    get_sync_engine,
    get_sync_session_factory,
)
from app.main import app as fastapi_app  # noqa: E402

import app.models  # noqa: F401,E402  (registers every mapper)

SAMPLE_VIDEO = Path(__file__).resolve().parents[2] / "data" / "samples" / "sample_landscape.mp4"


@pytest.fixture(scope="session", autouse=True)
def _create_schema():
    engine = get_sync_engine()
    Base.metadata.create_all(engine)
    yield
    engine.dispose()
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP_ROOT


@pytest.fixture
def sync_session():
    factory = get_sync_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http:
        yield http
    await get_async_engine().dispose()


@pytest_asyncio.fixture
async def auth_client(client: AsyncClient) -> AsyncClient:
    """A client with a freshly registered account attached."""
    import uuid as _uuid

    # example.com is the reserved-for-documentation domain that
    # email-validator still accepts; .test is rejected as special-use.
    email = f"user-{_uuid.uuid4().hex[:10]}@example.com"
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "correct-horse-battery", "full_name": "Test User"},
    )
    assert response.status_code == 201, response.text
    token = response.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return client


@pytest.fixture(scope="session")
def sample_video_path() -> Path:
    if not SAMPLE_VIDEO.exists():
        pytest.skip(
            "Sample video missing. Run: python scripts/make_sample_video.py"
        )
    return SAMPLE_VIDEO


@pytest.fixture(autouse=True)
def _reset_provider_cache():
    from app.providers.registry import reset_providers

    reset_providers()
    yield
    reset_providers()


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: exercises ffmpeg and takes minutes")
    config.addinivalue_line("markers", "integration: end-to-end pipeline test")
