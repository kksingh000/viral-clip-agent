"""Engine and session management.

The API runs on the async engine; Celery workers and Alembic run on the sync
engine. Both are created lazily so that importing the package never opens a
connection.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _ensure_sqlite_dir(url: str) -> None:
    if not url.startswith("sqlite"):
        return
    path_part = url.split("///")[-1]
    if path_part and path_part != ":memory:":
        Path(path_part).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _engine_kwargs(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"echo": settings.database_echo, "future": True}
    return {
        "echo": settings.database_echo,
        "future": True,
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "pool_pre_ping": True,
        "pool_recycle": 1800,
    }


def _apply_sqlite_pragmas(engine: Engine | AsyncEngine) -> None:
    target = engine.sync_engine if isinstance(engine, AsyncEngine) else engine
    if target.dialect.name != "sqlite":
        return

    @event.listens_for(target, "connect")
    def _set_pragmas(dbapi_connection, _record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=10000")
        cursor.close()


@lru_cache(maxsize=1)
def get_async_engine() -> AsyncEngine:
    _ensure_sqlite_dir(settings.database_url)
    engine = create_async_engine(settings.database_url, **_engine_kwargs(settings.database_url))
    _apply_sqlite_pragmas(engine)
    logger.info("async engine created", extra={"dialect": engine.dialect.name})
    return engine


@lru_cache(maxsize=1)
def get_sync_engine() -> Engine:
    url = settings.sync_database_url
    _ensure_sqlite_dir(url)
    engine = create_engine(url, **_engine_kwargs(url))
    _apply_sqlite_pragmas(engine)
    return engine


@lru_cache(maxsize=1)
def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=get_async_engine(), expire_on_commit=False, autoflush=False
    )


@lru_cache(maxsize=1)
def get_sync_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_sync_engine(), expire_on_commit=False, autoflush=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped async session."""
    factory = get_async_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def async_session_scope() -> AsyncIterator[AsyncSession]:
    factory = get_async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for synchronous (worker) code."""
    factory = get_sync_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


async def dispose_engines() -> None:
    await get_async_engine().dispose()
    get_sync_engine().dispose()
