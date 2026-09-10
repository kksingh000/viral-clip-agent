"""Alembic environment.

Runs against the synchronous driver derived from ``DATABASE_URL`` so the same
configuration drives the app, the workers and migrations.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text

from app.core.config import settings
from app.database.base import Base

# Importing the models package registers every mapper on Base.metadata.
import app.models  # noqa: F401

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", settings.sync_database_url)
target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    """Ignore objects Alembic should not manage."""
    if type_ == "table" and name in {"spatial_ref_sys", "alembic_version"}:
        return False
    return True


def _configure(connection=None, url=None) -> None:
    context.configure(
        connection=connection,
        url=url,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
        # SQLite cannot ALTER most things in place; batch mode rewrites the
        # table instead, so the same migrations run on both backends.
        render_as_batch=settings.sync_database_url.startswith("sqlite"),
        user_module_prefix="sa_types.",
        dialect_opts={"paramstyle": "named"},
    )


def run_migrations_offline() -> None:
    _configure(url=settings.sync_database_url)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = settings.sync_database_url
    connectable = engine_from_config(
        section, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        if connection.dialect.name == "postgresql":
            # pgvector is optional; the Vector type degrades to text without it.
            try:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                connection.commit()
            except Exception:  # pragma: no cover - insufficient privileges
                connection.rollback()
        _configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
