"""Portable column types.

The production target is PostgreSQL (with ``pgvector``); SQLite is supported so
the full pipeline can be exercised locally and in CI without a database server.
These types pick the native PostgreSQL implementation when available and fall
back to a portable representation otherwise.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import CHAR, JSON, DateTime, String, Text, TypeDecorator
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Dialect


class UTCDateTime(TypeDecorator):
    """Timezone-aware timestamps on every backend.

    PostgreSQL round-trips ``timestamptz`` correctly; SQLite has no timezone
    support and hands back naive values, which then cannot be subtracted from
    the aware values the application creates. This normalises both directions:
    everything is stored as UTC and always read back as aware UTC.
    """

    impl = DateTime
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        return dialect.type_descriptor(DateTime(timezone=True))

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        return value

    def process_result_value(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class EnumString(TypeDecorator):
    """A string column that reads back as its Python enum member.

    Declaring these columns as a bare ``String`` stored the value correctly but
    returned a plain ``str``, so ``loaded.status is JobStatus.CANCELLED`` was
    silently always ``False`` -- which disabled job cancellation and stopped
    Instagram publishing from ever receiving its media URL. Round-tripping
    through the enum makes identity comparisons valid everywhere and validates
    on write, so an invalid status fails at the boundary instead of being
    stored.

    A value that is not a member of the enum (a row written by an older
    version) is passed through unchanged rather than raising, so a deployment
    can roll forward without a data migration.
    """

    impl = String
    cache_ok = True

    def __init__(self, enum_class: Any, length: int = 32) -> None:
        self.enum_class = enum_class
        super().__init__(length)

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        if isinstance(value, self.enum_class):
            return value.value
        # Validate on write: a typo becomes an error here rather than a row
        # nothing can ever match.
        return self.enum_class(value).value

    def process_result_value(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        try:
            return self.enum_class(value)
        except ValueError:  # pragma: no cover - forward compatibility
            return value


class GUID(TypeDecorator):
    """UUID column: native ``uuid`` on PostgreSQL, ``CHAR(32)`` elsewhere."""

    impl = CHAR
    cache_ok = True
    python_type = uuid.UUID

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(postgresql.UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        if dialect.name == "postgresql":
            return value
        return value.hex

    def process_result_value(self, value: Any, dialect: Dialect) -> uuid.UUID | None:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


class JSONB(TypeDecorator):
    """``JSONB`` on PostgreSQL, generic ``JSON`` elsewhere."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(postgresql.JSONB())
        return dialect.type_descriptor(JSON())


class Vector(TypeDecorator):
    """Embedding column.

    Uses ``pgvector`` when the extension and package are present, otherwise
    stores a JSON array of floats. :mod:`app.services.embeddings` reads
    :attr:`Vector.native` to decide between an indexed SQL nearest-neighbour
    query and an in-process cosine scan.
    """

    impl = Text
    cache_ok = True

    native = False

    def __init__(self, dimensions: int = 1536) -> None:
        self.dimensions = dimensions
        super().__init__()

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            try:
                from pgvector.sqlalchemy import Vector as PGVector  # type: ignore

                type(self).native = True
                return dialect.type_descriptor(PGVector(self.dimensions))
            except Exception:  # pragma: no cover - optional dependency
                pass
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        if dialect.name == "postgresql" and type(self).native:
            return list(value)
        return json.dumps([float(x) for x in value])

    def process_result_value(self, value: Any, dialect: Dialect) -> Sequence[float] | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return list(value)
        try:
            return json.loads(value)
        except (TypeError, ValueError):  # pragma: no cover - corrupt row
            return None
