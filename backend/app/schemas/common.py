"""Shared response envelopes and query helpers."""

from __future__ import annotations

from typing import Generic, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")

ORM_CONFIG = ConfigDict(from_attributes=True, populate_by_name=True)


class APIModel(BaseModel):
    model_config = ORM_CONFIG


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total

    @classmethod
    def of(
        cls, items: Sequence[T], *, total: int, limit: int, offset: int
    ) -> "Page[T]":
        return cls(items=list(items), total=total, limit=limit, offset=offset)


class PaginationParams(BaseModel):
    limit: int = Field(default=25, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class ErrorResponse(BaseModel):
    error_code: str
    error_message: str
    details: dict = Field(default_factory=dict)
    request_id: str | None = None


class MessageResponse(BaseModel):
    message: str


class HealthComponent(BaseModel):
    name: str
    status: str
    detail: str | None = None
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    components: list[HealthComponent] = Field(default_factory=list)
