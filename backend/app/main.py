"""FastAPI application factory.

The API is deliberately thin: it validates, authorises, records, and hands
heavy work to the worker fleet. No request path transcodes, transcribes or
calls a model.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.middleware.gzip import GZipMiddleware

from app import __version__
from app.api.v1 import media
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.errors import AppError
from app.core.logging import (
    configure_logging,
    get_logger,
    new_request_id,
    request_id_var,
)
from app.schemas.common import HealthComponent, HealthResponse

logger = get_logger(__name__)

DESCRIPTION = """\
An AI content-repurposing platform: discover momentum, find the self-contained
moments inside authorized long-form video, and turn them into vertical shorts.

**Rights model.** Trend discovery reads public metadata only and never
downloads media. Every discovered video is recorded with
`authorization_status = UNKNOWN` and cannot be processed or published until an
account holder records that they own it, hold a licence, or are otherwise
authorized. Discovering a URL never implies permission to republish it.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(settings.log_level, settings.log_json)
    logger.info(
        "starting api",
        extra={
            "version": __version__,
            "environment": settings.environment,
            "llm_provider": settings.llm_provider,
            "transcription_provider": settings.transcription_provider,
            "storage_provider": settings.storage_provider,
        },
    )
    from app.services.dispatch import resolve_mode, shutdown

    resolve_mode()
    try:
        yield
    finally:
        shutdown()
        from app.database.session import dispose_engines

        await dispose_engines()
        logger.info("api stopped")


#: Paths that serve already-compressed binaries. Running them through gzip
#: burns CPU for no size win and buffers large files in memory.
_UNCOMPRESSED_PREFIXES = ("/media/",)
_UNCOMPRESSED_SUFFIXES = ("/download",)


class SelectiveGZipMiddleware(GZipMiddleware):
    """GZip, except for media responses.

    Starlette's middleware compresses by size alone, with no content-type
    awareness, so a 200 MB mp4 download would be run through the compressor.
    """

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[override]
        if scope.get("type") == "http":
            path = scope.get("path", "")
            if path.startswith(_UNCOMPRESSED_PREFIXES) or path.endswith(
                _UNCOMPRESSED_SUFFIXES
            ):
                await self.app(scope, receive, send)
                return
        await super().__call__(scope, receive, send)


def _serialisable_errors(errors: list[dict]) -> list[dict]:
    """Make Pydantic validation errors safe to JSON-encode.

    A validator that raises ``ValueError`` -- which is how every cross-field
    rule in this API is expressed -- leaves the exception *object* in
    ``ctx["error"]``. Encoding that raises ``TypeError`` inside the error
    handler, turning a 422 into a 500 with no useful body. Values are coerced
    to strings and the input echo is trimmed so a large payload cannot be
    reflected back wholesale.
    """
    cleaned: list[dict] = []
    for error in errors:
        entry = {
            "type": str(error.get("type", "")),
            "loc": [str(part) for part in error.get("loc", ())],
            "msg": str(error.get("msg", "")),
        }
        context = error.get("ctx")
        if isinstance(context, dict):
            entry["ctx"] = {key: str(value)[:200] for key, value in context.items()}
        if "input" in error:
            entry["input"] = str(error["input"])[:200]
        cleaned.append(entry)
    return cleaned


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(SelectiveGZipMiddleware, minimum_size=1024)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or new_request_id()
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(elapsed_ms, 1),
                "request_id": request_id,
            },
        )
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.exception("unhandled application error", extra={"path": request.url.path})
        else:
            logger.info(
                "request rejected",
                extra={
                    "path": request.url.path,
                    "error_code": exc.error_code,
                    "status": exc.status_code,
                },
            )
        payload = exc.to_dict()
        payload["request_id"] = request_id_var.get()
        headers = {}
        if exc.error_code == "rate_limited":
            retry_after = exc.details.get("retry_after_seconds")
            if retry_after:
                headers["Retry-After"] = str(retry_after)
        return JSONResponse(status_code=exc.status_code, content=payload, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = _serialisable_errors(exc.errors()[:20])
        return JSONResponse(
            status_code=422,
            content={
                "error_code": "validation_error",
                # Surface the first message so a client sees *what* was wrong
                # without having to dig through the details array.
                "error_message": (
                    errors[0]["msg"] if errors else "The request is invalid."
                ),
                "details": {"errors": errors},
                "request_id": request_id_var.get(),
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", extra={"path": request.url.path})
        # Never leak internals to the client; the request id ties the response
        # to the full stack trace in the logs.
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "internal_error",
                "error_message": "An unexpected error occurred.",
                "details": {},
                "request_id": request_id_var.get(),
            },
        )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        components: list[HealthComponent] = []
        overall = "ok"

        # Database
        started = time.perf_counter()
        try:
            from app.database.session import get_async_session_factory

            async with get_async_session_factory()() as session:
                await session.execute(text("SELECT 1"))
            components.append(
                HealthComponent(
                    name="database",
                    status="ok",
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            )
        except Exception as exc:  # noqa: BLE001
            overall = "degraded"
            components.append(
                HealthComponent(name="database", status="error", detail=str(exc)[:200])
            )

        # Task dispatch
        from app.services.dispatch import resolve_mode

        mode = resolve_mode()
        components.append(
            HealthComponent(
                name="task_queue",
                status="ok" if mode == "broker" else "degraded",
                detail=(
                    "Celery broker reachable."
                    if mode == "broker"
                    else f"Running tasks in-process ({mode}); not suitable for production."
                ),
            )
        )
        if mode != "broker":
            overall = "degraded" if overall == "ok" else overall

        # ffmpeg
        try:
            from app.video.ffmpeg import ffmpeg_path, ffprobe_path

            components.append(
                HealthComponent(
                    name="ffmpeg",
                    status="ok",
                    detail=(
                        f"{ffmpeg_path()}"
                        + ("" if ffprobe_path() else " (ffprobe missing; using fallback probe)")
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            overall = "degraded"
            components.append(
                HealthComponent(name="ffmpeg", status="error", detail=str(exc)[:200])
            )

        # Providers -- report honestly when a mock is in use.
        components.append(
            HealthComponent(
                name="llm",
                status="ok" if settings.llm_provider != "mock" else "degraded",
                detail=(
                    f"provider={settings.llm_provider}"
                    + (
                        "; agents will use deterministic fallbacks"
                        if settings.llm_provider == "mock"
                        else f", model={settings.llm_model}"
                    )
                ),
            )
        )
        components.append(
            HealthComponent(
                name="transcription",
                status="ok" if settings.transcription_provider != "mock" else "degraded",
                detail=(
                    f"provider={settings.transcription_provider}"
                    + (
                        "; transcripts will be structural only"
                        if settings.transcription_provider == "mock"
                        else ""
                    )
                ),
            )
        )

        return HealthResponse(
            status=overall,
            version=__version__,
            environment=settings.environment,
            components=components,
        )

    @app.get("/", tags=["system"], include_in_schema=False)
    async def root() -> dict:
        return {
            "name": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    app.include_router(api_router, prefix=settings.api_v1_prefix)
    app.include_router(media.router)
    return app


app = create_app()
