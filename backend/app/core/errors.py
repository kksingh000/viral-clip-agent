"""Typed application errors, each carrying a stable ``error_code``."""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for every error the application raises deliberately."""

    error_code = "internal_error"
    status_code = 500
    retryable = False

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "error_message": self.message,
            "details": self.details,
        }


class NotFoundError(AppError):
    error_code = "not_found"
    status_code = 404


class ConflictError(AppError):
    error_code = "conflict"
    status_code = 409


class ValidationFailure(AppError):
    error_code = "validation_error"
    status_code = 422


class AuthenticationError(AppError):
    error_code = "unauthenticated"
    status_code = 401


class ForbiddenError(AppError):
    error_code = "forbidden"
    status_code = 403


class RateLimitError(AppError):
    error_code = "rate_limited"
    status_code = 429
    retryable = True


class AuthorizationStatusError(AppError):
    """Raised when content rights do not permit the requested operation."""

    error_code = "content_not_authorized"
    status_code = 403


class ProviderError(AppError):
    error_code = "provider_error"
    status_code = 502
    retryable = True


class LLMResponseError(ProviderError):
    error_code = "llm_invalid_response"


class TranscriptionError(ProviderError):
    error_code = "transcription_failed"


class FFmpegError(AppError):
    error_code = "ffmpeg_failed"
    status_code = 500
    retryable = True

    def __init__(
        self,
        message: str,
        *,
        command: list[str] | None = None,
        stderr: str | None = None,
        returncode: int | None = None,
    ) -> None:
        super().__init__(
            message,
            details={
                "command": " ".join(command) if command else None,
                "returncode": returncode,
                "stderr": (stderr or "")[-4000:],
            },
        )


class MediaAnalysisError(AppError):
    error_code = "media_analysis_failed"
    status_code = 500
    retryable = True


class PipelineError(AppError):
    error_code = "pipeline_failed"
    status_code = 500
    retryable = True
