"""Task dispatch with a documented local fallback.

In a normal deployment tasks go to Redis and are picked up by a Celery worker.
For local development without a broker, ``dispatch`` runs the task on a small
background thread pool instead: the API still returns immediately, the job row
still tracks progress, and the dashboard behaves identically. The mode in use
is logged once at startup and reported by ``GET /health``.

This is a development convenience, not a production queue: there is no
durability, no retry across restarts and no cross-process fairness. Set
``CELERY_BROKER_URL`` (or ``REDIS_URL``) to a reachable broker for anything
real.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

DispatchMode = Literal["broker", "eager", "thread"]

_lock = threading.Lock()
_mode: DispatchMode | None = None
_executor: ThreadPoolExecutor | None = None


def _probe_broker() -> bool:
    """Is the configured broker reachable right now?"""
    url = settings.broker_url
    if not url.startswith("redis"):
        # Non-Redis brokers (RabbitMQ, SQS) are assumed to be intentional
        # production configuration; do not second-guess them.
        return True
    try:
        import redis

        client = redis.from_url(url, socket_connect_timeout=1.5, socket_timeout=1.5)
        client.ping()
        client.close()
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
        logger.warning(
            "task broker is not reachable; falling back to in-process execution",
            extra={"broker": url.split("@")[-1], "error": str(exc)},
        )
        return False


def resolve_mode(*, refresh: bool = False) -> DispatchMode:
    global _mode
    with _lock:
        if _mode is not None and not refresh:
            return _mode
        if settings.celery_task_always_eager:
            _mode = "eager"
        elif _probe_broker():
            _mode = "broker"
        else:
            _mode = "thread"
        logger.info("task dispatch mode selected", extra={"mode": _mode})
        return _mode


def _thread_pool() -> ThreadPoolExecutor:
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="inline-task"
            )
        return _executor


def dispatch(task: Any, *args: Any, **kwargs: Any) -> str | None:
    """Send ``task`` for execution. Returns the Celery task id when there is one."""
    mode = resolve_mode()
    if mode == "broker":
        async_result = task.apply_async(args=args, kwargs=kwargs)
        return str(async_result.id)
    if mode == "eager":
        # Celery's own eager mode: synchronous, inside this process.
        result = task.apply(args=args, kwargs=kwargs)
        return str(result.id)

    def _run() -> None:
        try:
            task.apply(args=args, kwargs=kwargs)
        except Exception:  # noqa: BLE001 - the task wrapper already records failures
            logger.exception("inline task raised", extra={"task": task.name})

    _thread_pool().submit(_run)
    return None


def shutdown() -> None:
    global _executor
    with _lock:
        if _executor is not None:
            _executor.shutdown(wait=False, cancel_futures=False)
            _executor = None
