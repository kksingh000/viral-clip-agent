"""Celery application.

Heavy work (transcription, ffmpeg, model calls) runs here, never in the API
process. Queues are separated so a long render cannot starve short jobs.
"""

from __future__ import annotations

from celery import Celery
from celery.signals import setup_logging, task_postrun, task_prerun

from app.core.config import settings
from app.core.logging import configure_logging, get_logger, job_id_var

logger = get_logger(__name__)

celery_app = Celery(
    "viral_clip_agent",
    broker=settings.broker_url,
    backend=settings.result_backend,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    # A render or a transcription must not be handed to another worker just
    # because this one is busy; late ack plus no prefetch keeps distribution
    # fair and makes redelivery safe.
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    task_time_limit=3 * 3600,
    task_soft_time_limit=3 * 3600 - 300,
    result_expires=7 * 24 * 3600,
    broker_connection_retry_on_startup=True,
    task_always_eager=settings.celery_task_always_eager,
    task_eager_propagates=False,
    task_default_queue="default",
    task_routes={
        "app.workers.tasks.analyze_video_task": {"queue": "analysis"},
        "app.workers.tasks.generate_clips_task": {"queue": "render"},
        "app.workers.tasks.regenerate_clip_task": {"queue": "render"},
        "app.workers.tasks.generate_variants_task": {"queue": "render"},
        "app.workers.tasks.trend_discovery_task": {"queue": "discovery"},
        "app.workers.tasks.publish_clip_task": {"queue": "publishing"},
        "app.workers.tasks.sync_analytics_task": {"queue": "publishing"},
    },
)

celery_app.conf.beat_schedule = {
    "refresh-trending-metrics": {
        "task": "app.workers.tasks.refresh_trend_metrics_task",
        "schedule": 30 * 60.0,
    },
    "sync-published-analytics": {
        "task": "app.workers.tasks.sync_all_analytics_task",
        "schedule": 6 * 3600.0,
    },
    "reap-stale-jobs": {
        "task": "app.workers.tasks.reap_stale_jobs_task",
        "schedule": 15 * 60.0,
    },
}


@setup_logging.connect
def _configure_worker_logging(**_kwargs) -> None:
    configure_logging(settings.log_level, settings.log_json)


@task_prerun.connect
def _bind_job_id(task_id=None, args=None, **_kwargs) -> None:
    if args:
        job_id_var.set(str(args[0]))


@task_postrun.connect
def _clear_job_id(**_kwargs) -> None:
    job_id_var.set(None)
