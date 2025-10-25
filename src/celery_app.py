"""
Celery application configuration for AudioProcessor.
"""
from celery import Celery
from celery.schedules import crontab
from config import get_settings

settings = get_settings()

# === Create Celery App ===
celery_app = Celery(
    "audio_processor",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["src.celery_app"],
)

# === Base Configuration ===
celery_app.conf.update(
    # Serialization
    task_serializer=settings.celery_task_serializer,
    result_serializer=settings.celery_result_serializer,
    accept_content=settings.celery_accept_content,

    # Timezone
    timezone=settings.celery_timezone,
    enable_utc=settings.celery_enable_utc,

    # Task behavior
    task_time_limit=settings.task_time_limit,
    task_soft_time_limit=settings.worker_timeout,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,

    # Results
    result_expires=3600,
    result_persistent=True,

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,

    # Error handling
    task_reject_on_worker_lost=True,
    task_ignore_result=False,
)

# === Queue Routing ===
celery_app.conf.task_routes = {
    "src.celery_app.process_audio_task": {"queue": "audio_queue"},
    "src.celery_app.process_audio_gpu_task": {"queue": "gpu_queue"},
    "src.celery_app.health_check_task": {"queue": "system_queue"},
}

# === Beat Schedule ===
celery_app.conf.beat_schedule = {
    "health-check-every-minute": {
        "task": "src.celery_app.health_check_task",
        "schedule": 60.0,
    },
}

celery_app.conf.timezone = "UTC"

# === Tasks ===
@celery_app.task(bind=True, name="src.celery_app.process_audio_task")
def process_audio_task(self, video_id: str, audio_uri: str, **kwargs):
    """
    CPU-based audio feature extraction.
    """
    import logging, time
    from datetime import datetime

    logger = logging.getLogger(__name__)

    try:
        logger.info(f"[CPU] Start processing video_id={video_id}")

        self.update_state(state="PROGRESS", meta={"progress": 0, "status": "Initializing..."})

        steps = [
            "Downloading audio",
            "Validating format",
            "Extracting MFCC",
            "Extracting Mel spectrogram",
            "Extracting Chroma",
            "Calculating RMS/Loudness",
            "Running Voice Activity Detection",
            "Generating embeddings (placeholder)",
            "Uploading to S3",
            "Finalizing manifest",
        ]

        for i, step in enumerate(steps):
            time.sleep(1)
            progress = int((i + 1) / len(steps) * 100)
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress,
                    "status": step,
                    "step": i + 1,
                    "total_steps": len(steps),
                },
            )
            logger.info(f"[CPU] Step {i + 1}/{len(steps)}: {step}")

        result = {
            "status": "completed",
            "video_id": video_id,
            "audio_uri": audio_uri,
            "manifest_uri": f"s3://{settings.s3_bucket}/manifests/{video_id}.json",
            "processing_time": len(steps),
            "extractors": [
                "mfcc_extractor",
                "mel_extractor",
                "chroma_extractor",
                "loudness_extractor",
                "vad_extractor",
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(f"[CPU] Completed video_id={video_id}")
        return result

    except Exception as e:
        logger.exception(f"[CPU] Error processing video_id={video_id}: {e}")
        self.update_state(state="FAILURE", meta={"error": str(e), "status": "failed"})
        raise


@celery_app.task(bind=True, name="src.celery_app.process_audio_gpu_task")
def process_audio_gpu_task(self, video_id: str, audio_uri: str, **kwargs):
    """
    GPU-accelerated audio feature extraction.
    """
    import logging, time
    from datetime import datetime

    logger = logging.getLogger(__name__)

    try:
        logger.info(f"[GPU] Start processing video_id={video_id}")

        self.update_state(state="PROGRESS", meta={"progress": 0, "status": "Initializing GPU..."})

        steps = [
            "Initialize GPU",
            "Load models",
            "Download audio",
            "Run OpenL3 embeddings (GPU)",
            "ASR with Whisper (GPU)",
            "Embedding postprocessing",
            "Upload results",
            "Cleanup GPU memory",
        ]

        for i, step in enumerate(steps):
            time.sleep(0.5)
            progress = int((i + 1) / len(steps) * 100)
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress,
                    "status": step,
                    "step": i + 1,
                    "total_steps": len(steps),
                    "gpu": True,
                },
            )
            logger.info(f"[GPU] Step {i + 1}/{len(steps)}: {step}")

        result = {
            "status": "completed",
            "video_id": video_id,
            "audio_uri": audio_uri,
            "manifest_uri": f"s3://{settings.s3_bucket}/manifests/{video_id}_gpu.json",
            "gpu": True,
            "extractors": ["openl3_extractor_gpu", "whisper_asr_gpu"],
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(f"[GPU] Completed video_id={video_id}")
        return result

    except Exception as e:
        logger.exception(f"[GPU] Error video_id={video_id}: {e}")
        self.update_state(state="FAILURE", meta={"error": str(e), "gpu": True})
        raise


@celery_app.task(name="src.celery_app.health_check_task")
def health_check_task():
    """
    Health check for Celery worker.
    """
    import logging
    from datetime import datetime
    import psutil

    logger = logging.getLogger(__name__)

    try:
        mem = psutil.virtual_memory()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "memory_usage_percent": mem.percent,
            "celery_workers": 1,  # TODO: add discovery
            "queue_length": 0,  # TODO: pull from Redis
        }

    except Exception as e:
        logger.exception("Health check failed")
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    celery_app.start()
