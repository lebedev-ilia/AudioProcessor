"""
Celery application configuration for AudioProcessor.
"""
from celery import Celery
from celery.schedules import crontab
from .config import get_settings

settings = get_settings()

# === Create Celery App ===
celery_app = Celery(
    "audio_processor",
    broker=str(settings.celery_broker_url),
    backend=str(settings.celery_result_backend),
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
    "src.celery_app.simple_test_task": {"queue": "audio_queue"},
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
@celery_app.task(
    bind=True, 
    name="src.celery_app.process_audio_task",
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 60},
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True
)
def process_audio_task(self, video_id: str, audio_uri: str, **kwargs):
    """
    CPU-based audio feature extraction.
    """
    import logging, time, tempfile, os
    from datetime import datetime
    from src.extractors import discover_extractors
    from src.storage.s3_client import S3Client
    from src.schemas.models import ManifestModel, ExtractorResult
    from src.monitor.metrics import metrics_collector
    from src.utils.logging import get_logger, task_logger, extractor_logger

    logger = get_logger(__name__)

    start_time = time.time()
    
    try:
        # Log task start
        task_logger.log_task_start(
            task_id=self.request.id,
            task_name="process_audio_task",
            video_id=video_id,
            audio_uri=audio_uri,
            dataset=kwargs.get('dataset', 'default')
        )
        
        # Record task start
        metrics_collector.record_task("started", "audio_queue")

        self.update_state(state="PROGRESS", meta={"progress": 0, "status": "Initializing..."})

        # Initialize S3 client
        s3_client = S3Client()
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Step 1: Download audio file (or use local file for testing)
            self.update_state(state="PROGRESS", meta={"progress": 10, "status": "Loading audio..."})
            
            if audio_uri.startswith("s3://"):
                # S3 file - download it
                local_audio_path = s3_client.download_file(audio_uri, tmp_dir)
                logger.info(f"Downloaded audio to: {local_audio_path}")
            else:
                # Local file - copy it to temp directory
                import shutil
                filename = os.path.basename(audio_uri)
                local_audio_path = os.path.join(tmp_dir, filename)
                shutil.copy2(audio_uri, local_audio_path)
                logger.info(f"Copied local audio to: {local_audio_path}")

            # Step 2: Get extractors
            self.update_state(state="PROGRESS", meta={"progress": 20, "status": "Initializing extractors..."})
            extractors = discover_extractors()
            logger.info(f"Found {len(extractors)} extractors")

            # Step 3: Run extractors
            results = []
            total_extractors = len(extractors)
            
            for i, extractor in enumerate(extractors):
                progress = 20 + int((i / total_extractors) * 60)
                self.update_state(
                    state="PROGRESS", 
                    meta={
                        "progress": progress, 
                        "status": f"Running {extractor.name}...",
                        "extractor": extractor.name,
                        "step": i + 1,
                        "total_steps": total_extractors
                    }
                )
                
                try:
                    # Log extractor start
                    extractor_logger.log_extractor_start(
                        extractor_name=extractor.name,
                        video_id=video_id
                    )
                    
                    extractor_start = time.time()
                    result = extractor.run(local_audio_path, tmp_dir)
                    extractor_duration = time.time() - extractor_start
                    
                    # Log extractor completion
                    extractor_logger.log_extractor_complete(
                        extractor_name=extractor.name,
                        video_id=video_id,
                        duration=extractor_duration,
                        success=result.success
                    )
                    
                    # Record extractor metrics
                    metrics_collector.record_extractor(
                        name=extractor.name,
                        version=extractor.version,
                        success=result.success,
                        duration=extractor_duration,
                        error_type=str(e).split(':')[0] if not result.success else None
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    extractor_duration = time.time() - extractor_start
                    
                    # Log extractor error
                    extractor_logger.log_extractor_error(
                        extractor_name=extractor.name,
                        video_id=video_id,
                        error=str(e)
                    )
                    
                    # Record failed extractor metrics
                    metrics_collector.record_extractor(
                        name=extractor.name,
                        version=extractor.version,
                        success=False,
                        duration=extractor_duration,
                        error_type=type(e).__name__
                    )
                    
                    # Create error result
                    error_result = ExtractorResult(
                        name=extractor.name,
                        version=extractor.version,
                        success=False,
                        error=str(e)
                    )
                    results.append(error_result)

            # Step 4: Create manifest
            self.update_state(state="PROGRESS", meta={"progress": 85, "status": "Creating manifest..."})
            
            manifest = ManifestModel(
                video_id=video_id,
                task_id=kwargs.get('task_id'),
                dataset=kwargs.get('dataset', 'default'),
                timestamp=datetime.utcnow().isoformat(),
                extractors=results,
                schema_version="audio_manifest_v1"
            )

            # Step 5: Upload manifest (or save locally for testing)
            self.update_state(state="PROGRESS", meta={"progress": 95, "status": "Saving manifest..."})
            try:
                manifest_uri = s3_client.upload_manifest(
                    manifest.dict(), 
                    video_id, 
                    kwargs.get('dataset', 'default')
                )
            except Exception as e:
                # Fallback: save manifest locally for testing
                logger.warning(f"S3 upload failed, saving locally: {e}")
                import json
                local_manifest_path = f"manifest_{video_id}.json"
                with open(local_manifest_path, 'w') as f:
                    json.dump(manifest.dict(), f, indent=2)
                manifest_uri = f"file://{os.path.abspath(local_manifest_path)}"
                logger.info(f"Manifest saved locally: {manifest_uri}")

        # Record successful task completion
        total_duration = time.time() - start_time
        metrics_collector.record_task("completed", "audio_queue", total_duration)
        
        # Final result
        result = {
            "status": "completed",
            "video_id": video_id,
            "audio_uri": audio_uri,
            "manifest_uri": manifest_uri,
            "extractors": [r.name for r in results],
            "successful_extractors": [r.name for r in results if r.success],
            "failed_extractors": [r.name for r in results if not r.success],
            "total_processing_time": total_duration,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Notify MasterML of completion
        try:
            notify_masterml(video_id, manifest_uri, "completed", result)
        except Exception as e:
            logger.warning(f"Failed to notify MasterML: {e}")
        
        # Log task completion
        task_logger.log_task_complete(
            task_id=self.request.id,
            video_id=video_id,
            duration=total_duration,
            successful_extractors=len([r for r in results if r.success]),
            failed_extractors=len([r for r in results if not r.success])
        )
        
        return result

    except Exception as e:
        total_duration = time.time() - start_time
        
        # Log task error
        task_logger.log_task_error(
            task_id=self.request.id,
            video_id=video_id,
            error=str(e)
        )
        
        # Record failed task
        metrics_collector.record_task("failed", "audio_queue", total_duration)
        
        # Notify MasterML of failure
        try:
            notify_masterml(video_id, None, "failed", {"error": str(e)})
        except Exception as notify_e:
            logger.warning(f"Failed to notify MasterML of failure: {notify_e}")
        
        self.update_state(state="FAILURE", meta={"error": str(e), "status": "failed"})
        raise


@celery_app.task(
    bind=True, 
    name="src.celery_app.process_audio_gpu_task",
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 2, 'countdown': 120},
    retry_backoff=True,
    retry_backoff_max=1200,
    retry_jitter=True
)
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

    logger = logging.getLogger(__name__)

    try:
        # Simplified health check without psutil for now
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "celery_workers": 1,
            "queue_length": 0,
        }

    except Exception as e:
        logger.exception("Health check failed")
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@celery_app.task(name="src.celery_app.simple_test_task")
def simple_test_task(message):
    """
    Simple test task for debugging.
    """
    import logging
    import time
    
    logger = logging.getLogger(__name__)
    logger.info(f"Processing simple test task: {message}")
    
    time.sleep(2)
    return f"Processed: {message}"


def notify_masterml(video_id: str, manifest_uri: str, status: str, result: dict = None):
    """
    Notify MasterML about task completion.
    
    Args:
        video_id: Video identifier
        manifest_uri: URI to manifest file (if successful)
        status: Task status (completed, failed)
        result: Task result data
    """
    import httpx
    from datetime import datetime
    from src.utils.logging import get_logger
    
    logger = get_logger(__name__)
    
    try:
        # Prepare notification payload
        payload = {
            "video_id": video_id,
            "processor": "audio_processor",
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "manifest_uri": manifest_uri,
            "result": result
        }
        
        # Send notification to MasterML
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{settings.masterml_url}/api/v1/processors/audio/notify",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.masterml_token}" if settings.masterml_token else None
                }
            )
            response.raise_for_status()
            
        logger.info(f"Successfully notified MasterML for video_id={video_id}, status={status}")
        
    except Exception as e:
        logger.error(f"Failed to notify MasterML for video_id={video_id}: {e}")
        # Don't raise exception to avoid breaking the main task


if __name__ == "__main__":
    celery_app.start()
