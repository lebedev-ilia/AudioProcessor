"""
Celery application configuration for AudioProcessor.
"""
from celery import Celery
from config import get_settings

# Get settings
settings = get_settings()

# Create Celery app
app = Celery(
    'audio_processor',
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=['src.celery_app']
)

# Configure Celery
app.conf.update(
    task_serializer=settings.celery_task_serializer,
    result_serializer=settings.celery_result_serializer,
    accept_content=settings.celery_accept_content,
    timezone=settings.celery_timezone,
    enable_utc=settings.celery_enable_utc,
    
    # Task settings
    task_time_limit=settings.task_time_limit,
    task_soft_time_limit=settings.worker_timeout,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    
    # Result settings
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Queue settings
    task_routes={
        'src.celery_app.process_audio_task': {'queue': 'audio_queue'},
        'src.celery_app.process_audio_gpu_task': {'queue': 'gpu_queue'},
    },
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Error handling
    task_reject_on_worker_lost=True,
    task_ignore_result=False,
)

# Task definitions
@app.task(bind=True, name='src.celery_app.process_audio_task')
def process_audio_task(self, video_id: str, audio_uri: str, **kwargs):
    """
    Process audio file and extract features.
    
    Args:
        video_id: Unique video identifier
        audio_uri: S3 URI to audio file
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with processing results
    """
    import logging
    import time
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting audio processing for video_id: {video_id}")
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'progress': 0, 'status': 'Initializing...'}
        )
        
        # TODO: Implement actual processing logic
        # This is a placeholder implementation
        
        # Simulate processing steps
        steps = [
            "Downloading audio file",
            "Validating audio format", 
            "Extracting MFCC features",
            "Extracting Mel spectrogram",
            "Extracting Chroma features",
            "Extracting RMS/Loudness",
            "Running Voice Activity Detection",
            "Generating OpenL3 embeddings",
            "Uploading results to S3",
            "Creating manifest"
        ]
        
        for i, step in enumerate(steps):
            # Simulate processing time
            time.sleep(1)
            
            # Update progress
            progress = int((i + 1) / len(steps) * 100)
            self.update_state(
                state='PROGRESS',
                meta={
                    'progress': progress,
                    'status': step,
                    'current_step': i + 1,
                    'total_steps': len(steps)
                }
            )
            
            logger.info(f"Processing step {i + 1}/{len(steps)}: {step}")
        
        # Simulate final result
        result = {
            'status': 'completed',
            'video_id': video_id,
            'audio_uri': audio_uri,
            'manifest_uri': f"s3://bucket/manifests/{video_id}.json",
            'processing_time': len(steps),
            'extractors_used': [
                'mfcc_extractor',
                'mel_extractor', 
                'chroma_extractor',
                'loudness_extractor',
                'vad_extractor',
                'openl3_extractor'
            ],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Successfully completed audio processing for video_id: {video_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing audio for video_id {video_id}: {e}")
        
        # Update task state with error
        self.update_state(
            state='FAILURE',
            meta={
                'error': str(e),
                'status': 'Failed'
            }
        )
        
        raise


@app.task(bind=True, name='src.celery_app.process_audio_gpu_task')
def process_audio_gpu_task(self, video_id: str, audio_uri: str, **kwargs):
    """
    Process audio file using GPU-accelerated extractors.
    
    Args:
        video_id: Unique video identifier
        audio_uri: S3 URI to audio file
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with processing results
    """
    import logging
    import time
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting GPU-accelerated audio processing for video_id: {video_id}")
        
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'progress': 0, 'status': 'Initializing GPU processing...'}
        )
        
        # TODO: Implement actual GPU processing logic
        # This is a placeholder implementation
        
        # Simulate GPU processing steps
        steps = [
            "Initializing GPU",
            "Loading models to GPU",
            "Downloading audio file",
            "Extracting OpenL3 embeddings (GPU)",
            "Running ASR with Whisper (GPU)",
            "Processing embeddings",
            "Uploading results to S3",
            "Cleaning up GPU memory"
        ]
        
        for i, step in enumerate(steps):
            # Simulate processing time (GPU steps are faster)
            time.sleep(0.5)
            
            # Update progress
            progress = int((i + 1) / len(steps) * 100)
            self.update_state(
                state='PROGRESS',
                meta={
                    'progress': progress,
                    'status': step,
                    'current_step': i + 1,
                    'total_steps': len(steps),
                    'gpu_used': True
                }
            )
            
            logger.info(f"GPU processing step {i + 1}/{len(steps)}: {step}")
        
        # Simulate final result
        result = {
            'status': 'completed',
            'video_id': video_id,
            'audio_uri': audio_uri,
            'manifest_uri': f"s3://bucket/manifests/{video_id}_gpu.json",
            'processing_time': len(steps) * 0.5,
            'gpu_used': True,
            'extractors_used': [
                'openl3_extractor_gpu',
                'whisper_asr_gpu',
                'sentiment_extractor',
                'ner_extractor'
            ],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Successfully completed GPU audio processing for video_id: {video_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error in GPU processing for video_id {video_id}: {e}")
        
        # Update task state with error
        self.update_state(
            state='FAILURE',
            meta={
                'error': str(e),
                'status': 'Failed',
                'gpu_used': True
            }
        )
        
        raise


@app.task(name='src.celery_app.health_check_task')
def health_check_task():
    """
    Periodic health check task.
    
    Returns:
        Health status information
    """
    import logging
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    
    try:
        # TODO: Implement actual health checks
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'celery_workers': 1,  # TODO: Get actual worker count
            'queue_length': 0,    # TODO: Get actual queue length
            'memory_usage': 0,    # TODO: Get actual memory usage
        }
        
        logger.debug("Health check completed successfully")
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }


# Periodic tasks configuration
from celery.schedules import crontab

app.conf.beat_schedule = {
    'health-check-every-minute': {
        'task': 'src.celery_app.health_check_task',
        'schedule': 60.0,  # Run every minute
    },
}

app.conf.timezone = 'UTC'


if __name__ == '__main__':
    app.start()
