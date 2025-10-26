"""
Unified API endpoints for audio processing.
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
from typing import Dict, Any, List

from .config import get_settings, Settings
from .schemas.unified_models import (
    UnifiedProcessRequest,
    UnifiedProcessResponse, 
    BatchProcessRequest,
    BatchProcessResponse,
    ProcessingMode
)
from .celery_app import celery_app
from .unified_celery_tasks import (
    unified_process_task,
    unified_batch_task,
    unified_process_video_task,
    unified_process_audio_task
)
from .utils.logging import get_logger, request_logger

logger = get_logger(__name__)

# Create rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create router
router = APIRouter(prefix="/unified", tags=["unified"])

# Get settings
settings = get_settings()


@router.post("/process", response_model=UnifiedProcessResponse)
@limiter.limit("10/minute")
async def unified_process_audio(
    request: Request, 
    process_request: UnifiedProcessRequest
):
    """
    Unified audio processing endpoint.
    
    This endpoint can extract:
    - Only aggregated features (traditional AudioProcessor behavior)
    - Only per-segment sequences 
    - Both aggregated features AND per-segment sequences
    
    Args:
        process_request: Unified processing request
        
    Returns:
        Unified processing response with task ID
    """
    try:
        # Determine input URI
        input_uri = process_request.audio_uri or process_request.video_uri
        if not input_uri:
            raise HTTPException(
                status_code=400,
                detail="Either audio_uri or video_uri must be provided"
            )
        
        # Validate URI format
        if not (input_uri.startswith("s3://") or 
                input_uri.endswith(('.wav', '.mp3', '.flac', '.m4a', '.mp4', '.avi', '.mov', '.mkv'))):
            raise HTTPException(
                status_code=400,
                detail="Invalid URI format. Must be S3 URI or local audio/video file"
            )
        
        # Determine processing mode
        processing_mode = process_request.processing_mode
        if process_request.aggregates_only:
            processing_mode = ProcessingMode.AGGREGATES_ONLY
        
        # Choose appropriate task based on input type
        if process_request.audio_uri:
            task = celery_app.send_task(
                'src.unified_celery_tasks.unified_process_audio_task',
                args=[
                    process_request.video_id,
                    process_request.audio_uri,
                    processing_mode.value,
                    process_request.segment_config,
                    process_request.extractor_names,
                    process_request.output_dir,
                    process_request.task_id,
                    process_request.dataset,
                    process_request.meta
                ]
            )
            message = "Unified audio processing request accepted"
        else:
            task = celery_app.send_task(
                'src.unified_celery_tasks.unified_process_video_task',
                args=[
                    process_request.video_id,
                    process_request.video_uri,
                    processing_mode.value,
                    process_request.segment_config,
                    process_request.extractor_names,
                    process_request.output_dir,
                    process_request.task_id,
                    process_request.dataset,
                    process_request.meta
                ]
            )
            message = "Unified video processing request accepted"
        
        logger.info(f"Submitted unified processing task {task.id} for video_id: {process_request.video_id}, "
                   f"mode: {processing_mode.value}")
        
        return UnifiedProcessResponse(
            accepted=True,
            celery_task_id=task.id,
            message=message,
            processing_mode=processing_mode
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in unified processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/batch", response_model=BatchProcessResponse)
@limiter.limit("5/minute")
async def unified_batch_process(
    request: Request,
    batch_request: BatchProcessRequest
):
    """
    Unified batch processing endpoint.
    
    Process multiple videos in batch with unified processing.
    
    Args:
        batch_request: Batch processing request
        
    Returns:
        Batch processing response with task ID
    """
    try:
        # Validate videos list
        if not batch_request.videos:
            raise HTTPException(
                status_code=400,
                detail="Videos list cannot be empty"
            )
        
        # Validate each video
        for i, video in enumerate(batch_request.videos):
            if "video_id" not in video:
                raise HTTPException(
                    status_code=400,
                    detail=f"Video {i}: video_id is required"
                )
            
            if "audio_uri" not in video and "video_uri" not in video:
                raise HTTPException(
                    status_code=400,
                    detail=f"Video {i}: Either audio_uri or video_uri must be provided"
                )
        
        # Submit batch task
        task = celery_app.send_task(
            'src.unified_celery_tasks.unified_batch_task',
            args=[
                batch_request.videos,
                batch_request.processing_mode.value,
                batch_request.segment_config,
                batch_request.extractor_names,
                batch_request.output_dir,
                None,  # task_id
                "batch"  # dataset
            ]
        )
        
        logger.info(f"Submitted unified batch processing task {task.id} for {len(batch_request.videos)} videos, "
                   f"mode: {batch_request.processing_mode.value}")
        
        return BatchProcessResponse(
            accepted=True,
            celery_task_id=task.id,
            total_videos=len(batch_request.videos),
            processing_mode=batch_request.processing_mode,
            message=f"Unified batch processing started for {len(batch_request.videos)} videos"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in unified batch processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/task/{task_id}")
async def get_unified_task_status(task_id: str):
    """
    Get the status of a unified processing task.
    
    Args:
        task_id: The task identifier
        
    Returns:
        Task status information
    """
    try:
        # Get task result from Celery
        task_result = celery_app.AsyncResult(task_id)
        
        if task_result.state == 'PENDING':
            status = "pending"
            progress = 0.0
            result = None
            error = None
        elif task_result.state == 'PROGRESS':
            status = "processing"
            progress = task_result.info.get('progress', 0.0)
            result = task_result.info
            error = None
        elif task_result.state == 'SUCCESS':
            status = "completed"
            progress = 100.0
            result = task_result.result
            error = None
        elif task_result.state == 'FAILURE':
            status = "failed"
            progress = 0.0
            result = None
            error = str(task_result.info)
        else:
            status = "pending"
            progress = 0.0
            result = None
            error = None
        
        return {
            "task_id": task_id,
            "status": status,
            "progress": progress,
            "result": result,
            "error": error
        }
        
    except Exception as e:
        logger.error(f"Error getting unified task status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/config")
async def get_unified_config():
    """
    Get default unified processing configuration.
    
    Returns:
        Default configuration for unified processing
    """
    try:
        from .segment_config import get_default_config
        
        config = get_default_config()
        
        from .extractors import discover_extractors
        
        # Get available extractors
        extractors = discover_extractors()
        
        return {
            "default_segment_config": config.to_dict(),
            "available_processing_modes": [mode.value for mode in ProcessingMode],
            "available_extractors": [
                {
                    "name": extractor.name,
                    "version": extractor.version,
                    "description": getattr(extractor, 'description', 'No description available'),
                    "category": getattr(extractor, 'category', 'unknown')
                }
                for extractor in extractors
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting unified config: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/examples")
async def get_unified_examples():
    """
    Get example requests for unified processing.
    
    Returns:
        Example requests for different processing modes
    """
    return {
        "aggregates_only": {
            "video_id": "example_video_001",
            "audio_uri": "s3://bucket/audio.wav",
            "processing_mode": "aggregates_only",
            "extractor_names": ["clap_extractor", "loudness_extractor", "vad_extractor"],
            "dataset": "training"
        },
        "segments_only": {
            "video_id": "example_video_002",
            "video_uri": "s3://bucket/video.mp4",
            "processing_mode": "segments_only",
            "segment_config": {
                "segment_len": 3.0,
                "hop": 1.5,
                "max_seq_len": 128,
                "k_start": 16,
                "k_end": 16
            },
            "dataset": "training"
        },
        "both": {
            "video_id": "example_video_003",
            "audio_uri": "s3://bucket/audio.wav",
            "processing_mode": "both",
            "segment_config": {
                "segment_len": 3.0,
                "hop": 1.5,
                "max_seq_len": 64,
                "pca_dims": {
                    "clap": 128,
                    "wav2vec": 64
                }
            },
            "extractor_names": ["clap_extractor", "loudness_extractor", "vad_extractor", "advanced_embeddings"],
            "dataset": "training"
        },
        "batch_example": {
            "videos": [
                {
                    "video_id": "batch_video_001",
                    "audio_uri": "s3://bucket/audio1.wav"
                },
                {
                    "video_id": "batch_video_002",
                    "video_uri": "s3://bucket/video2.mp4"
                }
            ],
            "processing_mode": "both",
            "segment_config": {
                "segment_len": 3.0,
                "max_seq_len": 128
            }
        }
    }
