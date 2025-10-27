"""
Unified API endpoints for async GPU-optimized audio processing.
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Query
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
from typing import Dict, Any, List, Optional

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
    async_unified_process_task,
    async_unified_batch_task,
    async_unified_process_video_task,
    async_unified_process_audio_task
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
    process_request: UnifiedProcessRequest,
    max_cpu_workers: int = Query(8, description="Maximum CPU workers for extractors"),
    max_gpu_workers: int = Query(2, description="Maximum GPU workers for extractors"),
    max_io_workers: int = Query(16, description="Maximum I/O workers for file operations"),
    gpu_batch_size: int = Query(8, description="Batch size for GPU processing")
):
    """
    Unified async audio processing endpoint with GPU optimization.
    
    This endpoint can extract:
    - Only aggregated features (traditional AudioProcessor behavior)
    - Only per-segment sequences 
    - Both aggregated features AND per-segment sequences
    
    All extractors run in parallel for maximum performance with GPU optimization.
    
    Args:
        process_request: Unified processing request
        max_cpu_workers: Maximum CPU workers for extractors
        max_gpu_workers: Maximum GPU workers for extractors
        max_io_workers: Maximum I/O workers for file operations
        gpu_batch_size: Batch size for GPU processing
        
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
                'src.unified_celery_tasks.async_unified_process_audio_task',
                args=[
                    process_request.video_id,
                    process_request.audio_uri,
                    processing_mode.value,
                    process_request.segment_config,
                    process_request.extractor_names,
                    process_request.output_dir,
                    process_request.task_id,
                    process_request.dataset,
                    process_request.meta,
                    max_cpu_workers,
                    max_gpu_workers,
                    max_io_workers,
                    gpu_batch_size
                ]
            )
            message = "Unified async audio processing request accepted"
        else:
            task = celery_app.send_task(
                'src.unified_celery_tasks.async_unified_process_video_task',
                args=[
                    process_request.video_id,
                    process_request.video_uri,
                    processing_mode.value,
                    process_request.segment_config,
                    process_request.extractor_names,
                    process_request.output_dir,
                    process_request.task_id,
                    process_request.dataset,
                    process_request.meta,
                    max_cpu_workers,
                    max_gpu_workers,
                    max_io_workers,
                    gpu_batch_size
                ]
            )
            message = "Unified async video processing request accepted"
        
        logger.info(f"Submitted unified async processing task {task.id} for video_id: {process_request.video_id}, "
                   f"mode: {processing_mode.value}, workers: CPU={max_cpu_workers}, GPU={max_gpu_workers}")
        
        return UnifiedProcessResponse(
            accepted=True,
            celery_task_id=task.id,
            message=message,
            processing_mode=processing_mode
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in unified async processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/batch", response_model=BatchProcessResponse)
@limiter.limit("5/minute")
async def unified_batch_process(
    request: Request,
    batch_request: BatchProcessRequest,
    max_concurrent_videos: int = Query(4, description="Maximum concurrent video processing"),
    max_cpu_workers: int = Query(8, description="Maximum CPU workers for extractors"),
    max_gpu_workers: int = Query(2, description="Maximum GPU workers for extractors"),
    max_io_workers: int = Query(16, description="Maximum I/O workers for file operations"),
    gpu_batch_size: int = Query(8, description="Batch size for GPU processing")
):
    """
    Unified async batch processing endpoint with GPU optimization.
    
    Process multiple videos concurrently with parallel extractors.
    
    Args:
        batch_request: Batch processing request
        max_concurrent_videos: Maximum concurrent video processing
        max_cpu_workers: Maximum CPU workers for extractors
        max_gpu_workers: Maximum GPU workers for extractors
        max_io_workers: Maximum I/O workers for file operations
        gpu_batch_size: Batch size for GPU processing
        
    Returns:
        Batch processing response with task ID
    """
    try:
        # Validate batch request
        if not batch_request.videos:
            raise HTTPException(
                status_code=400,
                detail="Videos list cannot be empty"
            )
        
        if len(batch_request.videos) > 100:  # Reasonable limit
            raise HTTPException(
                status_code=400,
                detail="Too many videos in batch. Maximum 100 videos per batch."
            )
        
        # Determine processing mode
        processing_mode = batch_request.processing_mode
        if batch_request.aggregates_only:
            processing_mode = ProcessingMode.AGGREGATES_ONLY
        
        # Submit async batch task
        task = celery_app.send_task(
            'src.unified_celery_tasks.async_unified_batch_task',
            args=[
                batch_request.videos,
                processing_mode.value,
                batch_request.segment_config,
                batch_request.extractor_names,
                batch_request.output_dir,
                batch_request.task_id,
                batch_request.dataset,
                max_concurrent_videos,
                max_cpu_workers,
                max_gpu_workers,
                max_io_workers,
                gpu_batch_size
            ]
        )
        
        logger.info(f"Submitted unified async batch processing task {task.id} for {len(batch_request.videos)} videos, "
                   f"mode: {processing_mode.value}, concurrent: {max_concurrent_videos}, "
                   f"workers: CPU={max_cpu_workers}, GPU={max_gpu_workers}")
        
        return BatchProcessResponse(
            accepted=True,
            celery_task_id=task.id,
            message=f"Unified async batch processing request accepted for {len(batch_request.videos)} videos",
            processing_mode=processing_mode,
            total_videos=len(batch_request.videos)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in unified async batch processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/task/{task_id}")
async def get_unified_task_status(task_id: str):
    """
    Get the status of a unified async processing task.
    
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
            ],
            "async_processing": True,
            "gpu_optimization": True,
            "parallel_extractors": True,
            "default_max_cpu_workers": 8,
            "default_max_gpu_workers": 2,
            "default_max_io_workers": 16,
            "default_gpu_batch_size": 8,
            "default_max_concurrent_videos": 4
        }
        
    except Exception as e:
        logger.error(f"Error getting unified config: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/examples")
async def get_unified_examples():
    """
    Get example requests for unified async processing.
    
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
        },
        "query_parameters": {
            "max_cpu_workers": 8,
            "max_gpu_workers": 2,
            "max_io_workers": 16,
            "gpu_batch_size": 8,
            "max_concurrent_videos": 4
        }
    }