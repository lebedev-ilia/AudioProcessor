"""
Async Unified API endpoints for parallel audio processing.
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
from .async_unified_celery_tasks import (
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
router = APIRouter(prefix="/async", tags=["async"])

# Get settings
settings = get_settings()


@router.post("/process", response_model=UnifiedProcessResponse)
@limiter.limit("10/minute")
async def async_unified_process_audio(
    request: Request, 
    process_request: UnifiedProcessRequest,
    max_cpu_workers: int = 8,
    max_gpu_workers: int = 2,
    max_io_workers: int = 16,
    gpu_batch_size: int = 8
):
    """
    Async unified audio processing endpoint with parallel execution.
    
    This endpoint can extract:
    - Only aggregated features (traditional AudioProcessor behavior)
    - Only per-segment sequences 
    - Both aggregated features AND per-segment sequences
    
    All extractors run in parallel for maximum performance.
    
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
                'src.async_unified_celery_tasks.async_unified_process_audio_task',
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
            message = "Async unified audio processing request accepted"
        else:
            task = celery_app.send_task(
                'src.async_unified_celery_tasks.async_unified_process_video_task',
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
            message = "Async unified video processing request accepted"
        
        logger.info(f"Submitted async unified processing task {task.id} for video_id: {process_request.video_id}, "
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
        logger.error(f"Error in async unified processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/batch", response_model=BatchProcessResponse)
@limiter.limit("5/minute")
async def async_unified_batch_process(
    request: Request,
    batch_request: BatchProcessRequest,
    max_concurrent_videos: int = 4,
    max_cpu_workers: int = 8,
    max_gpu_workers: int = 2,
    max_io_workers: int = 16,
    gpu_batch_size: int = 8
):
    """
    Async unified batch processing endpoint with parallel execution.
    
    Processes multiple videos concurrently with parallel extractors.
    
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
            'src.async_unified_celery_tasks.async_unified_batch_task',
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
        
        logger.info(f"Submitted async unified batch processing task {task.id} for {len(batch_request.videos)} videos, "
                   f"mode: {processing_mode.value}, concurrent: {max_concurrent_videos}, "
                   f"workers: CPU={max_cpu_workers}, GPU={max_gpu_workers}")
        
        return BatchProcessResponse(
            accepted=True,
            celery_task_id=task.id,
            message=f"Async unified batch processing request accepted for {len(batch_request.videos)} videos",
            processing_mode=processing_mode,
            total_videos=len(batch_request.videos)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in async unified batch processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/task/{task_id}")
async def get_async_task_status(task_id: str):
    """
    Get status of async unified processing task.
    
    Args:
        task_id: Celery task ID
        
    Returns:
        Task status and result information
    """
    try:
        # Get task result
        task_result = celery_app.AsyncResult(task_id)
        
        if task_result.state == 'PENDING':
            response = {
                'task_id': task_id,
                'state': task_result.state,
                'status': 'Task is waiting to be processed...'
            }
        elif task_result.state == 'PROGRESS':
            response = {
                'task_id': task_id,
                'state': task_result.state,
                'current': task_result.info.get('current', 0),
                'total': task_result.info.get('total', 1),
                'status': task_result.info.get('status', '')
            }
        elif task_result.state == 'SUCCESS':
            response = {
                'task_id': task_id,
                'state': task_result.state,
                'result': task_result.result
            }
        else:  # FAILURE
            response = {
                'task_id': task_id,
                'state': task_result.state,
                'error': str(task_result.info)
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting async task status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/config")
async def get_async_config():
    """
    Get async processing configuration and capabilities.
    
    Returns:
        Configuration information for async processing
    """
    return {
        "async_processing": True,
        "parallel_extractors": True,
        "parallel_segments": True,
        "parallel_batch": True,
        "default_max_cpu_workers": 8,
        "default_max_gpu_workers": 2,
        "default_max_io_workers": 16,
        "default_gpu_batch_size": 8,
        "default_max_concurrent_videos": 4,
        "supported_processing_modes": [
            "aggregates_only",
            "segments_only", 
            "both"
        ],
        "extractor_categories": {
            "cpu_extractors": [
                "mfcc_extractor",
                "mel_extractor", 
                "chroma_extractor",
                "loudness_extractor",
                "vad_extractor",
                "pitch_extractor",
                "spectral_extractor",
                "tempo_extractor",
                "quality_extractor",
                "onset_extractor",
                "voice_quality_extractor",
                "phoneme_analysis_extractor",
                "advanced_spectral_extractor",
                "music_analysis_extractor",
                "sound_event_detection_extractor",
                "rhythmic_analysis_extractor"
            ],
            "gpu_extractors": [
                "clap_extractor",
                "advanced_embeddings",
                "asr_extractor",
                "emotion_recognition_extractor",
                "source_separation_extractor",
                "speaker_diarization_extractor"
            ]
        },
        "performance_notes": {
            "cpu_extractors": "Run in parallel with ProcessPoolExecutor",
            "gpu_extractors": "Run in parallel with limited concurrency",
            "segments": "Process segments in parallel within each video",
            "batch": "Process multiple videos concurrently",
            "expected_speedup": "3-15x depending on workload and resources"
        }
    }


@router.get("/examples")
async def get_async_examples():
    """
    Get examples of async processing requests.
    
    Returns:
        Example requests for async processing
    """
    return {
        "single_audio_processing": {
            "endpoint": "/async/process",
            "method": "POST",
            "example_request": {
                "video_id": "example_001",
                "audio_uri": "s3://bucket/audio.wav",
                "processing_mode": "both",
                "segment_config": {
                    "segment_len": 3.0,
                    "hop": 1.5,
                    "max_seq_len": 32
                },
                "extractor_names": ["mfcc_extractor", "clap_extractor", "vad_extractor"]
            },
            "query_parameters": {
                "max_cpu_workers": 8,
                "max_gpu_workers": 2,
                "max_io_workers": 16,
                "gpu_batch_size": 8
            }
        },
        "batch_processing": {
            "endpoint": "/async/batch",
            "method": "POST", 
            "example_request": {
                "videos": [
                    {
                        "video_id": "batch_001",
                        "audio_uri": "s3://bucket/audio1.wav"
                    },
                    {
                        "video_id": "batch_002", 
                        "video_uri": "s3://bucket/video2.mp4"
                    }
                ],
                "processing_mode": "aggregates_only",
                "dataset": "example_dataset"
            },
            "query_parameters": {
                "max_concurrent_videos": 4,
                "max_cpu_workers": 8,
                "max_gpu_workers": 2,
                "max_io_workers": 16,
                "gpu_batch_size": 8
            }
        },
        "task_status": {
            "endpoint": "/async/task/{task_id}",
            "method": "GET",
            "example_url": "/async/task/12345-abcd-6789-efgh"
        }
    }
