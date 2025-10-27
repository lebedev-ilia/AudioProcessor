"""
Async Celery tasks for unified audio processing with parallel execution.
"""

import os
import json
import asyncio
import time
from typing import Dict, Any, List, Optional
from celery import current_task
from datetime import datetime

from .celery_app import celery_app
from .unified_processor import AsyncUnifiedAudioProcessor
from .segment_config import SegmentConfig, create_config
from .schemas.unified_models import ProcessingMode, UnifiedTaskResult, BatchTaskResult
from .utils.logging import get_logger

logger = get_logger(__name__)


@celery_app.task(bind=True, name='src.unified_celery_tasks.async_unified_process_task')
def async_unified_process_task(
    self,
    video_id: str,
    input_uri: str,
    processing_mode: str = "aggregates_only",
    segment_config: Optional[Dict[str, Any]] = None,
    extractor_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    task_id: Optional[str] = None,
    dataset: str = "default",
    meta: Optional[Dict[str, Any]] = None,
    max_cpu_workers: int = 8,
    max_gpu_workers: int = 2,
    max_io_workers: int = 16,
    gpu_batch_size: int = 8
) -> Dict[str, Any]:
    """
    Async unified audio processing task.
    
    Args:
        video_id: Video identifier
        input_uri: Path to audio/video file
        processing_mode: Processing mode ("aggregates_only", "segments_only", "both")
        segment_config: Configuration for segment processing
        extractor_names: List of extractor names to use
        output_dir: Output directory for results
        task_id: Task identifier
        dataset: Dataset name
        meta: Additional metadata
        max_cpu_workers: Maximum CPU workers for extractors
        max_gpu_workers: Maximum GPU workers for extractors
        max_io_workers: Maximum I/O workers for file operations
        gpu_batch_size: Batch size for GPU processing
        
    Returns:
        Processing result dictionary
    """
    start_time = time.time()
    logger.info(f"Starting async unified processing task for {video_id}")
    
    try:
        # Update task progress
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 0, 'status': 'Initializing async processor...'}
        )
        
        # Create segment config if provided
        config = None
        if segment_config and processing_mode != "aggregates_only":
            config = create_config(**segment_config)
        
        # Initialize async processor
        processor = AsyncUnifiedAudioProcessor(
            config=config,
            max_cpu_workers=max_cpu_workers,
            max_gpu_workers=max_gpu_workers,
            max_io_workers=max_io_workers,
            gpu_batch_size=gpu_batch_size
        )
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'status': 'Async processor initialized'}
        )
        
        # Determine output directory
        if output_dir is None:
            output_dir = f"async_unified_output/{dataset}"
        
        # Process audio asynchronously
        aggregates_only = (processing_mode == "aggregates_only")
        
        # Run async processing in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                processor.process_audio_async(
                    input_uri=input_uri,
                    video_id=video_id,
                    aggregates_only=aggregates_only,
                    segment_config=segment_config,
                    extractor_names=extractor_names,
                    output_dir=output_dir
                )
            )
        finally:
            loop.close()
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 90, 'status': 'Async processing completed'}
        )
        
        # Add timing information
        processing_time = time.time() - start_time
        result.update({
            "processing_time": processing_time,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "processing_mode": processing_mode,
            "async_processing": True,
            "max_cpu_workers": max_cpu_workers,
            "max_gpu_workers": max_gpu_workers
        })
        
        # Update final progress
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 100, 'status': 'Async task completed'}
        )
        
        logger.info(f"Async unified processing completed for {video_id} in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        error_msg = f"Async unified processing failed for {video_id}: {str(e)}"
        logger.error(error_msg)
        
        # Return error result
        return {
            "video_id": video_id,
            "success": False,
            "error": error_msg,
            "aggregates_extracted": False,
            "segments_extracted": False,
            "processing_time": time.time() - start_time,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "processing_mode": processing_mode,
            "async_processing": True
        }


@celery_app.task(bind=True, name='src.unified_celery_tasks.async_unified_batch_task')
def async_unified_batch_task(
    self,
    videos: List[Dict[str, Any]],
    processing_mode: str = "aggregates_only",
    segment_config: Optional[Dict[str, Any]] = None,
    extractor_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    task_id: Optional[str] = None,
    dataset: str = "default",
    max_concurrent_videos: int = 4,
    max_cpu_workers: int = 8,
    max_gpu_workers: int = 2,
    max_io_workers: int = 16,
    gpu_batch_size: int = 8
) -> Dict[str, Any]:
    """
    Async unified batch processing task.
    
    Args:
        videos: List of video data dictionaries
        processing_mode: Processing mode
        segment_config: Configuration for segment processing
        extractor_names: List of extractor names to use
        output_dir: Output directory for results
        task_id: Task identifier
        dataset: Dataset name
        max_concurrent_videos: Maximum concurrent video processing
        max_cpu_workers: Maximum CPU workers for extractors
        max_gpu_workers: Maximum GPU workers for extractors
        max_io_workers: Maximum I/O workers for file operations
        gpu_batch_size: Batch size for GPU processing
        
    Returns:
        Batch processing result dictionary
    """
    start_time = time.time()
    logger.info(f"Starting async unified batch processing for {len(videos)} videos")
    
    try:
        # Update task progress
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 0, 'status': 'Initializing async batch processing...'}
        )
        
        # Create segment config if provided
        config = None
        if segment_config and processing_mode != "aggregates_only":
            config = create_config(**segment_config)
        
        # Initialize async processor
        processor = AsyncUnifiedAudioProcessor(
            config=config,
            max_cpu_workers=max_cpu_workers,
            max_gpu_workers=max_gpu_workers,
            max_io_workers=max_io_workers,
            gpu_batch_size=gpu_batch_size
        )
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 5, 'status': 'Async processor initialized'}
        )
        
        # Determine output directory
        if output_dir is None:
            output_dir = f"async_unified_batch_output/{dataset}"
        
        # Process batch asynchronously
        aggregates_only = (processing_mode == "aggregates_only")
        
        # Run async batch processing in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            batch_result = loop.run_until_complete(
                processor.process_batch_async(
                    video_data=videos,
                    aggregates_only=aggregates_only,
                    segment_config=segment_config,
                    extractor_names=extractor_names,
                    output_dir=output_dir,
                    max_concurrent_videos=max_concurrent_videos
                )
            )
        finally:
            loop.close()
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 95, 'status': 'Async batch processing completed'}
        )
        
        # Add timing information
        processing_time = time.time() - start_time
        batch_result.update({
            "total_processing_time": processing_time,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "processing_mode": processing_mode,
            "async_processing": True,
            "max_concurrent_videos": max_concurrent_videos,
            "max_cpu_workers": max_cpu_workers,
            "max_gpu_workers": max_gpu_workers
        })
        
        # Update final progress
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 100, 'status': 'Async batch task completed'}
        )
        
        logger.info(f"Async unified batch processing completed: {batch_result['successful']} successful, "
                   f"{batch_result['failed']} failed in {processing_time:.2f}s")
        return batch_result
        
    except Exception as e:
        error_msg = f"Async unified batch processing failed: {str(e)}"
        logger.error(error_msg)
        
        # Return error result
        return {
            "total_videos": len(videos),
            "successful": 0,
            "failed": len(videos),
            "error": error_msg,
            "processing_mode": processing_mode,
            "async_processing": True,
            "total_processing_time": time.time() - start_time,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "completed_at": datetime.utcnow().isoformat() + "Z"
        }


@celery_app.task(bind=True, name='src.async_unified_celery_tasks.async_unified_process_video_task')
def async_unified_process_video_task(
    self,
    video_id: str,
    video_uri: str,
    processing_mode: str = "aggregates_only",
    segment_config: Optional[Dict[str, Any]] = None,
    extractor_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    task_id: Optional[str] = None,
    dataset: str = "default",
    meta: Optional[Dict[str, Any]] = None,
    max_cpu_workers: int = 8,
    max_gpu_workers: int = 2,
    max_io_workers: int = 16,
    gpu_batch_size: int = 8
) -> Dict[str, Any]:
    """
    Async unified video processing task (wrapper for video files).
    
    Args:
        video_id: Video identifier
        video_uri: Path to video file
        processing_mode: Processing mode
        segment_config: Configuration for segment processing
        extractor_names: List of extractor names to use
        output_dir: Output directory for results
        task_id: Task identifier
        dataset: Dataset name
        meta: Additional metadata
        max_cpu_workers: Maximum CPU workers for extractors
        max_gpu_workers: Maximum GPU workers for extractors
        max_io_workers: Maximum I/O workers for file operations
        gpu_batch_size: Batch size for GPU processing
        
    Returns:
        Processing result dictionary
    """
    # This is a wrapper that calls the main async_unified_process_task
    return async_unified_process_task.delay(
        video_id=video_id,
        input_uri=video_uri,
        processing_mode=processing_mode,
        segment_config=segment_config,
        extractor_names=extractor_names,
        output_dir=output_dir,
        task_id=task_id,
        dataset=dataset,
        meta=meta,
        max_cpu_workers=max_cpu_workers,
        max_gpu_workers=max_gpu_workers,
        max_io_workers=max_io_workers,
        gpu_batch_size=gpu_batch_size
    ).get()


@celery_app.task(bind=True, name='src.async_unified_celery_tasks.async_unified_process_audio_task')
def async_unified_process_audio_task(
    self,
    video_id: str,
    audio_uri: str,
    processing_mode: str = "aggregates_only",
    segment_config: Optional[Dict[str, Any]] = None,
    extractor_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    task_id: Optional[str] = None,
    dataset: str = "default",
    meta: Optional[Dict[str, Any]] = None,
    max_cpu_workers: int = 8,
    max_gpu_workers: int = 2,
    max_io_workers: int = 16,
    gpu_batch_size: int = 8
) -> Dict[str, Any]:
    """
    Async unified audio processing task (wrapper for audio files).
    
    Args:
        video_id: Video identifier
        audio_uri: Path to audio file
        processing_mode: Processing mode
        segment_config: Configuration for segment processing
        extractor_names: List of extractor names to use
        output_dir: Output directory for results
        task_id: Task identifier
        dataset: Dataset name
        meta: Additional metadata
        max_cpu_workers: Maximum CPU workers for extractors
        max_gpu_workers: Maximum GPU workers for extractors
        max_io_workers: Maximum I/O workers for file operations
        gpu_batch_size: Batch size for GPU processing
        
    Returns:
        Processing result dictionary
    """
    # This is a wrapper that calls the main async_unified_process_task
    return async_unified_process_task.delay(
        video_id=video_id,
        input_uri=audio_uri,
        processing_mode=processing_mode,
        segment_config=segment_config,
        extractor_names=extractor_names,
        output_dir=output_dir,
        task_id=task_id,
        dataset=dataset,
        meta=meta,
        max_cpu_workers=max_cpu_workers,
        max_gpu_workers=max_gpu_workers,
        max_io_workers=max_io_workers,
        gpu_batch_size=gpu_batch_size
    ).get()
