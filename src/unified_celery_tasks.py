"""
Celery tasks for unified audio processing.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from celery import current_task
from datetime import datetime

from .celery_app import celery_app
from .unified_processor import UnifiedAudioProcessor
from .segment_config import SegmentConfig, create_config
from .schemas.unified_models import ProcessingMode, UnifiedTaskResult, BatchTaskResult
from .utils.logging import get_logger

logger = get_logger(__name__)


@celery_app.task(bind=True, name='src.unified_celery_tasks.unified_process_task')
def unified_process_task(
    self,
    video_id: str,
    input_uri: str,
    processing_mode: str = "aggregates_only",
    segment_config: Optional[Dict[str, Any]] = None,
    extractor_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    task_id: Optional[str] = None,
    dataset: str = "default",
    meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Unified audio processing task.
    
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
        
    Returns:
        Processing result dictionary
    """
    start_time = time.time()
    logger.info(f"Starting unified processing task for {video_id}")
    
    try:
        # Update task progress
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 0, 'status': 'Initializing...'}
        )
        
        # Create segment config if provided
        config = None
        if segment_config and processing_mode != "aggregates_only":
            config = create_config(**segment_config)
        
        # Initialize processor
        processor = UnifiedAudioProcessor(config)
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'status': 'Processor initialized'}
        )
        
        # Determine output directory
        if output_dir is None:
            output_dir = f"unified_output/{dataset}"
        
        # Process audio
        aggregates_only = (processing_mode == "aggregates_only")
        
        result = processor.process_audio(
            input_uri=input_uri,
            video_id=video_id,
            aggregates_only=aggregates_only,
            segment_config=segment_config,
            extractor_names=extractor_names,
            output_dir=output_dir
        )
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 90, 'status': 'Processing completed'}
        )
        
        # Add timing information
        processing_time = time.time() - start_time
        result.update({
            "processing_time": processing_time,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "processing_mode": processing_mode
        })
        
        # Update final progress
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 100, 'status': 'Task completed'}
        )
        
        logger.info(f"Unified processing completed for {video_id} in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        error_msg = f"Unified processing failed for {video_id}: {str(e)}"
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
            "processing_mode": processing_mode
        }


@celery_app.task(bind=True, name='src.unified_celery_tasks.unified_batch_task')
def unified_batch_task(
    self,
    videos: List[Dict[str, Any]],
    processing_mode: str = "aggregates_only",
    segment_config: Optional[Dict[str, Any]] = None,
    extractor_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    task_id: Optional[str] = None,
    dataset: str = "default"
) -> Dict[str, Any]:
    """
    Unified batch processing task.
    
    Args:
        videos: List of video data dictionaries
        processing_mode: Processing mode
        segment_config: Configuration for segment processing
        extractor_names: List of extractor names to use
        output_dir: Output directory for results
        task_id: Task identifier
        dataset: Dataset name
        
    Returns:
        Batch processing result dictionary
    """
    start_time = time.time()
    logger.info(f"Starting unified batch processing for {len(videos)} videos")
    
    try:
        # Update task progress
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 0, 'status': 'Initializing batch processing...'}
        )
        
        # Create segment config if provided
        config = None
        if segment_config and processing_mode != "aggregates_only":
            config = create_config(**segment_config)
        
        # Initialize processor
        processor = UnifiedAudioProcessor(config)
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 5, 'status': 'Processor initialized'}
        )
        
        # Determine output directory
        if output_dir is None:
            output_dir = f"unified_batch_output/{dataset}"
        
        # Process batch
        aggregates_only = (processing_mode == "aggregates_only")
        
        batch_result = processor.process_batch(
            video_data=videos,
            aggregates_only=aggregates_only,
            segment_config=segment_config,
            extractor_names=extractor_names,
            output_dir=output_dir
        )
        
        # Update progress
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 95, 'status': 'Batch processing completed'}
        )
        
        # Add timing information
        processing_time = time.time() - start_time
        batch_result.update({
            "total_processing_time": processing_time,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "processing_mode": processing_mode
        })
        
        # Update final progress
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 100, 'status': 'Batch task completed'}
        )
        
        logger.info(f"Unified batch processing completed: {batch_result['successful']} successful, "
                   f"{batch_result['failed']} failed in {processing_time:.2f}s")
        return batch_result
        
    except Exception as e:
        error_msg = f"Unified batch processing failed: {str(e)}"
        logger.error(error_msg)
        
        # Return error result
        return {
            "total_videos": len(videos),
            "successful": 0,
            "failed": len(videos),
            "error": error_msg,
            "processing_mode": processing_mode,
            "total_processing_time": time.time() - start_time,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "completed_at": datetime.utcnow().isoformat() + "Z"
        }


@celery_app.task(bind=True, name='src.unified_celery_tasks.unified_process_video_task')
def unified_process_video_task(
    self,
    video_id: str,
    video_uri: str,
    processing_mode: str = "aggregates_only",
    segment_config: Optional[Dict[str, Any]] = None,
    extractor_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    task_id: Optional[str] = None,
    dataset: str = "default",
    meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Unified video processing task (wrapper for video files).
    
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
        
    Returns:
        Processing result dictionary
    """
    # This is a wrapper that calls the main unified_process_task
    return unified_process_task.delay(
        video_id=video_id,
        input_uri=video_uri,
        processing_mode=processing_mode,
        segment_config=segment_config,
        extractor_names=extractor_names,
        output_dir=output_dir,
        task_id=task_id,
        dataset=dataset,
        meta=meta
    ).get()


@celery_app.task(bind=True, name='src.unified_celery_tasks.unified_process_audio_task')
def unified_process_audio_task(
    self,
    video_id: str,
    audio_uri: str,
    processing_mode: str = "aggregates_only",
    segment_config: Optional[Dict[str, Any]] = None,
    extractor_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    task_id: Optional[str] = None,
    dataset: str = "default",
    meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Unified audio processing task (wrapper for audio files).
    
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
        
    Returns:
        Processing result dictionary
    """
    # This is a wrapper that calls the main unified_process_task
    return unified_process_task.delay(
        video_id=video_id,
        input_uri=audio_uri,
        processing_mode=processing_mode,
        segment_config=segment_config,
        extractor_names=extractor_names,
        output_dir=output_dir,
        task_id=task_id,
        dataset=dataset,
        meta=meta
    ).get()
