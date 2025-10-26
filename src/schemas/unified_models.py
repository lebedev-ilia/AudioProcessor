"""
Pydantic models for unified audio processing API.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class ProcessingMode(str, Enum):
    """Processing mode enumeration."""
    AGGREGATES_ONLY = "aggregates_only"
    SEGMENTS_ONLY = "segments_only" 
    BOTH = "both"


class UnifiedProcessRequest(BaseModel):
    """Request model for unified audio processing."""
    video_id: str = Field(..., description="Unique video identifier")
    audio_uri: Optional[str] = Field(None, description="S3 URI to audio file")
    video_uri: Optional[str] = Field(None, description="S3 URI to video file (alternative to audio_uri)")
    task_id: Optional[str] = Field(None, description="Optional task identifier")
    dataset: str = Field(default="default", description="Dataset name")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Unified processing parameters
    processing_mode: ProcessingMode = Field(default=ProcessingMode.AGGREGATES_ONLY, description="Processing mode")
    aggregates_only: bool = Field(default=True, description="Extract only aggregated features (deprecated, use processing_mode)")
    
    # Segment processing parameters (used when processing_mode != AGGREGATES_ONLY)
    segment_config: Optional[Dict[str, Any]] = Field(None, description="Configuration for segment processing")
    
    # Extractor selection
    extractor_names: Optional[List[str]] = Field(None, description="List of extractor names to use (None for all)")
    
    # Output configuration
    output_dir: Optional[str] = Field(None, description="Output directory for results")
    
    @validator('audio_uri', 'video_uri')
    def validate_uri_fields(cls, v, values):
        """Validate that either audio_uri or video_uri is provided, but not both."""
        audio_uri = values.get('audio_uri')
        video_uri = values.get('video_uri')
        
        # If this is the second field being validated, check both
        if 'video_uri' in values:
            if not audio_uri and not video_uri:
                raise ValueError('Either audio_uri or video_uri must be provided')
            if audio_uri and video_uri:
                raise ValueError('Cannot provide both audio_uri and video_uri')
        
        return v
    
    @validator('processing_mode')
    def validate_processing_mode(cls, v, values):
        """Validate processing mode consistency."""
        aggregates_only = values.get('aggregates_only', True)
        
        # Handle backward compatibility
        if aggregates_only and v == ProcessingMode.BOTH:
            return ProcessingMode.AGGREGATES_ONLY
        elif not aggregates_only and v == ProcessingMode.AGGREGATES_ONLY:
            return ProcessingMode.BOTH
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "video_123",
                "audio_uri": "s3://bucket/audio.wav",
                "processing_mode": "both",
                "segment_config": {
                    "segment_len": 3.0,
                    "hop": 1.5,
                    "max_seq_len": 128
                },
                "extractor_names": ["clap_extractor", "loudness_extractor", "vad_extractor"],
                "dataset": "training",
                "meta": {
                    "duration": 120,
                    "quality": "high"
                }
            }
        }


class UnifiedProcessResponse(BaseModel):
    """Response model for unified audio processing request."""
    accepted: bool = Field(..., description="Whether request was accepted")
    celery_task_id: str = Field(..., description="Celery task identifier")
    message: Optional[str] = Field(None, description="Additional message")
    processing_mode: ProcessingMode = Field(..., description="Processing mode that will be used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "accepted": True,
                "celery_task_id": "celery-task-789",
                "message": "Unified audio processing started",
                "processing_mode": "both"
            }
        }


class UnifiedTaskResult(BaseModel):
    """Result model for unified processing task."""
    video_id: str = Field(..., description="Video identifier")
    success: bool = Field(..., description="Whether processing was successful")
    processing_mode: ProcessingMode = Field(..., description="Processing mode used")
    
    # Aggregated features results
    aggregates_extracted: bool = Field(default=False, description="Whether aggregated features were extracted")
    manifest_path: Optional[str] = Field(None, description="Path to manifest file with aggregated features")
    extractor_results: Optional[List[Dict[str, Any]]] = Field(None, description="Extractor results")
    
    # Segment features results
    segments_extracted: bool = Field(default=False, description="Whether segment features were extracted")
    segment_files: Dict[str, str] = Field(default_factory=dict, description="Paths to segment feature files")
    num_segments: Optional[int] = Field(None, description="Number of segments created")
    num_selected_segments: Optional[int] = Field(None, description="Number of segments selected")
    feature_shape: Optional[List[int]] = Field(None, description="Shape of feature matrix")
    
    # Error information
    error: Optional[str] = Field(None, description="Error message if processing failed")
    
    # Timing information
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")
    created_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "video_123",
                "success": True,
                "processing_mode": "both",
                "aggregates_extracted": True,
                "manifest_path": "/output/video_123_manifest.json",
                "segments_extracted": True,
                "segment_files": {
                    "features_file": "/output/video_123_features.npy",
                    "mask_file": "/output/video_123_mask.npy",
                    "meta_file": "/output/video_123_meta.json"
                },
                "num_segments": 20,
                "num_selected_segments": 16,
                "feature_shape": [16, 256],
                "processing_time": 45.2
            }
        }


class BatchProcessRequest(BaseModel):
    """Request model for batch processing."""
    videos: List[Dict[str, Any]] = Field(..., description="List of video data")
    processing_mode: ProcessingMode = Field(default=ProcessingMode.AGGREGATES_ONLY, description="Processing mode")
    segment_config: Optional[Dict[str, Any]] = Field(None, description="Configuration for segment processing")
    extractor_names: Optional[List[str]] = Field(None, description="List of extractor names to use")
    output_dir: Optional[str] = Field(None, description="Output directory for results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "videos": [
                    {
                        "video_id": "video_001",
                        "audio_uri": "s3://bucket/audio1.wav"
                    },
                    {
                        "video_id": "video_002", 
                        "video_uri": "s3://bucket/video2.mp4"
                    }
                ],
                "processing_mode": "both",
                "segment_config": {
                    "segment_len": 3.0,
                    "max_seq_len": 64
                }
            }
        }


class BatchProcessResponse(BaseModel):
    """Response model for batch processing."""
    accepted: bool = Field(..., description="Whether batch request was accepted")
    celery_task_id: str = Field(..., description="Celery task identifier")
    total_videos: int = Field(..., description="Total number of videos in batch")
    processing_mode: ProcessingMode = Field(..., description="Processing mode")
    message: Optional[str] = Field(None, description="Additional message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "accepted": True,
                "celery_task_id": "batch-task-456",
                "total_videos": 10,
                "processing_mode": "both",
                "message": "Batch processing started"
            }
        }


class BatchTaskResult(BaseModel):
    """Result model for batch processing task."""
    total_videos: int = Field(..., description="Total number of videos")
    successful: int = Field(..., description="Number of successfully processed videos")
    failed: int = Field(..., description="Number of failed videos")
    processing_mode: ProcessingMode = Field(..., description="Processing mode used")
    results: List[UnifiedTaskResult] = Field(..., description="Individual video results")
    total_processing_time: Optional[float] = Field(None, description="Total processing time")
    created_at: Optional[datetime] = Field(None, description="Batch start time")
    completed_at: Optional[datetime] = Field(None, description="Batch completion time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_videos": 10,
                "successful": 8,
                "failed": 2,
                "processing_mode": "both",
                "results": [
                    {
                        "video_id": "video_001",
                        "success": True,
                        "aggregates_extracted": True,
                        "segments_extracted": True
                    }
                ],
                "total_processing_time": 120.5
            }
        }
