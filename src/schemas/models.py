"""
Pydantic models for AudioProcessor.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessRequest(BaseModel):
    """Request model for audio processing."""
    video_id: str = Field(..., description="Unique video identifier")
    audio_uri: str = Field(..., description="S3 URI to audio file")
    task_id: Optional[str] = Field(None, description="Optional task identifier")
    dataset: str = Field(default="default", description="Dataset name")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "video_123",
                "audio_uri": "s3://bucket/audio.wav",
                "task_id": "task_456",
                "dataset": "training",
                "meta": {
                    "duration": 120,
                    "quality": "high"
                }
            }
        }


class ProcessResponse(BaseModel):
    """Response model for audio processing request."""
    accepted: bool = Field(..., description="Whether request was accepted")
    celery_task_id: str = Field(..., description="Celery task identifier")
    message: Optional[str] = Field(None, description="Additional message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "accepted": True,
                "celery_task_id": "celery-task-789",
                "message": "Audio processing started"
            }
        }


class ExtractorResult(BaseModel):
    """Result from a single extractor."""
    name: str = Field(..., description="Extractor name")
    version: str = Field(..., description="Extractor version")
    success: bool = Field(..., description="Whether extraction was successful")
    payload: Optional[Dict[str, Any]] = Field(None, description="Extracted features")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "mfcc_extractor",
                "version": "0.1.0",
                "success": True,
                "payload": {
                    "mfcc_mean": [0.1, 0.2, 0.3],
                    "mfcc_std": [0.05, 0.1, 0.15]
                },
                "processing_time": 2.5
            }
        }


class ManifestModel(BaseModel):
    """Manifest containing all extraction results."""
    video_id: str = Field(..., description="Video identifier")
    task_id: Optional[str] = Field(None, description="Task identifier")
    dataset: str = Field(..., description="Dataset name")
    timestamp: str = Field(..., description="Processing timestamp")
    extractors: List[ExtractorResult] = Field(..., description="List of extractor results")
    schema_version: str = Field(default="audio_manifest_v1", description="Schema version")
    total_processing_time: Optional[float] = Field(None, description="Total processing time")
    manifest_uri: Optional[str] = Field(None, description="S3 URI to manifest file")
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "video_123",
                "task_id": "task_456",
                "dataset": "training",
                "timestamp": "2023-10-25T10:30:00Z",
                "extractors": [
                    {
                        "name": "mfcc_extractor",
                        "version": "0.1.0",
                        "success": True,
                        "payload": {"mfcc_mean": [0.1, 0.2]}
                    }
                ],
                "schema_version": "audio_manifest_v1",
                "total_processing_time": 15.5
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Check timestamp")
    version: str = Field(..., description="Service version")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")
    dependencies: Dict[str, str] = Field(default_factory=dict, description="Dependency status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2023-10-25T10:30:00Z",
                "version": "1.0.0",
                "uptime": 3600.0,
                "dependencies": {
                    "redis": "healthy",
                    "s3": "healthy"
                }
            }
        }


class TaskStatusResponse(BaseModel):
    """Task status response."""
    task_id: str = Field(..., description="Task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: Optional[datetime] = Field(None, description="Task creation time")
    updated_at: Optional[datetime] = Field(None, description="Last update time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "celery-task-789",
                "status": "processing",
                "progress": 45.0,
                "created_at": "2023-10-25T10:30:00Z",
                "updated_at": "2023-10-25T10:32:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid audio file format",
                "detail": "File must be in WAV, MP3, or FLAC format",
                "timestamp": "2023-10-25T10:30:00Z",
                "request_id": "req-123"
            }
        }
