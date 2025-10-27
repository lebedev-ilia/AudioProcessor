"""
FastAPI application for AudioProcessor.
"""
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import time
from datetime import datetime
from typing import Dict, Any

from .config import get_settings, Settings
from .schemas.models import (
    ProcessRequest, 
    ProcessResponse, 
    HealthResponse, 
    ErrorResponse,
    TaskStatusResponse,
    TaskStatus
)
from .celery_app import celery_app
from .extractors import discover_extractors
from .monitor.metrics import get_metrics, metrics_collector
from .utils.logging import setup_logging, get_logger, request_logger
from .health import health_checker
from .unified_api import router as unified_router

# Setup structured logging
setup_logging()
logger = get_logger(__name__)

# Create rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="AudioProcessor",
    description="Микросервис для извлечения аудио признаков из медиафайлов",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Get settings
settings = get_settings()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include unified async API router
app.include_router(unified_router)

# Store startup time for uptime calculation
startup_time = time.time()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and record metrics."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Record metrics
    metrics_collector.record_request(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code),
        duration=process_time
    )
    
    # Log request with structured logging
    request_logger.log_request(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration=process_time,
        correlation_id=request.headers.get("X-Correlation-ID")
    )
    
    return response


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "service": "AudioProcessor",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Calculate uptime
        uptime = time.time() - startup_time
        
        # Run comprehensive health checks
        health_results = await health_checker.check_all()
        
        # Extract dependency statuses
        dependencies = {}
        for check_name, check_result in health_results["checks"].items():
            dependencies[check_name] = check_result["status"]
        
        # Determine overall status
        overall_status = health_results["status"]
        
        # Return appropriate HTTP status
        if overall_status == "unhealthy":
            raise HTTPException(status_code=503, detail="Service unhealthy")
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat() + "Z",
            version="1.0.0",
            uptime=uptime,
            dependencies=dependencies
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/process", response_model=ProcessResponse)
@limiter.limit("10/minute")
async def process_audio(request: Request, process_request: ProcessRequest):
    """
    Process audio or video file and extract features.
    
    This endpoint accepts a request to process an audio or video file and returns
    a task ID for tracking the processing status.
    """
    try:
        # Determine if we're processing audio or video
        if process_request.audio_uri:
            # Audio processing
            if not (process_request.audio_uri.startswith("s3://") or process_request.audio_uri.endswith(('.wav', '.mp3', '.flac', '.m4a'))):
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid audio_uri. Must be an S3 URI (s3://bucket/path) or local audio file"
                )
            
            # Submit audio processing task
            task = celery_app.send_task(
                'src.celery_app.process_audio_task',
                args=[process_request.video_id, process_request.audio_uri],
                kwargs={
                    'task_id': process_request.task_id,
                    'dataset': process_request.dataset,
                    'meta': process_request.meta
                }
            )
            
            message = "Audio processing request accepted"
            
        elif process_request.video_uri:
            # Video processing
            if not (process_request.video_uri.startswith("s3://") or process_request.video_uri.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'))):
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid video_uri. Must be an S3 URI (s3://bucket/path) or local video file"
                )
            
            # Submit video processing task
            task = celery_app.send_task(
                'src.celery_app.process_video_task',
                args=[process_request.video_id, process_request.video_uri],
                kwargs={
                    'task_id': process_request.task_id,
                    'dataset': process_request.dataset,
                    'meta': process_request.meta
                }
            )
            
            message = "Video processing request accepted"
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Either audio_uri or video_uri must be provided"
            )
        
        # Record task submission metrics
        metrics_collector.record_task("submitted", "audio_queue")
        
        logger.info(f"Submitted processing task {task.id} for video_id: {process_request.video_id}")
        
        return ProcessResponse(
            accepted=True,
            celery_task_id=task.id,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the status of a processing task.
    
    Args:
        task_id: The task identifier
        
    Returns:
        Task status information
    """
    try:
        # Get task result from Celery
        task_result = celery_app.AsyncResult(task_id)
        
        if task_result.state == 'PENDING':
            status = TaskStatus.PENDING
            progress = 0.0
            result = None
            error = None
        elif task_result.state == 'PROGRESS':
            status = TaskStatus.PROCESSING
            progress = task_result.info.get('progress', 0.0)
            result = task_result.info
            error = None
        elif task_result.state == 'SUCCESS':
            status = TaskStatus.COMPLETED
            progress = 100.0
            result = task_result.result
            error = None
        elif task_result.state == 'FAILURE':
            status = TaskStatus.FAILED
            progress = 0.0
            result = None
            error = str(task_result.info)
        else:
            status = TaskStatus.PENDING
            progress = 0.0
            result = None
            error = None
        
        return TaskStatusResponse(
            task_id=task_id,
            status=status,
            progress=progress,
            result=result,
            error=error,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/extractors")
async def list_extractors():
    """
    List available extractors.
    
    Returns:
        List of available extractors with their information
    """
    try:
        # Get actual extractors
        extractors = discover_extractors()
        
        extractor_info = []
        for extractor in extractors:
            extractor_info.append({
                "name": extractor.name,
                "version": extractor.version,
                "description": extractor.description,
                "status": "available"
            })
        
        return {"extractors": extractor_info}
        
    except Exception as e:
        logger.error(f"Error listing extractors: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check endpoint with full diagnostic information.
    
    Returns:
        Detailed health check results
    """
    try:
        health_results = await health_checker.check_all()
        return health_results
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/health/{check_name}")
async def specific_health_check(check_name: str):
    """
    Run a specific health check.
    
    Args:
        check_name: Name of the health check to run
        
    Returns:
        Specific health check result
    """
    try:
        result = await health_checker.check_specific(check_name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Specific health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/metrics")
async def get_prometheus_metrics():
    """
    Get Prometheus metrics.
    
    Returns:
        Prometheus metrics in text format
    """
    try:
        from fastapi.responses import PlainTextResponse
        metrics_data = get_metrics()
        return PlainTextResponse(content=metrics_data, media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.utcnow().isoformat() + "Z"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.utcnow().isoformat() + "Z"
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )
