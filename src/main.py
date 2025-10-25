"""
FastAPI application for AudioProcessor.
"""
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
from .monitor.metrics import get_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AudioProcessor",
    description="Микросервис для извлечения аудио признаков из медиафайлов",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

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

# Store startup time for uptime calculation
startup_time = time.time()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
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
        
        # Check dependencies (placeholder for now)
        dependencies = {
            "redis": "healthy",  # TODO: Add actual Redis health check
            "s3": "healthy",     # TODO: Add actual S3 health check
        }
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat() + "Z",
            version="1.0.0",
            uptime=uptime,
            dependencies=dependencies
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/process", response_model=ProcessResponse)
async def process_audio(request: ProcessRequest):
    """
    Process audio file and extract features.
    
    This endpoint accepts a request to process an audio file and returns
    a task ID for tracking the processing status.
    """
    try:
        # Validate request (allow local files for testing)
        if not (request.audio_uri.startswith("s3://") or request.audio_uri.endswith(('.wav', '.mp3', '.flac', '.m4a'))):
            raise HTTPException(
                status_code=400, 
                detail="Invalid audio_uri. Must be an S3 URI (s3://bucket/path) or local audio file"
            )
        
        # Submit task to Celery
        task = celery_app.send_task(
            'src.celery_app.process_audio_task',
            args=[request.video_id, request.audio_uri],
            kwargs={
                'task_id': request.task_id,
                'dataset': request.dataset,
                'meta': request.meta
            }
        )
        
        logger.info(f"Submitted processing task {task.id} for video_id: {request.video_id}")
        
        return ProcessResponse(
            accepted=True,
            celery_task_id=task.id,
            message="Audio processing request accepted"
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
