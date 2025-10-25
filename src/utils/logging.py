"""
Structured logging configuration for AudioProcessor.
"""
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import structlog
from ..config import get_settings

settings = get_settings()


class StructuredLogger:
    """Structured logger with JSON output."""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup structured logging configuration."""
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self.logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self.logger.debug(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with structured data."""
        self.logger.critical(message, **kwargs)


class RequestLogger:
    """Logger for HTTP requests with correlation ID."""
    
    def __init__(self):
        self.logger = StructuredLogger("request")
    
    def log_request(self, method: str, path: str, status_code: int, 
                   duration: float, correlation_id: str = None, **kwargs):
        """Log HTTP request with structured data."""
        self.logger.info(
            "HTTP request processed",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=round(duration * 1000, 2),
            correlation_id=correlation_id,
            **kwargs
        )
    
    def log_error(self, method: str, path: str, error: str, 
                 correlation_id: str = None, **kwargs):
        """Log HTTP request error."""
        self.logger.error(
            "HTTP request error",
            method=method,
            path=path,
            error=error,
            correlation_id=correlation_id,
            **kwargs
        )


class TaskLogger:
    """Logger for Celery tasks."""
    
    def __init__(self):
        self.logger = StructuredLogger("task")
    
    def log_task_start(self, task_id: str, task_name: str, video_id: str, **kwargs):
        """Log task start."""
        self.logger.info(
            "Task started",
            task_id=task_id,
            task_name=task_name,
            video_id=video_id,
            **kwargs
        )
    
    def log_task_progress(self, task_id: str, progress: float, status: str, **kwargs):
        """Log task progress."""
        self.logger.info(
            "Task progress",
            task_id=task_id,
            progress=progress,
            status=status,
            **kwargs
        )
    
    def log_task_complete(self, task_id: str, video_id: str, duration: float, **kwargs):
        """Log task completion."""
        self.logger.info(
            "Task completed",
            task_id=task_id,
            video_id=video_id,
            duration_seconds=duration,
            **kwargs
        )
    
    def log_task_error(self, task_id: str, video_id: str, error: str, **kwargs):
        """Log task error."""
        self.logger.error(
            "Task failed",
            task_id=task_id,
            video_id=video_id,
            error=error,
            **kwargs
        )


class ExtractorLogger:
    """Logger for extractor operations."""
    
    def __init__(self):
        self.logger = StructuredLogger("extractor")
    
    def log_extractor_start(self, extractor_name: str, video_id: str, **kwargs):
        """Log extractor start."""
        self.logger.info(
            "Extractor started",
            extractor_name=extractor_name,
            video_id=video_id,
            **kwargs
        )
    
    def log_extractor_complete(self, extractor_name: str, video_id: str, 
                              duration: float, success: bool, **kwargs):
        """Log extractor completion."""
        self.logger.info(
            "Extractor completed",
            extractor_name=extractor_name,
            video_id=video_id,
            duration_seconds=duration,
            success=success,
            **kwargs
        )
    
    def log_extractor_error(self, extractor_name: str, video_id: str, 
                           error: str, **kwargs):
        """Log extractor error."""
        self.logger.error(
            "Extractor failed",
            extractor_name=extractor_name,
            video_id=video_id,
            error=error,
            **kwargs
        )


class S3Logger:
    """Logger for S3 operations."""
    
    def __init__(self):
        self.logger = StructuredLogger("s3")
    
    def log_operation(self, operation: str, bucket: str, key: str, 
                     duration: float, success: bool, **kwargs):
        """Log S3 operation."""
        self.logger.info(
            "S3 operation",
            operation=operation,
            bucket=bucket,
            key=key,
            duration_seconds=duration,
            success=success,
            **kwargs
        )
    
    def log_error(self, operation: str, bucket: str, key: str, error: str, **kwargs):
        """Log S3 operation error."""
        self.logger.error(
            "S3 operation failed",
            operation=operation,
            bucket=bucket,
            key=key,
            error=error,
            **kwargs
        )


# Global logger instances
request_logger = RequestLogger()
task_logger = TaskLogger()
extractor_logger = ExtractorLogger()
s3_logger = S3Logger()


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)


def setup_logging():
    """Setup application-wide logging configuration."""
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format='%(message)s',
        stream=sys.stdout
    )
    
    # Set log levels for third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("celery").setLevel(logging.INFO)
    logging.getLogger("redis").setLevel(logging.WARNING)
