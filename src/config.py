"""
Configuration settings for AudioProcessor.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AnyUrl, field_validator
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings."""

    # === API ===
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_workers: int = Field(1, env="API_WORKERS")
    debug: bool = Field(False, env="DEBUG")
    reload: bool = Field(False, env="RELOAD")

    # === Celery ===
    celery_broker_url: AnyUrl = Field("redis://localhost:6380/0", env="CELERY_BROKER_URL")
    celery_result_backend: AnyUrl = Field("redis://localhost:6380/0", env="CELERY_RESULT_BACKEND")
    celery_task_serializer: str = Field("json", env="CELERY_TASK_SERIALIZER")
    celery_result_serializer: str = Field("json", env="CELERY_RESULT_SERIALIZER")
    celery_accept_content: List[str] = Field(default_factory=lambda: ["json"], env="CELERY_ACCEPT_CONTENT")
    celery_timezone: str = Field("UTC", env="CELERY_TIMEZONE")
    celery_enable_utc: bool = Field(True, env="CELERY_ENABLE_UTC")

    # === S3 ===
    s3_endpoint: AnyUrl = Field("https://s3.amazonaws.com", env="S3_ENDPOINT")
    s3_access_key: str = Field("", env="S3_ACCESS_KEY")
    s3_secret_key: str = Field("", env="S3_SECRET_KEY")
    s3_bucket: str = Field("audio-features", env="S3_BUCKET")
    s3_region: str = Field("us-east-1", env="S3_REGION")

    # === MasterML ===
    masterml_url: AnyUrl = Field("http://localhost:8000", env="MASTERML_URL")
    masterml_token: Optional[str] = Field(None, env="MASTERML_TOKEN")

    # === Logging ===
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    log_file: Optional[str] = Field(None, env="LOG_FILE")

    # === Monitoring ===
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")

    # === Audio Processing ===
    audio_sample_rate: int = Field(22050, env="AUDIO_SAMPLE_RATE")
    audio_chunk_size: int = Field(1024, env="AUDIO_CHUNK_SIZE")
    audio_max_duration: int = Field(3600, env="AUDIO_MAX_DURATION")

    # === GPU ===
    cuda_visible_devices: str = Field("0", env="CUDA_VISIBLE_DEVICES")
    use_gpu: bool = Field(True, env="USE_GPU")
    gpu_memory_limit: float = Field(0.8, env="GPU_MEMORY_LIMIT")
    gpu_batch_size: int = Field(8, env="GPU_BATCH_SIZE")
    
    # === Parallel Processing ===
    max_cpu_workers: int = Field(8, env="MAX_CPU_WORKERS")
    max_gpu_workers: int = Field(2, env="MAX_GPU_WORKERS")
    max_io_workers: int = Field(16, env="MAX_IO_WORKERS")
    max_concurrent_videos: int = Field(4, env="MAX_CONCURRENT_VIDEOS")
    max_segment_workers: int = Field(8, env="MAX_SEGMENT_WORKERS")
    segment_batch_size: int = Field(16, env="SEGMENT_BATCH_SIZE")
    max_s3_downloads: int = Field(16, env="MAX_S3_DOWNLOADS")
    max_file_operations: int = Field(32, env="MAX_FILE_OPERATIONS")
    extractor_timeout: int = Field(300, env="EXTRACTOR_TIMEOUT")

    # === Security ===
    api_key: Optional[str] = Field(None, env="API_KEY")
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"], env="CORS_ORIGINS")

    # === Performance ===
    max_workers: int = Field(4, env="MAX_WORKERS")
    worker_timeout: int = Field(3600, env="WORKER_TIMEOUT")
    task_time_limit: int = Field(3600, env="TASK_TIME_LIMIT")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # === Validators ===
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            # поддержка форматов:
            # "http://a.com,http://b.com" или '["http://a.com", "http://b.com"]'
            if v.startswith("["):
                import json
                return json.loads(v)
            return [origin.strip() for origin in v.split(",")]
        return v


# Singleton
settings = Settings()


def get_settings() -> Settings:
    """Return application settings singleton."""
    return settings
