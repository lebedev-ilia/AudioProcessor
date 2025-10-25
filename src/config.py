"""
Configuration settings for AudioProcessor.
"""
from pydantic import BaseSettings, Field
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    debug: bool = Field(default=False, env="DEBUG")
    reload: bool = Field(default=False, env="RELOAD")
    
    # Celery Configuration
    celery_broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    celery_task_serializer: str = Field(default="json", env="CELERY_TASK_SERIALIZER")
    celery_result_serializer: str = Field(default="json", env="CELERY_RESULT_SERIALIZER")
    celery_accept_content: List[str] = Field(default=["json"], env="CELERY_ACCEPT_CONTENT")
    celery_timezone: str = Field(default="UTC", env="CELERY_TIMEZONE")
    celery_enable_utc: bool = Field(default=True, env="CELERY_ENABLE_UTC")
    
    # S3 Configuration
    s3_endpoint: str = Field(default="https://s3.amazonaws.com", env="S3_ENDPOINT")
    s3_access_key: str = Field(default="", env="S3_ACCESS_KEY")
    s3_secret_key: str = Field(default="", env="S3_SECRET_KEY")
    s3_bucket: str = Field(default="audio-features", env="S3_BUCKET")
    s3_region: str = Field(default="us-east-1", env="S3_REGION")
    
    # MasterML Configuration
    masterml_url: str = Field(default="http://localhost:8000", env="MASTERML_URL")
    masterml_token: str = Field(default="", env="MASTERML_TOKEN")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Monitoring
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    
    # Audio Processing
    audio_sample_rate: int = Field(default=22050, env="AUDIO_SAMPLE_RATE")
    audio_chunk_size: int = Field(default=1024, env="AUDIO_CHUNK_SIZE")
    audio_max_duration: int = Field(default=3600, env="AUDIO_MAX_DURATION")
    
    # Model Configuration
    openl3_model: str = Field(default="openl3_audio_embedding", env="OPENL3_MODEL")
    openl3_content_type: str = Field(default="music", env="OPENL3_CONTENT_TYPE")
    openl3_input_repr: str = Field(default="mel256", env="OPENL3_INPUT_REPR")
    openl3_embedding_size: int = Field(default=512, env="OPENL3_EMBEDDING_SIZE")
    
    # GPU Configuration
    cuda_visible_devices: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")
    use_gpu: bool = Field(default=False, env="USE_GPU")
    
    # Security
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    cors_origins: List[str] = Field(default=["http://localhost:3000"], env="CORS_ORIGINS")
    
    # Performance
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    worker_timeout: int = Field(default=3600, env="WORKER_TIMEOUT")
    task_time_limit: int = Field(default=3600, env="TASK_TIME_LIMIT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
