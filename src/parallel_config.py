"""
Configuration for parallel processing in AudioProcessor.

This module provides configuration classes and settings for:
1. Resource allocation (CPU, GPU, I/O workers)
2. Concurrency limits and semaphores
3. Performance tuning parameters
4. Machine-specific configurations
"""

import os
from typing import Dict, Any, Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
from .utils.logging import get_logger

logger = get_logger(__name__)


class ParallelConfig(BaseSettings):
    """Configuration for parallel processing."""
    
    # === Resource Configuration ===
    max_cpu_workers: int = Field(8, env="MAX_CPU_WORKERS", description="Maximum CPU workers for extractors")
    max_gpu_workers: int = Field(2, env="MAX_GPU_WORKERS", description="Maximum GPU workers for extractors")
    max_io_workers: int = Field(16, env="MAX_IO_WORKERS", description="Maximum I/O workers for file operations")
    max_concurrent_videos: int = Field(4, env="MAX_CONCURRENT_VIDEOS", description="Maximum concurrent video processing")
    
    # === GPU Configuration ===
    gpu_batch_size: int = Field(8, env="GPU_BATCH_SIZE", description="Batch size for GPU processing")
    gpu_memory_limit: float = Field(0.8, env="GPU_MEMORY_LIMIT", description="GPU memory usage limit (0.0-1.0)")
    
    # === Segment Processing ===
    max_segment_workers: int = Field(8, env="MAX_SEGMENT_WORKERS", description="Maximum workers for segment processing")
    segment_batch_size: int = Field(16, env="SEGMENT_BATCH_SIZE", description="Batch size for segment processing")
    
    # === I/O Configuration ===
    max_s3_downloads: int = Field(16, env="MAX_S3_DOWNLOADS", description="Maximum concurrent S3 downloads")
    max_file_operations: int = Field(32, env="MAX_FILE_OPERATIONS", description="Maximum concurrent file operations")
    
    # === Performance Tuning ===
    extractor_timeout: int = Field(300, env="EXTRACTOR_TIMEOUT", description="Timeout for extractor execution (seconds)")
    segment_timeout: int = Field(60, env="SEGMENT_TIMEOUT", description="Timeout for segment processing (seconds)")
    batch_timeout: int = Field(1800, env="BATCH_TIMEOUT", description="Timeout for batch processing (seconds)")
    
    # === Backpressure Configuration ===
    enable_backpressure: bool = Field(True, env="ENABLE_BACKPRESSURE", description="Enable backpressure control")
    gpu_queue_threshold: int = Field(100, env="GPU_QUEUE_THRESHOLD", description="GPU queue length threshold for backpressure")
    cpu_queue_threshold: int = Field(200, env="CPU_QUEUE_THRESHOLD", description="CPU queue length threshold for backpressure")
    
    # === Monitoring ===
    enable_metrics: bool = Field(True, env="ENABLE_PARALLEL_METRICS", description="Enable parallel processing metrics")
    metrics_interval: int = Field(30, env="METRICS_INTERVAL", description="Metrics collection interval (seconds)")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class MachineConfig:
    """Machine-specific configurations for different deployment scenarios."""
    
    @staticmethod
    def get_small_dev_config() -> ParallelConfig:
        """Configuration for small development machine (4 cores, no GPU)."""
        return ParallelConfig(
            max_cpu_workers=2,
            max_gpu_workers=0,
            max_io_workers=4,
            max_concurrent_videos=1,
            gpu_batch_size=1,
            max_segment_workers=2,
            segment_batch_size=8,
            max_s3_downloads=4,
            max_file_operations=8
        )
    
    @staticmethod
    def get_prod_cpu_config() -> ParallelConfig:
        """Configuration for production CPU-only machine (32 cores)."""
        return ParallelConfig(
            max_cpu_workers=12,
            max_gpu_workers=0,
            max_io_workers=24,
            max_concurrent_videos=6,
            gpu_batch_size=1,
            max_segment_workers=12,
            segment_batch_size=16,
            max_s3_downloads=16,
            max_file_operations=32
        )
    
    @staticmethod
    def get_prod_gpu_config() -> ParallelConfig:
        """Configuration for production GPU machine (32 cores + 2 GPUs)."""
        return ParallelConfig(
            max_cpu_workers=8,
            max_gpu_workers=2,
            max_io_workers=16,
            max_concurrent_videos=4,
            gpu_batch_size=8,
            max_segment_workers=8,
            segment_batch_size=16,
            max_s3_downloads=16,
            max_file_operations=32
        )
    
    @staticmethod
    def get_auto_config() -> ParallelConfig:
        """Auto-detect configuration based on system resources."""
        import multiprocessing
        import psutil
        
        cpu_count = multiprocessing.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Detect GPU availability
        gpu_count = 0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
        except ImportError:
            pass
        
        # Auto-configure based on resources
        if gpu_count > 0:
            # GPU machine
            max_cpu_workers = min(8, cpu_count // 2)
            max_gpu_workers = min(2, gpu_count)
            max_concurrent_videos = min(4, cpu_count // 4)
        else:
            # CPU-only machine
            max_cpu_workers = min(12, cpu_count // 2)
            max_gpu_workers = 0
            max_concurrent_videos = min(6, cpu_count // 4)
        
        max_io_workers = min(32, cpu_count)
        max_segment_workers = min(16, cpu_count)
        
        return ParallelConfig(
            max_cpu_workers=max_cpu_workers,
            max_gpu_workers=max_gpu_workers,
            max_io_workers=max_io_workers,
            max_concurrent_videos=max_concurrent_videos,
            gpu_batch_size=8 if gpu_count > 0 else 1,
            max_segment_workers=max_segment_workers,
            segment_batch_size=16,
            max_s3_downloads=16,
            max_file_operations=32
        )


def get_parallel_config(machine_type: str = "auto") -> ParallelConfig:
    """
    Get parallel processing configuration.
    
    Args:
        machine_type: Type of machine configuration ("auto", "small_dev", "prod_cpu", "prod_gpu")
        
    Returns:
        ParallelConfig instance
    """
    if machine_type == "auto":
        return MachineConfig.get_auto_config()
    elif machine_type == "small_dev":
        return MachineConfig.get_small_dev_config()
    elif machine_type == "prod_cpu":
        return MachineConfig.get_prod_cpu_config()
    elif machine_type == "prod_gpu":
        return MachineConfig.get_prod_gpu_config()
    else:
        logger.warning(f"Unknown machine type: {machine_type}, using auto configuration")
        return MachineConfig.get_auto_config()


def get_extractor_categories() -> Dict[str, List[str]]:
    """
    Get extractor categories for parallel processing.
    
    Returns:
        Dictionary with CPU and GPU extractor lists
    """
    return {
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
    }


def get_performance_estimates() -> Dict[str, Dict[str, Any]]:
    """
    Get performance estimates for parallel processing.
    
    Returns:
        Dictionary with performance estimates
    """
    return {
        "extractors": {
            "cpu_parallel": {
                "speedup": "3-5x",
                "description": "CPU extractors running in parallel"
            },
            "gpu_parallel": {
                "speedup": "2-6x", 
                "description": "GPU extractors with batching"
            }
        },
        "segments": {
            "parallel_processing": {
                "speedup": "4-8x",
                "description": "Segments processed in parallel"
            }
        },
        "batch": {
            "concurrent_videos": {
                "speedup": "2-4x",
                "description": "Multiple videos processed concurrently"
            }
        },
        "overall": {
            "total_speedup": "5-15x",
            "description": "Combined parallel processing benefits"
        }
    }


# Global configuration instance
_parallel_config: Optional[ParallelConfig] = None


def get_global_parallel_config() -> ParallelConfig:
    """Get global parallel configuration instance."""
    global _parallel_config
    if _parallel_config is None:
        machine_type = os.getenv("MACHINE_TYPE", "auto")
        _parallel_config = get_parallel_config(machine_type)
        logger.info(f"Initialized parallel config for machine type: {machine_type}")
        logger.info(f"CPU workers: {_parallel_config.max_cpu_workers}, "
                   f"GPU workers: {_parallel_config.max_gpu_workers}, "
                   f"IO workers: {_parallel_config.max_io_workers}")
    return _parallel_config


def update_global_parallel_config(config: ParallelConfig):
    """Update global parallel configuration."""
    global _parallel_config
    _parallel_config = config
    logger.info("Updated global parallel configuration")


# Example usage
if __name__ == "__main__":
    # Test different configurations
    configs = {
        "auto": get_parallel_config("auto"),
        "small_dev": get_parallel_config("small_dev"),
        "prod_cpu": get_parallel_config("prod_cpu"),
        "prod_gpu": get_parallel_config("prod_gpu")
    }
    
    for name, config in configs.items():
        print(f"\n{name.upper()} Configuration:")
        print(f"  CPU Workers: {config.max_cpu_workers}")
        print(f"  GPU Workers: {config.max_gpu_workers}")
        print(f"  IO Workers: {config.max_io_workers}")
        print(f"  Concurrent Videos: {config.max_concurrent_videos}")
        print(f"  GPU Batch Size: {config.gpu_batch_size}")
    
    # Test extractor categories
    categories = get_extractor_categories()
    print(f"\nExtractor Categories:")
    print(f"  CPU Extractors: {len(categories['cpu_extractors'])}")
    print(f"  GPU Extractors: {len(categories['gpu_extractors'])}")
    
    # Test performance estimates
    estimates = get_performance_estimates()
    print(f"\nPerformance Estimates:")
    print(f"  Overall Speedup: {estimates['overall']['total_speedup']}")
    print(f"  CPU Parallel: {estimates['extractors']['cpu_parallel']['speedup']}")
    print(f"  GPU Parallel: {estimates['extractors']['gpu_parallel']['speedup']}")
    print(f"  Segment Parallel: {estimates['segments']['parallel_processing']['speedup']}")
