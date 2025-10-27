"""
Smart GPU detection and resource management for AudioProcessor.
Automatically detects GPU availability and configures extractors accordingly.
Enhanced with GPU-optimized configurations and performance tuning.
"""

import os
import torch
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from .utils.logging import get_logger
from .gpu_config import get_gpu_config_manager, GPUType, GPUConfig

logger = get_logger(__name__)

@dataclass
class GPUInfo:
    """Information about GPU availability and configuration."""
    available: bool
    device_count: int
    device_name: str
    memory_total: Optional[int] = None  # in MB
    memory_free: Optional[int] = None   # in MB
    cuda_version: Optional[str] = None

@dataclass
class SmartConfig:
    """Smart configuration based on available resources with GPU optimization."""
    gpu_available: bool
    max_cpu_workers: int
    max_gpu_workers: int
    max_io_workers: int
    gpu_batch_size: int
    cpu_extractors: List[str]
    gpu_extractors: List[str]
    hybrid_extractors: List[str]  # Can work on both CPU and GPU
    device: str
    gpu_semaphore_enabled: bool
    
    # GPU optimization settings
    gpu_type: GPUType
    gpu_config: GPUConfig
    extractor_configs: Dict[str, Dict[str, Any]]
    performance_estimates: Dict[str, Any]

class SmartGPUDetector:
    """Smart GPU detection and resource configuration."""
    
    def __init__(self):
        self.gpu_info = self._detect_gpu()
        self.smart_config = self._create_smart_config()
        self._log_detection_results()
    
    def _detect_gpu(self) -> GPUInfo:
        """Detect GPU availability and capabilities."""
        logger.info("🔍 Detecting GPU availability...")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        
        if not cuda_available:
            logger.info("❌ CUDA not available - using CPU-only mode")
            return GPUInfo(
                available=False,
                device_count=0,
                device_name="CPU",
                cuda_version=None
            )
        
        # Get GPU information
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        
        # Get memory information
        memory_total = None
        memory_free = None
        try:
            memory_total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)  # MB
            memory_free = (torch.cuda.get_device_properties(0).total_memory - 
                          torch.cuda.memory_allocated(0)) // (1024 * 1024)  # MB
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")
        
        logger.info(f"✅ GPU detected: {device_name}")
        logger.info(f"   Device count: {device_count}")
        logger.info(f"   CUDA version: {cuda_version}")
        if memory_total:
            logger.info(f"   Total memory: {memory_total} MB")
        if memory_free:
            logger.info(f"   Free memory: {memory_free} MB")
        
        return GPUInfo(
            available=True,
            device_count=device_count,
            device_name=device_name,
            memory_total=memory_total,
            memory_free=memory_free,
            cuda_version=cuda_version
        )
    
    def _create_smart_config(self) -> SmartConfig:
        """Create smart configuration based on detected resources with GPU optimization."""
        logger.info("⚙️ Creating smart configuration with GPU optimization...")
        
        # Get GPU configuration manager
        gpu_config_manager = get_gpu_config_manager()
        gpu_type = gpu_config_manager.gpu_type
        gpu_config = gpu_config_manager.get_optimized_config(gpu_type)
        extractor_configs = gpu_config_manager.get_extractor_configs(gpu_type)
        performance_estimates = gpu_config_manager.get_performance_estimates(gpu_type)
        
        # Define extractor categories
        cpu_only_extractors = [
            "mfcc_extractor", "mel_extractor", "chroma_extractor", 
            "loudness_extractor", "vad_extractor", "pitch_extractor",
            "spectral_extractor", "tempo_extractor", "quality_extractor",
            "onset_extractor", "voice_quality_extractor", "phoneme_analysis_extractor",
            "advanced_spectral_extractor", "music_analysis_extractor", 
            "rhythmic_analysis_extractor"
        ]
        
        gpu_optimized_extractors = [
            "optimized_clap_extractor", "optimized_asr_extractor", "speaker_diarization_extractor",
            "optimized_emotion_recognition_extractor", "source_separation_extractor",
            "sound_event_detection_extractor", "optimized_advanced_embeddings"
        ]
        
        hybrid_extractors = [
            "optimized_clap_extractor", "optimized_emotion_recognition_extractor"  # Can fallback to CPU
        ]
        
        if self.gpu_info.available:
            # GPU available - use GPU-optimized configuration
            logger.info(f"🚀 GPU mode: Using {gpu_type.value} GPU-optimized configuration")
            logger.info(f"   Performance: {performance_estimates['estimated_speedup']} speedup expected")
            
            return SmartConfig(
                gpu_available=True,
                max_cpu_workers=gpu_config.max_cpu_workers,
                max_gpu_workers=gpu_config.max_gpu_workers,
                max_io_workers=gpu_config.max_io_workers,
                gpu_batch_size=gpu_config.batch_size,
                cpu_extractors=cpu_only_extractors,
                gpu_extractors=gpu_optimized_extractors,
                hybrid_extractors=hybrid_extractors,
                device="cuda:0",
                gpu_semaphore_enabled=True,
                gpu_type=gpu_type,
                gpu_config=gpu_config,
                extractor_configs=extractor_configs,
                performance_estimates=performance_estimates
            )
        else:
            # No GPU - use CPU-only configuration
            logger.info("💻 CPU mode: Using CPU-only configuration")
            
            return SmartConfig(
                gpu_available=False,
                max_cpu_workers=gpu_config.max_cpu_workers,
                max_gpu_workers=0,
                max_io_workers=gpu_config.max_io_workers,
                gpu_batch_size=1,  # Not used in CPU mode
                cpu_extractors=cpu_only_extractors + gpu_optimized_extractors,
                gpu_extractors=[],  # No GPU extractors in CPU mode
                hybrid_extractors=hybrid_extractors,
                device="cpu",
                gpu_semaphore_enabled=False,
                gpu_type=gpu_type,
                gpu_config=gpu_config,
                extractor_configs=extractor_configs,
                performance_estimates=performance_estimates
            )
    
    def _log_detection_results(self):
        """Log the detection and configuration results."""
        logger.info("📊 Smart GPU Detection Results:")
        logger.info(f"   GPU Available: {self.smart_config.gpu_available}")
        logger.info(f"   GPU Type: {self.smart_config.gpu_type.value}")
        logger.info(f"   Device: {self.smart_config.device}")
        logger.info(f"   CPU Workers: {self.smart_config.max_cpu_workers}")
        logger.info(f"   GPU Workers: {self.smart_config.max_gpu_workers}")
        logger.info(f"   IO Workers: {self.smart_config.max_io_workers}")
        logger.info(f"   GPU Batch Size: {self.smart_config.gpu_batch_size}")
        logger.info(f"   GPU Semaphore: {self.smart_config.gpu_semaphore_enabled}")
        logger.info(f"   CPU Extractors: {len(self.smart_config.cpu_extractors)}")
        logger.info(f"   GPU Extractors: {len(self.smart_config.gpu_extractors)}")
        logger.info(f"   Hybrid Extractors: {len(self.smart_config.hybrid_extractors)}")
        
        # Log GPU optimization settings
        if self.smart_config.gpu_available:
            logger.info("🚀 GPU Optimization Settings:")
            logger.info(f"   Mixed Precision: {self.smart_config.gpu_config.mixed_precision}")
            logger.info(f"   Tensor Core Optimization: {self.smart_config.gpu_config.tensor_core_optimization}")
            logger.info(f"   Model Caching: {self.smart_config.gpu_config.model_caching}")
            logger.info(f"   Memory Limit: {self.smart_config.gpu_config.gpu_memory_limit:.1%}")
            logger.info(f"   Model Precision: {self.smart_config.gpu_config.model_precision}")
            
            # Log performance estimates
            logger.info("📈 Performance Estimates:")
            for metric, value in self.smart_config.performance_estimates.items():
                logger.info(f"   {metric.replace('_', ' ').title()}: {value}")
    
    def get_smart_config(self) -> SmartConfig:
        """Get the smart configuration."""
        return self.smart_config
    
    def get_gpu_info(self) -> GPUInfo:
        """Get GPU information."""
        return self.gpu_info
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return self.gpu_info.available
    
    def get_optimal_device(self) -> str:
        """Get the optimal device for processing."""
        return self.smart_config.device
    
    def should_use_gpu_semaphore(self) -> bool:
        """Check if GPU semaphore should be used."""
        return self.smart_config.gpu_semaphore_enabled
    
    def get_extractor_categories(self) -> Dict[str, List[str]]:
        """Get extractor categories based on available resources."""
        return {
            "cpu_extractors": self.smart_config.cpu_extractors,
            "gpu_extractors": self.smart_config.gpu_extractors,
            "hybrid_extractors": self.smart_config.hybrid_extractors
        }
    
    def get_worker_config(self) -> Dict[str, int]:
        """Get worker configuration."""
        return {
            "max_cpu_workers": self.smart_config.max_cpu_workers,
            "max_gpu_workers": self.smart_config.max_gpu_workers,
            "max_io_workers": self.smart_config.max_io_workers,
            "gpu_batch_size": self.smart_config.gpu_batch_size
        }

# Global instance for easy access
_smart_detector: Optional[SmartGPUDetector] = None

def get_smart_detector() -> SmartGPUDetector:
    """Get the global smart GPU detector instance."""
    global _smart_detector
    if _smart_detector is None:
        _smart_detector = SmartGPUDetector()
    return _smart_detector

def get_smart_config() -> SmartConfig:
    """Get the smart configuration."""
    return get_smart_detector().get_smart_config()

def is_gpu_available() -> bool:
    """Check if GPU is available."""
    return get_smart_detector().is_gpu_available()

def get_optimal_device() -> str:
    """Get the optimal device for processing."""
    return get_smart_detector().get_optimal_device()
