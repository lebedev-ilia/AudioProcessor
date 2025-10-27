"""
GPU-optimized configurations for maximum performance.

This module provides:
1. GPU-specific configuration presets
2. Automatic GPU resource detection and optimization
3. Memory-aware configuration adjustment
4. Performance-optimized settings for different GPU types
5. Dynamic configuration based on available resources
"""

import os
import torch
import psutil
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from .utils.logging import get_logger

logger = get_logger(__name__)


class GPUType(Enum):
    """GPU type classification for optimization."""
    HIGH_END = "high_end"      # RTX 4090, A100, H100
    MID_RANGE = "mid_range"    # RTX 3080, RTX 4070, V100
    ENTRY_LEVEL = "entry_level" # RTX 3060, GTX 1660, T4
    LOW_MEMORY = "low_memory"  # < 4GB VRAM
    CPU_ONLY = "cpu_only"      # No GPU available


@dataclass
class GPUConfig:
    """GPU-optimized configuration."""
    
    # === GPU Settings ===
    gpu_enabled: bool = True
    gpu_count: int = 1
    gpu_memory_limit: float = 0.9  # Use 90% of available memory
    mixed_precision: bool = True
    tensor_core_optimization: bool = True
    
    # === Batch Processing ===
    batch_size: int = 8
    max_batch_size: int = 16
    dynamic_batching: bool = True
    batch_timeout_ms: int = 100
    
    # === Memory Management ===
    memory_efficient_attention: bool = True
    gradient_checkpointing: bool = True
    model_caching: bool = True
    cache_size_limit: float = 0.3  # 30% of GPU memory for caching
    
    # === Worker Configuration ===
    max_gpu_workers: int = 2
    max_cpu_workers: int = 8
    max_io_workers: int = 16
    
    # === Model Settings ===
    model_precision: str = "fp16"  # fp16, fp32, bf16
    enable_optimization: bool = True
    enable_compilation: bool = True
    
    # === Performance Tuning ===
    prefetch_factor: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # === Monitoring ===
    enable_profiling: bool = False
    memory_monitoring: bool = True
    performance_tracking: bool = True


class GPUConfigManager:
    """Manager for GPU-optimized configurations."""
    
    def __init__(self):
        self.logger = logger
        self.gpu_info = self._detect_gpu_info()
        self.gpu_type = self._classify_gpu_type()
        
    def _detect_gpu_info(self) -> Dict[str, Any]:
        """Detect GPU information and capabilities."""
        gpu_info = {
            "available": False,
            "count": 0,
            "devices": [],
            "total_memory": 0,
            "free_memory": 0,
            "compute_capability": None,
            "cuda_version": None
        }
        
        if not torch.cuda.is_available():
            self.logger.info("CUDA not available - using CPU-only configuration")
            return gpu_info
        
        try:
            gpu_info["available"] = True
            gpu_info["count"] = torch.cuda.device_count()
            gpu_info["cuda_version"] = torch.version.cuda
            
            # Get information for each GPU
            for i in range(gpu_info["count"]):
                device_info = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_free": torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i),
                    "compute_capability": torch.cuda.get_device_properties(i).major,
                    "multiprocessor_count": torch.cuda.get_device_properties(i).multi_processor_count
                }
                gpu_info["devices"].append(device_info)
                gpu_info["total_memory"] += device_info["memory_total"]
                gpu_info["free_memory"] += device_info["memory_free"]
            
            # Get compute capability from first device
            if gpu_info["devices"]:
                gpu_info["compute_capability"] = gpu_info["devices"][0]["compute_capability"]
            
            self.logger.info(f"Detected {gpu_info['count']} GPU(s) with {gpu_info['total_memory'] // 1024**3}GB total memory")
            
        except Exception as e:
            self.logger.error(f"Failed to detect GPU info: {e}")
            gpu_info["available"] = False
        
        return gpu_info
    
    def _classify_gpu_type(self) -> GPUType:
        """Classify GPU type based on capabilities with universal detection."""
        if not self.gpu_info["available"]:
            return GPUType.CPU_ONLY
        
        total_memory_gb = self.gpu_info["total_memory"] / (1024**3)
        compute_capability = self.gpu_info["compute_capability"]
        device_name = self.gpu_info.get("device_name", "").lower()
        
        # Get major compute capability
        major_capability = compute_capability[0] if isinstance(compute_capability, (tuple, list)) else 0
        
        # High-end GPUs (16GB+ VRAM or high-end models)
        if (total_memory_gb >= 16 or 
            any(name in device_name for name in ['rtx 4090', 'rtx 4080', 'a100', 'h100', 'v100', 'a6000', 'a40']) or
            major_capability >= 8):
            return GPUType.HIGH_END
        
        # Mid-range GPUs (8-16GB VRAM or mid-range models)
        elif (total_memory_gb >= 8 or 
              any(name in device_name for name in ['rtx 3080', 'rtx 4070', 'rtx 3090', 'rtx 3070', 'rtx 4060 ti', 'a5000', 'a4000']) or
              major_capability >= 7):
            return GPUType.MID_RANGE
        
        # Entry-level GPUs (4-8GB VRAM or entry-level models)
        elif (total_memory_gb >= 4 or 
              any(name in device_name for name in ['rtx 2060', 'rtx 3060', 'rtx 3050', 'rtx 4060', 'gtx 1660', 't4', 'a2000', 'a10']) or
              major_capability >= 6):
            return GPUType.ENTRY_LEVEL
        
        # Low memory GPUs (2-4GB VRAM)
        elif total_memory_gb >= 2:
            return GPUType.LOW_MEMORY
        
        # No GPU or very low memory
        else:
            return GPUType.CPU_ONLY
    
    def get_optimized_config(self, gpu_type: Optional[GPUType] = None) -> GPUConfig:
        """Get optimized configuration for GPU type."""
        if gpu_type is None:
            gpu_type = self.gpu_type
        
        self.logger.info(f"Generating optimized configuration for {gpu_type.value} GPU")
        
        if gpu_type == GPUType.HIGH_END:
            return self._get_high_end_config()
        elif gpu_type == GPUType.MID_RANGE:
            return self._get_mid_range_config()
        elif gpu_type == GPUType.ENTRY_LEVEL:
            return self._get_entry_level_config()
        elif gpu_type == GPUType.LOW_MEMORY:
            return self._get_low_memory_config()
        else:  # CPU_ONLY
            return self._get_cpu_only_config()
    
    def _get_high_end_config(self) -> GPUConfig:
        """Configuration for high-end GPUs (RTX 4090, A100, H100)."""
        return GPUConfig(
            gpu_enabled=True,
            gpu_count=min(self.gpu_info["count"], 2),
            gpu_memory_limit=0.95,
            mixed_precision=True,
            tensor_core_optimization=True,
            batch_size=16,
            max_batch_size=32,
            dynamic_batching=True,
            batch_timeout_ms=50,
            memory_efficient_attention=True,
            gradient_checkpointing=False,
            model_caching=True,
            cache_size_limit=0.4,
            max_gpu_workers=4,
            max_cpu_workers=12,
            max_io_workers=24,
            model_precision="bf16",
            enable_optimization=True,
            enable_compilation=True,
            prefetch_factor=4,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            enable_profiling=True,
            memory_monitoring=True,
            performance_tracking=True
        )
    
    def _get_mid_range_config(self) -> GPUConfig:
        """Configuration for mid-range GPUs (RTX 3080, RTX 4070, V100)."""
        return GPUConfig(
            gpu_enabled=True,
            gpu_count=min(self.gpu_info["count"], 2),
            gpu_memory_limit=0.9,
            mixed_precision=True,
            tensor_core_optimization=True,
            batch_size=8,
            max_batch_size=16,
            dynamic_batching=True,
            batch_timeout_ms=100,
            memory_efficient_attention=True,
            gradient_checkpointing=True,
            model_caching=True,
            cache_size_limit=0.3,
            max_gpu_workers=2,
            max_cpu_workers=8,
            max_io_workers=16,
            model_precision="fp16",
            enable_optimization=True,
            enable_compilation=True,
            prefetch_factor=2,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            enable_profiling=False,
            memory_monitoring=True,
            performance_tracking=True
        )
    
    def _get_entry_level_config(self) -> GPUConfig:
        """Configuration for entry-level GPUs (RTX 2060, RTX 3060, GTX 1660, T4, etc.)."""
        # Get actual GPU memory for dynamic configuration
        memory_gb = self.gpu_info.get("total_memory", 0) / (1024**3)
        
        # Dynamic memory limit based on actual GPU memory
        if memory_gb >= 8:
            memory_limit = 0.85
            batch_size = 6
            max_batch_size = 12
        elif memory_gb >= 6:
            memory_limit = 0.80
            batch_size = 4
            max_batch_size = 8
        else:  # 4GB or less
            memory_limit = 0.75
            batch_size = 2
            max_batch_size = 4
        
        return GPUConfig(
            gpu_enabled=True,
            gpu_count=1,
            gpu_memory_limit=memory_limit,
            mixed_precision=True,
            tensor_core_optimization=True,  # Most modern GPUs support Tensor Cores
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            dynamic_batching=True,
            batch_timeout_ms=250,  # Longer timeout for smaller batches
            memory_efficient_attention=True,
            gradient_checkpointing=True,
            model_caching=True,
            cache_size_limit=0.2,
            max_gpu_workers=1,
            max_cpu_workers=6,
            max_io_workers=12,
            model_precision="fp16",
            enable_optimization=True,
            enable_compilation=False,
            prefetch_factor=1,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False,
            enable_profiling=False,
            memory_monitoring=True,
            performance_tracking=False
        )
    
    def _get_low_memory_config(self) -> GPUConfig:
        """Configuration for low memory GPUs (< 4GB VRAM)."""
        return GPUConfig(
            gpu_enabled=True,
            gpu_count=1,
            gpu_memory_limit=0.7,
            mixed_precision=True,
            tensor_core_optimization=False,
            batch_size=2,
            max_batch_size=4,
            dynamic_batching=True,
            batch_timeout_ms=500,
            memory_efficient_attention=True,
            gradient_checkpointing=True,
            model_caching=False,
            cache_size_limit=0.1,
            max_gpu_workers=1,
            max_cpu_workers=4,
            max_io_workers=8,
            model_precision="fp16",
            enable_optimization=True,
            enable_compilation=False,
            prefetch_factor=1,
            num_workers=1,
            pin_memory=False,
            persistent_workers=False,
            enable_profiling=False,
            memory_monitoring=True,
            performance_tracking=False
        )
    
    def _get_cpu_only_config(self) -> GPUConfig:
        """Configuration for CPU-only processing."""
        return GPUConfig(
            gpu_enabled=False,
            gpu_count=0,
            gpu_memory_limit=0.0,
            mixed_precision=False,
            tensor_core_optimization=False,
            batch_size=1,
            max_batch_size=1,
            dynamic_batching=False,
            batch_timeout_ms=1000,
            memory_efficient_attention=False,
            gradient_checkpointing=False,
            model_caching=False,
            cache_size_limit=0.0,
            max_gpu_workers=0,
            max_cpu_workers=psutil.cpu_count(),
            max_io_workers=psutil.cpu_count() * 2,
            model_precision="fp32",
            enable_optimization=False,
            enable_compilation=False,
            prefetch_factor=1,
            num_workers=1,
            pin_memory=False,
            persistent_workers=False,
            enable_profiling=False,
            memory_monitoring=False,
            performance_tracking=False
        )
    
    def get_extractor_configs(self, gpu_type: Optional[GPUType] = None) -> Dict[str, Dict[str, Any]]:
        """Get optimized configurations for specific extractors."""
        if gpu_type is None:
            gpu_type = self.gpu_type
        
        configs = {}
        
        if gpu_type == GPUType.HIGH_END:
            configs = {
                # GPU-optimized extractors with enhanced settings
                "spectral_extractor": {
                    "device": "cuda",
                    "batch_size": 16,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": True,
                    "n_fft": 2048,
                    "hop_length": 512,
                    "enable_profiling": True
                },
                "mfcc_extractor": {
                    "device": "cuda",
                    "batch_size": 16,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": True,
                    "n_mfcc": 13,
                    "n_fft": 2048,
                    "hop_length": 512,
                    "enable_profiling": True
                },
                "chroma_extractor": {
                    "device": "cuda",
                    "batch_size": 16,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": True,
                    "n_chroma": 12,
                    "n_fft": 2048,
                    "hop_length": 512,
                    "enable_profiling": True
                },
                "clap_extractor": {
                    "device": "cuda",
                    "batch_size": 16,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": True,
                    "max_audio_length": 15.0,
                    "enable_profiling": True
                },
                "asr_extractor": {
                    "device": "cuda",
                    "model_name": "large",
                    "batch_size": 8,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": True,
                    "enable_profiling": True
                },
                "advanced_embeddings_extractor": {
                    "device": "cuda",
                    "batch_size": 16,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": True,
                    "max_audio_length": 15.0,
                    "enable_profiling": True
                },
                "emotion_recognition_extractor": {
                    "device": "cuda",
                    "batch_size": 16,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": True,
                    "enable_profiling": True
                }
            }
        elif gpu_type == GPUType.MID_RANGE:
            configs = {
                # GPU-optimized extractors for mid-range GPUs
                "spectral_extractor": {
                    "device": "cuda",
                    "batch_size": 8,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": True,
                    "n_fft": 2048,
                    "hop_length": 512,
                    "enable_profiling": False
                },
                "mfcc_extractor": {
                    "device": "cuda",
                    "batch_size": 8,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": True,
                    "n_mfcc": 13,
                    "n_fft": 2048,
                    "hop_length": 512,
                    "enable_profiling": False
                },
                "chroma_extractor": {
                    "device": "cuda",
                    "batch_size": 8,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": True,
                    "n_chroma": 12,
                    "n_fft": 2048,
                    "hop_length": 512,
                    "enable_profiling": False
                },
                "clap_extractor": {
                    "device": "cuda",
                    "batch_size": 8,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": True,
                    "max_audio_length": 10.0,
                    "enable_profiling": False
                },
                "asr_extractor": {
                    "device": "cuda",
                    "model_name": "medium",
                    "batch_size": 4,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": True,
                    "enable_profiling": False
                },
                "advanced_embeddings_extractor": {
                    "device": "cuda",
                    "batch_size": 8,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": True,
                    "max_audio_length": 10.0,
                    "enable_profiling": False
                },
                "emotion_recognition_extractor": {
                    "device": "cuda",
                    "batch_size": 8,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": True,
                    "enable_profiling": False
                }
            }
        elif gpu_type == GPUType.ENTRY_LEVEL:
            configs = {
                # GPU-optimized extractors for entry-level GPUs
                "spectral_extractor": {
                    "device": "cuda",
                    "batch_size": 4,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": False,
                    "n_fft": 2048,
                    "hop_length": 512,
                    "enable_profiling": False
                },
                "mfcc_extractor": {
                    "device": "cuda",
                    "batch_size": 4,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": False,
                    "n_mfcc": 13,
                    "n_fft": 2048,
                    "hop_length": 512,
                    "enable_profiling": False
                },
                "chroma_extractor": {
                    "device": "cuda",
                    "batch_size": 4,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": False,
                    "n_chroma": 12,
                    "n_fft": 2048,
                    "hop_length": 512,
                    "enable_profiling": False
                },
                "clap_extractor": {
                    "device": "cuda",
                    "batch_size": 4,
                    "use_mixed_precision": True,
                    "enable_caching": True,
                    "enable_tensor_cores": False,
                    "max_audio_length": 8.0,
                    "enable_profiling": False
                },
                "asr_extractor": {
                    "device": "cuda",
                    "model_name": "small",
                    "batch_size": 2,
                    "use_mixed_precision": False,
                    "enable_caching": True,
                    "enable_tensor_cores": False,
                    "enable_profiling": False
                },
                "advanced_embeddings_extractor": {
                    "device": "cuda",
                    "batch_size": 4,
                    "use_mixed_precision": False,
                    "enable_caching": True,
                    "enable_tensor_cores": False,
                    "max_audio_length": 8.0,
                    "enable_profiling": False
                },
                "emotion_recognition_extractor": {
                    "device": "cuda",
                    "batch_size": 4,
                    "use_mixed_precision": False,
                    "enable_caching": True,
                    "enable_tensor_cores": False,
                    "enable_profiling": False
                }
            }
        elif gpu_type == GPUType.LOW_MEMORY:
            configs = {
                # GPU-optimized extractors
                "gpu_optimized_spectral": {
                    "device": "cuda",
                    "batch_size": 2,
                    "use_mixed_precision": True,
                    "enable_caching": False,
                    "n_fft": 1024,
                    "hop_length": 256
                },
                "gpu_optimized_mfcc": {
                    "device": "cuda",
                    "batch_size": 2,
                    "use_mixed_precision": True,
                    "enable_caching": False,
                    "n_mfcc": 13,
                    "n_fft": 1024,
                    "hop_length": 256
                },
                "gpu_optimized_chroma": {
                    "device": "cuda",
                    "batch_size": 2,
                    "use_mixed_precision": True,
                    "enable_caching": False,
                    "n_chroma": 12,
                    "n_fft": 1024,
                    "hop_length": 256
                },
                "optimized_clap_extractor": {
                    "device": "cuda",
                    "batch_size": 2,
                    "use_mixed_precision": True,
                    "enable_caching": False,
                    "max_audio_length": 5.0
                },
                "optimized_asr_extractor": {
                    "device": "cuda",
                    "model_name": "base",
                    "batch_size": 1,
                    "use_mixed_precision": False,
                    "enable_caching": False
                },
                "optimized_advanced_embeddings": {
                    "device": "cuda",
                    "batch_size": 2,
                    "use_mixed_precision": False,
                    "enable_caching": False,
                    "max_audio_length": 5.0
                },
                "emotion_recognition": {
                    "batch_size": 2,
                    "use_mixed_precision": False,
                    "enable_caching": False
                }
            }
        else:  # CPU_ONLY
            configs = {
                # Fallback to CPU versions
                "gpu_optimized_spectral": {
                    "device": "cpu",
                    "batch_size": 1,
                    "use_mixed_precision": False,
                    "enable_caching": False,
                    "n_fft": 2048,
                    "hop_length": 512
                },
                "gpu_optimized_mfcc": {
                    "device": "cpu",
                    "batch_size": 1,
                    "use_mixed_precision": False,
                    "enable_caching": False,
                    "n_mfcc": 13,
                    "n_fft": 2048,
                    "hop_length": 512
                },
                "gpu_optimized_chroma": {
                    "device": "cpu",
                    "batch_size": 1,
                    "use_mixed_precision": False,
                    "enable_caching": False,
                    "n_chroma": 12,
                    "n_fft": 2048,
                    "hop_length": 512
                },
                "optimized_clap_extractor": {
                    "device": "cpu",
                    "batch_size": 1,
                    "use_mixed_precision": False,
                    "enable_caching": False,
                    "max_audio_length": 10.0
                },
                "optimized_asr_extractor": {
                    "device": "cpu",
                    "model_name": "base",
                    "batch_size": 1,
                    "use_mixed_precision": False,
                    "enable_caching": False
                },
                "optimized_advanced_embeddings": {
                    "device": "cpu",
                    "batch_size": 1,
                    "use_mixed_precision": False,
                    "enable_caching": False,
                    "max_audio_length": 10.0
                },
                "emotion_recognition": {
                    "batch_size": 1,
                    "use_mixed_precision": False,
                    "enable_caching": False
                }
            }
        
        return configs
    
    def get_performance_estimates(self, gpu_type: Optional[GPUType] = None) -> Dict[str, Any]:
        """Get performance estimates for GPU type."""
        if gpu_type is None:
            gpu_type = self.gpu_type
        
        estimates = {
            GPUType.HIGH_END: {
                "throughput": "Very High",
                "latency": "Very Low",
                "memory_efficiency": "High",
                "power_efficiency": "Medium",
                "estimated_speedup": "10-20x"
            },
            GPUType.MID_RANGE: {
                "throughput": "High",
                "latency": "Low",
                "memory_efficiency": "High",
                "power_efficiency": "High",
                "estimated_speedup": "5-10x"
            },
            GPUType.ENTRY_LEVEL: {
                "throughput": "Medium",
                "latency": "Medium",
                "memory_efficiency": "Medium",
                "power_efficiency": "High",
                "estimated_speedup": "2-5x"
            },
            GPUType.LOW_MEMORY: {
                "throughput": "Low",
                "latency": "High",
                "memory_efficiency": "Low",
                "power_efficiency": "Very High",
                "estimated_speedup": "1-2x"
            },
            GPUType.CPU_ONLY: {
                "throughput": "Very Low",
                "latency": "Very High",
                "memory_efficiency": "Very Low",
                "power_efficiency": "Low",
                "estimated_speedup": "1x"
            }
        }
        
        return estimates.get(gpu_type, estimates[GPUType.CPU_ONLY])
    
    def optimize_for_workload(self, workload_type: str) -> GPUConfig:
        """Optimize configuration for specific workload type."""
        base_config = self.get_optimized_config()
        
        if workload_type == "batch_processing":
            # Optimize for high throughput
            base_config.batch_size = min(base_config.batch_size * 2, base_config.max_batch_size)
            base_config.dynamic_batching = True
            base_config.batch_timeout_ms = 50
            base_config.max_gpu_workers = min(base_config.max_gpu_workers + 1, 4)
            
        elif workload_type == "real_time":
            # Optimize for low latency
            base_config.batch_size = max(1, base_config.batch_size // 2)
            base_config.dynamic_batching = False
            base_config.batch_timeout_ms = 10
            base_config.enable_compilation = True
            
        elif workload_type == "memory_constrained":
            # Optimize for memory efficiency
            base_config.batch_size = max(1, base_config.batch_size // 2)
            base_config.gpu_memory_limit = 0.7
            base_config.model_caching = False
            base_config.gradient_checkpointing = True
            base_config.memory_efficient_attention = True
            
        elif workload_type == "accuracy_critical":
            # Optimize for accuracy
            base_config.mixed_precision = False
            base_config.model_precision = "fp32"
            base_config.batch_size = max(1, base_config.batch_size // 2)
            base_config.enable_optimization = False
        
        self.logger.info(f"Optimized configuration for {workload_type} workload")
        return base_config


# Global configuration manager instance
_config_manager: Optional[GPUConfigManager] = None


def get_gpu_config_manager() -> GPUConfigManager:
    """Get global GPU configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = GPUConfigManager()
        logger.info("Initialized global GPU configuration manager")
    return _config_manager


def get_optimized_gpu_config(gpu_type: Optional[GPUType] = None) -> GPUConfig:
    """Get optimized GPU configuration."""
    manager = get_gpu_config_manager()
    return manager.get_optimized_config(gpu_type)


def get_extractor_configs(gpu_type: Optional[GPUType] = None) -> Dict[str, Dict[str, Any]]:
    """Get optimized extractor configurations."""
    manager = get_gpu_config_manager()
    return manager.get_extractor_configs(gpu_type)


# Example usage
if __name__ == "__main__":
    # Test GPU configuration manager
    manager = get_gpu_config_manager()
    
    print(f"GPU Type: {manager.gpu_type.value}")
    print(f"GPU Info: {manager.gpu_info}")
    
    # Get optimized configuration
    config = manager.get_optimized_config()
    print(f"\nOptimized Configuration:")
    print(f"  GPU Enabled: {config.gpu_enabled}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Mixed Precision: {config.mixed_precision}")
    print(f"  Max GPU Workers: {config.max_gpu_workers}")
    
    # Get extractor configurations
    extractor_configs = manager.get_extractor_configs()
    print(f"\nExtractor Configurations:")
    for extractor, config in extractor_configs.items():
        print(f"  {extractor}: {config}")
    
    # Get performance estimates
    performance = manager.get_performance_estimates()
    print(f"\nPerformance Estimates:")
    for metric, value in performance.items():
        print(f"  {metric}: {value}")
