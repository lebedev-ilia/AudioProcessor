"""
Advanced GPU Monitoring and Profiling System for AudioProcessor.

This module provides:
1. Real-time GPU metrics collection
2. Performance profiling and analysis
3. Memory usage tracking
4. Temperature and power monitoring
5. Automatic optimization recommendations
"""

import time
import torch
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import queue
import json
from collections import deque
import numpy as np

# Try to import NVIDIA monitoring libraries
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml not available. Limited GPU monitoring capabilities.")

from .utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GPUMetrics:
    """GPU metrics snapshot."""
    timestamp: float
    device_id: int
    device_name: str
    memory_total: int  # bytes
    memory_used: int   # bytes
    memory_free: int   # bytes
    memory_utilization: float  # 0.0-1.0
    gpu_utilization: float     # 0.0-1.0
    temperature: Optional[float] = None  # Celsius
    power_usage: Optional[float] = None  # Watts
    clock_graphics: Optional[int] = None  # MHz
    clock_memory: Optional[int] = None    # MHz
    fan_speed: Optional[float] = None    # %
    processes_count: int = 0
    compute_mode: Optional[str] = None


@dataclass
class PerformanceProfile:
    """Performance profiling data."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: int
    memory_after: int
    memory_peak: int
    gpu_utilization_avg: float
    gpu_utilization_peak: float
    temperature_avg: float
    temperature_peak: float
    power_avg: float
    power_peak: float
    batch_size: int
    input_size: int
    output_size: int


class GPUProfiler:
    """Advanced GPU profiler with detailed metrics collection."""
    
    def __init__(self, device_id: int = 0, enable_detailed_profiling: bool = True):
        """
        Initialize GPU profiler.
        
        Args:
            device_id: GPU device ID to monitor
            enable_detailed_profiling: Enable detailed profiling (may impact performance)
        """
        self.device_id = device_id
        self.enable_detailed_profiling = enable_detailed_profiling
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        
        # Initialize NVIDIA monitoring
        if NVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self.device_name = pynvml.nvmlDeviceGetName(self.handle).decode('utf-8')
                logger.info(f"GPU Profiler initialized for device: {self.device_name}")
            except Exception as e:
                logger.error(f"Failed to initialize NVIDIA monitoring: {e}")
                self.handle = None
                self.device_name = "Unknown"
        else:
            self.handle = None
            self.device_name = "CPU"
        
        # Profiling state
        self.active_profiles: Dict[str, PerformanceProfile] = {}
        self.completed_profiles: List[PerformanceProfile] = []
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 metrics
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Performance tracking
        self.performance_stats = {
            "total_operations": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0,
            "peak_memory_usage": 0,
            "peak_gpu_utilization": 0.0,
            "peak_temperature": 0.0,
            "operations_per_second": 0.0
        }
        
        logger.info(f"GPU Profiler initialized on device {device_id}")
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous GPU monitoring."""
        if self.is_monitoring:
            logger.warning("GPU monitoring is already running")
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started GPU monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous GPU monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Stopped GPU monitoring")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                metrics = self._collect_metrics()
                if metrics:
                    self.metrics_history.append(metrics)
                    self._update_performance_stats(metrics)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(interval)
    
    def _collect_metrics(self) -> Optional[GPUMetrics]:
        """Collect current GPU metrics."""
        if not torch.cuda.is_available():
            return None
        
        try:
            timestamp = time.time()
            
            # Basic PyTorch metrics
            memory_allocated = torch.cuda.memory_allocated(self.device_id)
            memory_reserved = torch.cuda.memory_reserved(self.device_id)
            memory_total = torch.cuda.get_device_properties(self.device_id).total_memory
            memory_free = memory_total - memory_reserved
            
            # Initialize metrics
            metrics = GPUMetrics(
                timestamp=timestamp,
                device_id=self.device_id,
                device_name=self.device_name,
                memory_total=memory_total,
                memory_used=memory_allocated,
                memory_free=memory_free,
                memory_utilization=memory_reserved / memory_total,
                gpu_utilization=0.0,  # Will be updated if NVML available
                temperature=None,
                power_usage=None,
                processes_count=0
            )
            
            # Detailed metrics from NVML
            if self.handle and NVML_AVAILABLE:
                try:
                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    metrics.gpu_utilization = util.gpu / 100.0
                    
                    # Temperature
                    metrics.temperature = pynvml.nvmlDeviceGetTemperature(
                        self.handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    
                    # Power usage
                    metrics.power_usage = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert to Watts
                    
                    # Clock speeds
                    try:
                        graphics_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
                        memory_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
                        metrics.clock_graphics = graphics_clock
                        metrics.clock_memory = memory_clock
                    except:
                        pass
                    
                    # Fan speed
                    try:
                        fan_speed = pynvml.nvmlDeviceGetFanSpeed(self.handle)
                        metrics.fan_speed = fan_speed
                    except:
                        pass
                    
                    # Process count
                    try:
                        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
                        metrics.processes_count = len(processes)
                    except:
                        pass
                    
                    # Compute mode
                    try:
                        compute_mode = pynvml.nvmlDeviceGetComputeMode(self.handle)
                        metrics.compute_mode = str(compute_mode)
                    except:
                        pass
                        
                except Exception as e:
                    logger.debug(f"Error collecting detailed GPU metrics: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
            return None
    
    def start_profile(self, operation_name: str, batch_size: int = 1, 
                     input_size: int = 0, output_size: int = 0) -> str:
        """Start profiling an operation."""
        profile_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        # Get initial metrics
        initial_metrics = self._collect_metrics()
        memory_before = initial_metrics.memory_used if initial_metrics else 0
        
        profile = PerformanceProfile(
            operation_name=operation_name,
            start_time=time.time(),
            end_time=0.0,
            duration=0.0,
            memory_before=memory_before,
            memory_after=0,
            memory_peak=memory_before,
            gpu_utilization_avg=0.0,
            gpu_utilization_peak=0.0,
            temperature_avg=0.0,
            temperature_peak=0.0,
            power_avg=0.0,
            power_peak=0.0,
            batch_size=batch_size,
            input_size=input_size,
            output_size=output_size
        )
        
        self.active_profiles[profile_id] = profile
        logger.debug(f"Started profiling: {profile_id}")
        return profile_id
    
    def end_profile(self, profile_id: str) -> Optional[PerformanceProfile]:
        """End profiling an operation."""
        if profile_id not in self.active_profiles:
            logger.warning(f"Profile {profile_id} not found")
            return None
        
        profile = self.active_profiles[profile_id]
        profile.end_time = time.time()
        profile.duration = profile.end_time - profile.start_time
        
        # Get final metrics
        final_metrics = self._collect_metrics()
        if final_metrics:
            profile.memory_after = final_metrics.memory_used
            profile.memory_peak = max(profile.memory_peak, final_metrics.memory_used)
            profile.gpu_utilization_avg = final_metrics.gpu_utilization
            profile.gpu_utilization_peak = final_metrics.gpu_utilization
            profile.temperature_avg = final_metrics.temperature or 0.0
            profile.temperature_peak = final_metrics.temperature or 0.0
            profile.power_avg = final_metrics.power_usage or 0.0
            profile.power_peak = final_metrics.power_usage or 0.0
        
        # Move to completed profiles
        del self.active_profiles[profile_id]
        self.completed_profiles.append(profile)
        
        # Update performance stats
        self._update_performance_stats_from_profile(profile)
        
        logger.debug(f"Completed profiling: {profile_id} (duration: {profile.duration:.3f}s)")
        return profile
    
    def _update_performance_stats(self, metrics: GPUMetrics):
        """Update performance statistics from metrics."""
        if metrics.memory_used > self.performance_stats["peak_memory_usage"]:
            self.performance_stats["peak_memory_usage"] = metrics.memory_used
        
        if metrics.gpu_utilization > self.performance_stats["peak_gpu_utilization"]:
            self.performance_stats["peak_gpu_utilization"] = metrics.gpu_utilization
        
        if metrics.temperature and metrics.temperature > self.performance_stats["peak_temperature"]:
            self.performance_stats["peak_temperature"] = metrics.temperature
    
    def _update_performance_stats_from_profile(self, profile: PerformanceProfile):
        """Update performance statistics from completed profile."""
        self.performance_stats["total_operations"] += 1
        self.performance_stats["total_duration"] += profile.duration
        self.performance_stats["avg_duration"] = (
            self.performance_stats["total_duration"] / self.performance_stats["total_operations"]
        )
        
        if profile.memory_peak > self.performance_stats["peak_memory_usage"]:
            self.performance_stats["peak_memory_usage"] = profile.memory_peak
        
        if profile.gpu_utilization_peak > self.performance_stats["peak_gpu_utilization"]:
            self.performance_stats["peak_gpu_utilization"] = profile.gpu_utilization_peak
        
        if profile.temperature_peak > self.performance_stats["peak_temperature"]:
            self.performance_stats["peak_temperature"] = profile.temperature_peak
        
        # Calculate operations per second
        if self.performance_stats["total_duration"] > 0:
            self.performance_stats["operations_per_second"] = (
                self.performance_stats["total_operations"] / self.performance_stats["total_duration"]
            )
    
    def get_current_metrics(self) -> Optional[GPUMetrics]:
        """Get current GPU metrics."""
        return self._collect_metrics()
    
    def get_metrics_history(self, duration_minutes: int = 10) -> List[GPUMetrics]:
        """Get metrics history for the specified duration."""
        cutoff_time = time.time() - (duration_minutes * 60)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        current_metrics = self.get_current_metrics()
        
        summary = {
            "device_info": {
                "device_id": self.device_id,
                "device_name": self.device_name,
                "cuda_available": torch.cuda.is_available(),
                "nvml_available": NVML_AVAILABLE
            },
            "current_metrics": current_metrics.__dict__ if current_metrics else None,
            "performance_stats": self.performance_stats.copy(),
            "active_profiles": len(self.active_profiles),
            "completed_profiles": len(self.completed_profiles),
            "monitoring_active": self.is_monitoring
        }
        
        # Add recent metrics statistics
        if self.metrics_history:
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
            summary["recent_averages"] = {
                "memory_utilization": np.mean([m.memory_utilization for m in recent_metrics]),
                "gpu_utilization": np.mean([m.gpu_utilization for m in recent_metrics]),
                "temperature": np.mean([m.temperature or 0 for m in recent_metrics]),
                "power_usage": np.mean([m.power_usage or 0 for m in recent_metrics])
            }
        
        return summary
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on collected metrics."""
        recommendations = []
        
        if not self.metrics_history:
            return ["No metrics available for recommendations"]
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Memory utilization recommendations
        avg_memory_util = np.mean([m.memory_utilization for m in recent_metrics])
        if avg_memory_util > 0.9:
            recommendations.append("High memory utilization detected. Consider reducing batch size or enabling gradient checkpointing.")
        elif avg_memory_util < 0.3:
            recommendations.append("Low memory utilization. Consider increasing batch size for better GPU utilization.")
        
        # GPU utilization recommendations
        avg_gpu_util = np.mean([m.gpu_utilization for m in recent_metrics])
        if avg_gpu_util < 0.5:
            recommendations.append("Low GPU utilization. Consider increasing batch size or enabling mixed precision.")
        
        # Temperature recommendations
        avg_temp = np.mean([m.temperature or 0 for m in recent_metrics])
        if avg_temp > 80:
            recommendations.append("High GPU temperature detected. Consider improving cooling or reducing workload.")
        
        # Power recommendations
        avg_power = np.mean([m.power_usage or 0 for m in recent_metrics])
        if avg_power > 300:  # Assuming high-end GPU
            recommendations.append("High power usage detected. Consider optimizing model or reducing precision.")
        
        return recommendations
    
    def export_profile_data(self, filepath: str):
        """Export profiling data to JSON file."""
        data = {
            "device_info": {
                "device_id": self.device_id,
                "device_name": self.device_name,
                "cuda_available": torch.cuda.is_available()
            },
            "performance_stats": self.performance_stats,
            "completed_profiles": [
                {
                    "operation_name": p.operation_name,
                    "duration": p.duration,
                    "memory_peak": p.memory_peak,
                    "gpu_utilization_avg": p.gpu_utilization_avg,
                    "temperature_avg": p.temperature_avg,
                    "batch_size": p.batch_size
                }
                for p in self.completed_profiles
            ],
            "metrics_history": [
                {
                    "timestamp": m.timestamp,
                    "memory_utilization": m.memory_utilization,
                    "gpu_utilization": m.gpu_utilization,
                    "temperature": m.temperature,
                    "power_usage": m.power_usage
                }
                for m in list(self.metrics_history)
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported profiling data to {filepath}")


# Global profiler instance
_gpu_profiler: Optional[GPUProfiler] = None


def get_gpu_profiler(device_id: int = 0) -> GPUProfiler:
    """Get global GPU profiler instance."""
    global _gpu_profiler
    if _gpu_profiler is None:
        _gpu_profiler = GPUProfiler(device_id)
    return _gpu_profiler


def start_gpu_monitoring(device_id: int = 0, interval: float = 1.0):
    """Start global GPU monitoring."""
    profiler = get_gpu_profiler(device_id)
    profiler.start_monitoring(interval)


def stop_gpu_monitoring():
    """Stop global GPU monitoring."""
    global _gpu_profiler
    if _gpu_profiler:
        _gpu_profiler.stop_monitoring()


# Context manager for profiling
class GPUProfileContext:
    """Context manager for GPU profiling."""
    
    def __init__(self, operation_name: str, batch_size: int = 1, 
                 input_size: int = 0, output_size: int = 0):
        self.operation_name = operation_name
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.profile_id = None
        self.profiler = get_gpu_profiler()
    
    def __enter__(self):
        self.profile_id = self.profiler.start_profile(
            self.operation_name, self.batch_size, self.input_size, self.output_size
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profile_id:
            self.profiler.end_profile(self.profile_id)


# Example usage
if __name__ == "__main__":
    # Test GPU profiler
    profiler = get_gpu_profiler()
    
    # Start monitoring
    profiler.start_monitoring(interval=0.5)
    
    try:
        # Simulate some GPU operations
        with GPUProfileContext("test_operation", batch_size=8):
            if torch.cuda.is_available():
                # Simulate GPU work
                x = torch.randn(1000, 1000, device="cuda")
                y = torch.mm(x, x.t())
                del x, y
                torch.cuda.empty_cache()
        
        # Wait a bit
        time.sleep(2)
        
        # Get performance summary
        summary = profiler.get_performance_summary()
        print("Performance Summary:")
        print(json.dumps(summary, indent=2))
        
        # Get recommendations
        recommendations = profiler.get_optimization_recommendations()
        print("\nOptimization Recommendations:")
        for rec in recommendations:
            print(f"- {rec}")
    
    finally:
        profiler.stop_monitoring()
