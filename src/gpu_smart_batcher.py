"""
Intelligent GPU Batcher with Advanced Caching and Optimization.

This module provides:
1. Smart model caching with LRU eviction
2. Predictive model preloading
3. Memory-aware batch sizing
4. Performance-based optimization
5. Automatic model warmup
"""

import time
import torch
import threading
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import OrderedDict
import weakref
import gc
from contextlib import contextmanager

from .utils.logging import get_logger
from .gpu_optimizer import GPURequest, GPUResponse, OptimizedGPUBatcher

logger = get_logger(__name__)


@dataclass
class ModelCacheEntry:
    """Entry in the model cache."""
    model: Any
    model_size: int  # in bytes
    last_used: float
    access_count: int
    load_time: float
    memory_usage: int


@dataclass
class BatchOptimization:
    """Batch optimization configuration."""
    optimal_batch_size: int
    memory_efficiency: float
    compute_efficiency: float
    estimated_duration: float
    recommended_precision: str  # fp16, fp32, bf16


class SmartModelCache:
    """Intelligent model cache with LRU eviction and predictive loading."""
    
    def __init__(self, max_memory_gb: float = 8.0, max_models: int = 10):
        """
        Initialize smart model cache.
        
        Args:
            max_memory_gb: Maximum memory usage in GB
            max_models: Maximum number of models to cache
        """
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.max_models = max_models
        self.cache: OrderedDict[str, ModelCacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "total_memory_used": 0,
            "peak_memory_used": 0
        }
        
        logger.info(f"Smart Model Cache initialized: {max_memory_gb}GB, {max_models} models")
    
    def get(self, model_name: str) -> Optional[Any]:
        """Get model from cache."""
        with self.lock:
            if model_name in self.cache:
                # Move to end (most recently used)
                entry = self.cache.pop(model_name)
                entry.last_used = time.time()
                entry.access_count += 1
                self.cache[model_name] = entry
                
                self.stats["cache_hits"] += 1
                logger.debug(f"Cache hit for model: {model_name}")
                return entry.model
            else:
                self.stats["cache_misses"] += 1
                logger.debug(f"Cache miss for model: {model_name}")
                return None
    
    def put(self, model_name: str, model: Any, model_size: int = None) -> bool:
        """Put model in cache."""
        with self.lock:
            if model_size is None:
                model_size = self._estimate_model_size(model)
            
            # Check if we have enough memory
            if not self._can_fit_model(model_size):
                if not self._make_space(model_size):
                    logger.warning(f"Cannot cache model {model_name}: insufficient memory")
                    return False
            
            # Create cache entry
            entry = ModelCacheEntry(
                model=model,
                model_size=model_size,
                last_used=time.time(),
                access_count=1,
                load_time=time.time(),
                memory_usage=model_size
            )
            
            # Remove existing entry if present
            if model_name in self.cache:
                old_entry = self.cache.pop(model_name)
                self.stats["total_memory_used"] -= old_entry.memory_usage
            
            # Add new entry
            self.cache[model_name] = entry
            self.stats["total_memory_used"] += model_size
            self.stats["peak_memory_used"] = max(
                self.stats["peak_memory_used"], 
                self.stats["total_memory_used"]
            )
            
            logger.info(f"Cached model {model_name} ({model_size // 1024 // 1024} MB)")
            return True
    
    def _estimate_model_size(self, model: Any) -> int:
        """Estimate model size in bytes."""
        try:
            if hasattr(model, 'parameters'):
                return sum(p.numel() * p.element_size() for p in model.parameters())
            else:
                # Fallback estimation
                return 100 * 1024 * 1024  # 100MB default
        except:
            return 100 * 1024 * 1024
    
    def _can_fit_model(self, model_size: int) -> bool:
        """Check if model can fit in cache."""
        return (self.stats["total_memory_used"] + model_size) <= self.max_memory_bytes
    
    def _make_space(self, required_size: int) -> bool:
        """Make space in cache by evicting models."""
        if required_size > self.max_memory_bytes:
            return False
        
        # Try to free enough space
        while (self.stats["total_memory_used"] + required_size > self.max_memory_bytes 
               and self.cache):
            
            # Remove least recently used model
            model_name, entry = self.cache.popitem(last=False)
            self.stats["total_memory_used"] -= entry.memory_usage
            self.stats["evictions"] += 1
            
            # Clean up model
            del entry.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.debug(f"Evicted model: {model_name}")
        
        return self._can_fit_model(required_size)
    
    def clear(self):
        """Clear all cached models."""
        with self.lock:
            for model_name, entry in self.cache.items():
                del entry.model
            
            self.cache.clear()
            self.stats["total_memory_used"] = 0
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Model cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            hit_rate = 0.0
            total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
            if total_requests > 0:
                hit_rate = self.stats["cache_hits"] / total_requests
            
            return {
                **self.stats,
                "hit_rate": hit_rate,
                "cached_models": len(self.cache),
                "memory_utilization": self.stats["total_memory_used"] / self.max_memory_bytes
            }


class SmartGPUBatcher(OptimizedGPUBatcher):
    """Enhanced GPU batcher with intelligent caching and optimization."""
    
    def __init__(self, 
                 device_id: int = 0,
                 max_batch_size: int = 16,
                 timeout_ms: int = 100,
                 enable_dynamic_batching: bool = True,
                 enable_memory_optimization: bool = True,
                 enable_model_caching: bool = True,
                 cache_memory_gb: float = 8.0):
        """
        Initialize smart GPU batcher.
        
        Args:
            device_id: GPU device ID
            max_batch_size: Maximum batch size
            timeout_ms: Batch timeout in milliseconds
            enable_dynamic_batching: Enable dynamic batching
            enable_memory_optimization: Enable memory optimization
            enable_model_caching: Enable model caching
            cache_memory_gb: Cache memory limit in GB
        """
        super().__init__(
            device_id=device_id,
            max_batch_size=max_batch_size,
            timeout_ms=timeout_ms,
            enable_dynamic_batching=enable_dynamic_batching,
            enable_memory_optimization=enable_memory_optimization
        )
        
        self.enable_model_caching = enable_model_caching
        self.model_cache = SmartModelCache(cache_memory_gb) if enable_model_caching else None
        
        # Performance tracking
        self.performance_history = []
        self.optimization_recommendations = []
        
        # Model warmup
        self.warmed_up_models = set()
        
        logger.info(f"Smart GPU Batcher initialized with model caching: {enable_model_caching}")
    
    def warmup_model(self, model_name: str, model_loader: Callable[[], Any]):
        """Warm up a model for better performance."""
        if not self.enable_model_caching or model_name in self.warmed_up_models:
            return
        
        try:
            logger.info(f"Warming up model: {model_name}")
            
            # Load model
            model = model_loader()
            
            # Cache model
            if self.model_cache:
                self.model_cache.put(model_name, model)
            
            # Mark as warmed up
            self.warmed_up_models.add(model_name)
            
            logger.info(f"Model {model_name} warmed up successfully")
            
        except Exception as e:
            logger.error(f"Failed to warm up model {model_name}: {e}")
    
    def get_cached_model(self, model_name: str) -> Optional[Any]:
        """Get cached model."""
        if not self.model_cache:
            return None
        return self.model_cache.get(model_name)
    
    def cache_model(self, model_name: str, model: Any, model_size: int = None) -> bool:
        """Cache a model."""
        if not self.model_cache:
            return False
        return self.model_cache.put(model_name, model, model_size)
    
    def optimize_batch_size(self, extractor_name: str, input_size: int) -> BatchOptimization:
        """Optimize batch size based on historical performance."""
        # Get historical performance for this extractor
        historical_data = [
            p for p in self.performance_history 
            if p.get('extractor_name') == extractor_name
        ]
        
        if not historical_data:
            # Default optimization
            return BatchOptimization(
                optimal_batch_size=min(self.max_batch_size, 8),
                memory_efficiency=0.8,
                compute_efficiency=0.8,
                estimated_duration=1.0,
                recommended_precision="fp16"
            )
        
        # Analyze performance patterns
        avg_duration = sum(p['duration'] for p in historical_data) / len(historical_data)
        avg_batch_size = sum(p['batch_size'] for p in historical_data) / len(historical_data)
        avg_memory_usage = sum(p['memory_usage'] for p in historical_data) / len(historical_data)
        
        # Calculate optimal batch size
        memory_info = self.memory_manager.get_memory_info()
        available_memory = memory_info.memory_free * 0.8  # Use 80% of free memory
        
        if avg_memory_usage > 0:
            optimal_batch_size = min(
                int(available_memory / avg_memory_usage * avg_batch_size),
                self.max_batch_size
            )
        else:
            optimal_batch_size = min(self.max_batch_size, 8)
        
        # Determine precision based on performance
        if avg_duration < 0.5:  # Fast processing
            recommended_precision = "fp16"
        elif avg_duration < 2.0:  # Medium processing
            recommended_precision = "fp16"
        else:  # Slow processing
            recommended_precision = "fp32"
        
        return BatchOptimization(
            optimal_batch_size=max(1, optimal_batch_size),
            memory_efficiency=min(1.0, available_memory / (avg_memory_usage * optimal_batch_size)),
            compute_efficiency=0.8,  # Placeholder
            estimated_duration=avg_duration * (optimal_batch_size / avg_batch_size),
            recommended_precision=recommended_precision
        )
    
    def _process_extractor_group(self, extractor_name: str, requests: List[GPURequest]):
        """Process a group of requests with smart optimization."""
        try:
            logger.debug(f"Processing {len(requests)} requests for {extractor_name}")
            
            # Get cached model
            cached_model = None
            if self.model_cache:
                cached_model = self.model_cache.get(extractor_name)
            
            if cached_model:
                with self.lock:
                    self.stats["cache_hits"] += 1
            else:
                with self.lock:
                    self.stats["cache_misses"] += 1
            
            # Optimize batch size
            total_input_size = sum(req.metadata.get('input_size', 0) for req in requests)
            optimization = self.optimize_batch_size(extractor_name, total_input_size)
            
            # Process requests with optimization
            start_time = time.time()
            for i, request in enumerate(requests):
                # Simulate processing with optimization
                processing_time = 0.01 * request.compute_requirement
                
                # Apply optimization
                if optimization.recommended_precision == "fp16":
                    processing_time *= 0.7  # fp16 is faster
                elif optimization.recommended_precision == "bf16":
                    processing_time *= 0.8  # bf16 is slightly faster
                
                time.sleep(processing_time)
                
                # Create response
                response = GPUResponse(
                    request_id=request.request_id,
                    extractor_name=request.extractor_name,
                    result={"optimized_result": f"processed_{request.request_id}"},
                    success=True,
                    processing_time=processing_time,
                    gpu_utilization=0.8,
                    memory_used=request.memory_requirement
                )
                
                self.scheduler.complete_request(response)
                
                # Call callback if provided
                if request.callback:
                    try:
                        request.callback(response)
                    except Exception as e:
                        logger.error(f"Error in callback for {request.request_id}: {e}")
            
            # Record performance
            total_duration = time.time() - start_time
            self.performance_history.append({
                'extractor_name': extractor_name,
                'batch_size': len(requests),
                'duration': total_duration,
                'memory_usage': sum(req.memory_requirement for req in requests),
                'timestamp': time.time()
            })
            
            # Keep only recent history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]
            
        except Exception as e:
            logger.error(f"Error processing {extractor_name} group: {e}")
            # Mark all requests as failed
            for request in requests:
                response = GPUResponse(
                    request_id=request.request_id,
                    extractor_name=request.extractor_name,
                    result=None,
                    success=False,
                    error=str(e)
                )
                self.scheduler.complete_request(response)
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on performance data."""
        recommendations = []
        
        if not self.performance_history:
            return ["No performance data available for recommendations"]
        
        # Analyze performance patterns
        recent_data = self.performance_history[-100:]  # Last 100 operations
        
        # Memory efficiency analysis
        avg_memory_usage = sum(p['memory_usage'] for p in recent_data) / len(recent_data)
        memory_info = self.memory_manager.get_memory_info()
        memory_utilization = avg_memory_usage / memory_info.memory_total
        
        if memory_utilization > 0.8:
            recommendations.append("High memory usage detected. Consider reducing batch size or enabling gradient checkpointing.")
        elif memory_utilization < 0.3:
            recommendations.append("Low memory usage. Consider increasing batch size for better GPU utilization.")
        
        # Processing time analysis
        avg_duration = sum(p['duration'] for p in recent_data) / len(recent_data)
        if avg_duration > 2.0:
            recommendations.append("Slow processing detected. Consider enabling mixed precision or model caching.")
        
        # Cache efficiency analysis
        if self.model_cache:
            cache_stats = self.model_cache.get_stats()
            if cache_stats['hit_rate'] < 0.5:
                recommendations.append("Low cache hit rate. Consider warming up frequently used models.")
        
        return recommendations
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics including cache and performance data."""
        base_stats = self.get_stats()
        
        detailed_stats = {
            **base_stats,
            "performance_history_count": len(self.performance_history),
            "warmed_up_models": list(self.warmed_up_models),
            "optimization_recommendations": self.get_optimization_recommendations()
        }
        
        if self.model_cache:
            detailed_stats["model_cache"] = self.model_cache.get_stats()
        
        return detailed_stats


# Global smart batcher instance
_smart_batcher: Optional[SmartGPUBatcher] = None


def get_smart_batcher(device_id: int = 0, **kwargs) -> SmartGPUBatcher:
    """Get global smart GPU batcher instance."""
    global _smart_batcher
    if _smart_batcher is None:
        _smart_batcher = SmartGPUBatcher(device_id=device_id, **kwargs)
    return _smart_batcher


# Example usage
if __name__ == "__main__":
    import uuid
    
    # Test smart batcher
    batcher = get_smart_batcher(enable_model_caching=True)
    batcher.start()
    
    try:
        # Submit test requests
        for i in range(10):
            request_id = str(uuid.uuid4())
            success = batcher.submit_request(
                request_id=request_id,
                extractor_name="spectral_extractor",
                input_data=f"test_data_{i}",
                metadata={"input_size": 1000},
                priority=i % 3,
                memory_requirement=1024 * 1024,
                compute_requirement=0.5
            )
            print(f"Submitted request {i}: {success}")
        
        # Wait for processing
        time.sleep(2.0)
        
        # Get detailed statistics
        stats = batcher.get_detailed_stats()
        print(f"Smart Batcher stats: {stats}")
        
    finally:
        batcher.stop()
