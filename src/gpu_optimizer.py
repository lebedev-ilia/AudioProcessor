"""
Advanced GPU Optimizer for maximum GPU utilization in AudioProcessor.

This module provides:
1. Dynamic GPU memory management
2. Intelligent batching and queuing
3. GPU resource monitoring and optimization
4. Automatic scaling and load balancing
5. Memory-efficient model loading and caching
"""

import asyncio
import time
import numpy as np
import torch
import psutil
import gc
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import logging
from dataclasses import dataclass, field
from queue import PriorityQueue, Empty
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import weakref

from .utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GPUResourceInfo:
    """Information about GPU resources and utilization."""
    device_id: int
    device_name: str
    memory_total: int  # in bytes
    memory_allocated: int  # in bytes
    memory_free: int  # in bytes
    memory_utilization: float  # 0.0-1.0
    compute_utilization: float  # 0.0-1.0 (estimated)
    temperature: Optional[float] = None  # in Celsius
    power_usage: Optional[float] = None  # in Watts
    is_available: bool = True


@dataclass
class GPURequest:
    """GPU processing request with priority and resource requirements."""
    request_id: str
    extractor_name: str
    input_data: Any
    metadata: Dict[str, Any]
    priority: int = 0  # Higher number = higher priority
    memory_requirement: int = 0  # Estimated memory requirement in bytes
    compute_requirement: float = 1.0  # Estimated compute requirement (0.0-1.0)
    timestamp: float = field(default_factory=time.time)
    callback: Optional[Callable] = None
    timeout: float = 300.0  # Request timeout in seconds


@dataclass
class GPUResponse:
    """GPU processing response."""
    request_id: str
    extractor_name: str
    result: Any
    success: bool
    error: Optional[str] = None
    processing_time: float = 0.0
    gpu_utilization: float = 0.0
    memory_used: int = 0


class GPUMemoryManager:
    """Advanced GPU memory management with caching and optimization."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.memory_limit = 0.9  # Use 90% of available memory
        self.cache_size_limit = 0.3  # Use 30% for model caching
        self._model_cache = {}
        self._cache_refs = weakref.WeakValueDictionary()
        self._memory_stats = {
            "total_allocated": 0,
            "cache_allocated": 0,
            "peak_usage": 0,
            "allocation_count": 0
        }
        
    def get_memory_info(self) -> GPUResourceInfo:
        """Get current GPU memory information."""
        if not torch.cuda.is_available():
            return GPUResourceInfo(
                device_id=self.device_id,
                device_name="CPU",
                memory_total=psutil.virtual_memory().total,
                memory_allocated=0,
                memory_free=psutil.virtual_memory().available,
                memory_utilization=0.0,
                compute_utilization=0.0,
                is_available=True
            )
        
        try:
            memory_allocated = torch.cuda.memory_allocated(self.device_id)
            memory_reserved = torch.cuda.memory_reserved(self.device_id)
            memory_total = torch.cuda.get_device_properties(self.device_id).total_memory
            memory_free = memory_total - memory_reserved
            
            return GPUResourceInfo(
                device_id=self.device_id,
                device_name=torch.cuda.get_device_name(self.device_id),
                memory_total=memory_total,
                memory_allocated=memory_allocated,
                memory_free=memory_free,
                memory_utilization=memory_reserved / memory_total,
                compute_utilization=self._estimate_compute_utilization(),
                is_available=True
            )
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return GPUResourceInfo(
                device_id=self.device_id,
                device_name="Unknown",
                memory_total=0,
                memory_allocated=0,
                memory_free=0,
                memory_utilization=0.0,
                compute_utilization=0.0,
                is_available=False
            )
    
    def _estimate_compute_utilization(self) -> float:
        """Estimate GPU compute utilization (simplified)."""
        # This is a simplified estimation - in production, you'd use nvidia-ml-py
        try:
            # Check if there are active CUDA operations
            torch.cuda.synchronize()
            return 0.5  # Placeholder - would need nvidia-ml-py for accurate measurement
        except:
            return 0.0
    
    def can_allocate(self, size_bytes: int) -> bool:
        """Check if we can allocate the requested amount of memory."""
        memory_info = self.get_memory_info()
        if not memory_info.is_available:
            return False
        
        available_memory = memory_info.memory_free
        return size_bytes <= available_memory * self.memory_limit
    
    def allocate_memory(self, size_bytes: int) -> bool:
        """Allocate memory and track usage."""
        if not self.can_allocate(size_bytes):
            return False
        
        try:
            # Allocate a dummy tensor to reserve memory
            dummy_tensor = torch.empty(size_bytes // 4, dtype=torch.float32, device=self.device)
            self._memory_stats["total_allocated"] += size_bytes
            self._memory_stats["allocation_count"] += 1
            self._memory_stats["peak_usage"] = max(
                self._memory_stats["peak_usage"],
                self._memory_stats["total_allocated"]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to allocate {size_bytes} bytes: {e}")
            return False
    
    def cache_model(self, model_name: str, model: torch.nn.Module) -> bool:
        """Cache a model in GPU memory."""
        try:
            model_size = sum(p.numel() * p.element_size() for p in model.parameters())
            
            if not self.can_allocate(model_size):
                logger.warning(f"Cannot cache model {model_name}: insufficient memory")
                return False
            
            self._model_cache[model_name] = model
            self._cache_refs[model_name] = model
            self._memory_stats["cache_allocated"] += model_size
            logger.info(f"Cached model {model_name} ({model_size // 1024 // 1024} MB)")
            return True
        except Exception as e:
            logger.error(f"Failed to cache model {model_name}: {e}")
            return False
    
    def get_cached_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """Get a cached model."""
        return self._model_cache.get(model_name)
    
    def clear_cache(self):
        """Clear model cache to free memory."""
        self._model_cache.clear()
        self._cache_refs.clear()
        self._memory_stats["cache_allocated"] = 0
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleared GPU model cache")
    
    def optimize_memory(self):
        """Optimize GPU memory usage."""
        # Clear unused cache
        self.clear_cache()
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Reset memory stats
        self._memory_stats["total_allocated"] = 0
        self._memory_stats["allocation_count"] = 0
        
        logger.info("Optimized GPU memory usage")


class GPURequestScheduler:
    """Intelligent GPU request scheduler with priority and resource management."""
    
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.request_queue = PriorityQueue(maxsize=max_queue_size)
        self.pending_requests: Dict[str, GPURequest] = {}
        self.completed_requests: Dict[str, GPUResponse] = {}
        self.lock = threading.Lock()
        
    def submit_request(self, request: GPURequest) -> bool:
        """Submit a GPU request with priority."""
        try:
            # Use negative priority for max-heap behavior
            priority_item = (-request.priority, request.timestamp, request)
            self.request_queue.put_nowait(priority_item)
            
            with self.lock:
                self.pending_requests[request.request_id] = request
            
            logger.debug(f"Submitted GPU request: {request.request_id} (priority: {request.priority})")
            return True
        except:
            logger.warning(f"GPU request queue full, dropping request: {request.request_id}")
            return False
    
    def get_next_request(self, timeout: float = 1.0) -> Optional[GPURequest]:
        """Get the next highest priority request."""
        try:
            priority_item = self.request_queue.get(timeout=timeout)
            _, _, request = priority_item
            return request
        except Empty:
            return None
    
    def complete_request(self, response: GPUResponse):
        """Mark a request as completed."""
        with self.lock:
            if response.request_id in self.pending_requests:
                del self.pending_requests[response.request_id]
            self.completed_requests[response.request_id] = response
    
    def get_response(self, request_id: str) -> Optional[GPUResponse]:
        """Get response for a completed request."""
        with self.lock:
            return self.completed_requests.pop(request_id, None)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            return {
                "queue_size": self.request_queue.qsize(),
                "pending_requests": len(self.pending_requests),
                "completed_requests": len(self.completed_requests)
            }


class OptimizedGPUBatcher:
    """Optimized GPU batcher with advanced features."""
    
    def __init__(self, 
                 device_id: int = 0,
                 max_batch_size: int = 16,
                 timeout_ms: int = 100,
                 enable_dynamic_batching: bool = True,
                 enable_memory_optimization: bool = True):
        self.device_id = device_id
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.enable_dynamic_batching = enable_dynamic_batching
        self.enable_memory_optimization = enable_memory_optimization
        
        # Initialize components
        self.memory_manager = GPUMemoryManager(device_id)
        self.scheduler = GPURequestScheduler()
        
        # Processing state
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "total_processing_time": 0.0,
            "average_batch_size": 0.0,
            "memory_optimizations": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(f"Initialized OptimizedGPUBatcher on device {device_id}")
    
    def start(self):
        """Start the GPU batcher processing thread."""
        if self.is_running:
            logger.warning("GPU Batcher is already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Optimized GPU Batcher started")
    
    def stop(self):
        """Stop the GPU batcher processing thread."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        logger.info("Optimized GPU Batcher stopped")
    
    def submit_request(self, 
                      request_id: str,
                      extractor_name: str,
                      input_data: Any,
                      metadata: Dict[str, Any],
                      priority: int = 0,
                      memory_requirement: int = 0,
                      compute_requirement: float = 1.0,
                      callback: Optional[Callable] = None) -> bool:
        """Submit a GPU processing request."""
        request = GPURequest(
            request_id=request_id,
            extractor_name=extractor_name,
            input_data=input_data,
            metadata=metadata,
            priority=priority,
            memory_requirement=memory_requirement,
            compute_requirement=compute_requirement,
            callback=callback
        )
        
        success = self.scheduler.submit_request(request)
        if success:
            with self.lock:
                self.stats["total_requests"] += 1
        
        return success
    
    def get_response(self, request_id: str, timeout: float = 30.0) -> Optional[GPUResponse]:
        """Get response for a request."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.scheduler.get_response(request_id)
            if response:
                return response
            time.sleep(0.01)
        
        logger.warning(f"Timeout waiting for GPU response: {request_id}")
        return None
    
    def _processing_loop(self):
        """Main processing loop for GPU batcher."""
        logger.info("Optimized GPU Batcher processing loop started")
        
        while self.is_running:
            try:
                # Collect requests for batch
                batch = self._collect_optimized_batch()
                
                if batch:
                    # Process batch
                    self._process_optimized_batch(batch)
                
                # Memory optimization
                if self.enable_memory_optimization:
                    self._optimize_memory_if_needed()
                
            except Exception as e:
                logger.error(f"Error in GPU batcher processing loop: {e}")
                time.sleep(0.1)
        
        logger.info("Optimized GPU Batcher processing loop stopped")
    
    def _collect_optimized_batch(self) -> List[GPURequest]:
        """Collect requests for optimized batch processing."""
        batch = []
        start_time = time.time()
        
        # Get first request (blocking)
        request = self.scheduler.get_next_request(timeout=1.0)
        if not request:
            return []
        
        batch.append(request)
        
        # Collect additional requests based on optimization strategy
        if self.enable_dynamic_batching:
            batch = self._collect_dynamic_batch(batch, start_time)
        else:
            batch = self._collect_static_batch(batch, start_time)
        
        if batch:
            logger.debug(f"Collected batch of {len(batch)} requests")
        
        return batch
    
    def _collect_dynamic_batch(self, initial_batch: List[GPURequest], start_time: float) -> List[GPURequest]:
        """Collect batch using intelligent dynamic batching strategy."""
        batch = initial_batch.copy()
        current_memory_requirement = sum(req.memory_requirement for req in batch)
        current_compute_requirement = sum(req.compute_requirement for req in batch)
        
        # Get current GPU metrics for intelligent batching
        memory_info = self.memory_manager.get_memory_info()
        gpu_utilization = self._get_gpu_utilization()
        
        # Adaptive timeout based on GPU utilization
        adaptive_timeout = self.timeout_ms
        if gpu_utilization < 0.3:  # Low utilization, wait longer for more requests
            adaptive_timeout = min(self.timeout_ms * 2, 500)
        elif gpu_utilization > 0.8:  # High utilization, process quickly
            adaptive_timeout = max(self.timeout_ms // 2, 10)
        
        # Collect more requests based on resource requirements and GPU state
        while len(batch) < self.max_batch_size:
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Check adaptive timeout
            if elapsed_ms >= adaptive_timeout:
                break
            
            # Get next request
            request = self.scheduler.get_next_request(timeout=0.01)
            if not request:
                break
            
            # Check if we can add this request
            new_memory_req = current_memory_requirement + request.memory_requirement
            new_compute_req = current_compute_requirement + request.compute_requirement
            
            # Dynamic memory threshold based on GPU utilization
            memory_threshold = 0.8 if gpu_utilization < 0.5 else 0.7
            if new_memory_req > memory_info.memory_free * memory_threshold:
                logger.debug(f"Memory constraint reached (threshold: {memory_threshold:.1%}), stopping batch collection")
                break
            
            # Adaptive compute constraints
            max_compute = 2.0 if gpu_utilization < 0.5 else 1.5
            if new_compute_req > max_compute:
                logger.debug(f"Compute constraint reached (max: {max_compute}), stopping batch collection")
                break
            
            # Check if request is compatible with current batch
            if not self._is_request_compatible(batch, request):
                logger.debug(f"Request {request.request_id} not compatible with current batch")
                break
            
            batch.append(request)
            current_memory_requirement = new_memory_req
            current_compute_requirement = new_compute_req
        
        return batch
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization."""
        try:
            if torch.cuda.is_available():
                # Simple utilization estimation based on memory usage
                memory_info = self.memory_manager.get_memory_info()
                return memory_info.memory_utilization
        except:
            pass
        return 0.0
    
    def _is_request_compatible(self, batch: List[GPURequest], request: GPURequest) -> bool:
        """Check if request is compatible with current batch."""
        if not batch:
            return True
        
        # Check if extractor types are compatible for batching
        batch_extractors = {req.extractor_name for req in batch}
        if len(batch_extractors) > 1 and request.extractor_name not in batch_extractors:
            # Mixed extractor types - check if they can be batched together
            return self._can_batch_extractors(list(batch_extractors) + [request.extractor_name])
        
        return True
    
    def _can_batch_extractors(self, extractor_names: List[str]) -> bool:
        """Check if multiple extractor types can be batched together."""
        # Define compatible extractor groups
        compatible_groups = [
            {"spectral_extractor", "mfcc_extractor", "chroma_extractor"},
            {"clap_extractor", "advanced_embeddings_extractor"},
            {"asr_extractor"}
        ]
        
        for group in compatible_groups:
            if all(name in group for name in extractor_names):
                return True
        
        return False
    
    def _collect_static_batch(self, initial_batch: List[GPURequest], start_time: float) -> List[GPURequest]:
        """Collect batch using static batching strategy."""
        batch = initial_batch.copy()
        
        # Collect additional requests (non-blocking)
        while len(batch) < self.max_batch_size:
            request = self.scheduler.get_next_request(timeout=0.01)
            if not request:
                break
            batch.append(request)
        
        return batch
    
    def _process_optimized_batch(self, batch: List[GPURequest]):
        """Process a batch of GPU requests with optimization."""
        if not batch:
            return
        
        start_time = time.time()
        
        try:
            # Group requests by extractor type for efficient processing
            extractor_groups = {}
            for request in batch:
                if request.extractor_name not in extractor_groups:
                    extractor_groups[request.extractor_name] = []
                extractor_groups[request.extractor_name].append(request)
            
            # Process each extractor group
            for extractor_name, requests in extractor_groups.items():
                self._process_extractor_group(extractor_name, requests)
            
            # Update statistics
            processing_time = time.time() - start_time
            with self.lock:
                self.stats["total_batches"] += 1
                self.stats["total_processing_time"] += processing_time
                self.stats["average_batch_size"] = (
                    (self.stats["average_batch_size"] * (self.stats["total_batches"] - 1) + len(batch)) 
                    / self.stats["total_batches"]
                )
            
            logger.debug(f"Processed optimized batch of {len(batch)} requests in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing optimized GPU batch: {e}")
            # Mark all requests as failed
            for request in batch:
                response = GPUResponse(
                    request_id=request.request_id,
                    extractor_name=request.extractor_name,
                    result=None,
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time
                )
                self.scheduler.complete_request(response)
    
    def _process_extractor_group(self, extractor_name: str, requests: List[GPURequest]):
        """Process a group of requests for a specific extractor."""
        try:
            logger.debug(f"Processing {len(requests)} requests for {extractor_name}")
            
            # Check if model is cached
            cached_model = self.memory_manager.get_cached_model(extractor_name)
            if cached_model:
                with self.lock:
                    self.stats["cache_hits"] += 1
            else:
                with self.lock:
                    self.stats["cache_misses"] += 1
            
            # Process requests (placeholder implementation)
            for request in requests:
                # Simulate processing time based on compute requirement
                processing_time = 0.01 * request.compute_requirement
                time.sleep(processing_time)
                
                # Create mock response
                response = GPUResponse(
                    request_id=request.request_id,
                    extractor_name=request.extractor_name,
                    result={"optimized_result": f"processed_{request.request_id}"},
                    success=True,
                    processing_time=processing_time,
                    gpu_utilization=0.8,  # Placeholder
                    memory_used=request.memory_requirement
                )
                
                self.scheduler.complete_request(response)
                
                # Call callback if provided
                if request.callback:
                    try:
                        request.callback(response)
                    except Exception as e:
                        logger.error(f"Error in callback for {request.request_id}: {e}")
        
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
    
    def _optimize_memory_if_needed(self):
        """Optimize memory if utilization is too high."""
        memory_info = self.memory_manager.get_memory_info()
        
        if memory_info.memory_utilization > 0.8:  # If using more than 80% of memory
            logger.info("High memory utilization detected, optimizing...")
            self.memory_manager.optimize_memory()
            
            with self.lock:
                self.stats["memory_optimizations"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        memory_info = self.memory_manager.get_memory_info()
        queue_stats = self.scheduler.get_queue_stats()
        
        with self.lock:
            stats = self.stats.copy()
            stats.update({
                "memory_info": {
                    "total": memory_info.memory_total,
                    "allocated": memory_info.memory_allocated,
                    "free": memory_info.memory_free,
                    "utilization": memory_info.memory_utilization
                },
                "queue_stats": queue_stats,
                "average_processing_time": (
                    stats["total_processing_time"] / max(1, stats["total_batches"])
                )
            })
        
        return stats


class GPUOptimizer:
    """Main GPU optimizer that manages multiple GPU batchers and resources."""
    
    def __init__(self, gpu_count: Optional[int] = None):
        self.gpu_count = gpu_count or (torch.cuda.device_count() if torch.cuda.is_available() else 0)
        self.batchers: List[OptimizedGPUBatcher] = []
        self.current_batcher = 0
        
        # Initialize batchers for each GPU
        for device_id in range(self.gpu_count):
            batcher = OptimizedGPUBatcher(device_id=device_id)
            self.batchers.append(batcher)
        
        logger.info(f"Initialized GPU Optimizer with {self.gpu_count} GPUs")
    
    def start(self):
        """Start all GPU batchers."""
        for batcher in self.batchers:
            batcher.start()
        logger.info("All GPU batchers started")
    
    def stop(self):
        """Stop all GPU batchers."""
        for batcher in self.batchers:
            batcher.stop()
        logger.info("All GPU batchers stopped")
    
    def submit_request(self, 
                      request_id: str,
                      extractor_name: str,
                      input_data: Any,
                      metadata: Dict[str, Any],
                      priority: int = 0,
                      memory_requirement: int = 0,
                      compute_requirement: float = 1.0,
                      callback: Optional[Callable] = None) -> bool:
        """Submit request to the best available GPU batcher."""
        if not self.batchers:
            logger.error("No GPU batchers available")
            return False
        
        # Round-robin selection (could be improved with load balancing)
        batcher = self.batchers[self.current_batcher]
        self.current_batcher = (self.current_batcher + 1) % len(self.batchers)
        
        return batcher.submit_request(
            request_id=request_id,
            extractor_name=extractor_name,
            input_data=input_data,
            metadata=metadata,
            priority=priority,
            memory_requirement=memory_requirement,
            compute_requirement=compute_requirement,
            callback=callback
        )
    
    def get_response(self, request_id: str, timeout: float = 30.0) -> Optional[GPUResponse]:
        """Get response from any batcher."""
        # Try all batchers
        for batcher in self.batchers:
            response = batcher.get_response(request_id, timeout=0.1)
            if response:
                return response
        
        # If not found, wait longer on first batcher
        return self.batchers[0].get_response(request_id, timeout) if self.batchers else None
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all batchers."""
        all_stats = {}
        for i, batcher in enumerate(self.batchers):
            all_stats[f"gpu_{i}"] = batcher.get_stats()
        return all_stats
    
    def optimize_all_memory(self):
        """Optimize memory on all GPUs."""
        for i, batcher in enumerate(self.batchers):
            logger.info(f"Optimizing memory on GPU {i}")
            batcher.memory_manager.optimize_memory()


# Global GPU optimizer instance
_gpu_optimizer: Optional[GPUOptimizer] = None


def get_gpu_optimizer() -> GPUOptimizer:
    """Get global GPU optimizer instance."""
    global _gpu_optimizer
    if _gpu_optimizer is None:
        _gpu_optimizer = GPUOptimizer()
        _gpu_optimizer.start()
        logger.info("Initialized global GPU optimizer")
    
    return _gpu_optimizer


def shutdown_gpu_optimizer():
    """Shutdown global GPU optimizer."""
    global _gpu_optimizer
    if _gpu_optimizer:
        _gpu_optimizer.stop()
        _gpu_optimizer = None
        logger.info("Shutdown global GPU optimizer")


# Example usage
if __name__ == "__main__":
    import uuid
    
    # Test GPU optimizer
    optimizer = get_gpu_optimizer()
    
    try:
        # Submit test requests
        for i in range(10):
            request_id = str(uuid.uuid4())
            success = optimizer.submit_request(
                request_id=request_id,
                extractor_name="clap_extractor",
                input_data=f"test_data_{i}",
                metadata={"test": True},
                priority=i % 3,  # Varying priorities
                memory_requirement=1024 * 1024,  # 1MB
                compute_requirement=0.5
            )
            print(f"Submitted request {i}: {success}")
        
        # Wait for responses
        time.sleep(2.0)
        
        # Get statistics
        stats = optimizer.get_all_stats()
        print(f"GPU Optimizer stats: {stats}")
        
    finally:
        shutdown_gpu_optimizer()
