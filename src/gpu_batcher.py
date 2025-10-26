"""
GPU Batcher for efficient GPU extractor processing.

This module provides a GPU batching system that:
1. Collects GPU extractor requests in batches
2. Processes batches efficiently on GPU
3. Manages GPU memory and resources
4. Provides backpressure control
"""

import asyncio
import time
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Callable, Tuple
import logging
from dataclasses import dataclass
from queue import Queue, Empty
import threading
from concurrent.futures import ThreadPoolExecutor

from .utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GPURequest:
    """GPU processing request."""
    request_id: str
    extractor_name: str
    input_data: Any
    metadata: Dict[str, Any]
    timestamp: float
    callback: Optional[Callable] = None


@dataclass
class GPUResponse:
    """GPU processing response."""
    request_id: str
    extractor_name: str
    result: Any
    success: bool
    error: Optional[str] = None
    processing_time: float = 0.0


class GPUBatcher:
    """GPU batcher for efficient processing of GPU extractors."""
    
    def __init__(self, 
                 batch_size: int = 8,
                 timeout_ms: int = 100,
                 max_queue_size: int = 1000,
                 gpu_id: int = 0):
        """
        Initialize GPU batcher.
        
        Args:
            batch_size: Maximum batch size for processing
            timeout_ms: Maximum wait time in milliseconds before processing batch
            max_queue_size: Maximum queue size for backpressure
            gpu_id: GPU device ID
        """
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.max_queue_size = max_queue_size
        self.gpu_id = gpu_id
        
        # Request queue and response storage
        self.request_queue = Queue(maxsize=max_queue_size)
        self.pending_requests: Dict[str, GPURequest] = {}
        self.responses: Dict[str, GPUResponse] = {}
        
        # Processing state
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # GPU device setup
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        logger.info(f"GPU Batcher initialized on device: {self.device}")
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "total_processing_time": 0.0,
            "average_batch_size": 0.0,
            "queue_overflows": 0
        }
    
    def start(self):
        """Start the GPU batcher processing thread."""
        if self.is_running:
            logger.warning("GPU Batcher is already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("GPU Batcher started")
    
    def stop(self):
        """Stop the GPU batcher processing thread."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        logger.info("GPU Batcher stopped")
    
    def submit_request(self, 
                      request_id: str,
                      extractor_name: str,
                      input_data: Any,
                      metadata: Dict[str, Any],
                      callback: Optional[Callable] = None) -> bool:
        """
        Submit a GPU processing request.
        
        Args:
            request_id: Unique request identifier
            extractor_name: Name of the extractor
            input_data: Input data for processing
            metadata: Additional metadata
            callback: Optional callback function
            
        Returns:
            True if request was submitted successfully, False if queue is full
        """
        try:
            request = GPURequest(
                request_id=request_id,
                extractor_name=extractor_name,
                input_data=input_data,
                metadata=metadata,
                timestamp=time.time(),
                callback=callback
            )
            
            # Try to add to queue (non-blocking)
            self.request_queue.put_nowait(request)
            
            with self.lock:
                self.pending_requests[request_id] = request
                self.stats["total_requests"] += 1
            
            logger.debug(f"Submitted GPU request: {request_id} for {extractor_name}")
            return True
            
        except:
            # Queue is full
            with self.lock:
                self.stats["queue_overflows"] += 1
            logger.warning(f"GPU request queue full, dropping request: {request_id}")
            return False
    
    def get_response(self, request_id: str, timeout: float = 30.0) -> Optional[GPUResponse]:
        """
        Get response for a request.
        
        Args:
            request_id: Request identifier
            timeout: Maximum wait time in seconds
            
        Returns:
            GPUResponse if available, None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                if request_id in self.responses:
                    response = self.responses.pop(request_id)
                    if request_id in self.pending_requests:
                        del self.pending_requests[request_id]
                    return response
            
            time.sleep(0.01)  # Small delay to avoid busy waiting
        
        # Timeout
        with self.lock:
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
        
        logger.warning(f"Timeout waiting for GPU response: {request_id}")
        return None
    
    def _processing_loop(self):
        """Main processing loop for GPU batcher."""
        logger.info("GPU Batcher processing loop started")
        
        while self.is_running:
            try:
                # Collect requests for batch
                batch = self._collect_batch()
                
                if batch:
                    # Process batch
                    self._process_batch(batch)
                
            except Exception as e:
                logger.error(f"Error in GPU batcher processing loop: {e}")
                time.sleep(0.1)  # Small delay on error
        
        logger.info("GPU Batcher processing loop stopped")
    
    def _collect_batch(self) -> List[GPURequest]:
        """Collect requests for batch processing."""
        batch = []
        start_time = time.time()
        
        # Collect first request (blocking)
        try:
            request = self.request_queue.get(timeout=1.0)
            batch.append(request)
        except Empty:
            return []
        
        # Collect additional requests (non-blocking)
        while len(batch) < self.batch_size:
            try:
                request = self.request_queue.get_nowait()
                batch.append(request)
            except Empty:
                break
        
        # Check timeout
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms >= self.timeout_ms and len(batch) > 0:
            logger.debug(f"Processing batch of {len(batch)} requests (timeout: {elapsed_ms:.1f}ms)")
            return batch
        
        # Wait for more requests if batch is small
        if len(batch) < self.batch_size:
            try:
                request = self.request_queue.get(timeout=(self.timeout_ms - elapsed_ms) / 1000.0)
                batch.append(request)
            except Empty:
                pass
        
        if batch:
            logger.debug(f"Processing batch of {len(batch)} requests")
        
        return batch
    
    def _process_batch(self, batch: List[GPURequest]):
        """Process a batch of GPU requests."""
        if not batch:
            return
        
        start_time = time.time()
        
        try:
            # Group requests by extractor type
            extractor_groups = {}
            for request in batch:
                if request.extractor_name not in extractor_groups:
                    extractor_groups[request.extractor_name] = []
                extractor_groups[request.extractor_name].append(request)
            
            # Process each extractor group
            for extractor_name, requests in extractor_groups.items():
                self._process_extractor_batch(extractor_name, requests)
            
            # Update statistics
            processing_time = time.time() - start_time
            with self.lock:
                self.stats["total_batches"] += 1
                self.stats["total_processing_time"] += processing_time
                self.stats["average_batch_size"] = (
                    (self.stats["average_batch_size"] * (self.stats["total_batches"] - 1) + len(batch)) 
                    / self.stats["total_batches"]
                )
            
            logger.debug(f"Processed batch of {len(batch)} requests in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing GPU batch: {e}")
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
                self._store_response(response)
    
    def _process_extractor_batch(self, extractor_name: str, requests: List[GPURequest]):
        """Process a batch of requests for a specific extractor."""
        try:
            # This is a placeholder for actual GPU processing
            # In real implementation, this would:
            # 1. Load the appropriate model for the extractor
            # 2. Prepare input data in batch format
            # 3. Run inference on GPU
            # 4. Process results
            
            logger.debug(f"Processing {len(requests)} requests for {extractor_name}")
            
            # Simulate GPU processing
            for request in requests:
                # Simulate processing time
                time.sleep(0.01)
                
                # Create mock response
                response = GPUResponse(
                    request_id=request.request_id,
                    extractor_name=request.extractor_name,
                    result={"mock_result": f"processed_{request.request_id}"},
                    success=True,
                    processing_time=0.01
                )
                
                self._store_response(response)
                
                # Call callback if provided
                if request.callback:
                    try:
                        request.callback(response)
                    except Exception as e:
                        logger.error(f"Error in callback for {request.request_id}: {e}")
        
        except Exception as e:
            logger.error(f"Error processing {extractor_name} batch: {e}")
            # Mark all requests as failed
            for request in requests:
                response = GPUResponse(
                    request_id=request.request_id,
                    extractor_name=request.extractor_name,
                    result=None,
                    success=False,
                    error=str(e)
                )
                self._store_response(response)
    
    def _store_response(self, response: GPUResponse):
        """Store response and notify waiting threads."""
        with self.lock:
            self.responses[response.request_id] = response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        with self.lock:
            stats = self.stats.copy()
            stats["queue_size"] = self.request_queue.qsize()
            stats["pending_requests"] = len(self.pending_requests)
            stats["average_processing_time"] = (
                stats["total_processing_time"] / max(1, stats["total_batches"])
            )
        return stats
    
    def reset_stats(self):
        """Reset batcher statistics."""
        with self.lock:
            self.stats = {
                "total_requests": 0,
                "total_batches": 0,
                "total_processing_time": 0.0,
                "average_batch_size": 0.0,
                "queue_overflows": 0
            }


class GPUManager:
    """Manager for multiple GPU batchers."""
    
    def __init__(self, gpu_count: int = 1, batch_size: int = 8):
        """
        Initialize GPU manager.
        
        Args:
            gpu_count: Number of GPUs to use
            batch_size: Batch size for each GPU
        """
        self.gpu_count = gpu_count
        self.batch_size = batch_size
        self.batchers: List[GPUBatcher] = []
        self.current_gpu = 0
        
        # Initialize batchers
        for gpu_id in range(gpu_count):
            batcher = GPUBatcher(
                batch_size=batch_size,
                gpu_id=gpu_id
            )
            self.batchers.append(batcher)
        
        logger.info(f"GPU Manager initialized with {gpu_count} GPUs")
    
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
                      callback: Optional[Callable] = None) -> bool:
        """Submit request to next available GPU batcher."""
        # Round-robin selection
        gpu_id = self.current_gpu
        self.current_gpu = (self.current_gpu + 1) % self.gpu_count
        
        return self.batchers[gpu_id].submit_request(
            request_id, extractor_name, input_data, metadata, callback
        )
    
    def get_response(self, request_id: str, timeout: float = 30.0) -> Optional[GPUResponse]:
        """Get response from any batcher."""
        # Try all batchers
        for batcher in self.batchers:
            response = batcher.get_response(request_id, timeout=0.1)
            if response:
                return response
        
        # If not found, wait longer on first batcher
        return self.batchers[0].get_response(request_id, timeout)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all batchers."""
        all_stats = {}
        for i, batcher in enumerate(self.batchers):
            all_stats[f"gpu_{i}"] = batcher.get_stats()
        return all_stats


# Global GPU manager instance
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get global GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        gpu_count = 1
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
        
        _gpu_manager = GPUManager(gpu_count=gpu_count)
        _gpu_manager.start()
        logger.info(f"Initialized global GPU manager with {gpu_count} GPUs")
    
    return _gpu_manager


def shutdown_gpu_manager():
    """Shutdown global GPU manager."""
    global _gpu_manager
    if _gpu_manager:
        _gpu_manager.stop()
        _gpu_manager = None
        logger.info("Shutdown global GPU manager")


# Example usage
if __name__ == "__main__":
    import uuid
    
    # Test GPU batcher
    batcher = GPUBatcher(batch_size=4, timeout_ms=50)
    batcher.start()
    
    try:
        # Submit test requests
        for i in range(10):
            request_id = str(uuid.uuid4())
            success = batcher.submit_request(
                request_id=request_id,
                extractor_name="clap_extractor",
                input_data=f"test_data_{i}",
                metadata={"test": True}
            )
            print(f"Submitted request {i}: {success}")
        
        # Wait for responses
        time.sleep(2.0)
        
        # Get statistics
        stats = batcher.get_stats()
        print(f"Batcher stats: {stats}")
        
    finally:
        batcher.stop()
