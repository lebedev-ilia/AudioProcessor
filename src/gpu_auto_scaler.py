"""
GPU Auto Scaler for dynamic resource management.

This module provides:
1. Automatic GPU resource scaling based on workload
2. Dynamic batch size adjustment
3. Memory usage monitoring and optimization
4. Performance-based scaling decisions
5. Load balancing across multiple GPUs
"""

import asyncio
import time
import psutil
import torch
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from .utils.logging import get_logger
from .gpu_optimizer import get_gpu_optimizer, GPUOptimizer
from .gpu_config import get_gpu_config_manager, GPUType, GPUConfig

logger = get_logger(__name__)


class ScalingTrigger(Enum):
    """Triggers for scaling operations."""
    MEMORY_HIGH = "memory_high"
    MEMORY_LOW = "memory_low"
    QUEUE_LONG = "queue_long"
    QUEUE_SHORT = "queue_short"
    PERFORMANCE_POOR = "performance_poor"
    PERFORMANCE_GOOD = "performance_good"
    WORKLOAD_HIGH = "workload_high"
    WORKLOAD_LOW = "workload_low"


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    gpu_utilization: float
    memory_utilization: float
    queue_length: int
    processing_time: float
    throughput: float
    error_rate: float
    timestamp: float


@dataclass
class ScalingAction:
    """Action to take for scaling."""
    action_type: str
    target_value: Any
    reason: str
    priority: int
    estimated_impact: str


class GPUAutoScaler:
    """Automatic GPU resource scaler."""
    
    def __init__(self, 
                 check_interval: float = 30.0,
                 scaling_threshold: float = 0.8,
                 memory_threshold: float = 0.85,
                 performance_threshold: float = 0.7):
        """
        Initialize GPU auto scaler.
        
        Args:
            check_interval: Interval between scaling checks (seconds)
            scaling_threshold: Threshold for triggering scaling actions
            memory_threshold: Memory utilization threshold
            performance_threshold: Performance threshold for scaling
        """
        self.check_interval = check_interval
        self.scaling_threshold = scaling_threshold
        self.memory_threshold = memory_threshold
        self.performance_threshold = performance_threshold
        
        # Components
        self.gpu_optimizer = get_gpu_optimizer()
        self.gpu_config_manager = get_gpu_config_manager()
        
        # State
        self.is_running = False
        self.scaling_task: Optional[asyncio.Task] = None
        self.metrics_history: List[ScalingMetrics] = []
        self.scaling_actions: List[ScalingAction] = []
        
        # Configuration
        self.current_config = self.gpu_config_manager.get_optimized_config()
        self.base_config = self.current_config
        
        # Statistics
        self.stats = {
            "scaling_events": 0,
            "successful_scalings": 0,
            "failed_scalings": 0,
            "total_uptime": 0.0,
            "average_response_time": 0.0
        }
        
        logger.info("GPU Auto Scaler initialized")
    
    async def start(self):
        """Start the auto scaler."""
        if self.is_running:
            logger.warning("Auto scaler is already running")
            return
        
        self.is_running = True
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        logger.info("GPU Auto Scaler started")
    
    async def stop(self):
        """Stop the auto scaler."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass
        
        logger.info("GPU Auto Scaler stopped")
    
    async def _scaling_loop(self):
        """Main scaling loop."""
        logger.info("GPU Auto Scaler loop started")
        
        while self.is_running:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics (last 100)
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]
                
                # Analyze metrics and determine scaling actions
                actions = await self._analyze_metrics(metrics)
                
                # Execute scaling actions
                if actions:
                    await self._execute_scaling_actions(actions)
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(self.check_interval)
        
        logger.info("GPU Auto Scaler loop stopped")
    
    async def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        try:
            # GPU metrics
            gpu_utilization = 0.0
            memory_utilization = 0.0
            
            if torch.cuda.is_available():
                # Get GPU utilization (simplified)
                gpu_utilization = self._get_gpu_utilization()
                
                # Get memory utilization
                memory_allocated = torch.cuda.memory_allocated(0)
                memory_total = torch.cuda.get_device_properties(0).total_memory
                memory_utilization = memory_allocated / memory_total
            
            # Queue metrics
            queue_length = self._get_queue_length()
            
            # Processing metrics
            processing_time = self._get_average_processing_time()
            throughput = self._get_throughput()
            error_rate = self._get_error_rate()
            
            return ScalingMetrics(
                gpu_utilization=gpu_utilization,
                memory_utilization=memory_utilization,
                queue_length=queue_length,
                processing_time=processing_time,
                throughput=throughput,
                error_rate=error_rate,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return ScalingMetrics(
                gpu_utilization=0.0,
                memory_utilization=0.0,
                queue_length=0,
                processing_time=0.0,
                throughput=0.0,
                error_rate=0.0,
                timestamp=time.time()
            )
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            # This is a simplified implementation
            # In production, you'd use nvidia-ml-py for accurate GPU utilization
            return 0.5  # Placeholder
        except:
            return 0.0
    
    def _get_queue_length(self) -> int:
        """Get current queue length."""
        try:
            stats = self.gpu_optimizer.get_all_stats()
            total_queue = 0
            for gpu_stats in stats.values():
                if "queue_stats" in gpu_stats:
                    total_queue += gpu_stats["queue_stats"].get("queue_size", 0)
            return total_queue
        except:
            return 0
    
    def _get_average_processing_time(self) -> float:
        """Get average processing time."""
        try:
            stats = self.gpu_optimizer.get_all_stats()
            total_time = 0.0
            count = 0
            for gpu_stats in stats.values():
                if "average_processing_time" in gpu_stats:
                    total_time += gpu_stats["average_processing_time"]
                    count += 1
            return total_time / count if count > 0 else 0.0
        except:
            return 0.0
    
    def _get_throughput(self) -> float:
        """Get current throughput (requests per second)."""
        try:
            # Calculate based on recent metrics
            if len(self.metrics_history) < 2:
                return 0.0
            
            recent_metrics = self.metrics_history[-5:]  # Last 5 measurements
            time_span = recent_metrics[-1].timestamp - recent_metrics[0].timestamp
            
            if time_span > 0:
                # Estimate throughput based on processing time and queue length
                avg_processing_time = sum(m.processing_time for m in recent_metrics) / len(recent_metrics)
                if avg_processing_time > 0:
                    return 1.0 / avg_processing_time
            
            return 0.0
        except:
            return 0.0
    
    def _get_error_rate(self) -> float:
        """Get current error rate."""
        try:
            # This would be calculated from actual error statistics
            # For now, return a placeholder
            return 0.0
        except:
            return 0.0
    
    async def _analyze_metrics(self, metrics: ScalingMetrics) -> List[ScalingAction]:
        """Analyze metrics and determine scaling actions."""
        actions = []
        
        # Memory-based scaling
        if metrics.memory_utilization > self.memory_threshold:
            actions.append(ScalingAction(
                action_type="reduce_batch_size",
                target_value=max(1, self.current_config.batch_size // 2),
                reason=f"High memory utilization: {metrics.memory_utilization:.2%}",
                priority=1,
                estimated_impact="Reduce memory usage, may increase latency"
            ))
        elif metrics.memory_utilization < 0.5 and self.current_config.batch_size < self.base_config.batch_size:
            actions.append(ScalingAction(
                action_type="increase_batch_size",
                target_value=min(self.base_config.batch_size, self.current_config.batch_size * 2),
                reason=f"Low memory utilization: {metrics.memory_utilization:.2%}",
                priority=2,
                estimated_impact="Increase throughput, may increase memory usage"
            ))
        
        # Queue-based scaling
        if metrics.queue_length > 50:  # High queue length
            actions.append(ScalingAction(
                action_type="increase_workers",
                target_value=min(4, self.current_config.max_gpu_workers + 1),
                reason=f"High queue length: {metrics.queue_length}",
                priority=1,
                estimated_impact="Process more requests in parallel"
            ))
        elif metrics.queue_length < 5 and self.current_config.max_gpu_workers > 1:
            actions.append(ScalingAction(
                action_type="decrease_workers",
                target_value=max(1, self.current_config.max_gpu_workers - 1),
                reason=f"Low queue length: {metrics.queue_length}",
                priority=3,
                estimated_impact="Reduce resource usage"
            ))
        
        # Performance-based scaling
        if metrics.processing_time > 10.0:  # High processing time
            actions.append(ScalingAction(
                action_type="enable_optimization",
                target_value=True,
                reason=f"High processing time: {metrics.processing_time:.2f}s",
                priority=1,
                estimated_impact="Improve processing speed"
            ))
        
        # Throughput-based scaling
        if metrics.throughput < 0.1:  # Low throughput
            actions.append(ScalingAction(
                action_type="increase_batch_size",
                target_value=min(self.base_config.batch_size, self.current_config.batch_size + 2),
                reason=f"Low throughput: {metrics.throughput:.2f} req/s",
                priority=2,
                estimated_impact="Increase throughput"
            ))
        
        return actions
    
    async def _execute_scaling_actions(self, actions: List[ScalingAction]):
        """Execute scaling actions."""
        for action in sorted(actions, key=lambda x: x.priority):
            try:
                await self._execute_scaling_action(action)
                self.stats["successful_scalings"] += 1
                logger.info(f"Executed scaling action: {action.action_type} = {action.target_value}")
            except Exception as e:
                self.stats["failed_scalings"] += 1
                logger.error(f"Failed to execute scaling action {action.action_type}: {e}")
        
        self.stats["scaling_events"] += len(actions)
    
    async def _execute_scaling_action(self, action: ScalingAction):
        """Execute a single scaling action."""
        if action.action_type == "reduce_batch_size":
            self.current_config.batch_size = action.target_value
            # Update GPU optimizer batch size
            for batcher in self.gpu_optimizer.batchers:
                batcher.batch_size = action.target_value
        
        elif action.action_type == "increase_batch_size":
            self.current_config.batch_size = action.target_value
            # Update GPU optimizer batch size
            for batcher in self.gpu_optimizer.batchers:
                batcher.batch_size = action.target_value
        
        elif action.action_type == "increase_workers":
            self.current_config.max_gpu_workers = action.target_value
            # Note: In a real implementation, you'd need to restart workers
        
        elif action.action_type == "decrease_workers":
            self.current_config.max_gpu_workers = action.target_value
            # Note: In a real implementation, you'd need to restart workers
        
        elif action.action_type == "enable_optimization":
            self.current_config.enable_optimization = action.target_value
            # Update GPU optimizer settings
        
        # Store the action
        self.scaling_actions.append(action)
        
        # Keep only recent actions (last 50)
        if len(self.scaling_actions) > 50:
            self.scaling_actions = self.scaling_actions[-50:]
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        return {
            **self.stats,
            "current_config": {
                "batch_size": self.current_config.batch_size,
                "max_gpu_workers": self.current_config.max_gpu_workers,
                "memory_limit": self.current_config.gpu_memory_limit,
                "mixed_precision": self.current_config.mixed_precision
            },
            "recent_actions": len(self.scaling_actions),
            "metrics_history_length": len(self.metrics_history)
        }
    
    def get_recent_metrics(self, count: int = 10) -> List[ScalingMetrics]:
        """Get recent metrics."""
        return self.metrics_history[-count:] if self.metrics_history else []
    
    def get_recent_actions(self, count: int = 10) -> List[ScalingAction]:
        """Get recent scaling actions."""
        return self.scaling_actions[-count:] if self.scaling_actions else []


# Global auto scaler instance
_auto_scaler: Optional[GPUAutoScaler] = None


def get_auto_scaler() -> GPUAutoScaler:
    """Get global auto scaler instance."""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = GPUAutoScaler()
    return _auto_scaler


async def start_auto_scaler():
    """Start the global auto scaler."""
    scaler = get_auto_scaler()
    await scaler.start()


async def stop_auto_scaler():
    """Stop the global auto scaler."""
    scaler = get_auto_scaler()
    await scaler.stop()


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_auto_scaler():
        # Test auto scaler
        scaler = GPUAutoScaler(check_interval=5.0)
        
        try:
            await scaler.start()
            
            # Run for 60 seconds
            await asyncio.sleep(60)
            
            # Get statistics
            stats = scaler.get_scaling_stats()
            print(f"Auto Scaler Stats: {stats}")
            
            # Get recent metrics
            metrics = scaler.get_recent_metrics(5)
            print(f"Recent Metrics: {len(metrics)}")
            
            # Get recent actions
            actions = scaler.get_recent_actions(5)
            print(f"Recent Actions: {len(actions)}")
            
        finally:
            await scaler.stop()
    
    # Run test
    asyncio.run(test_auto_scaler())
