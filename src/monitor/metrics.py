"""
Prometheus metrics for AudioProcessor.
"""
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from typing import Dict, Any
import time

# Request metrics
audio_requests_total = Counter(
    'audio_requests_total',
    'Total audio processing requests',
    ['method', 'endpoint', 'status']
)

audio_request_duration = Histogram(
    'audio_request_duration_seconds',
    'Audio request processing duration',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
)

# Task metrics
audio_tasks_total = Counter(
    'audio_tasks_total',
    'Total audio processing tasks',
    ['status', 'queue']
)

audio_task_duration = Histogram(
    'audio_task_duration_seconds',
    'Audio task processing duration',
    ['extractor', 'queue'],
    buckets=[1, 5, 10, 30, 60, 300, 600, 1800, 3600]
)

audio_task_queue_length = Gauge(
    'audio_task_queue_length',
    'Length of audio processing queue',
    ['queue']
)

# Extractor metrics
audio_extractor_success_total = Counter(
    'audio_extractor_success_total',
    'Successful extractor runs',
    ['extractor_name', 'version']
)

audio_extractor_failures_total = Counter(
    'audio_extractor_failures_total',
    'Failed extractor runs',
    ['extractor_name', 'version', 'error_type']
)

audio_extractor_duration = Histogram(
    'audio_extractor_duration_seconds',
    'Extractor processing duration',
    ['extractor_name'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
)

# Feature extraction metrics
audio_features_extracted_total = Counter(
    'audio_features_extracted_total',
    'Total audio features extracted',
    ['feature_type', 'extractor_name']
)

audio_file_size_bytes = Histogram(
    'audio_file_size_bytes',
    'Size of processed audio files',
    buckets=[1024, 10240, 102400, 1048576, 10485760, 104857600, 1073741824]  # 1KB to 1GB
)

audio_duration_seconds = Histogram(
    'audio_duration_seconds',
    'Duration of processed audio files',
    buckets=[1, 5, 10, 30, 60, 300, 600, 1800, 3600, 7200]  # 1s to 2h
)

# System metrics
audio_processor_info = Info(
    'audio_processor_info',
    'AudioProcessor service information'
)

audio_processor_uptime_seconds = Gauge(
    'audio_processor_uptime_seconds',
    'AudioProcessor service uptime in seconds'
)

audio_processor_memory_usage_bytes = Gauge(
    'audio_processor_memory_usage_bytes',
    'AudioProcessor memory usage in bytes'
)

# S3 metrics
s3_operations_total = Counter(
    's3_operations_total',
    'Total S3 operations',
    ['operation', 'status']
)

s3_operation_duration = Histogram(
    's3_operation_duration_seconds',
    'S3 operation duration',
    ['operation'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

s3_data_transferred_bytes = Counter(
    's3_data_transferred_bytes',
    'Total data transferred to/from S3',
    ['direction']  # upload, download
)

# Model metrics
model_load_duration = Histogram(
    'model_load_duration_seconds',
    'Model loading duration',
    ['model_name'],
    buckets=[1, 5, 10, 30, 60, 300, 600]
)

model_inference_duration = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    ['model_name'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

# GPU metrics (if available)
gpu_utilization_percent = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

gpu_memory_usage_bytes = Gauge(
    'gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['gpu_id']
)

gpu_memory_total_bytes = Gauge(
    'gpu_memory_total_bytes',
    'Total GPU memory in bytes',
    ['gpu_id']
)


class MetricsCollector:
    """Metrics collector for AudioProcessor."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.start_time = time.time()
        self._update_info()
    
    def _update_info(self):
        """Update service information."""
        audio_processor_info.info({
            'version': '1.0.0',
            'service': 'audio_processor',
            'python_version': '3.10'
        })
    
    def update_uptime(self):
        """Update uptime metric."""
        uptime = time.time() - self.start_time
        audio_processor_uptime_seconds.set(uptime)
    
    def record_request(self, method: str, endpoint: str, status: str, duration: float):
        """Record request metrics."""
        audio_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()
        
        audio_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_task(self, status: str, queue: str, duration: float = None, extractor: str = None):
        """Record task metrics."""
        audio_tasks_total.labels(
            status=status,
            queue=queue
        ).inc()
        
        if duration is not None:
            audio_task_duration.labels(
                extractor=extractor or 'unknown',
                queue=queue
            ).observe(duration)
    
    def record_extractor(self, name: str, version: str, success: bool, duration: float, error_type: str = None):
        """Record extractor metrics."""
        if success:
            audio_extractor_success_total.labels(
                extractor_name=name,
                version=version
            ).inc()
        else:
            audio_extractor_failures_total.labels(
                extractor_name=name,
                version=version,
                error_type=error_type or 'unknown'
            ).inc()
        
        audio_extractor_duration.labels(
            extractor_name=name
        ).observe(duration)
    
    def record_features(self, feature_type: str, extractor_name: str, count: int = 1):
        """Record feature extraction metrics."""
        audio_features_extracted_total.labels(
            feature_type=feature_type,
            extractor_name=extractor_name
        ).inc(count)
    
    def record_audio_file(self, size_bytes: int, duration_seconds: float):
        """Record audio file metrics."""
        audio_file_size_bytes.observe(size_bytes)
        audio_duration_seconds.observe(duration_seconds)
    
    def record_s3_operation(self, operation: str, status: str, duration: float, bytes_transferred: int = 0):
        """Record S3 operation metrics."""
        s3_operations_total.labels(
            operation=operation,
            status=status
        ).inc()
        
        s3_operation_duration.labels(
            operation=operation
        ).observe(duration)
        
        if bytes_transferred > 0:
            direction = 'upload' if operation in ['put_object', 'upload_file'] else 'download'
            s3_data_transferred_bytes.labels(
                direction=direction
            ).inc(bytes_transferred)
    
    def record_model_operation(self, model_name: str, operation: str, duration: float):
        """Record model operation metrics."""
        if operation == 'load':
            model_load_duration.labels(
                model_name=model_name
            ).observe(duration)
        elif operation == 'inference':
            model_inference_duration.labels(
                model_name=model_name
            ).observe(duration)
    
    def record_gpu_metrics(self, gpu_id: str, utilization: float, memory_used: int, memory_total: int):
        """Record GPU metrics."""
        gpu_utilization_percent.labels(gpu_id=gpu_id).set(utilization)
        gpu_memory_usage_bytes.labels(gpu_id=gpu_id).set(memory_used)
        gpu_memory_total_bytes.labels(gpu_id=gpu_id).set(memory_total)
    
    def update_queue_length(self, queue: str, length: int):
        """Update queue length metric."""
        audio_task_queue_length.labels(queue=queue).set(length)
    
    def update_memory_usage(self, memory_bytes: int):
        """Update memory usage metric."""
        audio_processor_memory_usage_bytes.set(memory_bytes)


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics() -> str:
    """Get Prometheus metrics in text format."""
    metrics_collector.update_uptime()
    return generate_latest().decode('utf-8')
