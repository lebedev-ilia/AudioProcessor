"""
Mock fixtures for testing.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List
import tempfile
import os


class MockFixtures:
    """Mock fixtures for testing."""
    
    @staticmethod
    def create_mock_extractor(name: str, version: str = "1.0.0", 
                            success: bool = True, payload: Dict[str, Any] = None) -> Mock:
        """Create a mock extractor."""
        if payload is None:
            payload = {f"{name}_feature": 0.5}
        
        mock_extractor = Mock()
        mock_extractor.name = name
        mock_extractor.version = version
        
        mock_result = Mock()
        mock_result.success = success
        mock_result.payload = payload
        mock_result.error = None if success else "Mock error"
        
        mock_extractor.run.return_value = mock_result
        return mock_extractor
    
    @staticmethod
    def create_mock_s3_client() -> Mock:
        """Create a mock S3 client."""
        mock_s3 = Mock()
        mock_s3.download_file.return_value = "/tmp/test_audio.wav"
        mock_s3.upload_manifest.return_value = "s3://bucket/manifest.json"
        mock_s3.file_exists.return_value = True
        mock_s3.get_file_info.return_value = {
            "size": 1024,
            "last_modified": "2024-01-01T00:00:00Z",
            "content_type": "audio/wav"
        }
        mock_s3.list_files.return_value = ["s3://bucket/file1.wav", "s3://bucket/file2.wav"]
        mock_s3.delete_file.return_value = True
        return mock_s3
    
    @staticmethod
    def create_mock_celery_task() -> Mock:
        """Create a mock Celery task."""
        mock_task = Mock()
        mock_task.id = "test-task-123"
        mock_task.state = "PENDING"
        mock_task.result = None
        mock_task.info = {"progress": 0, "status": "Initializing..."}
        return mock_task
    
    @staticmethod
    def create_mock_metrics_collector() -> Mock:
        """Create a mock metrics collector."""
        mock_metrics = Mock()
        mock_metrics.record_task = Mock()
        mock_metrics.record_extractor = Mock()
        mock_metrics.record_api_request = Mock()
        return mock_metrics
    
    @staticmethod
    def create_mock_logger() -> Mock:
        """Create a mock logger."""
        mock_logger = Mock()
        mock_logger.info = Mock()
        mock_logger.error = Mock()
        mock_logger.warning = Mock()
        mock_logger.debug = Mock()
        return mock_logger


@pytest.fixture
def mock_extractors():
    """Mock all extractors."""
    extractors = [
        MockFixtures.create_mock_extractor("mfcc_extractor", payload={
            "mfcc_0_mean": 0.5, "mfcc_0_std": 0.1, "mfcc_mean": [0.5, 0.3, 0.2]
        }),
        MockFixtures.create_mock_extractor("mel_extractor", payload={
            "mel64_mean_0": 0.3, "mel64_std_0": 0.05, "mel64_mean": [0.3, 0.2, 0.1]
        }),
        MockFixtures.create_mock_extractor("chroma_extractor", payload={
            "chroma_0_mean": 0.4, "chroma_0_std": 0.08, "chroma_mean": [0.4, 0.3, 0.2]
        }),
        MockFixtures.create_mock_extractor("loudness_extractor", payload={
            "rms_mean": 0.2, "rms_std": 0.05, "loudness_lufs": -20.0
        }),
        MockFixtures.create_mock_extractor("vad_extractor", payload={
            "voiced_fraction": 0.8, "f0_mean": 200.0, "f0_std": 50.0
        }),
        MockFixtures.create_mock_extractor("clap_extractor", payload={
            "clap_0": 0.1, "clap_1": 0.2, "clap_embedding": [0.1, 0.2, 0.3]
        })
    ]
    return extractors


@pytest.fixture
def mock_failing_extractors():
    """Mock extractors with some failures."""
    extractors = [
        MockFixtures.create_mock_extractor("mfcc_extractor", success=True),
        MockFixtures.create_mock_extractor("mel_extractor", success=False),
        MockFixtures.create_mock_extractor("chroma_extractor", success=True),
        MockFixtures.create_mock_extractor("loudness_extractor", success=False),
        MockFixtures.create_mock_extractor("vad_extractor", success=True),
        MockFixtures.create_mock_extractor("clap_extractor", success=True)
    ]
    return extractors


@pytest.fixture
def mock_s3_client():
    """Mock S3 client."""
    return MockFixtures.create_mock_s3_client()


@pytest.fixture
def mock_celery_task():
    """Mock Celery task."""
    return MockFixtures.create_mock_celery_task()


@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector."""
    return MockFixtures.create_mock_metrics_collector()


@pytest.fixture
def mock_logger():
    """Mock logger."""
    return MockFixtures.create_mock_logger()


@pytest.fixture
def mock_settings():
    """Mock settings."""
    with patch('src.config.get_settings') as mock_get_settings:
        mock_settings = Mock()
        mock_settings.s3_endpoint = "http://localhost:9000"
        mock_settings.s3_access_key = "test_key"
        mock_settings.s3_secret_key = "test_secret"
        mock_settings.s3_region = "us-east-1"
        mock_settings.s3_bucket = "test-bucket"
        mock_settings.celery_broker_url = "redis://localhost:6379/0"
        mock_settings.celery_result_backend = "redis://localhost:6379/0"
        mock_settings.masterml_url = "http://masterml:8000"
        mock_settings.masterml_token = "test-token"
        mock_get_settings.return_value = mock_settings
        yield mock_settings


@pytest.fixture
def mock_temp_dir():
    """Mock temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def mock_manifest_data():
    """Mock manifest data."""
    return {
        "video_id": "test_video",
        "task_id": "test-task-123",
        "dataset": "test",
        "timestamp": "2024-01-01T00:00:00Z",
        "extractors": [
            {
                "name": "mfcc_extractor",
                "version": "1.0.0",
                "success": True,
                "payload": {"mfcc_0_mean": 0.5}
            },
            {
                "name": "mel_extractor",
                "version": "1.0.0",
                "success": True,
                "payload": {"mel64_mean_0": 0.3}
            }
        ],
        "schema_version": "audio_manifest_v1"
    }


@pytest.fixture
def mock_process_request():
    """Mock process request data."""
    return {
        "video_id": "test_video",
        "audio_uri": "s3://test-bucket/audio.wav",
        "dataset": "test",
        "priority": "normal",
        "gpu": False
    }


@pytest.fixture
def mock_process_response():
    """Mock process response data."""
    return {
        "accepted": True,
        "celery_task_id": "test-task-123",
        "estimated_completion_time": 60.0
    }


@pytest.fixture
def mock_task_status_pending():
    """Mock pending task status."""
    return {
        "task_id": "test-task-123",
        "status": "PENDING",
        "progress": 0,
        "status_message": "Initializing...",
        "result": None
    }


@pytest.fixture
def mock_task_status_progress():
    """Mock task in progress status."""
    return {
        "task_id": "test-task-123",
        "status": "PROGRESS",
        "progress": 50,
        "status_message": "Running extractors...",
        "current_extractor": "mfcc_extractor",
        "step": 3,
        "total_steps": 6,
        "result": None
    }


@pytest.fixture
def mock_task_status_success():
    """Mock successful task status."""
    return {
        "task_id": "test-task-123",
        "status": "SUCCESS",
        "progress": 100,
        "status_message": "Completed",
        "result": {
            "status": "completed",
            "video_id": "test_video",
            "manifest_uri": "s3://bucket/manifest.json",
            "extractors": ["mfcc_extractor", "mel_extractor"],
            "total_processing_time": 45.2
        }
    }


@pytest.fixture
def mock_task_status_failure():
    """Mock failed task status."""
    return {
        "task_id": "test-task-123",
        "status": "FAILURE",
        "progress": 0,
        "status_message": "Failed",
        "error": "Processing failed",
        "result": None
    }


@pytest.fixture
def mock_extractor_list():
    """Mock extractor list response."""
    return {
        "extractors": [
            {
                "name": "mfcc_extractor",
                "version": "1.0.0",
                "description": "MFCC feature extraction",
                "status": "available",
                "category": "core",
                "dependencies": ["librosa"],
                "estimated_duration": 5.0
            },
            {
                "name": "mel_extractor",
                "version": "1.0.0",
                "description": "Mel spectrogram extraction",
                "status": "available",
                "category": "core",
                "dependencies": ["librosa"],
                "estimated_duration": 3.0
            }
        ]
    }


@pytest.fixture
def mock_extractor_status():
    """Mock individual extractor status."""
    return {
        "name": "mfcc_extractor",
        "version": "1.0.0",
        "status": "available",
        "description": "MFCC feature extraction",
        "last_used": "2024-01-01T00:00:00Z",
        "success_rate": 0.95,
        "average_duration": 4.2
    }


@pytest.fixture
def mock_health_status():
    """Mock health status response."""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "1.0.0",
        "uptime": 3600.0,
        "memory_usage": 0.5,
        "cpu_usage": 0.2,
        "disk_usage": 0.3
    }


@pytest.fixture
def mock_metrics_response():
    """Mock metrics response."""
    return """# HELP audio_processor_tasks_total Total number of tasks processed
# TYPE audio_processor_tasks_total counter
audio_processor_tasks_total{status="completed"} 100
audio_processor_tasks_total{status="failed"} 5

# HELP audio_processor_extractors_total Total number of extractor runs
# TYPE audio_processor_extractors_total counter
audio_processor_extractors_total{name="mfcc_extractor",status="success"} 95
audio_processor_extractors_total{name="mfcc_extractor",status="error"} 5
"""


@pytest.fixture
def mock_boto_client():
    """Mock boto3 client."""
    with patch('boto3.client') as mock_client:
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        yield mock_client_instance


@pytest.fixture
def mock_boto_resource():
    """Mock boto3 resource."""
    with patch('boto3.resource') as mock_resource:
        mock_resource_instance = Mock()
        mock_resource.return_value = mock_resource_instance
        yield mock_resource_instance


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client."""
    with patch('httpx.Client') as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        yield mock_client


@pytest.fixture
def mock_celery_app():
    """Mock Celery app."""
    with patch('src.celery_app.celery_app') as mock_app:
        yield mock_app


@pytest.fixture
def mock_discover_extractors():
    """Mock discover_extractors function."""
    with patch('src.extractors.discover_extractors') as mock_discover:
        yield mock_discover


@pytest.fixture
def mock_s3_client_class():
    """Mock S3Client class."""
    with patch('src.storage.s3_client.S3Client') as mock_s3_class:
        yield mock_s3_class


@pytest.fixture
def mock_metrics_collector_class():
    """Mock metrics collector class."""
    with patch('src.monitor.metrics.metrics_collector') as mock_metrics:
        yield mock_metrics


@pytest.fixture
def mock_logger_class():
    """Mock logger class."""
    with patch('src.utils.logging.get_logger') as mock_get_logger:
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        yield mock_logger


@pytest.fixture
def mock_task_logger():
    """Mock task logger."""
    with patch('src.utils.logging.task_logger') as mock_task_logger:
        yield mock_task_logger


@pytest.fixture
def mock_extractor_logger():
    """Mock extractor logger."""
    with patch('src.utils.logging.extractor_logger') as mock_extractor_logger:
        yield mock_extractor_logger


@pytest.fixture
def mock_datetime():
    """Mock datetime."""
    with patch('datetime.datetime') as mock_dt:
        mock_dt.utcnow.return_value.isoformat.return_value = "2024-01-01T00:00:00Z"
        yield mock_dt


@pytest.fixture
def mock_time():
    """Mock time module."""
    with patch('time.time') as mock_time_func:
        mock_time_func.side_effect = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]  # Simulate time progression
        yield mock_time_func


@pytest.fixture
def mock_tempfile():
    """Mock tempfile module."""
    with patch('tempfile.TemporaryDirectory') as mock_tempdir:
        mock_tempdir.return_value.__enter__.return_value = "/tmp"
        yield mock_tempdir


@pytest.fixture
def mock_os():
    """Mock os module."""
    with patch('os.makedirs') as mock_makedirs, \
         patch('os.path.basename') as mock_basename, \
         patch('os.path.join') as mock_join, \
         patch('os.unlink') as mock_unlink:
        
        mock_basename.return_value = "test_file.wav"
        mock_join.return_value = "/tmp/test_file.wav"
        
        yield {
            'makedirs': mock_makedirs,
            'basename': mock_basename,
            'join': mock_join,
            'unlink': mock_unlink
        }


@pytest.fixture
def mock_json():
    """Mock json module."""
    with patch('json.dump') as mock_dump, \
         patch('json.load') as mock_load:
        
        yield {
            'dump': mock_dump,
            'load': mock_load
        }


@pytest.fixture
def mock_shutil():
    """Mock shutil module."""
    with patch('shutil.copy2') as mock_copy2:
        yield mock_copy2


@pytest.fixture
def mock_open():
    """Mock open function."""
    from unittest.mock import mock_open as original_mock_open
    return original_mock_open()


# Context managers for testing
@pytest.fixture
def mock_context_managers():
    """Mock context managers."""
    with patch('tempfile.TemporaryDirectory') as mock_tempdir, \
         patch('builtins.open', mock_open()) as mock_file:
        
        mock_tempdir.return_value.__enter__.return_value = "/tmp"
        
        yield {
            'tempdir': mock_tempdir,
            'file': mock_file
        }
