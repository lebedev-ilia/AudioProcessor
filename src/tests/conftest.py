"""
Pytest configuration and shared fixtures.
"""
import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope="session")
def test_audio_dir():
    """Create a temporary directory for test audio files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Clean up temporary files after each test."""
    temp_files = []
    
    def track_temp_file(file_path):
        temp_files.append(file_path)
        return file_path
    
    yield track_temp_file
    
    # Cleanup
    for file_path in temp_files:
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except OSError:
                pass  # Ignore cleanup errors


@pytest.fixture
def mock_environment():
    """Mock environment variables for testing."""
    import os
    from unittest.mock import patch
    
    env_vars = {
        'S3_ENDPOINT': 'http://localhost:9000',
        'S3_ACCESS_KEY': 'test_key',
        'S3_SECRET_KEY': 'test_secret',
        'S3_REGION': 'us-east-1',
        'S3_BUCKET': 'test-bucket',
        'CELERY_BROKER_URL': 'redis://localhost:6379/0',
        'CELERY_RESULT_BACKEND': 'redis://localhost:6379/0',
        'MASTERML_URL': 'http://masterml:8000',
        'MASTERML_TOKEN': 'test-token'
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def mock_logging():
    """Mock logging for tests."""
    import logging
    from unittest.mock import patch
    
    with patch('logging.getLogger') as mock_get_logger:
        mock_logger = patch('logging.Logger')
        mock_logger.info = patch('logging.Logger.info')
        mock_logger.error = patch('logging.Logger.error')
        mock_logger.warning = patch('logging.Logger.warning')
        mock_logger.debug = patch('logging.Logger.debug')
        
        mock_get_logger.return_value = mock_logger
        
        yield mock_logger


@pytest.fixture
def mock_psutil():
    """Mock psutil for tests."""
    from unittest.mock import patch
    
    with patch('psutil.Process') as mock_process:
        mock_process_instance = patch('psutil.Process')
        mock_process_instance.memory_info.return_value = patch('psutil.Process.memory_info')
        mock_process_instance.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        mock_process_instance.cpu_percent.return_value = 50.0
        
        mock_process.return_value = mock_process_instance
        
        yield mock_process_instance


@pytest.fixture
def mock_soundfile():
    """Mock soundfile for tests."""
    from unittest.mock import patch
    
    with patch('soundfile.read') as mock_read, \
         patch('soundfile.write') as mock_write, \
         patch('soundfile.info') as mock_info:
        
        # Mock read
        mock_read.return_value = (patch('numpy.array'), 22050)
        
        # Mock write
        mock_write.return_value = None
        
        # Mock info
        mock_info.return_value = patch('soundfile.SoundFile')
        mock_info.return_value.samplerate = 22050
        mock_info.return_value.frames = 22050
        mock_info.return_value.channels = 1
        
        yield {
            'read': mock_read,
            'write': mock_write,
            'info': mock_info
        }


@pytest.fixture
def mock_librosa():
    """Mock librosa for tests."""
    from unittest.mock import patch
    
    with patch('librosa.load') as mock_load, \
         patch('librosa.feature.mfcc') as mock_mfcc, \
         patch('librosa.feature.melspectrogram') as mock_mel, \
         patch('librosa.feature.chroma') as mock_chroma, \
         patch('librosa.feature.rms') as mock_rms, \
         patch('librosa.feature.zero_crossing_rate') as mock_zcr, \
         patch('librosa.piptrack') as mock_piptrack:
        
        # Mock load
        mock_load.return_value = (patch('numpy.array'), 22050)
        
        # Mock features
        mock_mfcc.return_value = patch('numpy.array')
        mock_mel.return_value = patch('numpy.array')
        mock_chroma.return_value = patch('numpy.array')
        mock_rms.return_value = patch('numpy.array')
        mock_zcr.return_value = patch('numpy.array')
        mock_piptrack.return_value = (patch('numpy.array'), patch('numpy.array'))
        
        yield {
            'load': mock_load,
            'mfcc': mock_mfcc,
            'mel': mock_mel,
            'chroma': mock_chroma,
            'rms': mock_rms,
            'zcr': mock_zcr,
            'piptrack': mock_piptrack
        }


@pytest.fixture
def mock_openl3():
    """Mock openl3 for tests."""
    from unittest.mock import patch
    
    with patch('openl3.get_audio_embedding') as mock_get_embedding:
        mock_get_embedding.return_value = (patch('numpy.array'), patch('numpy.array'))
        
        yield mock_get_embedding


@pytest.fixture
def mock_numpy():
    """Mock numpy for tests."""
    from unittest.mock import patch
    
    with patch('numpy.array') as mock_array, \
         patch('numpy.mean') as mock_mean, \
         patch('numpy.std') as mock_std, \
         patch('numpy.min') as mock_min, \
         patch('numpy.max') as mock_max, \
         patch('numpy.concatenate') as mock_concatenate:
        
        # Mock array
        mock_array.return_value = patch('numpy.array')
        
        # Mock statistics
        mock_mean.return_value = 0.5
        mock_std.return_value = 0.1
        mock_min.return_value = 0.0
        mock_max.return_value = 1.0
        mock_concatenate.return_value = patch('numpy.array')
        
        yield {
            'array': mock_array,
            'mean': mock_mean,
            'std': mock_std,
            'min': mock_min,
            'max': mock_max,
            'concatenate': mock_concatenate
        }


@pytest.fixture
def mock_boto3():
    """Mock boto3 for tests."""
    from unittest.mock import patch
    
    with patch('boto3.client') as mock_client, \
         patch('boto3.resource') as mock_resource:
        
        # Mock client
        mock_client_instance = patch('boto3.client')
        mock_client_instance.download_file.return_value = None
        mock_client_instance.upload_file.return_value = None
        mock_client_instance.head_object.return_value = {'ContentLength': 1024}
        mock_client_instance.list_objects_v2.return_value = {'Contents': []}
        mock_client_instance.delete_object.return_value = {}
        
        mock_client.return_value = mock_client_instance
        
        # Mock resource
        mock_resource_instance = patch('boto3.resource')
        mock_resource.return_value = mock_resource_instance
        
        yield {
            'client': mock_client_instance,
            'resource': mock_resource_instance
        }


@pytest.fixture
def mock_celery():
    """Mock Celery for tests."""
    from unittest.mock import patch
    
    with patch('celery.Celery') as mock_celery_class, \
         patch('celery.result.AsyncResult') as mock_async_result:
        
        # Mock Celery app
        mock_celery_app = patch('celery.Celery')
        mock_celery_app.conf = patch('celery.Celery.conf')
        mock_celery_app.task = patch('celery.Celery.task')
        
        mock_celery_class.return_value = mock_celery_app
        
        # Mock AsyncResult
        mock_result = patch('celery.result.AsyncResult')
        mock_result.state = 'PENDING'
        mock_result.result = None
        mock_result.info = {'progress': 0, 'status': 'Initializing...'}
        
        mock_async_result.return_value = mock_result
        
        yield {
            'app': mock_celery_app,
            'result': mock_result
        }


@pytest.fixture
def mock_httpx():
    """Mock httpx for tests."""
    from unittest.mock import patch
    
    with patch('httpx.Client') as mock_client_class:
        mock_client = patch('httpx.Client')
        mock_response = patch('httpx.Response')
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        yield mock_client


@pytest.fixture
def mock_fastapi():
    """Mock FastAPI for tests."""
    from unittest.mock import patch
    
    with patch('fastapi.FastAPI') as mock_fastapi_class, \
         patch('fastapi.testclient.TestClient') as mock_testclient_class:
        
        # Mock FastAPI app
        mock_app = patch('fastapi.FastAPI')
        mock_app.get = patch('fastapi.FastAPI.get')
        mock_app.post = patch('fastapi.FastAPI.post')
        
        mock_fastapi_class.return_value = mock_app
        
        # Mock TestClient
        mock_client = patch('fastapi.testclient.TestClient')
        mock_client.get.return_value = patch('fastapi.testclient.Response')
        mock_client.post.return_value = patch('fastapi.testclient.Response')
        
        mock_testclient_class.return_value = mock_client
        
        yield {
            'app': mock_app,
            'client': mock_client
        }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "s3: mark test as requiring S3 access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test names
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        elif "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        elif "gpu" in item.nodeid:
            item.add_marker(pytest.mark.gpu)
        elif "s3" in item.nodeid:
            item.add_marker(pytest.mark.s3)
        else:
            item.add_marker(pytest.mark.unit)


def pytest_runtest_setup(item):
    """Setup for each test."""
    # Skip integration tests if not requested
    if item.get_closest_marker("integration"):
        if not item.config.getoption("--integration"):
            pytest.skip("Integration tests not requested (use --integration)")
    
    # Skip performance tests if not requested
    if item.get_closest_marker("performance"):
        if not item.config.getoption("--performance"):
            pytest.skip("Performance tests not requested (use --performance)")
    
    # Skip GPU tests if not requested
    if item.get_closest_marker("gpu"):
        if not item.config.getoption("--gpu"):
            pytest.skip("GPU tests not requested (use --gpu)")
    
    # Skip S3 tests if not requested
    if item.get_closest_marker("s3"):
        if not item.config.getoption("--s3"):
            pytest.skip("S3 tests not requested (use --s3)")


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--integration", action="store_true", default=False,
        help="run integration tests"
    )
    parser.addoption(
        "--performance", action="store_true", default=False,
        help="run performance tests"
    )
    parser.addoption(
        "--gpu", action="store_true", default=False,
        help="run GPU tests"
    )
    parser.addoption(
        "--s3", action="store_true", default=False,
        help="run S3 tests"
    )
    parser.addoption(
        "--slow", action="store_true", default=False,
        help="run slow tests"
    )


# Test data fixtures
@pytest.fixture
def sample_audio_data():
    """Sample audio data for testing."""
    import numpy as np
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5])


@pytest.fixture
def sample_audio_features():
    """Sample audio features for testing."""
    return {
        'mfcc_0_mean': 0.5,
        'mfcc_0_std': 0.1,
        'mel64_mean_0': 0.3,
        'mel64_std_0': 0.05,
        'chroma_0_mean': 0.4,
        'chroma_0_std': 0.08,
        'rms_mean': 0.2,
        'rms_std': 0.05,
        'loudness_lufs': -20.0,
        'voiced_fraction': 0.8,
        'f0_mean': 200.0,
        'clap_0': 0.1,
        'clap_1': 0.2
    }


@pytest.fixture
def sample_manifest_data():
    """Sample manifest data for testing."""
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
            }
        ],
        "schema_version": "audio_manifest_v1"
    }


@pytest.fixture
def sample_process_request():
    """Sample process request data for testing."""
    return {
        "video_id": "test_video",
        "audio_uri": "s3://test-bucket/audio.wav",
        "dataset": "test",
        "priority": "normal",
        "gpu": False
    }
