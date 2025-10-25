"""
Test suite for Celery tasks.
"""
import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from celery import Celery
from celery.result import AsyncResult

# Import the tasks we want to test
from src.celery_app import (
    process_audio_task,
    process_audio_gpu_task,
    health_check_task,
    simple_test_task,
    notify_masterml
)


class TestCeleryTasks:
    """Test cases for Celery tasks."""
    
    @pytest.fixture
    def mock_celery_app(self):
        """Mock Celery app."""
        with patch('src.celery_app.celery_app') as mock_app:
            yield mock_app
    
    @pytest.fixture
    def mock_task_request(self):
        """Mock Celery task request."""
        request = Mock()
        request.id = "test-task-123"
        request.retries = 0
        return request
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create a sample audio file for testing."""
        import numpy as np
        import soundfile as sf
        
        # Generate a test audio signal (1 second of 440 Hz sine wave)
        sample_rate = 22050
        duration = 1.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Add some noise for more realistic testing
        noise = np.random.normal(0, 0.1, len(audio))
        audio = audio + noise
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            yield tmp.name
        
        # Cleanup
        os.unlink(tmp.name)
    
    @pytest.fixture
    def mock_extractors(self):
        """Mock extractors for testing."""
        with patch('src.extractors.discover_extractors') as mock_discover:
            # Create mock extractors
            mock_extractor1 = Mock()
            mock_extractor1.name = "mfcc_extractor"
            mock_extractor1.version = "1.0.0"
            mock_extractor1.run.return_value = Mock(
                success=True,
                payload={"mfcc_0_mean": 0.5, "mfcc_0_std": 0.1},
                error=None
            )
            
            mock_extractor2 = Mock()
            mock_extractor2.name = "mel_extractor"
            mock_extractor2.version = "1.0.0"
            mock_extractor2.run.return_value = Mock(
                success=True,
                payload={"mel64_mean_0": 0.3, "mel64_std_0": 0.05},
                error=None
            )
            
            mock_discover.return_value = [mock_extractor1, mock_extractor2]
            yield [mock_extractor1, mock_extractor2]
    
    @pytest.fixture
    def mock_s3_client(self):
        """Mock S3 client."""
        with patch('src.storage.s3_client.S3Client') as mock_s3_class:
            mock_s3_instance = Mock()
            mock_s3_class.return_value = mock_s3_instance
            mock_s3_instance.download_file.return_value = "/tmp/test_audio.wav"
            mock_s3_instance.upload_manifest.return_value = "s3://bucket/manifests/test_video.json"
            yield mock_s3_instance
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Mock metrics collector."""
        with patch('src.monitor.metrics.metrics_collector') as mock_metrics:
            yield mock_metrics
    
    @pytest.fixture
    def mock_loggers(self):
        """Mock loggers."""
        with patch('src.utils.logging.get_logger') as mock_get_logger, \
             patch('src.utils.logging.task_logger') as mock_task_logger, \
             patch('src.utils.logging.extractor_logger') as mock_extractor_logger:
            
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            yield {
                'logger': mock_logger,
                'task_logger': mock_task_logger,
                'extractor_logger': mock_extractor_logger
            }
    
    def test_simple_test_task(self):
        """Test simple test task."""
        result = simple_test_task("test message")
        assert result == "Processed: test message"
    
    def test_health_check_task_success(self):
        """Test health check task success."""
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value.isoformat.return_value = "2024-01-01T00:00:00"
            
            result = health_check_task()
            
            assert result["status"] == "healthy"
            assert "timestamp" in result
            assert result["celery_workers"] == 1
            assert result["queue_length"] == 0
    
    def test_health_check_task_failure(self):
        """Test health check task failure."""
        with patch('datetime.datetime') as mock_datetime, \
             patch('logging') as mock_logging:
            
            mock_datetime.utcnow.return_value.isoformat.return_value = "2024-01-01T00:00:00"
            mock_logging.getLogger.return_value.exception.side_effect = Exception("Test error")
            
            result = health_check_task()
            
            assert result["status"] == "unhealthy"
            assert "error" in result
            assert "timestamp" in result
    
    def test_process_audio_task_success(self, mock_task_request, sample_audio_file, 
                                       mock_extractors, mock_s3_client, mock_metrics_collector, mock_loggers):
        """Test successful audio processing task."""
        # Mock the task instance
        task_instance = Mock()
        task_instance.request = mock_task_request
        task_instance.update_state = Mock()
        
        # Mock settings
        with patch('src.config.get_settings') as mock_get_settings:
            mock_get_settings.return_value.s3_bucket = "test-bucket"
            
            # Mock tempfile
            with patch('tempfile.TemporaryDirectory') as mock_tempdir:
                mock_tempdir.return_value.__enter__.return_value = "/tmp"
                
                # Mock shutil for local file copying
                with patch('shutil.copy2') as mock_copy2:
                    # Mock datetime
                    with patch('datetime.datetime') as mock_datetime:
                        mock_datetime.utcnow.return_value.isoformat.return_value = "2024-01-01T00:00:00"
                        
                        # Mock notify_masterml
                        with patch('src.celery_app.notify_masterml') as mock_notify:
                            result = process_audio_task(
                                video_id="test_video",
                                audio_uri=sample_audio_file,
                                dataset="test"
                            )
                            
                            # Verify result
                            assert result["status"] == "completed"
                            assert result["video_id"] == "test_video"
                            assert result["audio_uri"] == sample_audio_file
                            assert "manifest_uri" in result
                            assert "extractors" in result
                            assert "successful_extractors" in result
                            assert "failed_extractors" in result
                            assert "total_processing_time" in result
                            assert "timestamp" in result
                            
                            # Verify task state updates
                            assert task_instance.update_state.call_count >= 5  # Multiple progress updates
                            
                            # Verify extractors were called
                            for extractor in mock_extractors:
                                extractor.run.assert_called_once()
                            
                            # Verify metrics were recorded
                            assert mock_metrics_collector.record_task.call_count >= 2  # Start and completion
                            assert mock_metrics_collector.record_extractor.call_count == len(mock_extractors)
                            
                            # Verify MasterML notification
                            mock_notify.assert_called_once()
    
    def test_process_audio_task_s3_download(self, mock_task_request, mock_extractors, 
                                           mock_s3_client, mock_metrics_collector, mock_loggers):
        """Test audio processing task with S3 download."""
        # Mock the task instance
        task_instance = Mock()
        task_instance.request = mock_task_request
        task_instance.update_state = Mock()
        
        # Mock settings
        with patch('src.config.get_settings') as mock_get_settings:
            mock_get_settings.return_value.s3_bucket = "test-bucket"
            
            # Mock tempfile
            with patch('tempfile.TemporaryDirectory') as mock_tempdir:
                mock_tempdir.return_value.__enter__.return_value = "/tmp"
                
                # Mock datetime
                with patch('datetime.datetime') as mock_datetime:
                    mock_datetime.utcnow.return_value.isoformat.return_value = "2024-01-01T00:00:00"
                    
                    # Mock notify_masterml
                    with patch('src.celery_app.notify_masterml') as mock_notify:
                        result = process_audio_task(
                            task_instance,
                            video_id="test_video",
                            audio_uri="s3://test-bucket/audio.wav",
                            dataset="test"
                        )
                        
                        # Verify S3 download was called
                        mock_s3_client.download_file.assert_called_once_with(
                            "s3://test-bucket/audio.wav", "/tmp"
                        )
                        
                        # Verify result
                        assert result["status"] == "completed"
    
    def test_process_audio_task_extractor_failure(self, mock_task_request, sample_audio_file, 
                                                 mock_s3_client, mock_metrics_collector, mock_loggers):
        """Test audio processing task with extractor failure."""
        # Mock the task instance
        task_instance = Mock()
        task_instance.request = mock_task_request
        task_instance.update_state = Mock()
        
        # Mock extractors with one failing
        with patch('src.extractors.discover_extractors') as mock_discover:
            mock_extractor1 = Mock()
            mock_extractor1.name = "mfcc_extractor"
            mock_extractor1.version = "1.0.0"
            mock_extractor1.run.return_value = Mock(
                success=True,
                payload={"mfcc_0_mean": 0.5},
                error=None
            )
            
            mock_extractor2 = Mock()
            mock_extractor2.name = "mel_extractor"
            mock_extractor2.version = "1.0.0"
            mock_extractor2.run.side_effect = Exception("Extractor failed")
            
            mock_discover.return_value = [mock_extractor1, mock_extractor2]
            
            # Mock settings
            with patch('src.config.get_settings') as mock_get_settings:
                mock_get_settings.return_value.s3_bucket = "test-bucket"
                
                # Mock tempfile
                with patch('tempfile.TemporaryDirectory') as mock_tempdir:
                    mock_tempdir.return_value.__enter__.return_value = "/tmp"
                    
                    # Mock shutil for local file copying
                    with patch('shutil.copy2') as mock_copy2:
                        # Mock datetime
                        with patch('datetime.datetime') as mock_datetime:
                            mock_datetime.utcnow.return_value.isoformat.return_value = "2024-01-01T00:00:00"
                            
                            # Mock notify_masterml
                            with patch('src.celery_app.notify_masterml') as mock_notify:
                        result = process_audio_task(
                            video_id="test_video",
                            audio_uri=sample_audio_file,
                                    dataset="test"
                                )
                                
                                # Verify result still completed but with failed extractor
                                assert result["status"] == "completed"
                                assert "mfcc_extractor" in result["successful_extractors"]
                                assert "mel_extractor" in result["failed_extractors"]
    
    def test_process_audio_task_s3_upload_fallback(self, mock_task_request, sample_audio_file, 
                                                  mock_extractors, mock_s3_client, mock_metrics_collector, mock_loggers):
        """Test audio processing task with S3 upload fallback to local."""
        # Mock the task instance
        task_instance = Mock()
        task_instance.request = mock_task_request
        task_instance.update_state = Mock()
        
        # Mock S3 upload failure
        mock_s3_client.upload_manifest.side_effect = Exception("S3 upload failed")
        
        # Mock settings
        with patch('src.config.get_settings') as mock_get_settings:
            mock_get_settings.return_value.s3_bucket = "test-bucket"
            
            # Mock tempfile
            with patch('tempfile.TemporaryDirectory') as mock_tempdir:
                mock_tempdir.return_value.__enter__.return_value = "/tmp"
                
                # Mock shutil for local file copying
                with patch('shutil.copy2') as mock_copy2:
                    # Mock datetime
                    with patch('datetime.datetime') as mock_datetime:
                        mock_datetime.utcnow.return_value.isoformat.return_value = "2024-01-01T00:00:00"
                        
                        # Mock notify_masterml
                        with patch('src.celery_app.notify_masterml') as mock_notify:
                            # Mock json.dump for local manifest saving
                            with patch('json.dump') as mock_json_dump:
                                with patch('builtins.open', mock_open()) as mock_file:
                        result = process_audio_task(
                            video_id="test_video",
                            audio_uri=sample_audio_file,
                                        dataset="test"
                                    )
                                    
                                    # Verify result
                                    assert result["status"] == "completed"
                                    assert result["manifest_uri"].startswith("file://")
                                    
                                    # Verify local manifest was saved
                                    mock_file.assert_called()
                                    mock_json_dump.assert_called()
    
    def test_process_audio_task_complete_failure(self, mock_task_request, mock_metrics_collector, mock_loggers):
        """Test audio processing task with complete failure."""
        # Mock the task instance
        task_instance = Mock()
        task_instance.request = mock_task_request
        task_instance.update_state = Mock()
        
        # Mock settings
        with patch('src.config.get_settings') as mock_get_settings:
            mock_get_settings.return_value.s3_bucket = "test-bucket"
            
            # Mock tempfile to raise exception
            with patch('tempfile.TemporaryDirectory', side_effect=Exception("Temp dir failed")):
                # Mock notify_masterml
                with patch('src.celery_app.notify_masterml') as mock_notify:
                    with pytest.raises(Exception, match="Temp dir failed"):
                        process_audio_task(
                            task_instance,
                            video_id="test_video",
                            audio_uri="s3://test-bucket/audio.wav",
                            dataset="test"
                        )
                    
                    # Verify failure metrics were recorded
                    mock_metrics_collector.record_task.assert_called_with("failed", "audio_queue", pytest.any(float))
                    
                    # Verify MasterML notification of failure
                    mock_notify.assert_called_once()
    
    def test_process_audio_gpu_task_success(self, mock_task_request):
        """Test successful GPU audio processing task."""
        # Mock the task instance
        task_instance = Mock()
        task_instance.request = mock_task_request
        task_instance.update_state = Mock()
        
        # Mock settings
        with patch('src.config.get_settings') as mock_get_settings:
            mock_get_settings.return_value.s3_bucket = "test-bucket"
            
            # Mock datetime
            with patch('datetime.datetime') as mock_datetime:
                mock_datetime.utcnow.return_value.isoformat.return_value = "2024-01-01T00:00:00"
                
                result = process_audio_gpu_task(
                    task_instance,
                    video_id="test_video",
                    audio_uri="s3://test-bucket/audio.wav"
                )
                
                # Verify result
                assert result["status"] == "completed"
                assert result["video_id"] == "test_video"
                assert result["audio_uri"] == "s3://test-bucket/audio.wav"
                assert result["gpu"] is True
                assert "manifest_uri" in result
                assert "extractors" in result
                assert "timestamp" in result
                
                # Verify task state updates
                assert task_instance.update_state.call_count >= 8  # Multiple progress updates
    
    def test_process_audio_gpu_task_failure(self, mock_task_request):
        """Test GPU audio processing task failure."""
        # Mock the task instance
        task_instance = Mock()
        task_instance.request = mock_task_request
        task_instance.update_state = Mock()
        
        # Mock settings
        with patch('src.config.get_settings') as mock_get_settings:
            mock_get_settings.return_value.s3_bucket = "test-bucket"
            
            # Mock datetime to raise exception
            with patch('celery_app.datetime', side_effect=Exception("GPU error")):
                with pytest.raises(Exception, match="GPU error"):
                    process_audio_gpu_task(
                        task_instance,
                        video_id="test_video",
                        audio_uri="s3://test-bucket/audio.wav"
                    )
                
                # Verify failure state was set
                task_instance.update_state.assert_called_with(
                    state="FAILURE", 
                    meta={"error": "GPU error", "gpu": True}
                )


class TestNotifyMasterML:
    """Test cases for MasterML notification."""
    
    def test_notify_masterml_success(self):
        """Test successful MasterML notification."""
        with patch('httpx.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            
            with patch('src.config.get_settings') as mock_get_settings:
                mock_get_settings.return_value.masterml_url = "http://masterml:8000"
                mock_get_settings.return_value.masterml_token = "test-token"
                
                with patch('datetime.datetime') as mock_datetime:
                    mock_datetime.utcnow.return_value.isoformat.return_value = "2024-01-01T00:00:00"
                    
                    notify_masterml(
                        video_id="test_video",
                        manifest_uri="s3://bucket/manifest.json",
                        status="completed",
                        result={"extractors": ["mfcc", "mel"]}
                    )
                    
                    # Verify HTTP request was made
                    mock_client.post.assert_called_once()
                    call_args = mock_client.post.call_args
                    assert call_args[0][0] == "http://localhost:8000/api/v1/processors/audio/notify"
                    assert call_args[1]["headers"]["Authorization"] == "Bearer test-token"
    
    def test_notify_masterml_failure(self):
        """Test MasterML notification failure."""
        with patch('httpx.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.post.side_effect = Exception("Network error")
            
            with patch('src.config.get_settings') as mock_get_settings:
                mock_get_settings.return_value.masterml_url = "http://masterml:8000"
                mock_get_settings.return_value.masterml_token = "test-token"
                
                # Should not raise exception
                notify_masterml(
                    video_id="test_video",
                    manifest_uri="s3://bucket/manifest.json",
                    status="failed",
                    result={"error": "Processing failed"}
                )
                
                # Verify HTTP request was attempted
                mock_client.post.assert_called_once()


def mock_open():
    """Mock open function for testing."""
    from unittest.mock import mock_open as original_mock_open
    return original_mock_open()


@pytest.mark.integration
class TestCeleryTasksIntegration:
    """Integration tests for Celery tasks (require real Celery broker)."""
    
    def test_celery_task_execution(self):
        """Test actual Celery task execution (requires broker)."""
        pytest.skip("Integration tests require real Celery broker")
    
    def test_celery_worker_health(self):
        """Test Celery worker health (requires broker)."""
        pytest.skip("Integration tests require real Celery broker")
