"""
Tests for video processing functionality.
"""
import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from src.main import app
from src.schemas.models import ProcessRequest
from src.extractors.video_audio_extractor import VideoAudioExtractor
from src.core.utils import validate_video_file


class TestVideoAudioExtractor:
    """Test VideoAudioExtractor functionality."""
    
    def test_video_audio_extractor_init(self):
        """Test VideoAudioExtractor initialization."""
        extractor = VideoAudioExtractor()
        assert extractor.name == "video_audio_extractor"
        assert extractor.version == "1.0.0"
        assert extractor.description == "Extract audio from video files using ffmpeg"
    
    def test_validate_video_file_valid(self):
        """Test video file validation with valid file."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b"fake video content")
            tmp_file.flush()
            
            assert validate_video_file(tmp_file.name) == True
            
            os.unlink(tmp_file.name)
    
    def test_validate_video_file_invalid_extension(self):
        """Test video file validation with invalid extension."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(b"fake content")
            tmp_file.flush()
            
            assert validate_video_file(tmp_file.name) == False
            
            os.unlink(tmp_file.name)
    
    def test_validate_video_file_nonexistent(self):
        """Test video file validation with nonexistent file."""
        assert validate_video_file("/nonexistent/file.mp4") == False
    
    def test_validate_video_file_empty(self):
        """Test video file validation with empty file."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            # Empty file
            pass
            
            assert validate_video_file(tmp_file.name) == False
            
            os.unlink(tmp_file.name)
    
    @patch('subprocess.run')
    def test_extract_audio_from_video_success(self, mock_run):
        """Test successful audio extraction from video."""
        # Mock ffmpeg command success
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        extractor = VideoAudioExtractor()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a fake video file
            video_path = os.path.join(tmp_dir, "test_video.mp4")
            with open(video_path, 'w') as f:
                f.write("fake video content")
            
            # Mock ffprobe for metadata
            with patch.object(extractor, '_get_video_metadata') as mock_metadata:
                mock_metadata.return_value = {
                    'duration': 120.0,
                    'size': 1024000,
                    'format_name': 'mp4'
                }
                
                audio_path, metadata = extractor._extract_audio_from_video(video_path, tmp_dir)
                
                assert audio_path is not None
                assert os.path.exists(audio_path)
                assert metadata['duration'] == 120.0
    
    @patch('subprocess.run')
    def test_extract_audio_from_video_failure(self, mock_run):
        """Test audio extraction failure."""
        # Mock ffmpeg command failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "ffmpeg error"
        mock_run.return_value = mock_result
        
        extractor = VideoAudioExtractor()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = os.path.join(tmp_dir, "test_video.mp4")
            with open(video_path, 'w') as f:
                f.write("fake video content")
            
            audio_path, metadata = extractor._extract_audio_from_video(video_path, tmp_dir)
            
            assert audio_path is None
            assert metadata == {}
    
    @patch('subprocess.run')
    def test_get_video_metadata_success(self, mock_run):
        """Test successful video metadata extraction."""
        # Mock ffprobe success
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            'format': {
                'duration': '120.5',
                'size': '1024000',
                'bit_rate': '128000',
                'format_name': 'mp4',
                'format_long_name': 'QuickTime / MOV'
            },
            'streams': [
                {
                    'codec_type': 'video',
                    'codec_name': 'h264',
                    'width': 1920,
                    'height': 1080,
                    'r_frame_rate': '30/1'
                },
                {
                    'codec_type': 'audio',
                    'codec_name': 'aac',
                    'sample_rate': '44100',
                    'channels': 2
                }
            ]
        })
        mock_run.return_value = mock_result
        
        extractor = VideoAudioExtractor()
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b"fake video content")
            tmp_file.flush()
            
            metadata = extractor._get_video_metadata(tmp_file.name)
            
            assert metadata['duration'] == 120.5
            assert metadata['video_codec'] == 'h264'
            assert metadata['video_width'] == 1920
            assert metadata['video_height'] == 1080
            assert metadata['audio_codec'] == 'aac'
            assert metadata['audio_sample_rate'] == 44100
            
            os.unlink(tmp_file.name)


class TestVideoProcessingAPI:
    """Test video processing API endpoints."""
    
    def test_process_request_validation_audio_uri(self):
        """Test ProcessRequest validation with audio_uri."""
        request = ProcessRequest(
            video_id="test_video_123",
            audio_uri="s3://bucket/audio.wav"
        )
        assert request.audio_uri == "s3://bucket/audio.wav"
        assert request.video_uri is None
    
    def test_process_request_validation_video_uri(self):
        """Test ProcessRequest validation with video_uri."""
        request = ProcessRequest(
            video_id="test_video_123",
            video_uri="s3://bucket/video.mp4"
        )
        assert request.video_uri == "s3://bucket/video.mp4"
        assert request.audio_uri is None
    
    def test_process_request_validation_both_uris(self):
        """Test ProcessRequest validation with both URIs (should fail)."""
        with pytest.raises(ValueError, match="Cannot provide both audio_uri and video_uri"):
            ProcessRequest(
                video_id="test_video_123",
                audio_uri="s3://bucket/audio.wav",
                video_uri="s3://bucket/video.mp4"
            )
    
    def test_process_request_validation_no_uris(self):
        """Test ProcessRequest validation with no URIs (should fail)."""
        with pytest.raises(ValueError, match="Either audio_uri or video_uri must be provided"):
            ProcessRequest(
                video_id="test_video_123"
            )
    
    @patch('src.celery_app.celery_app.send_task')
    def test_process_video_endpoint(self, mock_send_task):
        """Test video processing endpoint."""
        mock_task = Mock()
        mock_task.id = "test_task_123"
        mock_send_task.return_value = mock_task
        
        client = TestClient(app)
        
        response = client.post("/process", json={
            "video_id": "test_video_123",
            "video_uri": "s3://bucket/video.mp4",
            "dataset": "test"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] == True
        assert data["celery_task_id"] == "test_task_123"
        assert "Video processing request accepted" in data["message"]
        
        # Verify the correct task was called
        mock_send_task.assert_called_once_with(
            'src.celery_app.process_video_task',
            args=["test_video_123", "s3://bucket/video.mp4"],
            kwargs={
                'task_id': None,
                'dataset': 'test',
                'meta': {}
            }
        )
    
    def test_process_video_endpoint_invalid_extension(self):
        """Test video processing endpoint with invalid file extension."""
        client = TestClient(app)
        
        response = client.post("/process", json={
            "video_id": "test_video_123",
            "video_uri": "s3://bucket/video.txt"
        })
        
        assert response.status_code == 400
        assert "Invalid video_uri" in response.json()["detail"]
    
    def test_process_video_endpoint_local_file(self):
        """Test video processing endpoint with local file."""
        with patch('src.celery_app.celery_app.send_task') as mock_send_task:
            mock_task = Mock()
            mock_task.id = "test_task_123"
            mock_send_task.return_value = mock_task
            
            client = TestClient(app)
            
            response = client.post("/process", json={
                "video_id": "test_video_123",
                "video_uri": "/local/path/video.mp4"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["accepted"] == True


class TestVideoProcessingIntegration:
    """Integration tests for video processing."""
    
    @patch('src.celery_app.process_video_task.delay')
    def test_video_processing_workflow(self, mock_delay):
        """Test complete video processing workflow."""
        mock_task = Mock()
        mock_task.id = "test_task_123"
        mock_delay.return_value = mock_task
        
        client = TestClient(app)
        
        # Submit video processing request
        response = client.post("/process", json={
            "video_id": "integration_test_video",
            "video_uri": "s3://test-bucket/test_video.mp4",
            "dataset": "integration_test",
            "meta": {
                "source": "test",
                "quality": "high"
            }
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] == True
        
        # Verify task was submitted
        mock_delay.assert_called_once_with(
            "integration_test_video",
            "s3://test-bucket/test_video.mp4",
            task_id=None,
            dataset="integration_test",
            meta={
                "source": "test",
                "quality": "high"
            }
        )


if __name__ == "__main__":
    pytest.main([__file__])
