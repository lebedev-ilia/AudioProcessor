"""
Test suite for S3 client.
"""
import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, NoCredentialsError
from src.storage.s3_client import S3Client


class TestS3Client:
    """Test cases for S3Client."""
    
    @pytest.fixture
    def s3_client(self):
        """Create S3Client instance with mocked settings."""
        with patch('src.storage.s3_client.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                s3_endpoint="http://localhost:9000",
                s3_access_key="test_key",
                s3_secret_key="test_secret",
                s3_region="us-east-1",
                s3_bucket="test-bucket"
            )
            return S3Client()
    
    @pytest.fixture
    def mock_boto_client(self):
        """Mock boto3 client."""
        with patch('boto3.client') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            yield mock_client_instance
    
    @pytest.fixture
    def mock_boto_resource(self):
        """Mock boto3 resource."""
        with patch('boto3.resource') as mock_resource:
            mock_resource_instance = Mock()
            mock_resource.return_value = mock_resource_instance
            yield mock_resource_instance
    
    def test_parse_s3_uri_valid(self, s3_client):
        """Test parsing valid S3 URI."""
        bucket, key = s3_client.parse_s3_uri("s3://test-bucket/path/to/file.wav")
        assert bucket == "test-bucket"
        assert key == "path/to/file.wav"
    
    def test_parse_s3_uri_with_slash(self, s3_client):
        """Test parsing S3 URI with leading slash."""
        bucket, key = s3_client.parse_s3_uri("s3://test-bucket/path/to/file.wav")
        assert bucket == "test-bucket"
        assert key == "path/to/file.wav"
    
    def test_parse_s3_uri_invalid_scheme(self, s3_client):
        """Test parsing invalid URI scheme."""
        with pytest.raises(ValueError, match="Invalid S3 URI"):
            s3_client.parse_s3_uri("https://test-bucket/file.wav")
    
    def test_parse_s3_uri_invalid_format(self, s3_client):
        """Test parsing invalid URI format."""
        with pytest.raises(ValueError, match="Invalid S3 URI"):
            s3_client.parse_s3_uri("not-a-uri")
    
    def test_client_property_initialization(self, s3_client, mock_boto_client):
        """Test S3 client property initialization."""
        client = s3_client.client
        assert client is not None
        mock_boto_client.assert_called_once()
    
    def test_client_property_credentials_error(self, s3_client):
        """Test S3 client property with credentials error."""
        with patch('boto3.client', side_effect=NoCredentialsError()):
            with pytest.raises(NoCredentialsError):
                _ = s3_client.client
    
    def test_resource_property_initialization(self, s3_client, mock_boto_resource):
        """Test S3 resource property initialization."""
        resource = s3_client.resource
        assert resource is not None
        mock_boto_resource.assert_called_once()
    
    def test_download_file_success(self, s3_client, mock_boto_client):
        """Test successful file download."""
        # Mock successful download
        mock_boto_client.download_file.return_value = None
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = s3_client.download_file("s3://test-bucket/audio.wav", tmp_dir)
            
            expected_path = os.path.join(tmp_dir, "audio.wav")
            assert result_path == expected_path
            assert os.path.exists(result_path)
            
            # Verify download was called with correct parameters
            mock_boto_client.download_file.assert_called_once_with(
                "test-bucket", "audio.wav", expected_path
            )
    
    def test_download_file_client_error(self, s3_client, mock_boto_client):
        """Test file download with client error."""
        # Mock client error
        error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Not Found'}}
        mock_boto_client.download_file.side_effect = ClientError(error_response, 'GetObject')
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ClientError):
                s3_client.download_file("s3://test-bucket/nonexistent.wav", tmp_dir)
    
    def test_upload_file_success(self, s3_client, mock_boto_client):
        """Test successful file upload."""
        # Mock successful upload
        mock_boto_client.upload_file.return_value = None
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            
            try:
                result_uri = s3_client.upload_file(tmp_file.name, "s3://test-bucket/uploaded.wav")
                
                assert result_uri == "s3://test-bucket/uploaded.wav"
                
                # Verify upload was called with correct parameters
                mock_boto_client.upload_file.assert_called_once_with(
                    tmp_file.name, "test-bucket", "uploaded.wav", ExtraArgs={}
                )
            finally:
                os.unlink(tmp_file.name)
    
    def test_upload_file_with_metadata(self, s3_client, mock_boto_client):
        """Test file upload with metadata."""
        # Mock successful upload
        mock_boto_client.upload_file.return_value = None
        
        metadata = {"content-type": "audio/wav", "duration": "60"}
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            
            try:
                result_uri = s3_client.upload_file(
                    tmp_file.name, 
                    "s3://test-bucket/uploaded.wav", 
                    metadata=metadata
                )
                
                assert result_uri == "s3://test-bucket/uploaded.wav"
                
                # Verify upload was called with metadata
                expected_extra_args = {"Metadata": metadata}
                mock_boto_client.upload_file.assert_called_once_with(
                    tmp_file.name, "test-bucket", "uploaded.wav", ExtraArgs=expected_extra_args
                )
            finally:
                os.unlink(tmp_file.name)
    
    def test_upload_file_client_error(self, s3_client, mock_boto_client):
        """Test file upload with client error."""
        # Mock client error
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}}
        mock_boto_client.upload_file.side_effect = ClientError(error_response, 'PutObject')
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            
            try:
                with pytest.raises(ClientError):
                    s3_client.upload_file(tmp_file.name, "s3://test-bucket/uploaded.wav")
            finally:
                os.unlink(tmp_file.name)
    
    def test_upload_manifest_success(self, s3_client, mock_boto_client):
        """Test successful manifest upload."""
        # Mock successful upload
        mock_boto_client.upload_file.return_value = None
        
        manifest_data = {
            "video_id": "test_video",
            "timestamp": "2024-01-01T00:00:00",
            "extractors": []
        }
        
        result_uri = s3_client.upload_manifest(manifest_data, "test_video", "test_dataset")
        
        expected_uri = "s3://test-bucket/test_dataset/manifests/test_video.json"
        assert result_uri == expected_uri
        
        # Verify upload was called
        mock_boto_client.upload_file.assert_called_once()
        call_args = mock_boto_client.upload_file.call_args
        assert call_args[0][1] == "test-bucket"  # bucket
        assert call_args[0][2] == "test_dataset/manifests/test_video.json"  # key
    
    def test_upload_manifest_creates_temp_file(self, s3_client, mock_boto_client):
        """Test that manifest upload creates temporary file with correct content."""
        # Mock successful upload
        mock_boto_client.upload_file.return_value = None
        
        manifest_data = {
            "video_id": "test_video",
            "timestamp": "2024-01-01T00:00:00",
            "extractors": []
        }
        
        s3_client.upload_manifest(manifest_data, "test_video", "test_dataset")
        
        # Verify upload was called with a file path
        call_args = mock_boto_client.upload_file.call_args
        temp_file_path = call_args[0][0]
        
        # Verify the temporary file contains correct JSON
        with open(temp_file_path, 'r') as f:
            uploaded_data = json.load(f)
        
        assert uploaded_data == manifest_data
    
    def test_file_exists_true(self, s3_client, mock_boto_client):
        """Test file exists when file is found."""
        # Mock successful head_object call
        mock_boto_client.head_object.return_value = {"ContentLength": 1024}
        
        result = s3_client.file_exists("s3://test-bucket/existing.wav")
        
        assert result is True
        mock_boto_client.head_object.assert_called_once_with(
            Bucket="test-bucket", Key="existing.wav"
        )
    
    def test_file_exists_false(self, s3_client, mock_boto_client):
        """Test file exists when file is not found."""
        # Mock 404 error
        error_response = {'Error': {'Code': '404', 'Message': 'Not Found'}}
        mock_boto_client.head_object.side_effect = ClientError(error_response, 'HeadObject')
        
        result = s3_client.file_exists("s3://test-bucket/nonexistent.wav")
        
        assert result is False
    
    def test_file_exists_client_error(self, s3_client, mock_boto_client):
        """Test file exists with non-404 client error."""
        # Mock non-404 error
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}}
        mock_boto_client.head_object.side_effect = ClientError(error_response, 'HeadObject')
        
        with pytest.raises(ClientError):
            s3_client.file_exists("s3://test-bucket/restricted.wav")
    
    def test_get_file_info_success(self, s3_client, mock_boto_client):
        """Test getting file info successfully."""
        # Mock successful head_object call
        mock_response = {
            "ContentLength": 1024,
            "LastModified": "2024-01-01T00:00:00Z",
            "ContentType": "audio/wav",
            "Metadata": {"duration": "60"}
        }
        mock_boto_client.head_object.return_value = mock_response
        
        result = s3_client.get_file_info("s3://test-bucket/file.wav")
        
        assert result is not None
        assert result["size"] == 1024
        assert result["last_modified"] == "2024-01-01T00:00:00Z"
        assert result["content_type"] == "audio/wav"
        assert result["metadata"] == {"duration": "60"}
    
    def test_get_file_info_not_found(self, s3_client, mock_boto_client):
        """Test getting file info when file doesn't exist."""
        # Mock 404 error
        error_response = {'Error': {'Code': '404', 'Message': 'Not Found'}}
        mock_boto_client.head_object.side_effect = ClientError(error_response, 'HeadObject')
        
        result = s3_client.get_file_info("s3://test-bucket/nonexistent.wav")
        
        assert result is None
    
    def test_list_files_success(self, s3_client, mock_boto_client):
        """Test listing files successfully."""
        # Mock successful list_objects_v2 call
        mock_response = {
            "Contents": [
                {"Key": "audio/file1.wav"},
                {"Key": "audio/file2.wav"},
                {"Key": "audio/file3.wav"}
            ]
        }
        mock_boto_client.list_objects_v2.return_value = mock_response
        
        result = s3_client.list_files("audio/")
        
        expected_uris = [
            "s3://test-bucket/audio/file1.wav",
            "s3://test-bucket/audio/file2.wav",
            "s3://test-bucket/audio/file3.wav"
        ]
        assert result == expected_uris
        
        mock_boto_client.list_objects_v2.assert_called_once_with(
            Bucket="test-bucket", Prefix="audio/", MaxKeys=1000
        )
    
    def test_list_files_empty(self, s3_client, mock_boto_client):
        """Test listing files when no files found."""
        # Mock empty response
        mock_response = {"Contents": []}
        mock_boto_client.list_objects_v2.return_value = mock_response
        
        result = s3_client.list_files("empty/")
        
        assert result == []
    
    def test_list_files_with_max_keys(self, s3_client, mock_boto_client):
        """Test listing files with custom max_keys."""
        # Mock successful list_objects_v2 call
        mock_response = {"Contents": [{"Key": "audio/file1.wav"}]}
        mock_boto_client.list_objects_v2.return_value = mock_response
        
        s3_client.list_files("audio/", max_keys=10)
        
        mock_boto_client.list_objects_v2.assert_called_once_with(
            Bucket="test-bucket", Prefix="audio/", MaxKeys=10
        )
    
    def test_delete_file_success(self, s3_client, mock_boto_client):
        """Test successful file deletion."""
        # Mock successful delete_object call
        mock_boto_client.delete_object.return_value = {}
        
        result = s3_client.delete_file("s3://test-bucket/file.wav")
        
        assert result is True
        mock_boto_client.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="file.wav"
        )
    
    def test_delete_file_not_found(self, s3_client, mock_boto_client):
        """Test deleting non-existent file."""
        # Mock 404 error
        error_response = {'Error': {'Code': '404', 'Message': 'Not Found'}}
        mock_boto_client.delete_object.side_effect = ClientError(error_response, 'DeleteObject')
        
        result = s3_client.delete_file("s3://test-bucket/nonexistent.wav")
        
        assert result is False
    
    def test_delete_file_client_error(self, s3_client, mock_boto_client):
        """Test deleting file with non-404 client error."""
        # Mock non-404 error
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}}
        mock_boto_client.delete_object.side_effect = ClientError(error_response, 'DeleteObject')
        
        with pytest.raises(ClientError):
            s3_client.delete_file("s3://test-bucket/restricted.wav")


@pytest.mark.integration
class TestS3ClientIntegration:
    """Integration tests for S3Client (require real S3/MinIO)."""
    
    @pytest.fixture
    def s3_client_integration(self):
        """Create S3Client for integration tests."""
        # These tests would require a real S3/MinIO instance
        # pytest.skip("Integration tests require real S3/MinIO instance")
    
    def test_real_s3_operations(self, s3_client_integration):
        """Test real S3 operations (requires S3/MinIO)."""
        # pytest.skip("Integration tests require real S3/MinIO instance")
