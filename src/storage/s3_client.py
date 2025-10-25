"""
S3 client for AudioProcessor.
"""
import boto3
import os
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse
from botocore.exceptions import ClientError, NoCredentialsError
from ..config import get_settings

logger = logging.getLogger(__name__)


class S3Client:
    """S3 client for file operations."""
    
    def __init__(self):
        """Initialize S3 client."""
        self.settings = get_settings()
        self._client = None
        self._resource = None
    
    @property
    def client(self):
        """Get S3 client."""
        if self._client is None:
            try:
                self._client = boto3.client(
                    's3',
                    endpoint_url=str(self.settings.s3_endpoint),
                    aws_access_key_id=self.settings.s3_access_key,
                    aws_secret_access_key=self.settings.s3_secret_key,
                    region_name=self.settings.s3_region
                )
            except NoCredentialsError:
                logger.error("AWS credentials not found")
                raise
        return self._client
    
    @property
    def resource(self):
        """Get S3 resource."""
        if self._resource is None:
            try:
                self._resource = boto3.resource(
                    's3',
                    endpoint_url=self.settings.s3_endpoint,
                    aws_access_key_id=self.settings.s3_access_key,
                    aws_secret_access_key=self.settings.s3_secret_key,
                    region_name=self.settings.s3_region
                )
            except NoCredentialsError:
                logger.error("AWS credentials not found")
                raise
        return self._resource
    
    def parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """
        Parse S3 URI into bucket and key.
        
        Args:
            s3_uri: S3 URI (s3://bucket/key)
            
        Returns:
            Tuple of (bucket, key)
        """
        parsed = urlparse(s3_uri)
        if parsed.scheme != 's3':
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        
        return bucket, key
    
    def download_file(self, s3_uri: str, local_path: str) -> str:
        """
        Download file from S3 to local path.
        
        Args:
            s3_uri: S3 URI to download
            local_path: Local directory to save file
            
        Returns:
            Path to downloaded file
        """
        try:
            bucket, key = self.parse_s3_uri(s3_uri)
            
            # Create local directory if it doesn't exist
            os.makedirs(local_path, exist_ok=True)
            
            # Generate local filename
            filename = os.path.basename(key)
            local_file_path = os.path.join(local_path, filename)
            
            logger.info(f"Downloading {s3_uri} to {local_file_path}")
            
            # Download file
            self.client.download_file(bucket, key, local_file_path)
            
            logger.info(f"Successfully downloaded {s3_uri}")
            return local_file_path
            
        except ClientError as e:
            logger.error(f"Error downloading {s3_uri}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading {s3_uri}: {e}")
            raise
    
    def upload_file(self, local_path: str, s3_uri: str, metadata: Optional[Dict[str, str]] = None) -> str:
        """
        Upload file from local path to S3.
        
        Args:
            local_path: Local file path
            s3_uri: S3 URI to upload to
            metadata: Optional metadata to attach
            
        Returns:
            S3 URI of uploaded file
        """
        try:
            bucket, key = self.parse_s3_uri(s3_uri)
            
            logger.info(f"Uploading {local_path} to {s3_uri}")
            
            # Upload file
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            self.client.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
            
            logger.info(f"Successfully uploaded {local_path}")
            return s3_uri
            
        except ClientError as e:
            logger.error(f"Error uploading {local_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading {local_path}: {e}")
            raise
    
    def upload_manifest(self, manifest_data: Dict[str, Any], video_id: str, dataset: str = "default") -> str:
        """
        Upload manifest to S3.
        
        Args:
            manifest_data: Manifest data to upload
            video_id: Video identifier
            dataset: Dataset name
            
        Returns:
            S3 URI of uploaded manifest
        """
        try:
            import json
            import tempfile
            
            # Create temporary file for manifest
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(manifest_data, tmp_file, indent=2, default=str)
                tmp_file_path = tmp_file.name
            
            # Generate S3 key
            s3_key = f"{dataset}/manifests/{video_id}.json"
            s3_uri = f"s3://{self.settings.s3_bucket}/{s3_key}"
            
            # Upload manifest
            self.upload_file(tmp_file_path, s3_uri)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            logger.info(f"Successfully uploaded manifest for {video_id}")
            return s3_uri
            
        except Exception as e:
            logger.error(f"Error uploading manifest for {video_id}: {e}")
            raise
    
    def file_exists(self, s3_uri: str) -> bool:
        """
        Check if file exists in S3.
        
        Args:
            s3_uri: S3 URI to check
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            bucket, key = self.parse_s3_uri(s3_uri)
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise
    
    def get_file_info(self, s3_uri: str) -> Optional[Dict[str, Any]]:
        """
        Get file information from S3.
        
        Args:
            s3_uri: S3 URI to get info for
            
        Returns:
            File information dictionary or None if not found
        """
        try:
            bucket, key = self.parse_s3_uri(s3_uri)
            response = self.client.head_object(Bucket=bucket, Key=key)
            
            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'content_type': response.get('ContentType', ''),
                'metadata': response.get('Metadata', {})
            }
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return None
            raise
    
    def list_files(self, prefix: str, max_keys: int = 1000) -> list[str]:
        """
        List files in S3 with given prefix.
        
        Args:
            prefix: S3 key prefix
            max_keys: Maximum number of keys to return
            
        Returns:
            List of S3 URIs
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=self.settings.s3_bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            s3_uris = []
            for obj in response.get('Contents', []):
                s3_uri = f"s3://{self.settings.s3_bucket}/{obj['Key']}"
                s3_uris.append(s3_uri)
            
            return s3_uris
            
        except ClientError as e:
            logger.error(f"Error listing files with prefix {prefix}: {e}")
            raise
    
    def delete_file(self, s3_uri: str) -> bool:
        """
        Delete file from S3.
        
        Args:
            s3_uri: S3 URI to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            bucket, key = self.parse_s3_uri(s3_uri)
            self.client.delete_object(Bucket=bucket, Key=key)
            
            logger.info(f"Successfully deleted {s3_uri}")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            logger.error(f"Error deleting {s3_uri}: {e}")
            raise
