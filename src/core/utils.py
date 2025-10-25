"""
Utility functions for AudioProcessor.
"""
import os
import tempfile
import subprocess
import logging
from typing import Optional, Dict, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)


def create_temp_directory(prefix: str = "audio_processor") -> str:
    """
    Create a temporary directory for processing.
    
    Args:
        prefix: Prefix for the temporary directory name
        
    Returns:
        Path to the created temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    logger.debug(f"Created temporary directory: {temp_dir}")
    return temp_dir


def cleanup_temp_directory(temp_path: str):
    """
    Clean up temporary directory and its contents.
    
    Args:
        temp_path: Path to the temporary directory
    """
    try:
        import shutil
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
            logger.debug(f"Cleaned up temporary directory: {temp_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary directory {temp_path}: {e}")


def run_subprocess(
    command: list,
    timeout: int = 300,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None
) -> subprocess.CompletedProcess:
    """
    Run a subprocess command with timeout and error handling.
    
    Args:
        command: Command to run as a list
        timeout: Timeout in seconds
        cwd: Working directory
        env: Environment variables
        
    Returns:
        CompletedProcess object
        
    Raises:
        subprocess.TimeoutExpired: If command times out
        subprocess.CalledProcessError: If command fails
    """
    logger.debug(f"Running command: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            timeout=timeout,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(f"Command completed successfully: {result.returncode}")
        return result
    except subprocess.TimeoutExpired as e:
        logger.error(f"Command timed out after {timeout}s: {' '.join(command)}")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}: {e.stderr}")
        raise


def validate_audio_file(file_path: str) -> bool:
    """
    Validate that the file is a valid audio file.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False
    
    # Check file extension
    valid_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma', '.aiff', '.au'}
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext not in valid_extensions:
        logger.error(f"Invalid audio file extension: {file_ext}")
        return False
    
    # Check file size (must be > 0)
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        logger.error(f"Audio file is empty: {file_path}")
        return False
    
    logger.debug(f"Audio file validation passed: {file_path}")
    return True


def get_audio_info(file_path: str) -> Dict[str, Any]:
    """
    Get basic information about an audio file using ffprobe.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Dictionary with audio file information
    """
    try:
        command = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            file_path
        ]
        
        result = run_subprocess(command, timeout=30)
        info = json.loads(result.stdout)
        
        # Extract relevant information
        audio_info = {
            'duration': float(info['format'].get('duration', 0)),
            'size': int(info['format'].get('size', 0)),
            'bit_rate': int(info['format'].get('bit_rate', 0)),
            'format_name': info['format'].get('format_name', ''),
            'format_long_name': info['format'].get('format_long_name', '')
        }
        
        # Get audio stream info
        audio_streams = [s for s in info['streams'] if s['codec_type'] == 'audio']
        if audio_streams:
            stream = audio_streams[0]
            audio_info.update({
                'sample_rate': int(stream.get('sample_rate', 0)),
                'channels': int(stream.get('channels', 0)),
                'codec_name': stream.get('codec_name', ''),
                'codec_long_name': stream.get('codec_long_name', '')
            })
        
        return audio_info
        
    except Exception as e:
        logger.error(f"Failed to get audio info for {file_path}: {e}")
        return {}


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def format_file_size(bytes_size: int) -> str:
    """
    Format file size in bytes to human-readable string.
    
    Args:
        bytes_size: File size in bytes
        
    Returns:
        Formatted file size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def get_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        ISO formatted timestamp string
    """
    return datetime.utcnow().isoformat() + "Z"


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    import re
    # Remove or replace invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores and dots
    safe_name = safe_name.strip('_.')
    return safe_name


def ensure_directory_exists(directory_path: str):
    """
    Ensure that a directory exists, create it if it doesn't.
    
    Args:
        directory_path: Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        logger.debug(f"Created directory: {directory_path}")


def get_file_extension(file_path: str) -> str:
    """
    Get file extension in lowercase.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (including dot) in lowercase
    """
    return os.path.splitext(file_path)[1].lower()
