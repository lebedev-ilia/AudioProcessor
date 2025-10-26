"""
Video Audio Extractor for extracting audio from video files.
"""
import os
import tempfile
import subprocess
import librosa
import numpy as np
from typing import Dict, Any, Optional
from ..core.base_extractor import BaseExtractor
from ..utils.logging import get_logger

logger = get_logger(__name__)


class VideoAudioExtractor(BaseExtractor):
    """
    Video Audio Extractor for extracting audio from video files.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "video_audio_extractor"
        self.version = "1.0.0"
        self.description = "Extract audio from video files using ffmpeg"
        
    def run(self, video_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Extract audio from video file and return audio file path.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted audio
            
        Returns:
            Dictionary with extracted audio file path and metadata
        """
        try:
            logger.info(f"Starting video audio extraction from: {video_path}")
            
            # Validate video file
            if not self._validate_video_file(video_path):
                return {
                    "success": False,
                    "error": "Invalid video file",
                    "processing_time": 0.0
                }
            
            # Extract audio using ffmpeg
            audio_path, metadata = self._extract_audio_from_video(video_path, output_dir)
            
            if not audio_path:
                return {
                    "success": False,
                    "error": "Failed to extract audio from video",
                    "processing_time": 0.0
                }
            
            # Get audio info
            audio_info = self._get_audio_info(audio_path)
            
            result = {
                "success": True,
                "extracted_audio_path": audio_path,
                "video_metadata": metadata,
                "audio_info": audio_info,
                "processing_time": 0.0  # Placeholder for now
            }
            
            logger.info(f"Successfully extracted audio from video: {video_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting audio from video {video_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": 0.0
            }
    
    def _validate_video_file(self, video_path: str) -> bool:
        """
        Validate that the file is a valid video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if valid, False otherwise
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            return False
        
        # Check file extension
        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv'}
        file_ext = os.path.splitext(video_path)[1].lower()
        
        if file_ext not in valid_extensions:
            logger.error(f"Invalid video file extension: {file_ext}")
            return False
        
        # Check file size (must be > 0)
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            logger.error(f"Video file is empty: {video_path}")
            return False
        
        logger.debug(f"Video file validation passed: {video_path}")
        return True
    
    def _extract_audio_from_video(self, video_path: str, output_dir: str) -> tuple[Optional[str], Dict[str, Any]]:
        """
        Extract audio from video using ffmpeg.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted audio
            
        Returns:
            Tuple of (audio_path, metadata)
        """
        try:
            # Generate output audio filename
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            audio_path = os.path.join(output_dir, f"{video_name}_extracted_audio.wav")
            
            # Get video metadata first
            metadata = self._get_video_metadata(video_path)
            
            # Extract audio using ffmpeg
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '44100',  # Sample rate 44.1kHz
                '-ac', '2',  # Stereo
                '-y',  # Overwrite output file
                audio_path
            ]
            
            logger.info(f"Running ffmpeg command: {' '.join(command)}")
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode != 0:
                logger.error(f"ffmpeg failed: {result.stderr}")
                return None, {}
            
            if not os.path.exists(audio_path):
                logger.error(f"Audio file was not created: {audio_path}")
                return None, {}
            
            logger.info(f"Successfully extracted audio to: {audio_path}")
            return audio_path, metadata
            
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg command timed out")
            return None, {}
        except Exception as e:
            logger.error(f"Error running ffmpeg: {e}")
            return None, {}
    
    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """
        Get video metadata using ffprobe.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video metadata
        """
        try:
            command = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"ffprobe failed: {result.stderr}")
                return {}
            
            import json
            info = json.loads(result.stdout)
            
            # Extract relevant information
            metadata = {
                'duration': float(info['format'].get('duration', 0)),
                'size': int(info['format'].get('size', 0)),
                'bit_rate': int(info['format'].get('bit_rate', 0)),
                'format_name': info['format'].get('format_name', ''),
                'format_long_name': info['format'].get('format_long_name', '')
            }
            
            # Get video stream info
            video_streams = [s for s in info['streams'] if s['codec_type'] == 'video']
            if video_streams:
                stream = video_streams[0]
                metadata.update({
                    'video_codec': stream.get('codec_name', ''),
                    'video_width': int(stream.get('width', 0)),
                    'video_height': int(stream.get('height', 0)),
                    'video_fps': eval(stream.get('r_frame_rate', '0/1')) if stream.get('r_frame_rate') else 0
                })
            
            # Get audio stream info
            audio_streams = [s for s in info['streams'] if s['codec_type'] == 'audio']
            if audio_streams:
                stream = audio_streams[0]
                metadata.update({
                    'audio_codec': stream.get('codec_name', ''),
                    'audio_sample_rate': int(stream.get('sample_rate', 0)),
                    'audio_channels': int(stream.get('channels', 0))
                })
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get video metadata for {video_path}: {e}")
            return {}
    
    def _get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """
        Get basic information about the extracted audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with audio file information
        """
        try:
            # Load audio with librosa to get basic info
            audio, sr = librosa.load(audio_path, sr=None)
            
            return {
                'duration': len(audio) / sr,
                'sample_rate': sr,
                'channels': 1 if audio.ndim == 1 else audio.shape[0],
                'samples': len(audio),
                'file_size': os.path.getsize(audio_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get audio info for {audio_path}: {e}")
            return {}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python video_audio_extractor.py <video_file>")
        sys.exit(1)
    
    video_file = sys.argv[1]
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmp_dir:
        extractor = VideoAudioExtractor()
        result = extractor.run(video_file, tmp_dir)
        
        print(f"Extraction result: {result}")
        
        if result.get('success'):
            print(f"Extracted audio saved to: {result['extracted_audio_path']}")
            print(f"Video metadata: {result['video_metadata']}")
            print(f"Audio info: {result['audio_info']}")
        else:
            print(f"Extraction failed: {result.get('error')}")
