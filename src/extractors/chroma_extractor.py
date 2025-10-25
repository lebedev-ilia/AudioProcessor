"""
Chroma extractor for audio feature extraction.

This extractor implements:
- Chroma features (12 tonal classes)
- STFT-based chroma calculation
- Normalization and statistical aggregation
"""

import librosa
import numpy as np
import soundfile as sf
from typing import Dict, Any, Tuple
import logging
from src.core.base_extractor import BaseExtractor
from src.schemas.models import ExtractorResult

logger = logging.getLogger(__name__)


class ChromaExtractor(BaseExtractor):
    """Extractor for Chroma features (tonal/harmonic information)."""
    
    name = "chroma_extractor"
    version = "1.0.0"
    description = "Chroma feature extraction for tonal/harmonic analysis"
    category = "core"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 4.0
    
    def __init__(self):
        """Initialize Chroma extractor with default parameters."""
        super().__init__()
        
        # Chroma parameters
        self.n_chroma = 12  # Number of chroma bins (12 semitones)
        self.n_fft = 2048  # FFT window size
        self.hop_length = 512  # Hop length for STFT
        self.fmin = 32.7  # Minimum frequency (C1 = 32.7 Hz)
        self.norm = 2  # Normalization method (L2 norm)
        
        self.logger.info(f"Initialized {self.name} v{self.version}")
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract Chroma features from audio file.
        
        Args:
            input_uri: Path to input audio file
            tmp_path: Path to temporary directory (unused for this extractor)
            
        Returns:
            ExtractorResult with Chroma features
        """
        self._log_extraction_start(input_uri)
        
        try:
            # Load audio file
            audio, sample_rate = self._load_audio(input_uri)
            
            # Extract Chroma features
            chroma_features = self._extract_chroma_features(audio, sample_rate)
            
            # Calculate statistical aggregation
            chroma_stats = self._calculate_statistics(chroma_features)
            
            # Create successful result
            result = self._create_result(
                success=True,
                payload=chroma_stats,
                processing_time=None  # Will be set by base class timing
            )
            
            self._log_extraction_success(input_uri, 0.0)  # Time will be updated
            return result
            
        except Exception as e:
            error_msg = f"Chroma extraction failed: {str(e)}"
            self._log_extraction_error(input_uri, error_msg, 0.0)
            
            return self._create_result(
                success=False,
                error=error_msg,
                processing_time=None
            )
    
    def _load_audio(self, input_uri: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa.
        
        Args:
            input_uri: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio with librosa (automatically resamples to 22050 Hz)
            audio, sr = librosa.load(
                input_uri,
                sr=22050,  # Standard sample rate for audio analysis
                mono=True,  # Convert to mono
                res_type='kaiser_fast'  # Fast resampling
            )
            
            self.logger.debug(f"Loaded audio: {len(audio)} samples at {sr} Hz")
            return audio, sr
            
        except Exception as e:
            self.logger.error(f"Failed to load audio file {input_uri}: {str(e)}")
            raise
    
    def _extract_chroma_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract Chroma features from audio.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of audio
            
        Returns:
            Chroma features array of shape (n_chroma, n_frames)
        """
        try:
            # Extract Chroma features using STFT
            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=sample_rate,
                n_chroma=self.n_chroma,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                norm=self.norm
            )
            
            self.logger.debug(f"Extracted Chroma features: {chroma.shape}")
            return chroma
            
        except Exception as e:
            self.logger.error(f"Failed to extract Chroma features: {str(e)}")
            raise
    
    def _calculate_statistics(self, chroma_features: np.ndarray) -> Dict[str, Any]:
        """
        Calculate statistical aggregations for Chroma features.
        
        Args:
            chroma_features: Chroma features array of shape (n_chroma, n_frames)
            
        Returns:
            Dictionary with statistical features
        """
        try:
            # Calculate mean and std across time (axis=1)
            chroma_mean = np.mean(chroma_features, axis=1)
            chroma_std = np.std(chroma_features, axis=1)
            
            # Calculate min and max across time
            chroma_min = np.min(chroma_features, axis=1)
            chroma_max = np.max(chroma_features, axis=1)
            
            # Create feature dictionary
            features = {}
            
            # Add individual chroma features (chroma_0..11_mean)
            for i in range(self.n_chroma):
                features[f"chroma_{i}_mean"] = float(chroma_mean[i])
                features[f"chroma_{i}_std"] = float(chroma_std[i])
                features[f"chroma_{i}_min"] = float(chroma_min[i])
                features[f"chroma_{i}_max"] = float(chroma_max[i])
            
            # Add overall arrays for convenience
            features["chroma_mean"] = chroma_mean.tolist()
            features["chroma_std"] = chroma_std.tolist()
            features["chroma_min"] = chroma_min.tolist()
            features["chroma_max"] = chroma_max.tolist()
            
            # Add summary statistics
            features["chroma_mean_overall"] = float(np.mean(chroma_mean))
            features["chroma_std_overall"] = float(np.std(chroma_mean))
            features["chroma_range"] = float(np.max(chroma_max) - np.min(chroma_min))
            
            # Add tonal strength metrics
            features["chroma_tonal_strength"] = float(np.max(chroma_mean))
            features["chroma_tonal_centroid"] = float(np.sum(chroma_mean * np.arange(self.n_chroma)) / np.sum(chroma_mean))
            
            # Add key profile analysis (simplified)
            major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])  # C major
            minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])  # A minor
            
            # Calculate correlation with major and minor profiles
            major_correlation = np.corrcoef(chroma_mean, major_profile)[0, 1]
            minor_correlation = np.corrcoef(chroma_mean, minor_profile)[0, 1]
            
            features["chroma_major_correlation"] = float(major_correlation) if not np.isnan(major_correlation) else 0.0
            features["chroma_minor_correlation"] = float(minor_correlation) if not np.isnan(minor_correlation) else 0.0
            
            self.logger.debug(f"Calculated Chroma statistics: {len(features)} features")
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Chroma statistics: {str(e)}")
            raise
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get extractor parameters.
        
        Returns:
            Dictionary with extractor parameters
        """
        return {
            "n_chroma": self.n_chroma,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "fmin": self.fmin,
            "norm": self.norm
        }


# For running as a module
if __name__ == "__main__":
    import sys
    import json
    import tempfile
    import os
    
    if len(sys.argv) != 2:
        print("Usage: python chroma_extractor.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file {audio_file} not found")
        sys.exit(1)
    
    # Create extractor and run
    extractor = ChromaExtractor()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = extractor.run(audio_file, tmp_dir)
        
        # Print result as JSON
        print(json.dumps(result.dict(), indent=2))
