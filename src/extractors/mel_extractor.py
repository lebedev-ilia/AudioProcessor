"""
Mel Spectrogram extractor for audio feature extraction.

This extractor implements:
- Mel spectrogram with 64 mel bands
- Temporal aggregation (mean across time)
- Optimized mel filterbank parameters
"""

import librosa
import numpy as np
import soundfile as sf
from typing import Dict, Any, Tuple
import logging
from src.core.base_extractor import BaseExtractor
from src.schemas.models import ExtractorResult

logger = logging.getLogger(__name__)


class MelExtractor(BaseExtractor):
    """Extractor for Mel spectrogram features."""
    
    name = "mel_extractor"
    version = "1.0.0"
    description = "Mel spectrogram feature extraction with 64 mel bands"
    category = "core"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 3.0
    
    def __init__(self):
        """Initialize Mel extractor with default parameters."""
        super().__init__()
        
        # Mel spectrogram parameters
        self.n_mels = 64  # Number of mel bands (as specified in requirements)
        self.n_fft = 2048  # FFT window size
        self.hop_length = 512  # Hop length for STFT
        self.fmin = 0  # Minimum frequency
        self.fmax = None  # Maximum frequency (None = Nyquist)
        self.power = 2.0  # Power for mel spectrogram
        
        self.logger.info(f"Initialized {self.name} v{self.version}")
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract Mel spectrogram features from audio file.
        
        Args:
            input_uri: Path to input audio file
            tmp_path: Path to temporary directory (unused for this extractor)
            
        Returns:
            ExtractorResult with Mel spectrogram features
        """
        self._log_extraction_start(input_uri)
        
        try:
            # Load audio file
            audio, sample_rate = self._load_audio(input_uri)
            
            # Extract Mel spectrogram features
            mel_features = self._extract_mel_features(audio, sample_rate)
            
            # Calculate temporal aggregation
            mel_stats = self._calculate_temporal_aggregation(mel_features)
            
            # Create successful result
            result = self._create_result(
                success=True,
                payload=mel_stats,
                processing_time=None  # Will be set by base class timing
            )
            
            self._log_extraction_success(input_uri, 0.0)  # Time will be updated
            return result
            
        except Exception as e:
            error_msg = f"Mel spectrogram extraction failed: {str(e)}"
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
    
    def _extract_mel_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract Mel spectrogram features from audio.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of audio
            
        Returns:
            Mel spectrogram features array of shape (n_mels, n_frames)
        """
        try:
            # Extract Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                fmin=self.fmin,
                fmax=self.fmax,
                power=self.power
            )
            
            # Convert to log scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            self.logger.debug(f"Extracted Mel spectrogram: {mel_spec_db.shape}")
            return mel_spec_db
            
        except Exception as e:
            self.logger.error(f"Failed to extract Mel spectrogram: {str(e)}")
            raise
    
    def _calculate_temporal_aggregation(self, mel_features: np.ndarray) -> Dict[str, Any]:
        """
        Calculate temporal aggregation for Mel spectrogram features.
        
        Args:
            mel_features: Mel spectrogram array of shape (n_mels, n_frames)
            
        Returns:
            Dictionary with aggregated features
        """
        try:
            # Calculate mean across time (axis=1)
            mel_mean = np.mean(mel_features, axis=1)
            
            # Calculate std across time
            mel_std = np.std(mel_features, axis=1)
            
            # Calculate min and max across time
            mel_min = np.min(mel_features, axis=1)
            mel_max = np.max(mel_features, axis=1)
            
            # Create feature dictionary
            features = {}
            
            # Add individual mel band features (mel64_mean[0..63])
            for i in range(self.n_mels):
                features[f"mel64_mean_{i}"] = float(mel_mean[i])
                features[f"mel64_std_{i}"] = float(mel_std[i])
                features[f"mel64_min_{i}"] = float(mel_min[i])
                features[f"mel64_max_{i}"] = float(mel_max[i])
            
            # Add overall arrays for convenience
            features["mel64_mean"] = mel_mean.tolist()
            features["mel64_std"] = mel_std.tolist()
            features["mel64_min"] = mel_min.tolist()
            features["mel64_max"] = mel_max.tolist()
            
            # Add summary statistics
            features["mel64_mean_overall"] = float(np.mean(mel_mean))
            features["mel64_std_overall"] = float(np.std(mel_mean))
            features["mel64_range"] = float(np.max(mel_max) - np.min(mel_min))
            
            self.logger.debug(f"Calculated temporal aggregation: {len(features)} features")
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to calculate temporal aggregation: {str(e)}")
            raise
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get extractor parameters.
        
        Returns:
            Dictionary with extractor parameters
        """
        return {
            "n_mels": self.n_mels,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "power": self.power
        }


# For running as a module
if __name__ == "__main__":
    import sys
    import json
    import tempfile
    import os
    
    if len(sys.argv) != 2:
        print("Usage: python mel_extractor.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file {audio_file} not found")
        sys.exit(1)
    
    # Create extractor and run
    extractor = MelExtractor()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = extractor.run(audio_file, tmp_dir)
        
        # Print result as JSON
        print(json.dumps(result.dict(), indent=2))
