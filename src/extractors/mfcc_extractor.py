"""
MFCC (Mel-Frequency Cepstral Coefficients) extractor for audio feature extraction.

This extractor implements:
- MFCC coefficients (13 coefficients)
- Delta MFCC coefficients (13 coefficients) 
- Statistical aggregation (mean, std)
"""

import librosa
import numpy as np
import soundfile as sf
from typing import Dict, Any, Tuple
import logging
from src.core.base_extractor import BaseExtractor
from src.schemas.models import ExtractorResult

logger = logging.getLogger(__name__)


class MFCCExtractor(BaseExtractor):
    """Extractor for MFCC and delta MFCC features."""
    
    name = "mfcc_extractor"
    version = "0.1.0"
    description = "MFCC and delta MFCC feature extraction with statistical aggregation"
    
    def __init__(self):
        """Initialize MFCC extractor with default parameters."""
        super().__init__()
        
        # MFCC parameters
        self.n_mfcc = 13  # Number of MFCC coefficients
        self.n_fft = 2048  # FFT window size
        self.hop_length = 512  # Hop length for STFT
        self.n_mels = 128  # Number of mel bands
        self.fmin = 0  # Minimum frequency
        self.fmax = None  # Maximum frequency (None = Nyquist)
        
        # Delta parameters
        self.delta_width = 9  # Width for delta calculation
        
        self.logger.info(f"Initialized {self.name} v{self.version}")
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract MFCC features from audio file.
        
        Args:
            input_uri: Path to input audio file
            tmp_path: Path to temporary directory (unused for this extractor)
            
        Returns:
            ExtractorResult with MFCC features
        """
        self._log_extraction_start(input_uri)
        
        try:
            # Load audio file
            audio, sample_rate = self._load_audio(input_uri)
            
            # Extract MFCC features
            mfcc_features = self._extract_mfcc_features(audio, sample_rate)
            
            # Extract delta MFCC features
            delta_features = self._extract_delta_features(mfcc_features)
            
            # Calculate statistical aggregations
            mfcc_stats = self._calculate_statistics(mfcc_features, "mfcc")
            delta_stats = self._calculate_statistics(delta_features, "mfcc_delta")
            
            # Combine all features
            payload = {**mfcc_stats, **delta_stats}
            
            # Create successful result
            result = self._create_result(
                success=True,
                payload=payload,
                processing_time=None  # Will be set by base class timing
            )
            
            self._log_extraction_success(input_uri, 0.0)  # Time will be updated
            return result
            
        except Exception as e:
            error_msg = f"MFCC extraction failed: {str(e)}"
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
    
    def _extract_mfcc_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of audio
            
        Returns:
            MFCC features array of shape (n_mfcc, n_frames)
        """
        try:
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax
            )
            
            self.logger.debug(f"Extracted MFCC features: {mfcc.shape}")
            return mfcc
            
        except Exception as e:
            self.logger.error(f"Failed to extract MFCC features: {str(e)}")
            raise
    
    def _extract_delta_features(self, mfcc: np.ndarray) -> np.ndarray:
        """
        Extract delta (first derivative) of MFCC features.
        
        Args:
            mfcc: MFCC features array
            
        Returns:
            Delta MFCC features array
        """
        try:
            # Calculate delta features
            delta_mfcc = librosa.feature.delta(
                mfcc,
                width=self.delta_width
            )
            
            self.logger.debug(f"Extracted delta MFCC features: {delta_mfcc.shape}")
            return delta_mfcc
            
        except Exception as e:
            self.logger.error(f"Failed to extract delta MFCC features: {str(e)}")
            raise
    
    def _calculate_statistics(self, features: np.ndarray, prefix: str) -> Dict[str, Any]:
        """
        Calculate statistical aggregations for features.
        
        Args:
            features: Feature array of shape (n_features, n_frames)
            prefix: Prefix for feature names
            
        Returns:
            Dictionary with statistical features
        """
        try:
            # Calculate mean and std across time (axis=1)
            mean_features = np.mean(features, axis=1)
            std_features = np.std(features, axis=1)
            
            # Create feature dictionary
            stats = {}
            
            # Add mean features
            for i in range(len(mean_features)):
                stats[f"{prefix}_{i}_mean"] = float(mean_features[i])
            
            # Add std features
            for i in range(len(std_features)):
                stats[f"{prefix}_{i}_std"] = float(std_features[i])
            
            # Add overall statistics
            stats[f"{prefix}_mean"] = mean_features.tolist()
            stats[f"{prefix}_std"] = std_features.tolist()
            
            self.logger.debug(f"Calculated statistics for {prefix}: {len(stats)} features")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to calculate statistics for {prefix}: {str(e)}")
            raise
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get extractor parameters.
        
        Returns:
            Dictionary with extractor parameters
        """
        return {
            "n_mfcc": self.n_mfcc,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "delta_width": self.delta_width
        }


# For running as a module
if __name__ == "__main__":
    import sys
    import json
    import tempfile
    import os
    
    if len(sys.argv) != 2:
        print("Usage: python mfcc_extractor.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file {audio_file} not found")
        sys.exit(1)
    
    # Create extractor and run
    extractor = MFCCExtractor()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = extractor.run(audio_file, tmp_dir)
        
        # Print result as JSON
        print(json.dumps(result.dict(), indent=2))
