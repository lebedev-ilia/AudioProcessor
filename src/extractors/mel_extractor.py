"""
Mel Spectrogram extractor for audio feature extraction with GPU fallback.

This extractor implements:
- Mel spectrogram with 64 mel bands
- Temporal aggregation (mean across time)
- Optimized mel filterbank parameters
- GPU acceleration with CPU fallback
"""

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from typing import Dict, Any, Tuple
import logging
from src.core.base_extractor import BaseExtractor
from src.schemas.models import ExtractorResult

logger = logging.getLogger(__name__)


class MelExtractor(BaseExtractor):
    """Extractor for Mel spectrogram features with GPU acceleration and CPU fallback."""
    
    name = "mel_extractor"
    version = "2.0.0"
    description = "Mel spectrogram feature extraction with GPU acceleration and CPU fallback"
    category = "core"
    dependencies = ["librosa", "numpy", "torch", "torchaudio"]
    estimated_duration = 2.0  # Faster with GPU
    
    def __init__(self, device: str = "auto"):
        """
        Initialize Mel extractor with GPU acceleration and CPU fallback.
        
        Args:
            device: Device to use for processing ("auto", "cuda", or "cpu")
        """
        super().__init__()
        
        # Device detection with fallback
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        # Mel spectrogram parameters
        self.n_mels = 64  # Number of mel bands (as specified in requirements)
        self.n_fft = 2048  # FFT window size
        self.hop_length = 512  # Hop length for STFT
        self.fmin = 0  # Minimum frequency
        self.fmax = None  # Maximum frequency (None = Nyquist)
        self.power = 2.0  # Power for mel spectrogram
        self.sample_rate = 22050
        
        # Initialize GPU transforms if available
        self.gpu_transform = None
        if self.device == "cuda":
            try:
                self.gpu_transform = T.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                    f_min=self.fmin,
                    f_max=self.fmax,
                    power=self.power
                ).to(self.device)
                self.logger.info("GPU Mel transform initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize GPU transform: {e}")
                self.device = "cpu"
                self.gpu_transform = None
        
        self.logger.info(f"Initialized {self.name} v{self.version} on {self.device}")
    
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
        Extract Mel spectrogram features from audio with GPU acceleration and CPU fallback.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of audio
            
        Returns:
            Mel spectrogram features array of shape (n_mels, n_frames)
        """
        try:
            if self.device == "cuda" and self.gpu_transform is not None:
                # Use GPU acceleration
                return self._extract_mel_features_gpu(audio, sample_rate)
            else:
                # Use CPU fallback
                return self._extract_mel_features_cpu(audio, sample_rate)
                
        except Exception as e:
            self.logger.error(f"Failed to extract Mel spectrogram: {str(e)}")
            raise
    
    def _extract_mel_features_gpu(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract Mel spectrogram using GPU acceleration."""
        try:
            # Convert to tensor and move to GPU
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                resampler = T.Resample(sample_rate, self.sample_rate).to(self.device)
                audio_tensor = resampler(audio_tensor)
            
            # Extract Mel spectrogram using GPU
            with torch.no_grad():
                mel_spec = self.gpu_transform(audio_tensor)
                
                # Convert to log scale (dB)
                mel_spec_db = torch.log10(mel_spec + 1e-10) * 10
                
                # Convert back to numpy
                mel_spec_db = mel_spec_db.squeeze().cpu().numpy()
            
            self.logger.debug(f"Extracted Mel spectrogram (GPU): {mel_spec_db.shape}")
            return mel_spec_db
            
        except Exception as e:
            self.logger.warning(f"GPU Mel extraction failed, falling back to CPU: {e}")
            return self._extract_mel_features_cpu(audio, sample_rate)
    
    def _extract_mel_features_cpu(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract Mel spectrogram using CPU fallback."""
        try:
            # Extract Mel spectrogram using librosa
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
            
            self.logger.debug(f"Extracted Mel spectrogram (CPU): {mel_spec_db.shape}")
            return mel_spec_db
            
        except Exception as e:
            self.logger.error(f"CPU Mel extraction failed: {str(e)}")
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
            
            # Add device information
            features["device_used"] = self.device
            features["gpu_accelerated"] = self.device == "cuda"
            
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
