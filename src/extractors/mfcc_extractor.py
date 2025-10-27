"""
GPU-Optimized MFCC (Mel-Frequency Cepstral Coefficients) extractor for audio feature extraction.

GPU-accelerated replacement for MFCCExtractor using PyTorch and torchaudio.
This extractor implements:
- MFCC coefficients (13 coefficients)
- Delta MFCC coefficients (13 coefficients) 
- Statistical aggregation (mean, std)
"""

import torch
import torchaudio.transforms as T
import numpy as np
from typing import Dict, Any, Tuple
import logging
from src.core.base_extractor import BaseExtractor
from src.schemas.models import ExtractorResult
from src.core.gpu_audio_utils import get_gpu_audio_processor
from src.core.audio_utils import load_audio_mono, ensure_mono_tensor, validate_audio_shape

logger = logging.getLogger(__name__)


class MFCCExtractor(BaseExtractor):
    """GPU-optimized extractor for MFCC and delta MFCC features."""
    
    name = "mfcc_extractor"
    version = "3.0.0"
    description = "GPU-accelerated MFCC and delta MFCC feature extraction with statistical aggregation"
    category = "spectral"
    dependencies = ["torch", "torchaudio"]
    estimated_duration = 2.0  # Much faster than CPU version
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize GPU-optimized MFCC extractor.
        
        Args:
            device: Device to use for processing ("cuda" or "cpu")
        """
        super().__init__()
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        # MFCC parameters
        self.n_mfcc = 13  # Number of MFCC coefficients
        self.n_fft = 2048  # FFT window size
        self.hop_length = 512  # Hop length for STFT
        self.n_mels = 128  # Number of mel bands
        self.fmin = 0  # Minimum frequency
        self.fmax = None  # Maximum frequency (None = Nyquist)
        self.sample_rate = 22050
        
        # Delta parameters
        self.delta_width = 9  # Width for delta calculation
        
        # Initialize GPU audio processor
        self.gpu_processor = get_gpu_audio_processor(self.device, self.sample_rate)
        
        # Initialize MFCC transform
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
                "f_min": self.fmin,
                "f_max": self.fmax
            }
        ).to(self.device)
        
        self.logger.info(f"GPU-Optimized MFCC Extractor initialized on device: {self.device}")
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract MFCC features from audio file using GPU acceleration.
        
        Args:
            input_uri: Path to input audio file
            tmp_path: Path to temporary directory (unused for this extractor)
            
        Returns:
            ExtractorResult with MFCC features
        """
        self._log_extraction_start(input_uri)
        
        try:
            # Load audio file with proper mono conversion
            audio_tensor, sample_rate = load_audio_mono(input_uri, self.sample_rate)
            
            # Validate audio shape for MFCC (expects [1, 1, N])
            if not validate_audio_shape(audio_tensor, expected_channels=1):
                raise ValueError(f"Invalid audio shape for MFCC: {audio_tensor.shape}")
            
            # Move to device
            audio_tensor = audio_tensor.to(self.device)
            
            # Extract MFCC features using GPU
            mfcc_features = self._extract_mfcc_features_gpu(audio_tensor, sample_rate)
            
            # Extract delta MFCC features using GPU
            delta_features = self._extract_delta_features_gpu(mfcc_features)
            
            # Calculate statistical aggregations using GPU
            mfcc_stats = self._calculate_statistics_gpu(mfcc_features, "mfcc")
            delta_stats = self._calculate_statistics_gpu(delta_features, "mfcc_delta")
            
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
            error_msg = f"GPU-optimized MFCC extraction failed: {str(e)}"
            self._log_extraction_error(input_uri, error_msg, 0.0)
            
            return self._create_result(
                success=False,
                error=error_msg,
                processing_time=None
            )
    
    def _extract_mfcc_features_gpu(self, audio_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Extract MFCC features from audio using GPU acceleration.
        
        Args:
            audio_tensor: Audio tensor on GPU
            sample_rate: Sample rate of audio
            
        Returns:
            MFCC features tensor of shape (1, n_mfcc, n_frames)
        """
        try:
            # Extract MFCC features using GPU transform
            mfcc = self.mfcc_transform(audio_tensor)  # Shape: [1, 1, n_mfcc, n_frames]
            
            # Remove extra channel dimension for consistency
            mfcc = mfcc.squeeze(1)  # Shape: [1, n_mfcc, n_frames]
            
            self.logger.debug(f"Extracted MFCC features: {mfcc.shape}")
            return mfcc
            
        except Exception as e:
            self.logger.error(f"Failed to extract MFCC features: {str(e)}")
            raise
    
    def _extract_delta_features_gpu(self, mfcc: torch.Tensor) -> torch.Tensor:
        """
        Extract delta (first derivative) of MFCC features using GPU.
        
        Args:
            mfcc: MFCC features tensor
            
        Returns:
            Delta MFCC features tensor
        """
        try:
            # Calculate delta features using GPU
            delta_mfcc = self._compute_delta_gpu(mfcc, width=self.delta_width)
            
            self.logger.debug(f"Extracted delta MFCC features: {delta_mfcc.shape}")
            return delta_mfcc
            
        except Exception as e:
            self.logger.error(f"Failed to extract delta MFCC features: {str(e)}")
            raise
    
    def _compute_delta_gpu(self, features: torch.Tensor, width: int = 9) -> torch.Tensor:
        """
        Compute delta features using GPU operations.
        
        Args:
            features: Input features tensor
            width: Width for delta calculation
            
        Returns:
            Delta features tensor
        """
        # Pad features for delta computation
        pad_width = width // 2
        padded = torch.nn.functional.pad(features, (pad_width, pad_width), mode='replicate')
        
        # Compute delta using finite differences
        delta = torch.diff(padded, dim=1)
        
        # Apply smoothing window
        if width > 1:
            # Create smoothing kernel
            kernel = torch.ones(1, 1, width - 1, device=self.device) / (width - 1)
            
            # Apply convolution for smoothing
            delta = torch.nn.functional.conv1d(
                delta.unsqueeze(0), 
                kernel, 
                padding=width // 2
            ).squeeze(0)
        
        return delta
    
    def _calculate_statistics_gpu(self, features: torch.Tensor, prefix: str) -> Dict[str, Any]:
        """
        Calculate statistical aggregations for features using GPU.
        
        Args:
            features: Feature tensor of shape (n_features, n_frames)
            prefix: Prefix for feature names
            
        Returns:
            Dictionary with statistical features
        """
        try:
            # Calculate mean and std across time (axis=1) using GPU
            mean_features = torch.mean(features, dim=1)
            std_features = torch.std(features, dim=1)
            
            # Create feature dictionary
            stats = {}
            
            # Add individual coefficient statistics
            for i in range(len(mean_features)):
                stats[f"{prefix}_{i}_mean"] = float(mean_features[i].item())
                stats[f"{prefix}_{i}_std"] = float(std_features[i].item())
            
            # Add overall statistics as lists
            stats[f"{prefix}_mean"] = mean_features.cpu().numpy().tolist()
            stats[f"{prefix}_std"] = std_features.cpu().numpy().tolist()
            
            # Additional statistical features
            stats[f"{prefix}_min"] = torch.min(features, dim=1)[0].cpu().numpy().tolist()
            stats[f"{prefix}_max"] = torch.max(features, dim=1)[0].cpu().numpy().tolist()
            stats[f"{prefix}_median"] = torch.median(features, dim=1)[0].cpu().numpy().tolist()
            
            # Range and coefficient of variation
            feature_range = torch.max(features, dim=1)[0] - torch.min(features, dim=1)[0]
            feature_cv = std_features / (torch.abs(mean_features) + 1e-10)
            
            stats[f"{prefix}_range"] = feature_range.cpu().numpy().tolist()
            stats[f"{prefix}_cv"] = feature_cv.cpu().numpy().tolist()
            
            # Global statistics across all coefficients
            stats[f"{prefix}_global_mean"] = float(torch.mean(features).item())
            stats[f"{prefix}_global_std"] = float(torch.std(features).item())
            stats[f"{prefix}_global_min"] = float(torch.min(features).item())
            stats[f"{prefix}_global_max"] = float(torch.max(features).item())
            
            self.logger.debug(f"Calculated GPU statistics for {prefix}: {len(stats)} features")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to calculate GPU statistics for {prefix}: {str(e)}")
            raise
    
    def _extract_advanced_mfcc_features_gpu(self, audio_tensor: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
        """
        Extract advanced MFCC features using GPU.
        
        Args:
            audio_tensor: Audio tensor on GPU
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with advanced MFCC features
        """
        features = {}
        
        # Basic MFCC
        mfcc = self._extract_mfcc_features_gpu(audio_tensor, sample_rate)
        
        # Delta MFCC
        delta_mfcc = self._extract_delta_features_gpu(mfcc)
        
        # Delta-Delta MFCC (second derivative)
        delta_delta_mfcc = self._extract_delta_features_gpu(delta_mfcc)
        
        # Statistical features for each type
        mfcc_stats = self._calculate_statistics_gpu(mfcc, "mfcc")
        delta_stats = self._calculate_statistics_gpu(delta_mfcc, "mfcc_delta")
        delta_delta_stats = self._calculate_statistics_gpu(delta_delta_mfcc, "mfcc_delta_delta")
        
        features.update(mfcc_stats)
        features.update(delta_stats)
        features.update(delta_delta_stats)
        
        # Energy features
        energy = torch.sum(torch.abs(audio_tensor) ** 2)
        features["energy"] = float(energy.item())
        
        # Zero crossing rate
        zcr = self.gpu_processor.zero_crossing_rate(
            audio_tensor,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        zcr_stats = self.gpu_processor.compute_statistics(zcr, "zcr")
        features.update(zcr_stats)
        
        # Spectral centroid
        spectral_centroid = self.gpu_processor.spectral_centroid(
            audio_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        centroid_stats = self.gpu_processor.compute_statistics(spectral_centroid, "spectral_centroid")
        features.update(centroid_stats)
        
        return features
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get extractor parameters.
        
        Returns:
            Dictionary with extractor parameters
        """
        return {
            "device": self.device,
            "n_mfcc": self.n_mfcc,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "delta_width": self.delta_width,
            "sample_rate": self.sample_rate
        }


# For running as a module
if __name__ == "__main__":
    import sys
    import json
    import tempfile
    import os
    
    if len(sys.argv) != 2:
        print("Usage: python gpu_optimized_mfcc_extractor.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file {audio_file} not found")
        sys.exit(1)
    
    # Create extractor and run
    extractor = GPUOptimizedMFCCExtractor()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = extractor.run(audio_file, tmp_dir)
        
        # Print result as JSON
        print(json.dumps(result.dict(), indent=2))
