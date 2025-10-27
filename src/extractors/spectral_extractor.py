"""
GPU-Optimized Spectral Extractor for spectral characteristics
GPU-accelerated replacement for SpectralExtractor using PyTorch and torchaudio.
Extracts zero-crossing rate, spectral centroid, bandwidth, rolloff, flatness
"""

import torch
import numpy as np
from typing import Dict, Any
from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.gpu_audio_utils import get_gpu_audio_processor
import logging

logger = logging.getLogger(__name__)


class SpectralExtractor(BaseExtractor):
    """
    GPU-Optimized Spectral Extractor for spectral characteristics
    GPU-accelerated replacement using PyTorch and torchaudio
    """
    
    name = "spectral_extractor"
    version = "3.0.0"
    description = "GPU-accelerated spectral characteristics: ZCR, centroid, bandwidth, rolloff, flatness"
    category = "spectral"
    dependencies = ["torch", "torchaudio"]
    estimated_duration = 2.0  # Much faster than CPU version
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize GPU-optimized spectral extractor.
        
        Args:
            device: Device to use for processing ("cuda" or "cpu")
        """
        super().__init__()
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.sample_rate = 22050
        self.hop_length = 512
        self.frame_length = 2048
        self.n_fft = 2048
        
        # Initialize GPU audio processor
        self.gpu_processor = get_gpu_audio_processor(self.device, self.sample_rate)
        
        logger.info(f"GPU-Optimized Spectral Extractor initialized on device: {self.device}")
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract spectral features from audio file using GPU acceleration.
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with spectral features
        """
        try:
            self.logger.info(f"Starting GPU-optimized spectral extraction for {input_uri}")
            
            # Load audio using GPU processor
            audio_tensor, sr = self.gpu_processor.load_audio(input_uri, self.sample_rate)
            
            # Extract spectral features using GPU
            features = self._extract_spectral_features_gpu(audio_tensor, sr)
            
            self.logger.info(f"GPU-optimized spectral extraction completed successfully")
            
            return ExtractorResult(
                name=self.name,
                version=self.version,
                success=True,
                payload=features
            )
            
        except Exception as e:
            self.logger.error(f"GPU-optimized spectral extraction failed: {e}")
            return ExtractorResult(
                name=self.name,
                version=self.version,
                success=False,
                error=str(e)
            )
    
    def _extract_spectral_features_gpu(self, audio_tensor: torch.Tensor, sr: int) -> Dict[str, Any]:
        """
        Extract spectral features from audio using GPU acceleration.
        
        Args:
            audio_tensor: Audio tensor on GPU
            sr: Sample rate
            
        Returns:
            Dictionary of spectral features
        """
        features = {}
        
        # Zero Crossing Rate (ZCR)
        zcr = self.gpu_processor.zero_crossing_rate(
            audio_tensor,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        zcr_stats = self.gpu_processor.compute_statistics(zcr, "zcr")
        features.update(zcr_stats)
        
        # Spectral Centroid
        spectral_centroids = self.gpu_processor.spectral_centroid(
            audio_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        centroid_stats = self.gpu_processor.compute_statistics(spectral_centroids, "spectral_centroid")
        features.update(centroid_stats)
        
        # Spectral Bandwidth
        spectral_bandwidth = self.gpu_processor.spectral_bandwidth(
            audio_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        bandwidth_stats = self.gpu_processor.compute_statistics(spectral_bandwidth, "spectral_bandwidth")
        features.update(bandwidth_stats)
        
        # Spectral Rolloff
        spectral_rolloff = self.gpu_processor.spectral_rolloff(
            audio_tensor,
            roll_percent=0.85,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        rolloff_stats = self.gpu_processor.compute_statistics(spectral_rolloff, "spectral_rolloff")
        features.update(rolloff_stats)
        
        # Spectral Flatness (Wiener entropy)
        spectral_flatness = self.gpu_processor.spectral_flatness(
            audio_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        flatness_stats = self.gpu_processor.compute_statistics(spectral_flatness, "spectral_flatness")
        features.update(flatness_stats)
        
        # Additional spectral features
        
        # Spectral Contrast
        spectral_contrast = self.gpu_processor.spectral_contrast(
            audio_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_bands=6
        )
        
        # Mean contrast across frequency bands
        contrast_mean = torch.mean(spectral_contrast, dim=0)
        contrast_std = torch.std(spectral_contrast, dim=0)
        
        features["spectral_contrast_mean"] = float(torch.mean(contrast_mean).item())
        features["spectral_contrast_std"] = float(torch.mean(contrast_std).item())
        
        # Spectral Flux (rate of change in spectrum)
        spectral_flux = self.gpu_processor.spectral_flux(
            audio_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        flux_stats = self.gpu_processor.compute_statistics(spectral_flux, "spectral_flux")
        features.update(flux_stats)
        
        # Spectral Entropy
        spectral_entropy = self.gpu_processor.spectral_entropy(
            audio_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        entropy_stats = self.gpu_processor.compute_statistics(spectral_entropy, "spectral_entropy")
        features.update(entropy_stats)
        
        # Spectral Kurtosis and Skewness
        features["spectral_centroid_skewness"] = self.gpu_processor.compute_skewness(spectral_centroids)
        features["spectral_centroid_kurtosis"] = self.gpu_processor.compute_kurtosis(spectral_centroids)
        
        features["spectral_bandwidth_skewness"] = self.gpu_processor.compute_skewness(spectral_bandwidth)
        features["spectral_bandwidth_kurtosis"] = self.gpu_processor.compute_kurtosis(spectral_bandwidth)
        
        # Spectral Shape Descriptors
        centroid_mean = features["spectral_centroid_mean"]
        bandwidth_mean = features["spectral_bandwidth_mean"]
        rolloff_mean = features["spectral_rolloff_mean"]
        
        features["spectral_centroid_bandwidth_ratio"] = (
            centroid_mean / bandwidth_mean if bandwidth_mean > 0 else 0.0
        )
        
        features["spectral_rolloff_centroid_ratio"] = (
            rolloff_mean / centroid_mean if centroid_mean > 0 else 0.0
        )
        
        # Frequency domain statistics
        features["spectral_centroid_normalized"] = centroid_mean / (sr / 2)
        features["spectral_rolloff_normalized"] = rolloff_mean / (sr / 2)
        
        # Additional advanced features
        
        # Spectral irregularity (measure of spectral smoothness)
        stft = self.gpu_processor.stft(audio_tensor, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = torch.abs(stft)
        
        # Compute spectral irregularity
        magnitude_diff = torch.diff(magnitude, dim=0)
        irregularity = torch.mean(torch.abs(magnitude_diff), dim=0)
        irregularity_stats = self.gpu_processor.compute_statistics(irregularity, "spectral_irregularity")
        features.update(irregularity_stats)
        
        # Spectral slope (linear regression slope of log-magnitude spectrum)
        log_magnitude = torch.log(magnitude + 1e-10)
        freqs = torch.linspace(0, sr // 2, magnitude.shape[0], device=self.device)
        
        # Compute slope for each frame
        slopes = []
        for i in range(log_magnitude.shape[1]):
            frame_log_mag = log_magnitude[:, i]
            valid_mask = torch.isfinite(frame_log_mag)
            
            if torch.sum(valid_mask) > 1:
                valid_freqs = freqs[valid_mask]
                valid_log_mag = frame_log_mag[valid_mask]
                
                # Linear regression
                n = valid_freqs.shape[0]
                sum_x = torch.sum(valid_freqs)
                sum_y = torch.sum(valid_log_mag)
                sum_xy = torch.sum(valid_freqs * valid_log_mag)
                sum_x2 = torch.sum(valid_freqs ** 2)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2 + 1e-10)
                slopes.append(slope)
            else:
                slopes.append(torch.tensor(0.0, device=self.device))
        
        if slopes:
            slopes_tensor = torch.stack(slopes)
            slope_stats = self.gpu_processor.compute_statistics(slopes_tensor, "spectral_slope")
            features.update(slope_stats)
        
        # Spectral decrease (rate of decrease of spectral amplitude)
        spectral_decrease = torch.mean(magnitude[1:, :] - magnitude[:-1, :], dim=0)
        decrease_stats = self.gpu_processor.compute_statistics(spectral_decrease, "spectral_decrease")
        features.update(decrease_stats)
        
        # Spectral variation (standard deviation of spectral amplitude)
        spectral_variation = torch.std(magnitude, dim=0)
        variation_stats = self.gpu_processor.compute_statistics(spectral_variation, "spectral_variation")
        features.update(variation_stats)
        
        return features
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get extractor parameters.
        
        Returns:
            Dictionary with extractor parameters
        """
        return {
            "device": self.device,
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "frame_length": self.frame_length,
            "n_fft": self.n_fft
        }


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python gpu_optimized_spectral_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = GPUOptimizedSpectralExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
