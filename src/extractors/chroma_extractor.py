"""
GPU-Optimized Chroma Extractor for chroma feature extraction.

GPU-accelerated replacement for ChromaExtractor using PyTorch and torchaudio.
Extracts chroma features from audio using GPU acceleration.
"""

import torch
import numpy as np
import librosa
from typing import Dict, Any, Tuple
import logging
from src.core.base_extractor import BaseExtractor
from src.schemas.models import ExtractorResult

logger = logging.getLogger(__name__)


class ChromaExtractor(BaseExtractor):
    """GPU-optimized extractor for chroma features."""
    
    name = "chroma_extractor"
    version = "3.0.0"
    description = "GPU-accelerated chroma feature extraction with statistical aggregation"
    category = "spectral"
    dependencies = ["torch", "torchaudio"]
    estimated_duration = 2.0  # Much faster than CPU version
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize GPU-optimized chroma extractor.
        
        Args:
            device: Device to use for processing ("cuda" or "cpu")
        """
        super().__init__()
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        # Chroma parameters
        self.n_chroma = 12  # Number of chroma bins
        self.n_fft = 2048  # FFT window size
        self.hop_length = 512  # Hop length for STFT
        self.norm = 2.0  # Normalization for chroma
        self.sample_rate = 22050
        
        self.logger.info(f"GPU-Optimized Chroma Extractor initialized on device: {self.device}")
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract chroma features from audio file using librosa.
        
        Args:
            input_uri: Path to input audio file
            tmp_path: Path to temporary directory (unused for this extractor)
            
        Returns:
            ExtractorResult with chroma features
        """
        try:
            # Load audio file using librosa
            audio, sr = librosa.load(input_uri, sr=self.sample_rate)
            
            # Extract chroma features using librosa
            chroma_features = self._extract_chroma_features(audio, sr)
            
            # Calculate statistical aggregations
            chroma_stats = self._calculate_statistics(chroma_features, "chroma")
            
            # Create successful result
            result = self._create_result(
                success=True,
                payload=chroma_stats,
                processing_time=None  # Will be set by base class timing
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Chroma extraction failed: {str(e)}"
            return self._create_result(
                success=False,
                error=error_msg,
                processing_time=None
            )
    
    def _extract_chroma_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract chroma features from audio using librosa.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate of audio
            
        Returns:
            Chroma features array of shape (n_chroma, n_frames)
        """
        try:
            # Extract chroma features using librosa
            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=sample_rate,
                n_chroma=self.n_chroma,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                norm=self.norm
            )
            
            self.logger.debug(f"Extracted chroma features: {chroma.shape}")
            return chroma
            
        except Exception as e:
            self.logger.error(f"Failed to extract chroma features: {str(e)}")
            raise
    
    def _calculate_statistics(self, chroma_features: np.ndarray, prefix: str) -> Dict[str, Any]:
        """
        Calculate statistical aggregations for chroma features.
        
        Args:
            chroma_features: Chroma features array of shape (n_chroma, n_frames)
            prefix: Prefix for feature names
            
        Returns:
            Dictionary with statistical features
        """
        try:
            # Calculate mean and std across time (axis=1)
            mean_features = np.mean(chroma_features, axis=1)
            std_features = np.std(chroma_features, axis=1)
            
            # Create feature dictionary
            stats = {}
            
            # Add individual chroma bin statistics
            for i in range(len(mean_features)):
                stats[f"{prefix}_{i}_mean"] = float(mean_features[i])
                stats[f"{prefix}_{i}_std"] = float(std_features[i])
            
            # Add overall statistics as lists
            stats[f"{prefix}_mean"] = mean_features.tolist()
            stats[f"{prefix}_std"] = std_features.tolist()
            
            # Additional statistical features
            stats[f"{prefix}_min"] = np.min(chroma_features, axis=1).tolist()
            stats[f"{prefix}_max"] = np.max(chroma_features, axis=1).tolist()
            stats[f"{prefix}_median"] = np.median(chroma_features, axis=1).tolist()
            
            # Range and coefficient of variation
            feature_range = np.max(chroma_features, axis=1) - np.min(chroma_features, axis=1)
            feature_cv = std_features / (np.abs(mean_features) + 1e-10)
            
            stats[f"{prefix}_range"] = feature_range.tolist()
            stats[f"{prefix}_cv"] = feature_cv.tolist()
            
            # Global statistics across all chroma bins
            stats[f"{prefix}_global_mean"] = float(np.mean(chroma_features))
            stats[f"{prefix}_global_std"] = float(np.std(chroma_features))
            stats[f"{prefix}_global_min"] = float(np.min(chroma_features))
            stats[f"{prefix}_global_max"] = float(np.max(chroma_features))
            
            # Chroma-specific features
            
            # Chroma energy (sum of all chroma bins)
            chroma_energy = np.sum(chroma_features, axis=0)
            stats[f"{prefix}_energy_mean"] = float(np.mean(chroma_energy))
            stats[f"{prefix}_energy_std"] = float(np.std(chroma_energy))
            stats[f"{prefix}_energy_min"] = float(np.min(chroma_energy))
            stats[f"{prefix}_energy_max"] = float(np.max(chroma_energy))
            
            # Chroma centroid (weighted average of chroma bins)
            chroma_centroid = np.sum(np.arange(self.n_chroma).reshape(-1, 1) * chroma_features, axis=0) / (np.sum(chroma_features, axis=0) + 1e-10)
            stats[f"{prefix}_centroid_mean"] = float(np.mean(chroma_centroid))
            stats[f"{prefix}_centroid_std"] = float(np.std(chroma_centroid))
            stats[f"{prefix}_centroid_min"] = float(np.min(chroma_centroid))
            stats[f"{prefix}_centroid_max"] = float(np.max(chroma_centroid))
            
            # Chroma spread (standard deviation of chroma distribution)
            chroma_spread = np.sqrt(np.sum(((np.arange(self.n_chroma).reshape(-1, 1) - chroma_centroid.reshape(1, -1)) ** 2) * chroma_features, axis=0) / (np.sum(chroma_features, axis=0) + 1e-10))
            stats[f"{prefix}_spread_mean"] = float(np.mean(chroma_spread))
            stats[f"{prefix}_spread_std"] = float(np.std(chroma_spread))
            stats[f"{prefix}_spread_min"] = float(np.min(chroma_spread))
            stats[f"{prefix}_spread_max"] = float(np.max(chroma_spread))
            
            # Chroma flux (rate of change in chroma features)
            chroma_flux = np.sum(np.diff(chroma_features, axis=1) ** 2, axis=0)
            stats[f"{prefix}_flux_mean"] = float(np.mean(chroma_flux))
            stats[f"{prefix}_flux_std"] = float(np.std(chroma_flux))
            stats[f"{prefix}_flux_min"] = float(np.min(chroma_flux))
            stats[f"{prefix}_flux_max"] = float(np.max(chroma_flux))
            
            # Chroma entropy (entropy of chroma distribution)
            chroma_normalized = chroma_features / (np.sum(chroma_features, axis=0, keepdims=True) + 1e-10)
            chroma_entropy = -np.sum(chroma_normalized * np.log2(chroma_normalized + 1e-10), axis=0)
            stats[f"{prefix}_entropy_mean"] = float(np.mean(chroma_entropy))
            stats[f"{prefix}_entropy_std"] = float(np.std(chroma_entropy))
            stats[f"{prefix}_entropy_min"] = float(np.min(chroma_entropy))
            stats[f"{prefix}_entropy_max"] = float(np.max(chroma_entropy))
            
            # Chroma irregularity (measure of chroma smoothness)
            chroma_diff = np.diff(chroma_features, axis=0)
            chroma_irregularity = np.mean(np.abs(chroma_diff), axis=0)
            stats[f"{prefix}_irregularity_mean"] = float(np.mean(chroma_irregularity))
            stats[f"{prefix}_irregularity_std"] = float(np.std(chroma_irregularity))
            stats[f"{prefix}_irregularity_min"] = float(np.min(chroma_irregularity))
            stats[f"{prefix}_irregularity_max"] = float(np.max(chroma_irregularity))
            
            # Chroma variation (standard deviation of chroma features across time)
            chroma_variation = np.std(chroma_features, axis=1)
            stats[f"{prefix}_variation_mean"] = float(np.mean(chroma_variation))
            stats[f"{prefix}_variation_std"] = float(np.std(chroma_variation))
            stats[f"{prefix}_variation_min"] = float(np.min(chroma_variation))
            stats[f"{prefix}_variation_max"] = float(np.max(chroma_variation))
            
            # Chroma correlation matrix (simplified - just pairwise correlations)
            chroma_corr = np.corrcoef(chroma_features)
            stats[f"{prefix}_correlation_mean"] = float(np.mean(chroma_corr))
            stats[f"{prefix}_correlation_std"] = float(np.std(chroma_corr))
            
            # Chroma tonality (measure of tonal content)
            chroma_tonality = np.sum(chroma_features ** 2, axis=0) / (np.sum(chroma_features, axis=0) + 1e-10) ** 2
            stats[f"{prefix}_tonality_mean"] = float(np.mean(chroma_tonality))
            stats[f"{prefix}_tonality_std"] = float(np.std(chroma_tonality))
            stats[f"{prefix}_tonality_min"] = float(np.min(chroma_tonality))
            stats[f"{prefix}_tonality_max"] = float(np.max(chroma_tonality))
            
            self.logger.debug(f"Calculated chroma statistics: {len(stats)} features")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to calculate chroma statistics: {str(e)}")
            raise
    
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get extractor parameters.
        
        Returns:
            Dictionary with extractor parameters
        """
        return {
            "device": self.device,
            "n_chroma": self.n_chroma,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "norm": self.norm,
            "sample_rate": self.sample_rate
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
