"""
Spectral Extractor for spectral characteristics
Extracts zero-crossing rate, spectral centroid, bandwidth, rolloff, flatness
"""

import numpy as np
import librosa
from typing import Dict, Any
from src.core.base_extractor import BaseExtractor, ExtractorResult


class SpectralExtractor(BaseExtractor):
    """
    Spectral Extractor for spectral characteristics
    Extracts ZCR, spectral centroid, bandwidth, rolloff, flatness
    """
    
    name = "spectral"
    version = "1.0.0"
    description = "Spectral characteristics: ZCR, centroid, bandwidth, rolloff, flatness"
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 22050
        self.hop_length = 512
        self.frame_length = 2048
        self.n_fft = 2048
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract spectral features from audio file
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with spectral features
        """
        try:
            self.logger.info(f"Starting spectral extraction for {input_uri}")
            
            # Load audio
            audio, sr = librosa.load(input_uri, sr=self.sample_rate)
            
            # Extract spectral features
            features = self._extract_spectral_features(audio, sr)
            
            self.logger.info(f"Spectral extraction completed successfully")
            
            return ExtractorResult(
                name=self.name,
                version=self.version,
                success=True,
                payload=features
            )
            
        except Exception as e:
            self.logger.error(f"Spectral extraction failed: {e}")
            return ExtractorResult(
                name=self.name,
                version=self.version,
                success=False,
                error=str(e)
            )
    
    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract spectral features from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of spectral features
        """
        features = {}
        
        # Zero Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))
        features["zcr_min"] = float(np.min(zcr))
        features["zcr_max"] = float(np.max(zcr))
        features["zcr_median"] = float(np.median(zcr))
        
        # Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio,
            sr=sr,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )[0]
        
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
        features["spectral_centroid_std"] = float(np.std(spectral_centroids))
        features["spectral_centroid_min"] = float(np.min(spectral_centroids))
        features["spectral_centroid_max"] = float(np.max(spectral_centroids))
        features["spectral_centroid_median"] = float(np.median(spectral_centroids))
        
        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )[0]
        
        features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
        features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))
        features["spectral_bandwidth_min"] = float(np.min(spectral_bandwidth))
        features["spectral_bandwidth_max"] = float(np.max(spectral_bandwidth))
        features["spectral_bandwidth_median"] = float(np.median(spectral_bandwidth))
        
        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            roll_percent=0.85  # 85% energy rolloff
        )[0]
        
        features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
        features["spectral_rolloff_std"] = float(np.std(spectral_rolloff))
        features["spectral_rolloff_min"] = float(np.min(spectral_rolloff))
        features["spectral_rolloff_max"] = float(np.max(spectral_rolloff))
        features["spectral_rolloff_median"] = float(np.median(spectral_rolloff))
        
        # Spectral Flatness (Wiener entropy)
        spectral_flatness = librosa.feature.spectral_flatness(
            y=audio,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )[0]
        
        features["spectral_flatness_mean"] = float(np.mean(spectral_flatness))
        features["spectral_flatness_std"] = float(np.std(spectral_flatness))
        features["spectral_flatness_min"] = float(np.min(spectral_flatness))
        features["spectral_flatness_max"] = float(np.max(spectral_flatness))
        features["spectral_flatness_median"] = float(np.median(spectral_flatness))
        
        # Additional spectral features
        
        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=sr,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        
        # Mean contrast across frequency bands
        features["spectral_contrast_mean"] = float(np.mean(spectral_contrast))
        features["spectral_contrast_std"] = float(np.std(spectral_contrast))
        
        # Spectral Flux (rate of change in spectrum)
        stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=self.n_fft)
        magnitude = np.abs(stft)
        
        # Calculate spectral flux as sum of squared differences
        spectral_flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
        
        features["spectral_flux_mean"] = float(np.mean(spectral_flux))
        features["spectral_flux_std"] = float(np.std(spectral_flux))
        features["spectral_flux_min"] = float(np.min(spectral_flux))
        features["spectral_flux_max"] = float(np.max(spectral_flux))
        
        # Spectral Entropy
        spectral_entropy = self._calculate_spectral_entropy(magnitude)
        features["spectral_entropy_mean"] = float(np.mean(spectral_entropy))
        features["spectral_entropy_std"] = float(np.std(spectral_entropy))
        
        # Spectral Kurtosis and Skewness
        features["spectral_centroid_skewness"] = self._calculate_skewness(spectral_centroids)
        features["spectral_centroid_kurtosis"] = self._calculate_kurtosis(spectral_centroids)
        
        features["spectral_bandwidth_skewness"] = self._calculate_skewness(spectral_bandwidth)
        features["spectral_bandwidth_kurtosis"] = self._calculate_kurtosis(spectral_bandwidth)
        
        # Spectral Shape Descriptors
        features["spectral_centroid_bandwidth_ratio"] = (
            features["spectral_centroid_mean"] / features["spectral_bandwidth_mean"]
            if features["spectral_bandwidth_mean"] > 0 else 0.0
        )
        
        features["spectral_rolloff_centroid_ratio"] = (
            features["spectral_rolloff_mean"] / features["spectral_centroid_mean"]
            if features["spectral_centroid_mean"] > 0 else 0.0
        )
        
        # Frequency domain statistics
        features["spectral_centroid_normalized"] = features["spectral_centroid_mean"] / (sr / 2)
        features["spectral_rolloff_normalized"] = features["spectral_rolloff_mean"] / (sr / 2)
        
        return features
    
    def _calculate_spectral_entropy(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Calculate spectral entropy for each frame
        
        Args:
            magnitude: Magnitude spectrum
            
        Returns:
            Spectral entropy array
        """
        # Normalize magnitude to get probability distribution
        magnitude_norm = magnitude / (np.sum(magnitude, axis=0) + 1e-10)
        
        # Calculate entropy: -sum(p * log2(p))
        entropy = -np.sum(magnitude_norm * np.log2(magnitude_norm + 1e-10), axis=0)
        
        return entropy
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return float(skewness)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3
        return float(kurtosis)


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python spectral_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = SpectralExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
