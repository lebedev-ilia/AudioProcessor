"""
Advanced Spectral Extractor for advanced spectral analysis
Extracts spectral flux, spectral contrast, spectral entropy, and LPC coefficients
"""

import numpy as np
import librosa
from typing import Dict, Any, List, Tuple, Optional
from src.core.base_extractor import BaseExtractor, ExtractorResult


class AdvancedSpectralExtractor(BaseExtractor):
    """
    Advanced Spectral Extractor for advanced spectral analysis
    Extracts spectral flux, spectral contrast, spectral entropy, and LPC coefficients
    """
    
    name = "advanced_spectral"
    version = "1.0.0"
    description = "Advanced spectral analysis: flux, contrast, entropy, LPC"
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 22050
        self.hop_length = 512
        self.frame_length = 2048
        self.n_fft = 2048
        self.lpc_order = 12  # LPC order
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract advanced spectral features from audio file
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with advanced spectral features
        """
        try:
            self.logger.info(f"Starting advanced spectral extraction for {input_uri}")
            
            # Load audio
            audio, sr = librosa.load(input_uri, sr=self.sample_rate)
            
            # Extract advanced spectral features with timing
            features, processing_time = self._time_execution(self._extract_advanced_spectral_features, audio, sr)
            
            self.logger.info(f"Advanced spectral extraction completed successfully in {processing_time:.3f}s")
            
            return self._create_result(
                success=True,
                payload=features,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Advanced spectral extraction failed: {e}")
            return self._create_result(
                success=False,
                error=str(e)
            )
    
    def _extract_advanced_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract advanced spectral features from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of advanced spectral features
        """
        features = {}
        
        # Spectral flux
        spectral_flux_features = self._extract_spectral_flux(audio, sr)
        features.update(spectral_flux_features)
        
        # Spectral contrast
        spectral_contrast_features = self._extract_spectral_contrast(audio, sr)
        features.update(spectral_contrast_features)
        
        # Spectral entropy
        spectral_entropy_features = self._extract_spectral_entropy(audio, sr)
        features.update(spectral_entropy_features)
        
        # LPC coefficients
        lpc_features = self._extract_lpc_coefficients(audio, sr)
        features.update(lpc_features)
        
        # Spectral irregularity
        irregularity_features = self._extract_spectral_irregularity(audio, sr)
        features.update(irregularity_features)
        
        # Spectral rolloff and centroid variations
        spectral_variations = self._extract_spectral_variations(audio, sr)
        features.update(spectral_variations)
        
        return features
    
    def _extract_spectral_flux(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract spectral flux features
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with spectral flux features
        """
        try:
            # Calculate STFT
            stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=self.n_fft)
            magnitude = np.abs(stft)
            
            # Calculate spectral flux (difference between consecutive frames)
            spectral_flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
            
            # Calculate statistics
            flux_mean = float(np.mean(spectral_flux))
            flux_std = float(np.std(spectral_flux))
            flux_min = float(np.min(spectral_flux))
            flux_max = float(np.max(spectral_flux))
            flux_median = float(np.median(spectral_flux))
            
            # Calculate flux percentiles
            flux_p25 = float(np.percentile(spectral_flux, 25))
            flux_p75 = float(np.percentile(spectral_flux, 75))
            flux_p90 = float(np.percentile(spectral_flux, 90))
            
            # Calculate flux variability
            flux_cv = flux_std / (flux_mean + 1e-10)
            
            return {
                "spectral_flux_mean": flux_mean,
                "spectral_flux_std": flux_std,
                "spectral_flux_min": flux_min,
                "spectral_flux_max": flux_max,
                "spectral_flux_median": flux_median,
                "spectral_flux_p25": flux_p25,
                "spectral_flux_p75": flux_p75,
                "spectral_flux_p90": flux_p90,
                "spectral_flux_cv": flux_cv
            }
            
        except Exception as e:
            self.logger.warning(f"Spectral flux extraction failed: {e}")
            return {
                "spectral_flux_mean": 0.0,
                "spectral_flux_std": 0.0,
                "spectral_flux_min": 0.0,
                "spectral_flux_max": 0.0,
                "spectral_flux_median": 0.0,
                "spectral_flux_p25": 0.0,
                "spectral_flux_p75": 0.0,
                "spectral_flux_p90": 0.0,
                "spectral_flux_cv": 0.0
            }
    
    def _extract_spectral_contrast(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract spectral contrast features
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with spectral contrast features
        """
        try:
            # Calculate spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(
                y=audio,
                sr=sr,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            
            # Calculate statistics for each frequency band
            contrast_features = {}
            
            for i in range(spectral_contrast.shape[0]):
                band_contrast = spectral_contrast[i]
                
                contrast_features[f"spectral_contrast_band_{i}_mean"] = float(np.mean(band_contrast))
                contrast_features[f"spectral_contrast_band_{i}_std"] = float(np.std(band_contrast))
                contrast_features[f"spectral_contrast_band_{i}_min"] = float(np.min(band_contrast))
                contrast_features[f"spectral_contrast_band_{i}_max"] = float(np.max(band_contrast))
            
            # Calculate overall contrast statistics
            overall_contrast = np.mean(spectral_contrast, axis=0)
            contrast_features["spectral_contrast_overall_mean"] = float(np.mean(overall_contrast))
            contrast_features["spectral_contrast_overall_std"] = float(np.std(overall_contrast))
            contrast_features["spectral_contrast_overall_min"] = float(np.min(overall_contrast))
            contrast_features["spectral_contrast_overall_max"] = float(np.max(overall_contrast))
            
            return contrast_features
            
        except Exception as e:
            self.logger.warning(f"Spectral contrast extraction failed: {e}")
            return {
                "spectral_contrast_overall_mean": 0.0,
                "spectral_contrast_overall_std": 0.0,
                "spectral_contrast_overall_min": 0.0,
                "spectral_contrast_overall_max": 0.0
            }
    
    def _extract_spectral_entropy(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract spectral entropy features
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with spectral entropy features
        """
        try:
            # Calculate STFT
            stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=self.n_fft)
            magnitude = np.abs(stft)
            
            # Calculate spectral entropy for each frame
            spectral_entropy = []
            
            for i in range(magnitude.shape[1]):
                frame_spectrum = magnitude[:, i]
                
                # Normalize to probability distribution
                frame_spectrum = frame_spectrum / (np.sum(frame_spectrum) + 1e-10)
                
                # Calculate entropy
                entropy = -np.sum(frame_spectrum * np.log2(frame_spectrum + 1e-10))
                spectral_entropy.append(entropy)
            
            spectral_entropy = np.array(spectral_entropy)
            
            # Calculate statistics
            entropy_mean = float(np.mean(spectral_entropy))
            entropy_std = float(np.std(spectral_entropy))
            entropy_min = float(np.min(spectral_entropy))
            entropy_max = float(np.max(spectral_entropy))
            entropy_median = float(np.median(spectral_entropy))
            
            # Calculate entropy percentiles
            entropy_p25 = float(np.percentile(spectral_entropy, 25))
            entropy_p75 = float(np.percentile(spectral_entropy, 75))
            entropy_p90 = float(np.percentile(spectral_entropy, 90))
            
            # Calculate entropy variability
            entropy_cv = entropy_std / (entropy_mean + 1e-10)
            
            return {
                "spectral_entropy_mean": entropy_mean,
                "spectral_entropy_std": entropy_std,
                "spectral_entropy_min": entropy_min,
                "spectral_entropy_max": entropy_max,
                "spectral_entropy_median": entropy_median,
                "spectral_entropy_p25": entropy_p25,
                "spectral_entropy_p75": entropy_p75,
                "spectral_entropy_p90": entropy_p90,
                "spectral_entropy_cv": entropy_cv
            }
            
        except Exception as e:
            self.logger.warning(f"Spectral entropy extraction failed: {e}")
            return {
                "spectral_entropy_mean": 0.0,
                "spectral_entropy_std": 0.0,
                "spectral_entropy_min": 0.0,
                "spectral_entropy_max": 0.0,
                "spectral_entropy_median": 0.0,
                "spectral_entropy_p25": 0.0,
                "spectral_entropy_p75": 0.0,
                "spectral_entropy_p90": 0.0,
                "spectral_entropy_cv": 0.0
            }
    
    def _extract_lpc_coefficients(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract LPC coefficients
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with LPC coefficient features
        """
        try:
            from scipy.signal import lfilter
            
            # Pre-emphasize the signal
            pre_emphasis = 0.97
            emphasized_signal = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
            
            # Calculate LPC coefficients using autocorrelation method
            autocorr = np.correlate(emphasized_signal, emphasized_signal, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Solve Yule-Walker equations using Levinson-Durbin algorithm
            lpc_coeffs = self._levinson_durbin(autocorr, self.lpc_order)
            
            # Extract LPC coefficient features
            lpc_features = {}
            
            # Individual coefficients
            for i in range(1, min(self.lpc_order + 1, len(lpc_coeffs))):
                lpc_features[f"lpc_coeff_{i}"] = float(lpc_coeffs[i])
            
            # LPC coefficient statistics
            if len(lpc_coeffs) > 1:
                lpc_features["lpc_coeff_mean"] = float(np.mean(lpc_coeffs[1:]))
                lpc_features["lpc_coeff_std"] = float(np.std(lpc_coeffs[1:]))
                lpc_features["lpc_coeff_min"] = float(np.min(lpc_coeffs[1:]))
                lpc_features["lpc_coeff_max"] = float(np.max(lpc_coeffs[1:]))
            else:
                lpc_features["lpc_coeff_mean"] = 0.0
                lpc_features["lpc_coeff_std"] = 0.0
                lpc_features["lpc_coeff_min"] = 0.0
                lpc_features["lpc_coeff_max"] = 0.0
            
            # LPC prediction error
            predicted_signal = lfilter([0] + lpc_coeffs[1:], [1], emphasized_signal)
            prediction_error = emphasized_signal - predicted_signal
            lpc_features["lpc_prediction_error"] = float(np.mean(prediction_error ** 2))
            
            return lpc_features
            
        except Exception as e:
            self.logger.warning(f"LPC coefficient extraction failed: {e}")
            return {
                "lpc_coeff_mean": 0.0,
                "lpc_coeff_std": 0.0,
                "lpc_coeff_min": 0.0,
                "lpc_coeff_max": 0.0,
                "lpc_prediction_error": 0.0
            }
    
    def _levinson_durbin(self, autocorr: np.ndarray, order: int) -> np.ndarray:
        """
        Levinson-Durbin algorithm for solving Yule-Walker equations
        
        Args:
            autocorr: Autocorrelation sequence
            order: LPC order
            
        Returns:
            LPC coefficients
        """
        try:
            # Initialize
            a = np.zeros(order + 1)
            a[0] = 1.0
            
            # Levinson-Durbin recursion
            for i in range(1, order + 1):
                if i == 1:
                    k = -autocorr[1] / autocorr[0]
                else:
                    k = -(autocorr[i] + np.dot(a[1:i], autocorr[i-1:0:-1])) / (autocorr[0] + np.dot(a[1:i], autocorr[1:i]))
                
                a[1:i+1] = a[1:i+1] + k * a[i-1:0:-1]
                a[i] = k
            
            return a
            
        except Exception as e:
            self.logger.warning(f"Levinson-Durbin algorithm failed: {e}")
            return np.zeros(order + 1)
    
    def _extract_spectral_irregularity(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract spectral irregularity features
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with spectral irregularity features
        """
        try:
            # Calculate STFT
            stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=self.n_fft)
            magnitude = np.abs(stft)
            
            # Calculate spectral irregularity for each frame
            spectral_irregularity = []
            
            for i in range(magnitude.shape[1]):
                frame_spectrum = magnitude[:, i]
                
                # Calculate irregularity as sum of squared differences between adjacent bins
                irregularity = np.sum((frame_spectrum[1:] - frame_spectrum[:-1]) ** 2)
                spectral_irregularity.append(irregularity)
            
            spectral_irregularity = np.array(spectral_irregularity)
            
            # Calculate statistics
            irregularity_mean = float(np.mean(spectral_irregularity))
            irregularity_std = float(np.std(spectral_irregularity))
            irregularity_min = float(np.min(spectral_irregularity))
            irregularity_max = float(np.max(spectral_irregularity))
            irregularity_median = float(np.median(spectral_irregularity))
            
            return {
                "spectral_irregularity_mean": irregularity_mean,
                "spectral_irregularity_std": irregularity_std,
                "spectral_irregularity_min": irregularity_min,
                "spectral_irregularity_max": irregularity_max,
                "spectral_irregularity_median": irregularity_median
            }
            
        except Exception as e:
            self.logger.warning(f"Spectral irregularity extraction failed: {e}")
            return {
                "spectral_irregularity_mean": 0.0,
                "spectral_irregularity_std": 0.0,
                "spectral_irregularity_min": 0.0,
                "spectral_irregularity_max": 0.0,
                "spectral_irregularity_median": 0.0
            }
    
    def _extract_spectral_variations(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract spectral rolloff and centroid variations
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with spectral variation features
        """
        try:
            # Calculate spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio,
                sr=sr,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )[0]
            
            # Calculate spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio,
                sr=sr,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )[0]
            
            # Calculate variations
            rolloff_variation = float(np.std(spectral_rolloff) / (np.mean(spectral_rolloff) + 1e-10))
            centroid_variation = float(np.std(spectral_centroid) / (np.mean(spectral_centroid) + 1e-10))
            
            # Calculate correlation between rolloff and centroid
            correlation = float(np.corrcoef(spectral_rolloff, spectral_centroid)[0, 1])
            
            return {
                "spectral_rolloff_variation": rolloff_variation,
                "spectral_centroid_variation": centroid_variation,
                "spectral_rolloff_centroid_correlation": correlation
            }
            
        except Exception as e:
            self.logger.warning(f"Spectral variations extraction failed: {e}")
            return {
                "spectral_rolloff_variation": 0.0,
                "spectral_centroid_variation": 0.0,
                "spectral_rolloff_centroid_correlation": 0.0
            }


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python advanced_spectral_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = AdvancedSpectralExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
