"""
Pitch Extractor for fundamental frequency (f0) estimation
Extracts pitch-related features using multiple algorithms
"""

import os
import numpy as np
import librosa
from typing import Dict, Any, Optional
from src.core.base_extractor import BaseExtractor, ExtractorResult


class PitchExtractor(BaseExtractor):
    """
    Pitch Extractor for fundamental frequency estimation
    Uses multiple algorithms: pyin, yin, crepe for robust pitch detection
    """
    
    name = "pitch"
    version = "1.0.0"
    description = "Fundamental frequency (f0) estimation using multiple algorithms"
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 22050
        self.hop_length = 512
        self.frame_length = 2048
        self.fmin = 50.0  # Minimum frequency (Hz)
        self.fmax = 2000.0  # Maximum frequency (Hz)
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract pitch features from audio file
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with pitch features
        """
        try:
            self.logger.info(f"Starting pitch extraction for {input_uri}")
            
            # Load audio
            audio, sr = librosa.load(input_uri, sr=self.sample_rate)
            
            # Extract pitch features with timing
            features, processing_time = self._time_execution(self._extract_pitch_features, audio, sr)
            
            self.logger.info(f"Pitch extraction completed successfully in {processing_time:.3f}s")
            
            return self._create_result(
                success=True,
                payload=features,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Pitch extraction failed: {e}")
            return self._create_result(
                success=False,
                error=str(e)
            )
    
    def _extract_pitch_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract pitch features using multiple algorithms
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of pitch features
        """
        features = {}
        
        # PYIN algorithm (most robust)
        try:
            f0_pyin, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=sr,
                hop_length=self.hop_length,
                frame_length=self.frame_length
            )
            
            # Remove NaN values
            f0_pyin_clean = f0_pyin[~np.isnan(f0_pyin)]
            voiced_flag_clean = voiced_flag[~np.isnan(voiced_flag)]
            
            if len(f0_pyin_clean) > 0:
                features.update(self._calculate_pitch_stats(f0_pyin_clean, "pyin"))
                features["voiced_fraction_pyin"] = np.mean(voiced_flag_clean)
                features["voiced_probability_mean_pyin"] = np.mean(voiced_probs[~np.isnan(voiced_probs)])
            else:
                features.update(self._get_zero_pitch_stats("pyin"))
                features["voiced_fraction_pyin"] = 0.0
                features["voiced_probability_mean_pyin"] = 0.0
                
        except Exception as e:
            self.logger.warning(f"PYIN pitch estimation failed: {e}")
            features.update(self._get_zero_pitch_stats("pyin"))
            features["voiced_fraction_pyin"] = 0.0
            features["voiced_probability_mean_pyin"] = 0.0
        
        # YIN algorithm
        try:
            f0_yin = librosa.yin(
                audio,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=sr,
                hop_length=self.hop_length,
                frame_length=self.frame_length
            )
            
            # Remove NaN values
            f0_yin_clean = f0_yin[~np.isnan(f0_yin)]
            
            if len(f0_yin_clean) > 0:
                features.update(self._calculate_pitch_stats(f0_yin_clean, "yin"))
            else:
                features.update(self._get_zero_pitch_stats("yin"))
                
        except Exception as e:
            self.logger.warning(f"YIN pitch estimation failed: {e}")
            features.update(self._get_zero_pitch_stats("yin"))
        
        # CREPE algorithm (if available)
        try:
            f0_crepe = self._extract_crepe_pitch(audio, sr)
            if f0_crepe is not None and len(f0_crepe) > 0:
                features.update(self._calculate_pitch_stats(f0_crepe, "crepe"))
            else:
                features.update(self._get_zero_pitch_stats("crepe"))
        except Exception as e:
            self.logger.warning(f"CREPE pitch estimation failed: {e}")
            features.update(self._get_zero_pitch_stats("crepe"))
        
        # Overall pitch statistics (using best available method)
        best_f0 = None
        best_method = "pyin"
        
        if "f0_mean_pyin" in features and features["f0_mean_pyin"] > 0:
            best_f0 = f0_pyin_clean if len(f0_pyin_clean) > 0 else None
            best_method = "pyin"
        elif "f0_mean_yin" in features and features["f0_mean_yin"] > 0:
            best_f0 = f0_yin_clean if len(f0_yin_clean) > 0 else None
            best_method = "yin"
        elif "f0_mean_crepe" in features and features["f0_mean_crepe"] > 0:
            best_f0 = f0_crepe
            best_method = "crepe"
        
        if best_f0 is not None and len(best_f0) > 0:
            features["f0_mean"] = np.mean(best_f0)
            features["f0_std"] = np.std(best_f0)
            features["f0_min"] = np.min(best_f0)
            features["f0_max"] = np.max(best_f0)
            features["f0_median"] = np.median(best_f0)
            features["f0_method"] = best_method
        else:
            features["f0_mean"] = 0.0
            features["f0_std"] = 0.0
            features["f0_min"] = 0.0
            features["f0_max"] = 0.0
            features["f0_median"] = 0.0
            features["f0_method"] = "none"
        
        # Pitch stability metrics
        if best_f0 is not None and len(best_f0) > 1:
            # Pitch variation (jitter-like measure)
            pitch_diff = np.diff(best_f0)
            features["pitch_variation"] = np.std(pitch_diff)
            features["pitch_stability"] = 1.0 / (1.0 + features["pitch_variation"])
            
            # Pitch range
            features["pitch_range"] = features["f0_max"] - features["f0_min"]
            
            # Pitch distribution
            features["pitch_skewness"] = self._calculate_skewness(best_f0)
            features["pitch_kurtosis"] = self._calculate_kurtosis(best_f0)
        else:
            features["pitch_variation"] = 0.0
            features["pitch_stability"] = 0.0
            features["pitch_range"] = 0.0
            features["pitch_skewness"] = 0.0
            features["pitch_kurtosis"] = 0.0
        
        return features
    
    def _extract_crepe_pitch(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        Extract pitch using CREPE algorithm
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Pitch array or None if failed
        """
        try:
            import crepe
            
            # CREPE expects 16kHz sample rate
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio
            
            # Run CREPE
            time, frequency, confidence, activation = crepe.predict(
                audio_16k,
                sr=16000,
                model_capacity='medium',
                viterbi=True
            )
            
            # Filter by confidence threshold
            confidence_threshold = 0.3
            valid_indices = confidence > confidence_threshold
            f0_crepe = frequency[valid_indices]
            
            # Remove zero frequencies
            f0_crepe = f0_crepe[f0_crepe > 0]
            
            return f0_crepe if len(f0_crepe) > 0 else None
            
        except ImportError:
            self.logger.warning("CREPE not available. Install with: pip install crepe")
            return None
        except Exception as e:
            self.logger.warning(f"CREPE pitch estimation failed: {e}")
            return None
    
    def _calculate_pitch_stats(self, f0: np.ndarray, method: str) -> Dict[str, Any]:
        """
        Calculate pitch statistics for a given method
        
        Args:
            f0: Pitch array
            method: Method name (pyin, yin, crepe)
            
        Returns:
            Dictionary of pitch statistics
        """
        stats = {}
        
        if len(f0) > 0:
            stats[f"f0_mean_{method}"] = np.mean(f0)
            stats[f"f0_std_{method}"] = np.std(f0)
            stats[f"f0_min_{method}"] = np.min(f0)
            stats[f"f0_max_{method}"] = np.max(f0)
            stats[f"f0_median_{method}"] = np.median(f0)
            stats[f"f0_count_{method}"] = len(f0)
            
            # Percentiles
            stats[f"f0_p25_{method}"] = np.percentile(f0, 25)
            stats[f"f0_p75_{method}"] = np.percentile(f0, 75)
            stats[f"f0_p90_{method}"] = np.percentile(f0, 90)
        else:
            stats.update(self._get_zero_pitch_stats(method))
        
        return stats
    
    def _get_zero_pitch_stats(self, method: str) -> Dict[str, Any]:
        """
        Get zero pitch statistics for failed methods
        
        Args:
            method: Method name
            
        Returns:
            Dictionary of zero statistics
        """
        return {
            f"f0_mean_{method}": 0.0,
            f"f0_std_{method}": 0.0,
            f"f0_min_{method}": 0.0,
            f"f0_max_{method}": 0.0,
            f"f0_median_{method}": 0.0,
            f"f0_count_{method}": 0,
            f"f0_p25_{method}": 0.0,
            f"f0_p75_{method}": 0.0,
            f"f0_p90_{method}": 0.0
        }
    
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
        print("Usage: python pitch_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = PitchExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
