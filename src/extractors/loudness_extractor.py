"""
RMS/Loudness extractor for audio feature extraction.

This extractor implements:
- RMS (Root Mean Square) energy analysis
- LUFS loudness measurement using pyloudnorm
- Peak amplitude detection
- Clip detection
- Statistical aggregation
"""

import librosa
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from typing import Dict, Any, Tuple
import logging
from src.core.base_extractor import BaseExtractor
from src.schemas.models import ExtractorResult

logger = logging.getLogger(__name__)


class LoudnessExtractor(BaseExtractor):
    """Extractor for RMS energy and loudness features."""
    
    name = "loudness_extractor"
    version = "1.0.0"
    description = "RMS energy and LUFS loudness feature extraction"
    category = "core"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 2.0
    
    def __init__(self):
        """Initialize Loudness extractor with default parameters."""
        super().__init__()
        
        # RMS parameters
        self.frame_length = 2048  # Frame length for RMS calculation
        self.hop_length = 512  # Hop length for RMS calculation
        
        # LUFS parameters
        self.lufs_target = -23.0  # Target LUFS level
        self.lufs_tolerance = 0.1  # Tolerance for LUFS measurement
        
        self.logger.info(f"Initialized {self.name} v{self.version}")
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract RMS and loudness features from audio file.
        
        Args:
            input_uri: Path to input audio file
            tmp_path: Path to temporary directory (unused for this extractor)
            
        Returns:
            ExtractorResult with RMS and loudness features
        """
        self._log_extraction_start(input_uri)
        
        try:
            # Load audio file
            audio, sample_rate = self._load_audio(input_uri)
            
            # Extract RMS features
            rms_features = self._extract_rms_features(audio, sample_rate)
            
            # Extract LUFS loudness features
            lufs_features = self._extract_lufs_features(audio, sample_rate)
            
            # Extract peak amplitude features
            peak_features = self._extract_peak_features(audio)
            
            # Extract clip detection features
            clip_features = self._extract_clip_features(audio)
            
            # Combine all features
            payload = {**rms_features, **lufs_features, **peak_features, **clip_features}
            
            # Create successful result
            result = self._create_result(
                success=True,
                payload=payload,
                processing_time=None  # Will be set by base class timing
            )
            
            self._log_extraction_success(input_uri, 0.0)  # Time will be updated
            return result
            
        except Exception as e:
            error_msg = f"Loudness extraction failed: {str(e)}"
            self._log_extraction_error(input_uri, error_msg, 0.0)
            
            return self._create_result(
                success=False,
                error=error_msg,
                processing_time=None
            )
    
    def _load_audio(self, input_uri: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file using soundfile (preserves original sample rate for LUFS).
        
        Args:
            input_uri: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio with soundfile to preserve original sample rate
            audio, sr = sf.read(input_uri, dtype='float32')
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            self.logger.debug(f"Loaded audio: {len(audio)} samples at {sr} Hz")
            return audio, sr
            
        except Exception as e:
            self.logger.error(f"Failed to load audio file {input_uri}: {str(e)}")
            raise
    
    def _extract_rms_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Extract RMS energy features.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with RMS features
        """
        try:
            # Calculate RMS using librosa
            rms = librosa.feature.rms(
                y=audio,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )[0]  # Remove singleton dimension
            
            # Calculate statistics
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))
            rms_min = float(np.min(rms))
            rms_max = float(np.max(rms))
            rms_median = float(np.median(rms))
            
            # Calculate RMS percentile
            rms_p25 = float(np.percentile(rms, 25))
            rms_p75 = float(np.percentile(rms, 75))
            
            # Calculate RMS range and coefficient of variation
            rms_range = rms_max - rms_min
            rms_cv = rms_std / rms_mean if rms_mean > 0 else 0.0
            
            features = {
                "rms_mean": rms_mean,
                "rms_std": rms_std,
                "rms_min": rms_min,
                "rms_max": rms_max,
                "rms_median": rms_median,
                "rms_p25": rms_p25,
                "rms_p75": rms_p75,
                "rms_range": rms_range,
                "rms_cv": rms_cv,
                "rms_array": rms.tolist()  # Full RMS array
            }
            
            self.logger.debug(f"Extracted RMS features: {len(features)} features")
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract RMS features: {str(e)}")
            raise
    
    def _extract_lufs_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Extract LUFS loudness features using pyloudnorm.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with LUFS features
        """
        try:
            # Create loudness meter
            meter = pyln.Meter(sample_rate)
            
            # Calculate integrated loudness
            loudness = meter.integrated_loudness(audio)
            
            # Calculate momentary loudness (3-second window)
            momentary_loudness = meter.momentary_loudness(audio)
            
            # Calculate short-term loudness (400ms window)
            short_term_loudness = meter.short_term_loudness(audio)
            
            # Calculate statistics for momentary loudness
            momentary_mean = float(np.mean(momentary_loudness))
            momentary_std = float(np.std(momentary_loudness))
            momentary_min = float(np.min(momentary_loudness))
            momentary_max = float(np.max(momentary_loudness))
            
            # Calculate statistics for short-term loudness
            short_term_mean = float(np.mean(short_term_loudness))
            short_term_std = float(np.std(short_term_loudness))
            short_term_min = float(np.min(short_term_loudness))
            short_term_max = float(np.max(short_term_loudness))
            
            # Calculate loudness range (LRA)
            lra = float(np.max(short_term_loudness) - np.min(short_term_loudness))
            
            # Calculate peak level
            peak_level = float(np.max(np.abs(audio)))
            peak_db = 20 * np.log10(peak_level) if peak_level > 0 else -np.inf
            
            features = {
                "loudness_lufs": loudness,
                "loudness_momentary_mean": momentary_mean,
                "loudness_momentary_std": momentary_std,
                "loudness_momentary_min": momentary_min,
                "loudness_momentary_max": momentary_max,
                "loudness_short_term_mean": short_term_mean,
                "loudness_short_term_std": short_term_std,
                "loudness_short_term_min": short_term_min,
                "loudness_short_term_max": short_term_max,
                "loudness_range_lra": lra,
                "peak_level": peak_level,
                "peak_db": peak_db,
                "momentary_loudness_array": momentary_loudness.tolist(),
                "short_term_loudness_array": short_term_loudness.tolist()
            }
            
            self.logger.debug(f"Extracted LUFS features: {len(features)} features")
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract LUFS features: {str(e)}")
            # Return default values if LUFS calculation fails
            return {
                "loudness_lufs": -70.0,  # Default quiet level
                "loudness_momentary_mean": -70.0,
                "loudness_momentary_std": 0.0,
                "loudness_momentary_min": -70.0,
                "loudness_momentary_max": -70.0,
                "loudness_short_term_mean": -70.0,
                "loudness_short_term_std": 0.0,
                "loudness_short_term_min": -70.0,
                "loudness_short_term_max": -70.0,
                "loudness_range_lra": 0.0,
                "peak_level": 0.0,
                "peak_db": -np.inf,
                "momentary_loudness_array": [],
                "short_term_loudness_array": []
            }
    
    def _extract_peak_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract peak amplitude features.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with peak features
        """
        try:
            # Calculate peak amplitude
            peak_amplitude = float(np.max(np.abs(audio)))
            
            # Calculate peak-to-peak amplitude
            peak_to_peak = float(np.max(audio) - np.min(audio))
            
            # Calculate crest factor (peak to RMS ratio)
            rms = np.sqrt(np.mean(audio**2))
            crest_factor = peak_amplitude / rms if rms > 0 else 0.0
            
            # Calculate peak count (number of samples at peak level)
            peak_threshold = peak_amplitude * 0.95  # 95% of peak
            peak_count = int(np.sum(np.abs(audio) >= peak_threshold))
            peak_fraction = peak_count / len(audio)
            
            features = {
                "peak_amplitude": peak_amplitude,
                "peak_to_peak": peak_to_peak,
                "crest_factor": crest_factor,
                "peak_count": peak_count,
                "peak_fraction": peak_fraction
            }
            
            self.logger.debug(f"Extracted peak features: {len(features)} features")
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract peak features: {str(e)}")
            raise
    
    def _extract_clip_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract clip detection features.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with clip features
        """
        try:
            # Define clipping threshold (typically 0.99 or 0.95)
            clip_threshold = 0.99
            
            # Count clipped samples
            clipped_samples = np.sum(np.abs(audio) >= clip_threshold)
            clip_fraction = clipped_samples / len(audio)
            
            # Count hard clipping (samples at exactly 1.0 or -1.0)
            hard_clipped = np.sum((audio >= 1.0) | (audio <= -1.0))
            hard_clip_fraction = hard_clipped / len(audio)
            
            # Calculate clipping severity (how much over threshold)
            over_threshold = np.abs(audio) - clip_threshold
            over_threshold = over_threshold[over_threshold > 0]
            clip_severity = float(np.mean(over_threshold)) if len(over_threshold) > 0 else 0.0
            
            features = {
                "clip_fraction": clip_fraction,
                "hard_clip_fraction": hard_clip_fraction,
                "clip_severity": clip_severity,
                "clipped_samples": int(clipped_samples),
                "hard_clipped_samples": int(hard_clipped)
            }
            
            self.logger.debug(f"Extracted clip features: {len(features)} features")
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract clip features: {str(e)}")
            raise
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get extractor parameters.
        
        Returns:
            Dictionary with extractor parameters
        """
        return {
            "frame_length": self.frame_length,
            "hop_length": self.hop_length,
            "lufs_target": self.lufs_target,
            "lufs_tolerance": self.lufs_tolerance
        }


# For running as a module
if __name__ == "__main__":
    import sys
    import json
    import tempfile
    import os
    
    if len(sys.argv) != 2:
        print("Usage: python loudness_extractor.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file {audio_file} not found")
        sys.exit(1)
    
    # Create extractor and run
    extractor = LoudnessExtractor()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = extractor.run(audio_file, tmp_dir)
        
        # Print result as JSON
        print(json.dumps(result.dict(), indent=2))
