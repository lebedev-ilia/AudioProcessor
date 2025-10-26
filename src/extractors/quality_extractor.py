"""
Quality Extractor for audio quality assessment
Extracts SNR, clip detection, and other quality metrics
"""

import numpy as np
import librosa
from typing import Dict, Any, Tuple
from src.core.base_extractor import BaseExtractor, ExtractorResult


class QualityExtractor(BaseExtractor):
    """
    Quality Extractor for audio quality assessment
    Extracts SNR, clip detection, hum detection, and quality metrics
    """
    
    name = "quality"
    version = "1.0.0"
    description = "Audio quality assessment: SNR, clipping, hum detection"
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 22050
        self.hop_length = 512
        self.frame_length = 2048
        self.clip_threshold = 0.99  # Threshold for clipping detection
        self.hum_frequencies = [50, 60, 100, 120, 150, 180]  # Common hum frequencies
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract quality features from audio file
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with quality features
        """
        try:
            self.logger.info(f"Starting quality extraction for {input_uri}")
            
            # Load audio
            audio, sr = librosa.load(input_uri, sr=self.sample_rate)
            
            # Extract quality features with timing
            features, processing_time = self._time_execution(self._extract_quality_features, audio, sr)
            
            self.logger.info(f"Quality extraction completed successfully in {processing_time:.3f}s")
            
            return self._create_result(
                success=True,
                payload=features,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Quality extraction failed: {e}")
            return self._create_result(
                success=False,
                error=str(e)
            )
    
    def _extract_quality_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract quality features from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of quality features
        """
        features = {}
        
        # Basic audio properties
        features["duration_seconds"] = len(audio) / sr
        features["sample_rate"] = sr
        features["num_samples"] = len(audio)
        
        # Amplitude statistics
        features["peak_amplitude"] = float(np.max(np.abs(audio)))
        features["rms_amplitude"] = float(np.sqrt(np.mean(audio ** 2)))
        features["dynamic_range"] = features["peak_amplitude"] / (features["rms_amplitude"] + 1e-10)
        
        # Clipping detection
        clip_features = self._detect_clipping(audio)
        features.update(clip_features)
        
        # SNR estimation
        snr_features = self._estimate_snr(audio, sr)
        features.update(snr_features)
        
        # Hum detection
        hum_features = self._detect_hum(audio, sr)
        features.update(hum_features)
        
        # Distortion detection
        distortion_features = self._detect_distortion(audio, sr)
        features.update(distortion_features)
        
        # Spectral quality metrics
        spectral_quality = self._assess_spectral_quality(audio, sr)
        features.update(spectral_quality)
        
        # Overall quality score
        features["overall_quality_score"] = self._calculate_overall_quality(features)
        
        return features
    
    def _detect_clipping(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Detect audio clipping
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with clipping features
        """
        # Count clipped samples
        clipped_samples = np.sum(np.abs(audio) >= self.clip_threshold)
        total_samples = len(audio)
        
        clip_fraction = clipped_samples / total_samples if total_samples > 0 else 0.0
        
        # Find clipping regions
        clipping_mask = np.abs(audio) >= self.clip_threshold
        clipping_regions = self._find_clipping_regions(clipping_mask)
        
        return {
            "clip_fraction": float(clip_fraction),
            "clip_count": int(clipped_samples),
            "clip_regions_count": len(clipping_regions),
            "max_clip_duration": float(np.max([region[1] - region[0] for region in clipping_regions]) / self.sample_rate) if clipping_regions else 0.0,
            "is_clipped": bool(clip_fraction > 0.001)  # 0.1% threshold
        }
    
    def _find_clipping_regions(self, clipping_mask: np.ndarray) -> list:
        """
        Find continuous clipping regions
        
        Args:
            clipping_mask: Boolean array indicating clipped samples
            
        Returns:
            List of (start, end) tuples for clipping regions
        """
        regions = []
        in_region = False
        start = 0
        
        for i, is_clipped in enumerate(clipping_mask):
            if is_clipped and not in_region:
                start = i
                in_region = True
            elif not is_clipped and in_region:
                regions.append((start, i))
                in_region = False
        
        # Handle case where clipping continues to end
        if in_region:
            regions.append((start, len(clipping_mask)))
        
        return regions
    
    def _estimate_snr(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Estimate Signal-to-Noise Ratio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with SNR features
        """
        try:
            # Method 1: Spectral subtraction approach
            stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=self.frame_length)
            magnitude = np.abs(stft)
            
            # Estimate noise floor from quietest 10% of frames
            frame_energies = np.sum(magnitude ** 2, axis=0)
            noise_threshold = np.percentile(frame_energies, 10)
            noise_frames = frame_energies <= noise_threshold
            
            if np.any(noise_frames):
                noise_spectrum = np.mean(magnitude[:, noise_frames], axis=1)
                signal_spectrum = np.mean(magnitude, axis=1)
                
                # Calculate SNR in dB
                signal_power = np.mean(signal_spectrum ** 2)
                noise_power = np.mean(noise_spectrum ** 2)
                
                if noise_power > 0:
                    snr_db = 10 * np.log10(signal_power / noise_power)
                else:
                    snr_db = 100.0  # Very high SNR
            else:
                snr_db = 0.0
            
            # Method 2: Time domain approach
            # Estimate noise from quiet segments
            window_size = int(0.1 * sr)  # 100ms windows
            window_energies = []
            
            for i in range(0, len(audio) - window_size, window_size):
                window = audio[i:i + window_size]
                energy = np.mean(window ** 2)
                window_energies.append(energy)
            
            if window_energies:
                # Use quietest 20% of windows as noise estimate
                noise_energy = np.percentile(window_energies, 20)
                signal_energy = np.mean(window_energies)
                
                if noise_energy > 0:
                    snr_db_td = 10 * np.log10(signal_energy / noise_energy)
                else:
                    snr_db_td = 100.0
            else:
                snr_db_td = 0.0
            
            # Use the more conservative estimate
            snr_db_final = min(snr_db, snr_db_td)
            
            return {
                "snr_estimate_db": float(snr_db_final),
                "snr_spectral_db": float(snr_db),
                "snr_temporal_db": float(snr_db_td),
                "noise_floor_estimate": float(noise_power) if 'noise_power' in locals() else 0.0
            }
            
        except Exception as e:
            self.logger.warning(f"SNR estimation failed: {e}")
            return {
                "snr_estimate_db": 0.0,
                "snr_spectral_db": 0.0,
                "snr_temporal_db": 0.0,
                "noise_floor_estimate": 0.0
            }
    
    def _detect_hum(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Detect electrical hum (50/60Hz and harmonics)
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with hum detection features
        """
        hum_features = {}
        
        try:
            # Get frequency spectrum
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(audio), 1/sr)
            magnitude = np.abs(fft)
            
            # Check for hum at specific frequencies
            hum_detected = {}
            hum_strength = {}
            
            for hum_freq in self.hum_frequencies:
                if hum_freq < sr / 2:  # Within Nyquist limit
                    # Find closest frequency bin
                    freq_idx = np.argmin(np.abs(freqs - hum_freq))
                    
                    # Check if there's a peak at this frequency
                    window_size = 5  # Check ±5 bins around the frequency
                    start_idx = max(0, freq_idx - window_size)
                    end_idx = min(len(magnitude), freq_idx + window_size + 1)
                    
                    local_magnitude = magnitude[start_idx:end_idx]
                    peak_magnitude = magnitude[freq_idx]
                    avg_magnitude = np.mean(local_magnitude)
                    
                    # Hum is detected if peak is significantly above local average
                    if peak_magnitude > 2 * avg_magnitude:
                        hum_detected[f"hum_{hum_freq}hz"] = True
                        hum_strength[f"hum_strength_{hum_freq}hz"] = float(peak_magnitude / avg_magnitude)
                    else:
                        hum_detected[f"hum_{hum_freq}hz"] = False
                        hum_strength[f"hum_strength_{hum_freq}hz"] = 0.0
            
            hum_features.update(hum_detected)
            hum_features.update(hum_strength)
            
            # Overall hum detection
            any_hum = any(hum_detected.values())
            hum_features["hum_detected"] = bool(any_hum)
            hum_features["hum_count"] = sum(hum_detected.values())
            
        except Exception as e:
            self.logger.warning(f"Hum detection failed: {e}")
            hum_features = {
                "hum_detected": False,
                "hum_count": 0
            }
            for hum_freq in self.hum_frequencies:
                hum_features[f"hum_{hum_freq}hz"] = False
                hum_features[f"hum_strength_{hum_freq}hz"] = 0.0
        
        return hum_features
    
    def _detect_distortion(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Detect audio distortion
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with distortion features
        """
        try:
            # Harmonic distortion analysis
            stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=self.frame_length)
            magnitude = np.abs(stft)
            
            # Calculate total harmonic distortion (THD) approximation
            # This is a simplified approach
            fundamental_freq = 440.0  # A4 note as reference
            fundamental_bin = int(fundamental_freq * self.frame_length / sr)
            
            if fundamental_bin < len(magnitude) // 2:
                fundamental_magnitude = np.mean(magnitude[fundamental_bin, :])
                
                # Check harmonics
                harmonic_magnitudes = []
                for harmonic in range(2, 6):  # 2nd to 5th harmonic
                    harmonic_bin = fundamental_bin * harmonic
                    if harmonic_bin < len(magnitude) // 2:
                        harmonic_magnitude = np.mean(magnitude[harmonic_bin, :])
                        harmonic_magnitudes.append(harmonic_magnitude)
                
                if harmonic_magnitudes and fundamental_magnitude > 0:
                    thd = np.sqrt(np.sum(np.array(harmonic_magnitudes) ** 2)) / fundamental_magnitude
                else:
                    thd = 0.0
            else:
                thd = 0.0
            
            # Spectral flatness as distortion indicator
            spectral_flatness = librosa.feature.spectral_flatness(y=audio, hop_length=self.hop_length)[0]
            avg_spectral_flatness = np.mean(spectral_flatness)
            
            return {
                "total_harmonic_distortion": float(thd),
                "spectral_flatness_distortion": float(avg_spectral_flatness),
                "distortion_detected": bool(thd > 0.1 or avg_spectral_flatness > 0.5)
            }
            
        except Exception as e:
            self.logger.warning(f"Distortion detection failed: {e}")
            return {
                "total_harmonic_distortion": 0.0,
                "spectral_flatness_distortion": 0.0,
                "distortion_detected": False
            }
    
    def _assess_spectral_quality(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Assess spectral quality metrics
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with spectral quality features
        """
        try:
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            return {
                "spectral_centroid_mean": float(np.mean(spectral_centroid)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
                "zcr_mean": float(np.mean(zcr)),
                "spectral_quality_score": self._calculate_spectral_quality_score(
                    spectral_centroid, spectral_rolloff, spectral_bandwidth, zcr
                )
            }
            
        except Exception as e:
            self.logger.warning(f"Spectral quality assessment failed: {e}")
            return {
                "spectral_centroid_mean": 0.0,
                "spectral_rolloff_mean": 0.0,
                "spectral_bandwidth_mean": 0.0,
                "zcr_mean": 0.0,
                "spectral_quality_score": 0.0
            }
    
    def _calculate_spectral_quality_score(self, centroid: np.ndarray, rolloff: np.ndarray, 
                                        bandwidth: np.ndarray, zcr: np.ndarray) -> float:
        """
        Calculate overall spectral quality score
        
        Args:
            centroid: Spectral centroid array
            rolloff: Spectral rolloff array
            bandwidth: Spectral bandwidth array
            zcr: Zero crossing rate array
            
        Returns:
            Spectral quality score (0-1)
        """
        try:
            # Normalize metrics to [0, 1] range
            centroid_norm = np.mean(centroid) / (self.sample_rate / 2)
            rolloff_norm = np.mean(rolloff) / (self.sample_rate / 2)
            bandwidth_norm = np.mean(bandwidth) / (self.sample_rate / 2)
            zcr_norm = np.mean(zcr)
            
            # Combine metrics (higher is better for most)
            quality_score = (centroid_norm + rolloff_norm + bandwidth_norm + (1 - zcr_norm)) / 4
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception:
            return 0.0
    
    def _calculate_overall_quality(self, features: Dict[str, Any]) -> float:
        """
        Calculate overall quality score
        
        Args:
            features: Dictionary of all quality features
            
        Returns:
            Overall quality score (0-1)
        """
        try:
            # Weight different quality aspects
            weights = {
                'snr': 0.3,
                'clipping': 0.25,
                'hum': 0.2,
                'distortion': 0.15,
                'spectral': 0.1
            }
            
            # SNR score (0-1)
            snr_db = features.get('snr_estimate_db', 0)
            snr_score = min(snr_db / 40.0, 1.0)  # 40dB = perfect score
            
            # Clipping score (0-1, lower clipping is better)
            clip_fraction = features.get('clip_fraction', 1.0)
            clipping_score = max(0.0, 1.0 - clip_fraction * 100)  # Penalize clipping heavily
            
            # Hum score (0-1, no hum is better)
            hum_detected = features.get('hum_detected', True)
            hum_score = 0.0 if hum_detected else 1.0
            
            # Distortion score (0-1, lower distortion is better)
            distortion_detected = features.get('distortion_detected', True)
            distortion_score = 0.0 if distortion_detected else 1.0
            
            # Spectral quality score
            spectral_score = features.get('spectral_quality_score', 0.0)
            
            # Calculate weighted average
            overall_score = (
                weights['snr'] * snr_score +
                weights['clipping'] * clipping_score +
                weights['hum'] * hum_score +
                weights['distortion'] * distortion_score +
                weights['spectral'] * spectral_score
            )
            
            return float(np.clip(overall_score, 0.0, 1.0))
            
        except Exception:
            return 0.0


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python quality_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = QualityExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
