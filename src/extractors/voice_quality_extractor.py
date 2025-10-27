"""
Voice Quality Extractor for voice quality assessment
Extracts jitter, shimmer, HNR, formants, and other voice quality metrics
"""

import numpy as np
import librosa
from typing import Dict, Any, List, Tuple, Optional
from src.core.base_extractor import BaseExtractor, ExtractorResult


class VoiceQualityExtractor(BaseExtractor):
    """
    Voice Quality Extractor for voice quality assessment
    Extracts jitter, shimmer, HNR, formants, and other voice quality metrics
    """
    
    name = "voice_quality"
    version = "1.0.0"
    description = "Voice quality assessment: jitter, shimmer, HNR, formants"
    
    def __init__(self, device: str = "auto"):
        super().__init__()
        
        # Device detection with fallback
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        self.logger.info(f"Initialized {self.name} v{self.version} on {self.device}")
        self.sample_rate = 22050
        self.hop_length = 512
        self.frame_length = 2048
        self.fmin = 50.0  # Minimum frequency for pitch
        self.fmax = 2000.0  # Maximum frequency for pitch
        self.formant_order = 4  # Number of formants to extract
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract voice quality features from audio file
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with voice quality features
        """
        try:
            self.logger.info(f"Starting voice quality extraction for {input_uri}")
            
            # Load audio
            audio, sr = librosa.load(input_uri, sr=self.sample_rate)
            
            # Extract voice quality features with timing
            features, processing_time = self._time_execution(self._extract_voice_quality_features, audio, sr)
            
            self.logger.info(f"Voice quality extraction completed successfully in {processing_time:.3f}s")
            
            return self._create_result(
                success=True,
                payload=features,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Voice quality extraction failed: {e}")
            return self._create_result(
                success=False,
                error=str(e)
            )
    
    def _extract_voice_quality_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract voice quality features from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of voice quality features
        """
        features = {,
                "device_used": self.device,
                "gpu_accelerated": self.device == "cuda"
            }
        
        # Extract pitch for voice quality analysis
        f0 = self._extract_pitch(audio, sr)
        
        if len(f0) > 0 and np.any(f0 > 0):
            # Jitter (pitch period variability)
            jitter_features = self._calculate_jitter(f0, sr)
            features.update(jitter_features)
            
            # Shimmer (amplitude variability)
            shimmer_features = self._calculate_shimmer(audio, f0, sr)
            features.update(shimmer_features)
            
            # Harmonic-to-Noise Ratio (HNR)
            hnr_features = self._calculate_hnr(audio, f0, sr)
            features.update(hnr_features)
            
            # Formants
            formant_features = self._extract_formants(audio, sr)
            features.update(formant_features)
            
            # Voice quality indices
            quality_indices = self._calculate_voice_quality_indices(f0, jitter_features, shimmer_features, hnr_features)
            features.update(quality_indices)
            
        else:
            # No voice detected, return default values
            features = self._get_default_voice_quality_features()
        
        return features
    
    def _extract_pitch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract pitch using PYIN algorithm
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Pitch array
        """
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=sr,
                hop_length=self.hop_length,
                frame_length=self.frame_length
            )
            
            # Remove NaN values and keep only voiced frames
            f0_clean = f0[~np.isnan(f0)]
            voiced_clean = voiced_flag[~np.isnan(voiced_flag)]
            
            # Keep only voiced frames
            if len(f0_clean) > 0:
                f0_voiced = f0_clean[voiced_clean > 0.5]
                return f0_voiced
            else:
                return np.array([])
                
        except Exception as e:
            self.logger.warning(f"Pitch extraction failed: {e}")
            return np.array([])
    
    def _calculate_jitter(self, f0: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Calculate jitter (pitch period variability) metrics
        
        Args:
            f0: Pitch array
            sr: Sample rate
            
        Returns:
            Dictionary with jitter features
        """
        if len(f0) < 2:
            return {
                "jitter_local": 0.0,
                "jitter_rap": 0.0,
                "jitter_ppq5": 0.0,
                "jitter_ddp": 0.0
            }
        
        # Convert frequency to period
        periods = 1.0 / f0
        
        # Local jitter (relative period variability)
        period_diffs = np.abs(np.diff(periods))
        jitter_local = np.mean(period_diffs) / np.mean(periods) if np.mean(periods) > 0 else 0.0
        
        # RAP jitter (Relative Average Perturbation)
        if len(periods) >= 3:
            rap_periods = []
            for i in range(1, len(periods) - 1):
                rap = abs(periods[i] - (periods[i-1] + periods[i+1]) / 2)
                rap_periods.append(rap)
            jitter_rap = np.mean(rap_periods) / np.mean(periods) if np.mean(periods) > 0 else 0.0
        else:
            jitter_rap = 0.0
        
        # PPQ5 jitter (5-point Period Perturbation Quotient)
        if len(periods) >= 5:
            ppq5_periods = []
            for i in range(2, len(periods) - 2):
                ppq5 = abs(periods[i] - np.mean([periods[i-2], periods[i-1], periods[i+1], periods[i+2]]))
                ppq5_periods.append(ppq5)
            jitter_ppq5 = np.mean(ppq5_periods) / np.mean(periods) if np.mean(periods) > 0 else 0.0
        else:
            jitter_ppq5 = 0.0
        
        # DDP jitter (Difference of Differences of Periods)
        jitter_ddp = 3 * jitter_rap  # Approximation
        
        return {
            "jitter_local": float(jitter_local),
            "jitter_rap": float(jitter_rap),
            "jitter_ppq5": float(jitter_ppq5),
            "jitter_ddp": float(jitter_ddp)
        }
    
    def _calculate_shimmer(self, audio: np.ndarray, f0: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Calculate shimmer (amplitude variability) metrics
        
        Args:
            audio: Audio array
            f0: Pitch array
            sr: Sample rate
            
        Returns:
            Dictionary with shimmer features
        """
        if len(f0) == 0:
            return {
                "shimmer_local": 0.0,
                "shimmer_apq3": 0.0,
                "shimmer_apq5": 0.0,
                "shimmer_apq11": 0.0,
                "shimmer_dda": 0.0
            }
        
        # Extract amplitude envelope
        hop_length = self.hop_length
        frame_length = self.frame_length
        
        # Calculate RMS for each frame
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Align with pitch frames (assuming same hop length)
        if len(rms) >= len(f0):
            rms_aligned = rms[:len(f0)]
        else:
            rms_aligned = rms
        
        # Keep only voiced frames
        voiced_mask = f0 > 0
        if np.any(voiced_mask):
            rms_voiced = rms_aligned[voiced_mask]
        else:
            return {
                "shimmer_local": 0.0,
                "shimmer_apq3": 0.0,
                "shimmer_apq5": 0.0,
                "shimmer_apq11": 0.0,
                "shimmer_dda": 0.0
            }
        
        if len(rms_voiced) < 2:
            return {
                "shimmer_local": 0.0,
                "shimmer_apq3": 0.0,
                "shimmer_apq5": 0.0,
                "shimmer_apq11": 0.0,
                "shimmer_dda": 0.0
            }
        
        # Local shimmer (relative amplitude variability)
        amp_diffs = np.abs(np.diff(rms_voiced))
        shimmer_local = np.mean(amp_diffs) / np.mean(rms_voiced) if np.mean(rms_voiced) > 0 else 0.0
        
        # APQ3 shimmer (3-point Amplitude Perturbation Quotient)
        if len(rms_voiced) >= 3:
            apq3_amps = []
            for i in range(1, len(rms_voiced) - 1):
                apq3 = abs(rms_voiced[i] - np.mean([rms_voiced[i-1], rms_voiced[i+1]]))
                apq3_amps.append(apq3)
            shimmer_apq3 = np.mean(apq3_amps) / np.mean(rms_voiced) if np.mean(rms_voiced) > 0 else 0.0
        else:
            shimmer_apq3 = 0.0
        
        # APQ5 shimmer (5-point Amplitude Perturbation Quotient)
        if len(rms_voiced) >= 5:
            apq5_amps = []
            for i in range(2, len(rms_voiced) - 2):
                apq5 = abs(rms_voiced[i] - np.mean([rms_voiced[i-2], rms_voiced[i-1], rms_voiced[i+1], rms_voiced[i+2]]))
                apq5_amps.append(apq5)
            shimmer_apq5 = np.mean(apq5_amps) / np.mean(rms_voiced) if np.mean(rms_voiced) > 0 else 0.0
        else:
            shimmer_apq5 = 0.0
        
        # APQ11 shimmer (11-point Amplitude Perturbation Quotient)
        if len(rms_voiced) >= 11:
            apq11_amps = []
            for i in range(5, len(rms_voiced) - 5):
                apq11 = abs(rms_voiced[i] - np.mean(rms_voiced[i-5:i+6]))
                apq11_amps.append(apq11)
            shimmer_apq11 = np.mean(apq11_amps) / np.mean(rms_voiced) if np.mean(rms_voiced) > 0 else 0.0
        else:
            shimmer_apq11 = 0.0
        
        # DDA shimmer (Difference of Differences of Amplitudes)
        shimmer_dda = 3 * shimmer_apq3  # Approximation
        
        return {
            "shimmer_local": float(shimmer_local),
            "shimmer_apq3": float(shimmer_apq3),
            "shimmer_apq5": float(shimmer_apq5),
            "shimmer_apq11": float(shimmer_apq11),
            "shimmer_dda": float(shimmer_dda)
        }
    
    def _calculate_hnr(self, audio: np.ndarray, f0: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Calculate Harmonic-to-Noise Ratio (HNR)
        
        Args:
            audio: Audio array
            f0: Pitch array
            sr: Sample rate
            
        Returns:
            Dictionary with HNR features
        """
        if len(f0) == 0:
            return {
                "hnr_mean": 0.0,
                "hnr_std": 0.0,
                "hnr_min": 0.0,
                "hnr_max": 0.0
            }
        
        try:
            # Calculate HNR using autocorrelation method
            hnr_values = []
            
            # Process audio in frames
            frame_length = self.frame_length
            hop_length = self.hop_length
            
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                
                # Calculate autocorrelation
                autocorr = np.correlate(frame, frame, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                
                # Find fundamental period
                if len(f0) > i // hop_length:
                    current_f0 = f0[i // hop_length]
                    if current_f0 > 0:
                        period = int(sr / current_f0)
                        
                        if period < len(autocorr):
                            # HNR is the ratio of harmonic energy to noise energy
                            harmonic_energy = autocorr[period]
                            noise_energy = np.mean(autocorr[period+1:period*2]) if period*2 < len(autocorr) else 0
                            
                            if noise_energy > 0:
                                hnr = 10 * np.log10(harmonic_energy / noise_energy)
                                hnr_values.append(hnr)
            
            if hnr_values:
                return {
                    "hnr_mean": float(np.mean(hnr_values)),
                    "hnr_std": float(np.std(hnr_values)),
                    "hnr_min": float(np.min(hnr_values)),
                    "hnr_max": float(np.max(hnr_values))
                }
            else:
                return {
                    "hnr_mean": 0.0,
                    "hnr_std": 0.0,
                    "hnr_min": 0.0,
                    "hnr_max": 0.0
                }
                
        except Exception as e:
            self.logger.warning(f"HNR calculation failed: {e}")
            return {
                "hnr_mean": 0.0,
                "hnr_std": 0.0,
                "hnr_min": 0.0,
                "hnr_max": 0.0
            }
    
    def _extract_formants(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract formants using LPC analysis
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with formant features
        """
        try:
            from scipy.signal import lfilter
            
            # Pre-emphasize the signal
            pre_emphasis = 0.97
            emphasized_signal = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
            
            # Calculate LPC coefficients
            lpc_order = int(2 + sr / 1000)  # Rule of thumb: 2 + fs/1000
            
            # Use autocorrelation method for LPC
            autocorr = np.correlate(emphasized_signal, emphasized_signal, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Solve Yule-Walker equations
            R = autocorr[:lpc_order+1]
            A = np.zeros(lpc_order+1)
            A[0] = 1
            
            # Levinson-Durbin algorithm
            for i in range(1, lpc_order+1):
                if i == 1:
                    k = -R[1] / R[0]
                else:
                    k = -(R[i] + np.dot(A[1:i], R[i-1:0:-1])) / (R[0] + np.dot(A[1:i], R[1:i]))
                
                A[1:i+1] = A[1:i+1] + k * A[i-1:0:-1]
                A[i] = k
            
            # Find roots of LPC polynomial
            roots = np.roots(A)
            
            # Extract formants
            formants = []
            for root in roots:
                if np.iscomplex(root) and np.imag(root) > 0:
                    angle = np.angle(root)
                    frequency = angle * sr / (2 * np.pi)
                    if 0 < frequency < sr / 2:
                        formants.append(frequency)
            
            formants.sort()
            
            # Extract first 4 formants
            formant_features = {,
                "device_used": self.device,
                "gpu_accelerated": self.device == "cuda"
            }
            for i in range(1, 5):
                if i <= len(formants):
                    formant_features[f"formant_f{i}"] = float(formants[i-1])
                else:
                    formant_features[f"formant_f{i}"] = 0.0
            
            # Calculate formant bandwidths (simplified)
            for i in range(1, 5):
                if i <= len(formants):
                    # Approximate bandwidth calculation
                    bandwidth = 50.0 + 25.0 * i  # Rough approximation
                    formant_features[f"formant_bw{i}"] = float(bandwidth)
                else:
                    formant_features[f"formant_bw{i}"] = 0.0
            
            return formant_features
            
        except Exception as e:
            self.logger.warning(f"Formant extraction failed: {e}")
            return {
                "formant_f1": 0.0,
                "formant_f2": 0.0,
                "formant_f3": 0.0,
                "formant_f4": 0.0,
                "formant_bw1": 0.0,
                "formant_bw2": 0.0,
                "formant_bw3": 0.0,
                "formant_bw4": 0.0
            }
    
    def _calculate_voice_quality_indices(self, f0: np.ndarray, jitter_features: Dict, 
                                       shimmer_features: Dict, hnr_features: Dict) -> Dict[str, Any]:
        """
        Calculate overall voice quality indices
        
        Args:
            f0: Pitch array
            jitter_features: Jitter features
            shimmer_features: Shimmer features
            hnr_features: HNR features
            
        Returns:
            Dictionary with voice quality indices
        """
        # Voice Quality Index (VQI) - simplified version
        jitter_score = 1.0 - min(jitter_features.get("jitter_local", 0.0) * 100, 1.0)
        shimmer_score = 1.0 - min(shimmer_features.get("shimmer_local", 0.0) * 100, 1.0)
        hnr_score = min(hnr_features.get("hnr_mean", 0.0) / 20.0, 1.0)  # Normalize to 0-1
        
        vqi = (jitter_score + shimmer_score + hnr_score) / 3.0
        
        # Voice Activity Detection quality
        voiced_fraction = len(f0) / (len(f0) + 1e-10)  # Simplified
        
        return {
            "voice_quality_index": float(vqi),
            "jitter_score": float(jitter_score),
            "shimmer_score": float(shimmer_score),
            "hnr_score": float(hnr_score),
            "voiced_fraction": float(voiced_fraction),
            "voice_stability": float(1.0 - jitter_features.get("jitter_local", 0.0))
        }
    
    def _get_default_voice_quality_features(self) -> Dict[str, Any]:
        """
        Get default voice quality features when no voice is detected
        
        Returns:
            Dictionary with default features
        """
        return {
            "jitter_local": 0.0,
            "jitter_rap": 0.0,
            "jitter_ppq5": 0.0,
            "jitter_ddp": 0.0,
            "shimmer_local": 0.0,
            "shimmer_apq3": 0.0,
            "shimmer_apq5": 0.0,
            "shimmer_apq11": 0.0,
            "shimmer_dda": 0.0,
            "hnr_mean": 0.0,
            "hnr_std": 0.0,
            "hnr_min": 0.0,
            "hnr_max": 0.0,
            "formant_f1": 0.0,
            "formant_f2": 0.0,
            "formant_f3": 0.0,
            "formant_f4": 0.0,
            "formant_bw1": 0.0,
            "formant_bw2": 0.0,
            "formant_bw3": 0.0,
            "formant_bw4": 0.0,
            "voice_quality_index": 0.0,
            "jitter_score": 0.0,
            "shimmer_score": 0.0,
            "hnr_score": 0.0,
            "voiced_fraction": 0.0,
            "voice_stability": 0.0
        }


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
import torch
    
    if len(sys.argv) != 2:
        print("Usage: python voice_quality_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = VoiceQualityExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
