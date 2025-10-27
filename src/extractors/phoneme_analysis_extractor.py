"""
Phoneme Analysis Extractor for phonetic analysis
Extracts phoneme timeline and phoneme rate
"""

import numpy as np
import librosa
from typing import Dict, Any, List, Tuple, Optional
from src.core.base_extractor import BaseExtractor, ExtractorResult


class PhonemeAnalysisExtractor(BaseExtractor):
    """
    Phoneme Analysis Extractor for phonetic analysis
    Extracts phoneme timeline and phoneme rate
    """
    
    name = "phoneme_analysis"
    version = "1.0.0"
    description = "Phoneme analysis: timeline, rate, phonetic features"
    
    def __init__(self, device: str = "auto"):
        super().__init__()
        
        # Device detection with fallback
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        self.logger.info(f"Initialized {self.name} v{self.version} on {self.device}")
        self.sample_rate = 16000  # Standard for phoneme analysis
        self.hop_length = 512
        self.frame_length = 2048
        
        # Basic phoneme categories for rule-based analysis
        self.phoneme_categories = {
            "vowels": ["a", "e", "i", "o", "u"],
            "consonants": ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "y", "z"],
            "fricatives": ["f", "s", "sh", "th", "v", "z"],
            "stops": ["b", "d", "g", "k", "p", "t"],
            "nasals": ["m", "n", "ng"]
        }
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract phoneme analysis features from audio file
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with phoneme analysis features
        """
        try:
            self.logger.info(f"Starting phoneme analysis extraction for {input_uri}")
            
            # Load audio
            audio, sr = librosa.load(input_uri, sr=self.sample_rate)
            
            # Extract phoneme analysis features with timing
            features, processing_time = self._time_execution(self._extract_phoneme_features, audio, sr)
            
            self.logger.info(f"Phoneme analysis extraction completed successfully in {processing_time:.3f}s")
            
            return self._create_result(
                success=True,
                payload=features,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Phoneme analysis extraction failed: {e}")
            return self._create_result(
                success=False,
                error=str(e)
            )
    
    def _extract_phoneme_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract phoneme analysis features from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of phoneme analysis features
        """
        features = {,
                "device_used": self.device,
                "gpu_accelerated": self.device == "cuda"
            }
        
        # Extract phonetic features
        phonetic_features = self._extract_phonetic_features(audio, sr)
        
        # Estimate phoneme timeline (simplified approach)
        phoneme_timeline = self._estimate_phoneme_timeline(audio, sr, phonetic_features)
        features["phoneme_timeline"] = phoneme_timeline
        
        # Calculate phoneme rate
        phoneme_rate = self._calculate_phoneme_rate(phoneme_timeline, len(audio) / sr)
        features["phoneme_rate"] = phoneme_rate
        
        # Calculate phonetic statistics
        phonetic_stats = self._calculate_phonetic_statistics(phonetic_features)
        features.update(phonetic_stats)
        
        # Calculate speech rhythm features
        rhythm_features = self._calculate_speech_rhythm(audio, sr)
        features.update(rhythm_features)
        
        return features
    
    def _extract_phonetic_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract phonetic features from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of phonetic features
        """
        features = {,
                "device_used": self.device,
                "gpu_accelerated": self.device == "cuda"
            }
        
        # Pitch features
        f0 = librosa.yin(audio, fmin=50, fmax=2000, sr=sr, hop_length=self.hop_length)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 0:
            features["pitch_mean"] = float(np.mean(f0_clean))
            features["pitch_std"] = float(np.std(f0_clean))
            features["pitch_range"] = float(np.max(f0_clean) - np.min(f0_clean))
        else:
            features["pitch_mean"] = 0.0
            features["pitch_std"] = 0.0
            features["pitch_range"] = 0.0
        
        # Energy features
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        features["energy_mean"] = float(np.mean(rms))
        features["energy_std"] = float(np.std(rms))
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.hop_length)[0]
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
        features["spectral_centroid_std"] = float(np.std(spectral_centroid))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=self.hop_length)
        for i in range(min(5, mfcc.shape[0])):
            features[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
            features[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))
        
        # Formant-like features (simplified)
        formant_features = self._extract_simplified_formants(audio, sr)
        features.update(formant_features)
        
        return features
    
    def _extract_simplified_formants(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract simplified formant-like features
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with formant-like features
        """
        try:
            # Use spectral peaks as formant approximations
            stft = librosa.stft(audio, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            
            # Find spectral peaks
            freqs = librosa.fft_frequencies(sr=sr, n_fft=self.frame_length)
            
            # Calculate mean spectrum
            mean_spectrum = np.mean(magnitude, axis=1)
            
            # Find peaks in the spectrum
            from scipy.signal import find_peaks
            
            peaks, properties = find_peaks(mean_spectrum, height=np.max(mean_spectrum) * 0.1, distance=10)
            
            # Extract first few formant-like frequencies
            formant_features = {,
                "device_used": self.device,
                "gpu_accelerated": self.device == "cuda"
            }
            for i in range(1, 5):
                if i <= len(peaks):
                    formant_freq = freqs[peaks[i-1]]
                    formant_features[f"formant_f{i}"] = float(formant_freq)
                else:
                    formant_features[f"formant_f{i}"] = 0.0
            
            return formant_features
            
        except Exception as e:
            self.logger.warning(f"Formant extraction failed: {e}")
            return {
                "formant_f1": 0.0,
                "formant_f2": 0.0,
                "formant_f3": 0.0,
                "formant_f4": 0.0
            }
    
    def _estimate_phoneme_timeline(self, audio: np.ndarray, sr: int, phonetic_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Estimate phoneme timeline using acoustic features
        
        Args:
            audio: Audio array
            sr: Sample rate
            phonetic_features: Dictionary of phonetic features
            
        Returns:
            List of estimated phoneme segments
        """
        timeline = []
        
        # Use energy and spectral features to estimate phoneme boundaries
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
        
        # Find potential phoneme boundaries using energy changes
        energy_threshold = np.mean(rms) * 0.3
        zcr_threshold = np.mean(zcr) * 1.5
        
        # Segment audio based on energy and ZCR changes
        segments = []
        current_segment_start = 0
        current_phoneme_type = "silence"
        
        for i in range(1, len(rms)):
            time = i * self.hop_length / sr
            
            # Determine phoneme type based on acoustic features
            if rms[i] < energy_threshold:
                phoneme_type = "silence"
            elif zcr[i] > zcr_threshold:
                phoneme_type = "fricative"
            elif rms[i] > np.mean(rms) * 1.2:
                phoneme_type = "vowel"
            else:
                phoneme_type = "consonant"
            
            # Check for phoneme boundary
            if phoneme_type != current_phoneme_type:
                if current_segment_start < i:
                    segment_duration = (i - current_segment_start) * self.hop_length / sr
                    if segment_duration > 0.01:  # Minimum 10ms duration
                        segments.append({
                            "start": current_segment_start * self.hop_length / sr,
                            "end": i * self.hop_length / sr,
                            "duration": segment_duration,
                            "phoneme_type": current_phoneme_type,
                            "confidence": 0.5  # Low confidence for rule-based approach
                        })
                
                current_segment_start = i
                current_phoneme_type = phoneme_type
        
        # Add final segment
        if current_segment_start < len(rms):
            segment_duration = (len(rms) - current_segment_start) * self.hop_length / sr
            if segment_duration > 0.01:
                segments.append({
                    "start": current_segment_start * self.hop_length / sr,
                    "end": len(rms) * self.hop_length / sr,
                    "duration": segment_duration,
                    "phoneme_type": current_phoneme_type,
                    "confidence": 0.5
                })
        
        return segments
    
    def _calculate_phoneme_rate(self, phoneme_timeline: List[Dict[str, Any]], total_duration: float) -> Dict[str, float]:
        """
        Calculate phoneme rate metrics
        
        Args:
            phoneme_timeline: List of phoneme segments
            total_duration: Total audio duration
            
        Returns:
            Dictionary with phoneme rate metrics
        """
        if total_duration == 0:
            return {
                "phoneme_rate": 0.0,
                "phoneme_rate_std": 0.0,
                "phoneme_duration_mean": 0.0,
                "phoneme_duration_std": 0.0
            }
        
        # Count non-silence phonemes
        non_silence_phonemes = [p for p in phoneme_timeline if p["phoneme_type"] != "silence"]
        
        if len(non_silence_phonemes) == 0:
            return {
                "phoneme_rate": 0.0,
                "phoneme_rate_std": 0.0,
                "phoneme_duration_mean": 0.0,
                "phoneme_duration_std": 0.0
            }
        
        # Calculate phoneme rate (phonemes per second)
        phoneme_rate = len(non_silence_phonemes) / total_duration
        
        # Calculate phoneme durations
        phoneme_durations = [p["duration"] for p in non_silence_phonemes]
        phoneme_duration_mean = np.mean(phoneme_durations)
        phoneme_duration_std = np.std(phoneme_durations)
        
        # Calculate phoneme rate variability
        if len(non_silence_phonemes) > 1:
            # Calculate local phoneme rates
            local_rates = []
            for i in range(len(non_silence_phonemes) - 1):
                time_diff = non_silence_phonemes[i+1]["start"] - non_silence_phonemes[i]["start"]
                if time_diff > 0:
                    local_rates.append(1.0 / time_diff)
            
            phoneme_rate_std = np.std(local_rates) if local_rates else 0.0
        else:
            phoneme_rate_std = 0.0
        
        return {
            "phoneme_rate": float(phoneme_rate),
            "phoneme_rate_std": float(phoneme_rate_std),
            "phoneme_duration_mean": float(phoneme_duration_mean),
            "phoneme_duration_std": float(phoneme_duration_std)
        }
    
    def _calculate_phonetic_statistics(self, phonetic_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate phonetic statistics
        
        Args:
            phonetic_features: Dictionary of phonetic features
            
        Returns:
            Dictionary with phonetic statistics
        """
        stats = {}
        
        # Pitch statistics
        pitch_mean = phonetic_features.get("pitch_mean", 0.0)
        pitch_std = phonetic_features.get("pitch_std", 0.0)
        
        stats["pitch_variability"] = pitch_std / (pitch_mean + 1e-10)
        stats["pitch_stability"] = 1.0 / (1.0 + stats["pitch_variability"])
        
        # Energy statistics
        energy_mean = phonetic_features.get("energy_mean", 0.0)
        energy_std = phonetic_features.get("energy_std", 0.0)
        
        stats["energy_variability"] = energy_std / (energy_mean + 1e-10)
        stats["energy_stability"] = 1.0 / (1.0 + stats["energy_variability"])
        
        # Spectral statistics
        spectral_centroid_mean = phonetic_features.get("spectral_centroid_mean", 0.0)
        spectral_centroid_std = phonetic_features.get("spectral_centroid_std", 0.0)
        
        stats["spectral_variability"] = spectral_centroid_std / (spectral_centroid_mean + 1e-10)
        stats["spectral_stability"] = 1.0 / (1.0 + stats["spectral_variability"])
        
        # ZCR statistics
        zcr_mean = phonetic_features.get("zcr_mean", 0.0)
        zcr_std = phonetic_features.get("zcr_std", 0.0)
        
        stats["zcr_variability"] = zcr_std / (zcr_mean + 1e-10)
        stats["zcr_stability"] = 1.0 / (1.0 + stats["zcr_variability"])
        
        return stats
    
    def _calculate_speech_rhythm(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Calculate speech rhythm features
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with speech rhythm features
        """
        rhythm_features = {,
                "device_used": self.device,
                "gpu_accelerated": self.device == "cuda"
            }
        
        # Calculate tempo
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=self.hop_length)
        rhythm_features["tempo"] = float(tempo)
        rhythm_features["beat_count"] = len(beats)
        
        # Calculate rhythm regularity
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            rhythm_features["rhythm_regularity"] = 1.0 / (1.0 + np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-10))
        else:
            rhythm_features["rhythm_regularity"] = 0.0
        
        # Calculate speech rate (syllables per second approximation)
        # Use energy peaks as syllable approximation
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        
        # Find energy peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(rms, height=np.mean(rms) * 1.2, distance=5)
        
        speech_rate = len(peaks) / (len(audio) / sr) if len(audio) > 0 else 0.0
        rhythm_features["speech_rate"] = float(speech_rate)
        
        return rhythm_features


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
import torch
    
    if len(sys.argv) != 2:
        print("Usage: python phoneme_analysis_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = PhonemeAnalysisExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
