"""
Source Separation Extractor for audio source separation
Extracts stems, vocal fraction, and instrument probabilities
"""

import numpy as np
import librosa
from typing import Dict, Any, List, Tuple, Optional
from src.core.base_extractor import BaseExtractor, ExtractorResult


class SourceSeparationExtractor(BaseExtractor):
    """
    Source Separation Extractor for audio source separation
    Extracts stems, vocal fraction, and instrument probabilities
    """
    
    name = "source_separation"
    version = "1.0.0"
    description = "Source separation: stems, vocal fraction, instrument probabilities"
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 22050
        self.hop_length = 512
        self.frame_length = 2048
        
        # Instrument categories
        self.instrument_categories = {
            "vocal": ["voice", "singing", "speech"],
            "drums": ["kick", "snare", "hihat", "cymbal", "percussion"],
            "bass": ["bass", "low_frequency"],
            "guitar": ["guitar", "string"],
            "piano": ["piano", "keyboard"],
            "brass": ["trumpet", "saxophone", "horn"],
            "strings": ["violin", "cello", "viola", "string_section"]
        }
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract source separation features from audio file
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with source separation features
        """
        try:
            self.logger.info(f"Starting source separation extraction for {input_uri}")
            
            # Load audio
            audio, sr = librosa.load(input_uri, sr=self.sample_rate)
            
            # Extract source separation features with timing
            features, processing_time = self._time_execution(self._extract_source_separation_features, audio, sr)
            
            self.logger.info(f"Source separation extraction completed successfully in {processing_time:.3f}s")
            
            return self._create_result(
                success=True,
                payload=features,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Source separation extraction failed: {e}")
            return self._create_result(
                success=False,
                error=str(e)
            )
    
    def _extract_source_separation_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract source separation features from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of source separation features
        """
        features = {}
        
        # Harmonic-percussive separation
        harmonic_percussive = self._extract_harmonic_percussive_separation(audio)
        features.update(harmonic_percussive)
        
        # Vocal fraction estimation
        vocal_fraction = self._estimate_vocal_fraction(audio, sr)
        features["vocal_fraction"] = vocal_fraction
        
        # Instrument probabilities
        instrument_probs = self._estimate_instrument_probabilities(audio, sr)
        features["instrument_probs"] = instrument_probs
        
        # Stem analysis
        stem_analysis = self._analyze_stems(audio, sr)
        features.update(stem_analysis)
        
        # Source separation quality
        separation_quality = self._assess_separation_quality(audio, sr)
        features.update(separation_quality)
        
        return features
    
    def _extract_harmonic_percussive_separation(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract harmonic-percussive separation
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with harmonic-percussive features
        """
        try:
            # Separate harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            
            # Calculate energy ratios
            harmonic_energy = np.sum(y_harmonic ** 2)
            percussive_energy = np.sum(y_percussive ** 2)
            total_energy = harmonic_energy + percussive_energy
            
            harmonic_ratio = harmonic_energy / (total_energy + 1e-10)
            percussive_ratio = percussive_energy / (total_energy + 1e-10)
            
            # Calculate RMS for each component
            harmonic_rms = float(np.sqrt(np.mean(y_harmonic ** 2)))
            percussive_rms = float(np.sqrt(np.mean(y_percussive ** 2)))
            
            # Calculate spectral characteristics
            harmonic_centroid = librosa.feature.spectral_centroid(y=y_harmonic, sr=self.sample_rate)[0]
            percussive_centroid = librosa.feature.spectral_centroid(y=y_percussive, sr=self.sample_rate)[0]
            
            return {
                "harmonic_ratio": float(harmonic_ratio),
                "percussive_ratio": float(percussive_ratio),
                "harmonic_rms": harmonic_rms,
                "percussive_rms": percussive_rms,
                "harmonic_centroid_mean": float(np.mean(harmonic_centroid)),
                "percussive_centroid_mean": float(np.mean(percussive_centroid)),
                "harmonic_energy": float(harmonic_energy),
                "percussive_energy": float(percussive_energy)
            }
            
        except Exception as e:
            self.logger.warning(f"Harmonic-percussive separation failed: {e}")
            return {
                "harmonic_ratio": 0.5,
                "percussive_ratio": 0.5,
                "harmonic_rms": 0.0,
                "percussive_rms": 0.0,
                "harmonic_centroid_mean": 0.0,
                "percussive_centroid_mean": 0.0,
                "harmonic_energy": 0.0,
                "percussive_energy": 0.0
            }
    
    def _estimate_vocal_fraction(self, audio: np.ndarray, sr: int) -> float:
        """
        Estimate vocal fraction in the audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Estimated vocal fraction (0-1)
        """
        try:
            # Extract features that help identify vocals
            # 1. Pitch detection (vocals typically have clear pitch)
            f0 = librosa.yin(audio, fmin=80, fmax=400, sr=sr, hop_length=self.hop_length)
            f0_clean = f0[~np.isnan(f0)]
            
            # 2. Spectral centroid (vocals typically in mid-frequency range)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.hop_length)[0]
            
            # 3. Zero crossing rate (vocals typically have moderate ZCR)
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
            
            # 4. MFCC features (vocals have characteristic MFCC patterns)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            
            # Calculate vocal indicators
            vocal_indicators = []
            
            # Pitch-based indicator
            if len(f0_clean) > 0:
                pitch_stability = 1.0 / (1.0 + np.std(f0_clean) / (np.mean(f0_clean) + 1e-10))
                vocal_indicators.append(pitch_stability)
            else:
                vocal_indicators.append(0.0)
            
            # Spectral centroid indicator (vocals typically 1000-3000 Hz)
            centroid_mean = np.mean(spectral_centroid)
            if 1000 <= centroid_mean <= 3000:
                centroid_score = 1.0
            elif 800 <= centroid_mean <= 4000:
                centroid_score = 0.7
            else:
                centroid_score = 0.3
            vocal_indicators.append(centroid_score)
            
            # ZCR indicator (vocals typically 0.1-0.3)
            zcr_mean = np.mean(zcr)
            if 0.1 <= zcr_mean <= 0.3:
                zcr_score = 1.0
            elif 0.05 <= zcr_mean <= 0.4:
                zcr_score = 0.7
            else:
                zcr_score = 0.3
            vocal_indicators.append(zcr_score)
            
            # MFCC-based indicator (vocals have characteristic MFCC patterns)
            mfcc_1_mean = np.mean(mfcc[1])  # First MFCC coefficient
            mfcc_2_mean = np.mean(mfcc[2])  # Second MFCC coefficient
            
            # Simple heuristic: vocals typically have specific MFCC patterns
            mfcc_score = 0.5  # Default score
            if -20 <= mfcc_1_mean <= 0 and -10 <= mfcc_2_mean <= 10:
                mfcc_score = 0.8
            elif -30 <= mfcc_1_mean <= 10 and -20 <= mfcc_2_mean <= 20:
                mfcc_score = 0.6
            
            vocal_indicators.append(mfcc_score)
            
            # Combine indicators
            vocal_fraction = np.mean(vocal_indicators)
            
            return float(vocal_fraction)
            
        except Exception as e:
            self.logger.warning(f"Vocal fraction estimation failed: {e}")
            return 0.5  # Default to 50% if estimation fails
    
    def _estimate_instrument_probabilities(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Estimate instrument probabilities
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary mapping instrument categories to probabilities
        """
        try:
            # Extract spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.hop_length)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=self.hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=self.hop_length)[0]
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            
            # Extract rhythm features
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=self.hop_length)
            
            # Calculate instrument probabilities based on spectral characteristics
            instrument_probs = {}
            
            # Vocal probability (already calculated)
            vocal_fraction = self._estimate_vocal_fraction(audio, sr)
            instrument_probs["vocal"] = vocal_fraction
            
            # Drums probability (high percussive content, high tempo)
            percussive_ratio = self._extract_harmonic_percussive_separation(audio)["percussive_ratio"]
            tempo_score = min(tempo / 120.0, 1.0)  # Normalize tempo
            drums_prob = percussive_ratio * tempo_score * 0.8
            instrument_probs["drums"] = float(drums_prob)
            
            # Bass probability (low frequency content)
            low_freq_energy = self._calculate_low_frequency_energy(audio, sr)
            bass_prob = low_freq_energy * 0.7
            instrument_probs["bass"] = float(bass_prob)
            
            # Guitar probability (mid-frequency, harmonic content)
            mid_freq_energy = self._calculate_mid_frequency_energy(audio, sr)
            harmonic_ratio = self._extract_harmonic_percussive_separation(audio)["harmonic_ratio"]
            guitar_prob = mid_freq_energy * harmonic_ratio * 0.6
            instrument_probs["guitar"] = float(guitar_prob)
            
            # Piano probability (wide frequency range, harmonic content)
            wide_freq_score = self._calculate_frequency_range_score(audio, sr)
            piano_prob = wide_freq_score * harmonic_ratio * 0.5
            instrument_probs["piano"] = float(piano_prob)
            
            # Brass probability (high frequency, bright sound)
            high_freq_energy = self._calculate_high_frequency_energy(audio, sr)
            brightness_score = np.mean(spectral_centroid) / (sr / 2)
            brass_prob = high_freq_energy * brightness_score * 0.4
            instrument_probs["brass"] = float(brass_prob)
            
            # Strings probability (mid-high frequency, smooth sound)
            smoothness_score = 1.0 / (1.0 + np.std(spectral_centroid) / (np.mean(spectral_centroid) + 1e-10))
            strings_prob = mid_freq_energy * smoothness_score * 0.5
            instrument_probs["strings"] = float(strings_prob)
            
            # Normalize probabilities
            total_prob = sum(instrument_probs.values())
            if total_prob > 0:
                instrument_probs = {k: v / total_prob for k, v in instrument_probs.items()}
            
            return instrument_probs
            
        except Exception as e:
            self.logger.warning(f"Instrument probability estimation failed: {e}")
            return {category: 1.0 / len(self.instrument_categories) for category in self.instrument_categories.keys()}
    
    def _calculate_low_frequency_energy(self, audio: np.ndarray, sr: int) -> float:
        """Calculate low frequency energy (0-500 Hz)"""
        try:
            stft = librosa.stft(audio, hop_length=self.hop_length)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=self.frame_length)
            
            # Find frequency bins for low frequencies
            low_freq_mask = freqs <= 500
            low_freq_energy = np.sum(np.abs(stft[low_freq_mask]) ** 2)
            total_energy = np.sum(np.abs(stft) ** 2)
            
            return float(low_freq_energy / (total_energy + 1e-10))
        except:
            return 0.0
    
    def _calculate_mid_frequency_energy(self, audio: np.ndarray, sr: int) -> float:
        """Calculate mid frequency energy (500-4000 Hz)"""
        try:
            stft = librosa.stft(audio, hop_length=self.hop_length)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=self.frame_length)
            
            # Find frequency bins for mid frequencies
            mid_freq_mask = (freqs >= 500) & (freqs <= 4000)
            mid_freq_energy = np.sum(np.abs(stft[mid_freq_mask]) ** 2)
            total_energy = np.sum(np.abs(stft) ** 2)
            
            return float(mid_freq_energy / (total_energy + 1e-10))
        except:
            return 0.0
    
    def _calculate_high_frequency_energy(self, audio: np.ndarray, sr: int) -> float:
        """Calculate high frequency energy (4000+ Hz)"""
        try:
            stft = librosa.stft(audio, hop_length=self.hop_length)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=self.frame_length)
            
            # Find frequency bins for high frequencies
            high_freq_mask = freqs >= 4000
            high_freq_energy = np.sum(np.abs(stft[high_freq_mask]) ** 2)
            total_energy = np.sum(np.abs(stft) ** 2)
            
            return float(high_freq_energy / (total_energy + 1e-10))
        except:
            return 0.0
    
    def _calculate_frequency_range_score(self, audio: np.ndarray, sr: int) -> float:
        """Calculate frequency range score (how wide the frequency range is)"""
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.hop_length)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=self.hop_length)[0]
            
            # Wide frequency range indicates piano or full orchestra
            centroid_score = np.mean(spectral_centroid) / (sr / 2)
            bandwidth_score = np.mean(spectral_bandwidth) / (sr / 2)
            
            return float((centroid_score + bandwidth_score) / 2.0)
        except:
            return 0.0
    
    def _analyze_stems(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Analyze audio stems (simplified approach)
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with stem analysis
        """
        try:
            # Separate harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            
            # Analyze harmonic stem (melody, vocals, etc.)
            harmonic_analysis = self._analyze_harmonic_stem(y_harmonic, sr)
            
            # Analyze percussive stem (drums, percussion)
            percussive_analysis = self._analyze_percussive_stem(y_percussive, sr)
            
            # Combine analysis
            stem_analysis = {
                "harmonic_stem": harmonic_analysis,
                "percussive_stem": percussive_analysis,
                "separation_quality": self._assess_separation_quality(audio, sr)
            }
            
            return stem_analysis
            
        except Exception as e:
            self.logger.warning(f"Stem analysis failed: {e}")
            return {
                "harmonic_stem": {"energy": 0.0, "complexity": 0.0},
                "percussive_stem": {"energy": 0.0, "complexity": 0.0},
                "separation_quality": 0.0
            }
    
    def _analyze_harmonic_stem(self, harmonic_audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze harmonic stem"""
        try:
            # Calculate energy
            energy = float(np.sum(harmonic_audio ** 2))
            
            # Calculate complexity (spectral variation)
            spectral_centroid = librosa.feature.spectral_centroid(y=harmonic_audio, sr=sr, hop_length=self.hop_length)[0]
            complexity = float(np.std(spectral_centroid) / (np.mean(spectral_centroid) + 1e-10))
            
            # Calculate pitch content
            f0 = librosa.yin(harmonic_audio, fmin=80, fmax=400, sr=sr, hop_length=self.hop_length)
            f0_clean = f0[~np.isnan(f0)]
            pitch_content = len(f0_clean) / len(f0) if len(f0) > 0 else 0.0
            
            return {
                "energy": energy,
                "complexity": complexity,
                "pitch_content": float(pitch_content)
            }
        except:
            return {"energy": 0.0, "complexity": 0.0, "pitch_content": 0.0}
    
    def _analyze_percussive_stem(self, percussive_audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze percussive stem"""
        try:
            # Calculate energy
            energy = float(np.sum(percussive_audio ** 2))
            
            # Calculate rhythm complexity
            tempo, beats = librosa.beat.beat_track(y=percussive_audio, sr=sr, hop_length=self.hop_length)
            beat_count = len(beats)
            
            # Calculate percussive complexity
            zcr = librosa.feature.zero_crossing_rate(percussive_audio, hop_length=self.hop_length)[0]
            complexity = float(np.std(zcr) / (np.mean(zcr) + 1e-10))
            
            return {
                "energy": energy,
                "complexity": complexity,
                "beat_count": beat_count,
                "tempo": float(tempo)
            }
        except:
            return {"energy": 0.0, "complexity": 0.0, "beat_count": 0, "tempo": 0.0}
    
    def _assess_separation_quality(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Assess the quality of source separation
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with separation quality metrics
        """
        try:
            # Separate harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            
            # Calculate reconstruction error
            reconstructed = y_harmonic + y_percussive
            reconstruction_error = float(np.mean((audio - reconstructed) ** 2))
            
            # Calculate separation quality based on energy distribution
            harmonic_energy = np.sum(y_harmonic ** 2)
            percussive_energy = np.sum(y_percussive ** 2)
            total_energy = harmonic_energy + percussive_energy
            
            # Good separation should have clear energy distribution
            energy_balance = 1.0 - abs(harmonic_energy - percussive_energy) / (total_energy + 1e-10)
            
            # Calculate spectral separation quality
            harmonic_centroid = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr, hop_length=self.hop_length)[0]
            percussive_centroid = librosa.feature.spectral_centroid(y=y_percussive, sr=sr, hop_length=self.hop_length)[0]
            
            # Good separation should have different spectral characteristics
            spectral_separation = abs(np.mean(harmonic_centroid) - np.mean(percussive_centroid)) / (sr / 2)
            
            # Overall separation quality
            separation_quality = (energy_balance + spectral_separation) / 2.0
            
            return {
                "reconstruction_error": reconstruction_error,
                "energy_balance": float(energy_balance),
                "spectral_separation": float(spectral_separation),
                "separation_quality": float(separation_quality)
            }
            
        except Exception as e:
            self.logger.warning(f"Separation quality assessment failed: {e}")
            return {
                "reconstruction_error": 0.0,
                "energy_balance": 0.5,
                "spectral_separation": 0.0,
                "separation_quality": 0.0
            }


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python source_separation_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = SourceSeparationExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
