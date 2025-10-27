"""
Music Analysis Extractor for musical analysis
Extracts key/mode, chord estimates, danceability, energy, and other musical descriptors
"""

import numpy as np
import librosa
from typing import Dict, Any, List, Tuple, Optional
from src.core.base_extractor import BaseExtractor, ExtractorResult


class MusicAnalysisExtractor(BaseExtractor):
    """
    Music Analysis Extractor for musical analysis
    Extracts key/mode, chord estimates, danceability, energy, and other musical descriptors
    """
    
    name = "music_analysis"
    version = "1.0.0"
    description = "Music analysis: key/mode, chords, danceability, energy"
    
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
        
        # Key and mode mapping
        self.key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.mode_names = ['major', 'minor']
        
        # Chord templates (simplified)
        self.chord_templates = {
            'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C major
            'Cm': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # C minor
            'C7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # C dominant 7
            'Cmaj7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # C major 7
            'Cm7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # C minor 7
        }
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract music analysis features from audio file
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with music analysis features
        """
        try:
            self.logger.info(f"Starting music analysis extraction for {input_uri}")
            
            # Load audio
            audio, sr = librosa.load(input_uri, sr=self.sample_rate)
            
            # Extract music analysis features with timing
            features, processing_time = self._time_execution(self._extract_music_features, audio, sr)
            
            self.logger.info(f"Music analysis extraction completed successfully in {processing_time:.3f}s")
            
            return self._create_result(
                success=True,
                payload=features,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Music analysis extraction failed: {e}")
            return self._create_result(
                success=False,
                error=str(e)
            )
    
    def _extract_music_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract music analysis features from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of music analysis features
        """
        features = {,
                "device_used": self.device,
                "gpu_accelerated": self.device == "cuda"
            }
        
        # Key and mode detection
        key_mode_features = self._extract_key_mode(audio, sr)
        features.update(key_mode_features)
        
        # Chord estimation
        chord_features = self._extract_chord_estimates(audio, sr)
        features.update(chord_features)
        
        # Danceability and energy
        danceability_energy = self._extract_danceability_energy(audio, sr)
        features.update(danceability_energy)
        
        # Musical descriptors
        musical_descriptors = self._extract_musical_descriptors(audio, sr)
        features.update(musical_descriptors)
        
        # Harmonic analysis
        harmonic_features = self._extract_harmonic_features(audio, sr)
        features.update(harmonic_features)
        
        return features
    
    def _extract_key_mode(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract key and mode information
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with key and mode features
        """
        try:
            # Extract chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=self.hop_length)
            
            # Calculate key profile correlation
            key_profiles = self._get_key_profiles()
            
            # Calculate correlation with each key
            key_correlations = {}
            for key_name, profile in key_profiles.items():
                correlation = np.corrcoef(chroma.mean(axis=1), profile)[0, 1]
                key_correlations[key_name] = correlation
            
            # Find best matching key
            best_key = max(key_correlations, key=key_correlations.get)
            key_confidence = key_correlations[best_key]
            
            # Determine mode (major/minor)
            mode = "major" if "m" not in best_key else "minor"
            
            return {
                "key": best_key,
                "mode": mode,
                "key_confidence": float(key_confidence),
                "key_correlations": {k: float(v) for k, v in key_correlations.items()}
            }
            
        except Exception as e:
            self.logger.warning(f"Key/mode extraction failed: {e}")
            return {
                "key": "C",
                "mode": "major",
                "key_confidence": 0.0,
                "key_correlations": {}
            }
    
    def _get_key_profiles(self) -> Dict[str, np.ndarray]:
        """
        Get key profiles for correlation analysis
        
        Returns:
            Dictionary mapping key names to profiles
        """
        # Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        key_profiles = {}
        
        # Generate profiles for all keys
        for i, key_name in enumerate(self.key_names):
            # Major key
            major_key_profile = np.roll(major_profile, i)
            key_profiles[key_name] = major_key_profile
            
            # Minor key
            minor_key_profile = np.roll(minor_profile, i)
            key_profiles[f"{key_name}m"] = minor_key_profile
        
        return key_profiles
    
    def _extract_chord_estimates(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract chord estimates
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with chord features
        """
        try:
            # Extract chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=self.hop_length)
            
            # Calculate chord probabilities for each frame
            chord_probs = []
            chord_sequence = []
            
            for i in range(chroma.shape[1]):
                frame_chroma = chroma[:, i]
                
                # Calculate correlation with chord templates
                chord_scores = {}
                for chord_name, template in self.chord_templates.items():
                    correlation = np.corrcoef(frame_chroma, template)[0, 1]
                    chord_scores[chord_name] = correlation
                
                # Find best chord
                best_chord = max(chord_scores, key=chord_scores.get)
                best_score = chord_scores[best_chord]
                
                chord_sequence.append(best_chord)
                chord_probs.append(chord_scores)
            
            # Calculate chord statistics
            chord_counts = {}
            for chord in chord_sequence:
                chord_counts[chord] = chord_counts.get(chord, 0) + 1
            
            # Find most common chord
            most_common_chord = max(chord_counts, key=chord_counts.get) if chord_counts else "C"
            chord_diversity = len(chord_counts) / len(chord_sequence) if chord_sequence else 0.0
            
            # Calculate chord transition rate
            chord_changes = sum(1 for i in range(1, len(chord_sequence)) if chord_sequence[i] != chord_sequence[i-1])
            chord_transition_rate = chord_changes / len(chord_sequence) if len(chord_sequence) > 1 else 0.0
            
            return {
                "chord_sequence": chord_sequence,
                "most_common_chord": most_common_chord,
                "chord_diversity": float(chord_diversity),
                "chord_transition_rate": float(chord_transition_rate),
                "chord_counts": chord_counts
            }
            
        except Exception as e:
            self.logger.warning(f"Chord estimation failed: {e}")
            return {
                "chord_sequence": [],
                "most_common_chord": "C",
                "chord_diversity": 0.0,
                "chord_transition_rate": 0.0,
                "chord_counts": {}
            }
    
    def _extract_danceability_energy(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract danceability and energy features
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with danceability and energy features
        """
        try:
            # Calculate tempo
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=self.hop_length)
            
            # Calculate energy
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
            energy = float(np.mean(rms))
            
            # Calculate spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.hop_length)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=self.hop_length)[0]
            
            # Calculate danceability based on tempo, energy, and spectral features
            # Danceability: higher tempo, steady rhythm, good energy
            tempo_score = min(tempo / 140.0, 1.0)  # Normalize to 0-1, optimal around 120-140 BPM
            energy_score = min(energy / 0.2, 1.0)  # Normalize to 0-1
            rhythm_score = self._calculate_rhythm_score(beats, len(audio) / sr)
            
            danceability = (tempo_score * 0.4 + energy_score * 0.3 + rhythm_score * 0.3)
            
            # Calculate energy features
            energy_mean = float(np.mean(rms))
            energy_std = float(np.std(rms))
            energy_max = float(np.max(rms))
            energy_min = float(np.min(rms))
            
            # Calculate spectral energy
            spectral_energy = float(np.mean(spectral_centroid))
            spectral_energy_std = float(np.std(spectral_centroid))
            
            return {
                "danceability": float(danceability),
                "energy": float(energy),
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                "energy_max": energy_max,
                "energy_min": energy_min,
                "spectral_energy": spectral_energy,
                "spectral_energy_std": spectral_energy_std,
                "tempo": float(tempo),
                "rhythm_score": float(rhythm_score)
            }
            
        except Exception as e:
            self.logger.warning(f"Danceability/energy extraction failed: {e}")
            return {
                "danceability": 0.0,
                "energy": 0.0,
                "energy_mean": 0.0,
                "energy_std": 0.0,
                "energy_max": 0.0,
                "energy_min": 0.0,
                "spectral_energy": 0.0,
                "spectral_energy_std": 0.0,
                "tempo": 0.0,
                "rhythm_score": 0.0
            }
    
    def _calculate_rhythm_score(self, beats: np.ndarray, duration: float) -> float:
        """
        Calculate rhythm score based on beat regularity
        
        Args:
            beats: Beat positions
            duration: Audio duration
            
        Returns:
            Rhythm score (0-1)
        """
        if len(beats) < 2 or duration == 0:
            return 0.0
        
        # Calculate beat intervals
        beat_intervals = np.diff(beats)
        
        # Calculate regularity (inverse of coefficient of variation)
        if np.mean(beat_intervals) > 0:
            cv = np.std(beat_intervals) / np.mean(beat_intervals)
            regularity = 1.0 / (1.0 + cv)
        else:
            regularity = 0.0
        
        # Calculate beat density
        beat_density = len(beats) / duration
        
        # Combine regularity and density
        rhythm_score = (regularity * 0.7 + min(beat_density / 2.0, 1.0) * 0.3)
        
        return float(rhythm_score)
    
    def _extract_musical_descriptors(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract additional musical descriptors
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with musical descriptors
        """
        try:
            # Calculate tempo
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=self.hop_length)
            
            # Calculate spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=self.hop_length)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=self.hop_length)[0]
            
            # Calculate zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
            
            # Calculate MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            
            # Calculate musical descriptors
            descriptors = {
                "tempo": float(tempo),
                "beat_count": len(beats),
                "spectral_centroid_mean": float(np.mean(spectral_centroid)),
                "spectral_centroid_std": float(np.std(spectral_centroid)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "spectral_rolloff_std": float(np.std(spectral_rolloff)),
                "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
                "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
                "zcr_mean": float(np.mean(zcr)),
                "zcr_std": float(np.std(zcr))
            }
            
            # Add MFCC statistics
            for i in range(min(5, mfcc.shape[0])):
                descriptors[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
                descriptors[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))
            
            # Calculate musical complexity
            complexity = self._calculate_musical_complexity(spectral_centroid, spectral_bandwidth, zcr)
            descriptors["musical_complexity"] = complexity
            
            return descriptors
            
        except Exception as e:
            self.logger.warning(f"Musical descriptors extraction failed: {e}")
            return {
                "tempo": 0.0,
                "beat_count": 0,
                "musical_complexity": 0.0
            }
    
    def _calculate_musical_complexity(self, spectral_centroid: np.ndarray, 
                                    spectral_bandwidth: np.ndarray, zcr: np.ndarray) -> float:
        """
        Calculate musical complexity score
        
        Args:
            spectral_centroid: Spectral centroid array
            spectral_bandwidth: Spectral bandwidth array
            zcr: Zero crossing rate array
            
        Returns:
            Musical complexity score (0-1)
        """
        try:
            # Calculate variability in spectral features
            centroid_var = np.std(spectral_centroid) / (np.mean(spectral_centroid) + 1e-10)
            bandwidth_var = np.std(spectral_bandwidth) / (np.mean(spectral_bandwidth) + 1e-10)
            zcr_var = np.std(zcr) / (np.mean(zcr) + 1e-10)
            
            # Combine variabilities
            complexity = (centroid_var + bandwidth_var + zcr_var) / 3.0
            
            # Normalize to 0-1 range
            complexity = min(complexity, 1.0)
            
            return float(complexity)
            
        except Exception:
            return 0.0
    
    def _extract_harmonic_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract harmonic features
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with harmonic features
        """
        try:
            # Separate harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            
            # Calculate harmonic ratio
            harmonic_energy = np.sum(y_harmonic ** 2)
            percussive_energy = np.sum(y_percussive ** 2)
            total_energy = harmonic_energy + percussive_energy
            
            harmonic_ratio = harmonic_energy / (total_energy + 1e-10)
            percussive_ratio = percussive_energy / (total_energy + 1e-10)
            
            # Calculate harmonic features
            harmonic_features = {
                "harmonic_ratio": float(harmonic_ratio),
                "percussive_ratio": float(percussive_ratio),
                "harmonic_energy": float(harmonic_energy),
                "percussive_energy": float(percussive_energy)
            ,
                "device_used": self.device,
                "gpu_accelerated": self.device == "cuda"
            }
            
            # Calculate harmonic complexity
            if harmonic_ratio > 0.1:  # Only if there's significant harmonic content
                harmonic_centroid = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr, hop_length=self.hop_length)[0]
                harmonic_bandwidth = librosa.feature.spectral_bandwidth(y=y_harmonic, sr=sr, hop_length=self.hop_length)[0]
                
                harmonic_features["harmonic_centroid_mean"] = float(np.mean(harmonic_centroid))
                harmonic_features["harmonic_centroid_std"] = float(np.std(harmonic_centroid))
                harmonic_features["harmonic_bandwidth_mean"] = float(np.mean(harmonic_bandwidth))
                harmonic_features["harmonic_bandwidth_std"] = float(np.std(harmonic_bandwidth))
            else:
                harmonic_features["harmonic_centroid_mean"] = 0.0
                harmonic_features["harmonic_centroid_std"] = 0.0
                harmonic_features["harmonic_bandwidth_mean"] = 0.0
                harmonic_features["harmonic_bandwidth_std"] = 0.0
            
            return harmonic_features
            
        except Exception as e:
            self.logger.warning(f"Harmonic features extraction failed: {e}")
            return {
                "harmonic_ratio": 0.0,
                "percussive_ratio": 1.0,
                "harmonic_energy": 0.0,
                "percussive_energy": 0.0,
                "harmonic_centroid_mean": 0.0,
                "harmonic_centroid_std": 0.0,
                "harmonic_bandwidth_mean": 0.0,
                "harmonic_bandwidth_std": 0.0
            }


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
import torch
    
    if len(sys.argv) != 2:
        print("Usage: python music_analysis_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = MusicAnalysisExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
