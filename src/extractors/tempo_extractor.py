"""
Tempo Extractor for tempo and rhythm analysis
Extracts tempo (BPM), onset count, and rhythm-related features
"""

import numpy as np
import librosa
from typing import Dict, Any, Tuple
from src.core.base_extractor import BaseExtractor, ExtractorResult


class TempoExtractor(BaseExtractor):
    """
    Tempo Extractor for tempo and rhythm analysis
    Extracts BPM, onset count, beat positions, and rhythm features
    """
    
    name = "tempo"
    version = "1.0.0"
    description = "Tempo and rhythm analysis: BPM, onset count, beat positions"
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 22050
        self.hop_length = 512
        self.frame_length = 2048
        self.tempo_min = 60.0  # Minimum BPM
        self.tempo_max = 200.0  # Maximum BPM
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract tempo features from audio file
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with tempo features
        """
        try:
            self.logger.info(f"Starting tempo extraction for {input_uri}")
            
            # Load audio
            audio, sr = librosa.load(input_uri, sr=self.sample_rate)
            
            # Extract tempo features
            features = self._extract_tempo_features(audio, sr)
            
            self.logger.info(f"Tempo extraction completed successfully")
            
            return ExtractorResult(
                name=self.name,
                version=self.version,
                success=True,
                payload=features
            )
            
        except Exception as e:
            self.logger.error(f"Tempo extraction failed: {e}")
            return ExtractorResult(
                name=self.name,
                version=self.version,
                success=False,
                error=str(e)
            )
    
    def _extract_tempo_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract tempo and rhythm features from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of tempo features
        """
        features = {}
        
        # Tempo estimation
        tempo, beats = librosa.beat.beat_track(
            y=audio,
            sr=sr,
            hop_length=self.hop_length,
            start_bpm=self.tempo_min,
            tightness=100
        )
        
        features["tempo_bpm"] = float(tempo)
        
        # Convert beat frames to time
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
        features["beat_count"] = len(beats)
        features["beat_times"] = beat_times.tolist()
        
        # Onset detection and analysis
        onset_frames = librosa.onset.onset_detect(
            y=audio,
            sr=sr,
            hop_length=self.hop_length,
            units='frames'
        )
        
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
        features["onset_count"] = len(onset_frames)
        features["onset_times"] = onset_times.tolist()
        
        # Onset strength
        onset_strength = librosa.onset.onset_strength(
            y=audio,
            sr=sr,
            hop_length=self.hop_length
        )
        
        features["onset_strength_mean"] = float(np.mean(onset_strength))
        features["onset_strength_std"] = float(np.std(onset_strength))
        features["onset_strength_max"] = float(np.max(onset_strength))
        features["onset_strength_min"] = float(np.min(onset_strength))
        
        # Rhythm analysis
        if len(beat_times) > 1:
            # Beat intervals
            beat_intervals = np.diff(beat_times)
            features["beat_interval_mean"] = float(np.mean(beat_intervals))
            features["beat_interval_std"] = float(np.std(beat_intervals))
            features["beat_interval_min"] = float(np.min(beat_intervals))
            features["beat_interval_max"] = float(np.max(beat_intervals))
            
            # Tempo variability
            tempo_variability = np.std(beat_intervals) / np.mean(beat_intervals) if np.mean(beat_intervals) > 0 else 0
            features["tempo_variability"] = float(tempo_variability)
            
            # Rhythm regularity
            features["rhythm_regularity"] = 1.0 / (1.0 + tempo_variability)
        else:
            features["beat_interval_mean"] = 0.0
            features["beat_interval_std"] = 0.0
            features["beat_interval_min"] = 0.0
            features["beat_interval_max"] = 0.0
            features["tempo_variability"] = 0.0
            features["rhythm_regularity"] = 0.0
        
        # Onset density analysis
        if len(onset_times) > 0:
            audio_duration = len(audio) / sr
            features["onset_density"] = len(onset_times) / audio_duration  # onsets per second
            
            # Onset intervals
            if len(onset_times) > 1:
                onset_intervals = np.diff(onset_times)
                features["onset_interval_mean"] = float(np.mean(onset_intervals))
                features["onset_interval_std"] = float(np.std(onset_intervals))
                features["onset_interval_min"] = float(np.min(onset_intervals))
                features["onset_interval_max"] = float(np.max(onset_intervals))
            else:
                features["onset_interval_mean"] = 0.0
                features["onset_interval_std"] = 0.0
                features["onset_interval_min"] = 0.0
                features["onset_interval_max"] = 0.0
        else:
            features["onset_density"] = 0.0
            features["onset_interval_mean"] = 0.0
            features["onset_interval_std"] = 0.0
            features["onset_interval_min"] = 0.0
            features["onset_interval_max"] = 0.0
        
        # Tempo classification
        features["tempo_class"] = self._classify_tempo(tempo)
        
        # Rhythm complexity
        features["rhythm_complexity"] = self._calculate_rhythm_complexity(onset_strength, beat_times, onset_times)
        
        # Syncopation analysis
        features["syncopation"] = self._calculate_syncopation(beat_times, onset_times)
        
        # Meter estimation
        meter_info = self._estimate_meter(audio, sr)
        features.update(meter_info)
        
        return features
    
    def _classify_tempo(self, tempo: float) -> str:
        """
        Classify tempo into categories
        
        Args:
            tempo: Tempo in BPM
            
        Returns:
            Tempo classification
        """
        if tempo < 60:
            return "very_slow"
        elif tempo < 80:
            return "slow"
        elif tempo < 100:
            return "moderate"
        elif tempo < 120:
            return "medium"
        elif tempo < 140:
            return "fast"
        elif tempo < 180:
            return "very_fast"
        else:
            return "extremely_fast"
    
    def _calculate_rhythm_complexity(self, onset_strength: np.ndarray, beat_times: np.ndarray, onset_times: np.ndarray) -> float:
        """
        Calculate rhythm complexity based on onset patterns
        
        Args:
            onset_strength: Onset strength array
            beat_times: Beat time positions
            onset_times: Onset time positions
            
        Returns:
            Rhythm complexity score
        """
        if len(onset_times) == 0:
            return 0.0
        
        # Calculate variability in onset strength
        strength_variability = np.std(onset_strength) / (np.mean(onset_strength) + 1e-10)
        
        # Calculate onset-to-beat ratio
        if len(beat_times) > 0:
            onset_beat_ratio = len(onset_times) / len(beat_times)
        else:
            onset_beat_ratio = 0.0
        
        # Combine metrics for complexity score
        complexity = (strength_variability + onset_beat_ratio) / 2.0
        
        return float(min(complexity, 1.0))  # Normalize to [0, 1]
    
    def _calculate_syncopation(self, beat_times: np.ndarray, onset_times: np.ndarray) -> float:
        """
        Calculate syncopation level
        
        Args:
            beat_times: Beat time positions
            onset_times: Onset time positions
            
        Returns:
            Syncopation score
        """
        if len(beat_times) < 2 or len(onset_times) == 0:
            return 0.0
        
        # Calculate average beat interval
        beat_interval = np.mean(np.diff(beat_times))
        
        # Count onsets that are not on beats (syncopated)
        syncopated_onsets = 0
        tolerance = beat_interval * 0.1  # 10% tolerance
        
        for onset_time in onset_times:
            # Check if onset is close to any beat
            distances = np.abs(beat_times - onset_time)
            min_distance = np.min(distances)
            
            if min_distance > tolerance:
                syncopated_onsets += 1
        
        syncopation = syncopated_onsets / len(onset_times) if len(onset_times) > 0 else 0.0
        
        return float(syncopation)
    
    def _estimate_meter(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Estimate musical meter (time signature)
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with meter information
        """
        meter_info = {}
        
        try:
            # Get onset strength
            onset_strength = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=self.hop_length)
            
            # Estimate tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, hop_length=self.hop_length)
            
            # Simple meter estimation based on tempo ranges
            if tempo < 80:
                meter_info["estimated_meter"] = "4/4"
                meter_info["beats_per_measure"] = 4
            elif tempo < 120:
                meter_info["estimated_meter"] = "4/4"
                meter_info["beats_per_measure"] = 4
            elif tempo < 160:
                meter_info["estimated_meter"] = "4/4"
                meter_info["beats_per_measure"] = 4
            else:
                meter_info["estimated_meter"] = "4/4"  # Default
                meter_info["beats_per_measure"] = 4
            
            # Calculate accent pattern
            accent_strength = np.mean(onset_strength)
            meter_info["accent_strength"] = float(accent_strength)
            
        except Exception as e:
            self.logger.warning(f"Meter estimation failed: {e}")
            meter_info["estimated_meter"] = "4/4"
            meter_info["beats_per_measure"] = 4
            meter_info["accent_strength"] = 0.0
        
        return meter_info


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python tempo_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = TempoExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
