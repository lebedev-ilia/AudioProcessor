"""
Rhythmic Analysis Extractor for rhythmic analysis
Extracts beat positions, tempo variability, and onset strength time series
"""

import numpy as np
import librosa
from typing import Dict, Any, List, Tuple, Optional
from src.core.base_extractor import BaseExtractor, ExtractorResult


class RhythmicAnalysisExtractor(BaseExtractor):
    """
    Rhythmic Analysis Extractor for rhythmic analysis
    Extracts beat positions, tempo variability, and onset strength time series
    """
    
    name = "rhythmic_analysis"
    version = "1.0.0"
    description = "Rhythmic analysis: beat positions, tempo variability, onset strength"
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 22050
        self.hop_length = 512
        self.frame_length = 2048
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract rhythmic analysis features from audio file
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with rhythmic analysis features
        """
        try:
            self.logger.info(f"Starting rhythmic analysis extraction for {input_uri}")
            
            # Load audio
            audio, sr = librosa.load(input_uri, sr=self.sample_rate)
            
            # Extract rhythmic analysis features with timing
            features, processing_time = self._time_execution(self._extract_rhythmic_features, audio, sr)
            
            self.logger.info(f"Rhythmic analysis extraction completed successfully in {processing_time:.3f}s")
            
            return self._create_result(
                success=True,
                payload=features,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Rhythmic analysis extraction failed: {e}")
            return self._create_result(
                success=False,
                error=str(e)
            )
    
    def _extract_rhythmic_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract rhythmic analysis features from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of rhythmic analysis features
        """
        features = {}
        
        # Beat positions
        beat_positions = self._extract_beat_positions(audio, sr)
        features["beat_positions"] = beat_positions
        
        # Tempo variability
        tempo_variability = self._calculate_tempo_variability(audio, sr)
        features.update(tempo_variability)
        
        # Onset strength time series
        onset_strength_series = self._extract_onset_strength_time_series(audio, sr)
        features["onset_strength_time_series"] = onset_strength_series
        
        # Rhythmic patterns
        rhythmic_patterns = self._analyze_rhythmic_patterns(audio, sr)
        features.update(rhythmic_patterns)
        
        # Meter and time signature
        meter_analysis = self._analyze_meter(audio, sr)
        features.update(meter_analysis)
        
        # Syncopation analysis
        syncopation_analysis = self._analyze_syncopation(audio, sr)
        features.update(syncopation_analysis)
        
        return features
    
    def _extract_beat_positions(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """
        Extract beat positions from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            List of beat positions with metadata
        """
        try:
            # Calculate tempo and beat positions
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=self.hop_length)
            
            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
            
            # Calculate beat intervals
            beat_intervals = np.diff(beat_times) if len(beat_times) > 1 else []
            
            # Create beat position data
            beat_positions = []
            for i, beat_time in enumerate(beat_times):
                beat_data = {
                    "time": float(beat_time),
                    "beat_number": i + 1,
                    "tempo": float(tempo)
                }
                
                # Add interval information
                if i > 0:
                    beat_data["interval_from_previous"] = float(beat_intervals[i-1])
                else:
                    beat_data["interval_from_previous"] = 0.0
                
                # Add confidence (simplified)
                beat_data["confidence"] = 0.8  # Default confidence
                
                beat_positions.append(beat_data)
            
            return beat_positions
            
        except Exception as e:
            self.logger.warning(f"Beat position extraction failed: {e}")
            return []
    
    def _calculate_tempo_variability(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Calculate tempo variability
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with tempo variability features
        """
        try:
            # Calculate tempo and beat positions
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=self.hop_length)
            
            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
            
            # Calculate local tempos
            local_tempos = []
            if len(beat_times) > 1:
                for i in range(1, len(beat_times)):
                    interval = beat_times[i] - beat_times[i-1]
                    if interval > 0:
                        local_tempo = 60.0 / interval
                        local_tempos.append(local_tempo)
            
            if local_tempos:
                tempo_mean = float(np.mean(local_tempos))
                tempo_std = float(np.std(local_tempos))
                tempo_min = float(np.min(local_tempos))
                tempo_max = float(np.max(local_tempos))
                tempo_cv = tempo_std / (tempo_mean + 1e-10)  # Coefficient of variation
                
                # Calculate tempo stability
                tempo_stability = 1.0 / (1.0 + tempo_cv)
                
                # Calculate tempo acceleration/deceleration
                tempo_changes = np.diff(local_tempos)
                tempo_acceleration = float(np.mean(tempo_changes)) if len(tempo_changes) > 0 else 0.0
                tempo_acceleration_std = float(np.std(tempo_changes)) if len(tempo_changes) > 0 else 0.0
                
                return {
                    "tempo_mean": tempo_mean,
                    "tempo_std": tempo_std,
                    "tempo_min": tempo_min,
                    "tempo_max": tempo_max,
                    "tempo_cv": tempo_cv,
                    "tempo_stability": tempo_stability,
                    "tempo_acceleration": tempo_acceleration,
                    "tempo_acceleration_std": tempo_acceleration_std,
                    "global_tempo": float(tempo)
                }
            else:
                return {
                    "tempo_mean": float(tempo),
                    "tempo_std": 0.0,
                    "tempo_min": float(tempo),
                    "tempo_max": float(tempo),
                    "tempo_cv": 0.0,
                    "tempo_stability": 1.0,
                    "tempo_acceleration": 0.0,
                    "tempo_acceleration_std": 0.0,
                    "global_tempo": float(tempo)
                }
                
        except Exception as e:
            self.logger.warning(f"Tempo variability calculation failed: {e}")
            return {
                "tempo_mean": 0.0,
                "tempo_std": 0.0,
                "tempo_min": 0.0,
                "tempo_max": 0.0,
                "tempo_cv": 0.0,
                "tempo_stability": 0.0,
                "tempo_acceleration": 0.0,
                "tempo_acceleration_std": 0.0,
                "global_tempo": 0.0
            }
    
    def _extract_onset_strength_time_series(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract onset strength time series
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with onset strength time series
        """
        try:
            # Calculate onset strength
            onset_strength = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=self.hop_length)
            
            # Convert to time series
            times = librosa.frames_to_time(np.arange(len(onset_strength)), sr=sr, hop_length=self.hop_length)
            
            # Create time series data
            onset_strength_series = []
            for i, (time, strength) in enumerate(zip(times, onset_strength)):
                onset_strength_series.append({
                    "time": float(time),
                    "strength": float(strength)
                })
            
            # Calculate onset strength statistics
            onset_strength_mean = float(np.mean(onset_strength))
            onset_strength_std = float(np.std(onset_strength))
            onset_strength_min = float(np.min(onset_strength))
            onset_strength_max = float(np.max(onset_strength))
            onset_strength_median = float(np.median(onset_strength))
            
            # Calculate onset strength percentiles
            onset_strength_p25 = float(np.percentile(onset_strength, 25))
            onset_strength_p75 = float(np.percentile(onset_strength, 75))
            onset_strength_p90 = float(np.percentile(onset_strength, 90))
            
            # Calculate onset strength variability
            onset_strength_cv = onset_strength_std / (onset_strength_mean + 1e-10)
            
            return {
                "onset_strength_time_series": onset_strength_series,
                "onset_strength_mean": onset_strength_mean,
                "onset_strength_std": onset_strength_std,
                "onset_strength_min": onset_strength_min,
                "onset_strength_max": onset_strength_max,
                "onset_strength_median": onset_strength_median,
                "onset_strength_p25": onset_strength_p25,
                "onset_strength_p75": onset_strength_p75,
                "onset_strength_p90": onset_strength_p90,
                "onset_strength_cv": onset_strength_cv
            }
            
        except Exception as e:
            self.logger.warning(f"Onset strength time series extraction failed: {e}")
            return {
                "onset_strength_time_series": [],
                "onset_strength_mean": 0.0,
                "onset_strength_std": 0.0,
                "onset_strength_min": 0.0,
                "onset_strength_max": 0.0,
                "onset_strength_median": 0.0,
                "onset_strength_p25": 0.0,
                "onset_strength_p75": 0.0,
                "onset_strength_p90": 0.0,
                "onset_strength_cv": 0.0
            }
    
    def _analyze_rhythmic_patterns(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Analyze rhythmic patterns
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with rhythmic pattern analysis
        """
        try:
            # Calculate tempo and beat positions
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=self.hop_length)
            
            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
            
            # Calculate beat intervals
            beat_intervals = np.diff(beat_times) if len(beat_times) > 1 else []
            
            # Analyze rhythmic patterns
            patterns = {}
            
            if len(beat_intervals) > 0:
                # Calculate rhythm regularity
                rhythm_regularity = 1.0 / (1.0 + np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-10))
                patterns["rhythm_regularity"] = float(rhythm_regularity)
                
                # Calculate rhythm complexity
                rhythm_complexity = np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-10)
                patterns["rhythm_complexity"] = float(rhythm_complexity)
                
                # Calculate rhythm stability
                rhythm_stability = 1.0 / (1.0 + rhythm_complexity)
                patterns["rhythm_stability"] = float(rhythm_stability)
                
                # Analyze rhythm patterns (simplified)
                rhythm_patterns = self._detect_rhythm_patterns(beat_intervals)
                patterns["rhythm_patterns"] = rhythm_patterns
                
                # Calculate rhythm density
                rhythm_density = len(beat_intervals) / (len(audio) / sr) if len(audio) > 0 else 0.0
                patterns["rhythm_density"] = float(rhythm_density)
                
                # Calculate rhythm variation
                rhythm_variation = np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-10)
                patterns["rhythm_variation"] = float(rhythm_variation)
                
            else:
                patterns["rhythm_regularity"] = 0.0
                patterns["rhythm_complexity"] = 0.0
                patterns["rhythm_stability"] = 0.0
                patterns["rhythm_patterns"] = []
                patterns["rhythm_density"] = 0.0
                patterns["rhythm_variation"] = 0.0
            
            return patterns
            
        except Exception as e:
            self.logger.warning(f"Rhythmic pattern analysis failed: {e}")
            return {
                "rhythm_regularity": 0.0,
                "rhythm_complexity": 0.0,
                "rhythm_stability": 0.0,
                "rhythm_patterns": [],
                "rhythm_density": 0.0,
                "rhythm_variation": 0.0
            }
    
    def _detect_rhythm_patterns(self, beat_intervals: np.ndarray) -> List[str]:
        """
        Detect rhythm patterns in beat intervals
        
        Args:
            beat_intervals: Array of beat intervals
            
        Returns:
            List of detected rhythm patterns
        """
        try:
            patterns = []
            
            if len(beat_intervals) < 2:
                return patterns
            
            # Normalize intervals
            mean_interval = np.mean(beat_intervals)
            normalized_intervals = beat_intervals / (mean_interval + 1e-10)
            
            # Detect common rhythm patterns
            # Simple patterns based on interval ratios
            
            # Detect duple meter (1:1 ratio)
            duple_count = 0
            for interval in normalized_intervals:
                if 0.8 <= interval <= 1.2:  # Close to 1:1 ratio
                    duple_count += 1
            
            if duple_count / len(normalized_intervals) > 0.7:
                patterns.append("duple_meter")
            
            # Detect triple meter (1:2 ratio)
            triple_count = 0
            for i in range(0, len(normalized_intervals) - 1, 2):
                if i + 1 < len(normalized_intervals):
                    ratio = normalized_intervals[i] / (normalized_intervals[i + 1] + 1e-10)
                    if 1.5 <= ratio <= 2.5:  # Close to 1:2 ratio
                        triple_count += 1
            
            if triple_count > len(normalized_intervals) / 4:
                patterns.append("triple_meter")
            
            # Detect syncopation (irregular patterns)
            irregular_count = 0
            for interval in normalized_intervals:
                if interval < 0.7 or interval > 1.3:  # Significantly different from mean
                    irregular_count += 1
            
            if irregular_count / len(normalized_intervals) > 0.3:
                patterns.append("syncopated")
            
            # Detect steady rhythm
            if len(patterns) == 0:
                patterns.append("steady")
            
            return patterns
            
        except Exception as e:
            self.logger.warning(f"Rhythm pattern detection failed: {e}")
            return []
    
    def _analyze_meter(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Analyze meter and time signature
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with meter analysis
        """
        try:
            # Calculate tempo and beat positions
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=self.hop_length)
            
            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
            
            # Calculate beat intervals
            beat_intervals = np.diff(beat_times) if len(beat_times) > 1 else []
            
            meter_analysis = {}
            
            if len(beat_intervals) > 0:
                # Analyze meter based on beat intervals
                mean_interval = np.mean(beat_intervals)
                
                # Detect common time signatures
                # 4/4 time (most common)
                four_four_score = self._calculate_meter_score(beat_intervals, [1, 1, 1, 1])
                meter_analysis["four_four_score"] = four_four_score
                
                # 3/4 time (waltz)
                three_four_score = self._calculate_meter_score(beat_intervals, [1, 1, 1])
                meter_analysis["three_four_score"] = three_four_score
                
                # 2/4 time
                two_four_score = self._calculate_meter_score(beat_intervals, [1, 1])
                meter_analysis["two_four_score"] = two_four_score
                
                # 6/8 time
                six_eight_score = self._calculate_meter_score(beat_intervals, [1, 1, 1, 1, 1, 1])
                meter_analysis["six_eight_score"] = six_eight_score
                
                # Determine most likely meter
                meter_scores = {
                    "4/4": four_four_score,
                    "3/4": three_four_score,
                    "2/4": two_four_score,
                    "6/8": six_eight_score
                }
                
                best_meter = max(meter_scores, key=meter_scores.get)
                meter_analysis["meter"] = best_meter
                meter_analysis["meter_confidence"] = float(meter_scores[best_meter])
                
                # Calculate meter stability
                meter_stability = 1.0 / (1.0 + np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-10))
                meter_analysis["meter_stability"] = float(meter_stability)
                
            else:
                meter_analysis["four_four_score"] = 0.0
                meter_analysis["three_four_score"] = 0.0
                meter_analysis["two_four_score"] = 0.0
                meter_analysis["six_eight_score"] = 0.0
                meter_analysis["meter"] = "unknown"
                meter_analysis["meter_confidence"] = 0.0
                meter_analysis["meter_stability"] = 0.0
            
            return meter_analysis
            
        except Exception as e:
            self.logger.warning(f"Meter analysis failed: {e}")
            return {
                "four_four_score": 0.0,
                "three_four_score": 0.0,
                "two_four_score": 0.0,
                "six_eight_score": 0.0,
                "meter": "unknown",
                "meter_confidence": 0.0,
                "meter_stability": 0.0
            }
    
    def _calculate_meter_score(self, beat_intervals: np.ndarray, pattern: List[float]) -> float:
        """
        Calculate score for a specific meter pattern
        
        Args:
            beat_intervals: Array of beat intervals
            pattern: Pattern to match (e.g., [1, 1, 1, 1] for 4/4)
            
        Returns:
            Score for the pattern (0-1)
        """
        try:
            if len(beat_intervals) < len(pattern):
                return 0.0
            
            # Normalize intervals
            mean_interval = np.mean(beat_intervals)
            normalized_intervals = beat_intervals / (mean_interval + 1e-10)
            
            # Calculate pattern matching score
            total_score = 0.0
            pattern_count = 0
            
            for i in range(0, len(normalized_intervals) - len(pattern) + 1, len(pattern)):
                pattern_intervals = normalized_intervals[i:i + len(pattern)]
                
                # Calculate similarity to pattern
                pattern_similarity = 0.0
                for j, (interval, expected) in enumerate(zip(pattern_intervals, pattern)):
                    if j < len(pattern_intervals):
                        similarity = 1.0 - abs(interval - expected) / (expected + 1e-10)
                        pattern_similarity += max(0.0, similarity)
                
                pattern_similarity /= len(pattern)
                total_score += pattern_similarity
                pattern_count += 1
            
            return total_score / (pattern_count + 1e-10)
            
        except Exception as e:
            self.logger.warning(f"Meter score calculation failed: {e}")
            return 0.0
    
    def _analyze_syncopation(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Analyze syncopation in the audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with syncopation analysis
        """
        try:
            # Calculate tempo and beat positions
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=self.hop_length)
            
            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
            
            # Calculate onset strength
            onset_strength = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=self.hop_length)
            onset_times = librosa.frames_to_time(np.arange(len(onset_strength)), sr=sr, hop_length=self.hop_length)
            
            # Find strong onsets
            onset_threshold = np.mean(onset_strength) + np.std(onset_strength)
            strong_onsets = onset_times[onset_strength > onset_threshold]
            
            # Calculate syncopation
            syncopation_score = 0.0
            syncopation_count = 0
            
            if len(beat_times) > 1 and len(strong_onsets) > 0:
                # Calculate beat intervals
                beat_intervals = np.diff(beat_times)
                mean_beat_interval = np.mean(beat_intervals)
                
                # Check for onsets between beats (syncopation)
                for onset_time in strong_onsets:
                    # Find the closest beat
                    beat_distances = np.abs(beat_times - onset_time)
                    closest_beat_idx = np.argmin(beat_distances)
                    closest_beat_time = beat_times[closest_beat_idx]
                    
                    # Check if onset is between beats
                    if closest_beat_idx > 0:
                        prev_beat_time = beat_times[closest_beat_idx - 1]
                        beat_interval = closest_beat_time - prev_beat_time
                        
                        # Check if onset is in the second half of the beat interval (syncopation)
                        if onset_time > prev_beat_time + beat_interval * 0.5:
                            syncopation_count += 1
                
                # Calculate syncopation score
                syncopation_score = syncopation_count / len(strong_onsets) if len(strong_onsets) > 0 else 0.0
            
            # Calculate rhythm complexity (related to syncopation)
            rhythm_complexity = 0.0
            if len(beat_times) > 1:
                beat_intervals = np.diff(beat_times)
                rhythm_complexity = np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-10)
            
            return {
                "syncopation_score": float(syncopation_score),
                "syncopation_count": syncopation_count,
                "rhythm_complexity": float(rhythm_complexity),
                "strong_onsets_count": len(strong_onsets)
            }
            
        except Exception as e:
            self.logger.warning(f"Syncopation analysis failed: {e}")
            return {
                "syncopation_score": 0.0,
                "syncopation_count": 0,
                "rhythm_complexity": 0.0,
                "strong_onsets_count": 0
            }


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python rhythmic_analysis_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = RhythmicAnalysisExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
