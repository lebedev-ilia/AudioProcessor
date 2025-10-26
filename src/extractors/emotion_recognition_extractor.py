"""
Emotion Recognition Extractor for emotion analysis from audio
Extracts emotion probabilities, valence/arousal, and emotion time series
"""

import numpy as np
import librosa
from typing import Dict, Any, List, Tuple, Optional
from src.core.base_extractor import BaseExtractor, ExtractorResult


class EmotionRecognitionExtractor(BaseExtractor):
    """
    Emotion Recognition Extractor for emotion analysis from audio
    Extracts emotion probabilities, valence/arousal, and emotion time series
    """
    
    name = "emotion_recognition"
    version = "1.0.0"
    description = "Emotion recognition: probabilities, valence/arousal, time series"
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 16000  # Standard for emotion recognition models
        self.hop_length = 512
        self.frame_length = 2048
        self.emotion_labels = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
        self._emotion_model = None
        self._valence_arousal_model = None
    
    def _load_models(self):
        """Load emotion recognition models if not already loaded"""
        if self._emotion_model is None:
            try:
                # Try to load a pre-trained emotion recognition model
                # For now, we'll use a rule-based approach as fallback
                self.logger.info("Using rule-based emotion recognition (no external models loaded)")
                self._emotion_model = "rule_based"
                self._valence_arousal_model = "rule_based"
                
            except Exception as e:
                self.logger.warning(f"Failed to load emotion models: {e}")
                self._emotion_model = "rule_based"
                self._valence_arousal_model = "rule_based"
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract emotion recognition features from audio file
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with emotion recognition features
        """
        try:
            self.logger.info(f"Starting emotion recognition extraction for {input_uri}")
            
            # Load models
            self._load_models()
            
            # Load audio
            audio, sr = librosa.load(input_uri, sr=self.sample_rate)
            
            # Extract emotion recognition features with timing
            features, processing_time = self._time_execution(self._extract_emotion_features, audio, sr)
            
            self.logger.info(f"Emotion recognition extraction completed successfully in {processing_time:.3f}s")
            
            return self._create_result(
                success=True,
                payload=features,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Emotion recognition extraction failed: {e}")
            return self._create_result(
                success=False,
                error=str(e)
            )
    
    def _extract_emotion_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract emotion recognition features from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of emotion recognition features
        """
        features = {}
        
        # Extract acoustic features for emotion analysis
        acoustic_features = self._extract_acoustic_features(audio, sr)
        
        # Emotion probabilities (rule-based approach)
        emotion_probs = self._predict_emotion_probabilities(acoustic_features)
        features["emotion_probs"] = emotion_probs
        
        # Valence and arousal
        valence_arousal = self._predict_valence_arousal(acoustic_features)
        features["emotion_valence"] = valence_arousal["valence"]
        features["emotion_arousal"] = valence_arousal["arousal"]
        
        # Emotion time series
        emotion_time_series = self._extract_emotion_time_series(audio, sr)
        features["emotion_time_series"] = emotion_time_series
        
        # Dominant emotion
        dominant_emotion = max(emotion_probs, key=emotion_probs.get)
        features["dominant_emotion"] = dominant_emotion
        features["dominant_emotion_confidence"] = emotion_probs[dominant_emotion]
        
        # Emotion stability
        emotion_stability = self._calculate_emotion_stability(emotion_time_series)
        features["emotion_stability"] = emotion_stability
        
        return features
    
    def _extract_acoustic_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract acoustic features relevant for emotion recognition
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of acoustic features
        """
        features = {}
        
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
        features["energy_range"] = float(np.max(rms) - np.min(rms))
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.hop_length)[0]
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
        features["spectral_centroid_std"] = float(np.std(spectral_centroid))
        
        # Zero crossing rate (speech rate indicator)
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))
        
        # MFCC features (first few coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=self.hop_length)
        for i in range(min(5, mfcc.shape[0])):
            features[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
            features[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))
        
        # Tempo and rhythm
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=self.hop_length)
        features["tempo"] = float(tempo)
        features["beat_count"] = len(beats)
        
        # Voice activity
        voiced_frames = np.sum(f0 > 0)
        total_frames = len(f0)
        features["voiced_fraction"] = voiced_frames / total_frames if total_frames > 0 else 0.0
        
        return features
    
    def _predict_emotion_probabilities(self, acoustic_features: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict emotion probabilities based on acoustic features (rule-based)
        
        Args:
            acoustic_features: Dictionary of acoustic features
            
        Returns:
            Dictionary mapping emotion labels to probabilities
        """
        # Initialize probabilities
        probs = {emotion: 0.0 for emotion in self.emotion_labels}
        
        # Rule-based emotion prediction based on acoustic features
        pitch_mean = acoustic_features.get("pitch_mean", 0.0)
        pitch_std = acoustic_features.get("pitch_std", 0.0)
        energy_mean = acoustic_features.get("energy_mean", 0.0)
        energy_std = acoustic_features.get("energy_std", 0.0)
        tempo = acoustic_features.get("tempo", 0.0)
        zcr_mean = acoustic_features.get("zcr_mean", 0.0)
        
        # Happy: high pitch, high energy, fast tempo
        if pitch_mean > 200 and energy_mean > 0.1 and tempo > 120:
            probs["happy"] += 0.4
        elif pitch_mean > 180 and energy_mean > 0.08:
            probs["happy"] += 0.2
        
        # Sad: low pitch, low energy, slow tempo
        if pitch_mean < 150 and energy_mean < 0.05 and tempo < 80:
            probs["sad"] += 0.4
        elif pitch_mean < 170 and energy_mean < 0.07:
            probs["sad"] += 0.2
        
        # Angry: high pitch variation, high energy, fast tempo
        if pitch_std > 50 and energy_mean > 0.12 and tempo > 100:
            probs["angry"] += 0.4
        elif pitch_std > 40 and energy_mean > 0.1:
            probs["angry"] += 0.2
        
        # Fearful: high pitch, high energy variation, fast tempo
        if pitch_mean > 220 and energy_std > 0.05 and tempo > 110:
            probs["fearful"] += 0.4
        elif pitch_mean > 200 and energy_std > 0.03:
            probs["fearful"] += 0.2
        
        # Surprised: high pitch, high energy, fast tempo
        if pitch_mean > 250 and energy_mean > 0.15 and tempo > 130:
            probs["surprised"] += 0.4
        elif pitch_mean > 230 and energy_mean > 0.12:
            probs["surprised"] += 0.2
        
        # Disgusted: low pitch, low energy, slow tempo
        if pitch_mean < 140 and energy_mean < 0.04 and tempo < 70:
            probs["disgusted"] += 0.4
        elif pitch_mean < 160 and energy_mean < 0.06:
            probs["disgusted"] += 0.2
        
        # Neutral: moderate values
        if (150 <= pitch_mean <= 200 and 
            0.06 <= energy_mean <= 0.1 and 
            80 <= tempo <= 120 and
            pitch_std < 30):
            probs["neutral"] += 0.4
        else:
            probs["neutral"] += 0.1  # Default neutral probability
        
        # Normalize probabilities
        total_prob = sum(probs.values())
        if total_prob > 0:
            probs = {emotion: prob / total_prob for emotion, prob in probs.items()}
        else:
            # If no rules matched, assign equal probabilities
            probs = {emotion: 1.0 / len(self.emotion_labels) for emotion in self.emotion_labels}
        
        return probs
    
    def _predict_valence_arousal(self, acoustic_features: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict valence and arousal based on acoustic features
        
        Args:
            acoustic_features: Dictionary of acoustic features
            
        Returns:
            Dictionary with valence and arousal values
        """
        pitch_mean = acoustic_features.get("pitch_mean", 0.0)
        energy_mean = acoustic_features.get("energy_mean", 0.0)
        tempo = acoustic_features.get("tempo", 0.0)
        spectral_centroid = acoustic_features.get("spectral_centroid_mean", 0.0)
        
        # Valence (positive/negative): based on pitch and spectral centroid
        # Higher pitch and brighter sound -> more positive valence
        pitch_valence = min(pitch_mean / 300.0, 1.0)  # Normalize to 0-1
        spectral_valence = min(spectral_centroid / 4000.0, 1.0)  # Normalize to 0-1
        valence = (pitch_valence + spectral_valence) / 2.0
        
        # Arousal (calm/excited): based on energy and tempo
        # Higher energy and faster tempo -> higher arousal
        energy_arousal = min(energy_mean / 0.2, 1.0)  # Normalize to 0-1
        tempo_arousal = min(tempo / 200.0, 1.0)  # Normalize to 0-1
        arousal = (energy_arousal + tempo_arousal) / 2.0
        
        return {
            "valence": float(valence),
            "arousal": float(arousal)
        }
    
    def _extract_emotion_time_series(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """
        Extract emotion time series by analyzing audio in segments
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            List of emotion data for each time segment
        """
        time_series = []
        
        # Analyze audio in 1-second segments
        segment_length = sr  # 1 second
        hop_length = sr // 2  # 0.5 second overlap
        
        for i in range(0, len(audio) - segment_length, hop_length):
            segment = audio[i:i + segment_length]
            time_start = i / sr
            
            # Extract features for this segment
            segment_features = self._extract_acoustic_features(segment, sr)
            
            # Predict emotions for this segment
            emotion_probs = self._predict_emotion_probabilities(segment_features)
            valence_arousal = self._predict_valence_arousal(segment_features)
            
            # Find dominant emotion
            dominant_emotion = max(emotion_probs, key=emotion_probs.get)
            
            time_series.append({
                "time": time_start,
                "emotion_probs": emotion_probs,
                "valence": valence_arousal["valence"],
                "arousal": valence_arousal["arousal"],
                "dominant_emotion": dominant_emotion,
                "confidence": emotion_probs[dominant_emotion]
            })
        
        return time_series
    
    def _calculate_emotion_stability(self, emotion_time_series: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate emotion stability metrics
        
        Args:
            emotion_time_series: List of emotion data over time
            
        Returns:
            Dictionary with emotion stability metrics
        """
        if len(emotion_time_series) < 2:
            return {
                "emotion_stability": 0.0,
                "valence_stability": 0.0,
                "arousal_stability": 0.0,
                "emotion_changes": 0
            }
        
        # Calculate emotion changes
        emotion_changes = 0
        prev_emotion = emotion_time_series[0]["dominant_emotion"]
        
        for data in emotion_time_series[1:]:
            if data["dominant_emotion"] != prev_emotion:
                emotion_changes += 1
            prev_emotion = data["dominant_emotion"]
        
        # Calculate stability (inverse of changes)
        emotion_stability = 1.0 - (emotion_changes / len(emotion_time_series))
        
        # Calculate valence and arousal stability
        valences = [data["valence"] for data in emotion_time_series]
        arousals = [data["arousal"] for data in emotion_time_series]
        
        valence_stability = 1.0 - (np.std(valences) / (np.mean(valences) + 1e-10))
        arousal_stability = 1.0 - (np.std(arousals) / (np.mean(arousals) + 1e-10))
        
        return {
            "emotion_stability": float(emotion_stability),
            "valence_stability": float(valence_stability),
            "arousal_stability": float(arousal_stability),
            "emotion_changes": emotion_changes
        }


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python emotion_recognition_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = EmotionRecognitionExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
