"""
Sound Event Detection Extractor for audio event detection
Extracts sound event timeline and acoustic scene classification
"""

import numpy as np
import librosa
from typing import Dict, Any, List, Tuple, Optional
from src.core.base_extractor import BaseExtractor, ExtractorResult


class SoundEventDetectionExtractor(BaseExtractor):
    """
    Sound Event Detection Extractor for audio event detection
    Extracts sound event timeline and acoustic scene classification
    """
    
    name = "sound_event_detection"
    version = "1.0.0"
    description = "Sound event detection: event timeline, acoustic scene classification"
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 22050
        self.hop_length = 512
        self.frame_length = 2048
        
        # Common sound event categories
        self.event_categories = {
            "speech": ["speech", "talking", "conversation"],
            "music": ["music", "singing", "instrumental"],
            "noise": ["noise", "static", "hiss"],
            "silence": ["silence", "quiet"],
            "environmental": ["wind", "rain", "traffic", "birds", "animals"],
            "mechanical": ["engine", "motor", "machine", "fan"],
            "human": ["footsteps", "clapping", "laughing", "coughing"],
            "alarm": ["alarm", "siren", "bell", "beep"]
        }
        
        # Acoustic scene categories
        self.scene_categories = {
            "indoor": ["home", "office", "classroom", "restaurant", "store"],
            "outdoor": ["street", "park", "beach", "forest", "city"],
            "transport": ["car", "bus", "train", "plane", "subway"],
            "nature": ["forest", "beach", "mountain", "lake", "river"],
            "urban": ["street", "city", "construction", "traffic"],
            "quiet": ["library", "bedroom", "quiet_room"],
            "noisy": ["restaurant", "bar", "party", "concert"]
        }
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract sound event detection features from audio file
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with sound event detection features
        """
        try:
            self.logger.info(f"Starting sound event detection extraction for {input_uri}")
            
            # Load audio
            audio, sr = librosa.load(input_uri, sr=self.sample_rate)
            
            # Extract sound event detection features with timing
            features, processing_time = self._time_execution(self._extract_sound_event_features, audio, sr)
            
            self.logger.info(f"Sound event detection extraction completed successfully in {processing_time:.3f}s")
            
            return self._create_result(
                success=True,
                payload=features,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Sound event detection extraction failed: {e}")
            return self._create_result(
                success=False,
                error=str(e)
            )
    
    def _extract_sound_event_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract sound event detection features from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of sound event detection features
        """
        features = {}
        
        # Sound event timeline
        event_timeline = self._detect_sound_events(audio, sr)
        features["sed_event_timeline"] = event_timeline
        
        # Acoustic scene classification
        scene_classification = self._classify_acoustic_scene(audio, sr)
        features["acoustic_scene_label"] = scene_classification
        
        # Event statistics
        event_stats = self._calculate_event_statistics(event_timeline, len(audio) / sr)
        features.update(event_stats)
        
        # Scene features
        scene_features = self._extract_scene_features(audio, sr)
        features.update(scene_features)
        
        return features
    
    def _detect_sound_events(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """
        Detect sound events in the audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            List of detected sound events
        """
        try:
            events = []
            
            # Calculate energy-based event detection
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
            energy_threshold = np.mean(rms) + 2 * np.std(rms)
            
            # Find energy peaks (potential events)
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(rms, height=energy_threshold, distance=5)
            
            # Convert peaks to time
            peak_times = peaks * self.hop_length / sr
            
            # Classify each peak as an event
            for i, peak_time in enumerate(peak_times):
                # Extract audio segment around the peak
                start_sample = max(0, int((peak_time - 0.5) * sr))
                end_sample = min(len(audio), int((peak_time + 0.5) * sr))
                segment = audio[start_sample:end_sample]
                
                # Classify the event
                event_type = self._classify_event_type(segment, sr)
                confidence = self._calculate_event_confidence(segment, sr, event_type)
                
                events.append({
                    "start": float(peak_time - 0.5),
                    "end": float(peak_time + 0.5),
                    "duration": 1.0,
                    "event_type": event_type,
                    "confidence": confidence,
                    "energy": float(rms[peaks[i]])
                })
            
            # Merge nearby events of the same type
            events = self._merge_nearby_events(events)
            
            return events
            
        except Exception as e:
            self.logger.warning(f"Sound event detection failed: {e}")
            return []
    
    def _classify_event_type(self, segment: np.ndarray, sr: int) -> str:
        """
        Classify the type of sound event
        
        Args:
            segment: Audio segment
            sr: Sample rate
            
        Returns:
            Event type classification
        """
        try:
            # Extract features for classification
            features = self._extract_event_features(segment, sr)
            
            # Rule-based classification
            event_scores = {}
            
            # Speech detection
            speech_score = self._calculate_speech_score(features)
            event_scores["speech"] = speech_score
            
            # Music detection
            music_score = self._calculate_music_score(features)
            event_scores["music"] = music_score
            
            # Noise detection
            noise_score = self._calculate_noise_score(features)
            event_scores["noise"] = noise_score
            
            # Silence detection
            silence_score = self._calculate_silence_score(features)
            event_scores["silence"] = silence_score
            
            # Environmental sound detection
            env_score = self._calculate_environmental_score(features)
            event_scores["environmental"] = env_score
            
            # Mechanical sound detection
            mech_score = self._calculate_mechanical_score(features)
            event_scores["mechanical"] = mech_score
            
            # Human sound detection
            human_score = self._calculate_human_score(features)
            event_scores["human"] = human_score
            
            # Alarm detection
            alarm_score = self._calculate_alarm_score(features)
            event_scores["alarm"] = alarm_score
            
            # Find the event type with highest score
            best_event = max(event_scores, key=event_scores.get)
            
            return best_event if event_scores[best_event] > 0.3 else "unknown"
            
        except Exception as e:
            self.logger.warning(f"Event classification failed: {e}")
            return "unknown"
    
    def _extract_event_features(self, segment: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract features for event classification"""
        try:
            features = {}
            
            # Energy features
            rms = librosa.feature.rms(y=segment, hop_length=self.hop_length)[0]
            features["energy_mean"] = float(np.mean(rms))
            features["energy_std"] = float(np.std(rms))
            features["energy_max"] = float(np.max(rms))
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr, hop_length=self.hop_length)[0]
            features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
            features["spectral_centroid_std"] = float(np.std(spectral_centroid))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(segment, hop_length=self.hop_length)[0]
            features["zcr_mean"] = float(np.mean(zcr))
            features["zcr_std"] = float(np.std(zcr))
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            for i in range(min(5, mfcc.shape[0])):
                features[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
                features[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))
            
            # Pitch features
            f0 = librosa.yin(segment, fmin=50, fmax=2000, sr=sr, hop_length=self.hop_length)
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) > 0:
                features["pitch_mean"] = float(np.mean(f0_clean))
                features["pitch_std"] = float(np.std(f0_clean))
                features["pitch_present"] = 1.0
            else:
                features["pitch_mean"] = 0.0
                features["pitch_std"] = 0.0
                features["pitch_present"] = 0.0
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Event feature extraction failed: {e}")
            return {}
    
    def _calculate_speech_score(self, features: Dict[str, Any]) -> float:
        """Calculate speech detection score"""
        try:
            score = 0.0
            
            # Speech typically has pitch
            if features.get("pitch_present", 0) > 0.5:
                score += 0.4
            
            # Speech typically has moderate energy
            energy_mean = features.get("energy_mean", 0)
            if 0.05 <= energy_mean <= 0.2:
                score += 0.3
            
            # Speech typically has moderate ZCR
            zcr_mean = features.get("zcr_mean", 0)
            if 0.1 <= zcr_mean <= 0.3:
                score += 0.3
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _calculate_music_score(self, features: Dict[str, Any]) -> float:
        """Calculate music detection score"""
        try:
            score = 0.0
            
            # Music typically has pitch
            if features.get("pitch_present", 0) > 0.5:
                score += 0.3
            
            # Music typically has higher energy
            energy_mean = features.get("energy_mean", 0)
            if energy_mean > 0.1:
                score += 0.3
            
            # Music typically has lower ZCR (more harmonic)
            zcr_mean = features.get("zcr_mean", 0)
            if zcr_mean < 0.2:
                score += 0.4
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _calculate_noise_score(self, features: Dict[str, Any]) -> float:
        """Calculate noise detection score"""
        try:
            score = 0.0
            
            # Noise typically has no clear pitch
            if features.get("pitch_present", 0) < 0.3:
                score += 0.4
            
            # Noise typically has high ZCR
            zcr_mean = features.get("zcr_mean", 0)
            if zcr_mean > 0.3:
                score += 0.3
            
            # Noise typically has high spectral variation
            spectral_centroid_std = features.get("spectral_centroid_std", 0)
            if spectral_centroid_std > 1000:
                score += 0.3
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _calculate_silence_score(self, features: Dict[str, Any]) -> float:
        """Calculate silence detection score"""
        try:
            score = 0.0
            
            # Silence has very low energy
            energy_mean = features.get("energy_mean", 0)
            if energy_mean < 0.01:
                score += 0.5
            
            # Silence has no pitch
            if features.get("pitch_present", 0) < 0.1:
                score += 0.5
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _calculate_environmental_score(self, features: Dict[str, Any]) -> float:
        """Calculate environmental sound detection score"""
        try:
            score = 0.0
            
            # Environmental sounds typically have no clear pitch
            if features.get("pitch_present", 0) < 0.4:
                score += 0.3
            
            # Environmental sounds have moderate energy
            energy_mean = features.get("energy_mean", 0)
            if 0.02 <= energy_mean <= 0.15:
                score += 0.3
            
            # Environmental sounds have moderate ZCR
            zcr_mean = features.get("zcr_mean", 0)
            if 0.1 <= zcr_mean <= 0.4:
                score += 0.4
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _calculate_mechanical_score(self, features: Dict[str, Any]) -> float:
        """Calculate mechanical sound detection score"""
        try:
            score = 0.0
            
            # Mechanical sounds typically have no clear pitch
            if features.get("pitch_present", 0) < 0.3:
                score += 0.3
            
            # Mechanical sounds have moderate to high energy
            energy_mean = features.get("energy_mean", 0)
            if energy_mean > 0.05:
                score += 0.3
            
            # Mechanical sounds have high ZCR
            zcr_mean = features.get("zcr_mean", 0)
            if zcr_mean > 0.2:
                score += 0.4
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _calculate_human_score(self, features: Dict[str, Any]) -> float:
        """Calculate human sound detection score"""
        try:
            score = 0.0
            
            # Human sounds typically have pitch
            if features.get("pitch_present", 0) > 0.4:
                score += 0.4
            
            # Human sounds have moderate energy
            energy_mean = features.get("energy_mean", 0)
            if 0.03 <= energy_mean <= 0.2:
                score += 0.3
            
            # Human sounds have moderate ZCR
            zcr_mean = features.get("zcr_mean", 0)
            if 0.1 <= zcr_mean <= 0.3:
                score += 0.3
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _calculate_alarm_score(self, features: Dict[str, Any]) -> float:
        """Calculate alarm sound detection score"""
        try:
            score = 0.0
            
            # Alarms typically have pitch
            if features.get("pitch_present", 0) > 0.5:
                score += 0.3
            
            # Alarms have high energy
            energy_mean = features.get("energy_mean", 0)
            if energy_mean > 0.1:
                score += 0.3
            
            # Alarms have low ZCR (more tonal)
            zcr_mean = features.get("zcr_mean", 0)
            if zcr_mean < 0.2:
                score += 0.4
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _calculate_event_confidence(self, segment: np.ndarray, sr: int, event_type: str) -> float:
        """Calculate confidence for event classification"""
        try:
            features = self._extract_event_features(segment, sr)
            
            # Calculate confidence based on feature consistency
            confidence = 0.5  # Base confidence
            
            # Adjust based on event type
            if event_type == "speech":
                confidence = self._calculate_speech_score(features)
            elif event_type == "music":
                confidence = self._calculate_music_score(features)
            elif event_type == "noise":
                confidence = self._calculate_noise_score(features)
            elif event_type == "silence":
                confidence = self._calculate_silence_score(features)
            elif event_type == "environmental":
                confidence = self._calculate_environmental_score(features)
            elif event_type == "mechanical":
                confidence = self._calculate_mechanical_score(features)
            elif event_type == "human":
                confidence = self._calculate_human_score(features)
            elif event_type == "alarm":
                confidence = self._calculate_alarm_score(features)
            
            return float(confidence)
            
        except Exception as e:
            self.logger.warning(f"Event confidence calculation failed: {e}")
            return 0.5
    
    def _merge_nearby_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge nearby events of the same type"""
        try:
            if len(events) <= 1:
                return events
            
            merged_events = []
            current_event = events[0]
            
            for i in range(1, len(events)):
                next_event = events[i]
                
                # Check if events are close and of the same type
                time_gap = next_event["start"] - current_event["end"]
                same_type = current_event["event_type"] == next_event["event_type"]
                
                if time_gap <= 1.0 and same_type:  # Merge if within 1 second and same type
                    # Merge events
                    current_event["end"] = next_event["end"]
                    current_event["duration"] = current_event["end"] - current_event["start"]
                    current_event["confidence"] = (current_event["confidence"] + next_event["confidence"]) / 2
                    current_event["energy"] = max(current_event["energy"], next_event["energy"])
                else:
                    # Add current event and start new one
                    merged_events.append(current_event)
                    current_event = next_event
            
            # Add the last event
            merged_events.append(current_event)
            
            return merged_events
            
        except Exception as e:
            self.logger.warning(f"Event merging failed: {e}")
            return events
    
    def _classify_acoustic_scene(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Classify acoustic scene
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with scene classification
        """
        try:
            # Extract scene features
            scene_features = self._extract_scene_features(audio, sr)
            
            # Calculate scene probabilities
            scene_probs = {}
            
            # Indoor vs outdoor
            indoor_score = self._calculate_indoor_score(scene_features)
            outdoor_score = self._calculate_outdoor_score(scene_features)
            scene_probs["indoor"] = indoor_score
            scene_probs["outdoor"] = outdoor_score
            
            # Transport
            transport_score = self._calculate_transport_score(scene_features)
            scene_probs["transport"] = transport_score
            
            # Nature
            nature_score = self._calculate_nature_score(scene_features)
            scene_probs["nature"] = nature_score
            
            # Urban
            urban_score = self._calculate_urban_score(scene_features)
            scene_probs["urban"] = urban_score
            
            # Quiet vs noisy
            quiet_score = self._calculate_quiet_score(scene_features)
            noisy_score = self._calculate_noisy_score(scene_features)
            scene_probs["quiet"] = quiet_score
            scene_probs["noisy"] = noisy_score
            
            # Normalize probabilities
            total_prob = sum(scene_probs.values())
            if total_prob > 0:
                scene_probs = {k: v / total_prob for k, v in scene_probs.items()}
            
            # Find best scene
            best_scene = max(scene_probs, key=scene_probs.get)
            
            return {
                "scene_label": best_scene,
                "scene_confidence": float(scene_probs[best_scene]),
                "scene_probabilities": scene_probs
            }
            
        except Exception as e:
            self.logger.warning(f"Acoustic scene classification failed: {e}")
            return {
                "scene_label": "unknown",
                "scene_confidence": 0.0,
                "scene_probabilities": {}
            }
    
    def _extract_scene_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract features for scene classification"""
        try:
            features = {}
            
            # Energy features
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
            features["energy_mean"] = float(np.mean(rms))
            features["energy_std"] = float(np.std(rms))
            features["energy_max"] = float(np.max(rms))
            
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
            
            # Pitch features
            f0 = librosa.yin(audio, fmin=50, fmax=2000, sr=sr, hop_length=self.hop_length)
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) > 0:
                features["pitch_mean"] = float(np.mean(f0_clean))
                features["pitch_std"] = float(np.std(f0_clean))
                features["pitch_present"] = 1.0
            else:
                features["pitch_mean"] = 0.0
                features["pitch_std"] = 0.0
                features["pitch_present"] = 0.0
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Scene feature extraction failed: {e}")
            return {}
    
    def _calculate_indoor_score(self, features: Dict[str, Any]) -> float:
        """Calculate indoor scene score"""
        try:
            score = 0.0
            
            # Indoor scenes typically have lower energy
            energy_mean = features.get("energy_mean", 0)
            if energy_mean < 0.1:
                score += 0.3
            
            # Indoor scenes typically have more speech/music
            if features.get("pitch_present", 0) > 0.3:
                score += 0.4
            
            # Indoor scenes typically have lower ZCR
            zcr_mean = features.get("zcr_mean", 0)
            if zcr_mean < 0.3:
                score += 0.3
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _calculate_outdoor_score(self, features: Dict[str, Any]) -> float:
        """Calculate outdoor scene score"""
        try:
            score = 0.0
            
            # Outdoor scenes typically have higher energy
            energy_mean = features.get("energy_mean", 0)
            if energy_mean > 0.05:
                score += 0.3
            
            # Outdoor scenes typically have more environmental sounds
            if features.get("pitch_present", 0) < 0.4:
                score += 0.4
            
            # Outdoor scenes typically have higher ZCR
            zcr_mean = features.get("zcr_mean", 0)
            if zcr_mean > 0.2:
                score += 0.3
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _calculate_transport_score(self, features: Dict[str, Any]) -> float:
        """Calculate transport scene score"""
        try:
            score = 0.0
            
            # Transport scenes typically have high energy
            energy_mean = features.get("energy_mean", 0)
            if energy_mean > 0.1:
                score += 0.4
            
            # Transport scenes typically have mechanical sounds
            if features.get("pitch_present", 0) < 0.3:
                score += 0.3
            
            # Transport scenes typically have high ZCR
            zcr_mean = features.get("zcr_mean", 0)
            if zcr_mean > 0.3:
                score += 0.3
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _calculate_nature_score(self, features: Dict[str, Any]) -> float:
        """Calculate nature scene score"""
        try:
            score = 0.0
            
            # Nature scenes typically have moderate energy
            energy_mean = features.get("energy_mean", 0)
            if 0.02 <= energy_mean <= 0.1:
                score += 0.3
            
            # Nature scenes typically have environmental sounds
            if features.get("pitch_present", 0) < 0.4:
                score += 0.4
            
            # Nature scenes typically have moderate ZCR
            zcr_mean = features.get("zcr_mean", 0)
            if 0.1 <= zcr_mean <= 0.4:
                score += 0.3
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _calculate_urban_score(self, features: Dict[str, Any]) -> float:
        """Calculate urban scene score"""
        try:
            score = 0.0
            
            # Urban scenes typically have high energy
            energy_mean = features.get("energy_mean", 0)
            if energy_mean > 0.08:
                score += 0.4
            
            # Urban scenes typically have mechanical sounds
            if features.get("pitch_present", 0) < 0.4:
                score += 0.3
            
            # Urban scenes typically have high ZCR
            zcr_mean = features.get("zcr_mean", 0)
            if zcr_mean > 0.25:
                score += 0.3
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _calculate_quiet_score(self, features: Dict[str, Any]) -> float:
        """Calculate quiet scene score"""
        try:
            score = 0.0
            
            # Quiet scenes have low energy
            energy_mean = features.get("energy_mean", 0)
            if energy_mean < 0.03:
                score += 0.5
            
            # Quiet scenes have low ZCR
            zcr_mean = features.get("zcr_mean", 0)
            if zcr_mean < 0.2:
                score += 0.5
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _calculate_noisy_score(self, features: Dict[str, Any]) -> float:
        """Calculate noisy scene score"""
        try:
            score = 0.0
            
            # Noisy scenes have high energy
            energy_mean = features.get("energy_mean", 0)
            if energy_mean > 0.1:
                score += 0.5
            
            # Noisy scenes have high ZCR
            zcr_mean = features.get("zcr_mean", 0)
            if zcr_mean > 0.3:
                score += 0.5
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _calculate_event_statistics(self, events: List[Dict[str, Any]], duration: float) -> Dict[str, Any]:
        """Calculate event statistics"""
        try:
            if duration == 0:
                return {
                    "event_count": 0,
                    "event_density": 0.0,
                    "event_duration_mean": 0.0,
                    "event_duration_std": 0.0,
                    "event_types": {}
                }
            
            # Count events
            event_count = len(events)
            event_density = event_count / duration
            
            # Calculate event durations
            event_durations = [event["duration"] for event in events]
            event_duration_mean = np.mean(event_durations) if event_durations else 0.0
            event_duration_std = np.std(event_durations) if event_durations else 0.0
            
            # Count event types
            event_types = {}
            for event in events:
                event_type = event["event_type"]
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            return {
                "event_count": event_count,
                "event_density": float(event_density),
                "event_duration_mean": float(event_duration_mean),
                "event_duration_std": float(event_duration_std),
                "event_types": event_types
            }
            
        except Exception as e:
            self.logger.warning(f"Event statistics calculation failed: {e}")
            return {
                "event_count": 0,
                "event_density": 0.0,
                "event_duration_mean": 0.0,
                "event_duration_std": 0.0,
                "event_types": {}
            }


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python sound_event_detection_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = SoundEventDetectionExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
