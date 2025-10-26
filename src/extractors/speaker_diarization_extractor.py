"""
Speaker Diarization Extractor for speaker identification and segmentation
Extracts speaker timeline, speaker count, and speaker embeddings
"""

import numpy as np
import librosa
from typing import Dict, Any, List, Tuple, Optional
from src.core.base_extractor import BaseExtractor, ExtractorResult


class SpeakerDiarizationExtractor(BaseExtractor):
    """
    Speaker Diarization Extractor for speaker identification and segmentation
    Uses pyannote.audio for speaker diarization and speaker embeddings
    """
    
    name = "speaker_diarization"
    version = "1.0.0"
    description = "Speaker diarization: timeline, count, embeddings"
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 16000  # Standard for speaker diarization
        self.min_speakers = 1
        self.max_speakers = 10
        self._pipeline = None
        self._embedding_model = None
    
    def _load_models(self):
        """Load pyannote.audio models if not already loaded"""
        if self._pipeline is None:
            try:
                # Try to import and load models with comprehensive error handling
                try:
                    from pyannote.audio import Pipeline
                    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
                    
                    # Load diarization pipeline with error handling
                    try:
                        self._pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
                        self.logger.info("Successfully loaded diarization pipeline")
                    except Exception as pipeline_error:
                        self.logger.warning(f"Failed to load diarization pipeline: {pipeline_error}")
                        self._pipeline = None
                    
                    # Load speaker embedding model with error handling
                    try:
                        self._embedding_model = PretrainedSpeakerEmbedding(
                            "speechbrain/spkrec-ecapa-voxceleb",
                            device="cpu"  # Use CPU by default
                        )
                        self.logger.info("Successfully loaded speaker embedding model")
                    except Exception as embedding_error:
                        self.logger.warning(f"Failed to load embedding model: {embedding_error}")
                        self._embedding_model = None
                        
                except ImportError as import_error:
                    self.logger.warning(f"Failed to import pyannote.audio modules: {import_error}")
                    self._pipeline = None
                    self._embedding_model = None
                except Exception as e:
                    # Handle ForwardRef and other pyannote.audio issues
                    self.logger.warning(f"pyannote.audio compatibility issue: {e}")
                    self._pipeline = None
                    self._embedding_model = None
                
                if self._pipeline is None and self._embedding_model is None:
                    self.logger.info("Speaker diarization models not available - using fallback mode")
                
            except ImportError:
                raise ImportError("pyannote.audio not installed. Install with: pip install pyannote.audio")
            except Exception as e:
                raise RuntimeError(f"Failed to load speaker diarization models: {e}")
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract speaker diarization features from audio file
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with speaker diarization features
        """
        try:
            self.logger.info(f"Starting speaker diarization extraction for {input_uri}")
            
            # Load models
            self._load_models()
            
            # Load audio
            audio, sr = librosa.load(input_uri, sr=self.sample_rate)
            
            # Extract speaker diarization features with timing
            features, processing_time = self._time_execution(self._extract_diarization_features, audio, sr)
            
            self.logger.info(f"Speaker diarization extraction completed successfully in {processing_time:.3f}s")
            
            return self._create_result(
                success=True,
                payload=features,
                processing_time=processing_time
            )
            
        except Exception as e:
            error_msg = str(e) if e else "Unknown error occurred"
            self.logger.error(f"Speaker diarization extraction failed: {error_msg}")
            return self._create_result(
                success=False,
                error=error_msg
            )
    
    def _extract_diarization_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract speaker diarization features from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of speaker diarization features
        """
        features = {}
        
        try:
            # Check if pipeline is available
            if self._pipeline is None:
                # Fallback mode - return basic features without diarization
                self.logger.warning("Using fallback mode for speaker diarization")
                features = self._extract_fallback_features(audio, sr)
                return features
            
            # Run speaker diarization
            diarization = self._pipeline({"audio": audio, "sample_rate": sr})
            
            # Extract speaker timeline
            speaker_timeline = self._extract_speaker_timeline(diarization)
            features["diarization_timeline"] = speaker_timeline
            
            # Count speakers
            speakers = set()
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)
            
            features["num_speakers_detected"] = len(speakers)
            features["speaker_labels"] = list(speakers)
            
            # Extract speaker embeddings
            speaker_embeddings = self._extract_speaker_embeddings(audio, sr, diarization)
            features["speaker_embeddings"] = speaker_embeddings
            
            # Calculate speaker statistics
            speaker_stats = self._calculate_speaker_statistics(diarization)
            features.update(speaker_stats)
            
            # Calculate speaker change points
            change_points = self._calculate_speaker_changes(diarization)
            features["speaker_change_points"] = change_points
            
        except Exception as e:
            self.logger.warning(f"Speaker diarization failed: {e}")
            # Return default values
            features = self._get_default_diarization_features()
        
        return features
    
    def _extract_fallback_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract basic speaker features when diarization models are not available
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of basic speaker features
        """
        features = {
            "diarization_timeline": [],
            "num_speakers_detected": 1,  # Assume single speaker
            "speaker_labels": ["SPEAKER_00"],
            "speaker_embeddings": {},
            "speaker_change_points": [],
            "speaker_segments": [],
            "speaker_duration_stats": {
                "total_duration": len(audio) / sr,
                "speaking_time": len(audio) / sr,
                "silence_time": 0.0
            },
            "speaker_energy_stats": {
                "mean_energy": float(np.mean(audio ** 2)),
                "std_energy": float(np.std(audio ** 2)),
                "max_energy": float(np.max(audio ** 2)),
                "min_energy": float(np.min(audio ** 2))
            }
        }
        
        self.logger.info("Extracted basic speaker features in fallback mode")
        return features
    
    def _extract_speaker_timeline(self, diarization) -> List[Dict[str, Any]]:
        """
        Extract speaker timeline from diarization result
        
        Args:
            diarization: Pyannote diarization result
            
        Returns:
            List of speaker segments with timestamps
        """
        timeline = []
        
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            timeline.append({
                "speaker": speaker,
                "start": float(segment.start),
                "end": float(segment.end),
                "duration": float(segment.end - segment.start)
            })
        
        # Sort by start time
        timeline.sort(key=lambda x: x["start"])
        
        return timeline
    
    def _extract_speaker_embeddings(self, audio: np.ndarray, sr: int, diarization) -> Dict[str, List[float]]:
        """
        Extract speaker embeddings for each speaker
        
        Args:
            audio: Audio array
            sr: Sample rate
            diarization: Pyannote diarization result
            
        Returns:
            Dictionary mapping speaker labels to embeddings
        """
        embeddings = {}
        
        try:
            # Group segments by speaker
            speaker_segments = {}
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append(segment)
            
            # Extract embedding for each speaker
            for speaker, segments in speaker_segments.items():
                # Combine all segments for this speaker
                speaker_audio = []
                for segment in segments:
                    start_sample = int(segment.start * sr)
                    end_sample = int(segment.end * sr)
                    speaker_audio.extend(audio[start_sample:end_sample])
                
                if len(speaker_audio) > 0:
                    speaker_audio = np.array(speaker_audio)
                    
                    # Extract embedding
                    embedding = self._embedding_model({"waveform": speaker_audio.reshape(1, -1), "sample_rate": sr})
                    embeddings[speaker] = embedding[0].tolist()
                else:
                    embeddings[speaker] = [0.0] * 192  # Default embedding size for ECAPA-TDNN
        
        except Exception as e:
            self.logger.warning(f"Speaker embedding extraction failed: {e}")
            # Return empty embeddings
            speakers = set()
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)
            
            for speaker in speakers:
                embeddings[speaker] = [0.0] * 192
        
        return embeddings
    
    def _calculate_speaker_statistics(self, diarization) -> Dict[str, Any]:
        """
        Calculate speaker statistics from diarization result
        
        Args:
            diarization: Pyannote diarization result
            
        Returns:
            Dictionary with speaker statistics
        """
        stats = {}
        
        # Calculate total duration
        total_duration = 0.0
        for segment, _, _ in diarization.itertracks(yield_label=True):
            total_duration = max(total_duration, segment.end)
        
        stats["total_duration"] = total_duration
        
        # Calculate speaker durations
        speaker_durations = {}
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_durations:
                speaker_durations[speaker] = 0.0
            speaker_durations[speaker] += segment.end - segment.start
        
        stats["speaker_durations"] = speaker_durations
        
        # Calculate speaker fractions
        speaker_fractions = {}
        for speaker, duration in speaker_durations.items():
            speaker_fractions[speaker] = duration / total_duration if total_duration > 0 else 0.0
        
        stats["speaker_fractions"] = speaker_fractions
        
        # Calculate dominant speaker
        if speaker_fractions:
            dominant_speaker = max(speaker_fractions, key=speaker_fractions.get)
            stats["dominant_speaker"] = dominant_speaker
            stats["dominant_speaker_fraction"] = speaker_fractions[dominant_speaker]
        else:
            stats["dominant_speaker"] = None
            stats["dominant_speaker_fraction"] = 0.0
        
        # Calculate speaker balance (entropy)
        if len(speaker_fractions) > 1:
            entropy = -sum(f * np.log2(f) for f in speaker_fractions.values() if f > 0)
            max_entropy = np.log2(len(speaker_fractions))
            balance = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            balance = 0.0
        
        stats["speaker_balance"] = balance
        
        return stats
    
    def _calculate_speaker_changes(self, diarization) -> List[Dict[str, Any]]:
        """
        Calculate speaker change points
        
        Args:
            diarization: Pyannote diarization result
            
        Returns:
            List of speaker change points
        """
        change_points = []
        
        # Get all segments sorted by time
        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((segment.start, segment.end, speaker))
        
        segments.sort(key=lambda x: x[0])
        
        # Find speaker changes
        for i in range(1, len(segments)):
            prev_segment = segments[i-1]
            curr_segment = segments[i]
            
            if prev_segment[2] != curr_segment[2]:  # Different speakers
                change_points.append({
                    "time": float(curr_segment[0]),
                    "from_speaker": prev_segment[2],
                    "to_speaker": curr_segment[2],
                    "gap_duration": float(curr_segment[0] - prev_segment[1])
                })
        
        return change_points
    
    def _get_default_diarization_features(self) -> Dict[str, Any]:
        """
        Get default diarization features when extraction fails
        
        Returns:
            Dictionary with default features
        """
        return {
            "diarization_timeline": [],
            "num_speakers_detected": 1,
            "speaker_labels": ["SPEAKER_00"],
            "speaker_embeddings": {"SPEAKER_00": [0.0] * 192},
            "total_duration": 0.0,
            "speaker_durations": {"SPEAKER_00": 0.0},
            "speaker_fractions": {"SPEAKER_00": 1.0},
            "dominant_speaker": "SPEAKER_00",
            "dominant_speaker_fraction": 1.0,
            "speaker_balance": 0.0,
            "speaker_change_points": []
        }


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python speaker_diarization_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = SpeakerDiarizationExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
