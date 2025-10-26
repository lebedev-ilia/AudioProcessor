"""
ASR (Automatic Speech Recognition) Extractor using Whisper
Extracts transcriptions with word-level timestamps only
"""

import os
import json
import tempfile
import subprocess
from typing import Dict, List, Optional, Any
import numpy as np
import librosa
from src.core.base_extractor import BaseExtractor, ExtractorResult


class ASRExtractor(BaseExtractor):
    """
    ASR Extractor using OpenAI Whisper for speech recognition
    """
    
    name = "asr"
    version = "1.0.0"
    description = "Automatic Speech Recognition using Whisper"
    
    def __init__(self):
        super().__init__()
        self.model_name = os.getenv("WHISPER_MODEL", "base")
        self.language = os.getenv("WHISPER_LANGUAGE", None)  # None for auto-detect
        self.device = os.getenv("WHISPER_DEVICE", "cpu")  # cpu or cuda
        self._model = None
    
    def _load_model(self):
        """Load Whisper model if not already loaded"""
        if self._model is None:
            try:
                import whisper
                # Suppress verbose loading output
                import sys
                from contextlib import redirect_stdout
                with redirect_stdout(open('/dev/null', 'w')):
                    self._model = whisper.load_model(self.model_name, device=self.device)
                self.logger.info(f"Loaded Whisper model: {self.model_name} on {self.device}")
            except ImportError:
                raise ImportError("Whisper not installed. Install with: pip install openai-whisper")
            except Exception as e:
                raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract ASR features from audio file
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with ASR features
        """
        try:
            self.logger.info(f"Starting ASR extraction for {input_uri}")
            
            # Load model
            self._load_model()
            
            # Load audio
            audio, sr = librosa.load(input_uri, sr=16000)  # Whisper expects 16kHz
            
            # Transcribe audio
            result = self._model.transcribe(
                audio,
                language=self.language,
                word_timestamps=True,
                verbose=False
            )
            
            # Extract features
            features = self._extract_asr_features(result, audio, sr)
            
            self.logger.info(f"ASR extraction completed successfully")
            
            return ExtractorResult(
                name=self.name,
                version=self.version,
                success=True,
                payload=features
            )
            
        except Exception as e:
            self.logger.error(f"ASR extraction failed: {e}")
            return ExtractorResult(
                name=self.name,
                version=self.version,
                success=False,
                error=str(e)
            )
    
    def _extract_asr_features(self, whisper_result: Dict, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract ASR features from Whisper result with confidence metrics
        
        Args:
            whisper_result: Whisper transcription result
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of ASR features
        """
        features = {}
        
        # Basic transcription
        features["transcript_text"] = whisper_result.get("text", "").strip()
        features["language"] = whisper_result.get("language", "unknown")
        
        # Confidence metrics
        segments = whisper_result.get("segments", [])
        word_timestamps = []
        segment_confidences = []
        word_confidences = []
        
        for segment in segments:
            # Segment-level confidence
            segment_confidence = segment.get("avg_logprob", 0.0)
            segment_confidences.append(segment_confidence)
            
            # Extract word-level timestamps and confidences
            for word_info in segment.get("words", []):
                word_data = {
                    "word": word_info.get("word", "").strip(),
                    "start": word_info.get("start", 0.0),
                    "end": word_info.get("end", 0.0),
                    "confidence": word_info.get("probability", 0.0)
                }
                word_timestamps.append(word_data)
                word_confidences.append(word_info.get("probability", 0.0))
        
        features["word_timestamps"] = word_timestamps
        
        # Calculate overall confidence metrics
        if segment_confidences:
            # Convert log probabilities to confidence scores (0-1)
            segment_confidences_exp = [np.exp(conf) for conf in segment_confidences]
            features["transcript_confidence"] = float(np.mean(segment_confidences_exp))
            features["transcript_confidence_std"] = float(np.std(segment_confidences_exp))
            features["transcript_confidence_min"] = float(np.min(segment_confidences_exp))
            features["transcript_confidence_max"] = float(np.max(segment_confidences_exp))
        else:
            features["transcript_confidence"] = 0.0
            features["transcript_confidence_std"] = 0.0
            features["transcript_confidence_min"] = 0.0
            features["transcript_confidence_max"] = 0.0
        
        # Word-level confidence statistics
        if word_confidences:
            features["word_confidence_mean"] = float(np.mean(word_confidences))
            features["word_confidence_std"] = float(np.std(word_confidences))
            features["word_confidence_min"] = float(np.min(word_confidences))
            features["word_confidence_max"] = float(np.max(word_confidences))
        else:
            features["word_confidence_mean"] = 0.0
            features["word_confidence_std"] = 0.0
            features["word_confidence_min"] = 0.0
            features["word_confidence_max"] = 0.0
        
        # Language confidence (estimated from overall transcription quality)
        # Whisper doesn't provide explicit language confidence, so we estimate it
        if features["transcript_confidence"] > 0.8:
            features["language_confidence"] = 0.9
        elif features["transcript_confidence"] > 0.6:
            features["language_confidence"] = 0.7
        elif features["transcript_confidence"] > 0.4:
            features["language_confidence"] = 0.5
        else:
            features["language_confidence"] = 0.3
        
        # Additional metadata
        features["num_segments"] = len(segments)
        features["num_words"] = len(word_timestamps)
        features["audio_duration"] = len(audio) / sr
        
        return features
    
    def _run_subprocess(self, command: List[str], timeout: int = 300) -> subprocess.CompletedProcess:
        """
        Run subprocess with timeout and error handling
        
        Args:
            command: Command to run
            timeout: Timeout in seconds
            
        Returns:
            CompletedProcess result
        """
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )
            return result
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"ASR processing timeout after {timeout}s: {e}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ASR processing failed: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in ASR processing: {e}")


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python asr_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = ASRExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
