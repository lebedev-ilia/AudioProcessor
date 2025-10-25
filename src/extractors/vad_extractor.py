"""
VAD (Voice Activity Detection) extractor for audio feature extraction.

This extractor implements:
- Voice Activity Detection using WebRTC VAD
- Voiced fraction calculation
- F0 (fundamental frequency) estimation using pyin
- Pitch analysis and statistics
"""

import librosa
import numpy as np
import soundfile as sf
import webrtcvad
from typing import Dict, Any, Tuple, List
import logging
from src.core.base_extractor import BaseExtractor
from src.schemas.models import ExtractorResult

logger = logging.getLogger(__name__)


class VADExtractor(BaseExtractor):
    """Extractor for Voice Activity Detection and pitch features."""
    
    name = "vad_extractor"
    version = "0.1.0"
    description = "Voice Activity Detection and fundamental frequency extraction"
    
    def __init__(self):
        """Initialize VAD extractor with default parameters."""
        super().__init__()
        
        # VAD parameters
        self.vad_mode = 2  # VAD aggressiveness (0-3, 2 is balanced)
        self.frame_duration_ms = 30  # Frame duration in milliseconds
        self.sample_rate = 16000  # Sample rate for VAD (WebRTC requirement)
        
        # F0 estimation parameters
        self.fmin = 50.0  # Minimum frequency for F0 estimation
        self.fmax = 400.0  # Maximum frequency for F0 estimation
        self.hop_length = 512  # Hop length for F0 estimation
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(self.vad_mode)
        
        self.logger.info(f"Initialized {self.name} v{self.version}")
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract VAD and pitch features from audio file.
        
        Args:
            input_uri: Path to input audio file
            tmp_path: Path to temporary directory (unused for this extractor)
            
        Returns:
            ExtractorResult with VAD and pitch features
        """
        self._log_extraction_start(input_uri)
        
        try:
            # Load audio file
            audio, sample_rate = self._load_audio(input_uri)
            
            # Extract VAD features
            vad_features = self._extract_vad_features(audio, sample_rate)
            
            # Extract F0 features
            f0_features = self._extract_f0_features(audio, sample_rate)
            
            # Combine all features
            payload = {**vad_features, **f0_features}
            
            # Create successful result
            result = self._create_result(
                success=True,
                payload=payload,
                processing_time=None  # Will be set by base class timing
            )
            
            self._log_extraction_success(input_uri, 0.0)  # Time will be updated
            return result
            
        except Exception as e:
            error_msg = f"VAD extraction failed: {str(e)}"
            self._log_extraction_error(input_uri, error_msg, 0.0)
            
            return self._create_result(
                success=False,
                error=error_msg,
                processing_time=None
            )
    
    def _load_audio(self, input_uri: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa.
        
        Args:
            input_uri: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio with librosa (automatically resamples to 22050 Hz)
            audio, sr = librosa.load(
                input_uri,
                sr=22050,  # Standard sample rate for audio analysis
                mono=True,  # Convert to mono
                res_type='kaiser_fast'  # Fast resampling
            )
            
            self.logger.debug(f"Loaded audio: {len(audio)} samples at {sr} Hz")
            return audio, sr
            
        except Exception as e:
            self.logger.error(f"Failed to load audio file {input_uri}: {str(e)}")
            raise
    
    def _extract_vad_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Extract Voice Activity Detection features.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with VAD features
        """
        try:
            # Resample audio to 16kHz for VAD
            audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
            
            # Convert to 16-bit PCM
            audio_16bit = (audio_16k * 32767).astype(np.int16)
            
            # Calculate frame size
            frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
            
            # Process frames
            voiced_frames = 0
            total_frames = 0
            vad_decisions = []
            
            for i in range(0, len(audio_16bit) - frame_size + 1, frame_size):
                frame = audio_16bit[i:i + frame_size]
                
                # Ensure frame is exactly the right size
                if len(frame) == frame_size:
                    try:
                        is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                        vad_decisions.append(is_speech)
                        
                        if is_speech:
                            voiced_frames += 1
                        total_frames += 1
                        
                    except Exception as e:
                        # If VAD fails for this frame, assume no speech
                        vad_decisions.append(False)
                        total_frames += 1
            
            # Calculate voiced fraction
            voiced_fraction = voiced_frames / total_frames if total_frames > 0 else 0.0
            
            # Calculate additional VAD statistics
            vad_array = np.array(vad_decisions)
            speech_segments = self._find_speech_segments(vad_array)
            
            features = {
                "voiced_fraction": voiced_fraction,
                "voiced_frames": voiced_frames,
                "total_frames": total_frames,
                "speech_segments_count": len(speech_segments),
                "speech_segments": speech_segments,
                "vad_decisions": vad_decisions
            }
            
            self.logger.debug(f"Extracted VAD features: {len(features)} features")
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract VAD features: {str(e)}")
            # Return default values if VAD fails
            return {
                "voiced_fraction": 0.0,
                "voiced_frames": 0,
                "total_frames": 0,
                "speech_segments_count": 0,
                "speech_segments": [],
                "vad_decisions": []
            }
    
    def _extract_f0_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Extract F0 (fundamental frequency) features using pyin.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with F0 features
        """
        try:
            # Extract F0 using pyin
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=sample_rate,
                hop_length=self.hop_length
            )
            
            # Calculate F0 statistics (only for voiced frames)
            voiced_f0 = f0[voiced_flag]
            
            if len(voiced_f0) > 0:
                f0_mean = float(np.mean(voiced_f0))
                f0_std = float(np.std(voiced_f0))
                f0_min = float(np.min(voiced_f0))
                f0_max = float(np.max(voiced_f0))
                f0_median = float(np.median(voiced_f0))
                
                # Calculate F0 percentiles
                f0_p25 = float(np.percentile(voiced_f0, 25))
                f0_p75 = float(np.percentile(voiced_f0, 75))
                
                # Calculate F0 range and coefficient of variation
                f0_range = f0_max - f0_min
                f0_cv = f0_std / f0_mean if f0_mean > 0 else 0.0
                
                # Calculate F0 stability (inverse of coefficient of variation)
                f0_stability = 1.0 / (1.0 + f0_cv) if f0_cv > 0 else 1.0
                
            else:
                # No voiced frames found
                f0_mean = f0_std = f0_min = f0_max = f0_median = 0.0
                f0_p25 = f0_p75 = f0_range = f0_cv = f0_stability = 0.0
            
            # Calculate overall F0 statistics (including unvoiced frames)
            f0_overall_mean = float(np.mean(f0[~np.isnan(f0)])) if np.any(~np.isnan(f0)) else 0.0
            f0_overall_std = float(np.std(f0[~np.isnan(f0)])) if np.any(~np.isnan(f0)) else 0.0
            
            # Calculate voiced probability statistics
            voiced_prob_mean = float(np.mean(voiced_probs))
            voiced_prob_std = float(np.std(voiced_probs))
            
            features = {
                "f0_mean": f0_mean,
                "f0_std": f0_std,
                "f0_min": f0_min,
                "f0_max": f0_max,
                "f0_median": f0_median,
                "f0_p25": f0_p25,
                "f0_p75": f0_p75,
                "f0_range": f0_range,
                "f0_cv": f0_cv,
                "f0_stability": f0_stability,
                "f0_overall_mean": f0_overall_mean,
                "f0_overall_std": f0_overall_std,
                "voiced_prob_mean": voiced_prob_mean,
                "voiced_prob_std": voiced_prob_std,
                "f0_array": f0.tolist(),
                "voiced_flag_array": voiced_flag.tolist(),
                "voiced_probs_array": voiced_probs.tolist()
            }
            
            self.logger.debug(f"Extracted F0 features: {len(features)} features")
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract F0 features: {str(e)}")
            # Return default values if F0 extraction fails
            return {
                "f0_mean": 0.0,
                "f0_std": 0.0,
                "f0_min": 0.0,
                "f0_max": 0.0,
                "f0_median": 0.0,
                "f0_p25": 0.0,
                "f0_p75": 0.0,
                "f0_range": 0.0,
                "f0_cv": 0.0,
                "f0_stability": 0.0,
                "f0_overall_mean": 0.0,
                "f0_overall_std": 0.0,
                "voiced_prob_mean": 0.0,
                "voiced_prob_std": 0.0,
                "f0_array": [],
                "voiced_flag_array": [],
                "voiced_probs_array": []
            }
    
    def _find_speech_segments(self, vad_decisions: np.ndarray) -> List[Dict[str, Any]]:
        """
        Find continuous speech segments from VAD decisions.
        
        Args:
            vad_decisions: Array of VAD decisions
            
        Returns:
            List of speech segments with start/end information
        """
        try:
            segments = []
            in_speech = False
            start_frame = 0
            
            for i, decision in enumerate(vad_decisions):
                if decision and not in_speech:
                    # Start of speech segment
                    start_frame = i
                    in_speech = True
                elif not decision and in_speech:
                    # End of speech segment
                    segments.append({
                        "start_frame": start_frame,
                        "end_frame": i - 1,
                        "duration_frames": i - start_frame,
                        "duration_ms": (i - start_frame) * self.frame_duration_ms
                    })
                    in_speech = False
            
            # Handle case where speech continues to end of audio
            if in_speech:
                segments.append({
                    "start_frame": start_frame,
                    "end_frame": len(vad_decisions) - 1,
                    "duration_frames": len(vad_decisions) - start_frame,
                    "duration_ms": (len(vad_decisions) - start_frame) * self.frame_duration_ms
                })
            
            return segments
            
        except Exception as e:
            self.logger.error(f"Failed to find speech segments: {str(e)}")
            return []
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get extractor parameters.
        
        Returns:
            Dictionary with extractor parameters
        """
        return {
            "vad_mode": self.vad_mode,
            "frame_duration_ms": self.frame_duration_ms,
            "sample_rate": self.sample_rate,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "hop_length": self.hop_length
        }


# For running as a module
if __name__ == "__main__":
    import sys
    import json
    import tempfile
    import os
    
    if len(sys.argv) != 2:
        print("Usage: python vad_extractor.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file {audio_file} not found")
        sys.exit(1)
    
    # Create extractor and run
    extractor = VADExtractor()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = extractor.run(audio_file, tmp_dir)
        
        # Print result as JSON
        print(json.dumps(result.dict(), indent=2))
