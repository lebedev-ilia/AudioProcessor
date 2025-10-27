"""
Test script to verify all extractors work on CPU without CUDA.
This script tests all 22 extractors to ensure they have proper fallback mechanisms.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Force CPU mode for testing
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.cuda.is_available = lambda: False

from src.extractors import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CPUFallbackTester:
    """Test class to verify CPU fallback for all extractors."""
    
    def __init__(self):
        """Initialize the tester."""
        self.test_audio_path = "test_audio.wav"
        self.tmp_dir = "/tmp"
        self.results = {
            "passed": [],
            "failed": [],
            "skipped": []
        }
        
        # Create a simple test audio file
        self.create_test_audio()
    
    def create_test_audio(self):
        """Create a simple test audio file."""
        try:
            import numpy as np
            import soundfile as sf
            
            # Generate a simple sine wave
            duration = 2.0  # 2 seconds
            sample_rate = 22050
            frequency = 440.0  # A4 note
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = np.sin(2 * np.pi * frequency * t)
            
            # Add some noise
            noise = np.random.normal(0, 0.1, len(audio))
            audio = audio + noise
            
            # Normalize
            audio = audio / np.max(np.abs(audio))
            
            # Save as WAV
            sf.write(self.test_audio_path, audio, sample_rate)
            logger.info(f"Created test audio file: {self.test_audio_path}")
            
        except Exception as e:
            logger.error(f"Failed to create test audio: {e}")
            self.test_audio_path = None
    
    def test_extractor(self, extractor_class, extractor_name):
        """Test a single extractor."""
        try:
            logger.info(f"Testing {extractor_name}...")
            
            # Initialize extractor with CPU fallback
            extractor = extractor_class(device="cpu")
            
            # Verify device is set to CPU
            if hasattr(extractor, 'device'):
                assert extractor.device == "cpu", f"Device should be 'cpu', got '{extractor.device}'"
            else:
                logger.warning(f"{extractor_name} doesn't have device attribute")
            
            # Test extraction
            if self.test_audio_path and os.path.exists(self.test_audio_path):
                result = extractor.run(self.test_audio_path, self.tmp_dir)
                
                # Verify result
                assert result.success, f"Extraction failed: {result.error}"
                assert result.payload is not None, "Payload should not be None"
                
                # Check if device info is included
                if isinstance(result.payload, dict):
                    if "device_used" in result.payload:
                        assert result.payload["device_used"] == "cpu", f"Device should be 'cpu', got '{result.payload['device_used']}'"
                    if "gpu_accelerated" in result.payload:
                        assert result.payload["gpu_accelerated"] == False, "Should not be GPU accelerated"
                
                logger.info(f"✅ {extractor_name} passed")
                self.results["passed"].append(extractor_name)
            else:
                logger.warning(f"⏭️  {extractor_name} skipped (no test audio)")
                self.results["skipped"].append(extractor_name)
                
        except Exception as e:
            logger.error(f"❌ {extractor_name} failed: {e}")
            self.results["failed"].append(extractor_name)
    
    def test_all_extractors(self):
        """Test all available extractors."""
        logger.info("🧪 Testing all extractors with CPU fallback...")
        
        # List of all extractors to test
        extractors_to_test = [
            (ASRExtractor, "ASR Extractor"),
            (AdvancedEmbeddingsExtractor, "Advanced Embeddings Extractor"),
            (SpeakerDiarizationExtractor, "Speaker Diarization Extractor"),
            (CLAPExtractor, "CLAP Extractor"),
            (AdvancedSpectralExtractor, "Advanced Spectral Extractor"),
            (MFCCExtractor, "MFCC Extractor"),
            (ChromaExtractor, "Chroma Extractor"),
            (MelExtractor, "Mel Extractor"),
            (LoudnessExtractor, "Loudness Extractor"),
            (PitchExtractor, "Pitch Extractor"),
            (VADExtractor, "VAD Extractor"),
            (TempoExtractor, "Tempo Extractor"),
            (QualityExtractor, "Quality Extractor"),
            (OnsetExtractor, "Onset Extractor"),
            (VoiceQualityExtractor, "Voice Quality Extractor"),
            (PhonemeAnalysisExtractor, "Phoneme Analysis Extractor"),
            (RhythmicAnalysisExtractor, "Rhythmic Analysis Extractor"),
            (MusicAnalysisExtractor, "Music Analysis Extractor"),
            (VideoAudioExtractor, "Video Audio Extractor"),
            (SoundEventDetectionExtractor, "Sound Event Detection Extractor"),
            (SourceSeparationExtractor, "Source Separation Extractor"),
            (EmotionRecognitionExtractor, "Emotion Recognition Extractor")
        ]
        
        for extractor_class, extractor_name in extractors_to_test:
            try:
                self.test_extractor(extractor_class, extractor_name)
            except Exception as e:
                logger.error(f"Failed to test {extractor_name}: {e}")
                self.results["failed"].append(extractor_name)
    
    def print_results(self):
        """Print test results."""
        logger.info("\n📊 Test Results:")
        logger.info(f"✅ Passed: {len(self.results['passed'])} extractors")
        logger.info(f"❌ Failed: {len(self.results['failed'])} extractors")
        logger.info(f"⏭️  Skipped: {len(self.results['skipped'])} extractors")
        
        if self.results["passed"]:
            logger.info(f"\n✅ Passed extractors: {', '.join(self.results['passed'])}")
        
        if self.results["failed"]:
            logger.info(f"\n❌ Failed extractors: {', '.join(self.results['failed'])}")
        
        if self.results["skipped"]:
            logger.info(f"\n⏭️  Skipped extractors: {', '.join(self.results['skipped'])}")
        
        # Overall result
        total_tested = len(self.results["passed"]) + len(self.results["failed"])
        if total_tested > 0:
            success_rate = len(self.results["passed"]) / total_tested * 100
            logger.info(f"\n🎯 Success rate: {success_rate:.1f}%")
            
            if success_rate >= 90:
                logger.info("🎉 Excellent! Most extractors have proper CPU fallback.")
            elif success_rate >= 70:
                logger.info("👍 Good! Most extractors work on CPU.")
            else:
                logger.info("⚠️  Some extractors need fallback improvements.")
    
    def cleanup(self):
        """Clean up test files."""
        try:
            if self.test_audio_path and os.path.exists(self.test_audio_path):
                os.remove(self.test_audio_path)
                logger.info("Cleaned up test audio file")
        except Exception as e:
            logger.warning(f"Failed to cleanup: {e}")

def main():
    """Main function to run CPU fallback tests."""
    logger.info("🚀 Starting CPU fallback tests...")
    
    # Verify CUDA is disabled
    if torch.cuda.is_available():
        logger.warning("⚠️  CUDA is available but should be disabled for testing")
    else:
        logger.info("✅ CUDA is disabled - testing CPU fallback")
    
    tester = CPUFallbackTester()
    
    try:
        tester.test_all_extractors()
        tester.print_results()
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()
