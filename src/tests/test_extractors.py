"""
Test suite for audio extractors.
"""

import pytest
import numpy as np
import soundfile as sf
import tempfile
import os
from extractors import discover_extractors


@pytest.fixture
def sample_audio_file():
    """Create a sample audio file for testing."""
    # Generate a test audio signal (1 second of 440 Hz sine wave)
    sample_rate = 22050
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some noise for more realistic testing
    noise = np.random.normal(0, 0.1, len(audio))
    audio = audio + noise
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, sample_rate)
        yield tmp.name
    
    # Cleanup
    os.unlink(tmp.name)


def test_extractor_discovery():
    """Test that all extractors can be discovered."""
    extractors = discover_extractors()
    
    assert len(extractors) == 6, f"Expected 6 extractors, got {len(extractors)}"
    
    # Check that all expected extractors are present
    extractor_names = [extractor.name for extractor in extractors]
    expected_names = [
        'mfcc_extractor',
        'mel_extractor', 
        'chroma_extractor',
        'loudness_extractor',
        'vad_extractor',
        'clap_extractor'
    ]
    
    for expected_name in expected_names:
        assert expected_name in extractor_names, f"Missing extractor: {expected_name}"


def test_mfcc_extractor(sample_audio_file):
    """Test MFCC extractor."""
    from extractors.mfcc_extractor import MFCCExtractor
    
    extractor = MFCCExtractor()
    result = extractor.run(sample_audio_file, "/tmp")
    
    assert result.success, f"MFCC extraction failed: {result.error}"
    assert result.payload is not None, "MFCC payload is None"
    
    # Check that we have the expected features
    expected_features = [
        'mfcc_0_mean', 'mfcc_1_mean', 'mfcc_2_mean',  # First few MFCC means
        'mfcc_0_std', 'mfcc_1_std', 'mfcc_2_std',     # First few MFCC stds
        'mfcc_delta_0_mean', 'mfcc_delta_1_mean',     # First few delta means
        'mfcc_mean', 'mfcc_std'                        # Overall arrays
    ]
    
    for feature in expected_features:
        assert feature in result.payload, f"Missing MFCC feature: {feature}"


def test_mel_extractor(sample_audio_file):
    """Test Mel spectrogram extractor."""
    from extractors.mel_extractor import MelExtractor
    
    extractor = MelExtractor()
    result = extractor.run(sample_audio_file, "/tmp")
    
    assert result.success, f"Mel extraction failed: {result.error}"
    assert result.payload is not None, "Mel payload is None"
    
    # Check that we have the expected features
    expected_features = [
        'mel64_mean_0', 'mel64_mean_1', 'mel64_mean_2',  # First few mel means
        'mel64_std_0', 'mel64_std_1', 'mel64_std_2',     # First few mel stds
        'mel64_mean', 'mel64_std'                         # Overall arrays
    ]
    
    for feature in expected_features:
        assert feature in result.payload, f"Missing Mel feature: {feature}"


def test_chroma_extractor(sample_audio_file):
    """Test Chroma extractor."""
    from extractors.chroma_extractor import ChromaExtractor
    
    extractor = ChromaExtractor()
    result = extractor.run(sample_audio_file, "/tmp")
    
    assert result.success, f"Chroma extraction failed: {result.error}"
    assert result.payload is not None, "Chroma payload is None"
    
    # Check that we have the expected features
    expected_features = [
        'chroma_0_mean', 'chroma_1_mean', 'chroma_2_mean',  # First few chroma means
        'chroma_0_std', 'chroma_1_std', 'chroma_2_std',     # First few chroma stds
        'chroma_mean', 'chroma_std'                          # Overall arrays
    ]
    
    for feature in expected_features:
        assert feature in result.payload, f"Missing Chroma feature: {feature}"


def test_loudness_extractor(sample_audio_file):
    """Test Loudness extractor."""
    from extractors.loudness_extractor import LoudnessExtractor
    
    extractor = LoudnessExtractor()
    result = extractor.run(sample_audio_file, "/tmp")
    
    assert result.success, f"Loudness extraction failed: {result.error}"
    assert result.payload is not None, "Loudness payload is None"
    
    # Check that we have the expected features
    expected_features = [
        'rms_mean', 'rms_std', 'rms_min', 'rms_max',
        'loudness_lufs', 'peak_amplitude', 'clip_fraction'
    ]
    
    for feature in expected_features:
        assert feature in result.payload, f"Missing Loudness feature: {feature}"


def test_vad_extractor(sample_audio_file):
    """Test VAD extractor."""
    from extractors.vad_extractor import VADExtractor
    
    extractor = VADExtractor()
    result = extractor.run(sample_audio_file, "/tmp")
    
    assert result.success, f"VAD extraction failed: {result.error}"
    assert result.payload is not None, "VAD payload is None"
    
    # Check that we have the expected features
    expected_features = [
        'voiced_fraction', 'f0_mean', 'f0_std', 'f0_min', 'f0_max'
    ]
    
    for feature in expected_features:
        assert feature in result.payload, f"Missing VAD feature: {feature}"


def test_clap_extractor(sample_audio_file):
    """Test CLAP extractor."""
    from extractors.clap_extractor import CLAPExtractor
    
    extractor = CLAPExtractor()
    result = extractor.run(sample_audio_file, "/tmp")
    
    assert result.success, f"CLAP extraction failed: {result.error}"
    assert result.payload is not None, "CLAP payload is None"
    
    # Check that we have the expected features
    expected_features = [
        'clap_0', 'clap_1', 'clap_2',  # First few CLAP dimensions
        'clap_embedding', 'clap_mean', 'clap_std'
    ]
    
    for feature in expected_features:
        assert feature in result.payload, f"Missing CLAP feature: {feature}"


def test_all_extractors_integration(sample_audio_file):
    """Test that all extractors work together."""
    extractors = discover_extractors()
    
    results = []
    for extractor in extractors:
        result = extractor.run(sample_audio_file, "/tmp")
        results.append(result)
        
        assert result.success, f"Extractor {extractor.name} failed: {result.error}"
        assert result.payload is not None, f"Extractor {extractor.name} payload is None"
    
    # Check that we have results from all extractors
    assert len(results) == 6, f"Expected 6 results, got {len(results)}"
    
    # Check that all extractors have different names
    extractor_names = [result.name for result in results]
    assert len(set(extractor_names)) == 6, "Duplicate extractor names found"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
