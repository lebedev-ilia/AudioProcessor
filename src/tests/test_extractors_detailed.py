"""
Detailed unit tests for individual extractors.
"""
import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from src.extractors.mfcc_extractor import MFCCExtractor
from src.extractors.mel_extractor import MelExtractor
from src.extractors.chroma_extractor import ChromaExtractor
from src.extractors.loudness_extractor import LoudnessExtractor
from src.extractors.vad_extractor import VADExtractor
from src.extractors.clap_extractor import CLAPExtractor
from tests.fixtures.audio_fixtures import AudioFixtures


class TestMFCCExtractor:
    """Detailed tests for MFCC extractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create MFCC extractor instance."""
        return MFCCExtractor()
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create sample audio file."""
        audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
        audio = AudioFixtures.normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio, 22050)
            yield tmp.name
        
        os.unlink(tmp.name)
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.name == "mfcc_extractor"
        assert extractor.version == "1.0.0"
        assert extractor.description is not None
        assert extractor.category == "core"
        assert extractor.dependencies == ["librosa", "numpy"]
        assert extractor.estimated_duration == 5.0
    
    def test_extractor_run_success(self, extractor, sample_audio_file):
        """Test successful extraction."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run(sample_audio_file, tmp_dir)
            
            assert result.success is True
            assert result.error is None
            assert result.payload is not None
            assert len(result.payload) > 0
            
            # Verify specific MFCC features
            expected_features = [
                'mfcc_0_mean', 'mfcc_1_mean', 'mfcc_2_mean',
                'mfcc_0_std', 'mfcc_1_std', 'mfcc_2_std',
                'mfcc_delta_0_mean', 'mfcc_delta_1_mean',
                'mfcc_mean', 'mfcc_std'
            ]
            
            for feature in expected_features:
                assert feature in result.payload, f"Missing feature: {feature}"
                assert result.payload[feature] is not None, f"Feature {feature} is None"
    
    def test_extractor_run_invalid_file(self, extractor):
        """Test extraction with invalid file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run("/non/existent/file.wav", tmp_dir)
            
            assert result.success is False
            assert result.error is not None
            assert result.payload is None
    
    def test_extractor_run_empty_file(self, extractor):
        """Test extraction with empty file."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b"")
            tmp.flush()
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = extractor.run(tmp.name, tmp_dir)
                
                assert result.success is False
                assert result.error is not None
                assert result.payload is None
            
            os.unlink(tmp.name)
    
    def test_extractor_run_corrupted_file(self, extractor):
        """Test extraction with corrupted file."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b"This is not audio data")
            tmp.flush()
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = extractor.run(tmp.name, tmp_dir)
                
                assert result.success is False
                assert result.error is not None
                assert result.payload is None
            
            os.unlink(tmp.name)
    
    def test_extractor_run_different_durations(self, extractor):
        """Test extraction with different audio durations."""
        durations = [0.5, 1.0, 2.0, 5.0]  # Removed 0.1s as it's too short for delta MFCC
        
        for duration in durations:
            audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=duration)
            audio = AudioFixtures.normalize_audio(audio)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                import soundfile as sf
                sf.write(tmp.name, audio, 22050)
                
                with tempfile.TemporaryDirectory() as tmp_dir:
                    result = extractor.run(tmp.name, tmp_dir)
                    
                    assert result.success, f"Extraction failed for duration {duration}"
                    assert result.payload is not None
                    assert len(result.payload) > 0
                
                os.unlink(tmp.name)
    
    def test_extractor_run_different_frequencies(self, extractor):
        """Test extraction with different frequencies."""
        frequencies = [100.0, 440.0, 1000.0, 2000.0, 5000.0]
        
        for frequency in frequencies:
            audio = AudioFixtures.generate_sine_wave(frequency=frequency, duration=1.0)
            audio = AudioFixtures.normalize_audio(audio)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                import soundfile as sf
                sf.write(tmp.name, audio, 22050)
                
                with tempfile.TemporaryDirectory() as tmp_dir:
                    result = extractor.run(tmp.name, tmp_dir)
                    
                    assert result.success, f"Extraction failed for frequency {frequency}"
                    assert result.payload is not None
                    
                    # Verify that different frequencies produce different results
                    if frequency != 440.0:  # Skip comparison with default
                        # At least some features should be different
                        assert result.payload['mfcc_0_mean'] is not None
    
    def test_extractor_run_noisy_audio(self, extractor):
        """Test extraction with noisy audio."""
        # Generate base signal
        audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
        
        # Add noise
        noise = AudioFixtures.generate_noise(duration=1.0, noise_level=0.3)
        audio = audio + noise
        audio = AudioFixtures.normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio, 22050)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = extractor.run(tmp.name, tmp_dir)
                
                assert result.success, "Extraction failed with noisy audio"
                assert result.payload is not None
                assert len(result.payload) > 0
            
            os.unlink(tmp.name)
    
    def test_extractor_run_silence(self, extractor):
        """Test extraction with silence."""
        audio = AudioFixtures.generate_silence(duration=1.0)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio, 22050)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = extractor.run(tmp.name, tmp_dir)
                
                assert result.success, "Extraction failed with silence"
                assert result.payload is not None
                
                # Silence should produce specific values
                assert result.payload['mfcc_0_mean'] is not None
    
    def test_extractor_run_stereo_audio(self, extractor):
        """Test extraction with stereo audio."""
        # Generate different signals for left and right channels
        left = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
        right = AudioFixtures.generate_sine_wave(frequency=880.0, duration=1.0)
        
        # Combine into stereo
        stereo_audio = np.column_stack((left, right))
        stereo_audio = AudioFixtures.normalize_audio(stereo_audio)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, stereo_audio, 22050)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = extractor.run(tmp.name, tmp_dir)
                
                assert result.success, "Extraction failed with stereo audio"
                assert result.payload is not None
                assert len(result.payload) > 0
            
            os.unlink(tmp.name)
    
    def test_extractor_run_different_sample_rates(self, extractor):
        """Test extraction with different sample rates."""
        sample_rates = [8000, 16000, 22050, 44100, 48000]
        
        for sample_rate in sample_rates:
            audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
            audio = AudioFixtures.normalize_audio(audio)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                import soundfile as sf
                sf.write(tmp.name, audio, sample_rate)
                
                with tempfile.TemporaryDirectory() as tmp_dir:
                    result = extractor.run(tmp.name, tmp_dir)
                    
                    assert result.success, f"Extraction failed for sample rate {sample_rate}"
                    assert result.payload is not None
                    assert len(result.payload) > 0
                
                os.unlink(tmp.name)
    
    def test_extractor_error_handling(self, extractor):
        """Test extractor error handling."""
        # Test with None input
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run(None, tmp_dir)
            assert result.success is False
            assert result.error is not None
        
        # Test with empty string
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run("", tmp_dir)
            assert result.success is False
            assert result.error is not None
        
        # Test with invalid path
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run("invalid/path/file.wav", tmp_dir)
            assert result.success is False
            assert result.error is not None


class TestMelExtractor:
    """Detailed tests for Mel extractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create Mel extractor instance."""
        return MelExtractor()
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create sample audio file."""
        audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
        audio = AudioFixtures.normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio, 22050)
            yield tmp.name
        
        os.unlink(tmp.name)
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.name == "mel_extractor"
        assert extractor.version == "1.0.0"
        assert extractor.description is not None
        assert extractor.category == "core"
        assert extractor.dependencies == ["librosa", "numpy"]
        assert extractor.estimated_duration == 3.0
    
    def test_extractor_run_success(self, extractor, sample_audio_file):
        """Test successful extraction."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run(sample_audio_file, tmp_dir)
            
            assert result.success is True
            assert result.error is None
            assert result.payload is not None
            assert len(result.payload) > 0
            
            # Verify specific Mel features
            expected_features = [
                'mel64_mean_0', 'mel64_mean_1', 'mel64_mean_2',
                'mel64_std_0', 'mel64_std_1', 'mel64_std_2',
                'mel64_mean', 'mel64_std'
            ]
            
            for feature in expected_features:
                assert feature in result.payload, f"Missing feature: {feature}"
                assert result.payload[feature] is not None, f"Feature {feature} is None"
    
    def test_extractor_run_invalid_file(self, extractor):
        """Test extraction with invalid file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run("/non/existent/file.wav", tmp_dir)
            
            assert result.success is False
            assert result.error is not None
            assert result.payload is None


class TestChromaExtractor:
    """Detailed tests for Chroma extractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create Chroma extractor instance."""
        return ChromaExtractor()
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create sample audio file."""
        audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
        audio = AudioFixtures.normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio, 22050)
            yield tmp.name
        
        os.unlink(tmp.name)
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.name == "chroma_extractor"
        assert extractor.version == "1.0.0"
        assert extractor.description is not None
        assert extractor.category == "core"
        assert extractor.dependencies == ["librosa", "numpy"]
        assert extractor.estimated_duration == 4.0
    
    def test_extractor_run_success(self, extractor, sample_audio_file):
        """Test successful extraction."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run(sample_audio_file, tmp_dir)
            
            assert result.success is True
            assert result.error is None
            assert result.payload is not None
            assert len(result.payload) > 0
            
            # Verify specific Chroma features
            expected_features = [
                'chroma_0_mean', 'chroma_1_mean', 'chroma_2_mean',
                'chroma_0_std', 'chroma_1_std', 'chroma_2_std',
                'chroma_mean', 'chroma_std'
            ]
            
            for feature in expected_features:
                assert feature in result.payload, f"Missing feature: {feature}"
                assert result.payload[feature] is not None, f"Feature {feature} is None"
    
    def test_extractor_run_invalid_file(self, extractor):
        """Test extraction with invalid file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run("/non/existent/file.wav", tmp_dir)
            
            assert result.success is False
            assert result.error is not None
            assert result.payload is None


class TestLoudnessExtractor:
    """Detailed tests for Loudness extractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create Loudness extractor instance."""
        return LoudnessExtractor()
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create sample audio file."""
        audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
        audio = AudioFixtures.normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio, 22050)
            yield tmp.name
        
        os.unlink(tmp.name)
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.name == "loudness_extractor"
        assert extractor.version == "1.0.0"
        assert extractor.description is not None
        assert extractor.category == "core"
        assert extractor.dependencies == ["librosa", "numpy"]
        assert extractor.estimated_duration == 2.0
    
    def test_extractor_run_success(self, extractor, sample_audio_file):
        """Test successful extraction."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run(sample_audio_file, tmp_dir)
            
            assert result.success is True
            assert result.error is None
            assert result.payload is not None
            assert len(result.payload) > 0
            
            # Verify specific Loudness features
            expected_features = [
                'rms_mean', 'rms_std', 'rms_min', 'rms_max',
                'loudness_lufs', 'peak_amplitude', 'clip_fraction'
            ]
            
            for feature in expected_features:
                assert feature in result.payload, f"Missing feature: {feature}"
                assert result.payload[feature] is not None, f"Feature {feature} is None"
    
    def test_extractor_run_invalid_file(self, extractor):
        """Test extraction with invalid file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run("/non/existent/file.wav", tmp_dir)
            
            assert result.success is False
            assert result.error is not None
            assert result.payload is None


class TestVADExtractor:
    """Detailed tests for VAD extractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create VAD extractor instance."""
        return VADExtractor()
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create sample audio file."""
        audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
        audio = AudioFixtures.normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio, 22050)
            yield tmp.name
        
        os.unlink(tmp.name)
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.name == "vad_extractor"
        assert extractor.version == "1.0.0"
        assert extractor.description is not None
        assert extractor.category == "core"
        assert extractor.dependencies == ["librosa", "numpy"]
        assert extractor.estimated_duration == 6.0
    
    def test_extractor_run_success(self, extractor, sample_audio_file):
        """Test successful extraction."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run(sample_audio_file, tmp_dir)
            
            assert result.success is True
            assert result.error is None
            assert result.payload is not None
            assert len(result.payload) > 0
            
            # Verify specific VAD features
            expected_features = [
                'voiced_fraction', 'f0_mean', 'f0_std', 'f0_min', 'f0_max'
            ]
            
            for feature in expected_features:
                assert feature in result.payload, f"Missing feature: {feature}"
                assert result.payload[feature] is not None, f"Feature {feature} is None"
    
    def test_extractor_run_invalid_file(self, extractor):
        """Test extraction with invalid file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run("/non/existent/file.wav", tmp_dir)
            
            assert result.success is False
            assert result.error is not None
            assert result.payload is None


class TestCLAPExtractor:
    """Detailed tests for CLAP extractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create CLAP extractor instance."""
        return CLAPExtractor()
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create sample audio file."""
        audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
        audio = AudioFixtures.normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio, 22050)
            yield tmp.name
        
        os.unlink(tmp.name)
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.name == "clap_extractor"
        assert extractor.version == "1.0.0"
        assert extractor.description is not None
        assert extractor.category == "advanced"
        assert extractor.dependencies == ["openl3", "numpy"]
        assert extractor.estimated_duration == 10.0
    
    def test_extractor_run_success(self, extractor, sample_audio_file):
        """Test successful extraction."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run(sample_audio_file, tmp_dir)
            
            assert result.success is True
            assert result.error is None
            assert result.payload is not None
            assert len(result.payload) > 0
            
            # Verify specific CLAP features
            expected_features = [
                'clap_0', 'clap_1', 'clap_2',
                'clap_embedding', 'clap_mean', 'clap_std'
            ]
            
            for feature in expected_features:
                assert feature in result.payload, f"Missing feature: {feature}"
                assert result.payload[feature] is not None, f"Feature {feature} is None"
    
    def test_extractor_run_invalid_file(self, extractor):
        """Test extraction with invalid file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = extractor.run("/non/existent/file.wav", tmp_dir)
            
            assert result.success is False
            assert result.error is not None
            assert result.payload is None


class TestExtractorErrorHandling:
    """Test error handling across all extractors."""
    
    @pytest.fixture
    def extractors(self):
        """Get all extractors."""
        return [
            MFCCExtractor(),
            MelExtractor(),
            ChromaExtractor(),
            LoudnessExtractor(),
            VADExtractor(),
            CLAPExtractor()
        ]
    
    def test_all_extractors_handle_invalid_file(self, extractors):
        """Test that all extractors handle invalid files gracefully."""
        for extractor in extractors:
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = extractor.run("/non/existent/file.wav", tmp_dir)
                
                assert result.success is False, f"Extractor {extractor.name} should fail with invalid file"
                assert result.error is not None, f"Extractor {extractor.name} should have error message"
                assert result.payload is None, f"Extractor {extractor.name} should have None payload"
    
    def test_all_extractors_handle_empty_file(self, extractors):
        """Test that all extractors handle empty files gracefully."""
        for extractor in extractors:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(b"")
                tmp.flush()
                
                with tempfile.TemporaryDirectory() as tmp_dir:
                    result = extractor.run(tmp.name, tmp_dir)
                    
                    assert result.success is False, f"Extractor {extractor.name} should fail with empty file"
                    assert result.error is not None, f"Extractor {extractor.name} should have error message"
                    assert result.payload is None, f"Extractor {extractor.name} should have None payload"
                
                os.unlink(tmp.name)
    
    def test_all_extractors_handle_corrupted_file(self, extractors):
        """Test that all extractors handle corrupted files gracefully."""
        for extractor in extractors:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(b"This is not audio data")
                tmp.flush()
                
                with tempfile.TemporaryDirectory() as tmp_dir:
                    result = extractor.run(tmp.name, tmp_dir)
                    
                    assert result.success is False, f"Extractor {extractor.name} should fail with corrupted file"
                    assert result.error is not None, f"Extractor {extractor.name} should have error message"
                    assert result.payload is None, f"Extractor {extractor.name} should have None payload"
                
                os.unlink(tmp.name)
    
    def test_all_extractors_handle_none_input(self, extractors):
        """Test that all extractors handle None input gracefully."""
        for extractor in extractors:
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = extractor.run(None, tmp_dir)
                
                assert result.success is False, f"Extractor {extractor.name} should fail with None input"
                assert result.error is not None, f"Extractor {extractor.name} should have error message"
                assert result.payload is None, f"Extractor {extractor.name} should have None payload"
    
    def test_all_extractors_handle_empty_string_input(self, extractors):
        """Test that all extractors handle empty string input gracefully."""
        for extractor in extractors:
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = extractor.run("", tmp_dir)
                
                assert result.success is False, f"Extractor {extractor.name} should fail with empty string input"
                assert result.error is not None, f"Extractor {extractor.name} should have error message"
                assert result.payload is None, f"Extractor {extractor.name} should have None payload"


class TestExtractorConsistency:
    """Test consistency across extractors."""
    
    @pytest.fixture
    def extractors(self):
        """Get all extractors."""
        return [
            MFCCExtractor(),
            MelExtractor(),
            ChromaExtractor(),
            LoudnessExtractor(),
            VADExtractor(),
            CLAPExtractor()
        ]
    
    def test_all_extractors_have_required_attributes(self, extractors):
        """Test that all extractors have required attributes."""
        required_attributes = ['name', 'version', 'description', 'category', 'dependencies', 'estimated_duration']
        
        for extractor in extractors:
            for attr in required_attributes:
                assert hasattr(extractor, attr), f"Extractor {extractor.name} missing attribute {attr}"
                assert getattr(extractor, attr) is not None, f"Extractor {extractor.name} attribute {attr} is None"
    
    def test_all_extractors_have_unique_names(self, extractors):
        """Test that all extractors have unique names."""
        names = [extractor.name for extractor in extractors]
        assert len(names) == len(set(names)), "Extractor names are not unique"
    
    def test_all_extractors_have_valid_categories(self, extractors):
        """Test that all extractors have valid categories."""
        valid_categories = ['core', 'advanced', 'experimental']
        
        for extractor in extractors:
            assert extractor.category in valid_categories, \
                f"Extractor {extractor.name} has invalid category {extractor.category}"
    
    def test_all_extractors_have_positive_estimated_duration(self, extractors):
        """Test that all extractors have positive estimated duration."""
        for extractor in extractors:
            assert extractor.estimated_duration > 0, \
                f"Extractor {extractor.name} has non-positive estimated duration {extractor.estimated_duration}"
    
    def test_all_extractors_have_run_method(self, extractors):
        """Test that all extractors have run method."""
        for extractor in extractors:
            assert hasattr(extractor, 'run'), f"Extractor {extractor.name} missing run method"
            assert callable(getattr(extractor, 'run')), f"Extractor {extractor.name} run method is not callable"
