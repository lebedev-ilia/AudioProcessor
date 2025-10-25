"""
Audio fixtures for testing.
"""
import pytest
import numpy as np
import soundfile as sf
import tempfile
import os
from typing import Generator, Tuple


class AudioFixtures:
    """Audio fixtures for testing."""
    
    @staticmethod
    def generate_sine_wave(frequency: float = 440.0, duration: float = 1.0, 
                          sample_rate: int = 22050, amplitude: float = 0.5) -> np.ndarray:
        """Generate a sine wave audio signal."""
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = amplitude * np.sin(2 * np.pi * frequency * t)
        return audio
    
    @staticmethod
    def generate_noise(duration: float = 1.0, sample_rate: int = 22050, 
                      noise_level: float = 0.1) -> np.ndarray:
        """Generate white noise audio signal."""
        length = int(sample_rate * duration)
        noise = np.random.normal(0, noise_level, length)
        return noise
    
    @staticmethod
    def generate_chirp(start_freq: float = 100.0, end_freq: float = 1000.0, 
                      duration: float = 1.0, sample_rate: int = 22050) -> np.ndarray:
        """Generate a frequency chirp (sweep) signal."""
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Linear frequency sweep
        freq_sweep = start_freq + (end_freq - start_freq) * t / duration
        audio = 0.5 * np.sin(2 * np.pi * freq_sweep * t)
        return audio
    
    @staticmethod
    def generate_silence(duration: float = 1.0, sample_rate: int = 22050) -> np.ndarray:
        """Generate silence (zeros)."""
        length = int(sample_rate * duration)
        return np.zeros(length)
    
    @staticmethod
    def add_reverb(audio: np.ndarray, reverb_delay: int = 1000, 
                   reverb_decay: float = 0.3) -> np.ndarray:
        """Add simple reverb effect."""
        reverb = np.zeros_like(audio)
        for i in range(reverb_delay, len(audio)):
            reverb[i] = audio[i] + reverb_decay * reverb[i - reverb_delay]
        return reverb
    
    @staticmethod
    def add_clipping(audio: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """Add clipping distortion."""
        clipped = np.copy(audio)
        clipped[clipped > threshold] = threshold
        clipped[clipped < -threshold] = -threshold
        return clipped
    
    @staticmethod
    def normalize_audio(audio: np.ndarray, target_level: float = 0.8) -> np.ndarray:
        """Normalize audio to target level."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio * (target_level / max_val)
        return audio


@pytest.fixture
def sample_audio_file() -> Generator[str, None, None]:
    """Create a sample audio file for testing (1 second, 440 Hz sine wave)."""
    audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
    audio = AudioFixtures.normalize_audio(audio)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, 22050)
        yield tmp.name
    
    os.unlink(tmp.name)


@pytest.fixture
def short_audio_file() -> Generator[str, None, None]:
    """Create a short audio file (0.5 seconds)."""
    audio = AudioFixtures.generate_sine_wave(frequency=880.0, duration=0.5)
    audio = AudioFixtures.normalize_audio(audio)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, 22050)
        yield tmp.name
    
    os.unlink(tmp.name)


@pytest.fixture
def long_audio_file() -> Generator[str, None, None]:
    """Create a long audio file (10 seconds)."""
    audio = AudioFixtures.generate_sine_wave(frequency=220.0, duration=10.0)
    audio = AudioFixtures.normalize_audio(audio)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, 22050)
        yield tmp.name
    
    os.unlink(tmp.name)


@pytest.fixture
def noisy_audio_file() -> Generator[str, None, None]:
    """Create a noisy audio file."""
    # Generate base signal
    audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=2.0)
    
    # Add noise
    noise = AudioFixtures.generate_noise(duration=2.0, noise_level=0.2)
    audio = audio + noise
    
    # Normalize
    audio = AudioFixtures.normalize_audio(audio)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, 22050)
        yield tmp.name
    
    os.unlink(tmp.name)


@pytest.fixture
def chirp_audio_file() -> Generator[str, None, None]:
    """Create a chirp (frequency sweep) audio file."""
    audio = AudioFixtures.generate_chirp(start_freq=100.0, end_freq=2000.0, duration=3.0)
    audio = AudioFixtures.normalize_audio(audio)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, 22050)
        yield tmp.name
    
    os.unlink(tmp.name)


@pytest.fixture
def silence_audio_file() -> Generator[str, None, None]:
    """Create a silence audio file."""
    audio = AudioFixtures.generate_silence(duration=1.0)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, 22050)
        yield tmp.name
    
    os.unlink(tmp.name)


@pytest.fixture
def clipped_audio_file() -> Generator[str, None, None]:
    """Create a clipped (distorted) audio file."""
    audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0, amplitude=1.5)
    audio = AudioFixtures.add_clipping(audio, threshold=0.8)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, 22050)
        yield tmp.name
    
    os.unlink(tmp.name)


@pytest.fixture
def reverb_audio_file() -> Generator[str, None, None]:
    """Create an audio file with reverb."""
    audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=2.0)
    audio = AudioFixtures.add_reverb(audio, reverb_delay=500, reverb_decay=0.3)
    audio = AudioFixtures.normalize_audio(audio)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, 22050)
        yield tmp.name
    
    os.unlink(tmp.name)


@pytest.fixture
def low_sample_rate_audio_file() -> Generator[str, None, None]:
    """Create an audio file with low sample rate (8kHz)."""
    audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
    audio = AudioFixtures.normalize_audio(audio)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, 8000)  # 8kHz sample rate
        yield tmp.name
    
    os.unlink(tmp.name)


@pytest.fixture
def high_sample_rate_audio_file() -> Generator[str, None, None]:
    """Create an audio file with high sample rate (48kHz)."""
    audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
    audio = AudioFixtures.normalize_audio(audio)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, 48000)  # 48kHz sample rate
        yield tmp.name
    
    os.unlink(tmp.name)


@pytest.fixture
def stereo_audio_file() -> Generator[str, None, None]:
    """Create a stereo audio file."""
    # Generate different signals for left and right channels
    left = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
    right = AudioFixtures.generate_sine_wave(frequency=880.0, duration=1.0)
    
    # Combine into stereo
    stereo_audio = np.column_stack((left, right))
    stereo_audio = AudioFixtures.normalize_audio(stereo_audio)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, stereo_audio, 22050)
        yield tmp.name
    
    os.unlink(tmp.name)


@pytest.fixture
def mono_audio_file() -> Generator[str, None, None]:
    """Create a mono audio file."""
    audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
    audio = AudioFixtures.normalize_audio(audio)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, 22050)
        yield tmp.name
    
    os.unlink(tmp.name)


@pytest.fixture
def corrupted_audio_file() -> Generator[str, None, None]:
    """Create a corrupted audio file (invalid format)."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        # Write invalid audio data
        tmp.write(b"This is not valid audio data")
        tmp.flush()
        yield tmp.name
    
    os.unlink(tmp.name)


@pytest.fixture
def empty_audio_file() -> Generator[str, None, None]:
    """Create an empty audio file."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        # Write empty file
        tmp.write(b"")
        tmp.flush()
        yield tmp.name
    
    os.unlink(tmp.name)


@pytest.fixture
def very_loud_audio_file() -> Generator[str, None, None]:
    """Create a very loud audio file."""
    audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0, amplitude=2.0)
    # Don't normalize to keep it loud
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, 22050)
        yield tmp.name
    
    os.unlink(tmp.name)


@pytest.fixture
def very_quiet_audio_file() -> Generator[str, None, None]:
    """Create a very quiet audio file."""
    audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0, amplitude=0.01)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, 22050)
        yield tmp.name
    
    os.unlink(tmp.name)


@pytest.fixture
def multi_tone_audio_file() -> Generator[str, None, None]:
    """Create an audio file with multiple tones."""
    # Generate multiple sine waves
    tone1 = AudioFixtures.generate_sine_wave(frequency=440.0, duration=2.0, amplitude=0.3)
    tone2 = AudioFixtures.generate_sine_wave(frequency=880.0, duration=2.0, amplitude=0.2)
    tone3 = AudioFixtures.generate_sine_wave(frequency=1320.0, duration=2.0, amplitude=0.1)
    
    # Combine tones
    audio = tone1 + tone2 + tone3
    audio = AudioFixtures.normalize_audio(audio)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, 22050)
        yield tmp.name
    
    os.unlink(tmp.name)


@pytest.fixture
def audio_file_with_gaps() -> Generator[str, None, None]:
    """Create an audio file with silent gaps."""
    # Generate signal with gaps
    signal1 = AudioFixtures.generate_sine_wave(frequency=440.0, duration=0.5, amplitude=0.5)
    gap1 = AudioFixtures.generate_silence(duration=0.2)
    signal2 = AudioFixtures.generate_sine_wave(frequency=880.0, duration=0.5, amplitude=0.5)
    gap2 = AudioFixtures.generate_silence(duration=0.3)
    signal3 = AudioFixtures.generate_sine_wave(frequency=1320.0, duration=0.5, amplitude=0.5)
    
    # Combine
    audio = np.concatenate([signal1, gap1, signal2, gap2, signal3])
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, 22050)
        yield tmp.name
    
    os.unlink(tmp.name)


# Fixture for testing different audio formats
@pytest.fixture(params=['.wav', '.flac', '.mp3'])
def audio_file_different_formats(request) -> Generator[str, None, None]:
    """Create audio files in different formats."""
    audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
    audio = AudioFixtures.normalize_audio(audio)
    
    with tempfile.NamedTemporaryFile(suffix=request.param, delete=False) as tmp:
        try:
            sf.write(tmp.name, audio, 22050)
            yield tmp.name
        except Exception:
            # Some formats might not be supported
            pytest.skip(f"Format {request.param} not supported")
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)


# Fixture for testing different durations
@pytest.fixture(params=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
def audio_file_different_durations(request) -> Generator[str, None, None]:
    """Create audio files with different durations."""
    duration = request.param
    audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=duration)
    audio = AudioFixtures.normalize_audio(audio)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, 22050)
        yield tmp.name
    
    os.unlink(tmp.name)


# Fixture for testing different sample rates
@pytest.fixture(params=[8000, 16000, 22050, 44100, 48000])
def audio_file_different_sample_rates(request) -> Generator[str, None, None]:
    """Create audio files with different sample rates."""
    sample_rate = request.param
    audio = AudioFixtures.generate_sine_wave(frequency=440.0, duration=1.0)
    audio = AudioFixtures.normalize_audio(audio)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, audio, sample_rate)
        yield tmp.name
    
    os.unlink(tmp.name)
