"""
Audio feature extractors for AudioProcessor.
"""

from .mfcc_extractor import MFCCExtractor
from .mel_extractor import MelExtractor
from .chroma_extractor import ChromaExtractor
from .loudness_extractor import LoudnessExtractor
from .vad_extractor import VADExtractor
from .clap_extractor import CLAPExtractor
from .asr_extractor import ASRExtractor
from .pitch_extractor import PitchExtractor
from .spectral_extractor import SpectralExtractor
from .tempo_extractor import TempoExtractor
from .quality_extractor import QualityExtractor
from .onset_extractor import OnsetExtractor
from .speaker_diarization_extractor import SpeakerDiarizationExtractor
from .voice_quality_extractor import VoiceQualityExtractor
from .emotion_recognition_extractor import EmotionRecognitionExtractor
from .phoneme_analysis_extractor import PhonemeAnalysisExtractor
from .advanced_spectral_extractor import AdvancedSpectralExtractor
from .music_analysis_extractor import MusicAnalysisExtractor
from .source_separation_extractor import SourceSeparationExtractor
from .sound_event_detection_extractor import SoundEventDetectionExtractor
from .rhythmic_analysis_extractor import RhythmicAnalysisExtractor
from .advanced_embeddings_extractor import AdvancedEmbeddingsExtractor


def discover_extractors():
    """
    Discover and return all available extractors.
    
    Returns:
        List of extractor instances
    """
    return [
        MFCCExtractor(),
        MelExtractor(),
        ChromaExtractor(),
        LoudnessExtractor(),
        VADExtractor(),
        CLAPExtractor(),
        ASRExtractor(),
        PitchExtractor(),
        SpectralExtractor(),
        TempoExtractor(),
        QualityExtractor(),
        OnsetExtractor(),
        SpeakerDiarizationExtractor(),
        VoiceQualityExtractor(),
        EmotionRecognitionExtractor(),
        PhonemeAnalysisExtractor(),
        AdvancedSpectralExtractor(),
        MusicAnalysisExtractor(),
        SourceSeparationExtractor(),
        SoundEventDetectionExtractor(),
        RhythmicAnalysisExtractor(),
        AdvancedEmbeddingsExtractor(),
    ]


# Export all extractors
__all__ = [
    'MFCCExtractor',
    'MelExtractor',
    'ChromaExtractor',
    'LoudnessExtractor',
    'VADExtractor',
    'CLAPExtractor',
    'ASRExtractor',
    'PitchExtractor',
    'SpectralExtractor',
    'TempoExtractor',
    'QualityExtractor',
    'OnsetExtractor',
    'SpeakerDiarizationExtractor',
    'VoiceQualityExtractor',
    'EmotionRecognitionExtractor',
    'PhonemeAnalysisExtractor',
    'AdvancedSpectralExtractor',
    'MusicAnalysisExtractor',
    'SourceSeparationExtractor',
    'SoundEventDetectionExtractor',
    'RhythmicAnalysisExtractor',
    'AdvancedEmbeddingsExtractor',
    'discover_extractors'
]
