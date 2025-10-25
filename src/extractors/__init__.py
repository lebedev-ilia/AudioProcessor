"""
Audio feature extractors for AudioProcessor.
"""

from .mfcc_extractor import MFCCExtractor
from .mel_extractor import MelExtractor
from .chroma_extractor import ChromaExtractor
from .loudness_extractor import LoudnessExtractor
from .vad_extractor import VADExtractor
from .clap_extractor import CLAPExtractor


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
    ]


# Export all extractors
__all__ = [
    'MFCCExtractor',
    'MelExtractor', 
    'ChromaExtractor',
    'LoudnessExtractor',
    'VADExtractor',
    'CLAPExtractor',
    'discover_extractors'
]
