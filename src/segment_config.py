"""
Configuration module for per-segment audio feature extraction.

This module contains default configuration parameters for converting
AudioProcessor outputs to per-segment features for AudioTransformer.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os


@dataclass
class SegmentConfig:
    """Configuration for per-segment feature extraction."""
    
    # Segment parameters
    segment_len: float = 3.0          # Segment length in seconds
    hop: float = 1.5                  # Hop length in seconds (segment_len * 0.5)
    max_seq_len: int = 128            # Maximum number of segments per video
    
    # Boundary selection parameters
    k_start: int = 16                 # Number of segments to keep from start
    k_end: int = 16                   # Number of segments to keep from end
    
    # Importance weights for segment selection
    importance_weights: Dict[str, float] = None
    
    # PCA compression dimensions
    pca_dims: Dict[str, int] = None
    
    # Scaler configuration
    scaler_type: str = "StandardScaler"  # or RobustScaler
    
    # ASR confidence threshold
    asr_confidence_threshold: float = 0.3
    
    # Storage format
    storage_format: str = "npy+json"  # or TFRecord / LMDB
    dtype: str = "float32"
    
    # Output directories
    output_dir: str = "segment_features"
    artifacts_dir: str = "artifacts"
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.importance_weights is None:
            self.importance_weights = {
                "rms": 0.6,
                "voiced_fraction": 0.4
            }
        
        if self.pca_dims is None:
            self.pca_dims = {
                "clap": 128,
                "wav2vec": 64,
                "yamnet": 128
            }
    
    def get_feature_mapping(self) -> Dict[str, str]:
        """
        Get mapping from AudioProcessor extractor outputs to segment features.
        
        Returns:
            Dictionary mapping extractor.field to segment_feature_name
        """
        return {
            # Embeddings (compressed)
            "clap_extractor.clap_embedding": "clap_pca",
            "advanced_embeddings.wav2vec_embeddings": "wav2vec_pca",
            "advanced_embeddings.yamnet_embeddings": "yamnet_pca",
            
            # Acoustic features
            "loudness_extractor.rms_mean": "rms_mean",
            "loudness_extractor.rms_std": "rms_std",
            "vad_extractor.f0_mean": "f0_mean",
            "vad_extractor.f0_std": "f0_std",
            "vad_extractor.voiced_fraction": "voiced_fraction",
            
            # Spectral features
            "spectral_extractor.spectral_centroid_mean": "spectral_centroid_mean",
            "spectral_extractor.spectral_bandwidth_mean": "spectral_bandwidth_mean",
            "spectral_extractor.spectral_flatness_mean": "spectral_flatness_mean",
            
            # Tempo/Onset features
            "tempo_extractor.tempo_bpm": "tempo_bpm",
            "tempo_extractor.onset_density": "onset_density",
            "onset_extractor.onset_count_energy": "onset_count_energy",
            
            # Source separation
            "source_separation_extractor.vocal_fraction": "vocal_fraction",
            
            # Emotion features
            "emotion_recognition_extractor.emotion_valence": "emotion_valence",
            "emotion_recognition_extractor.emotion_arousal": "emotion_arousal",
            "emotion_recognition_extractor.dominant_emotion_confidence": "emotion_dom_conf",
            
            # Quality features
            "quality_extractor.snr_estimate_db": "snr_db",
            "quality_extractor.hum_detected": "hum_detected",
            
            # ASR features
            "asr_extractor.words_count": "words_count",
            "asr_extractor.words_per_sec": "words_per_sec",
            "asr_extractor.transcript_confidence": "asr_conf_mean",
            
            # Mel/MFCC features
            "mel_extractor.mel64_mean": "mel_mean_vector",
            "mfcc_extractor.mfcc_mean": "mfcc_mean_vector"
        }
    
    def get_array_fields(self) -> Dict[str, str]:
        """
        Get mapping of array fields that need time-based slicing.
        
        Returns:
            Dictionary mapping extractor.array_field to time_field
        """
        return {
            "clap_extractor.clap_embedding": "clap_times",
            "advanced_embeddings.wav2vec_embeddings": "wav2vec_times",
            "advanced_embeddings.yamnet_embeddings": "yamnet_times",
            "loudness_extractor.rms_array": "rms_times",
            "vad_extractor.f0_array": "f0_times",
            "vad_extractor.voiced_flag_array": "f0_times",
            "mel_extractor.mel64_array": "mel_times",
            "mfcc_extractor.mfcc_array": "mfcc_times"
        }
    
    def get_scalar_fields(self) -> list:
        """
        Get list of scalar fields that don't need time-based slicing.
        
        Returns:
            List of scalar field names
        """
        return [
            "loudness_extractor.rms_mean",
            "loudness_extractor.rms_std",
            "vad_extractor.f0_mean",
            "vad_extractor.f0_std",
            "vad_extractor.voiced_fraction",
            "spectral_extractor.spectral_centroid_mean",
            "spectral_extractor.spectral_bandwidth_mean",
            "spectral_extractor.spectral_flatness_mean",
            "tempo_extractor.tempo_bpm",
            "tempo_extractor.onset_density",
            "onset_extractor.onset_count_energy",
            "source_separation_extractor.vocal_fraction",
            "emotion_recognition_extractor.emotion_valence",
            "emotion_recognition_extractor.emotion_arousal",
            "emotion_recognition_extractor.dominant_emotion_confidence",
            "quality_extractor.snr_estimate_db",
            "quality_extractor.hum_detected",
            "asr_extractor.words_count",
            "asr_extractor.words_per_sec",
            "asr_extractor.transcript_confidence"
        ]
    
    def get_artifact_paths(self) -> Dict[str, str]:
        """
        Get paths for saved artifacts (PCA models, scalers).
        
        Returns:
            Dictionary with artifact file paths
        """
        return {
            "pca_clap": os.path.join(self.artifacts_dir, "pca_clap.joblib"),
            "pca_wav2vec": os.path.join(self.artifacts_dir, "pca_wav2vec.joblib"),
            "pca_yamnet": os.path.join(self.artifacts_dir, "pca_yamnet.joblib"),
            "scaler": os.path.join(self.artifacts_dir, "scaler.joblib"),
            "config": os.path.join(self.artifacts_dir, "segment_config.json")
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "segment_len": self.segment_len,
            "hop": self.hop,
            "max_seq_len": self.max_seq_len,
            "k_start": self.k_start,
            "k_end": self.k_end,
            "importance_weights": self.importance_weights,
            "pca_dims": self.pca_dims,
            "scaler_type": self.scaler_type,
            "asr_confidence_threshold": self.asr_confidence_threshold,
            "storage_format": self.storage_format,
            "dtype": self.dtype,
            "output_dir": self.output_dir,
            "artifacts_dir": self.artifacts_dir
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SegmentConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


# Default configuration instance
DEFAULT_CONFIG = SegmentConfig()


def get_default_config() -> SegmentConfig:
    """Get default configuration instance."""
    return DEFAULT_CONFIG


def create_config(
    segment_len: float = 3.0,
    hop: float = 1.5,
    max_seq_len: int = 128,
    k_start: int = 16,
    k_end: int = 16,
    importance_weights: Optional[Dict[str, float]] = None,
    pca_dims: Optional[Dict[str, int]] = None,
    scaler_type: str = "StandardScaler",
    asr_confidence_threshold: float = 0.3,
    storage_format: str = "npy+json",
    dtype: str = "float32",
    output_dir: str = "segment_features",
    artifacts_dir: str = "artifacts"
) -> SegmentConfig:
    """
    Create a custom configuration instance.
    
    Args:
        segment_len: Segment length in seconds
        hop: Hop length in seconds
        max_seq_len: Maximum number of segments per video
        k_start: Number of segments to keep from start
        k_end: Number of segments to keep from end
        importance_weights: Weights for importance-based selection
        pca_dims: PCA dimensions for different embeddings
        scaler_type: Type of scaler to use
        asr_confidence_threshold: ASR confidence threshold
        storage_format: Storage format for features
        dtype: Data type for features
        output_dir: Output directory for features
        artifacts_dir: Directory for artifacts
        
    Returns:
        SegmentConfig instance
    """
    return SegmentConfig(
        segment_len=segment_len,
        hop=hop,
        max_seq_len=max_seq_len,
        k_start=k_start,
        k_end=k_end,
        importance_weights=importance_weights,
        pca_dims=pca_dims,
        scaler_type=scaler_type,
        asr_confidence_threshold=asr_confidence_threshold,
        storage_format=storage_format,
        dtype=dtype,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir
    )
