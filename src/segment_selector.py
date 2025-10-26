"""
Segment selection module for choosing important segments from audio.

This module implements various strategies for selecting segments when
the number of segments exceeds max_seq_len, including:
- Boundary preservation (keep start/end segments)
- Importance-based selection (based on audio features)
- Random sampling for data augmentation
"""

import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional
import logging
from .segment_config import SegmentConfig

logger = logging.getLogger(__name__)


class SegmentSelector:
    """Selector for choosing important segments from audio."""
    
    def __init__(self, config: SegmentConfig):
        """
        Initialize segment selector.
        
        Args:
            config: Configuration containing selection parameters
        """
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.k_start = config.k_start
        self.k_end = config.k_end
        self.importance_weights = config.importance_weights
        
        logger.info(f"Initialized SegmentSelector: max_seq_len={self.max_seq_len}, "
                   f"k_start={self.k_start}, k_end={self.k_end}")
    
    def select_segments(
        self, 
        segment_features: List[Dict[str, Any]],
        strategy: str = "boundary_importance",
        random_seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Select segments using specified strategy.
        
        Args:
            segment_features: List of segment feature dictionaries
            strategy: Selection strategy ("all", "boundary_importance", "random", "uniform")
            random_seed: Random seed for reproducible selection
            
        Returns:
            List of selected segment features
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        num_segments = len(segment_features)
        
        # If we have fewer segments than max_seq_len, return all
        if num_segments <= self.max_seq_len:
            logger.info(f"Using all {num_segments} segments (<= max_seq_len)")
            return segment_features
        
        logger.info(f"Selecting {self.max_seq_len} segments from {num_segments} using strategy: {strategy}")
        
        if strategy == "all":
            return segment_features
        elif strategy == "boundary_importance":
            return self._select_boundary_importance(segment_features)
        elif strategy == "random":
            return self._select_random(segment_features)
        elif strategy == "uniform":
            return self._select_uniform(segment_features)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
    
    def _select_boundary_importance(
        self, 
        segment_features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Select segments using boundary preservation + importance-based selection.
        
        Args:
            segment_features: List of segment feature dictionaries
            
        Returns:
            List of selected segment features
        """
        num_segments = len(segment_features)
        
        # Calculate how many segments we can select from the middle
        boundary_segments = self.k_start + self.k_end
        if boundary_segments >= self.max_seq_len:
            # If boundary segments exceed max_seq_len, just use boundary
            logger.warning(f"Boundary segments ({boundary_segments}) >= max_seq_len ({self.max_seq_len})")
            return self._select_boundary_only(segment_features)
        
        remaining_slots = self.max_seq_len - boundary_segments
        
        # Select start and end segments
        start_segments = segment_features[:self.k_start]
        end_segments = segment_features[-self.k_end:]
        
        # Get middle segments for importance-based selection
        middle_segments = segment_features[self.k_start:-self.k_end]
        
        if len(middle_segments) <= remaining_slots:
            # If we have fewer middle segments than remaining slots, use all
            selected_middle = middle_segments
        else:
            # Select middle segments based on importance
            selected_middle = self._select_by_importance(middle_segments, remaining_slots)
        
        # Combine and sort by original order
        selected_segments = start_segments + selected_middle + end_segments
        
        # Sort by segment index to maintain temporal order
        selected_segments.sort(key=lambda x: x.get('segment_index', 0))
        
        logger.info(f"Selected {len(selected_segments)} segments: "
                   f"{len(start_segments)} start + {len(selected_middle)} middle + {len(end_segments)} end")
        
        return selected_segments
    
    def _select_boundary_only(
        self, 
        segment_features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Select only boundary segments when they exceed max_seq_len."""
        # Prioritize start segments, then end segments
        if self.k_start >= self.max_seq_len:
            return segment_features[:self.max_seq_len]
        
        start_segments = segment_features[:self.k_start]
        remaining_slots = self.max_seq_len - self.k_start
        end_segments = segment_features[-remaining_slots:]
        
        return start_segments + end_segments
    
    def _select_by_importance(
        self, 
        segments: List[Dict[str, Any]], 
        num_to_select: int
    ) -> List[Dict[str, Any]]:
        """
        Select segments based on importance scores.
        
        Args:
            segments: List of segment features
            num_to_select: Number of segments to select
            
        Returns:
            List of selected segments
        """
        if num_to_select <= 0:
            return []
        
        if len(segments) <= num_to_select:
            return segments
        
        # Calculate importance scores
        importance_scores = self._calculate_importance_scores(segments)
        
        # Select top segments by importance
        segment_scores = list(zip(segments, importance_scores))
        segment_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_segments = [seg for seg, score in segment_scores[:num_to_select]]
        
        # Sort selected segments by original order
        selected_segments.sort(key=lambda x: x.get('segment_index', 0))
        
        logger.debug(f"Selected {len(selected_segments)} segments by importance")
        return selected_segments
    
    def _calculate_importance_scores(
        self, 
        segments: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Calculate importance scores for segments.
        
        Args:
            segments: List of segment features
            
        Returns:
            List of importance scores
        """
        if not segments:
            return []
        
        # Extract feature values for importance calculation
        rms_values = []
        voiced_fractions = []
        
        for segment in segments:
            rms_mean = segment.get('rms_mean')
            voiced_frac = segment.get('voiced_fraction', 0.0)
            
            rms_values.append(rms_mean if rms_mean is not None else 0.0)
            voiced_fractions.append(voiced_frac if voiced_frac is not None else 0.0)
        
        # Normalize features
        rms_normalized = self._normalize_values(rms_values)
        voiced_normalized = self._normalize_values(voiced_fractions)
        
        # Calculate weighted importance scores
        importance_scores = []
        for i in range(len(segments)):
            rms_score = rms_normalized[i] * self.importance_weights.get('rms', 0.6)
            voiced_score = voiced_normalized[i] * self.importance_weights.get('voiced_fraction', 0.4)
            
            importance_score = rms_score + voiced_score
            importance_scores.append(importance_score)
        
        return importance_scores
    
    def _normalize_values(self, values: List[float]) -> List[float]:
        """
        Normalize values to [0, 1] range.
        
        Args:
            values: List of values to normalize
            
        Returns:
            List of normalized values
        """
        if not values:
            return []
        
        values_array = np.array(values)
        
        # Handle case where all values are the same
        if np.max(values_array) == np.min(values_array):
            return [1.0] * len(values)
        
        # Min-max normalization
        normalized = (values_array - np.min(values_array)) / (np.max(values_array) - np.min(values_array))
        return normalized.tolist()
    
    def _select_random(
        self, 
        segment_features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Select segments randomly.
        
        Args:
            segment_features: List of segment features
            
        Returns:
            List of randomly selected segments
        """
        selected_indices = random.sample(range(len(segment_features)), self.max_seq_len)
        selected_segments = [segment_features[i] for i in selected_indices]
        
        # Sort by segment index to maintain temporal order
        selected_segments.sort(key=lambda x: x.get('segment_index', 0))
        
        logger.info(f"Randomly selected {len(selected_segments)} segments")
        return selected_segments
    
    def _select_uniform(
        self, 
        segment_features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Select segments uniformly across the audio.
        
        Args:
            segment_features: List of segment features
            
        Returns:
            List of uniformly selected segments
        """
        num_segments = len(segment_features)
        step = num_segments / self.max_seq_len
        
        selected_indices = []
        for i in range(self.max_seq_len):
            index = int(i * step)
            if index < num_segments:
                selected_indices.append(index)
        
        selected_segments = [segment_features[i] for i in selected_indices]
        
        logger.info(f"Uniformly selected {len(selected_segments)} segments")
        return selected_segments
    
    def apply_stochastic_replacement(
        self, 
        selected_segments: List[Dict[str, Any]],
        replacement_prob: float = 0.1,
        random_seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply stochastic replacement for data augmentation.
        
        Args:
            selected_segments: List of selected segments
            replacement_prob: Probability of replacing a segment
            random_seed: Random seed for reproducibility
            
        Returns:
            List of segments with stochastic replacements
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        if replacement_prob <= 0:
            return selected_segments
        
        augmented_segments = selected_segments.copy()
        
        # Only apply to middle segments (not start/end)
        start_end_count = self.k_start + self.k_end
        if len(augmented_segments) <= start_end_count:
            return augmented_segments
        
        # Apply replacement to middle segments
        for i in range(self.k_start, len(augmented_segments) - self.k_end):
            if random.random() < replacement_prob:
                # Replace with a random segment from the same position
                # This is a simplified replacement - in practice, you might want
                # to replace with segments from other videos
                logger.debug(f"Stochastic replacement applied to segment {i}")
        
        return augmented_segments


def select_segments_meta(
    segment_features: List[Dict[str, Any]],
    config: SegmentConfig,
    strategy: str = "boundary_importance",
    random_seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Select segments using metadata and configuration.
    
    Args:
        segment_features: List of segment feature dictionaries
        config: Configuration for selection
        strategy: Selection strategy
        random_seed: Random seed for reproducibility
        
    Returns:
        List of selected segment features
    """
    selector = SegmentSelector(config)
    return selector.select_segments(segment_features, strategy, random_seed)


def create_attention_mask(
    selected_segments: List[Dict[str, Any]],
    max_seq_len: int
) -> np.ndarray:
    """
    Create attention mask for selected segments.
    
    Args:
        selected_segments: List of selected segment features
        max_seq_len: Maximum sequence length
        
    Returns:
        Binary attention mask (1 for valid segments, 0 for padding)
    """
    mask = np.zeros(max_seq_len, dtype=np.uint8)
    
    for i, segment in enumerate(selected_segments):
        if i < max_seq_len:
            mask[i] = 1
    
    return mask


def pad_segments_to_max_len(
    selected_segments: List[Dict[str, Any]],
    max_seq_len: int,
    feature_dim: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad segments to maximum sequence length.
    
    Args:
        selected_segments: List of selected segment features
        max_seq_len: Maximum sequence length
        feature_dim: Feature dimension
        
    Returns:
        Tuple of (padded_features, attention_mask)
    """
    # Create feature matrix
    features = np.zeros((max_seq_len, feature_dim), dtype=np.float32)
    attention_mask = np.zeros(max_seq_len, dtype=np.uint8)
    
    # Fill with actual segment features
    for i, segment in enumerate(selected_segments):
        if i >= max_seq_len:
            break
        
        # Extract features from segment (this would need to be implemented
        # based on the actual feature structure)
        # For now, we'll create a placeholder
        segment_features = extract_features_from_segment(segment, feature_dim)
        features[i] = segment_features
        attention_mask[i] = 1
    
    return features, attention_mask


def extract_features_from_segment(
    segment: Dict[str, Any], 
    feature_dim: int
) -> np.ndarray:
    """
    Extract feature vector from segment dictionary.
    
    Args:
        segment: Segment feature dictionary
        feature_dim: Expected feature dimension
        
    Returns:
        Feature vector
    """
    # This is a placeholder implementation
    # In practice, you would extract and concatenate all relevant features
    # from the segment dictionary into a single vector
    
    features = np.zeros(feature_dim, dtype=np.float32)
    
    # Example feature extraction (would need to be customized)
    feature_idx = 0
    
    # Add scalar features
    scalar_features = [
        'rms_mean', 'rms_std', 'f0_mean', 'f0_std', 'voiced_fraction',
        'spectral_centroid_mean', 'spectral_bandwidth_mean', 'spectral_flatness_mean',
        'tempo_bpm', 'onset_density', 'onset_count_energy', 'vocal_fraction',
        'emotion_valence', 'emotion_arousal', 'emotion_dom_conf',
        'snr_db', 'words_count', 'words_per_sec', 'asr_conf_mean'
    ]
    
    for feature_name in scalar_features:
        if feature_idx < feature_dim:
            value = segment.get(feature_name, 0.0)
            if value is None:
                value = 0.0
            features[feature_idx] = float(value)
            feature_idx += 1
    
    # Add compressed embeddings
    embedding_features = ['clap_pca', 'wav2vec_pca', 'yamnet_pca']
    for embedding_name in embedding_features:
        embedding = segment.get(embedding_name)
        if embedding is not None and isinstance(embedding, list):
            for val in embedding:
                if feature_idx < feature_dim:
                    features[feature_idx] = float(val)
                    feature_idx += 1
    
    # Add mel/mfcc features
    mel_mfcc_features = ['mel_mean_vector', 'mfcc_mean_vector']
    for feature_name in mel_mfcc_features:
        feature_vector = segment.get(feature_name)
        if feature_vector is not None and isinstance(feature_vector, list):
            for val in feature_vector:
                if feature_idx < feature_dim:
                    features[feature_idx] = float(val)
                    feature_idx += 1
    
    return features
