"""
Segment aggregator for converting AudioProcessor outputs to per-segment features.

This module contains functions for:
- Aggregating extractor outputs into per-segment features
- Handling different types of features (scalars, arrays, embeddings)
- Time-based slicing and aggregation
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from .segment_utils import (
    get_time_sliced_array, 
    get_time_sliced_embeddings, 
    aggregate_array_features,
    create_time_array
)
from .segment_config import SegmentConfig

logger = logging.getLogger(__name__)


class SegmentAggregator:
    """Aggregator for converting AudioProcessor outputs to per-segment features."""
    
    def __init__(self, config: SegmentConfig):
        """
        Initialize segment aggregator.
        
        Args:
            config: Configuration for segment processing
        """
        self.config = config
        self.feature_mapping = config.get_feature_mapping()
        self.array_fields = config.get_array_fields()
        self.scalar_fields = config.get_scalar_fields()
        
        logger.info(f"Initialized SegmentAggregator with {len(self.feature_mapping)} feature mappings")
    
    def aggregate_segment(
        self, 
        extractor_outputs: Dict[str, Any], 
        seg_start: float, 
        seg_end: float
    ) -> Dict[str, Any]:
        """
        Aggregate features for a single segment from extractor outputs.
        
        Args:
            extractor_outputs: Dictionary with extractor results
            seg_start: Segment start time in seconds
            seg_end: Segment end time in seconds
            
        Returns:
            Dictionary with aggregated features for the segment
        """
        features = {}
        
        # Add segment metadata
        features['segment_start'] = seg_start
        features['segment_end'] = seg_end
        features['segment_duration'] = seg_end - seg_start
        
        # Process CLAP embeddings
        features.update(self._process_clap_features(extractor_outputs, seg_start, seg_end))
        
        # Process advanced embeddings (wav2vec, yamnet)
        features.update(self._process_advanced_embeddings(extractor_outputs, seg_start, seg_end))
        
        # Process RMS/loudness features
        features.update(self._process_loudness_features(extractor_outputs, seg_start, seg_end))
        
        # Process VAD/pitch features
        features.update(self._process_vad_features(extractor_outputs, seg_start, seg_end))
        
        # Process spectral features
        features.update(self._process_spectral_features(extractor_outputs, seg_start, seg_end))
        
        # Process tempo/onset features
        features.update(self._process_tempo_features(extractor_outputs, seg_start, seg_end))
        
        # Process source separation features
        features.update(self._process_source_separation_features(extractor_outputs, seg_start, seg_end))
        
        # Process emotion features
        features.update(self._process_emotion_features(extractor_outputs, seg_start, seg_end))
        
        # Process quality features
        features.update(self._process_quality_features(extractor_outputs, seg_start, seg_end))
        
        # Process ASR features
        features.update(self._process_asr_features(extractor_outputs, seg_start, seg_end))
        
        # Process mel/mfcc features
        features.update(self._process_mel_mfcc_features(extractor_outputs, seg_start, seg_end))
        
        return features
    
    def _process_clap_features(
        self, 
        extractor_outputs: Dict[str, Any], 
        seg_start: float, 
        seg_end: float
    ) -> Dict[str, Any]:
        """Process CLAP embedding features."""
        features = {}
        
        # Get CLAP extractor result
        clap_result = extractor_outputs.get('clap_extractor')
        if not clap_result or not clap_result.get('success'):
            features['clap_mean'] = None
            features['clap_std'] = None
            return features
        
        payload = clap_result.get('payload', {})
        
        # For now, use global CLAP features (since CLAP typically processes entire audio)
        # In future, we could implement windowed CLAP processing
        clap_embedding = payload.get('clap_embedding')
        if clap_embedding is not None:
            clap_array = np.array(clap_embedding)
            features['clap_mean'] = float(np.mean(clap_array))
            features['clap_std'] = float(np.std(clap_array))
        else:
            features['clap_mean'] = None
            features['clap_std'] = None
        
        return features
    
    def _process_advanced_embeddings(
        self, 
        extractor_outputs: Dict[str, Any], 
        seg_start: float, 
        seg_end: float
    ) -> Dict[str, Any]:
        """Process advanced embedding features (wav2vec, yamnet)."""
        features = {}
        
        # Get advanced embeddings result
        adv_result = extractor_outputs.get('advanced_embeddings')
        if not adv_result or not adv_result.get('success'):
            features['wav2vec_mean'] = None
            features['wav2vec_std'] = None
            features['yamnet_mean'] = None
            features['yamnet_std'] = None
            return features
        
        payload = adv_result.get('payload', {})
        
        # Process wav2vec embeddings
        wav2vec_embeddings = payload.get('wav2vec_embeddings')
        if wav2vec_embeddings is not None:
            wav2vec_array = np.array(wav2vec_embeddings)
            if wav2vec_array.ndim == 2:
                features['wav2vec_mean'] = float(np.mean(wav2vec_array))
                features['wav2vec_std'] = float(np.std(wav2vec_array))
            else:
                features['wav2vec_mean'] = float(np.mean(wav2vec_array))
                features['wav2vec_std'] = float(np.std(wav2vec_array))
        else:
            features['wav2vec_mean'] = None
            features['wav2vec_std'] = None
        
        # Process yamnet embeddings
        yamnet_embeddings = payload.get('yamnet_embeddings')
        if yamnet_embeddings is not None:
            yamnet_array = np.array(yamnet_embeddings)
            if yamnet_array.ndim == 2:
                features['yamnet_mean'] = float(np.mean(yamnet_array))
                features['yamnet_std'] = float(np.std(yamnet_array))
            else:
                features['yamnet_mean'] = float(np.mean(yamnet_array))
                features['yamnet_std'] = float(np.std(yamnet_array))
        else:
            features['yamnet_mean'] = None
            features['yamnet_std'] = None
        
        return features
    
    def _process_loudness_features(
        self, 
        extractor_outputs: Dict[str, Any], 
        seg_start: float, 
        seg_end: float
    ) -> Dict[str, Any]:
        """Process RMS/loudness features."""
        features = {}
        
        # Get loudness extractor result
        loudness_result = extractor_outputs.get('loudness_extractor')
        if not loudness_result or not loudness_result.get('success'):
            features['rms_mean'] = None
            features['rms_std'] = None
            return features
        
        payload = loudness_result.get('payload', {})
        
        # Get RMS array and times
        rms_array = payload.get('rms_array')
        if rms_array is not None:
            rms_array = np.array(rms_array)
            
            # Create time array for RMS frames
            duration = seg_end - seg_start
            hop_length = 512  # Default hop length from loudness extractor
            sample_rate = 22050  # Default sample rate
            rms_times = create_time_array(duration, hop_length, sample_rate)
            
            # Slice RMS array for segment
            rms_slice = get_time_sliced_array(rms_times, rms_array, seg_start, seg_end)
            
            if rms_slice is not None and len(rms_slice) > 0:
                features['rms_mean'] = float(np.mean(rms_slice))
                features['rms_std'] = float(np.std(rms_slice))
            else:
                features['rms_mean'] = None
                features['rms_std'] = None
        else:
            # Fallback to global RMS features
            features['rms_mean'] = payload.get('rms_mean')
            features['rms_std'] = payload.get('rms_std')
        
        return features
    
    def _process_vad_features(
        self, 
        extractor_outputs: Dict[str, Any], 
        seg_start: float, 
        seg_end: float
    ) -> Dict[str, Any]:
        """Process VAD/pitch features."""
        features = {}
        
        # Get VAD extractor result
        vad_result = extractor_outputs.get('vad_extractor')
        if not vad_result or not vad_result.get('success'):
            features['f0_mean'] = None
            features['f0_std'] = None
            features['voiced_fraction'] = 0.0
            return features
        
        payload = vad_result.get('payload', {})
        
        # Process F0 features
        f0_array = payload.get('f0_array')
        if f0_array is not None:
            f0_array = np.array(f0_array)
            
            # Create time array for F0 frames
            duration = seg_end - seg_start
            hop_length = 512  # Default hop length from VAD extractor
            sample_rate = 22050  # Default sample rate
            f0_times = create_time_array(duration, hop_length, sample_rate)
            
            # Slice F0 array for segment
            f0_slice = get_time_sliced_array(f0_times, f0_array, seg_start, seg_end)
            
            if f0_slice is not None and len(f0_slice) > 0:
                # Remove NaN values
                valid_f0 = f0_slice[~np.isnan(f0_slice)]
                if len(valid_f0) > 0:
                    features['f0_mean'] = float(np.mean(valid_f0))
                    features['f0_std'] = float(np.std(valid_f0))
                else:
                    features['f0_mean'] = None
                    features['f0_std'] = None
            else:
                features['f0_mean'] = None
                features['f0_std'] = None
        else:
            # Fallback to global F0 features
            features['f0_mean'] = payload.get('f0_mean')
            features['f0_std'] = payload.get('f0_std')
        
        # Process voiced fraction
        voiced_flag_array = payload.get('voiced_flag_array')
        if voiced_flag_array is not None:
            voiced_flag_array = np.array(voiced_flag_array)
            
            # Create time array for voiced flag frames
            duration = seg_end - seg_start
            hop_length = 512
            sample_rate = 22050
            vad_times = create_time_array(duration, hop_length, sample_rate)
            
            # Slice voiced flag array for segment
            voiced_slice = get_time_sliced_array(vad_times, voiced_flag_array, seg_start, seg_end)
            
            if voiced_slice is not None and len(voiced_slice) > 0:
                features['voiced_fraction'] = float(np.mean(voiced_slice))
            else:
                features['voiced_fraction'] = 0.0
        else:
            # Fallback to global voiced fraction
            features['voiced_fraction'] = payload.get('voiced_fraction', 0.0)
        
        return features
    
    def _process_spectral_features(
        self, 
        extractor_outputs: Dict[str, Any], 
        seg_start: float, 
        seg_end: float
    ) -> Dict[str, Any]:
        """Process spectral features."""
        features = {}
        
        # Get spectral extractor result
        spectral_result = extractor_outputs.get('spectral_extractor')
        if not spectral_result or not spectral_result.get('success'):
            features['spectral_centroid_mean'] = None
            features['spectral_bandwidth_mean'] = None
            features['spectral_flatness_mean'] = None
            return features
        
        payload = spectral_result.get('payload', {})
        
        # Use global spectral features for now
        features['spectral_centroid_mean'] = payload.get('spectral_centroid_mean')
        features['spectral_bandwidth_mean'] = payload.get('spectral_bandwidth_mean')
        features['spectral_flatness_mean'] = payload.get('spectral_flatness_mean')
        
        return features
    
    def _process_tempo_features(
        self, 
        extractor_outputs: Dict[str, Any], 
        seg_start: float, 
        seg_end: float
    ) -> Dict[str, Any]:
        """Process tempo/onset features."""
        features = {}
        
        # Get tempo extractor result
        tempo_result = extractor_outputs.get('tempo_extractor')
        if tempo_result and tempo_result.get('success'):
            payload = tempo_result.get('payload', {})
            features['tempo_bpm'] = payload.get('tempo_bpm')
            features['onset_density'] = payload.get('onset_density')
        else:
            features['tempo_bpm'] = None
            features['onset_density'] = None
        
        # Get onset extractor result
        onset_result = extractor_outputs.get('onset_extractor')
        if onset_result and onset_result.get('success'):
            payload = onset_result.get('payload', {})
            features['onset_count_energy'] = payload.get('onset_count_energy')
        else:
            features['onset_count_energy'] = None
        
        return features
    
    def _process_source_separation_features(
        self, 
        extractor_outputs: Dict[str, Any], 
        seg_start: float, 
        seg_end: float
    ) -> Dict[str, Any]:
        """Process source separation features."""
        features = {}
        
        # Get source separation extractor result
        sep_result = extractor_outputs.get('source_separation_extractor')
        if sep_result and sep_result.get('success'):
            payload = sep_result.get('payload', {})
            features['vocal_fraction'] = payload.get('vocal_fraction')
        else:
            features['vocal_fraction'] = None
        
        return features
    
    def _process_emotion_features(
        self, 
        extractor_outputs: Dict[str, Any], 
        seg_start: float, 
        seg_end: float
    ) -> Dict[str, Any]:
        """Process emotion recognition features."""
        features = {}
        
        # Get emotion extractor result
        emotion_result = extractor_outputs.get('emotion_recognition_extractor')
        if emotion_result and emotion_result.get('success'):
            payload = emotion_result.get('payload', {})
            features['emotion_valence'] = payload.get('emotion_valence')
            features['emotion_arousal'] = payload.get('emotion_arousal')
            features['emotion_dom_conf'] = payload.get('dominant_emotion_confidence')
        else:
            features['emotion_valence'] = None
            features['emotion_arousal'] = None
            features['emotion_dom_conf'] = None
        
        return features
    
    def _process_quality_features(
        self, 
        extractor_outputs: Dict[str, Any], 
        seg_start: float, 
        seg_end: float
    ) -> Dict[str, Any]:
        """Process quality features."""
        features = {}
        
        # Get quality extractor result
        quality_result = extractor_outputs.get('quality_extractor')
        if quality_result and quality_result.get('success'):
            payload = quality_result.get('payload', {})
            features['snr_db'] = payload.get('snr_estimate_db')
            features['hum_detected'] = payload.get('hum_detected', False)
        else:
            features['snr_db'] = None
            features['hum_detected'] = False
        
        return features
    
    def _process_asr_features(
        self, 
        extractor_outputs: Dict[str, Any], 
        seg_start: float, 
        seg_end: float
    ) -> Dict[str, Any]:
        """Process ASR features."""
        features = {}
        
        # Get ASR extractor result
        asr_result = extractor_outputs.get('asr_extractor')
        if not asr_result or not asr_result.get('success'):
            features['words_count'] = 0
            features['words_per_sec'] = 0.0
            features['asr_conf_mean'] = None
            return features
        
        payload = asr_result.get('payload', {})
        
        # Process word timestamps
        word_timestamps = payload.get('word_timestamps', [])
        words_in_segment = []
        
        for word_info in word_timestamps:
            if isinstance(word_info, dict):
                start_time = word_info.get('start', 0)
                end_time = word_info.get('end', 0)
                word = word_info.get('word', '')
            else:
                # Handle different timestamp formats
                start_time, end_time, word = word_info[:3]
            
            # Check if word falls within segment
            if start_time >= seg_start and start_time < seg_end:
                words_in_segment.append(word)
        
        features['words_count'] = len(words_in_segment)
        segment_duration = seg_end - seg_start
        features['words_per_sec'] = len(words_in_segment) / segment_duration if segment_duration > 0 else 0.0
        
        # Process confidence
        transcript_confidence = payload.get('transcript_confidence')
        if transcript_confidence is not None and transcript_confidence >= self.config.asr_confidence_threshold:
            features['asr_conf_mean'] = transcript_confidence
        else:
            features['asr_conf_mean'] = None
        
        return features
    
    def _process_mel_mfcc_features(
        self, 
        extractor_outputs: Dict[str, Any], 
        seg_start: float, 
        seg_end: float
    ) -> Dict[str, Any]:
        """Process mel/mfcc features."""
        features = {}
        
        # Get mel extractor result
        mel_result = extractor_outputs.get('mel_extractor')
        if mel_result and mel_result.get('success'):
            payload = mel_result.get('payload', {})
            mel_mean = payload.get('mel64_mean')
            if mel_mean is not None:
                features['mel_mean_vector'] = np.array(mel_mean).tolist()
            else:
                features['mel_mean_vector'] = None
        else:
            features['mel_mean_vector'] = None
        
        # Get MFCC extractor result
        mfcc_result = extractor_outputs.get('mfcc_extractor')
        if mfcc_result and mfcc_result.get('success'):
            payload = mfcc_result.get('payload', {})
            mfcc_mean = payload.get('mfcc_mean')
            if mfcc_mean is not None:
                features['mfcc_mean_vector'] = np.array(mfcc_mean).tolist()
            else:
                features['mfcc_mean_vector'] = None
        else:
            features['mfcc_mean_vector'] = None
        
        return features


def aggregate_all_segments(
    extractor_outputs: Dict[str, Any],
    segments: List[Tuple[float, float]],
    config: SegmentConfig
) -> List[Dict[str, Any]]:
    """
    Aggregate features for all segments.
    
    Args:
        extractor_outputs: Dictionary with extractor results
        segments: List of (start, end) time tuples
        config: Configuration for processing
        
    Returns:
        List of feature dictionaries, one per segment
    """
    aggregator = SegmentAggregator(config)
    
    segment_features = []
    for i, (start, end) in enumerate(segments):
        features = aggregator.aggregate_segment(extractor_outputs, start, end)
        features['segment_index'] = i
        segment_features.append(features)
    
    logger.info(f"Aggregated features for {len(segments)} segments")
    return segment_features
