"""
Onset Extractor for onset detection and analysis
Extracts onset density, onset strength, and onset-related features
"""

import numpy as np
import librosa
from typing import Dict, Any, List
from src.core.base_extractor import BaseExtractor, ExtractorResult


class OnsetExtractor(BaseExtractor):
    """
    Onset Extractor for onset detection and analysis
    Extracts onset density, onset strength, and onset-related features
    """
    
    name = "onset"
    version = "1.0.0"
    description = "Onset detection and analysis: density, strength, patterns"
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 22050
        self.hop_length = 512
        self.frame_length = 2048
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Extract onset features from audio file
        
        Args:
            input_uri: Path to audio file
            tmp_path: Temporary directory for processing
            
        Returns:
            ExtractorResult with onset features
        """
        try:
            self.logger.info(f"Starting onset extraction for {input_uri}")
            
            # Load audio
            audio, sr = librosa.load(input_uri, sr=self.sample_rate)
            
            # Extract onset features with timing
            features, processing_time = self._time_execution(self._extract_onset_features, audio, sr)
            
            self.logger.info(f"Onset extraction completed successfully in {processing_time:.3f}s")
            
            return self._create_result(
                success=True,
                payload=features,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Onset extraction failed: {e}")
            return self._create_result(
                success=False,
                error=str(e)
            )
    
    def _extract_onset_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract onset features from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of onset features
        """
        features = {}
        
        # Onset strength
        onset_strength = librosa.onset.onset_strength(
            y=audio,
            sr=sr,
            hop_length=self.hop_length
        )
        
        features["onset_strength_mean"] = float(np.mean(onset_strength))
        features["onset_strength_std"] = float(np.std(onset_strength))
        features["onset_strength_max"] = float(np.max(onset_strength))
        features["onset_strength_min"] = float(np.min(onset_strength))
        features["onset_strength_median"] = float(np.median(onset_strength))
        
        # Onset detection with different methods
        onset_methods = ['energy', 'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth']
        
        for method in onset_methods:
            try:
                # Use different onset strength functions for different methods
                if method == 'energy':
                    onset_strength = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=self.hop_length)
                elif method == 'spectral_centroid':
                    onset_strength = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=self.hop_length, feature=librosa.feature.spectral_centroid)
                elif method == 'spectral_rolloff':
                    onset_strength = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=self.hop_length, feature=librosa.feature.spectral_rolloff)
                elif method == 'spectral_bandwidth':
                    onset_strength = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=self.hop_length, feature=librosa.feature.spectral_bandwidth)
                
                onset_frames = librosa.onset.onset_detect(
                    onset_envelope=onset_strength,
                    sr=sr,
                    hop_length=self.hop_length,
                    units='frames',
                    pre_max=3,
                    post_max=3,
                    pre_avg=3,
                    post_avg=5,
                    delta=0.2,
                    wait=10
                )
                
                onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
                
                features[f"onset_count_{method}"] = len(onset_frames)
                features[f"onset_times_{method}"] = onset_times.tolist()
                
                # Onset density
                audio_duration = len(audio) / sr
                features[f"onset_density_{method}"] = len(onset_frames) / audio_duration if audio_duration > 0 else 0.0
                
            except Exception as e:
                self.logger.warning(f"Onset detection with {method} failed: {e}")
                features[f"onset_count_{method}"] = 0
                features[f"onset_times_{method}"] = []
                features[f"onset_density_{method}"] = 0.0
        
        # Use energy-based onset detection as primary
        primary_onset_times = features.get("onset_times_energy", [])
        primary_onset_count = features.get("onset_count_energy", 0)
        
        # Onset density (primary metric)
        audio_duration = len(audio) / sr
        features["onset_density"] = primary_onset_count / audio_duration if audio_duration > 0 else 0.0
        
        # Onset intervals analysis
        if len(primary_onset_times) > 1:
            onset_intervals = np.diff(primary_onset_times)
            
            features["onset_interval_mean"] = float(np.mean(onset_intervals))
            features["onset_interval_std"] = float(np.std(onset_intervals))
            features["onset_interval_min"] = float(np.min(onset_intervals))
            features["onset_interval_max"] = float(np.max(onset_intervals))
            features["onset_interval_median"] = float(np.median(onset_intervals))
            
            # Onset regularity
            if features["onset_interval_mean"] > 0:
                features["onset_regularity"] = 1.0 / (1.0 + features["onset_interval_std"] / features["onset_interval_mean"])
            else:
                features["onset_regularity"] = 0.0
        else:
            features["onset_interval_mean"] = 0.0
            features["onset_interval_std"] = 0.0
            features["onset_interval_min"] = 0.0
            features["onset_interval_max"] = 0.0
            features["onset_interval_median"] = 0.0
            features["onset_regularity"] = 0.0
        
        # Onset clustering analysis
        clustering_features = self._analyze_onset_clustering(primary_onset_times)
        features.update(clustering_features)
        
        # Onset strength distribution
        strength_features = self._analyze_onset_strength_distribution(onset_strength)
        features.update(strength_features)
        
        # Temporal onset patterns
        temporal_features = self._analyze_temporal_onset_patterns(primary_onset_times, audio_duration)
        features.update(temporal_features)
        
        return features
    
    def _analyze_onset_clustering(self, onset_times: List[float]) -> Dict[str, Any]:
        """
        Analyze onset clustering patterns
        
        Args:
            onset_times: List of onset times
            
        Returns:
            Dictionary with clustering features
        """
        if len(onset_times) < 3:
            return {
                "onset_clusters_count": 0,
                "onset_cluster_size_mean": 0.0,
                "onset_cluster_density": 0.0,
                "onset_clustering_score": 0.0
            }
        
        # Simple clustering based on time gaps
        cluster_threshold = 0.5  # 500ms threshold for clustering
        clusters = []
        current_cluster = [onset_times[0]]
        
        for i in range(1, len(onset_times)):
            if onset_times[i] - onset_times[i-1] <= cluster_threshold:
                current_cluster.append(onset_times[i])
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [onset_times[i]]
        
        # Add last cluster
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
        
        # Calculate clustering features
        cluster_count = len(clusters)
        cluster_sizes = [len(cluster) for cluster in clusters]
        
        return {
            "onset_clusters_count": cluster_count,
            "onset_cluster_size_mean": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
            "onset_cluster_density": cluster_count / len(onset_times) if onset_times else 0.0,
            "onset_clustering_score": float(len([c for c in cluster_sizes if c > 2]) / len(onset_times)) if onset_times else 0.0
        }
    
    def _analyze_onset_strength_distribution(self, onset_strength: np.ndarray) -> Dict[str, Any]:
        """
        Analyze onset strength distribution
        
        Args:
            onset_strength: Onset strength array
            
        Returns:
            Dictionary with strength distribution features
        """
        if len(onset_strength) == 0:
            return {
                "onset_strength_skewness": 0.0,
                "onset_strength_kurtosis": 0.0,
                "onset_strength_p25": 0.0,
                "onset_strength_p75": 0.0,
                "onset_strength_p90": 0.0,
                "strong_onsets_ratio": 0.0
            }
        
        # Statistical measures
        skewness = self._calculate_skewness(onset_strength)
        kurtosis = self._calculate_kurtosis(onset_strength)
        
        # Percentiles
        p25 = np.percentile(onset_strength, 25)
        p75 = np.percentile(onset_strength, 75)
        p90 = np.percentile(onset_strength, 90)
        
        # Strong onsets ratio (above 75th percentile)
        strong_threshold = p75
        strong_onsets_ratio = np.sum(onset_strength > strong_threshold) / len(onset_strength)
        
        return {
            "onset_strength_skewness": float(skewness),
            "onset_strength_kurtosis": float(kurtosis),
            "onset_strength_p25": float(p25),
            "onset_strength_p75": float(p75),
            "onset_strength_p90": float(p90),
            "strong_onsets_ratio": float(strong_onsets_ratio)
        }
    
    def _analyze_temporal_onset_patterns(self, onset_times: List[float], duration: float) -> Dict[str, Any]:
        """
        Analyze temporal patterns of onsets
        
        Args:
            onset_times: List of onset times
            duration: Audio duration in seconds
            
        Returns:
            Dictionary with temporal pattern features
        """
        if len(onset_times) == 0 or duration == 0:
            return {
                "onset_temporal_distribution": "uniform",
                "onset_beginning_density": 0.0,
                "onset_middle_density": 0.0,
                "onset_end_density": 0.0,
                "onset_temporal_variation": 0.0
            }
        
        # Divide audio into thirds
        third = duration / 3
        beginning_onsets = [t for t in onset_times if t < third]
        middle_onsets = [t for t in onset_times if third <= t < 2 * third]
        end_onsets = [t for t in onset_times if t >= 2 * third]
        
        # Calculate densities
        beginning_density = len(beginning_onsets) / third
        middle_density = len(middle_onsets) / third
        end_density = len(end_onsets) / third
        
        # Determine temporal distribution pattern
        densities = [beginning_density, middle_density, end_density]
        max_density_idx = np.argmax(densities)
        
        if max_density_idx == 0:
            distribution = "beginning_heavy"
        elif max_density_idx == 1:
            distribution = "middle_heavy"
        elif max_density_idx == 2:
            distribution = "end_heavy"
        else:
            distribution = "uniform"
        
        # Calculate temporal variation
        temporal_variation = np.std(densities) / (np.mean(densities) + 1e-10)
        
        return {
            "onset_temporal_distribution": distribution,
            "onset_beginning_density": float(beginning_density),
            "onset_middle_density": float(middle_density),
            "onset_end_density": float(end_density),
            "onset_temporal_variation": float(temporal_variation)
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return float(skewness)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3
        return float(kurtosis)


# For running as standalone module
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python onset_extractor.py <audio_file>")
        sys.exit(1)
    
    extractor = OnsetExtractor()
    result = extractor.run(sys.argv[1], "/tmp")
    print(json.dumps(result.dict(), indent=2))
