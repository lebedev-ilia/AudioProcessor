"""
Main pipeline for converting AudioProcessor outputs to per-segment features.

This module provides the main pipeline that orchestrates the entire process:
1. Load AudioProcessor results
2. Create time segments
3. Aggregate features per segment
4. Compress embeddings with PCA
5. Select important segments
6. Save per-video data
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

from .segment_config import SegmentConfig, get_default_config
from .segment_utils import make_segments, validate_segment_bounds, get_segment_metadata
from .segment_aggregator import aggregate_all_segments
from .embedding_compressor import EmbeddingCompressor, collect_embeddings_from_extractors, compress_segment_embeddings
from .segment_selector import select_segments_meta, create_attention_mask, pad_segments_to_max_len, extract_features_from_segment
from .segment_storage import SegmentStorage, save_artifacts

logger = logging.getLogger(__name__)


class SegmentPipeline:
    """Main pipeline for per-segment feature extraction."""
    
    def __init__(self, config: Optional[SegmentConfig] = None):
        """
        Initialize segment pipeline.
        
        Args:
            config: Configuration for processing (uses default if None)
        """
        self.config = config or get_default_config()
        
        # Initialize components
        self.compressor = EmbeddingCompressor(self.config)
        self.storage = SegmentStorage(self.config)
        
        logger.info(f"Initialized SegmentPipeline with config: {self.config.storage_format}")
    
    def process_single_video(
        self,
        video_id: str,
        extractor_outputs: Dict[str, Any],
        duration: float,
        save_features: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single video from AudioProcessor outputs.
        
        Args:
            video_id: Video identifier
            extractor_outputs: Dictionary with extractor results
            duration: Audio duration in seconds
            save_features: Whether to save features to disk
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing video {video_id} (duration: {duration:.2f}s)")
        
        try:
            # Step 1: Create time segments
            segments = make_segments(
                duration=duration,
                segment_len=self.config.segment_len,
                hop=self.config.hop
            )
            
            # Validate segments
            if not validate_segment_bounds(segments, duration):
                raise ValueError(f"Invalid segment bounds for video {video_id}")
            
            logger.info(f"Created {len(segments)} segments for video {video_id}")
            
            # Step 2: Aggregate features per segment
            segment_features = aggregate_all_segments(
                extractor_outputs=extractor_outputs,
                segments=segments,
                config=self.config
            )
            
            logger.info(f"Aggregated features for {len(segment_features)} segments")
            
            # Step 3: Compress embeddings (if PCA models are available)
            try:
                segment_features = compress_segment_embeddings(segment_features, self.compressor)
                logger.info("Applied embedding compression")
            except Exception as e:
                logger.warning(f"Embedding compression failed: {e}")
            
            # Step 4: Select important segments
            selected_segments = select_segments_meta(
                segment_features=segment_features,
                config=self.config,
                strategy="boundary_importance"
            )
            
            logger.info(f"Selected {len(selected_segments)} segments from {len(segment_features)}")
            
            # Step 5: Create feature vectors and attention mask
            feature_vectors, attention_mask = self._create_feature_vectors(
                selected_segments, self.config.max_seq_len
            )
            
            # Step 6: Save features if requested
            file_paths = {}
            if save_features:
                metadata = {
                    "duration": duration,
                    "num_original_segments": len(segments),
                    "num_selected_segments": len(selected_segments),
                    "selection_strategy": "boundary_importance",
                    "segment_len": self.config.segment_len,
                    "hop": self.config.hop
                }
                
                file_paths = self.storage.save_per_video(
                    video_id=video_id,
                    selected_segments=selected_segments,
                    feature_vectors=feature_vectors,
                    attention_mask=attention_mask,
                    metadata=metadata
                )
            
            # Create result
            result = {
                "video_id": video_id,
                "success": True,
                "duration": duration,
                "num_segments": len(segments),
                "num_selected_segments": len(selected_segments),
                "feature_shape": feature_vectors.shape,
                "file_paths": file_paths,
                "processing_time": None  # Could add timing
            }
            
            logger.info(f"Successfully processed video {video_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process video {video_id}: {e}")
            return {
                "video_id": video_id,
                "success": False,
                "error": str(e),
                "duration": duration
            }
    
    def process_batch(
        self,
        video_data: List[Dict[str, Any]],
        fit_pca: bool = True,
        save_features: bool = True
    ) -> Dict[str, Any]:
        """
        Process a batch of videos.
        
        Args:
            video_data: List of video data dictionaries with keys:
                       - video_id: Video identifier
                       - extractor_outputs: AudioProcessor results
                       - duration: Audio duration
            fit_pca: Whether to fit PCA models on this batch
            save_features: Whether to save features to disk
            
        Returns:
            Dictionary with batch processing results
        """
        logger.info(f"Processing batch of {len(video_data)} videos")
        
        batch_results = {
            "total_videos": len(video_data),
            "successful": 0,
            "failed": 0,
            "results": [],
            "pca_fitted": False,
            "artifacts_saved": False
        }
        
        try:
            # Step 1: Fit PCA models if requested
            if fit_pca:
                logger.info("Fitting PCA models on batch data")
                self._fit_pca_models(video_data)
                batch_results["pca_fitted"] = True
            
            # Step 2: Process each video
            for video_info in video_data:
                video_id = video_info["video_id"]
                extractor_outputs = video_info["extractor_outputs"]
                duration = video_info["duration"]
                
                result = self.process_single_video(
                    video_id=video_id,
                    extractor_outputs=extractor_outputs,
                    duration=duration,
                    save_features=save_features
                )
                
                batch_results["results"].append(result)
                
                if result["success"]:
                    batch_results["successful"] += 1
                else:
                    batch_results["failed"] += 1
            
            # Step 3: Save artifacts if PCA was fitted
            if fit_pca and batch_results["pca_fitted"]:
                try:
                    artifact_paths = save_artifacts(
                        config=self.config,
                        pca_models=self.compressor.pca_models,
                        scalers=self.compressor.scalers
                    )
                    batch_results["artifact_paths"] = artifact_paths
                    batch_results["artifacts_saved"] = True
                    logger.info("Saved processing artifacts")
                except Exception as e:
                    logger.error(f"Failed to save artifacts: {e}")
            
            logger.info(f"Batch processing completed: {batch_results['successful']} successful, "
                       f"{batch_results['failed']} failed")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            batch_results["error"] = str(e)
        
        return batch_results
    
    def _fit_pca_models(self, video_data: List[Dict[str, Any]]):
        """Fit PCA models on batch data."""
        try:
            # Collect embeddings from all videos
            all_extractor_outputs = []
            for video_info in video_data:
                all_extractor_outputs.append(video_info["extractor_outputs"])
            
            # Collect embeddings
            embedding_data = collect_embeddings_from_extractors(
                all_extractor_outputs,
                embedding_types=list(self.config.pca_dims.keys())
            )
            
            # Fit PCA models
            fitted_models = self.compressor.fit_pca_models(
                embedding_data=embedding_data,
                save_models=True
            )
            
            logger.info(f"Fitted {len(fitted_models)} PCA models")
            
        except Exception as e:
            logger.error(f"Failed to fit PCA models: {e}")
            raise
    
    def _create_feature_vectors(
        self,
        selected_segments: List[Dict[str, Any]],
        max_seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create feature vectors and attention mask from selected segments.
        
        Args:
            selected_segments: List of selected segment features
            max_seq_len: Maximum sequence length
            
        Returns:
            Tuple of (feature_vectors, attention_mask)
        """
        # Calculate feature dimension
        feature_dim = self._calculate_feature_dim(selected_segments[0] if selected_segments else {})
        
        # Create feature matrix and attention mask
        feature_vectors, attention_mask = pad_segments_to_max_len(
            selected_segments=selected_segments,
            max_seq_len=max_seq_len,
            feature_dim=feature_dim
        )
        
        return feature_vectors, attention_mask
    
    def _calculate_feature_dim(self, sample_segment: Dict[str, Any]) -> int:
        """
        Calculate feature dimension from a sample segment.
        
        Args:
            sample_segment: Sample segment feature dictionary
            
        Returns:
            Feature dimension
        """
        # This is a simplified calculation - in practice, you'd want to
        # systematically count all features
        
        feature_dim = 0
        
        # Scalar features
        scalar_features = [
            'rms_mean', 'rms_std', 'f0_mean', 'f0_std', 'voiced_fraction',
            'spectral_centroid_mean', 'spectral_bandwidth_mean', 'spectral_flatness_mean',
            'tempo_bpm', 'onset_density', 'onset_count_energy', 'vocal_fraction',
            'emotion_valence', 'emotion_arousal', 'emotion_dom_conf',
            'snr_db', 'words_count', 'words_per_sec', 'asr_conf_mean'
        ]
        feature_dim += len(scalar_features)
        
        # Compressed embeddings
        for embedding_name in ['clap_pca', 'wav2vec_pca', 'yamnet_pca']:
            embedding = sample_segment.get(embedding_name)
            if embedding is not None and isinstance(embedding, list):
                feature_dim += len(embedding)
            else:
                # Use default dimensions if not available
                if embedding_name == 'clap_pca':
                    feature_dim += self.config.pca_dims.get('clap', 128)
                elif embedding_name == 'wav2vec_pca':
                    feature_dim += self.config.pca_dims.get('wav2vec', 64)
                elif embedding_name == 'yamnet_pca':
                    feature_dim += self.config.pca_dims.get('yamnet', 128)
        
        # Mel/MFCC features (simplified)
        feature_dim += 64 + 13  # mel64 + mfcc13
        
        return feature_dim
    
    def load_video_features(self, video_id: str) -> Dict[str, Any]:
        """
        Load features for a specific video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Dictionary with loaded features
        """
        return self.storage.load_per_video(video_id)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        storage_stats = self.storage.get_storage_stats()
        
        return {
            "config": self.config.to_dict(),
            "storage_stats": storage_stats,
            "pca_models_loaded": len(self.compressor.pca_models),
            "compressor_info": self.compressor.get_model_info()
        }


def process_manifest_file(
    manifest_path: str,
    config: Optional[SegmentConfig] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a manifest file from AudioProcessor.
    
    Args:
        manifest_path: Path to manifest JSON file
        config: Configuration for processing
        output_dir: Output directory (overrides config)
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Processing manifest file: {manifest_path}")
    
    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Update config if output directory specified
    if config is None:
        config = get_default_config()
    
    if output_dir is not None:
        config.output_dir = output_dir
    
    # Initialize pipeline
    pipeline = SegmentPipeline(config)
    
    # Extract video data from manifest
    video_data = []
    
    # Get video ID and duration from manifest
    video_id = manifest.get("video_id", "unknown")
    duration = manifest.get("duration", 0.0)
    
    # Convert extractor results to expected format
    extractor_outputs = {}
    for extractor_result in manifest.get("extractors", []):
        extractor_name = extractor_result.get("name")
        if extractor_name:
            extractor_outputs[extractor_name] = extractor_result
    
    video_data.append({
        "video_id": video_id,
        "extractor_outputs": extractor_outputs,
        "duration": duration
    })
    
    # Process batch
    results = pipeline.process_batch(
        video_data=video_data,
        fit_pca=True,
        save_features=True
    )
    
    logger.info(f"Manifest processing completed: {results['successful']} successful")
    return results


def create_dataloader(
    video_ids: List[str],
    storage: SegmentStorage,
    batch_size: int = 32,
    shuffle: bool = True
) -> List[Dict[str, Any]]:
    """
    Create a simple dataloader for training.
    
    Args:
        video_ids: List of video IDs to load
        storage: SegmentStorage instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        List of batches
    """
    if shuffle:
        import random
        random.shuffle(video_ids)
    
    batches = []
    for i in range(0, len(video_ids), batch_size):
        batch_video_ids = video_ids[i:i + batch_size]
        
        batch_data = []
        for video_id in batch_video_ids:
            try:
                data = storage.load_per_video(video_id)
                batch_data.append({
                    "video_id": video_id,
                    "features": data["feature_vectors"],
                    "attention_mask": data["attention_mask"],
                    "metadata": data["metadata"]
                })
            except Exception as e:
                logger.warning(f"Failed to load video {video_id}: {e}")
                continue
        
        if batch_data:
            batches.append(batch_data)
    
    logger.info(f"Created {len(batches)} batches from {len(video_ids)} videos")
    return batches
